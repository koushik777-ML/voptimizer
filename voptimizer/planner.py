"""Execution order tracing (Layer 2).

Prefetching only works if you know what runs next. One dummy forward pass with
temporary hooks records the order in which leaf modules execute, which turns
"stream the weights ahead of the compute" into a lookup instead of a guess.

The plan is a *sequence of steps*, not a set of modules. A module invoked twice
in one forward (weight sharing, a loop over a shared block) occupies two steps
with different successors, so callers index by step, not by name.

The plan is a prediction. Data-dependent control flow can diverge from the
traced order at runtime; :class:`ExecutionPlan` therefore supports name-based
lookup as a fallback and the hook layer counts mispredictions rather than
assuming the trace holds.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn


@dataclass(frozen=True)
class PlanStep:
    """One leaf-module invocation in traced execution order."""

    index: int
    name: str
    module_type: str
    param_bytes: int

    @property
    def param_gb(self) -> float:
        return self.param_bytes / float(1024**3)


@dataclass(frozen=True)
class ExecutionPlan:
    """Ordered record of one forward pass."""

    steps: tuple[PlanStep, ...] = ()
    _first_index: dict[str, int] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[PlanStep]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    @classmethod
    def of(cls, steps: list[PlanStep]) -> ExecutionPlan:
        first: dict[str, int] = {}
        for step in steps:
            first.setdefault(step.name, step.index)
        return cls(steps=tuple(steps), _first_index=first)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps)

    @property
    def total_param_bytes(self) -> int:
        """Bytes of the distinct modules in the plan.

        Counted once per module: a shared block occupying two steps still holds
        one copy of its weights.
        """
        return sum(
            self.steps[index].param_bytes for index in dict.fromkeys(self._first_index.values())
        )

    def index_of(self, name: str) -> int | None:
        """Step index of a module's first invocation, or None if untraced."""
        return self._first_index.get(name)

    def successors(
        self,
        index: int,
        window: int,
        budget_bytes: int | None = None,
    ) -> tuple[PlanStep, ...]:
        """The next ``window`` distinct modules after step ``index``.

        A window counted purely in layers is the wrong unit when layers differ
        in size: two 8 GB blocks do not fit where two 200 MB blocks did.
        ``budget_bytes`` caps the lookahead by resident bytes and always yields
        at least one step, so prefetch never stalls on an oversized layer.
        """
        if window <= 0:
            return ()

        out: list[PlanStep] = []
        seen: set[str] = set()
        used = 0
        for step in self.steps[index + 1 :]:
            if len(out) >= window:
                break
            if step.name in seen:
                continue
            if budget_bytes is not None and out and used + step.param_bytes > budget_bytes:
                break
            seen.add(step.name)
            used += step.param_bytes
            out.append(step)
        return tuple(out)


class ExecutionPlanner:
    """Traces a model's leaf-module execution order.

    Args:
        include: Optional predicate deciding which leaf modules are planned.
            Modules with no parameters (activations, dropout) are skipped by
            default — there is nothing to prefetch for them, and including them
            only dilutes the prefetch window.
    """

    def __init__(self, include: Callable[[str, nn.Module], bool] | None = None) -> None:
        self._include = include if include is not None else _has_parameters

    def trace(self, model: nn.Module, *args: Any, **kwargs: Any) -> ExecutionPlan:
        """Run one forward pass and record the order leaf modules execute in.

        The pass runs under ``no_grad`` and restores the model's training mode,
        so tracing costs one forward and leaves no state behind.
        """
        steps: list[PlanStep] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for name, module in _leaf_modules(model):
            if not self._include(name, module):
                continue
            handles.append(module.register_forward_pre_hook(self._recorder(steps, name, module)))

        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                model(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        return ExecutionPlan.of(steps)

    @staticmethod
    def _recorder(
        steps: list[PlanStep], name: str, module: nn.Module
    ) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        param_bytes = module_param_bytes(module)
        module_type = type(module).__name__

        def record(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            steps.append(
                PlanStep(
                    index=len(steps),
                    name=name,
                    module_type=module_type,
                    param_bytes=param_bytes,
                )
            )

        return record


def module_param_bytes(module: nn.Module) -> int:
    """Bytes held by a module's own parameters and buffers.

    ``recurse=False`` because a leaf module owns no children; recursing would
    double-count when a caller applies this to a container.
    """
    own = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
    return sum(tensor.numel() * tensor.element_size() for tensor in own)


def _leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if not list(module.children()) and name
    ]


def _has_parameters(_name: str, module: nn.Module) -> bool:
    return module_param_bytes(module) > 0
