"""PyTorch hook integration — the only module that touches hooks.

Every other layer works with :class:`LayerEvent` objects, so hook quirks stay
in one file. Three of those quirks drive the design:

* **Cursor desync.** A step cursor walking the plan drifts the moment execution
  diverges from the trace or a forward raises midway. The root module's
  pre-hook resets the cursor on every top-level forward, and a name mismatch
  resnaps the cursor to the plan instead of trusting it.
* **Recompute re-entry.** Activation checkpointing replays a segment's forward
  during backward. Those invocations are not new steps; counting them corrupts
  the schedule. :meth:`ModuleHookManager.paused` suppresses them.
* **Detachment.** Hooks outlive the object that registered them unless handles
  are released, so the manager is a context manager and ``detach`` is
  idempotent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from .planner import ExecutionPlan, PlanStep, module_param_bytes


@dataclass(frozen=True)
class LayerEvent:
    """What the decision layer is told when a layer starts or finishes."""

    name: str
    step_index: int
    prefetch: tuple[PlanStep, ...] = ()
    predicted: bool = True
    """False when execution diverged from the traced order at this step."""

    @property
    def prefetch_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.prefetch)


LayerCallback = Callable[[LayerEvent], None]


@dataclass
class HookStats:
    """Counters for diagnosing a plan that does not match runtime."""

    forwards: int = 0
    steps: int = 0
    mispredictions: int = 0
    unplanned: int = 0
    """Invocations of modules that were not in the trace at all."""

    @property
    def prediction_accuracy(self) -> float:
        return 1.0 - (self.mispredictions / self.steps) if self.steps else 1.0


class ModuleHookManager:
    """Attaches pre/post forward hooks and emits :class:`LayerEvent`s.

    Args:
        plan: Traced execution order.
        on_layer_start: Called before a layer runs — where prefetch is issued.
        on_layer_end: Called after a layer runs — where the policy cycle runs.
        prefetch_window: Layers of look-ahead reported on each start event.
        prefetch_budget_bytes: Caps look-ahead by resident bytes.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        on_layer_start: LayerCallback | None = None,
        on_layer_end: LayerCallback | None = None,
        prefetch_window: int = 2,
        prefetch_budget_bytes: int | None = None,
    ) -> None:
        self._plan = plan
        self._on_start = on_layer_start
        self._on_end = on_layer_end
        self._window = prefetch_window
        self._budget = prefetch_budget_bytes
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._cursor = 0
        self._last_index = -1
        self._paused = False
        self.stats = HookStats()

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    @property
    def cursor(self) -> int:
        """Index of the next expected step in the plan."""
        return self._cursor

    def attach(self, model: nn.Module) -> ModuleHookManager:
        """Hook every weight-bearing leaf module, plus the root for resets.

        Modules outside the plan are hooked too: a branch the trace never took
        still consumes VRAM, and leaving it unhooked would make the divergence
        invisible rather than merely unpredicted.
        """
        if self._handles:
            raise RuntimeError("hooks are already attached; call detach() first")

        planned = set(self._plan.names)
        for name, module in model.named_modules():
            if not name:
                continue
            is_leaf = not list(module.children())
            if name in planned or (is_leaf and module_param_bytes(module) > 0):
                self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
                self._handles.append(module.register_forward_hook(self._make_post_hook(name)))
        self._handles.append(model.register_forward_pre_hook(self._reset_hook))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._cursor = 0
        self._last_index = -1

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Suppress events, e.g. while a checkpointed segment is recomputed."""
        previous = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = previous

    def __enter__(self) -> ModuleHookManager:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.detach()

    def _reset_hook(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        if self._paused:
            return
        self.reset()
        self.stats.forwards += 1

    def _make_pre_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def pre_hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            if self._paused:
                return
            index, predicted = self._resolve(name)
            self.stats.steps += 1
            if index is None:
                self.stats.unplanned += 1
                self._last_index = -1
                self._emit(self._on_start, LayerEvent(name, -1, (), predicted=False))
                return
            if not predicted:
                self.stats.mispredictions += 1

            self._cursor = index + 1
            self._last_index = index
            self._emit(
                self._on_start,
                LayerEvent(
                    name=name,
                    step_index=index,
                    prefetch=self._plan.successors(index, self._window, self._budget),
                    predicted=predicted,
                ),
            )

        return pre_hook

    def _make_post_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def post_hook(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            if self._paused:
                return
            self._emit(self._on_end, LayerEvent(name=name, step_index=self._last_index))

        return post_hook

    def _resolve(self, name: str) -> tuple[int | None, bool]:
        """Map a running module to a plan step.

        Returns the step index and whether it matched the prediction. On a
        mismatch the cursor resnaps to the module's first planned step rather
        than letting the drift compound over the rest of the forward.
        """
        if self._cursor < len(self._plan) and self._plan[self._cursor].name == name:
            return self._cursor, True
        return self._plan.index_of(name), False

    @staticmethod
    def _emit(callback: LayerCallback | None, event: LayerEvent) -> None:
        if callback is not None:
            callback(event)
