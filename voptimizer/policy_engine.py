"""Strategy selection (Layer 3).

The policy engine is the only place that decides *what should happen*. It reads
observations and emits :class:`Action` commands; it never touches a tensor, a
CUDA stream or a hook. That split is what makes the decisions testable: a
decision is a pure function of a memory snapshot, a ledger snapshot and the
current layer event.

Two rules shape almost everything here:

* **Reclaim past the release point, not to the band edge.** The monitor
  de-escalates only once utilization falls a hysteresis margin below the band
  it entered. Freeing exactly to the band edge leaves the level unchanged, so
  the next poll asks for another eviction round and the run thrashes.
* **Never evict what prefetch is about to need.** The layers in the current
  event's prefetch window are, by construction, the tensors with the shortest
  time to next use -- evicting one converts a scheduled async copy into a
  synchronous stall on the critical path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import VOptimizerConfig
from .hooks import LayerEvent
from .monitor import MemorySnapshot, PressureLevel
from .registry import Location, RegistrySnapshot, TensorKind, TensorMeta
from .tuner import BandwidthTuner, CostModel

_RECLAIM_SLACK = 0.02
"""Extra fraction of budget freed past the release point.

Reclaiming to exactly the release point leaves the run one allocation away from
re-entering the band, which reproduces the thrash hysteresis exists to prevent.
"""

_MIN_PREFETCH_WINDOW = 1
"""The next layer must be resident for the forward to proceed, so look-ahead is
never reduced to zero no matter how severe the pressure."""


class ActionKind(str, Enum):
    PREFETCH = "prefetch"
    OFFLOAD = "offload"
    """Move weights or activations off the GPU; they can be brought back."""
    EVICT_KV = "evict_kv"
    """Drop KV blocks; they are recomputed by re-running attention, not reloaded."""
    CHECKPOINT = "checkpoint"
    """Turn activation recompute on or off. Only valid at a step boundary."""


@dataclass(frozen=True)
class Action:
    """One command for the action layer.

    ``reason`` exists so a log line explains why memory moved; a policy whose
    decisions cannot be explained after the fact cannot be tuned.
    """

    kind: ActionKind
    keys: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()
    bytes_estimate: int = 0
    enable: bool = True
    reason: str = ""


@dataclass(frozen=True)
class Strategy:
    """How the runtime should behave inside one pressure band."""

    level: PressureLevel
    prefetch_window: int
    checkpointing: bool
    offload_weights: bool
    evict_kv: bool
    reclaim: bool

    @property
    def name(self) -> str:
        return self.level.name.lower()


@dataclass(frozen=True)
class Decision:
    """The full response to one observation."""

    pressure: PressureLevel
    strategy: Strategy
    actions: tuple[Action, ...] = ()
    bytes_to_free: int = 0
    shortfall: int = 0
    """Bytes the deficit asked for that no evictable tensor could supply.

    Nonzero means the budget is not reachable by eviction alone -- the model
    does not fit even fully offloaded, or everything left is pinned. The action
    layer cannot fix that, so it is reported rather than silently absorbed.
    """

    def of_kind(self, kind: ActionKind) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if action.kind is kind)


@dataclass(frozen=True)
class _Band:
    prefetch_delta: int
    checkpointing: bool
    offload_weights: bool
    evict_kv: bool
    reclaim: bool


_BANDS: dict[PressureLevel, _Band] = {
    # Nothing is scarce: full look-ahead, no recompute, no movement.
    PressureLevel.NORMAL: _Band(0, False, False, False, False),
    # Start paying compute to save memory before transfers become urgent.
    PressureLevel.MODERATE: _Band(0, True, True, False, True),
    # Shorten look-ahead: prefetched layers are themselves resident bytes.
    PressureLevel.HIGH: _Band(-1, True, True, True, True),
    # Survival: keep only the layer that must run next.
    PressureLevel.CRITICAL: _Band(-2, True, True, True, True),
}


class PolicyEngine:
    """Maps observations to actions.

    Args:
        config: Supplies the prefetch window, thresholds and the two
            preference knobs.
        cost_model: Scores eviction candidates. Injectable so tests can pin
            victim order. The default locks its bandwidth calibration after
            ``config.warmup_steps`` observed transfers.
    """

    def __init__(
        self,
        config: VOptimizerConfig,
        cost_model: CostModel | None = None,
    ) -> None:
        self._config = config
        self._cost_model = (
            cost_model
            if cost_model is not None
            else CostModel(tuner=BandwidthTuner(warmup_samples=config.warmup_steps))
        )
        self._checkpointing = False

    @property
    def cost_model(self) -> CostModel:
        return self._cost_model

    @property
    def checkpointing_active(self) -> bool:
        return self._checkpointing

    def reset(self) -> None:
        self._checkpointing = False

    def strategy_for(self, level: PressureLevel) -> Strategy:
        """The band's behaviour, adjusted by the user's two preferences.

        ``latency_tolerance`` and ``throughput_priority`` pull in different
        directions and are applied to different levers:

        * A workload that cannot tolerate latency keeps its look-ahead. Its
          stalls are the visible cost, so pressure is answered by evicting more
          rather than by prefetching less.
        * A workload prioritising throughput defers activation checkpointing
          until HIGH, because recompute is a pure throughput tax paid on every
          step regardless of whether the memory was needed.
        """
        band = _BANDS[level]
        window = self._config.prefetch_window + band.prefetch_delta
        if self._config.latency_tolerance == "low" and level < PressureLevel.CRITICAL:
            window = self._config.prefetch_window
        elif self._config.latency_tolerance == "high" and level is PressureLevel.NORMAL:
            window = self._config.prefetch_window + 1

        checkpointing = band.checkpointing
        if self._config.throughput_priority == "high" and level < PressureLevel.HIGH:
            checkpointing = False

        return Strategy(
            level=level,
            prefetch_window=max(_MIN_PREFETCH_WINDOW, window),
            checkpointing=checkpointing,
            offload_weights=band.offload_weights and self._config.cpu_offload_enabled,
            evict_kv=band.evict_kv,
            reclaim=band.reclaim,
        )

    def decide(
        self,
        snapshot: MemorySnapshot,
        registry: RegistrySnapshot,
        event: LayerEvent | None = None,
        at_step_boundary: bool = False,
    ) -> Decision:
        """Choose actions for the current observation.

        Args:
            snapshot: Latest memory reading.
            registry: Immutable ledger view.
            event: The layer about to run, if the call is hook-driven. Supplies
                both the prefetch candidates and the protected set.
            at_step_boundary: Whether a checkpointing change may be applied.
                Mid-forward it may not: activations already saved for this
                step cannot retroactively become recomputed ones.
        """
        strategy = self.strategy_for(snapshot.pressure)
        actions: list[Action] = []

        if event is not None and event.prefetch:
            steps = event.prefetch[: strategy.prefetch_window]
            actions.append(
                Action(
                    kind=ActionKind.PREFETCH,
                    layers=tuple(step.name for step in steps),
                    bytes_estimate=sum(step.param_bytes for step in steps),
                    reason=f"{strategy.name} look-ahead {strategy.prefetch_window}",
                )
            )

        if at_step_boundary and strategy.checkpointing != self._checkpointing:
            self._checkpointing = strategy.checkpointing
            actions.append(
                Action(
                    kind=ActionKind.CHECKPOINT,
                    enable=strategy.checkpointing,
                    reason=f"pressure {strategy.name}",
                )
            )

        deficit = self._deficit(snapshot) if strategy.reclaim else 0
        shortfall = 0
        if deficit > 0:
            freed, reclaim_actions = self._reclaim(deficit, registry, strategy, event, snapshot)
            actions.extend(reclaim_actions)
            shortfall = max(0, deficit - freed)

        return Decision(
            pressure=snapshot.pressure,
            strategy=strategy,
            actions=tuple(actions),
            bytes_to_free=deficit,
            shortfall=shortfall,
        )

    def _deficit(self, snapshot: MemorySnapshot) -> int:
        """Bytes to free to drop clear of the current band.

        Uses reserved bytes for the same reason the monitor does: reserved is
        what the next allocation competes with. Freeing allocated bytes that
        the caching allocator then holds onto does not lower pressure.
        """
        thresholds = self._config.thresholds
        entry = {
            PressureLevel.MODERATE: thresholds.moderate,
            PressureLevel.HIGH: thresholds.high,
            PressureLevel.CRITICAL: thresholds.critical,
        }.get(snapshot.pressure)
        if entry is None or snapshot.budget <= 0:
            return 0
        target = max(0.0, entry - thresholds.hysteresis - _RECLAIM_SLACK)
        return max(0, snapshot.reserved - int(target * snapshot.budget))

    def _reclaim(
        self,
        deficit: int,
        registry: RegistrySnapshot,
        strategy: Strategy,
        event: LayerEvent | None,
        snapshot: MemorySnapshot,
    ) -> tuple[int, list[Action]]:
        protected = self._protected_owners(event)
        candidates = [
            meta
            for meta in registry.entries
            if meta.location is Location.GPU
            and not meta.pinned
            and meta.owner not in protected
            and self._kind_allowed(meta.kind, strategy)
        ]

        chosen: list[TensorMeta] = []
        freed = 0
        for meta in self._cost_model.rank(candidates, registry.taken_at):
            if freed >= deficit:
                break
            chosen.append(meta)
            freed += meta.size_bytes

        return freed, self._split_actions(chosen, strategy, snapshot)

    @staticmethod
    def _protected_owners(event: LayerEvent | None) -> frozenset[str]:
        """Layers whose tensors must stay resident: the one running now and the
        ones already scheduled to be copied in."""
        if event is None:
            return frozenset()
        return frozenset((event.name, *event.prefetch_names))

    @staticmethod
    def _kind_allowed(kind: TensorKind, strategy: Strategy) -> bool:
        if kind is TensorKind.KV_BLOCK:
            return strategy.evict_kv
        if kind in (TensorKind.WEIGHT, TensorKind.OPTIMIZER_STATE):
            return strategy.offload_weights
        return True

    @staticmethod
    def _split_actions(
        chosen: list[TensorMeta], strategy: Strategy, snapshot: MemorySnapshot
    ) -> list[Action]:
        """Group victims by destination.

        KV blocks are dropped rather than moved: they are cheap to rebuild from
        the sequence and copying them to host memory only defers the cost.
        Everything else goes to the next tier down.
        """
        kv = [meta for meta in chosen if meta.kind is TensorKind.KV_BLOCK]
        movable = [meta for meta in chosen if meta.kind is not TensorKind.KV_BLOCK]
        actions: list[Action] = []
        if movable:
            actions.append(
                Action(
                    kind=ActionKind.OFFLOAD,
                    keys=tuple(meta.key for meta in movable),
                    bytes_estimate=sum(meta.size_bytes for meta in movable),
                    reason=(f"{strategy.name} pressure at {snapshot.utilization:.2f} of budget"),
                )
            )
        if kv:
            actions.append(
                Action(
                    kind=ActionKind.EVICT_KV,
                    keys=tuple(meta.key for meta in kv),
                    bytes_estimate=sum(meta.size_bytes for meta in kv),
                    reason=f"{strategy.name} pressure",
                )
            )
        return actions


@dataclass
class PolicyTrace:
    """Rolling record of decisions, for tests and for explaining a run."""

    decisions: list[Decision] = field(default_factory=list)
    limit: int = 128

    def record(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        if len(self.decisions) > self.limit:
            del self.decisions[: len(self.decisions) - self.limit]
        return decision

    def bytes_freed(self) -> int:
        return sum(
            action.bytes_estimate
            for decision in self.decisions
            for action in decision.actions
            if action.kind in (ActionKind.OFFLOAD, ActionKind.EVICT_KV)
        )
