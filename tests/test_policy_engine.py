import pytest

from voptimizer.config import VOptimizerConfig
from voptimizer.hooks import LayerEvent
from voptimizer.monitor import MemorySnapshot, PressureLevel
from voptimizer.planner import PlanStep
from voptimizer.policy_engine import ActionKind, PolicyEngine, PolicyTrace
from voptimizer.registry import Location, RegistrySnapshot, TensorKind, TensorMeta

GB = 1024**3
BUDGET = 16 * GB


def snapshot(level: PressureLevel, utilization: float, budget: int = BUDGET) -> MemorySnapshot:
    reserved = int(utilization * budget)
    return MemorySnapshot(
        allocated=reserved,
        reserved=reserved,
        capacity=budget,
        budget=budget,
        utilization=utilization,
        pressure=level,
        timestamp=1.0,
    )


def entry(
    key: str,
    size_bytes: int = GB,
    kind: TensorKind = TensorKind.WEIGHT,
    owner: str | None = None,
    location: Location = Location.GPU,
    pinned: bool = False,
    last_access: float = 0.0,
) -> TensorMeta:
    return TensorMeta(
        key=key,
        size_bytes=size_bytes,
        kind=kind,
        location=location,
        owner=owner,
        pinned=pinned,
        last_access=last_access,
    )


def ledger(*entries: TensorMeta, now: float = 10.0) -> RegistrySnapshot:
    return RegistrySnapshot(entries=entries, taken_at=now)


def config(**kwargs) -> VOptimizerConfig:
    kwargs.setdefault("target_vram_gb", 16.0)
    return VOptimizerConfig(**kwargs)


def event(name: str = "blocks.1", prefetch: tuple[str, ...] = ()) -> LayerEvent:
    return LayerEvent(
        name=name,
        step_index=1,
        prefetch=tuple(
            PlanStep(index=i + 2, name=n, module_type="Linear", param_bytes=GB)
            for i, n in enumerate(prefetch)
        ),
    )


def test_normal_pressure_does_nothing_but_prefetch():
    engine = PolicyEngine(config())

    decision = engine.decide(
        snapshot(PressureLevel.NORMAL, 0.4),
        ledger(entry("w0")),
        event(prefetch=("a", "b")),
    )

    assert decision.bytes_to_free == 0
    assert [a.kind for a in decision.actions] == [ActionKind.PREFETCH]
    assert decision.of_kind(ActionKind.PREFETCH)[0].layers == ("a", "b")


def test_prefetch_window_shrinks_as_pressure_rises():
    engine = PolicyEngine(config(prefetch_window=3))
    windows = [
        engine.strategy_for(level).prefetch_window
        for level in (
            PressureLevel.NORMAL,
            PressureLevel.MODERATE,
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        )
    ]

    assert windows == [3, 3, 2, 1]


def test_prefetch_never_reaches_zero():
    """The next layer must be resident or the forward cannot proceed."""
    engine = PolicyEngine(config(prefetch_window=1))

    assert engine.strategy_for(PressureLevel.CRITICAL).prefetch_window == 1


def test_latency_sensitive_runs_keep_look_ahead_and_evict_instead():
    engine = PolicyEngine(config(prefetch_window=3, latency_tolerance="low"))

    assert engine.strategy_for(PressureLevel.HIGH).prefetch_window == 3
    assert engine.strategy_for(PressureLevel.CRITICAL).prefetch_window == 1


def test_throughput_priority_defers_checkpointing_to_high():
    engine = PolicyEngine(config(throughput_priority="high"))

    assert engine.strategy_for(PressureLevel.MODERATE).checkpointing is False
    assert engine.strategy_for(PressureLevel.HIGH).checkpointing is True


def test_offload_is_disabled_when_the_cpu_tier_is_off():
    engine = PolicyEngine(config(cpu_offload_enabled=False))

    assert engine.strategy_for(PressureLevel.HIGH).offload_weights is False


def test_reclaim_targets_past_the_release_point_not_the_band_edge():
    """Freeing to the band edge leaves the level unchanged and re-triggers."""
    cfg = config()
    engine = PolicyEngine(cfg)
    thresholds = cfg.thresholds
    snap = snapshot(PressureLevel.HIGH, 0.88)

    decision = engine.decide(snap, ledger())

    resulting_utilization = (snap.reserved - decision.bytes_to_free) / BUDGET
    assert resulting_utilization < thresholds.high - thresholds.hysteresis


def test_eviction_stops_once_the_deficit_is_covered():
    engine = PolicyEngine(config())
    entries = [entry(f"w{i}", size_bytes=GB) for i in range(8)]

    decision = engine.decide(snapshot(PressureLevel.MODERATE, 0.75), ledger(*entries))

    offload = decision.of_kind(ActionKind.OFFLOAD)[0]
    assert offload.bytes_estimate >= decision.bytes_to_free
    assert offload.bytes_estimate - decision.bytes_to_free < GB


def test_layers_in_the_prefetch_window_are_never_evicted():
    """Evicting a scheduled prefetch converts an async copy into a stall."""
    engine = PolicyEngine(config())
    entries = [
        entry("hot", owner="blocks.1"),
        entry("incoming", owner="blocks.2"),
        entry("cold", owner="blocks.7"),
    ]

    decision = engine.decide(
        snapshot(PressureLevel.CRITICAL, 0.97),
        ledger(*entries),
        event(name="blocks.1", prefetch=("blocks.2",)),
    )

    assert decision.of_kind(ActionKind.OFFLOAD)[0].keys == ("cold",)


def test_pinned_entries_are_never_evicted_even_under_a_large_deficit():
    engine = PolicyEngine(config())

    decision = engine.decide(
        snapshot(PressureLevel.CRITICAL, 0.99),
        ledger(entry("pinned", size_bytes=8 * GB, pinned=True)),
    )

    assert decision.actions == ()
    assert decision.shortfall == decision.bytes_to_free > 0


def test_non_gpu_entries_are_not_candidates():
    engine = PolicyEngine(config())

    decision = engine.decide(
        snapshot(PressureLevel.HIGH, 0.9),
        ledger(entry("already_out", size_bytes=8 * GB, location=Location.CPU)),
    )

    assert decision.actions == ()


def test_kv_blocks_are_dropped_not_offloaded():
    """Copying a KV block to host memory only defers the cost of rebuilding it."""
    engine = PolicyEngine(config())
    entries = [entry("kv0", kind=TensorKind.KV_BLOCK), entry("w0")]

    decision = engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(*entries))

    assert decision.of_kind(ActionKind.EVICT_KV)[0].keys == ("kv0",)
    assert "kv0" not in decision.of_kind(ActionKind.OFFLOAD)[0].keys


def test_kv_blocks_are_spared_below_high_pressure():
    engine = PolicyEngine(config())
    entries = [entry("kv0", size_bytes=8 * GB, kind=TensorKind.KV_BLOCK)]

    decision = engine.decide(snapshot(PressureLevel.MODERATE, 0.75), ledger(*entries))

    assert decision.actions == ()
    assert decision.shortfall > 0


def test_shortfall_is_reported_when_eviction_cannot_reach_the_budget():
    engine = PolicyEngine(config())

    decision = engine.decide(
        snapshot(PressureLevel.CRITICAL, 0.99),
        ledger(entry("small", size_bytes=1024)),
    )

    assert decision.shortfall == decision.bytes_to_free - 1024


def test_checkpointing_toggles_only_at_a_step_boundary():
    """Activations already saved this step cannot retroactively be recomputed."""
    engine = PolicyEngine(config())
    mid_forward = engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger())

    assert mid_forward.of_kind(ActionKind.CHECKPOINT) == ()
    assert engine.checkpointing_active is False

    boundary = engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(), at_step_boundary=True)

    assert boundary.of_kind(ActionKind.CHECKPOINT)[0].enable is True
    assert engine.checkpointing_active is True


def test_checkpointing_is_not_re_emitted_while_already_active():
    engine = PolicyEngine(config())
    engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(), at_step_boundary=True)

    again = engine.decide(snapshot(PressureLevel.CRITICAL, 0.97), ledger(), at_step_boundary=True)

    assert again.of_kind(ActionKind.CHECKPOINT) == ()


def test_checkpointing_is_disabled_again_when_pressure_clears():
    engine = PolicyEngine(config())
    engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(), at_step_boundary=True)

    cleared = engine.decide(snapshot(PressureLevel.NORMAL, 0.3), ledger(), at_step_boundary=True)

    assert cleared.of_kind(ActionKind.CHECKPOINT)[0].enable is False
    assert engine.checkpointing_active is False


def test_decide_without_an_event_still_reclaims():
    """Polling-driven calls have no layer context but must still act."""
    engine = PolicyEngine(config())

    decision = engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(entry("w0")))

    assert decision.of_kind(ActionKind.PREFETCH) == ()
    assert decision.of_kind(ActionKind.OFFLOAD)[0].keys == ("w0",)


def test_prefetch_list_is_truncated_to_the_current_window():
    engine = PolicyEngine(config(prefetch_window=3))

    decision = engine.decide(
        snapshot(PressureLevel.CRITICAL, 0.97),
        ledger(),
        event(prefetch=("a", "b", "c")),
    )

    assert decision.of_kind(ActionKind.PREFETCH)[0].layers == ("a",)


def test_zero_budget_snapshots_do_not_produce_a_deficit():
    engine = PolicyEngine(config())

    decision = engine.decide(snapshot(PressureLevel.HIGH, 0.9, budget=0), ledger())

    assert decision.bytes_to_free == 0


def test_default_cost_model_locks_calibration_after_config_warmup():
    engine = PolicyEngine(config(warmup_steps=2))
    tuner = engine.cost_model.tuner
    for _ in range(2):
        tuner.observe(GB, 0.1)

    assert tuner.locked is True


def test_reset_clears_checkpoint_state():
    engine = PolicyEngine(config())
    engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(), at_step_boundary=True)
    engine.reset()

    assert engine.checkpointing_active is False


def test_actions_carry_a_reason():
    engine = PolicyEngine(config())

    decision = engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(entry("w0")))

    assert all(action.reason for action in decision.actions)


def test_trace_keeps_a_bounded_history_and_totals_freed_bytes():
    engine = PolicyEngine(config())
    trace = PolicyTrace(limit=3)
    for _ in range(5):
        trace.record(engine.decide(snapshot(PressureLevel.HIGH, 0.9), ledger(entry("w0"))))

    assert len(trace.decisions) == 3
    assert trace.bytes_freed() == pytest.approx(3 * GB)
