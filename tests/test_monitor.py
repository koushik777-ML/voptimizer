import pytest

from voptimizer import PressureLevel, PressureThresholds, VOptimizerConfig, VRAMMonitor


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_monitor(mock_pressure, **config_kwargs):
    config = VOptimizerConfig(target_vram_gb=16.0, **config_kwargs)
    clock = FakeClock()
    return VRAMMonitor(config, mock_pressure=mock_pressure, time_fn=clock), clock


@pytest.mark.parametrize(
    "utilization, expected",
    [
        (0.0, PressureLevel.NORMAL),
        (0.69, PressureLevel.NORMAL),
        (0.70, PressureLevel.MODERATE),
        (0.84, PressureLevel.MODERATE),
        (0.85, PressureLevel.HIGH),
        (0.94, PressureLevel.HIGH),
        (0.95, PressureLevel.CRITICAL),
        (1.20, PressureLevel.CRITICAL),
    ],
)
def test_utilization_maps_to_pressure_band(utilization, expected):
    monitor, _ = make_monitor(utilization)
    assert monitor.poll().pressure is expected


@pytest.mark.parametrize("level", list(PressureLevel))
def test_injected_level_is_reported_verbatim(level):
    monitor, _ = make_monitor(level)
    assert monitor.poll().pressure is level


def test_injected_level_overrides_a_higher_previous_level():
    """Hysteresis must not stop a test from stepping pressure back down."""
    monitor, _ = make_monitor(PressureLevel.CRITICAL)
    assert monitor.poll(force=True).pressure is PressureLevel.CRITICAL

    monitor.set_mock_pressure(PressureLevel.NORMAL)
    assert monitor.poll(force=True).pressure is PressureLevel.NORMAL


def test_hysteresis_holds_level_inside_the_release_margin():
    thresholds = PressureThresholds(hysteresis=0.05)
    readings = [0.87, 0.83, 0.79]
    monitor, _ = make_monitor(lambda: readings.pop(0), thresholds=thresholds)

    assert monitor.poll(force=True).pressure is PressureLevel.HIGH
    # 0.83 is below the 0.85 entry but inside the 0.80 release margin.
    assert monitor.poll(force=True).pressure is PressureLevel.HIGH
    assert monitor.poll(force=True).pressure is PressureLevel.MODERATE


def test_escalation_is_immediate():
    readings = [0.10, 0.96]
    monitor, _ = make_monitor(lambda: readings.pop(0))
    assert monitor.poll(force=True).pressure is PressureLevel.NORMAL
    assert monitor.poll(force=True).pressure is PressureLevel.CRITICAL


def test_poll_interval_returns_cached_snapshot():
    readings = [0.10, 0.99]
    monitor, clock = make_monitor(lambda: readings.pop(0), poll_interval_s=1.0)

    first = monitor.poll()
    clock.advance(0.5)
    assert monitor.poll() is first
    assert readings == [0.99]

    clock.advance(0.6)
    assert monitor.poll().pressure is PressureLevel.CRITICAL


def test_force_bypasses_the_poll_interval():
    readings = [0.10, 0.99]
    monitor, _ = make_monitor(lambda: readings.pop(0), poll_interval_s=100.0)
    monitor.poll()
    assert monitor.poll(force=True).pressure is PressureLevel.CRITICAL


def test_snapshot_reports_headroom_and_fragmentation():
    monitor, _ = make_monitor(0.75)
    snapshot = monitor.poll()

    assert snapshot.budget == 16 * 1024**3
    assert snapshot.reserved == pytest.approx(0.75 * snapshot.budget, rel=1e-6)
    assert snapshot.headroom == pytest.approx(0.25 * snapshot.budget, rel=1e-3)
    assert snapshot.fragmentation == 0
    assert snapshot.reserved_gb == pytest.approx(12.0, rel=1e-3)


def test_budget_never_exceeds_device_capacity():
    """With no CUDA device the target stands; the clamp is exercised on GPU."""
    monitor, _ = make_monitor(0.5)
    assert monitor.budget_bytes() == 16 * 1024**3
    assert monitor.capacity_bytes() == 0


def test_reset_clears_cached_state():
    monitor, _ = make_monitor(0.99)
    monitor.poll()
    monitor.reset()
    assert monitor.last_snapshot is None
    assert monitor.pressure is PressureLevel.NORMAL
