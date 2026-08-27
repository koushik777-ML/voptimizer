import math

import pytest

from voptimizer.registry import Location, TensorKind, TensorMeta
from voptimizer.tuner import DEFAULT_H2D_BYTES_PER_S, BandwidthTuner, CostModel

GB = 1024**3


def meta(
    key: str,
    size_bytes: int = GB,
    last_access: float = 0.0,
    access_count: int = 0,
    recompute_cost_s: float = 0.0,
    pinned: bool = False,
    location: Location = Location.GPU,
    kind: TensorKind = TensorKind.WEIGHT,
) -> TensorMeta:
    return TensorMeta(
        key=key,
        size_bytes=size_bytes,
        kind=kind,
        location=location,
        recompute_cost_s=recompute_cost_s,
        pinned=pinned,
        last_access=last_access,
        access_count=access_count,
    )


def test_uncalibrated_tuner_uses_the_conservative_default():
    tuner = BandwidthTuner()

    assert tuner.bytes_per_s == DEFAULT_H2D_BYTES_PER_S
    assert tuner.calibrated is False
    assert tuner.locked is False


def test_median_ignores_a_slow_first_sample():
    """The first copy carries pinning and allocation cost; a mean would keep it."""
    tuner = BandwidthTuner(warmup_samples=10)
    tuner.observe(GB, 1.0)  # 1 GB/s outlier
    for _ in range(4):
        tuner.observe(GB, 0.1)  # 10 GB/s

    assert tuner.bytes_per_s == pytest.approx(10 * GB, rel=1e-6)


def test_tuner_locks_after_warmup_and_ignores_later_samples():
    tuner = BandwidthTuner(warmup_samples=3)
    for _ in range(3):
        tuner.observe(GB, 0.1)
    locked_rate = tuner.bytes_per_s

    tuner.observe(GB, 100.0)

    assert tuner.locked is True
    assert tuner.sample_count == 3
    assert tuner.bytes_per_s == locked_rate


@pytest.mark.parametrize("seconds", [0.0, -1.0])
def test_non_positive_durations_are_rejected(seconds):
    """A zero-duration copy means an async launch was timed, not the copy."""
    tuner = BandwidthTuner()
    tuner.observe(GB, seconds)

    assert tuner.calibrated is False


def test_zero_sized_transfers_are_rejected():
    tuner = BandwidthTuner()
    tuner.observe(0, 0.5)

    assert tuner.calibrated is False


def test_reset_reopens_calibration():
    tuner = BandwidthTuner(warmup_samples=1)
    tuner.observe(GB, 0.1)
    tuner.reset()

    assert tuner.locked is False
    assert tuner.bytes_per_s == DEFAULT_H2D_BYTES_PER_S


def test_restore_cost_tracks_measured_bandwidth():
    tuner = BandwidthTuner(warmup_samples=1)
    tuner.observe(GB, 0.5)  # 2 GB/s
    model = CostModel(tuner=tuner)

    assert model.restore_seconds(meta("w", size_bytes=GB)) == pytest.approx(0.5)


def test_nvme_entries_cost_more_to_restore_than_host_entries():
    model = CostModel()
    host = model.restore_seconds(meta("a", location=Location.CPU))
    nvme = model.restore_seconds(meta("b", location=Location.NVME))

    assert nvme > host


def test_recompute_wins_when_cheaper_than_the_copy():
    """An activation cheap to rebuild is never worth a full copy back."""
    model = CostModel()
    fast = meta("act", kind=TensorKind.ACTIVATION, recompute_cost_s=0.001)
    slow = meta("act2", kind=TensorKind.ACTIVATION, recompute_cost_s=100.0)

    assert model.restore_seconds(fast) == pytest.approx(0.001)
    assert model.restore_seconds(slow) == model.restore_seconds(meta("plain"))


def test_pinned_tensors_score_zero_and_are_never_ranked():
    model = CostModel()
    entries = [meta("pinned", pinned=True), meta("free")]

    assert model.score(entries[0], now=10.0) == 0.0
    assert [m.key for m in model.rank(entries, now=10.0)] == ["free"]


def test_stale_tensors_outrank_identical_fresh_ones():
    model = CostModel(half_life_s=0.5)
    fresh = meta("fresh", last_access=10.0)
    stale = meta("stale", last_access=0.0)

    assert model.score(stale, now=10.0) > model.score(fresh, now=10.0)


def test_hot_tensors_are_damped_relative_to_cold_ones():
    model = CostModel()
    cold = meta("cold", access_count=0)
    hot = meta("hot", access_count=100)

    assert model.score(cold, now=1.0) > model.score(hot, now=1.0)


def test_frequency_damping_is_sublinear():
    """1000 accesses must not make a tensor 1000x more protected than 1."""
    model = CostModel()
    once = model.score(meta("a", access_count=1), now=1.0)
    often = model.score(meta("b", access_count=1000), now=1.0)

    assert once / often < 10


def test_a_large_recently_used_tensor_can_still_outrank_a_tiny_ancient_one():
    """The recency floor is what keeps eviction able to free real bytes."""
    model = CostModel()
    huge_fresh = meta("huge", size_bytes=8 * GB, last_access=100.0)
    tiny_old = meta("tiny", size_bytes=1024, last_access=0.0)

    assert [m.key for m in model.rank([tiny_old, huge_fresh], now=100.0)] == ["huge", "tiny"]


def test_expensive_to_restore_tensors_rank_last():
    model = CostModel()
    cheap = meta("cheap", kind=TensorKind.ACTIVATION, recompute_cost_s=1e-4)
    costly = meta("costly", kind=TensorKind.ACTIVATION, recompute_cost_s=10.0)

    assert [m.key for m in model.rank([costly, cheap], now=1.0)] == ["cheap", "costly"]


def test_ranking_is_deterministic_for_equal_scores():
    model = CostModel()
    entries = [meta("c"), meta("a"), meta("b")]

    assert [m.key for m in model.rank(entries, now=1.0)] == ["a", "b", "c"]


def test_scores_stay_finite_for_zero_sized_entries():
    model = CostModel()

    assert math.isfinite(model.score(meta("empty", size_bytes=0), now=1.0))
