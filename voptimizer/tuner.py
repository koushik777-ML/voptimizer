"""Eviction cost model and its runtime calibration (Layer 3).

The README's original plan was a ``WeightTuner`` that learns four eviction
weights from observed latency during warmup. That is not implementable in a
trustworthy way: four coupled weights fitted to a few dozen noisy per-step
latency samples do not converge, and when the resulting policy misbehaves there
is no way to tell whether the model or the weights are wrong.

What replaces it is a deterministic cost model with exactly one calibrated
quantity: effective transfer bandwidth. Bandwidth is measurable directly, it is
the term the eviction decision is actually sensitive to, and a single scalar
fitted from direct observations of the thing itself converges in a handful of
samples. Everything else in the score is fixed arithmetic over facts the
registry already stores.

The question the score answers is:

    if I evict this tensor, how many bytes do I free per second of stall I will
    later pay to get it back?
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .registry import Location, TensorMeta

_BYTES_PER_GB = float(1024**3)

DEFAULT_H2D_BYTES_PER_S = 10.0 * _BYTES_PER_GB
"""Conservative pinned-memory host-to-device rate for a PCIe 4.0 x16 link.

Real links measure 20-25 GB/s; half of that is used as the cold-start guess so
an uncalibrated model over-estimates reload cost and under-evicts. Being too
cautious costs headroom, being too optimistic costs a stall.
"""

NVME_BANDWIDTH_RATIO = 0.25
"""NVMe reads are assumed a quarter of host-to-device speed absent a sample."""

_MIN_RESTORE_S = 1e-6


@dataclass
class BandwidthTuner:
    """Tracks effective transfer bandwidth from observed copies.

    Samples are summarized with a median rather than a mean: the first copies
    after a device switch include allocation and pinning costs that are not
    representative, and a single such outlier drags a mean far enough to change
    eviction decisions.

    The tuner locks after ``warmup_samples`` observations. A cost model that
    keeps drifting makes two identical pressure situations produce different
    decisions, which is not debuggable; a locked model is wrong in a fixed,
    findable way.
    """

    warmup_samples: int = 20
    default_bytes_per_s: float = DEFAULT_H2D_BYTES_PER_S
    _samples: list[float] = field(default_factory=list, repr=False)
    _locked: bool = False

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def calibrated(self) -> bool:
        """True once any real measurement has replaced the cold-start guess."""
        return bool(self._samples)

    @property
    def bytes_per_s(self) -> float:
        if not self._samples:
            return self.default_bytes_per_s
        return statistics.median(self._samples)

    def observe(self, size_bytes: int, seconds: float) -> None:
        """Record one completed transfer.

        Non-positive durations are ignored rather than clamped: a zero-duration
        copy means the caller timed an async launch instead of the copy, and
        folding an infinite rate into the model would silently disable eviction
        cost entirely.
        """
        if self._locked or size_bytes <= 0 or seconds <= 0:
            return
        self._samples.append(size_bytes / seconds)
        if len(self._samples) >= self.warmup_samples:
            self._locked = True

    def lock(self) -> None:
        self._locked = True

    def reset(self) -> None:
        self._samples.clear()
        self._locked = False


@dataclass(frozen=True)
class CostModel:
    """Scores tensors for eviction. Higher score is evicted first.

    Args:
        tuner: Supplies measured transfer bandwidth.
        half_life_s: Idle time at which a tensor is considered half forgotten.
            Sets the timescale of the staleness term; it should be on the order
            of one step, since a tensor untouched for several steps is a
            genuinely better victim than one used this step.
        recency_floor: Score multiplier for a tensor touched just now. Nonzero
            so that a huge just-used tensor can still be chosen over a tiny
            ancient one when nothing else will free enough bytes.
    """

    tuner: BandwidthTuner = field(default_factory=BandwidthTuner)
    half_life_s: float = 0.5
    recency_floor: float = 0.25

    def restore_seconds(self, meta: TensorMeta) -> float:
        """Stall incurred to make this tensor usable again after eviction.

        Recompute competes with transfer: an activation that costs 2 ms to
        recompute is not worth a 40 ms copy back from host memory, so the
        cheaper of the two paths is what eviction will actually pay.
        """
        rate = self.tuner.bytes_per_s
        if meta.location is Location.NVME:
            rate *= NVME_BANDWIDTH_RATIO
        transfer_s = meta.size_bytes / rate if rate > 0 else math.inf
        if meta.recompute_cost_s > 0:
            return max(_MIN_RESTORE_S, min(transfer_s, meta.recompute_cost_s))
        return max(_MIN_RESTORE_S, transfer_s)

    def score(self, meta: TensorMeta, now: float) -> float:
        """Bytes freed per second of future stall, discounted by recency.

        A pinned tensor scores zero: pinning is a hard constraint from the
        integration layer, and expressing it as a very low score instead would
        let a large enough deficit override it.
        """
        if meta.pinned:
            return 0.0
        yield_per_s = meta.size_bytes / self.restore_seconds(meta)
        return yield_per_s * self._recency(meta, now) * self._frequency(meta)

    def rank(self, entries: list[TensorMeta], now: float) -> list[TensorMeta]:
        """Evictable entries, best victim first. Zero-score entries are dropped.

        Ties break on key so a given ledger state always yields the same victim
        order; without it, dict iteration order decides evictions and two
        identical runs diverge.
        """
        scored = [(self.score(meta, now), meta) for meta in entries]
        scored = [(score, meta) for score, meta in scored if score > 0.0]
        scored.sort(key=lambda pair: (-pair[0], pair[1].key))
        return [meta for _, meta in scored]

    def _recency(self, meta: TensorMeta, now: float) -> float:
        age = meta.age(now)
        staleness = age / (age + self.half_life_s) if self.half_life_s > 0 else 1.0
        return self.recency_floor + (1.0 - self.recency_floor) * staleness

    @staticmethod
    def _frequency(meta: TensorMeta) -> float:
        """Damp hot tensors. Logarithmic so a 1000-access tensor is not
        1000x more protected than a 1-access one -- past a few accesses, the
        signal is "this is hot", and further counts add nothing."""
        return 1.0 / (1.0 + math.log1p(meta.access_count))
