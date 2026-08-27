"""Hardware memory observation (Layer 2).

The monitor only reports. It never decides and never moves a tensor.

Two numbers are tracked separately because they answer different questions:
``allocated`` is what live tensors occupy, ``reserved`` is what the caching
allocator holds from the driver. Freeing a tensor lowers the first and not the
second, so a policy driven by ``allocated`` alone under-reacts to fragmentation
and a policy driven by ``reserved`` alone over-reacts to cache that would be
reused. Both are surfaced; the pressure signal uses reserved bytes, since that
is what an allocation request actually competes with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import torch

from .config import PressureThresholds, VOptimizerConfig

_BYTES_PER_GB = float(1024**3)


class PressureLevel(IntEnum):
    """Ordered so levels can be compared directly."""

    NORMAL = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True)
class MemorySnapshot:
    """One observation of device memory, in bytes unless noted."""

    allocated: int
    reserved: int
    capacity: int
    budget: int
    utilization: float
    pressure: PressureLevel
    timestamp: float

    @property
    def fragmentation(self) -> int:
        """Bytes held by the caching allocator but not backing a live tensor."""
        return max(0, self.reserved - self.allocated)

    @property
    def headroom(self) -> int:
        return max(0, self.budget - self.reserved)

    @property
    def allocated_gb(self) -> float:
        return self.allocated / _BYTES_PER_GB

    @property
    def reserved_gb(self) -> float:
        return self.reserved / _BYTES_PER_GB

    @property
    def headroom_gb(self) -> float:
        return self.headroom / _BYTES_PER_GB


class VRAMMonitor:
    """Polls device memory and maps utilization to a pressure level.

    Args:
        config: Supplies the budget, poll interval and band thresholds.
        mock_pressure: Test hook. A fixed :class:`PressureLevel`, a fixed
            utilization fraction, or a callable returning either. When set, no
            CUDA call is made, so every pressure scenario is reproducible on a
            CPU-only box.
        time_fn: Injectable clock.
    """

    def __init__(
        self,
        config: VOptimizerConfig,
        mock_pressure: PressureLevel | float | Callable[[], PressureLevel | float] | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._thresholds = config.thresholds
        self._mock_pressure = mock_pressure
        self._time_fn = time_fn
        self._level = PressureLevel.NORMAL
        self._last: MemorySnapshot | None = None

    @property
    def is_mocked(self) -> bool:
        return self._mock_pressure is not None

    def set_mock_pressure(
        self,
        mock_pressure: PressureLevel | float | Callable[[], PressureLevel | float] | None,
    ) -> None:
        """Replace the injected pressure source. Test hook."""
        self._mock_pressure = mock_pressure

    @property
    def last_snapshot(self) -> MemorySnapshot | None:
        """Most recent observation, without forcing a new read."""
        return self._last

    @property
    def pressure(self) -> PressureLevel:
        return self._level

    def capacity_bytes(self) -> int:
        """Total device memory, or 0 when no CUDA device is available."""
        if self.is_mocked or not torch.cuda.is_available():
            return 0
        return int(torch.cuda.get_device_properties(self._config.device_index).total_memory)

    def budget_bytes(self) -> int:
        """Configured target, clamped to what the device physically has."""
        capacity = self.capacity_bytes()
        if capacity == 0:
            return self._config.target_vram_bytes
        return min(self._config.target_vram_bytes, capacity)

    def poll(self, force: bool = False) -> MemorySnapshot:
        """Read memory and update the pressure level.

        Returns the cached snapshot when called again inside
        ``poll_interval_s``, unless ``force`` is set.
        """
        now = self._time_fn()
        if (
            not force
            and self._last is not None
            and now - self._last.timestamp < self._config.poll_interval_s
        ):
            return self._last

        allocated, reserved, utilization = self._read(now)
        self._level = self._classify(utilization, self._level, self._thresholds)
        snapshot = MemorySnapshot(
            allocated=allocated,
            reserved=reserved,
            capacity=self.capacity_bytes(),
            budget=self.budget_bytes(),
            utilization=utilization,
            pressure=self._level,
            timestamp=now,
        )
        self._last = snapshot
        return snapshot

    def reset(self) -> None:
        """Drop cached state so the next poll starts from NORMAL."""
        self._level = PressureLevel.NORMAL
        self._last = None

    def _read(self, now: float) -> tuple[int, int, float]:
        budget = self.budget_bytes()
        mock = self._mock_pressure
        if mock is not None:
            if callable(mock):
                mock = mock()
            utilization = (
                self._representative_utilization(mock, self._thresholds)
                if isinstance(mock, PressureLevel)
                else float(mock)
            )
            reserved = int(utilization * budget)
            return reserved, reserved, utilization

        if not torch.cuda.is_available():
            return 0, 0, 0.0

        index = self._config.device_index
        allocated = int(torch.cuda.memory_allocated(index))
        reserved = int(torch.cuda.memory_reserved(index))
        return allocated, reserved, reserved / budget if budget else 0.0

    @staticmethod
    def _representative_utilization(level: PressureLevel, thresholds: PressureThresholds) -> float:
        """A utilization comfortably inside the requested band.

        Midpoints keep injected levels clear of the hysteresis margins, so a
        test that asks for HIGH gets HIGH regardless of the previous level.
        """
        if level is PressureLevel.NORMAL:
            return thresholds.moderate / 2
        if level is PressureLevel.MODERATE:
            return (thresholds.moderate + thresholds.high) / 2
        if level is PressureLevel.HIGH:
            return (thresholds.high + thresholds.critical) / 2
        return (thresholds.critical + 1.0) / 2

    @staticmethod
    def _classify(
        utilization: float, previous: PressureLevel, thresholds: PressureThresholds
    ) -> PressureLevel:
        """Map utilization to a band, holding the previous level inside the
        hysteresis margin so boundary noise cannot cause thrashing."""
        entry = (
            (PressureLevel.CRITICAL, thresholds.critical),
            (PressureLevel.HIGH, thresholds.high),
            (PressureLevel.MODERATE, thresholds.moderate),
        )
        raw = PressureLevel.NORMAL
        for level, threshold in entry:
            if utilization >= threshold:
                raw = level
                break

        if raw >= previous:
            return raw
        release = dict(entry)[previous] - thresholds.hysteresis
        return previous if utilization >= release else raw
