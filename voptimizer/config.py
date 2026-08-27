"""User-facing configuration for VOptimizer (Layer 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LatencyTolerance = Literal["low", "medium", "high"]
ThroughputPriority = Literal["low", "medium", "high"]

_VALID_TOLERANCE = ("low", "medium", "high")


@dataclass(frozen=True)
class PressureThresholds:
    """Budget utilization fractions at which each pressure level begins.

    ``hysteresis`` is the fraction a level must drop below its entry threshold
    before the monitor is allowed to report a lower level again. Without it the
    control loop oscillates between offloading and reloading the same tensors
    when utilization sits on a boundary.
    """

    moderate: float = 0.70
    high: float = 0.85
    critical: float = 0.95
    hysteresis: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 < self.moderate < self.high < self.critical < 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < moderate < high < critical < 1, got "
                f"{self.moderate}, {self.high}, {self.critical}"
            )
        if not 0.0 <= self.hysteresis < self.moderate:
            raise ValueError(f"hysteresis must be in [0, {self.moderate}), got {self.hysteresis}")


@dataclass
class VOptimizerConfig:
    """Declarative memory target for a wrapped model.

    Attributes:
        target_vram_gb: Ceiling VOptimizer schedules against. Clamped to the
            device capacity at runtime; a target above capacity is meaningless.
        latency_tolerance: How much per-step slowdown the workload accepts.
            Drives prefetch depth and how aggressively weights are streamed.
        throughput_priority: Relative weight of sustained throughput versus
            keeping headroom free.
        warmup_steps: Steps observed before the eviction cost model locks.
        prefetch_window: Layers of look-ahead for weight prefetch. Depth beyond
            what the PCIe link can sustain buys nothing but resident bytes.
        poll_interval_s: Minimum wall time between hardware memory reads.
        thresholds: Pressure band boundaries as fractions of the budget.
        cpu_offload_enabled: Allow the GPU -> CPU tier.
        nvme_offload_path: Enables the CPU -> NVMe tier when set.
        device_index: CUDA device VOptimizer manages.
    """

    target_vram_gb: float
    latency_tolerance: LatencyTolerance = "medium"
    throughput_priority: ThroughputPriority = "medium"
    warmup_steps: int = 20
    prefetch_window: int = 2
    poll_interval_s: float = 0.0
    thresholds: PressureThresholds = field(default_factory=PressureThresholds)
    cpu_offload_enabled: bool = True
    nvme_offload_path: str | None = None
    device_index: int = 0

    def __post_init__(self) -> None:
        if self.target_vram_gb <= 0:
            raise ValueError(f"target_vram_gb must be positive, got {self.target_vram_gb}")
        if self.latency_tolerance not in _VALID_TOLERANCE:
            raise ValueError(
                f"latency_tolerance must be one of {_VALID_TOLERANCE}, "
                f"got {self.latency_tolerance!r}"
            )
        if self.throughput_priority not in _VALID_TOLERANCE:
            raise ValueError(
                f"throughput_priority must be one of {_VALID_TOLERANCE}, "
                f"got {self.throughput_priority!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.prefetch_window < 0:
            raise ValueError(f"prefetch_window must be non-negative, got {self.prefetch_window}")
        if self.poll_interval_s < 0:
            raise ValueError(f"poll_interval_s must be non-negative, got {self.poll_interval_s}")
        if self.device_index < 0:
            raise ValueError(f"device_index must be non-negative, got {self.device_index}")

    @property
    def target_vram_bytes(self) -> int:
        return int(self.target_vram_gb * 1024**3)

    @property
    def nvme_offload_enabled(self) -> bool:
        return self.nvme_offload_path is not None
