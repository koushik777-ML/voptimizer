"""VOptimizer — adaptive VRAM management for PyTorch models.

Phase 1 (foundation) is implemented: configuration, memory observation and the
tensor ledger. ``VOptimizer.wrap`` arrives with the assembly phase.
"""

from .config import PressureThresholds, VOptimizerConfig
from .monitor import MemorySnapshot, PressureLevel, VRAMMonitor
from .registry import Location, RegistrySnapshot, TensorKind, TensorMeta, TensorRegistry

__all__ = [
    "Location",
    "MemorySnapshot",
    "PressureLevel",
    "PressureThresholds",
    "RegistrySnapshot",
    "TensorKind",
    "TensorMeta",
    "TensorRegistry",
    "VOptimizerConfig",
    "VRAMMonitor",
]

__version__ = "0.1.0"
