"""VOptimizer — adaptive VRAM management for PyTorch models.

Implemented so far: configuration, memory observation, the tensor ledger,
execution-order tracing and hook integration. The decision and action layers,
and with them ``VOptimizer.wrap``, arrive in later phases.
"""

from .config import PressureThresholds, VOptimizerConfig
from .hooks import HookStats, LayerEvent, ModuleHookManager
from .monitor import MemorySnapshot, PressureLevel, VRAMMonitor
from .planner import ExecutionPlan, ExecutionPlanner, PlanStep
from .registry import Location, RegistrySnapshot, TensorKind, TensorMeta, TensorRegistry

__all__ = [
    "ExecutionPlan",
    "ExecutionPlanner",
    "HookStats",
    "LayerEvent",
    "Location",
    "MemorySnapshot",
    "ModuleHookManager",
    "PlanStep",
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
