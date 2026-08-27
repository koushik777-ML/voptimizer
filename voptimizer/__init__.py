"""VOptimizer — adaptive VRAM management for PyTorch models.

Implemented so far: configuration, memory observation, the tensor ledger,
execution-order tracing, hook integration and the decision layer. The action
layer, and with it ``VOptimizer.wrap``, arrives in later phases.
"""

from .config import PressureThresholds, VOptimizerConfig
from .hooks import HookStats, LayerEvent, ModuleHookManager
from .monitor import MemorySnapshot, PressureLevel, VRAMMonitor
from .planner import ExecutionPlan, ExecutionPlanner, PlanStep
from .policy_engine import Action, ActionKind, Decision, PolicyEngine, PolicyTrace, Strategy
from .registry import Location, RegistrySnapshot, TensorKind, TensorMeta, TensorRegistry
from .tuner import BandwidthTuner, CostModel

__all__ = [
    "Action",
    "ActionKind",
    "BandwidthTuner",
    "CostModel",
    "Decision",
    "ExecutionPlan",
    "ExecutionPlanner",
    "HookStats",
    "LayerEvent",
    "Location",
    "MemorySnapshot",
    "ModuleHookManager",
    "PlanStep",
    "PolicyEngine",
    "PolicyTrace",
    "PressureLevel",
    "PressureThresholds",
    "RegistrySnapshot",
    "Strategy",
    "TensorKind",
    "TensorMeta",
    "TensorRegistry",
    "VOptimizerConfig",
    "VRAMMonitor",
]

__version__ = "0.1.0"
