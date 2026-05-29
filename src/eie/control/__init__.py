"""Phase 3 closed-loop control layer.

Provides the safety gate, control policies, and closed-loop controller
for low-risk autonomous dispatch of thermal, battery, load-shift, and
waste-heat assets.
"""

from eie.control.action import (
    ActionLog,
    ActionRecord,
    ActionStatus,
    ActuationResult,
    ControlAction,
    ControlTarget,
)
from eie.control.gate import GateVerdict, SafetyGate
from eie.control.loop import ClosedLoopController, LoopConfig, LoopIteration
from eie.control.policy import (
    BatteryPolicy,
    ControlPolicy,
    LoadShiftPolicy,
    PreCoolingPolicy,
    ThermalStoragePolicy,
    WasteHeatPolicy,
)

__all__ = [
    "ActionLog",
    "ActionRecord",
    "ActionStatus",
    "ActuationResult",
    "BatteryPolicy",
    "ClosedLoopController",
    "ControlAction",
    "ControlPolicy",
    "ControlTarget",
    "GateVerdict",
    "LoadShiftPolicy",
    "LoopConfig",
    "LoopIteration",
    "PreCoolingPolicy",
    "SafetyGate",
    "ThermalStoragePolicy",
    "WasteHeatPolicy",
]
