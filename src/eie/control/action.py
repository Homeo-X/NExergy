"""Control action schemas for Phase 3 closed-loop control.

ControlAction represents an advisory recommendation that a safety gate must
approve before any hardware actuation.  ActuationResult records what was
observed after execution.  ActionLog provides an append-only audit trail.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterator


class ControlTarget(str, enum.Enum):
    """Physical actuator that a ControlAction targets."""

    BUILDING_SETPOINT = "building_setpoint"
    THERMAL_STORAGE_CHARGE = "thermal_storage_charge"
    THERMAL_STORAGE_DISCHARGE = "thermal_storage_discharge"
    BATTERY_CHARGE = "battery_charge"
    BATTERY_DISCHARGE = "battery_discharge"
    LOAD_DEFER = "load_defer"
    WASTE_HEAT_ROUTE = "waste_heat_route"
    PRE_COOL_SETPOINT = "pre_cool_setpoint"


class ActionStatus(str, enum.Enum):
    """Lifecycle status of a ControlAction."""

    PENDING = "pending"
    GATE_APPROVED = "gate_approved"
    GATE_REJECTED = "gate_rejected"
    EXECUTED = "executed"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True)
class ControlAction:
    """An advisory control recommendation to be evaluated by the safety gate.

    This object is immutable.  Hardware actuation requires gate approval and
    an independent safety layer.  The safety_class determines which gate rules
    apply.

    Parameters
    ----------
    action_id : str
        Unique identifier for this action.
    timestamp : datetime
        When the action was generated.
    target : ControlTarget
        Which physical actuator this action targets.
    target_value : float
        Recommended setpoint or rate value.
    unit : str
        Physical unit for target_value.
    source_decision_id : str
        The optimizer DispatchDecision that generated this action.
    status : ActionStatus
        Current lifecycle status (default PENDING).
    safety_class : str
        Determines gate rules: "thermal", "battery", "load_shift",
        "waste_heat", or a blocked class.
    risk_level : str
        Advisory risk label: "low", "medium", or "high".
    is_reversible : bool
        Whether the action can be undone within the control horizon.
    """

    action_id: str
    timestamp: datetime
    target: ControlTarget
    target_value: float
    unit: str
    source_decision_id: str
    status: ActionStatus = ActionStatus.PENDING
    safety_class: str = "thermal"
    risk_level: str = "low"
    is_reversible: bool = True

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if not self.unit:
            raise ValueError("unit is required")
        if not self.source_decision_id:
            raise ValueError("source_decision_id is required")
        if self.risk_level not in ("low", "medium", "high"):
            raise ValueError(
                f"risk_level must be 'low', 'medium', or 'high'; got {self.risk_level!r}"
            )

    def with_status(self, new_status: ActionStatus) -> "ControlAction":
        """Return a new ControlAction with an updated status."""
        return ControlAction(
            action_id=self.action_id,
            timestamp=self.timestamp,
            target=self.target,
            target_value=self.target_value,
            unit=self.unit,
            source_decision_id=self.source_decision_id,
            status=new_status,
            safety_class=self.safety_class,
            risk_level=self.risk_level,
            is_reversible=self.is_reversible,
        )


@dataclass(frozen=True)
class ActuationResult:
    """Observed outcome after executing a ControlAction (or simulation thereof).

    Parameters
    ----------
    action_id : str
        Matches the ControlAction.action_id.
    simulated : bool
        True when this result comes from a physics simulation, False for real hardware.
    applied_value : float
        The value that was actually applied (may differ from target due to clamping).
    observed_delta : dict[str, float]
        Key/value pairs describing observed changes (e.g. {"soc_delta": 0.02}).
    success : bool
        Whether the actuation was accepted by the hardware layer (or simulation).
    notes : list[str]
        Human-readable notes about the execution.
    """

    action_id: str
    simulated: bool
    applied_value: float
    observed_delta: dict[str, float]
    success: bool
    notes: list[str]


if TYPE_CHECKING:
    from eie.control.gate import GateVerdict


@dataclass(frozen=True)
class ActionRecord:
    """Immutable record combining an action with its gate verdict and outcome."""

    action: ControlAction
    gate_verdict: "GateVerdict"
    executed_at: datetime | None
    observed_outcome: ActuationResult | None


@dataclass
class ActionLog:
    """Append-only audit log of ActionRecord entries.

    Records can only be added, never removed or modified.  Provides filtering
    helpers for target/status queries.
    """

    _records: list[ActionRecord] = field(default_factory=list, init=False, repr=False)

    def append(self, record: ActionRecord) -> None:
        """Append a new record.  Raises ValueError if action_id already exists."""
        existing_ids = {r.action.action_id for r in self._records}
        if record.action.action_id in existing_ids:
            raise ValueError(
                f"action_id {record.action.action_id!r} already exists in ActionLog"
            )
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ActionRecord]:
        return iter(list(self._records))

    def for_asset(self, target: ControlTarget) -> list[ActionRecord]:
        """Return records whose action.target matches *target*."""
        return [r for r in self._records if r.action.target == target]

    def recent(self, n: int) -> list[ActionRecord]:
        """Return the *n* most recent records (by insertion order)."""
        return list(self._records[-n:])

    def approved(self) -> list[ActionRecord]:
        """Return records where the gate verdict approved the action."""
        return [r for r in self._records if r.gate_verdict.approved]

    def rejected(self) -> list[ActionRecord]:
        """Return records where the gate verdict rejected the action."""
        return [r for r in self._records if not r.gate_verdict.approved]
