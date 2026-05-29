"""Control action schemas for Phase 3 closed-loop control."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterator


class ControlTarget(str, enum.Enum):
    BUILDING_SETPOINT = "building_setpoint"
    THERMAL_STORAGE_CHARGE = "thermal_storage_charge"
    THERMAL_STORAGE_DISCHARGE = "thermal_storage_discharge"
    BATTERY_CHARGE = "battery_charge"
    BATTERY_DISCHARGE = "battery_discharge"
    LOAD_DEFER = "load_defer"
    WASTE_HEAT_ROUTE = "waste_heat_route"
    PRE_COOL_SETPOINT = "pre_cool_setpoint"


class ActionStatus(str, enum.Enum):
    PENDING = "pending"
    GATE_APPROVED = "gate_approved"
    GATE_REJECTED = "gate_rejected"
    EXECUTED = "executed"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True)
class ControlAction:
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
            raise ValueError(f"risk_level must be 'low', 'medium', or 'high'; got {self.risk_level!r}")

    def with_status(self, new_status: ActionStatus) -> "ControlAction":
        return ControlAction(
            action_id=self.action_id, timestamp=self.timestamp, target=self.target,
            target_value=self.target_value, unit=self.unit,
            source_decision_id=self.source_decision_id, status=new_status,
            safety_class=self.safety_class, risk_level=self.risk_level,
            is_reversible=self.is_reversible,
        )


@dataclass(frozen=True)
class ActuationResult:
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
    action: ControlAction
    gate_verdict: "GateVerdict"
    executed_at: datetime | None
    observed_outcome: ActuationResult | None


@dataclass
class ActionLog:
    _records: list[ActionRecord] = field(default_factory=list, init=False, repr=False)

    def append(self, record: ActionRecord) -> None:
        existing_ids = {r.action.action_id for r in self._records}
        if record.action.action_id in existing_ids:
            raise ValueError(f"action_id {record.action.action_id!r} already exists in ActionLog")
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ActionRecord]:
        return iter(list(self._records))

    def for_asset(self, target: ControlTarget) -> list[ActionRecord]:
        return [r for r in self._records if r.action.target == target]

    def recent(self, n: int) -> list[ActionRecord]:
        return list(self._records[-n:])

    def approved(self) -> list[ActionRecord]:
        return [r for r in self._records if r.gate_verdict.approved]

    def rejected(self) -> list[ActionRecord]:
        return [r for r in self._records if not r.gate_verdict.approved]
