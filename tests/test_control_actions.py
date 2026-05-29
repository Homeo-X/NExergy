"""Tests for control action schemas, ActionLog, and lifecycle helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eie.control.action import (
    ActionLog,
    ActionRecord,
    ActionStatus,
    ActuationResult,
    ControlAction,
    ControlTarget,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def action(now: datetime) -> ControlAction:
    return ControlAction(
        action_id="act-001",
        timestamp=now,
        target=ControlTarget.BATTERY_CHARGE,
        target_value=1500.0,
        unit="W",
        source_decision_id="dec-001",
        status=ActionStatus.PENDING,
        safety_class="battery",
        risk_level="low",
        is_reversible=True,
    )


# ── ControlAction construction ─────────────────────────────────────────────

def test_action_defaults(action: ControlAction) -> None:
    assert action.status == ActionStatus.PENDING
    assert action.is_reversible is True


def test_action_requires_action_id(now: datetime) -> None:
    with pytest.raises((ValueError, TypeError)):
        ControlAction(
            action_id="",
            timestamp=now,
            target=ControlTarget.BATTERY_CHARGE,
            target_value=100.0,
            unit="W",
            source_decision_id="dec-1",
        )


def test_action_requires_unit(now: datetime) -> None:
    with pytest.raises((ValueError, TypeError)):
        ControlAction(
            action_id="act-x",
            timestamp=now,
            target=ControlTarget.BATTERY_CHARGE,
            target_value=100.0,
            unit="",
            source_decision_id="dec-1",
        )


def test_action_invalid_risk_level(now: datetime) -> None:
    with pytest.raises(ValueError, match="risk_level"):
        ControlAction(
            action_id="act-x",
            timestamp=now,
            target=ControlTarget.BATTERY_CHARGE,
            target_value=100.0,
            unit="W",
            source_decision_id="dec-1",
            risk_level="extreme",
        )


def test_action_requires_source_decision_id(now: datetime) -> None:
    with pytest.raises((ValueError, TypeError)):
        ControlAction(
            action_id="act-x",
            timestamp=now,
            target=ControlTarget.BATTERY_CHARGE,
            target_value=100.0,
            unit="W",
            source_decision_id="",
        )


def test_with_status(action: ControlAction) -> None:
    approved = action.with_status(ActionStatus.GATE_APPROVED)
    assert approved.status == ActionStatus.GATE_APPROVED
    assert action.status == ActionStatus.PENDING
    assert approved.action_id == action.action_id


def test_action_is_frozen(action: ControlAction) -> None:
    with pytest.raises((AttributeError, TypeError)):
        action.status = ActionStatus.EXECUTED  # type: ignore[misc]


# ── ControlTarget enum ──────────────────────────────────────────────────────────────

def test_all_targets_defined() -> None:
    targets = list(ControlTarget)
    assert ControlTarget.BATTERY_CHARGE in targets
    assert ControlTarget.BATTERY_DISCHARGE in targets
    assert ControlTarget.THERMAL_STORAGE_CHARGE in targets
    assert ControlTarget.THERMAL_STORAGE_DISCHARGE in targets
    assert ControlTarget.LOAD_DEFER in targets
    assert ControlTarget.WASTE_HEAT_ROUTE in targets
    assert ControlTarget.BUILDING_SETPOINT in targets
    assert ControlTarget.PRE_COOL_SETPOINT in targets


# ── ActuationResult ─────────────────────────────────────────────────────────────────

def test_actuation_result_fields(action: ControlAction) -> None:
    result = ActuationResult(
        action_id=action.action_id,
        simulated=True,
        applied_value=1500.0,
        observed_delta={"soc_delta_estimate": 0.05},
        success=True,
        notes=["simulated"],
    )
    assert result.simulated is True
    assert result.success is True
    assert result.applied_value == pytest.approx(1500.0)


# ── ActionLog ────────────────────────────────────────────────────────────────────────

def _make_gate_verdict(action_id: str, approved: bool) -> "GateVerdict":  # type: ignore[name-defined]
    from eie.control.gate import GateVerdict
    return GateVerdict(
        action_id=action_id,
        approved=approved,
        reasons=["test"],
        evaluated_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


def _make_record(action: ControlAction, approved: bool = True) -> ActionRecord:
    verdict = _make_gate_verdict(action.action_id, approved)
    return ActionRecord(
        action=action,
        gate_verdict=verdict,
        executed_at=action.timestamp if approved else None,
        observed_outcome=None,
    )


def test_action_log_append(action: ControlAction, now: datetime) -> None:
    log = ActionLog()
    record = _make_record(action)
    log.append(record)
    assert len(log) == 1


def test_action_log_no_duplicate_ids(action: ControlAction) -> None:
    log = ActionLog()
    record = _make_record(action)
    log.append(record)
    with pytest.raises(ValueError, match="already exists"):
        log.append(record)


def test_action_log_for_asset(now: datetime) -> None:
    log = ActionLog()
    a1 = ControlAction(
        action_id="a1", timestamp=now, target=ControlTarget.BATTERY_CHARGE,
        target_value=100.0, unit="W", source_decision_id="d1",
        safety_class="battery", risk_level="low",
    )
    a2 = ControlAction(
        action_id="a2", timestamp=now, target=ControlTarget.THERMAL_STORAGE_CHARGE,
        target_value=500.0, unit="W", source_decision_id="d1",
        safety_class="thermal", risk_level="low",
    )
    log.append(_make_record(a1))
    log.append(_make_record(a2))
    battery_records = log.for_asset(ControlTarget.BATTERY_CHARGE)
    assert len(battery_records) == 1
    assert battery_records[0].action.action_id == "a1"


def test_action_log_recent(now: datetime) -> None:
    log = ActionLog()
    for i in range(5):
        a = ControlAction(
            action_id=f"a{i}", timestamp=now, target=ControlTarget.LOAD_DEFER,
            target_value=3600.0, unit="s", source_decision_id="d1",
            safety_class="load_shift", risk_level="low",
        )
        log.append(_make_record(a))
    assert len(log.recent(3)) == 3
    assert log.recent(3)[-1].action.action_id == "a4"


def test_action_log_approved_rejected(now: datetime) -> None:
    log = ActionLog()
    a_ok = ControlAction(
        action_id="ok", timestamp=now, target=ControlTarget.BATTERY_CHARGE,
        target_value=100.0, unit="W", source_decision_id="d1",
        safety_class="battery", risk_level="low",
    )
    a_bad = ControlAction(
        action_id="bad", timestamp=now, target=ControlTarget.BATTERY_DISCHARGE,
        target_value=9000.0, unit="W", source_decision_id="d1",
        safety_class="battery", risk_level="low",
    )
    log.append(_make_record(a_ok, approved=True))
    log.append(_make_record(a_bad, approved=False))
    assert len(log.approved()) == 1
    assert len(log.rejected()) == 1


def test_action_log_iter(action: ControlAction) -> None:
    log = ActionLog()
    log.append(_make_record(action))
    records = list(log)
    assert len(records) == 1
