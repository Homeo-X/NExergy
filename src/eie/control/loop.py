"""Closed-loop controller for Phase 3 low-risk autonomous control."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from eie.control.action import ActionLog, ActionRecord, ActionStatus, ActuationResult, ControlAction
from eie.control.gate import GateVerdict, SafetyGate
from eie.control.policy import ControlPolicy
from eie.optimization.optimizer import OptimizationResult, ShadowOptimizer
from eie.optimization.snapshot import SiteSnapshot


@dataclass
class LoopConfig:
    optimizer: ShadowOptimizer
    policies: list[ControlPolicy]
    gate: SafetyGate
    max_actions_per_iteration: int = 5


@dataclass(frozen=True)
class LoopIteration:
    iteration_id: str
    timestamp: datetime
    snapshot_id: str
    optimization_result: OptimizationResult
    proposed_actions: list[ControlAction]
    approved_actions: list[ControlAction]
    rejected_actions: list[ControlAction]
    execution_results: list[ActuationResult]
    gate_verdicts: list[GateVerdict]
    exergy_improvement_j: float
    guard_bypass_count: int = 0

    def __post_init__(self) -> None:
        if self.guard_bypass_count != 0:
            raise ValueError("guard_bypass_count must be 0; the control loop never bypasses guards")


def _simulate_execution(action: ControlAction) -> ActuationResult:
    from eie.control.action import ControlTarget
    delta: dict[str, float] = {}
    applied = action.target_value
    if action.target == ControlTarget.BATTERY_CHARGE:
        delta["battery_charge_w"] = applied; delta["soc_delta_estimate"] = applied * 0.001
    elif action.target == ControlTarget.BATTERY_DISCHARGE:
        delta["battery_discharge_w"] = applied; delta["soc_delta_estimate"] = -applied * 0.001
    elif action.target in (ControlTarget.THERMAL_STORAGE_CHARGE, ControlTarget.THERMAL_STORAGE_DISCHARGE):
        delta["thermal_power_w"] = applied
    elif action.target in (ControlTarget.BUILDING_SETPOINT, ControlTarget.PRE_COOL_SETPOINT):
        delta["setpoint_k"] = applied
    elif action.target == ControlTarget.LOAD_DEFER:
        delta["deferred_s"] = applied
    elif action.target == ControlTarget.WASTE_HEAT_ROUTE:
        delta["routed_temp_k"] = applied
    return ActuationResult(action_id=action.action_id, simulated=True, applied_value=applied,
        observed_delta=delta, success=True, notes=["Phase 3 simulated execution — no hardware actuation"])


class ClosedLoopController:
    def __init__(self, config: LoopConfig) -> None:
        self.config = config
        self.action_log = ActionLog()

    def step(self, snapshot: SiteSnapshot, *, at: datetime | None = None) -> LoopIteration:
        timestamp = at if at is not None else datetime.now(timezone.utc)
        iteration_id = str(uuid.uuid4())
        opt_result = self.config.optimizer.optimise(snapshot, at=timestamp)
        proposed: list[ControlAction] = []
        if opt_result.recommended_decision is not None:
            for policy in self.config.policies:
                proposed.extend(policy.actions_from_decision(opt_result.recommended_decision, snapshot, timestamp))
        proposed = proposed[:self.config.max_actions_per_iteration]
        verdicts = self.config.gate.evaluate_batch(proposed, snapshot)
        verdict_map = {v.action_id: v for v in verdicts}
        approved: list[ControlAction] = []
        rejected: list[ControlAction] = []
        for action in proposed:
            if verdict_map[action.action_id].approved:
                approved.append(action.with_status(ActionStatus.GATE_APPROVED))
            else:
                rejected.append(action.with_status(ActionStatus.GATE_REJECTED))
        execution_results: list[ActuationResult] = []
        executed_actions: list[ControlAction] = []
        for action in approved:
            result = _simulate_execution(action)
            execution_results.append(result)
            executed = action.with_status(ActionStatus.EXECUTED if result.success else ActionStatus.FAILED)
            executed_actions.append(executed)
            self.action_log.append(ActionRecord(action=executed, gate_verdict=verdict_map[action.action_id], executed_at=timestamp, observed_outcome=result))
        for action in rejected:
            self.action_log.append(ActionRecord(action=action, gate_verdict=verdict_map[action.action_id], executed_at=None, observed_outcome=None))
        exergy_improvement = 0.0
        if opt_result.recommended_decision is not None:
            score = opt_result.objective_scores.get(opt_result.recommended_decision.decision_id)
            if score is not None:
                destruction = score.get("exergy_destruction_j")
                if destruction is not None:
                    exergy_improvement = max(0.0, -destruction)
        return LoopIteration(iteration_id=iteration_id, timestamp=timestamp, snapshot_id=snapshot.snapshot_id,
            optimization_result=opt_result, proposed_actions=proposed, approved_actions=executed_actions,
            rejected_actions=rejected, execution_results=execution_results, gate_verdicts=verdicts,
            exergy_improvement_j=exergy_improvement, guard_bypass_count=0)

    def run(self, snapshots: Sequence[SiteSnapshot], *, at: datetime | None = None) -> list[LoopIteration]:
        return [self.step(s, at=at if at is not None else s.timestamp) for s in snapshots]
