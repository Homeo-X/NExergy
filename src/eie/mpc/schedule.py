"""MPC schedule output schemas."""

from __future__ import annotations

from dataclasses import dataclass

from eie.control.action import ControlAction
from eie.control.gate import GateVerdict
from eie.forecasting.horizon import ForecastStep, TimeHorizon
from eie.optimization.optimizer import OptimizationResult


@dataclass(frozen=True)
class ScheduledAction:
    step: ForecastStep
    optimization_result: OptimizationResult
    control_actions: list[ControlAction]
    gate_verdicts: list[GateVerdict]
    approved: bool
    forecast_exergy_improvement_j: float

    @property
    def approved_actions(self) -> list[ControlAction]:
        approved_ids = {v.action_id for v in self.gate_verdicts if v.approved}
        return [a for a in self.control_actions if a.action_id in approved_ids]

    @property
    def rejected_actions(self) -> list[ControlAction]:
        rejected_ids = {v.action_id for v in self.gate_verdicts if not v.approved}
        return [a for a in self.control_actions if a.action_id in rejected_ids]


@dataclass(frozen=True)
class MPCSchedule:
    from datetime import datetime
    schedule_id: str
    created_at: "datetime"
    horizon: TimeHorizon
    snapshot_id: str
    steps: list[ScheduledAction]
    total_forecast_exergy_improvement_j: float

    def __post_init__(self) -> None:
        if len(self.steps) != self.horizon.n_steps:
            raise ValueError(f"steps length {len(self.steps)} != horizon.n_steps {self.horizon.n_steps}")

    @property
    def first_step_actions(self) -> list[ControlAction]:
        if not self.steps:
            return []
        return self.steps[0].approved_actions

    @property
    def approved_steps(self) -> list[ScheduledAction]:
        return [s for s in self.steps if s.approved]

    @property
    def n_approved_steps(self) -> int:
        return len(self.approved_steps)
