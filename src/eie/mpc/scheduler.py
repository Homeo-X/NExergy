"""Rolling-horizon Model Predictive Control scheduler."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from eie.control.gate import SafetyGate
from eie.control.policy import ControlPolicy
from eie.forecasting.bundle import ForecastBundle
from eie.mpc.schedule import MPCSchedule, ScheduledAction
from eie.optimization.optimizer import ShadowOptimizer
from eie.optimization.snapshot import SiteSnapshot


@dataclass
class MPCScheduler:
    optimizer: ShadowOptimizer
    policies: list[ControlPolicy]
    gate: SafetyGate

    def schedule(self, snapshot: SiteSnapshot, bundle: ForecastBundle, *, created_at: datetime | None = None) -> MPCSchedule:
        now = created_at or datetime.now(timezone.utc)
        steps: list[ScheduledAction] = []
        total_improvement = 0.0
        for step in bundle.horizon.steps:
            pv_point = bundle.pv_at(step.step_index)
            load_point = bundle.load_at(step.step_index)
            forecast_snapshot = self._build_forecast_snapshot(snapshot, pv_point.pv_power_w,
                load_point.thermal_load_w, load_point.electrical_base_load_w, step.start)
            opt_result = self.optimizer.optimise(forecast_snapshot, at=step.start)
            actions = []
            if opt_result.recommended_decision is not None:
                for policy in self.policies:
                    actions.extend(policy.actions_from_decision(opt_result.recommended_decision, forecast_snapshot, step.start))
            verdicts = self.gate.evaluate_batch(actions, forecast_snapshot)
            has_approved = any(v.approved for v in verdicts)
            improvement_j = 0.0
            if opt_result.recommended_decision is not None:
                score = opt_result.objective_scores.get(opt_result.recommended_decision.decision_id)
                if score is not None:
                    destruction = score.get("exergy_destruction_j")
                    if destruction is not None:
                        improvement_j = max(0.0, -destruction)
            total_improvement += improvement_j
            steps.append(ScheduledAction(step=step, optimization_result=opt_result, control_actions=actions,
                gate_verdicts=verdicts, approved=has_approved, forecast_exergy_improvement_j=improvement_j))
        return MPCSchedule(schedule_id=str(uuid.uuid4()), created_at=now, horizon=bundle.horizon,
            snapshot_id=snapshot.snapshot_id, steps=steps, total_forecast_exergy_improvement_j=total_improvement)

    def _build_forecast_snapshot(self, base: SiteSnapshot, pv_power_w: float,
                                  thermal_load_w: float, electrical_load_w: float, timestamp: datetime) -> SiteSnapshot:
        import dataclasses
        return dataclasses.replace(base, snapshot_id=f"forecast:{base.snapshot_id}:{timestamp.isoformat()}",
            timestamp=timestamp, available_pv_power_w=max(0.0, pv_power_w),
            building_electric_load_w=max(0.0, electrical_load_w), heat_demand_w=max(0.0, thermal_load_w))
