"""Safety gate for Phase 3 closed-loop control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from eie.control.action import ActionStatus, ControlAction, ControlTarget
from eie.optimization.snapshot import SiteSnapshot


@dataclass(frozen=True)
class GateVerdict:
    action_id: str
    approved: bool
    reasons: list[str]
    evaluated_at: datetime


class SafetyGate:
    ALLOWED_SAFETY_CLASSES: frozenset[str] = frozenset({"thermal", "battery", "load_shift", "waste_heat"})
    BLOCKED_CLASSES: frozenset[str] = frozenset({"critical_load", "hydrogen", "emergency_reserve", "grid_islanding", "fleet_trade"})

    def __init__(self, *, battery_protected_reserve: float = 0.20,
                 thermal_comfort_lower_k: float = 291.15, thermal_comfort_upper_k: float = 299.15,
                 waste_heat_max_temp_k: float = 373.15) -> None:
        self.battery_protected_reserve = battery_protected_reserve
        self.thermal_comfort_lower_k = thermal_comfort_lower_k
        self.thermal_comfort_upper_k = thermal_comfort_upper_k
        self.waste_heat_max_temp_k = waste_heat_max_temp_k

    def evaluate(self, action: ControlAction, snapshot: SiteSnapshot) -> GateVerdict:
        reasons: list[str] = []
        now = datetime.now(timezone.utc)
        if action.safety_class in self.BLOCKED_CLASSES:
            return GateVerdict(action_id=action.action_id, approved=False,
                reasons=[f"safety_class '{action.safety_class}' is blocked in Phase 3; allowed: {sorted(self.ALLOWED_SAFETY_CLASSES)}"],
                evaluated_at=now)
        if action.safety_class not in self.ALLOWED_SAFETY_CLASSES:
            return GateVerdict(action_id=action.action_id, approved=False,
                reasons=[f"unknown safety_class '{action.safety_class}'; allowed: {sorted(self.ALLOWED_SAFETY_CLASSES)}"],
                evaluated_at=now)
        if action.risk_level != "low":
            reasons.append(f"Phase 3 gate only admits risk_level='low'; got '{action.risk_level}'")
        if action.safety_class == "battery":
            reasons.extend(self._check_battery(action, snapshot))
        if action.safety_class == "thermal":
            reasons.extend(self._check_thermal(action, snapshot))
        if action.safety_class == "load_shift" and not action.is_reversible:
            reasons.append("load_shift actions must be reversible in Phase 3")
        if action.safety_class == "waste_heat":
            reasons.extend(self._check_waste_heat(action))
        approved = len(reasons) == 0
        if approved:
            reasons.append(f"all gate checks passed for {action.safety_class} action")
        return GateVerdict(action_id=action.action_id, approved=approved, reasons=reasons, evaluated_at=now)

    def _check_battery(self, action: ControlAction, snapshot: SiteSnapshot) -> list[str]:
        failures: list[str] = []
        current_soc = snapshot.battery_state.soc
        capacity_j = snapshot.battery_constraints.capacity_j
        rte = snapshot.battery_constraints.round_trip_efficiency
        horizon_s = snapshot.horizon_duration_s
        if action.target == ControlTarget.BATTERY_DISCHARGE:
            discharge_energy_j = action.target_value * horizon_s
            soc_delta = discharge_energy_j / (capacity_j * rte) if capacity_j > 0 else 0.0
            projected_soc = current_soc - soc_delta
            if projected_soc < self.battery_protected_reserve:
                failures.append(f"battery discharge would bring SOC to {projected_soc:.3f}, below protected reserve {self.battery_protected_reserve:.3f}")
        elif action.target == ControlTarget.BATTERY_CHARGE:
            if action.target_value > snapshot.battery_constraints.max_charge_power_w:
                failures.append(f"battery charge power {action.target_value:.1f} W exceeds max {snapshot.battery_constraints.max_charge_power_w:.1f} W")
        return failures

    def _check_thermal(self, action: ControlAction, snapshot: SiteSnapshot) -> list[str]:
        failures: list[str] = []
        if action.target in (ControlTarget.BUILDING_SETPOINT, ControlTarget.PRE_COOL_SETPOINT):
            if action.target_value < self.thermal_comfort_lower_k:
                failures.append(f"setpoint {action.target_value:.2f} K below comfort lower bound {self.thermal_comfort_lower_k:.2f} K")
            if action.target_value > self.thermal_comfort_upper_k:
                failures.append(f"setpoint {action.target_value:.2f} K above comfort upper bound {self.thermal_comfort_upper_k:.2f} K")
        elif action.target == ControlTarget.THERMAL_STORAGE_DISCHARGE:
            if action.target_value > snapshot.thermal_storage_constraints.max_discharge_power_w:
                failures.append(f"thermal discharge {action.target_value:.1f} W exceeds max {snapshot.thermal_storage_constraints.max_discharge_power_w:.1f} W")
        elif action.target == ControlTarget.THERMAL_STORAGE_CHARGE:
            if action.target_value > snapshot.thermal_storage_constraints.max_charge_power_w:
                failures.append(f"thermal charge {action.target_value:.1f} W exceeds max {snapshot.thermal_storage_constraints.max_charge_power_w:.1f} W")
        return failures

    def _check_waste_heat(self, action: ControlAction) -> list[str]:
        failures: list[str] = []
        if action.unit == "K" and action.target_value > self.waste_heat_max_temp_k:
            failures.append(f"waste-heat routing temperature {action.target_value:.2f} K exceeds safe bound {self.waste_heat_max_temp_k:.2f} K")
        return failures

    def evaluate_batch(self, actions: Sequence[ControlAction], snapshot: SiteSnapshot) -> list[GateVerdict]:
        return [self.evaluate(a, snapshot) for a in actions]
