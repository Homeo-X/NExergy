"""Safety gate for Phase 3 closed-loop control.

The SafetyGate is the independent safety layer that sits between the optimizer
and any hardware actuation.  It must be consulted before every ControlAction
is executed.  It cannot be bypassed.

Phase 3 permits only low-risk safety classes: thermal, battery, load_shift,
waste_heat.  Any action targeting a blocked class is unconditionally rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from eie.control.action import ActionStatus, ControlAction, ControlTarget
from eie.optimization.snapshot import SiteSnapshot


@dataclass(frozen=True)
class GateVerdict:
    """Immutable result of a SafetyGate evaluation.

    Parameters
    ----------
    action_id : str
        Matches the ControlAction.action_id that was evaluated.
    approved : bool
        True iff every gate check passed and the action is safe to execute.
    reasons : list[str]
        Human-readable explanations — present for both approvals and rejections.
    evaluated_at : datetime
        When the gate evaluation completed.
    """

    action_id: str
    approved: bool
    reasons: list[str]
    evaluated_at: datetime


class SafetyGate:
    """Independent safety layer for Phase 3 low-risk closed-loop control.

    Rules
    -----
    - Actions in BLOCKED_CLASSES are unconditionally rejected.
    - Battery actions must keep the SOC above the protected reserve.
    - Thermal actions must keep setpoints within comfort bounds.
    - Load-shift actions must be reversible.
    - Waste-heat actions must target temperatures within safe bounds.
    - Only risk_level="low" is admitted in Phase 3.

    Parameters
    ----------
    battery_protected_reserve : float
        Minimum battery SOC fraction that must remain after any battery action.
        Default 0.20 (20%).
    thermal_comfort_lower_k : float
        Minimum permitted thermal setpoint [K].  Default 291.15 K (18 °C).
    thermal_comfort_upper_k : float
        Maximum permitted thermal setpoint [K].  Default 299.15 K (26 °C).
    waste_heat_max_temp_k : float
        Maximum safe waste-heat routing temperature [K].  Default 373.15 K (100 °C).
    """

    ALLOWED_SAFETY_CLASSES: frozenset[str] = frozenset(
        {"thermal", "battery", "load_shift", "waste_heat"}
    )
    BLOCKED_CLASSES: frozenset[str] = frozenset(
        {
            "critical_load",
            "hydrogen",
            "emergency_reserve",
            "grid_islanding",
            "fleet_trade",
        }
    )

    def __init__(
        self,
        *,
        battery_protected_reserve: float = 0.20,
        thermal_comfort_lower_k: float = 291.15,
        thermal_comfort_upper_k: float = 299.15,
        waste_heat_max_temp_k: float = 373.15,
    ) -> None:
        self.battery_protected_reserve = battery_protected_reserve
        self.thermal_comfort_lower_k = thermal_comfort_lower_k
        self.thermal_comfort_upper_k = thermal_comfort_upper_k
        self.waste_heat_max_temp_k = waste_heat_max_temp_k

    def evaluate(self, action: ControlAction, snapshot: SiteSnapshot) -> GateVerdict:
        """Evaluate a single ControlAction against all gate rules."""
        reasons: list[str] = []
        now = datetime.now(timezone.utc)

        if action.safety_class in self.BLOCKED_CLASSES:
            return GateVerdict(
                action_id=action.action_id,
                approved=False,
                reasons=[
                    f"safety_class '{action.safety_class}' is blocked in Phase 3; "
                    f"allowed: {sorted(self.ALLOWED_SAFETY_CLASSES)}"
                ],
                evaluated_at=now,
            )

        if action.safety_class not in self.ALLOWED_SAFETY_CLASSES:
            return GateVerdict(
                action_id=action.action_id,
                approved=False,
                reasons=[
                    f"unknown safety_class '{action.safety_class}'; "
                    f"allowed: {sorted(self.ALLOWED_SAFETY_CLASSES)}"
                ],
                evaluated_at=now,
            )

        if action.risk_level != "low":
            reasons.append(
                f"Phase 3 gate only admits risk_level='low'; got '{action.risk_level}'"
            )

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

        return GateVerdict(
            action_id=action.action_id,
            approved=approved,
            reasons=reasons,
            evaluated_at=now,
        )

    def _check_battery(self, action: ControlAction, snapshot: SiteSnapshot) -> list[str]:
        failures: list[str] = []
        current_soc = snapshot.battery_state.soc
        capacity_j = snapshot.battery_constraints.capacity_j
        rte = snapshot.battery_constraints.round_trip_efficiency
        horizon_s = snapshot.horizon_duration_s

        if action.target == ControlTarget.BATTERY_DISCHARGE:
            discharge_power_w = action.target_value
            discharge_energy_j = discharge_power_w * horizon_s
            soc_delta = discharge_energy_j / (capacity_j * rte) if capacity_j > 0 else 0.0
            projected_soc = current_soc - soc_delta
            if projected_soc < self.battery_protected_reserve:
                failures.append(
                    f"battery discharge would bring SOC to {projected_soc:.3f}, "
                    f"below protected reserve {self.battery_protected_reserve:.3f}"
                )
        elif action.target == ControlTarget.BATTERY_CHARGE:
            if action.target_value > snapshot.battery_constraints.max_charge_power_w:
                failures.append(
                    f"battery charge power {action.target_value:.1f} W exceeds "
                    f"max {snapshot.battery_constraints.max_charge_power_w:.1f} W"
                )
        return failures

    def _check_thermal(self, action: ControlAction, snapshot: SiteSnapshot) -> list[str]:
        failures: list[str] = []
        if action.target in (
            ControlTarget.BUILDING_SETPOINT,
            ControlTarget.PRE_COOL_SETPOINT,
        ):
            if action.target_value < self.thermal_comfort_lower_k:
                failures.append(
                    f"setpoint {action.target_value:.2f} K below comfort lower bound "
                    f"{self.thermal_comfort_lower_k:.2f} K"
                )
            if action.target_value > self.thermal_comfort_upper_k:
                failures.append(
                    f"setpoint {action.target_value:.2f} K above comfort upper bound "
                    f"{self.thermal_comfort_upper_k:.2f} K"
                )
        elif action.target == ControlTarget.THERMAL_STORAGE_DISCHARGE:
            if action.target_value > snapshot.thermal_storage_constraints.max_discharge_power_w:
                failures.append(
                    f"thermal discharge {action.target_value:.1f} W exceeds "
                    f"max {snapshot.thermal_storage_constraints.max_discharge_power_w:.1f} W"
                )
        elif action.target == ControlTarget.THERMAL_STORAGE_CHARGE:
            if action.target_value > snapshot.thermal_storage_constraints.max_charge_power_w:
                failures.append(
                    f"thermal charge {action.target_value:.1f} W exceeds "
                    f"max {snapshot.thermal_storage_constraints.max_charge_power_w:.1f} W"
                )
        return failures

    def _check_waste_heat(self, action: ControlAction) -> list[str]:
        failures: list[str] = []
        if action.unit == "K" and action.target_value > self.waste_heat_max_temp_k:
            failures.append(
                f"waste-heat routing temperature {action.target_value:.2f} K exceeds "
                f"safe bound {self.waste_heat_max_temp_k:.2f} K"
            )
        return failures

    def evaluate_batch(
        self, actions: Sequence[ControlAction], snapshot: SiteSnapshot
    ) -> list[GateVerdict]:
        """Evaluate a sequence of actions, returning one verdict per action."""
        return [self.evaluate(a, snapshot) for a in actions]
