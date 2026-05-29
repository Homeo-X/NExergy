"""Control policies for Phase 3 low-risk closed-loop control."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from eie.control.action import ActionStatus, ControlAction, ControlTarget
from eie.optimization.decision import DispatchDecision
from eie.optimization.snapshot import SiteSnapshot


class ControlPolicy(ABC):
    @abstractmethod
    def actions_from_decision(self, decision: DispatchDecision, snapshot: SiteSnapshot, timestamp: datetime) -> list[ControlAction]: ...


class BatteryPolicy(ControlPolicy):
    def __init__(self, protected_reserve_fraction: float = 0.20) -> None:
        if not (0.0 <= protected_reserve_fraction < 1.0):
            raise ValueError("protected_reserve_fraction must be in [0, 1)")
        self.protected_reserve_fraction = protected_reserve_fraction

    def actions_from_decision(self, decision: DispatchDecision, snapshot: SiteSnapshot, timestamp: datetime) -> list[ControlAction]:
        if not decision.is_usable:
            return []
        charge_fraction = decision.variable("battery_charge_fraction")
        horizon_s = snapshot.horizon_duration_s
        capacity_j = snapshot.battery_constraints.capacity_j
        rte = snapshot.battery_constraints.round_trip_efficiency
        current_soc = snapshot.battery_state.soc
        max_charge_w = snapshot.battery_constraints.max_charge_power_w
        max_discharge_w = snapshot.battery_constraints.max_discharge_power_w
        if charge_fraction >= 0.5:
            raw_power = (charge_fraction - 0.5) * 2.0 * max_charge_w
            available_j = snapshot.battery_constraints.usable_charge_j(current_soc)
            max_from_capacity = available_j / horizon_s if horizon_s > 0 else 0.0
            target_w = min(raw_power, max_from_capacity, max_charge_w)
            if target_w <= 0.0:
                return []
            return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
                target=ControlTarget.BATTERY_CHARGE, target_value=target_w, unit="W",
                source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
                safety_class="battery", risk_level="low", is_reversible=True)]
        else:
            raw_power = (0.5 - charge_fraction) * 2.0 * max_discharge_w
            available_above_reserve = max(0.0, (current_soc - self.protected_reserve_fraction) * capacity_j)
            max_from_reserve = available_above_reserve / (horizon_s * rte) if horizon_s > 0 else 0.0
            target_w = min(raw_power, max_from_reserve, max_discharge_w)
            if target_w <= 0.0:
                return []
            return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
                target=ControlTarget.BATTERY_DISCHARGE, target_value=target_w, unit="W",
                source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
                safety_class="battery", risk_level="low", is_reversible=True)]


class ThermalStoragePolicy(ControlPolicy):
    def __init__(self, min_comfort_k: float, max_comfort_k: float) -> None:
        if min_comfort_k >= max_comfort_k:
            raise ValueError("min_comfort_k must be < max_comfort_k")
        self.min_comfort_k = min_comfort_k
        self.max_comfort_k = max_comfort_k

    def actions_from_decision(self, decision: DispatchDecision, snapshot: SiteSnapshot, timestamp: datetime) -> list[ControlAction]:
        if not decision.is_usable:
            return []
        ts_fraction = decision.variable("thermal_storage_fraction")
        delivered_temp_k = snapshot.delivered_heat_temperature_k
        if delivered_temp_k < self.min_comfort_k or delivered_temp_k > self.max_comfort_k:
            return []
        if ts_fraction >= 0.5:
            target_w = (ts_fraction - 0.5) * 2.0 * snapshot.thermal_storage_constraints.max_charge_power_w
            if target_w <= 0.0:
                return []
            return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
                target=ControlTarget.THERMAL_STORAGE_CHARGE, target_value=target_w, unit="W",
                source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
                safety_class="thermal", risk_level="low", is_reversible=True)]
        else:
            target_w = (0.5 - ts_fraction) * 2.0 * snapshot.thermal_storage_constraints.max_discharge_power_w
            if target_w <= 0.0:
                return []
            return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
                target=ControlTarget.THERMAL_STORAGE_DISCHARGE, target_value=target_w, unit="W",
                source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
                safety_class="thermal", risk_level="low", is_reversible=True)]


class PreCoolingPolicy(ControlPolicy):
    def __init__(self, comfort_lower_k: float, comfort_upper_k: float, activation_threshold: float = 0.3) -> None:
        if comfort_lower_k >= comfort_upper_k:
            raise ValueError("comfort_lower_k must be < comfort_upper_k")
        self.comfort_lower_k = comfort_lower_k
        self.comfort_upper_k = comfort_upper_k
        self.activation_threshold = activation_threshold

    def actions_from_decision(self, decision: DispatchDecision, snapshot: SiteSnapshot, timestamp: datetime) -> list[ControlAction]:
        if not decision.is_usable:
            return []
        hp_fraction = decision.variable("heat_pump_fraction")
        if hp_fraction < self.activation_threshold:
            return []
        scale = (hp_fraction - self.activation_threshold) / (1.0 - self.activation_threshold)
        setpoint_k = self.comfort_upper_k - scale * (self.comfort_upper_k - self.comfort_lower_k)
        setpoint_k = max(self.comfort_lower_k, min(self.comfort_upper_k, setpoint_k))
        return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
            target=ControlTarget.PRE_COOL_SETPOINT, target_value=setpoint_k, unit="K",
            source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
            safety_class="thermal", risk_level="low", is_reversible=True)]


class LoadShiftPolicy(ControlPolicy):
    def __init__(self, noncritical_load_ids: list[str], max_defer_s: float = 3600.0, activation_pv_threshold: float = 0.5) -> None:
        if not noncritical_load_ids:
            raise ValueError("noncritical_load_ids must not be empty")
        self.noncritical_load_ids = list(noncritical_load_ids)
        self.max_defer_s = max_defer_s
        self.activation_pv_threshold = activation_pv_threshold

    def actions_from_decision(self, decision: DispatchDecision, snapshot: SiteSnapshot, timestamp: datetime) -> list[ControlAction]:
        if not decision.is_usable:
            return []
        electric_load = snapshot.building_electric_load_w
        pv_available = snapshot.available_pv_power_w
        if electric_load <= 0 or (pv_available / electric_load) >= self.activation_pv_threshold:
            return []
        return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
            target=ControlTarget.LOAD_DEFER, target_value=self.max_defer_s, unit="s",
            source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
            safety_class="load_shift", risk_level="low", is_reversible=True)
            for _ in self.noncritical_load_ids]


class WasteHeatPolicy(ControlPolicy):
    def __init__(self, max_heat_rate_w: float, safe_temperature_bounds_k: tuple[float, float], activation_hp_threshold: float = 0.6) -> None:
        if max_heat_rate_w <= 0:
            raise ValueError("max_heat_rate_w must be > 0")
        min_k, max_k = safe_temperature_bounds_k
        if min_k >= max_k:
            raise ValueError("safe_temperature_bounds_k: min must be < max")
        self.max_heat_rate_w = max_heat_rate_w
        self.safe_temperature_bounds_k = safe_temperature_bounds_k
        self.activation_hp_threshold = activation_hp_threshold

    def actions_from_decision(self, decision: DispatchDecision, snapshot: SiteSnapshot, timestamp: datetime) -> list[ControlAction]:
        if not decision.is_usable:
            return []
        hp_fraction = decision.variable("heat_pump_fraction")
        if hp_fraction < self.activation_hp_threshold:
            return []
        min_k, max_k = self.safe_temperature_bounds_k
        route_temp_k = (min_k + max_k) / 2.0
        return [ControlAction(action_id=str(uuid.uuid4()), timestamp=timestamp,
            target=ControlTarget.WASTE_HEAT_ROUTE, target_value=route_temp_k, unit="K",
            source_decision_id=decision.decision_id, status=ActionStatus.PENDING,
            safety_class="waste_heat", risk_level="low", is_reversible=False)]
