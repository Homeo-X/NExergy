"""Site snapshot for shadow-mode optimization.

A SiteSnapshot captures the instantaneous physical state of the site that
the optimizer reasons about.  It is immutable and read-only; the optimizer
never modifies it.

The snapshot deliberately does NOT expose hardware control methods.
Hardware actuation belongs in a separate safety layer outside the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from eie.boundary.boundary import Boundary
from eie.core.errors import BoundaryError, DomainError, MissingReferenceError
from eie.core.temperature import require_ratio_temperature
from eie.flows.base import require_non_negative, require_probability
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.reference.state import ReferenceState


def _require_positive_finite(v: float, name: str) -> None:
    if not isfinite(v) or v <= 0:
        raise DomainError(f"{name} must be finite and > 0")


@dataclass(frozen=True)
class BatteryConstraints:
    """Physical and operational constraints for a battery asset."""

    max_charge_power_w: float
    max_discharge_power_w: float
    min_soc: float          # operational minimum SOC [0, 1]
    max_soc: float          # operational maximum SOC [0, 1]
    round_trip_efficiency: float   # DC-DC efficiency [0, 1]
    capacity_j: float              # total usable energy at SoH=1

    def __post_init__(self) -> None:
        _require_positive_finite(self.max_charge_power_w, "max_charge_power_w")
        _require_positive_finite(self.max_discharge_power_w, "max_discharge_power_w")
        require_probability(self.min_soc, "min_soc")
        require_probability(self.max_soc, "max_soc")
        if self.max_soc <= self.min_soc:
            raise DomainError("max_soc must be > min_soc")
        require_probability(self.round_trip_efficiency, "round_trip_efficiency")
        _require_positive_finite(self.capacity_j, "capacity_j")

    def usable_charge_j(self, current_soc: float) -> float:
        """Return energy [J] available to charge from current SOC to max_soc."""
        return max(0.0, (self.max_soc - current_soc) * self.capacity_j)

    def usable_discharge_j(self, current_soc: float) -> float:
        """Return energy [J] available to discharge from current SOC to min_soc."""
        return max(0.0, (current_soc - self.min_soc) * self.capacity_j)


@dataclass(frozen=True)
class ThermalStorageConstraints:
    """Physical and operational constraints for a thermal storage asset."""

    max_charge_power_w: float       # max rate energy can be added
    max_discharge_power_w: float    # max rate energy can be extracted
    charge_efficiency: float        # fraction of input that is stored
    discharge_efficiency: float     # fraction of stored that is delivered
    standby_loss_w: float           # constant parasitic loss

    def __post_init__(self) -> None:
        _require_positive_finite(self.max_charge_power_w, "max_charge_power_w")
        _require_positive_finite(self.max_discharge_power_w, "max_discharge_power_w")
        require_probability(self.charge_efficiency, "charge_efficiency")
        require_probability(self.discharge_efficiency, "discharge_efficiency")
        require_non_negative(self.standby_loss_w, "standby_loss_w")


@dataclass(frozen=True)
class GridConstraints:
    """Grid connection limits for the site."""

    import_limit_w: float       # maximum import rate from grid
    export_limit_w: float       # maximum export rate to grid
    import_tariff_per_kwh: float  # import electricity price [currency/kWh]
    export_tariff_per_kwh: float  # feed-in / export price [currency/kWh]
    carbon_intensity_kg_per_kwh: float  # grid carbon intensity

    def __post_init__(self) -> None:
        require_non_negative(self.import_limit_w, "import_limit_w")
        require_non_negative(self.export_limit_w, "export_limit_w")
        require_non_negative(self.import_tariff_per_kwh, "import_tariff_per_kwh")
        require_non_negative(self.export_tariff_per_kwh, "export_tariff_per_kwh")
        require_non_negative(self.carbon_intensity_kg_per_kwh, "carbon_intensity_kg_per_kwh")


@dataclass(frozen=True)
class SiteSnapshot:
    """Complete, immutable description of the site state for one optimization call.

    This is the sole input to ShadowOptimizer.optimise().  It carries:
    - The exergy reference environment and accounting boundary
    - Available renewable generation
    - Battery and thermal storage states and constraints
    - Electrical and thermal demand forecasts
    - Grid connection limits and tariffs
    - The accounting period to optimise over (horizon)

    The snapshot does NOT expose write methods or references to hardware.
    """

    snapshot_id: str
    timestamp: datetime
    reference_state: ReferenceState
    boundary: Boundary
    horizon_duration_s: float           # accounting period length [s]
    # Generation
    available_pv_power_w: float         # total forecasted PV power for period
    # Storage
    battery_state: BatteryState
    battery_constraints: BatteryConstraints
    thermal_storage_state: ThermalStorageState
    thermal_storage_constraints: ThermalStorageConstraints
    # Demand
    building_electric_load_w: float     # base building electrical load
    heat_pump_rated_cop: float          # rated COP of heat pump (heating mode)
    heat_pump_max_power_w: float        # rated electrical input to heat pump
    delivered_heat_temperature_k: float # supply temperature for space heating
    heat_demand_w: float                # instantaneous thermal demand
    # Grid
    grid_constraints: GridConstraints

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise DomainError("snapshot_id is required")
        if not isinstance(self.timestamp, datetime):
            raise DomainError("timestamp must be a datetime")
        if self.reference_state is None:
            raise MissingReferenceError("reference_state is required")
        if self.boundary is None:
            raise BoundaryError("boundary is required")
        if (
            self.boundary.reference_state_id
            != self.reference_state.reference_state_id
        ):
            raise BoundaryError(
                "boundary.reference_state_id does not match reference_state.reference_state_id"
            )
        _require_positive_finite(self.horizon_duration_s, "horizon_duration_s")
        require_non_negative(self.available_pv_power_w, "available_pv_power_w")
        require_non_negative(self.building_electric_load_w, "building_electric_load_w")
        _require_positive_finite(self.heat_pump_rated_cop, "heat_pump_rated_cop")
        _require_positive_finite(self.heat_pump_max_power_w, "heat_pump_max_power_w")
        require_ratio_temperature(self.delivered_heat_temperature_k, "delivered_heat_temperature_k")
        require_non_negative(self.heat_demand_w, "heat_demand_w")

    @property
    def horizon_start(self) -> datetime:
        return self.timestamp

    @property
    def horizon_end(self) -> datetime:
        return self.timestamp + timedelta(seconds=self.horizon_duration_s)

    @property
    def reference_temperature_k(self) -> float:
        return self.reference_state.ambient_temperature_k
