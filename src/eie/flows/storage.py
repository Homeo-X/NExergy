"""Storage state schemas."""

from __future__ import annotations

from dataclasses import dataclass

from eie.core.temperature import require_ratio_temperature
from eie.core.units import ENERGY
from eie.flows.base import (
    Metadata,
    require_binding,
    require_non_negative,
    require_positive,
    require_probability,
)


@dataclass(frozen=True)
class BatteryState:
    """Battery state bound to boundary and reference metadata."""

    storage_id: str
    stored_energy_j: float
    soc: float
    soh: float
    reserve_energy_j: float
    boundary_id: str
    reference_state_id: str
    metadata: Metadata

    def __post_init__(self) -> None:
        if not self.storage_id:
            raise ValueError("storage_id is required")
        require_non_negative(self.stored_energy_j, "stored_energy_j")
        require_probability(self.soc, "soc")
        require_probability(self.soh, "soh")
        require_non_negative(self.reserve_energy_j, "reserve_energy_j")
        if self.reserve_energy_j > self.stored_energy_j:
            raise ValueError("reserve_energy_j cannot exceed stored_energy_j")
        require_binding(self.boundary_id, self.reference_state_id)
        self.metadata.require_dimension(ENERGY)


@dataclass(frozen=True)
class ThermalLayer:
    """One thermal storage layer; no averaging is performed by the model."""

    temperature_k: float
    energy_j: float

    def __post_init__(self) -> None:
        require_ratio_temperature(self.temperature_k, "temperature_k")
        require_non_negative(self.energy_j, "energy_j")


@dataclass(frozen=True)
class ThermalStorageState:
    """Stratified thermal storage state."""

    storage_id: str
    layers: list[ThermalLayer]
    boundary_id: str
    reference_state_id: str
    metadata: Metadata

    def __post_init__(self) -> None:
        if not self.storage_id:
            raise ValueError("storage_id is required")
        if not self.layers:
            raise ValueError("thermal storage must have at least one layer")
        for layer in self.layers:
            if not isinstance(layer, ThermalLayer):
                raise ValueError("layers must contain ThermalLayer instances")
        require_binding(self.boundary_id, self.reference_state_id)
        self.metadata.require_dimension(ENERGY)
