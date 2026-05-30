"""Storage exergy functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eie.core.temperature import require_ratio_temperature
from eie.exergy.cooling import cooling_service_exergy_rate
from eie.exergy.heat import heat_exergy_rate
from eie.flows.base import require_non_negative, require_positive, require_probability
from eie.flows.storage import ThermalLayer


def battery_stored_exergy(stored_energy_j: float, availability_factor: float = 1.0) -> float:
    """Return approximate stored electrical exergy in a battery."""

    require_non_negative(stored_energy_j, "stored_energy_j")
    require_probability(availability_factor, "availability_factor")
    return stored_energy_j * availability_factor


@dataclass(frozen=True)
class ThermalStorageExergyBreakdown:
    hot_exergy_j: float
    cold_exergy_j: float

    @property
    def total_exergy_j(self) -> float:
        return self.hot_exergy_j + self.cold_exergy_j


def _coerce_layer(layer: ThermalLayer | dict[str, float] | tuple[float, float] | Any) -> ThermalLayer:
    if isinstance(layer, ThermalLayer):
        return layer
    if isinstance(layer, dict):
        return ThermalLayer(temperature_k=float(layer["temperature_k"]), energy_j=float(layer["energy_j"]))
    if isinstance(layer, tuple):
        temperature_k, energy_j = layer
        return ThermalLayer(temperature_k=float(temperature_k), energy_j=float(energy_j))
    if hasattr(layer, "temperature_k") and hasattr(layer, "energy_j"):
        return ThermalLayer(temperature_k=float(layer.temperature_k), energy_j=float(layer.energy_j))
    raise TypeError("thermal storage layer must expose temperature_k and energy_j")


def thermal_storage_exergy_breakdown(
    layers: list[ThermalLayer] | list[dict[str, float]] | list[tuple[float, float]],
    reference_temperature_k: float,
    *,
    include_cold_exergy: bool = True,
) -> ThermalStorageExergyBreakdown:
    """Integrate exergy layer by layer without average-temperature shortcuts."""

    require_ratio_temperature(reference_temperature_k, "reference_temperature_k")
    hot = 0.0
    cold = 0.0
    for raw_layer in layers:
        layer = _coerce_layer(raw_layer)
        if layer.energy_j == 0.0 or layer.temperature_k == reference_temperature_k:
            continue
        if layer.temperature_k > reference_temperature_k:
            hot += heat_exergy_rate(layer.energy_j, layer.temperature_k, reference_temperature_k)
        elif include_cold_exergy:
            cold += cooling_service_exergy_rate(layer.energy_j, layer.temperature_k, reference_temperature_k)
    return ThermalStorageExergyBreakdown(hot_exergy_j=hot, cold_exergy_j=cold)


def thermal_storage_exergy(
    layers: list[ThermalLayer] | list[dict[str, float]] | list[tuple[float, float]],
    reference_temperature_k: float,
    *,
    include_cold_exergy: bool = True,
) -> float:
    """Return total stratified thermal storage exergy."""

    return thermal_storage_exergy_breakdown(
        layers,
        reference_temperature_k,
        include_cold_exergy=include_cold_exergy,
    ).total_exergy_j
