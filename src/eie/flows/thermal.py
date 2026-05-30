"""Thermal flow schema."""

from __future__ import annotations

from dataclasses import dataclass

from eie.core.temperature import require_ratio_temperature
from eie.core.units import POWER
from eie.flows.base import Metadata, require_binding, require_non_negative, require_positive


@dataclass(frozen=True)
class ThermalFlow:
    """Heat-flow schema; hot/cold interpretation belongs to exergy functions."""

    flow_id: str
    heat_rate_w: float
    source_temperature_k: float
    sink_temperature_k: float | None
    boundary_temperature_k: float | None
    reference_state_id: str
    boundary_id: str
    metadata: Metadata

    def __post_init__(self) -> None:
        if not self.flow_id:
            raise ValueError("flow_id is required")
        require_non_negative(self.heat_rate_w, "heat_rate_w")
        require_ratio_temperature(self.source_temperature_k, "source_temperature_k")
        if self.sink_temperature_k is not None:
            require_ratio_temperature(self.sink_temperature_k, "sink_temperature_k")
        if self.boundary_temperature_k is not None:
            require_ratio_temperature(self.boundary_temperature_k, "boundary_temperature_k")
        require_binding(self.boundary_id, self.reference_state_id)
        self.metadata.require_dimension(POWER)
