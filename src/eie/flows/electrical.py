"""Electrical flow schema."""

from __future__ import annotations

from dataclasses import dataclass

from eie.core.units import POWER
from eie.flows.base import (
    Metadata,
    require_binding,
    require_non_negative,
    require_optional_probability,
    require_positive,
    require_probability,
)


@dataclass(frozen=True)
class ElectricalFlow:
    """Usable real electrical power with separate service-quality metadata."""

    flow_id: str
    real_power_w: float
    voltage_v: float | None
    frequency_hz: float | None
    power_factor: float | None
    thd: float | None
    availability_factor: float | None
    boundary_id: str
    reference_state_id: str
    metadata: Metadata

    def __post_init__(self) -> None:
        if not self.flow_id:
            raise ValueError("flow_id is required")
        require_non_negative(self.real_power_w, "real_power_w")
        if self.voltage_v is not None:
            require_positive(self.voltage_v, "voltage_v")
        if self.frequency_hz is not None:
            require_positive(self.frequency_hz, "frequency_hz")
        require_optional_probability(self.power_factor, "power_factor")
        require_optional_probability(self.availability_factor, "availability_factor")
        if self.thd is not None:
            require_probability(self.thd, "thd")
        require_binding(self.boundary_id, self.reference_state_id)
        self.metadata.require_dimension(POWER)
