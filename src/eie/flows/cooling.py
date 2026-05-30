"""Cooling service schema."""

from __future__ import annotations

from dataclasses import dataclass

from eie.core.temperature import require_ratio_temperature
from eie.core.units import POWER
from eie.flows.base import Metadata, require_binding, require_non_negative, require_positive


@dataclass(frozen=True)
class CoolingLoad:
    """Cooling or refrigeration service relative to a reference environment."""

    load_id: str
    cooling_rate_w: float
    cold_temperature_k: float
    reference_state_id: str
    boundary_id: str
    metadata: Metadata

    def __post_init__(self) -> None:
        if not self.load_id:
            raise ValueError("load_id is required")
        require_non_negative(self.cooling_rate_w, "cooling_rate_w")
        require_ratio_temperature(self.cold_temperature_k, "cold_temperature_k")
        require_binding(self.boundary_id, self.reference_state_id)
        self.metadata.require_dimension(POWER)
