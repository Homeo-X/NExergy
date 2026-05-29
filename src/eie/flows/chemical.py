"""Chemical flow schema for model-specific chemical exergy accounting."""

from __future__ import annotations

from dataclasses import dataclass

from eie.core.units import MASS_FLOW
from eie.flows.base import Metadata, require_binding, require_non_negative


@dataclass(frozen=True)
class ChemicalFlow:
    """Simple chemical flow with explicit model identity.

    Chemical exergy is intentionally model-specific. The kernel records the
    selected model and reference environment instead of pretending that fuel
    lower-heating-value is a universal exergy number.
    """

    flow_id: str
    mass_flow_kg_s: float
    specific_chemical_exergy_j_per_kg: float
    model_id: str
    reference_environment_id: str
    boundary_id: str
    reference_state_id: str
    metadata: Metadata

    def __post_init__(self) -> None:
        if not self.flow_id:
            raise ValueError("flow_id is required")
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.reference_environment_id:
            raise ValueError("reference_environment_id is required")
        require_non_negative(self.mass_flow_kg_s, "mass_flow_kg_s")
        require_non_negative(self.specific_chemical_exergy_j_per_kg, "specific_chemical_exergy_j_per_kg")
        require_binding(self.boundary_id, self.reference_state_id)
        self.metadata.require_dimension(MASS_FLOW)

    @property
    def exergy_rate_w(self) -> float:
        return self.mass_flow_kg_s * self.specific_chemical_exergy_j_per_kg
