"""Loss fingerprint records for avoidable exergy destruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from eie.flows.base import require_binding, require_non_negative, require_probability


@dataclass(frozen=True)
class LossFingerprint:
    """Structured description of a repeated or significant exergy loss."""

    loss_fingerprint_id: str
    timestamp: datetime
    boundary_id: str
    reference_state_id: str
    destroyed_exergy_j: float
    rejected_recoverable_exergy_j: float
    quality_waste_j: float
    curtailment_j: float
    storage_loss_j: float
    thermal_transfer_loss_j: float
    avoidability_score: float
    causal_hypotheses: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.loss_fingerprint_id:
            raise ValueError("loss_fingerprint_id is required")
        require_binding(self.boundary_id, self.reference_state_id)
        require_non_negative(self.destroyed_exergy_j, "destroyed_exergy_j")
        require_non_negative(self.rejected_recoverable_exergy_j, "rejected_recoverable_exergy_j")
        require_non_negative(self.quality_waste_j, "quality_waste_j")
        require_non_negative(self.curtailment_j, "curtailment_j")
        require_non_negative(self.storage_loss_j, "storage_loss_j")
        require_non_negative(self.thermal_transfer_loss_j, "thermal_transfer_loss_j")
        require_probability(self.avoidability_score, "avoidability_score")
