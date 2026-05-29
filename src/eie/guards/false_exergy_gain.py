"""False exergy gain detector and guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from eie.boundary.boundary import Boundary
from eie.core.enums import Carrier, Severity
from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance
from eie.exergy.quality import validate_quality_factor
from eie.guards.physics_guard import GuardResult
from eie.reference.state import ReferenceState


@dataclass(frozen=True)
class FalseExergyGainFinding:
    code: str
    severity: Severity
    message: str
    evidence: dict[str, object] = field(default_factory=dict)


def detect_false_exergy_gain(
    *,
    exergy_in_j: float | None = None,
    useful_exergy_j: float = 0.0,
    stored_exergy_delta_j: float = 0.0,
    recovered_exergy_j: float = 0.0,
    destroyed_exergy_j: float | None = None,
    entropy_generated_j_per_k: float | None = None,
    reference_state: ReferenceState | None = None,
    boundary: Boundary | None = None,
    reference_state_id: str | None = None,
    boundary_id: str | None = None,
    carrier: Carrier | None = None,
    quality_factor: float | None = None,
    high_grade_exergy_out_j: float | None = None,
    low_grade_exergy_in_j: float | None = None,
    work_input_j: float | None = None,
    at: datetime | None = None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> list[FalseExergyGainFinding]:
    """Return structured findings for impossible or suspicious exergy accounting."""

    findings: list[FalseExergyGainFinding] = []
    effective_reference_id = reference_state.reference_state_id if reference_state else reference_state_id
    effective_boundary_id = boundary.boundary_id if boundary else boundary_id

    if not effective_reference_id:
        findings.append(
            FalseExergyGainFinding(
                "missing_reference_state",
                Severity.CRITICAL,
                "exergy accounting is missing reference state",
            )
        )
    if not effective_boundary_id:
        findings.append(
            FalseExergyGainFinding(
                "missing_boundary",
                Severity.CRITICAL,
                "exergy accounting is missing boundary",
            )
        )
    if reference_state is not None and reference_state.is_stale(at=at, tolerance=tolerance):
        findings.append(
            FalseExergyGainFinding(
                "stale_reference_state",
                Severity.ERROR,
                "reference state is stale",
                {"reference_state_id": reference_state.reference_state_id},
            )
        )
    if boundary is not None and effective_reference_id and boundary.reference_state_id != effective_reference_id:
        findings.append(
            FalseExergyGainFinding(
                "boundary_reference_mismatch",
                Severity.ERROR,
                "boundary reference does not match accounting reference",
                {
                    "boundary_reference_state_id": boundary.reference_state_id,
                    "reference_state_id": effective_reference_id,
                },
            )
        )
    if exergy_in_j is not None:
        output_exergy = useful_exergy_j + stored_exergy_delta_j + recovered_exergy_j
        if exergy_in_j < -tolerance.absolute_j:
            findings.append(
                FalseExergyGainFinding(
                    "negative_exergy_input",
                    Severity.ERROR,
                    "exergy input is negative",
                    {"exergy_in_j": exergy_in_j},
                )
            )
        elif exergy_in_j <= tolerance.absolute_j and output_exergy > tolerance.absolute_j:
            findings.append(
                FalseExergyGainFinding(
                    "exergy_from_zero_input",
                    Severity.CRITICAL,
                    "useful/stored/recovered exergy appears without exergy input",
                    {"exergy_in_j": exergy_in_j, "output_exergy_j": output_exergy},
                )
            )
        elif exergy_in_j > tolerance.absolute_j:
            eta_x = output_exergy / exergy_in_j
            if eta_x > 1.0 + tolerance.efficiency:
                findings.append(
                    FalseExergyGainFinding(
                        "exergy_efficiency_above_one",
                        Severity.CRITICAL,
                        "exergy efficiency exceeds physically plausible limit",
                        {"eta_x": eta_x, "exergy_in_j": exergy_in_j, "output_exergy_j": output_exergy},
                    )
                )
    if destroyed_exergy_j is not None and destroyed_exergy_j < -tolerance.absolute_j:
        findings.append(
            FalseExergyGainFinding(
                "negative_destroyed_exergy",
                Severity.CRITICAL,
                "destroyed exergy is negative beyond tolerance",
                {"destroyed_exergy_j": destroyed_exergy_j},
            )
        )
    if entropy_generated_j_per_k is not None and entropy_generated_j_per_k < -tolerance.entropy_j_per_k:
        findings.append(
            FalseExergyGainFinding(
                "negative_entropy_generation",
                Severity.CRITICAL,
                "entropy generation is negative beyond tolerance",
                {"entropy_generated_j_per_k": entropy_generated_j_per_k},
            )
        )
    if quality_factor is not None and carrier is not None:
        try:
            validate_quality_factor(quality_factor, carrier)
        except Exception as exc:
            findings.append(
                FalseExergyGainFinding(
                    "invalid_quality_factor_for_carrier",
                    Severity.ERROR,
                    str(exc),
                    {"quality_factor": quality_factor, "carrier": Carrier(carrier).value},
                )
            )
    if high_grade_exergy_out_j is not None:
        low_grade = 0.0 if low_grade_exergy_in_j is None else low_grade_exergy_in_j
        work = 0.0 if work_input_j is None else work_input_j
        allowed = low_grade + work
        if high_grade_exergy_out_j > allowed + tolerance.absolute_j:
            findings.append(
                FalseExergyGainFinding(
                    "impossible_heat_upgrade_without_work",
                    Severity.CRITICAL,
                    "high-grade exergy output exceeds low-grade exergy plus work input",
                    {
                        "high_grade_exergy_out_j": high_grade_exergy_out_j,
                        "low_grade_exergy_in_j": low_grade,
                        "work_input_j": work,
                    },
                )
            )
    return findings


class FalseExergyGainGuard:
    """Guard wrapper around the false-exergy-gain detector."""

    guard_name = "FalseExergyGainGuard"

    def __init__(self, tolerance: Tolerance = DEFAULT_TOLERANCE) -> None:
        self.tolerance = tolerance

    def check(
        self,
        *,
        exergy_in_j: float | None = None,
        useful_exergy_j: float = 0.0,
        stored_exergy_delta_j: float = 0.0,
        recovered_exergy_j: float = 0.0,
        destroyed_exergy_j: float | None = None,
        entropy_generated_j_per_k: float | None = None,
        reference_state: ReferenceState | None = None,
        boundary: Boundary | None = None,
        reference_state_id: str | None = None,
        boundary_id: str | None = None,
        carrier: Carrier | None = None,
        quality_factor: float | None = None,
        high_grade_exergy_out_j: float | None = None,
        low_grade_exergy_in_j: float | None = None,
        work_input_j: float | None = None,
        at: datetime | None = None,
    ) -> GuardResult:
        findings = detect_false_exergy_gain(
            exergy_in_j=exergy_in_j,
            useful_exergy_j=useful_exergy_j,
            stored_exergy_delta_j=stored_exergy_delta_j,
            recovered_exergy_j=recovered_exergy_j,
            destroyed_exergy_j=destroyed_exergy_j,
            entropy_generated_j_per_k=entropy_generated_j_per_k,
            reference_state=reference_state,
            boundary=boundary,
            reference_state_id=reference_state_id,
            boundary_id=boundary_id,
            carrier=carrier,
            quality_factor=quality_factor,
            high_grade_exergy_out_j=high_grade_exergy_out_j,
            low_grade_exergy_in_j=low_grade_exergy_in_j,
            work_input_j=work_input_j,
            at=at,
            tolerance=self.tolerance,
        )
        blocking = [finding for finding in findings if finding.severity in {Severity.ERROR, Severity.CRITICAL}]
        if blocking:
            severity = Severity.CRITICAL if any(f.severity == Severity.CRITICAL for f in blocking) else Severity.ERROR
            return GuardResult(
                self.guard_name,
                False,
                severity,
                "false exergy gain risk detected",
                {"findings": [finding.__dict__ for finding in findings]},
            )
        return GuardResult(
            self.guard_name,
            True,
            Severity.INFO,
            "no false exergy gain finding",
            {"findings": [finding.__dict__ for finding in findings]},
        )
