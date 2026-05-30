"""Hot-heat and finite-stream thermal exergy equations."""

from __future__ import annotations

from math import isclose, log

from eie.core.errors import DomainError
from eie.core.temperature import require_ratio_temperature
from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance
from eie.flows.base import require_non_negative, require_positive


class ColdThermalDomainError(DomainError):
    """Raised when a hot-heat equation is asked to account for cold service."""


def heat_exergy_rate(
    heat_rate_w: float,
    source_temperature_k: float,
    reference_temperature_k: float,
    *,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> float:
    """Return exergy rate for heat supplied from a reservoir above ambient.

    Relation:
        Xdot = (1 - T0 / T) * Qdot

    This function is intentionally only for hot heat above the reference
    environment. Cooling/refrigeration service uses
    ``cooling_service_exergy_rate`` instead.
    """

    require_non_negative(heat_rate_w, "heat_rate_w")
    require_ratio_temperature(source_temperature_k, "source_temperature_k")
    require_ratio_temperature(reference_temperature_k, "reference_temperature_k")
    if isclose(source_temperature_k, reference_temperature_k, abs_tol=tolerance.temperature_k):
        return 0.0
    if source_temperature_k < reference_temperature_k:
        raise ColdThermalDomainError(
            "source_temperature_k is below reference_temperature_k; use the cooling model instead"
        )
    return heat_rate_w * (1.0 - reference_temperature_k / source_temperature_k)


def finite_stream_heat_exergy_rate(
    mass_flow_kg_s: float,
    cp_j_kg_k: float,
    t_in_k: float,
    t_out_k: float,
    reference_temperature_k: float,
    *,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> float:
    """Return exergy rate released by a finite hot stream cooled from Tin to Tout.

    For approximately constant heat capacity:
        Xdot = m_dot * cp * [(Tin - Tout) - T0 * ln(Tin / Tout)]

    The v0 implementation accepts this equation only for a stream cooling while
    remaining at or above the reference temperature. That avoids sign mistakes
    near cold-service boundaries.
    """

    require_non_negative(mass_flow_kg_s, "mass_flow_kg_s")
    require_non_negative(cp_j_kg_k, "cp_j_kg_k")
    require_ratio_temperature(t_in_k, "t_in_k")
    require_ratio_temperature(t_out_k, "t_out_k")
    require_ratio_temperature(reference_temperature_k, "reference_temperature_k")
    if mass_flow_kg_s == 0.0 or cp_j_kg_k == 0.0:
        return 0.0
    if t_in_k < t_out_k - tolerance.temperature_k:
        raise DomainError("finite_stream_heat_exergy_rate requires t_in_k >= t_out_k")
    if t_out_k < reference_temperature_k - tolerance.temperature_k:
        raise ColdThermalDomainError(
            "stream crosses below reference_temperature_k; split the stream or use a cold-service model"
        )
    if isclose(t_in_k, t_out_k, abs_tol=tolerance.temperature_k):
        return 0.0
    exergy = mass_flow_kg_s * cp_j_kg_k * ((t_in_k - t_out_k) - reference_temperature_k * log(t_in_k / t_out_k))
    if exergy < -tolerance.absolute_w:
        raise DomainError(f"finite stream exergy became negative ({exergy}); check temperatures and signs")
    return max(0.0, exergy)


def waste_heat_recoverable_exergy_rate(
    heat_rate_w: float,
    stream_temperature_k: float,
    reference_temperature_k: float,
) -> float:
    """Recoverable exergy rate for a simple waste heat reservoir above ambient."""

    return heat_exergy_rate(heat_rate_w, stream_temperature_k, reference_temperature_k)
