"""Cooling and refrigeration service exergy."""

from __future__ import annotations

from eie.flows.base import require_non_negative, require_positive


def cooling_service_exergy_rate(
    cooling_rate_w: float,
    cold_temperature_k: float,
    reference_temperature_k: float,
) -> float:
    """Return reversible minimum work rate for cooling below ambient.

    For refrigeration service at cold temperature Tc relative to environment T0:
        Wmin = Qc * (T0 / Tc - 1), for Tc < T0

    When Tc >= T0, no refrigeration work is required relative to ambient, so v0
    returns zero instead of fabricating cold exergy.
    """

    require_non_negative(cooling_rate_w, "cooling_rate_w")
    require_positive(cold_temperature_k, "cold_temperature_k")
    require_positive(reference_temperature_k, "reference_temperature_k")
    if cooling_rate_w == 0.0 or cold_temperature_k >= reference_temperature_k:
        return 0.0
    return cooling_rate_w * (reference_temperature_k / cold_temperature_k - 1.0)
