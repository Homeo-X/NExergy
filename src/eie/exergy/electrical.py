"""Electrical exergy and electrical service-availability models."""

from __future__ import annotations

from eie.core.errors import DomainError
from eie.flows.base import require_non_negative, require_probability


def electrical_exergy_rate(real_power_w: float) -> float:
    """Return thermodynamic exergy rate for usable real electrical work."""

    require_non_negative(real_power_w, "real_power_w")
    return real_power_w


def _service_factor(value: float, field: str) -> float:
    require_probability(value, field)
    return value


def electrical_service_exergy_rate(
    real_power_w: float,
    voltage_quality: float = 1.0,
    frequency_quality: float = 1.0,
    harmonic_quality: float = 1.0,
    availability: float = 1.0,
) -> float:
    """Return service-derated useful electrical availability.

    This is not a pure thermodynamic exergy destruction equation. It represents
    how much high-grade electrical work remains practically usable after power
    quality and availability derating.
    """

    require_non_negative(real_power_w, "real_power_w")
    factors = [
        _service_factor(voltage_quality, "voltage_quality"),
        _service_factor(frequency_quality, "frequency_quality"),
        _service_factor(harmonic_quality, "harmonic_quality"),
        _service_factor(availability, "availability"),
    ]
    service_power = real_power_w
    for factor in factors:
        service_power *= factor
    if service_power > real_power_w:
        raise DomainError("service derating cannot increase electrical real power")
    return service_power
