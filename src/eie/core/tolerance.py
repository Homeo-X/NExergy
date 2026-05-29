"""Numerical tolerances used by guards and thermodynamic accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    """Engineering tolerances for v0 calculations.

    The values are intentionally small because the simulation uses exact
    arithmetic-like inputs. Real deployments should configure these from
    measurement uncertainty and commissioning evidence.
    """

    absolute_w: float = 1.0e-6
    absolute_j: float = 1.0e-3
    relative: float = 1.0e-9
    temperature_k: float = 1.0e-9
    entropy_j_per_k: float = 1.0e-9
    efficiency: float = 1.0e-6
    stale_reference_seconds: float = 900.0

    def residual_limit(self, scale: float, *, absolute: float | None = None) -> float:
        base = self.absolute_j if absolute is None else absolute
        return max(base, abs(scale) * self.relative)


DEFAULT_TOLERANCE = Tolerance()
