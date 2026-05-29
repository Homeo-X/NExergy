"""Advisory dispatch decision schemas.

A DispatchDecision is advisory-only: it describes what the optimizer
recommends but never actuates hardware.  Optimization logic must never
bypass guard checks; hardware actuation requires an independent safety layer.

Every decision carries:
- The guard results that verified it
- An `is_advisory` flag (always True in v0)
- The dispatch variables and their values
- Feasibility notes explaining any constraint violations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any

from eie.core.errors import BoundaryError, DomainError, MissingReferenceError
from eie.guards.physics_guard import GuardResult


@dataclass(frozen=True)
class DispatchVariable:
    """Declaration of one controllable dispatch variable.

    Parameters
    ----------
    name : str
        Unique identifier used as key in DispatchDecision.variable_values.
    lower_bound : float
        Physical minimum value (inclusive).
    upper_bound : float
        Physical maximum value (inclusive).
    unit : str
        Physical unit for documentation.
    description : str
        Human-readable description of what this variable controls.
    """

    name: str
    lower_bound: float
    upper_bound: float
    unit: str
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise DomainError("DispatchVariable.name is required")
        if not isfinite(self.lower_bound) or not isfinite(self.upper_bound):
            raise DomainError("bounds must be finite")
        if self.lower_bound > self.upper_bound:
            raise DomainError(
                f"lower_bound {self.lower_bound} > upper_bound {self.upper_bound} "
                f"for variable {self.name!r}"
            )
        if not self.unit:
            raise DomainError("DispatchVariable.unit is required")

    def is_feasible(self, value: float) -> bool:
        return isfinite(value) and self.lower_bound <= value <= self.upper_bound

    def clamp(self, value: float) -> float:
        return max(self.lower_bound, min(self.upper_bound, value))

    def discrete_points(self, n: int) -> list[float]:
        """Return n evenly spaced values from lower_bound to upper_bound."""
        if n < 2:
            return [self.lower_bound]
        step = (self.upper_bound - self.lower_bound) / (n - 1)
        return [self.lower_bound + i * step for i in range(n)]


@dataclass(frozen=True)
class FeasibilityNote:
    """Structured note about why a candidate may be infeasible or sub-optimal."""

    code: str
    message: str
    severity: str   # "info" | "warning" | "infeasible"

    def __post_init__(self) -> None:
        if not self.code:
            raise DomainError("FeasibilityNote.code is required")
        if self.severity not in ("info", "warning", "infeasible"):
            raise DomainError(
                f"FeasibilityNote.severity must be 'info', 'warning', or 'infeasible', "
                f"got {self.severity!r}"
            )


@dataclass(frozen=True)
class DispatchDecision:
    """An advisory dispatch recommendation produced by the optimizer.

    This object is the unit of output from ShadowOptimizer.  It is never
    sent to hardware directly — it must be reviewed by a human or a
    hardware-specific safety layer before any actuation.

    Fields
    ------
    decision_id : str
        Unique identifier for this decision within an optimization run.
    timestamp : datetime
        When the decision was generated (not when it should be applied).
    horizon_start : datetime
        Start of the dispatch horizon this decision targets.
    horizon_end : datetime
        End of the dispatch horizon.
    boundary_id : str
        The accounting boundary this decision was computed under.
    reference_state_id : str
        The reference state used for exergy accounting.
    variable_values : dict[str, float]
        Mapping from variable name to recommended value.
    is_advisory : bool
        Always True in v0 — guards physical actuation at a higher layer.
    guard_verified : bool
        True when all guards passed for the simulated outcome of this decision.
    guard_results : list[GuardResult]
        Full guard output for the simulated outcome.
    feasibility_notes : list[FeasibilityNote]
        Structured notes about constraint compliance.
    metadata : dict[str, object]
        Additional diagnostic data (exergy balances, carbon, cost, etc.).
    """

    decision_id: str
    timestamp: datetime
    horizon_start: datetime
    horizon_end: datetime
    boundary_id: str
    reference_state_id: str
    variable_values: dict[str, float]
    is_advisory: bool
    guard_verified: bool
    guard_results: list[GuardResult]
    feasibility_notes: list[FeasibilityNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise DomainError("decision_id is required")
        if not isinstance(self.timestamp, datetime):
            raise DomainError("timestamp must be a datetime")
        if not isinstance(self.horizon_start, datetime):
            raise DomainError("horizon_start must be a datetime")
        if not isinstance(self.horizon_end, datetime):
            raise DomainError("horizon_end must be a datetime")
        if self.horizon_end < self.horizon_start:
            raise DomainError("horizon_end must be at or after horizon_start")
        if not self.boundary_id:
            raise BoundaryError("boundary_id is required")
        if not self.reference_state_id:
            raise MissingReferenceError("reference_state_id is required")
        if not self.is_advisory:
            raise DomainError(
                "DispatchDecision.is_advisory must be True in v0; hardware actuation "
                "requires an independent safety layer outside the optimizer"
            )

    @property
    def is_infeasible(self) -> bool:
        return any(n.severity == "infeasible" for n in self.feasibility_notes)

    @property
    def has_guard_failure(self) -> bool:
        return any(not r.passed for r in self.guard_results)

    @property
    def is_usable(self) -> bool:
        """Return True if this decision passed all guards and is not infeasible."""
        return self.guard_verified and not self.is_infeasible and not self.has_guard_failure

    def variable(self, name: str) -> float:
        if name not in self.variable_values:
            raise DomainError(f"variable {name!r} not in decision {self.decision_id!r}")
        return self.variable_values[name]


@dataclass(frozen=True)
class DispatchSpace:
    """Defines the feasible action space for one optimization problem.

    Parameters
    ----------
    variables : list[DispatchVariable]
        All controllable variables and their bounds.
    coupling_constraints : list[str]
        Human-readable descriptions of coupling constraints between variables.
        These are documented but not enforced automatically; the evaluator
        is responsible for infeasibility detection.
    """

    variables: list[DispatchVariable]
    coupling_constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.variables:
            raise DomainError("at least one DispatchVariable is required")
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise DomainError("variable names in DispatchSpace must be unique")

    def variable(self, name: str) -> DispatchVariable:
        for v in self.variables:
            if v.name == name:
                return v
        raise DomainError(f"no variable {name!r} in DispatchSpace")

    def is_feasible(self, values: dict[str, float]) -> bool:
        for var in self.variables:
            if not var.is_feasible(values.get(var.name, float("nan"))):
                return False
        return True

    def clamp(self, values: dict[str, float]) -> dict[str, float]:
        return {var.name: var.clamp(values.get(var.name, var.lower_bound)) for var in self.variables}

    def grid_points(self, n_per_variable: int) -> list[dict[str, float]]:
        """Generate a full grid of candidate dispatch points.

        Total points = n_per_variable ^ len(variables).
        Callers should choose n_per_variable conservatively for large spaces.
        """
        if n_per_variable < 1:
            raise DomainError("n_per_variable must be >= 1")
        from itertools import product
        axes = [v.discrete_points(n_per_variable) for v in self.variables]
        result = []
        for combo in product(*axes):
            result.append(
                {self.variables[i].name: combo[i] for i in range(len(self.variables))}
            )
        return result
