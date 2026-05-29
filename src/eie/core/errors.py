"""Domain-specific exceptions for the trusted exergy kernel."""


class ExergyKernelError(Exception):
    """Base exception for Exergy Kernel v0."""


class DomainError(ValueError, ExergyKernelError):
    """Raised when a value is outside the physical domain of an equation."""


class UnitError(ValueError, ExergyKernelError):
    """Raised when a value has a missing, unknown, or incompatible unit."""


class MissingReferenceError(ValueError, ExergyKernelError):
    """Raised when an exergy calculation lacks a required reference state."""


class BoundaryError(ValueError, ExergyKernelError):
    """Raised when a flow, ledger, or equation is not bound to a boundary."""


class StaleReferenceError(ValueError, ExergyKernelError):
    """Raised when a reference state is too old for trusted accounting."""


class PhysicsViolationError(ValueError, ExergyKernelError):
    """Raised when accounting violates a hard physics invariant."""
