# Contributing Guidelines

## Core Principles

1. **Physics correctness over feature breadth.** Every calculation must be
   thermodynamically sound and numerically auditable.
2. **Never describe the system as creating energy from nothing.**
3. **Every exergy value must be bound to an explicit reference state and
   boundary.** A value without `reference_state_id` or `boundary_id` is
   invalid by construction.
4. **Keep physical exergy separate from service usability.** Voltage,
   frequency, and harmonic derating are service quality metrics, not
   thermodynamic exergy destruction.
5. **`qX` is carrier-specific.** It is not a universal `X/E` invariant;
   apply the correct denominator for each carrier.
6. **Keep hot-heat and cooling calculations separate.** They use different
   thermodynamic relations and must not be conflated.
7. **Never replace stratified thermal storage integration with an
   average-temperature shortcut** unless a test and documentation explicitly
   justify it for the specific use case.
8. **Never remove or bypass `ReferenceGuard`, `BoundaryGuard`,
   `PhysicsGuard`, or `FalseExergyGainGuard`.** All four guards must run
   before any candidate is admitted to the feasible set.
9. **Never add real hardware actuation without an independent safety layer,**
   documented hazard analysis, tests, and a disabled-by-default interface.
10. **Optimization, learning, market, or fleet logic must never bypass guard
    checks.** `guard_bypass_count` must always be zero; this is an enforced
    dataclass invariant.
11. **Unit mismatch must fail explicitly.** Do not guess or silently convert
    incompatible units.
12. **Maintain tests for both normal cases and intentionally impossible
    cases.** Guard behaviour under bad inputs is part of the specification.
13. **Maintain property-based tests for thermodynamic invariants** when
    equations are added or modified.
14. **All corrections to ledger records must be new entries, not silent
    overwrites.** Immutability is an architectural invariant of the ledger
    and migration subsystems.

## Quality Gates

- Static types: `python -m mypy` (strict mode, must be clean)
- Tests: `python -m pytest` (354 tests, 90% branch coverage minimum)
- Coverage: `python -m pytest --cov=eie --cov-report=term-missing`

## Adding a New Thermodynamic Assumption

Document the assumption in the module docstring, cite the source (author,
year, equation number where applicable), and add a property-based test that
verifies the invariant holds across a representative input range.
