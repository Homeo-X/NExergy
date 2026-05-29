# Exergy Intelligence Engine

Exergy Intelligence Engine is a Python foundation for thermodynamic truth
accounting. It does not create energy from nothing. It represents, audits, and
checks useful-work potential relative to explicit reference states and system
boundaries.

The first deliverable is **Exergy Kernel v0**: a small, tested, advisory-only
kernel for energy, exergy, entropy, quality, ledger, and guard calculations.

## What Exergy Kernel v0 Implements

- Dynamic `ReferenceState` records with ambient temperature, pressure,
  confidence, validity windows, and optional grid/economic context.
- Explicit `Boundary` records for asset, subsystem, site, fleet, and region
  accounting.
- Typed physical schemas for electrical flows, thermal flows, cooling loads,
  chemical flows, battery state, thermal storage state, exergy flows, ledger
  entries, and loss fingerprints.
- Core equations for:
  - hot heat exergy above ambient,
  - finite hot-stream exergy,
  - cooling/refrigeration service exergy,
  - real electrical exergy,
  - service-derated electrical availability,
  - battery stored exergy,
  - stratified thermal storage exergy,
  - energy balance residual,
  - exergy balance residual.
- Carrier-specific quality grading that does not treat `qX = X/E` as a
  universal invariant.
- Append-only audit ledger with residual checks.
- Guard stack:
  - `ReferenceGuard`,
  - `BoundaryGuard`,
  - `PhysicsGuard`,
  - `FalseExergyGainGuard`.
- A deterministic simple-site simulation with PV, battery, heat pump,
  stratified thermal storage, building heat load, waste heat, ledger entries,
  normal guard checks, and intentionally impossible accounting checks.

## What It Does Not Implement Yet

- No direct PLC, SCADA, relay, inverter, valve, or other hardware actuation.
- No autonomous critical-load dispatch.
- No reinforcement learning.
- No fleet market coordination.
- No hydrogen safety automation.
- No black-start automation.
- No cloud orchestration.
- No full chemical exergy library beyond an explicit model-bound chemical flow
  schema.
- No automatic unit conversion. Unknown or mismatched units should fail instead
  of being guessed. The registered dimensional unit engine supports explicit
  scale-only conversions between compatible units such as `kW` to `W` and
  `kWh` to `J`.

## Important Scientific Limitations

- This is not a free-energy system.
- Energy is conserved; the kernel accounts for useful-work potential, not
  energy creation.
- Exergy depends on a reference environment. A value without a
  `reference_state_id` is invalid.
- Boundary definition matters. A value without a `boundary_id` is invalid.
- Raw equation functions are transparent numerical primitives. Auditable
  exergy records should be created through boundary/reference-bound schemas,
  `ExergyKernelV0`, or ledger entries.
- `qX` is carrier-specific and denominator-specific. It is not a universal
  invariant.
- Hot heat and cooling are separate thermodynamic services and use separate
  functions.
- Chemical exergy is model-specific and reference-environment-specific.
- Stratified thermal storage is integrated layer by layer; v0 does not use an
  average-temperature shortcut by default.
- Electrical real power is high-grade exergy, while voltage/frequency/harmonic
  derating is tracked as service usability rather than pure thermodynamic
  exergy destruction.
- Guard results are advisory evidence. They do not certify a real physical
  installation.

## Scientific Assumptions

- SI inputs are used by the core equations.
- Hot-heat exergy uses `Xdot = (1 - T0 / T) * Qdot` for a thermal reservoir
  above the reference temperature.
- Finite hot-stream exergy uses constant heat capacity:
  `Xdot = m_dot * cp * [(Tin - Tout) - T0 * ln(Tin / Tout)]`.
- Cooling service exergy uses reversible minimum work:
  `Wmin = Qc * (T0 / Tc - 1)` for `Tc < T0`.
- Electrical real power is treated as approximately equal to available
  electrical work.
- Battery stored exergy is approximated as stored energy times an availability
  factor.
- Exergy destruction is related to entropy generation with
  `X_destroyed = T0 * S_generated` under the ordinary engineering assumptions
  of a defined control volume and reference environment.
- Real deployments must set tolerances from measured uncertainty,
  commissioning data, manufacturer limits, and site hazard analysis.

## Safety Limitations

- v0 is simulation/advisory only.
- v0 has no hardware control API.
- Optimizers, learning systems, markets, or fleet logic must not bypass guards.
- Failed reference, boundary, physics, or false-gain checks should block any
  downstream decision that could affect equipment or safety.
- Safety thresholds for real assets must come from engineering design limits,
  manufacturer documentation, codes, standards, commissioning tests, and hazard
  analysis.
- This repository does not certify compliance with electrical, pressure,
  thermal, chemical, hydrogen, battery, industrial-control, or cybersecurity
  standards.

## Install

From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

If your shell requires quoting differently:

```powershell
python -m pip install -e . coverage hypothesis mypy pytest pytest-cov
```

## Run Tests

```powershell
python -m pytest
```

## Coverage Reporting

```powershell
python -m pytest --cov=eie --cov-report=term-missing --cov-report=xml
```

The configured coverage gate is 90% branch-aware coverage.

## Static Type Checks

```powershell
python -m mypy
```

Mypy checks the source package in strict mode and the package includes
`py.typed`.

## Run Simple Simulation

```powershell
python examples/run_simple_site.py
```

or, after editable install:

```powershell
eie-simple-site
```

The simulation prints a concise report and verifies that the normal accounting
case passes guards while the intentionally impossible case fails closed.

## Repository Layout

```text
src/eie/
  core/        constants, enums, errors, tolerances, units
  reference/   reference state and freshness validation
  boundary/    explicit accounting boundaries
  flows/       typed physical flow and storage schemas
  exergy/      equations, quality models, and ExergyKernelV0
  ledger/      append-only accounting records and audit checks
  guards/      reference, boundary, physics, and false-gain guards
  simulation/  simple deterministic site simulation
tests/         pytest, coverage, and property-based tests
examples/      runnable example entrypoint
```

## Roadmap

1. Add explicit schema versioning and event-sourced persistence.
2. Expand the strict dimensional-consistency engine with audited offset
   temperature handling only if it is genuinely needed.
3. Add time synchronization quality and stale telemetry handling.
4. Expand carrier-specific exergy models for chemical, pressure, radiative, and
   mechanical flows.
5. Add formal value-of-information logic for sensor placement.
6. Add exergoeconomic, exergoenvironmental, and lifecycle ledgers.
7. Add shadow-mode advisory optimization that cannot act before guard checks.
8. Add commissioning workflows and evidence storage before considering any
   controlled physical deployment.
