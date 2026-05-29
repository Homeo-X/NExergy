# Exergy Intelligence Engine

A Python foundation for thermodynamic truth accounting. The engine represents,
audits, and checks useful-work potential relative to explicit reference states
and system boundaries. It does not create energy from nothing.

## What Is Implemented (v0.2)

### Exergy Kernel v0
- Dynamic `ReferenceState` records with ambient temperature, pressure,
  confidence, validity windows, and optional grid/economic context.
- Explicit `Boundary` records for asset, subsystem, site, fleet, and region
  accounting.
- Typed physical schemas for electrical flows, thermal flows, cooling loads,
  chemical flows, battery state, thermal storage state, exergy flows, ledger
  entries, and loss fingerprints.
- Core equations: hot-heat exergy, finite hot-stream exergy, cooling service
  exergy, real electrical exergy, service-derated electrical availability,
  battery stored exergy, stratified thermal storage exergy, energy and exergy
  balance residuals.
- Carrier-specific quality grading.
- Append-only audit ledger with residual checks.
- Guard stack: `ReferenceGuard`, `BoundaryGuard`, `PhysicsGuard`,
  `FalseExergyGainGuard`.
- Deterministic simple-site simulation (PV, battery, heat pump, thermal
  storage, building load, waste heat).

### Chemical Exergy Models (`eie.chemical`)
- Szargut 2005 reference environment (T₀ = 298.15 K, P₀ = 101 325 Pa) with
  15 standard reference substances.
- Pure-substance database (~35 entries) with Szargut standard chemical
  exergies in J/mol and J/kg.
- Beta-factor correlations for liquid fuels (Szargut eq. 4.7) and solid fuels
  (Szargut eq. 4.5), using mass ratios H/C, O/C, S/C, N/C.
- Gaseous mixture exergy via mole-fraction weighting plus ideal mixing term.
- Separation and mixing exergy: KL-divergence formulation, desalination
  minimum work (osmotic-pressure integral model), CO₂ capture minimum work.
- `ChemicalReaction` with ΔG°, van 't Hoff equilibrium constant, spontaneity
  check, and exergy efficiency limit.
- 8 pre-built reactions: methane combustion, hydrogen combustion, CO
  combustion, steam methane reforming, water-gas shift, calcination,
  Haber-Bosch, glucose oxidation.
- 6 model classes (`PureSubstanceModel`, `BetaFactorLiquidModel`,
  `BetaFactorSolidModel`, `GaseousMixtureModel`, `ReactionExergyModel`, and
  the abstract `ChemicalExergyModel`).
- Pre-built fuel instances: natural gas (UK mix), biogas (AD), syngas (coal),
  hydrogen, methane, diesel, gasoline, ethanol, wood pellet biomass.
- `ExergyKernelV0.chemical_flow()` integration.

### Advisory Shadow Optimizer (`eie.optimization`)
- `OptimizationObjective` with minimise/maximise direction, weight, and
  normalisation.
- `pareto_nondominated()` Pareto frontier extraction.
- `DispatchVariable`, `DispatchSpace`, and `DispatchDecision` schemas; all
  decisions carry `is_advisory=True` and are guard-verified before admission.
- `SiteSnapshot` capturing the instantaneous physical site state (PV, battery,
  thermal storage, heat pump, building load, grid constraints).
- Guard-gated evaluator: `simulate_dispatch()` runs the full physics model and
  all four guards; any failing candidate is unconditionally rejected with
  no relax-constraints mode.
- `ShadowOptimizer`: discrete grid search O(n^k) + Pareto filtering +
  coordinate-descent bisection refinement; `guard_bypass_count = 0` is an
  enforced invariant.
- Default 5-objective set: exergy destruction, exergy efficiency, marginal
  carbon, operating cost, battery SOC.

### Schema Versioning (`eie.schema`)
- `SemanticVersion` with ordering, parsing, `bump_major/minor/patch`, and
  compatibility predicates.
- `FieldDeclaration` with `added_in`, `deprecated_in`, `removed_in` lifecycle
  tracking.
- `SchemaDeclaration` with `validate_record()`.
- `SchemaRegistry` with BFS migration-path finding.
- `SchemaMigrator`: immutable `VersionedRecord` migration — corrections are
  new entries, not silent overwrites.
- `SchemaChangelog`: append-only, breaking-change detection, `since()` queries.
- Domain schemas for all 7 EIE objects (LedgerEntry, ExergyFlow,
  ReferenceState, Boundary, ChemicalFlow, BatteryState, ThermalStorageState)
  through v0.3.0, with pre-built migration rules.

## What Is Not Implemented

- No direct PLC, SCADA, relay, inverter, valve, or other hardware actuation.
- No autonomous critical-load dispatch.
- No reinforcement learning or fleet market coordination.
- No hydrogen safety automation or black-start automation.
- No cloud orchestration.
- No automatic unit conversion; unknown or mismatched units fail explicitly.
- No formal value-of-information logic for sensor placement.
- No exergoeconomic, exergoenvironmental, or lifecycle ledgers.

## Scientific Limitations

- Energy is conserved; the kernel accounts for useful-work potential, not
  energy creation.
- Every exergy value must be bound to an explicit `reference_state_id` and
  `boundary_id`.
- `qX` is carrier-specific and denominator-specific; it is not a universal
  invariant.
- Hot heat and cooling are separate thermodynamic services.
- Chemical exergy is model-specific and reference-environment-specific.
- Stratified thermal storage is integrated layer by layer.
- Electrical voltage/frequency/harmonic derating is tracked as service
  usability, not pure thermodynamic exergy destruction.
- Guard results are advisory evidence; they do not certify a physical
  installation.

## Scientific Assumptions

- SI inputs throughout.
- Hot-heat exergy: `Ẋ = (1 − T₀/T) · Q̇` (thermal reservoir above T₀).
- Finite hot-stream exergy: `Ẋ = ṁ cₚ [(Tᵢₙ − Tₒᵤₜ) − T₀ ln(Tᵢₙ/Tₒᵤₜ)]`.
- Cooling service exergy: `Wₘᵢₙ = Qc · (T₀/Tc − 1)` for Tc < T₀.
- Electrical real power ≈ available electrical work.
- Battery stored exergy ≈ stored energy × availability factor.
- Exergy destruction: `X_destroyed = T₀ · S_generated`.
- Chemical exergy: Szargut 2005 standard-state reference environment.
- Beta correlations use mass ratios (kg/kg), not molar ratios.
- Desalination minimum work: osmotic-pressure integral model.

## Safety Limitations

- v0 is simulation and advisory only; it has no hardware control API.
- Optimizers, learning systems, markets, or fleet logic must not bypass guards.
- Failed reference, boundary, physics, or false-gain checks must block any
  downstream decision that could affect equipment or safety.
- Safety thresholds for real assets must come from engineering design limits,
  manufacturer documentation, codes, standards, commissioning tests, and hazard
  analysis.
- This repository does not certify compliance with any electrical, pressure,
  thermal, chemical, hydrogen, battery, industrial-control, or cybersecurity
  standard.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

## Coverage Report

```bash
python -m pytest --cov=eie --cov-report=term-missing
```

Coverage gate: 90% branch-aware.

## Static Type Checks

```bash
python -m mypy
```

Strict mode; the package ships `py.typed`.

## Run Simple Simulation

```bash
python examples/run_simple_site.py
```

or after editable install:

```bash
eie-simple-site
```

## Repository Layout

```
src/eie/
  core/          constants, enums, errors, tolerances, units
  reference/     reference state and freshness validation
  boundary/      explicit accounting boundaries
  flows/         typed physical flow and storage schemas
  exergy/        equations, quality models, ExergyKernelV0
  ledger/        append-only accounting records and audit checks
  guards/        reference, boundary, physics, and false-gain guards
  simulation/    simple deterministic site simulation
  chemical/      chemical exergy models, reactions, mixing, fuel database
  optimization/  advisory shadow optimizer, dispatch decisions, Pareto search
  schema/        schema versioning, migration, changelog, domain declarations
tests/           pytest, coverage, and property-based tests
examples/        runnable example entrypoints
```

## Roadmap

1. ~~Add explicit schema versioning and event-sourced persistence.~~ *(done in v0.2)*
2. Expand the dimensional-consistency engine with audited offset temperature
   handling if genuinely needed.
3. Add time synchronisation quality and stale telemetry handling.
4. ~~Expand carrier-specific exergy models for chemical flows.~~ *(done in v0.2)*
5. Add formal value-of-information logic for sensor placement.
6. Add exergoeconomic, exergoenvironmental, and lifecycle ledgers.
7. ~~Add shadow-mode advisory optimization that cannot act before guard checks.~~ *(done in v0.2)*
8. Add commissioning workflows and evidence storage before considering any
   controlled physical deployment.
