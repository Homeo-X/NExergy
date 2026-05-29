# Project Rules For Contributors And Automation

This repository implements Exergy Kernel v0 for the Exergy Intelligence Engine.

Persistent rules:

1. Preserve physics correctness over feature breadth.
2. Never describe the system as creating energy from nothing.
3. Every exergy value must remain bound to an explicit reference state and
   boundary.
4. Keep physical exergy separate from service usability and power-quality
   derating.
5. Treat `qX` as carrier-specific, not a universal `X/E` invariant.
6. Keep hot-heat and cooling calculations separate.
7. Never replace stratified thermal storage integration with an average
   temperature shortcut unless a test and documentation explicitly justify it.
8. Never remove or bypass `ReferenceGuard`, `BoundaryGuard`, `PhysicsGuard`, or
   `FalseExergyGainGuard`.
9. Never add real hardware actuation without an explicit safety layer,
   documented hazard analysis, tests, and a disabled-by-default interface.
10. Learning, market, fleet, or optimization logic must never affect safety
    directly or bypass guard checks.
11. Unit mismatch must fail explicitly.
12. Keep tests for normal cases and intentionally impossible cases.
13. Keep property-based tests for thermodynamic invariants when equations are
    expanded.
14. Maintain static type checks with `python -m mypy`.
15. Maintain coverage reporting with the configured 90% minimum.
16. Document every new thermodynamic assumption.
17. Corrections to ledger records must be new entries, not silent overwrites.
