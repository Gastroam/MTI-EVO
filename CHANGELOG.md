# Changelog

## 2.3.5 - 2026-02-13

- Hardened deterministic behavior in core pathways via seeded RNG ownership in `HolographicLattice`, neuron RNG injection, and RNG-only eviction sampling.
- Unified neuron input validation with a single `_validate_inputs()` path used by both `perceive()` and `adapt()`.
- Added explicit eviction modes (`full_scan`, `sample`, `deterministic_sample`) and kept capacity/eviction behavior testable.
- Added optional stimulation metrics while preserving backward-compatible default return type for `stimulate()`.
- Performed safe modularization of core code into:
  - `src/mti_evo/core/neuron.py`
  - `src/mti_evo/core/eviction.py`
  - `src/mti_evo/core/lattice.py`
  - `src/mti_evo/core/dense_tissue.py`
  - with `src/mti_evo/mti_core.py` as compatibility facade.
- Added and stabilized hardening invariants in `tests/integration/audit_test_suite.py`.
