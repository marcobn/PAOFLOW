# Internal Basis Workflow

This page documents the `projections()` basis-preset system, which controls how PAOFLOW constructs the PAO basis from DFT wavefunctions.

## Basis Presets

Three string-based presets are available alongside the legacy dictionary and `None` options:

| Preset | Description |
|--------|-------------|
| `"minimal"` | Pseudo-atomic functions from UPF files — valence shells only |
| `"standard"` | Swaps minimal pseudo radials for all-electron/pseudo-atom radials of the same shells |
| `"extended"` | Pure AE/pseudo-atom basis with a generous shell set including conduction states |

`"extended"` strictly improves projectability over both `"qe"` and `"minimal"` for every scalar-relativistic pseudo family.

## Augmentation Rule

An algorithmic approach determines which quantum shells to add based on:
- The periodic-table block (s/p/d/f) of the element
- The maximum principal quantum number across occupied shells

This ensures consistent basis augmentation without per-element manual configuration.

## Spin-Orbit Support

The `"extended"` preset enables j-resolved radials for spin-orbit pseudopotentials through a new pseudo-atom basis generator (`BASIS_PS/`). This fixes previous rank-deficiency issues that created spurious artefact bands in spin-orbit calculations.

## Files

| File | Role |
|------|------|
| `src/PAOFLOW/PAOFLOW.py` | API dispatch — routes preset strings to the resolver |
| `src/PAOFLOW/defs/basis_presets.py` | Preset resolution logic |
| `tests/unit/test_basis_presets.py` | 37 unit tests covering all preset variants |

## Validation

The preset system has been validated across 8 systems spanning multiple pseudo families (NC, US, PAW, and relativistic variants), demonstrating improved band structure representation without sacrificing numerical stability.
