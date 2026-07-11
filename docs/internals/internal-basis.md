# Internal Basis Workflow

This page documents the `projections()` basis-preset system — how PAOFLOW constructs the PAO basis from DFT wavefunctions and what options are available to control it.

## Basis Presets

Three string-based presets make it easy to choose the right basis for your calculation, alongside the legacy dictionary and `None` options:

| Preset | What it gives you |
|--------|------------------|
| `"minimal"` | Pseudo-atomic functions from the UPF files — valence shells only |
| `"standard"` | Same shells as minimal, but swaps the pseudo radials for all-electron/pseudo-atom radials |
| `"extended"` | A pure AE/pseudo-atom basis with a generous shell set that includes conduction states |

If you're unsure which to use, `"extended"` is a safe bet — it strictly improves projectability over both `"qe"` and `"minimal"` for every scalar-relativistic pseudo family.

## How Shell Augmentation Works

The preset system uses an algorithmic rule to decide which quantum shells to add, based on:
- The periodic-table block (s/p/d/f) of the element
- The maximum principal quantum number across occupied shells

This means you get consistent augmentation across the periodic table without needing per-element manual configuration.

## Spin-Orbit Support

The `"extended"` preset also enables j-resolved radials for spin-orbit pseudopotentials through a new pseudo-atom basis generator (`BASIS_PS/`). This fixes previous rank-deficiency issues that were causing spurious artifact bands in spin-orbit calculations.

## Files

| File | Role |
|------|------|
| `src/PAOFLOW/PAOFLOW.py` | API dispatch — routes preset strings to the resolver |
| `src/PAOFLOW/defs/basis_presets.py` | Preset resolution logic |
| `tests/unit/test_basis_presets.py` | 37 unit tests covering all preset variants |

## Validation

The preset system has been validated across 8 systems spanning norm-conserving, ultrasoft, PAW, and relativistic pseudo families. Improved band structure representation is consistent across the test set, with no loss of numerical stability.
