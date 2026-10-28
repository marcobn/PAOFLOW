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

## Ultrasoft and PAW pseudopotentials: the augmented metric

The pseudo-wavefunctions of an ultrasoft or PAW calculation are orthonormal in the metric
$S = 1 + \sum_a \sum_{ij} q^a_{ij} |\beta^a_i\rangle\langle\beta^a_j|$, not in the plain
plane-wave one. Since October 2026 the internal projections take this into account
(`projection/augmentation.py`, used by `calc_proj_k` and `calc_proj_and_ovlp_k` when any
species of a QE run carries augmentation charges):

- the projectors $\beta^a_i(\mathbf k+\mathbf G)$ and the integrated charges $q^a_{ij}$
  (`PP_Q` of the UPF) come from `elphon.pseudopotential_terms.KleinmanBylanderProjectors`
  (built with `require_norm_conserving=False`, which also fills its `Q` matrix);
- the atomic orbitals are Löwdin-orthonormalised with $\langle\phi_\mu|S|\phi_\nu\rangle$,
  the projections are $\langle\tilde\phi_i|S|\psi_{n\mathbf k}\rangle$ and the overlap
  written for ACBN0 is the augmented one, exactly as `projwfc.x` does;
- the stored bands are used as they are (`read_QE_wfc(..., orthonormalize=False)`); the driver
  checks $\langle\psi|S|\psi\rangle = 1$ at its first k-point and warns above $10^{-3}$.

Before this, the bands were re-orthonormalised with the plain overlap (a Löwdin step over the
`nbnd` pseudo-wavefunctions, whose plain overlaps deviate from orthonormality by up to 0.78 for
the Ga 3d states of PAW GaAs). That step mixes the bands where the `nbnd` cut crosses a
multiplet and breaks the point symmetry of the PAO Hamiltonian at those k-points: for GaAs
(example10 inputs, 62-orbital AE basis) $H(\mathbf k)$ violated the $C_{2z}$ relation by
0.65 eV at 16 of 4096 grid points, the eigenvalues interpolated at an off-grid k spread by
5–25 meV over the 24 $T_d$ images, the real-space Hamiltonian decayed slowly (median
$|H(n)| = 3$ meV beyond 40 bohr) and the second-order optical responses violated the tensor
symmetry by 2–5 %. With the augmented metric the gauge-invariant valence density matrix built
from the projections obeys the antiunitary $C_{2z}\mathcal T$ symmetry to $10^{-10}$ (as the
`projwfc.x` projections do), $\langle\psi|S|\psi\rangle - 1 = 3\times10^{-6}$, $H(\mathbf k)$
is symmetric on the grid, the interpolation spread is $3\times10^{-3}$ meV and the real-space
Hamiltonian is 20 times better localised. On the UPF pseudo-atomic basis (`minimal`) the
projectabilities agree with `projwfc.x` to $1.5\times10^{-3}$ ($6\times10^{-5}$ for the valence
bands).

Two consequences for ultrasoft / PAW runs:

- **Projectabilities change** (they were inflated by the re-orthonormalisation): the
  62-orbital all-electron basis of GaAs now projects the Ga 3d semicore bands at 0.91 and 12
  instead of 18 bands above 0.95. All-electron radial tables (`BASIS/`) are not the natural
  partners of a pseudo-wavefunction inside the augmentation sphere; prefer the pseudo-atom
  tables generated from the same pseudopotentials (`paoflow-genbasis-ps`, `BASIS_PS/`) or the
  UPF pseudo-atomic orbitals, and check the projectability vector.
- **Reference data move**: every integration example that uses internal projections on
  ultrasoft / PAW inputs (`example10`, `example17`) needs regenerated `Reference/` assets.

Not covered: spin-orbit runs with ultrasoft / PAW pseudopotentials (the spinor augmentation
$q^{so}_{ij}$ is not implemented; the old plain-overlap path is kept there with a warning, use
`read_atomic_proj_QE()` or norm-conserving pseudopotentials) and VASP input (PAW projectors of
the POTCAR are not read).

## Files

| File | Role |
|------|------|
| `src/PAOFLOW/PAOFLOW.py` | API dispatch — routes preset strings to the resolver |
| `src/PAOFLOW/defs/basis_presets.py` | Preset resolution logic |
| `tests/unit/test_basis_presets.py` | 37 unit tests covering all preset variants |
| `src/PAOFLOW/projection/augmentation.py` | Augmented overlap of ultrasoft / PAW pseudopotentials (`AugmentedOverlap`, `build_augmentation`) |
| `tests/unit/projection/test_augmented_projections.py` | Löwdin factor in a general metric, projector charges of the example10 PAW files |

## Validation

The preset system has been validated across 8 systems spanning norm-conserving, ultrasoft, PAW, and relativistic pseudo families. Improved band structure representation is consistent across the test set, with no loss of numerical stability.
