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

**Which radial function each spinor record takes (fixed 13 November 2026).** An $l > 0$ shell gives
$2(2l+1)$ records, which `assign_jm` labels as the $2l$ states $|j = l - \tfrac12, m_j\rangle$ followed
by the $2l + 2$ states $|j = l + \tfrac12, m_j\rangle$ (increasing $m_j$). Each record must carry the radial
function of its own $j$ (`spinor_record_sources`): $2l$ on the $j = l - \tfrac12$ table, $2l + 2$ on the
$j = l + \tfrac12$ one. The builders used to attach them per block of $2l + 1$, so
$|l + \tfrac12, -(l + \tfrac12)\rangle$ took the $j = l - \tfrac12$ function; the basis then broke time
reversal ($m_j \to -m_j$) and the rotations. In TaAs this put 1.8 meV of time-reversal and up to 8 meV of
point-group error into the PAO Hamiltonian near $E_F$ (spurious time-odd photocurrents); with the fix both
are at the scalar-relativistic level ($10^{-13}$ eV, $\le 40\,\mu$eV). The UPF route
(`build_pswfc_basis_all`) now pairs the two pseudo-orbitals of a shell by their `jchi` (PseudoDojo lists
$j = l + \tfrac12$ first). Tests: `tests/unit/projection/test_spinor_radials.py`.

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
  62-orbital all-electron basis of GaAs projects the Ga 3d semicore bands at 0.91 and 12
  instead of 18 bands above 0.95 in the augmented metric. All-electron radial tables are not
  the natural partners of a pseudo-wavefunction inside the augmentation sphere; see the next
  section for the basis route that replaces them.
- **Reference data move**: every integration example that uses internal projections on
  ultrasoft / PAW inputs (`example10`, `example17`) needs regenerated `Reference/` assets.

## Which shells come from where: `projections(pseudo_orbitals='auto')`

Since PAOFLOW 3.0.1 the shells of a configuration are taken from two sources
(`projection/do_atwfc_proj.py::build_basis_from_sources`, records ordered atom by atom as
`pao_sym` requires): the shells that the pseudopotential carries as pseudo-orbitals (`PP_PSWFC`,
the pseudo-atom's own solutions, consistent with the augmented metric) from the UPF, the other
shells of the preset or dictionary from the tables of `basispath`; the pseudo-orbitals of a
species that mixes the two are resampled on the mesh of its tables, so that every shell of a
species shares one radial mesh (the position-dipole tables require it). `pseudo_orbitals='auto'`
(default) does this for ultrasoft / PAW inputs and keeps the pure table basis for
norm-conserving ones; `True` / `False` force either choice (not with `'minimal'`, which is the
UPF basis already, nor with spin-orbit). On PAW GaAs (example10, 62 orbitals):

| basis | bands above 0.95 | Ga 3d bands | sp valence | conduction 9–17 |
|---|---|---|---|---|
| all-electron tables only (`pseudo_orbitals=False`) | 12 | 0.86–0.90 | 0.98 | 0.98 → 0.92 |
| UPF pseudo-orbitals + all-electron augmentation (`'auto'`) | 23 | 1.00 | 0.995 | 0.995 → 0.96 |

## Spin-orbit coupling with ultrasoft / PAW: the spinor augmentation
| all-electron tables only | 12 | 0.86–0.90 | 0.98 | 0.98 → 0.92 |
| UPF pseudo-orbitals + all-electron augmentation (`'auto'`) | 23 | 1.00 | 0.995 | 0.995 → 0.96 |
| `BASIS_PS` standard (generator, see below) | 14 | 1.00 | 0.99 | 0.98 → 0.72 |
| `BASIS_PS` extended (generator) | 17 | 1.00 | 0.99 | 0.99 → 0.89 |

The all-electron polarisation shells are more complete than the box states of the generator,
so the default route is the first; the GaAs example of `examples/nlo_examples` uses it with
`pthr = 0.95`.

## The generator with ultrasoft / PAW pseudopotentials

`paoflow-genbasis-ps` (`basis_gen/`) solves the pseudo-atom in a box; for ultrasoft / PAW it
is the generalized problem $H u = \varepsilon S u$. Its state selection by eigenvalue rank
produced ghosts on PAW files: for the Ga PAW the rank-0 states of the s and d channels sit at
−4.8 and −3.4 Ha with no overlap with the 4s and 3d pseudo-orbitals (the d ghost has a plain
norm of 10.8 for an augmented norm of 1: $S$ is nearly singular there, $q_{dd} = -0.9$), and
the written `BASIS_PS` tables projected the Ga 3d bands at 0.1. Since October 28 2026
(`basis_gen/radial.py`):

- a shell that the UPF carries as a pseudo-orbital is returned as stored (`prefer_pswfc=True`;
  the solver can only be worse than the generator's own atomic SCF);
- the other shells are selected by physics (`select_physical_states`): states confined to
  the augmentation sphere (more than 85 % of their norm inside the largest projector radius)
  or with a plain norm above three times the augmented one are ghosts, states that reproduce a
  stored pseudo-orbital are that orbital; the requested shell is then the k-th remaining state
  with k counted from one above the highest pseudo-orbital of the channel;
- for a channel without pseudo-orbitals the first valence n follows the core of the
  pseudopotential (`lowest_valence_n`: a noble-gas core plus the filled d / 4f shells of the
  period; As with 5 valence electrons has the core [Ar]3d10, so its lowest d shell is 4d, which
  the old atomic minimum called 3d and gave the one-node state for 4d).

Norm-conserving tables are unchanged within 0.01 (GeS: same 26 bands above 0.95). Tests in
`tests/unit/test_basis_gen_uspp_paw.py` (Ga PAW: pseudo-orbitals returned, ghosts excluded,
core rule) and `tests/unit/projection/test_basis_from_sources.py`.

### Spin-orbit coupling: the spinor augmentation

With fully relativistic ultrasoft / PAW pseudopotentials (`lspinorb`, `npol = 2`) the charges
are rotated into the spinor basis exactly as in QE's `init_us_1` (`qq_so`):

$$
S = 1 + \sum_a \sum_{kl} \sum_{s_1 s_2} q^{a, s_1 s_2}_{kl}\,
|\beta^a_k s_1\rangle\langle \beta^a_l s_2|, \qquad
q^{s_1 s_2}_{kl} = \sum_{ij}\sum_s q_{ij}\, f^{s_1 s}_{ki}\, f^{s s_2}_{jl},
$$

where $q_{ij}$ are the `PP_Q` charges of two projectors of the same atom and real harmonic (any
channel and $j$) and $f$ is the rotation from the real-harmonic projectors of one channel to its
$|l\,j\,m_j\rangle$ spinors (QE `fcoef`, restricted to one channel: QE zeroes it between
different channels before building `qq_so`). A scalar-relativistic species in a spin-orbit run
couples the spins diagonally. `KleinmanBylanderProjectors(spin_orbit=True,
require_norm_conserving=False)` stores the result as `Q_so` `(2, 2, nproj, nproj)`;
`AugmentedOverlap.becp` returns one block per spinor component and `overlap` contracts both,
so `augmented_proj_k` and the `<psi|S|psi>` check work unchanged on the spinor arrays (upper
component first, as QE stores them). Validation on the SnTe monolayer of example17 (Sn/Te
`rel-pbe-dn-kjpaw`, 68 projectors): the stored bands are orthonormal in the spinor metric to
$6\times10^{-6}$ (0.73 with the plain overlap), 32 of 38 bands project above 0.95 on the
minimal basis, and the symmetry of the Hamiltonian on the scf grid is exact ($10^{-13}$ eV).
Tests: `tests/unit/projection/test_spinor_augmentation.py` (the $f$ matrices are projectors on
the $j$ subspaces that sum to the identity; `Q_so` against the QE loop; the spinor overlap; the
SnTe metric, skipped unless `PAOFLOW_TEST_SNTE_SAVE` points to a `SnTe.save` of the case
`examples/cases/snte_mirror_chern`).

## VASP input: POTCAR datasets, the augmented metric and the generated basis

VASP is PAW-only, so every VASP run needs the augmented metric above. Since November 2026 the
PAW data come from the `POTCAR` datasets of the run (`src/PAOFLOW/inputs/read_potcar.py`,
`PotcarDataset`), exposed with the attributes of a `UPF` object so that every consumer
(`KleinmanBylanderProjectors`, the basis builders, the generator and the position dipoles) reads
them through one loader, `src/PAOFLOW/inputs/pseudopotentials.py` (`load_pseudopotentials`,
`load_pseudopotential`). The datasets are located by the `TITEL` of every atom type of
`vasprun.xml`: the `POTCAR` of the run folder (possibly concatenated), then
`potcar_dir/<name>/POTCAR` (`PAOFLOW(..., dft='VASP', potcar_dir=...)`), then
`$PAOFLOW_POTCAR_DIR`. Without them the projections keep the plain-overlap path with a warning;
`plain_overlap_warning` is printed whenever an ultrasoft / PAW run (every VASP run) is projected
with the plain overlap, including on request (`projections(plain_overlap=True)`), and
`vasp_license_notice` reminds once per process that VASP output and POTCAR files require a valid
VASP license.

What the reader provides, in bohr / Hartree:

- the projectors VASP applies to the plane waves: the `Reciprocal Space Part` tables
  $P_i(q) = 4\pi\int \tilde p_i(r) j_l(qr) r^2 dr$ on $q_i = i\,\mathrm{PSMAXN}/100$, injected
  into the radial table of `KleinmanBylanderProjectors` (`projector_q_table`) and divided by
  $\sqrt\Omega$. With them the WAVECAR bands are orthonormal in $S$ to $2\times10^{-6}$ (Si,
  `LREAL = .FALSE.`, the default; real-space projectors of `LREAL = Auto` runs give deviations
  of $10^{-3}$-$10^{-2}$, which the `<psi|S|psi>` check reports). The `Real Space Part` of the
  file is a real-space-optimised approximation (3-8 % off) and is not used;
- the integrated charges $q_{ij}$ (`augmentation charges`, equal to
  $\int (u_i u_j - \tilde u_i\tilde u_j) dr$ of the stored partial waves to $10^{-4}$), the
  partial waves $u = r\phi$, $\tilde u = r\tilde\phi$, the local potential
  (`local pseudopotential core`), the atomic pseudo valence density (transform of the
  reciprocal-space table, $Z_v$ electrons with the compensation charges) and the pseudised
  core density, on the logarithmic grid of the file extended geometrically to 12 bohr (the
  stored grids end at 2.6-6 angstrom: 4.9 bohr for Si);
- real-space projectors for the radial generator and the position dipoles: the inverse
  transform of the reciprocal tables, truncated at `RMAX` and re-biorthogonalised to the
  pseudo partial waves of each channel ($\langle\tilde p_i|\tilde\phi_j\rangle = \delta_{ij}$
  to $10^{-10}$; the raw transforms are dual to 1-5 %);
- the pseudo-orbitals (`pswfc`): the pseudo partial wave of every projector whose reference
  energy is a level of the `Atomic configuration` table and whose partial wave extends to the
  end of the grid (Si `3S 3P`, Ga_d `3D 4S 4P`, Fe `3D 4S`; the reference energies chosen at
  $-0.1$ Ha for the p channel of the transition metals, or the empty d level of As and I,
  give scattering partial waves cut off outside the sphere, which are not orbitals). Beyond
  the stored grid the orbital is continued by inward integration in the screened potential of
  the pseudo-atom, $\tilde v_{loc} + V_H[\tilde n_v] + V_{xc}[\tilde n_v + \tilde n_c]$, the
  `frozen_effective_potential` recipe of the QE route, which reproduces the local energy of the
  bound partial waves outside the sphere to 0.01 eV;
- the non-local strengths of the generator: by default the screened strengths of the
  reference atom, $D_{jk} = \varepsilon_k(\langle\tilde\phi_j|\tilde\phi_k\rangle_R + q_{jk})
  - \langle\tilde\phi_j|T + \tilde v_{eff}|\tilde\phi_k\rangle_R$ (Kresse-Joubert identity,
  kinetic term by parts, the column of a bound reference state exact), with which the bound
  reference states are eigenstates of the generalized box problem $Hu = \varepsilon Su$ within
  0.3 eV (overlap above 0.995, Si / Ga_d / Fe); the unscreened `D_ij` of the file (the analogue
  of QE's `PP_DIJ`, which the QE generator uses) misplace the Si 3s level by several eV and are
  kept as `nonlocal_strength='file'`.

The basis for VASP (and for QE without `basispath`) is generated at run time
(`src/PAOFLOW/projection/radial_shell_basis.py`, `build_generated_basis`): `projections(
configuration='minimal' | 'standard' | 'extended' | {elem: [...] | preset})` resolves the rule-based
shells without filtering by table files, rank 0 runs `basis_gen.driver.generate_basis_shells`
per element (the pseudo-orbitals as stored, the other shells from the box solver with the ghost
filter, one generalized eigenproblem per channel, cached), the functions are broadcast and
turned into the records of the projection on one uniform mesh per species (the scalar
duplication of `build_aewfc_basis` with spin-orbit coupling, j-resolved pairs of fully
relativistic UPFs otherwise). `r_box`, `n_points` and `save_basis` (tables written to
`outputdir/BASIS_PS/`) are the user controls; `paoflow-genbasis-ps --pseudo <POTCAR>` /
`--potcar-dir` write the same tables for the table route. The legacy
`projections(basispath=BASIS, configuration={...})` of the VASP examples still works and now
takes the pseudo-orbitals from the POTCAR and the augmenting shells from the tables, in the
augmented metric; with spin-orbit coupling this mixed route goes through
`build_mixed_basis_with_tables` (scalar pseudo-orbitals in both $j$ blocks, j-resolved tables when
present), where `build_basis_from_sources` used to refuse it and every shell came from the tables.

Si (example01 of `examples/recipes/vasp`, 18 bands): `minimal` 8 orbitals, 4 bands above
0.95; `standard` 26 orbitals, 12 bands; `extended` 44 orbitals, 16 bands (the shipped
all-electron tables with `3S 3P 3D 4S 4P 4D`: 36 orbitals, 15 bands). Pt with spin-orbit
coupling (example02): the spinor metric with the scalar datasets couples the spins diagonally,
$\langle\psi|S|\psi\rangle - 1 = 2\times10^{-6}$, 12 of 20 bands above 0.95 with `5D 6S 6P 7S 7P`.
Tests: `tests/unit/inputs/test_read_potcar.py`, `tests/unit/test_basis_gen_potcar.py`,
`tests/unit/projection/test_vasp_augmented_projections.py`,
`tests/unit/projection/test_radial_shell_basis.py` (the POTCAR and WAVECAR ones are skipped
without `PAOFLOW_POTCAR_DIR` / `../potpaw_PBE_64` and `../example01` next to the repository).

Not covered: the PAW augmentation of the position dipoles with spin-orbit coupling
(scalar-relativistic only; the dipoles themselves cover spin-orbit runs with norm-conserving
pseudopotentials, `position-dipoles.md`), Gamma-only WAVECARs, ultrasoft POTCARs (`LPAW = F`).

## Files

| File | Role |
|------|------|
| `src/PAOFLOW/PAOFLOW.py` | API dispatch — routes preset strings to the resolver |
| `src/PAOFLOW/inputs/basis_presets.py` | Preset resolution logic (rule-based shells; filtered by the table files only with `basispath`) |
| `src/PAOFLOW/inputs/read_potcar.py` | VASP POTCAR datasets with the attributes of a UPF (`PotcarDataset`, `read_potcar`, `find_potcar_datasets`) |
| `src/PAOFLOW/inputs/pseudopotentials.py` | One loader of the pseudopotential data of a run, UPF (QE) or POTCAR (VASP) |
| `src/PAOFLOW/basis_gen/driver.py` | Generator: `generate_basis_shells` (in memory), `generate_basis_for_pseudo` (tables) |
| `src/PAOFLOW/projection/radial_shell_basis.py` | Generated-basis route of `projections()` (`build_generated_basis`, `build_basis_from_radial_shells`) |
| `tests/unit/inputs/test_read_potcar.py`, `tests/unit/test_basis_gen_potcar.py`, `tests/unit/projection/test_vasp_augmented_projections.py`, `tests/unit/projection/test_radial_shell_basis.py` | VASP route (skipped without the POTCAR library / WAVECAR example) |
| `tests/unit/test_basis_presets.py` | 37 unit tests covering all preset variants |
| `src/PAOFLOW/projection/augmentation.py` | Augmented overlap of ultrasoft / PAW pseudopotentials (`AugmentedOverlap`, `build_augmentation`) |
| `tests/unit/projection/test_augmented_projections.py` | Löwdin factor in a general metric, projector charges of the example10 PAW files |

## Validation

The preset system has been validated across 8 systems spanning norm-conserving, ultrasoft, PAW, and relativistic pseudo families. Improved band structure representation is consistent across the test set, with no loss of numerical stability.
