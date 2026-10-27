# Electron–Phonon Coupling

This page documents the **PAO-interpolation route** of `PAOFLOW.elphon`: the
production path for computing electron–phonon (el‑ph) coupling and isotropic
Eliashberg superconducting properties ($\alpha^2F$, $\lambda$, $\omega_{\log}$,
$T_c$), and from these the temperature-dependent superconducting gap of the
isotropic Migdal–Eliashberg equations. The route reads the coarse-grid DFPT coupling computed by Quantum
ESPRESSO (QE) or EPW and interpolates it in the PAOFLOW pseudo-atomic-orbital
(PAO) gauge.

> **Recommended coupling source: EPW (`source='epw'`).** It is the only source
> whose matrix elements share the band gauge of PAOFLOW's projections for every
> q, which the interpolation requires. The ph.x sources (`'ahc'`, `'elphmat'`)
> remain available in the library, but their vertices are gauge-inconsistent for
> $q\neq\Gamma$ (the `paoflow-gen` menu offers only `ahc`, labelled "Gamma only"); see [Coupling sources and gauge consistency](#coupling-sources-and-gauge-consistency).

---

## Contents

- [Why this route](#why-this-route)
- [Theory](#theory)
- [Coupling sources and gauge consistency](#coupling-sources-and-gauge-consistency)
- [Module map](#module-map)
- [Required inputs](#required-inputs)
- [Workflow 1 — coarse-q](#workflow-1--coarse-q)
- [Workflow 2 — dense-q (k *and* q interpolation)](#workflow-2--dense-q-k-and-q-interpolation)
- [Symmetry reduction of the dense q-grid](#symmetry-reduction-of-the-dense-q-grid)
- [Pair-resolved Wigner–Seitz interpolation (several atoms per cell)](#pair-resolved-wignerseitz-interpolation-several-atoms-per-cell)
- [Isotropic Migdal–Eliashberg: gap, Padé, analytic continuation, linearised $T_c$](#isotropic-migdaleliashberg-gap-padé-analytic-continuation-linearised-t_c)
- [Anisotropic Migdal–Eliashberg (Fermi-surface restricted)](#anisotropic-migdaleliashberg-fermi-surface-restricted)
- [Phonon-assisted optical absorption](#phonon-assisted-optical-absorption)
- [Carrier transport in semiconductors and insulators (SERTA / MRTA)](#carrier-transport-in-semiconductors-and-insulators-serta--mrta)
- [Spectral functions and ARPES](#spectral-functions-and-arpes)
- [Screening correction of the coupling (Poncé 2018)](#screening-correction-of-the-coupling-poncé-2018)
- [Parallelisation and memory](#parallelisation-and-memory)
- [Adaptive smearing of the deltas](#adaptive-smearing-of-the-deltas)
- [Grid consistency rules](#grid-consistency-rules)
- [The `paoflow-gen elphon` CLI workflow](#the-paoflow-gen-elphon-cli-workflow)
- [Validation](#validation)
- [Practical notes and pitfalls](#practical-notes-and-pitfalls)
- [API summary](#api-summary)
- [References](#references)

---

## Why this route

- It uses the **full** DFPT coupling as computed by QE/EPW: bare local, bare
  nonlocal, induced (Hartree+xc), and any NLCC/ultrasoft augmentation. There is
  **no potential reconstruction** in PAOFLOW.
- The PAO gauge is a **fixed, deterministic atomic-orbital basis**: unlike a
  Wannier-function interpolation, no disentanglement, gauge-fixing or window
  selection is needed for the interpolation itself.
- Interpolation to dense grids reuses PAOFLOW's Wigner–Seitz generalised-Fourier
  machinery (the same used for `HRs` in every other PAOFLOW property).

Reference: L. A. Agapito and M. Bernardi, *Ab initio electron-phonon
interactions using atomic orbital wave functions*, [Phys. Rev. B **97**, 235146
(2018)](https://doi.org/10.1103/PhysRevB.97.235146).

---

## Theory

The isotropic Eliashberg spectral function and coupling constant are

$$
\alpha^2F(\omega) = \tfrac12\sum_{q\nu} w_q\,\lambda_{q\nu}\,\omega_{q\nu}\,
   \delta(\omega-\omega_{q\nu}),
\qquad
\lambda = \sum_{q\nu} w_q\,\lambda_{q\nu} = 2\int_0^\infty \frac{\alpha^2F(\omega)}{\omega}\,d\omega ,
$$

with $w_q$ the (normalised) q-point weights. The mode-resolved coupling is a
**Fermi-surface double delta**,

$$
\lambda_{q\nu} = \frac{1}{N(E_F)\,\omega_{q\nu}^2}\,\frac{1}{N_k}\sum_{k,mn}
   \big|\langle m,k{+}q|\,\partial V\cdot e_{q\nu}/\sqrt{M}\,|n,k\rangle\big|^2\,
   \delta(\varepsilon_{nk} - E_F)\,\delta(\varepsilon_{m,k+q}-E_F),
$$

with $N(E_F)$ the density of states per spin. This is identical to EPW's
$\lambda_{q\nu} = \gamma_{q\nu}/(\pi N_F \omega_{q\nu}^2)$: EPW's
$|g|^2 = |\cdot|^2/(2\omega)$ is compensated by its spin-summed k weights. $T_c$
follows from the McMillan / Allen–Dynes formula with Coulomb pseudopotential
$\mu^*$.

**The PAO-gauge vertex.** The Bloch-basis Cartesian deformation potential
$d_{mn,\kappa\alpha}(k,q) = \langle m,k{+}q|\partial_{u_{\kappa\alpha}}V|n,k\rangle$
is read from the coupling files (never recomputed). It is rotated into the PAO
gauge with the projections $A_{ni}(k) = \langle\phi_i|\psi_{nk}\rangle$ used to
build `HRs` ($H = A\,\varepsilon\,A^\dagger$),

$$
g^{\rm PAO}_{ij,\kappa\alpha}(k,q)
= \sum_{mn}\langle\phi_i|\psi_{m,k+q}\rangle\, d_{mn,\kappa\alpha}(k)\, \langle\psi_{nk}|\phi_j\rangle
= \big[A_{k+q}^{T}\, d_{\kappa\alpha}(k)\, A_k^{*}\big]_{ij},
$$

i.e. $\langle\phi_i|P_{k+q}\,\partial_{u_{\kappa\alpha}}V\,P_k|\phi_j\rangle$.
Each band index appears once as a bra and once as a ket, so the vertex does not
depend on the band phases, **provided $d$ and $A$ come from the same Kohn–Sham
states**. It is then Fourier-transformed $k\to R_e$ to a real-space vertex
$g_q(R_e)$ per coarse q (`vertex_pao_R`), the "half-transformed" object at the
heart of both workflows.

**Fourier conventions.** PAOFLOW builds `HRs = ifftn(Hks)`, so the electron
Hamiltonian is evaluated as $H(k)=\sum_R e^{-2\pi i k\cdot R}\,H(R)$. The vertex
uses its own pair: forward `fftn / N` in `vertex_pao_R` and back with
$e^{+2\pi i k\cdot R}$ in `lambda_q_dense_ws_fast`. Both are then evaluated at
the same physical k.

> **Corrections (October 2026).** Two errors affected every result obtained
> with this route before then:
> 1. the vertex was formed as $A_{k+q}^\dagger d A_k$, which assumes the
>    conjugate projection convention $\langle\psi|\phi\rangle$ and is not
>    independent of the band phases;
> 2. the dense electron Hamiltonian was rebuilt from `HRs` with
>    $e^{+2\pi i k\cdot R}$, i.e. at $-k$. Eigenvalues are unchanged (time
>    reversal), but the eigenvectors were complex-conjugated relative to the
>    vertex.
>
> Both are fixed and covered by unit tests
> (`test_vertex_pao_R_recovers_pao_operator_independent_of_band_phases`,
> `test_dense_electrons_follow_paoflow_fourier_convention`).

---

## Coupling sources and gauge consistency

| `source` | Coupling file | Pseudopotentials | $\psi_{k+q}$ in $d$ | Usable for interpolation |
|---|---|---|---|---|
| **`'dfpt'`** | `ph.x` outputs: `dvscf`, `patterns.N.xml`, `dyn` (no EPW) | norm-conserving (scalar or spin-orbit) | computed by PAOFLOW from the nscf save (grid point $k+q-G$) | **yes, all q** |
| **`'epw'`** | `prefix.epb*` (`epw.x`, `epbwrite = .true.`) | any (NC, US, PAW) | read from the nscf save (grid point $k+q-G$) | **yes, all q** |
| `'ahc'` | `ahc_gkk_iq*.bin` (`ph.x`, `electron_phonon='ahc'`) | norm-conserving | re-diagonalised by `ph.x`, arbitrary phases | only $q=\Gamma$ |
| `'elphmat'` | `elphmat.<iq>.dat` (PAOFLOW-patched `ph.x`) | any | re-diagonalised by `ph.x` | only $q=\Gamma$ |

`ph.x` recomputes the k+q states in its own nscf for every $q\neq\Gamma$ and
does not align their phases. AHC aligns only the k states, to the $q=\Gamma$
reference. The rotation $A_{k+q}^T d A_k^*$ then mixes two different band gauges.

- **At the coarse k-points, $\lambda_{q\nu}$ is unaffected**: the double-delta
  sum over complete degenerate blocks does not depend on band phases.
- **The real-space vertex $g_q(R_e)$ is not localised**, so any interpolation
  between coarse points (dense k, dense q) is wrong.

EPW instead evaluates both $\psi_k$ and $\psi_{k+q}$ from the nscf
wavefunctions, and covers the irreducible-q star by rotating coordinates. Its
matrix elements therefore share the gauge of PAOFLOW projections computed on
the same save. The details and diagnostics are in the wiki page
*PAOFLOW_elphon_AHC_symmetry_notes*.

The **`'dfpt'` source** (`elphon/dfpt_vertex.py`, October 2026) does the same
natively: it reads the induced potential `dvscf` of `ph.x` (the Hartree and
exchange-correlation response to each displacement pattern, which excludes the bare
ionic term), adds the bare local-pseudopotential derivative and the Kleinman–Bylander
derivative terms, and takes the matrix elements with the nscf wavefunctions of the
save that provides the projections, so the gauge consistency holds by construction.
The irreducible q of `ph.x` are unfolded to the full grid by rotating the induced
potential on the FFT grid (QE's `dfile_star` identity) and using the true wavefunctions
at $k$ and $k+Sq$; $-q$ partners use the complex-conjugate potential. The matrix
elements are either taken between the bands and rotated with $A$ (`basis='band'`), or
directly between the projected PAO functions $\tilde\phi_{j,k} = \sum_n A^*_{nj}(k)\psi_{nk}$
(`basis='pao'`), which is identical by linearity and never forms the band-basis array.
Spin-orbit coupling is supported (spinor wavefunctions, QE's `fcoef` coupling matrix of
the fully relativistic projectors); ultrasoft and PAW pseudopotentials are not (use EPW).
See [DFPT coupling port](dfpt-coupling-port.md) for the design, the parity with EPW and
the limits.

---

## Module map

Paths relative to `src/PAOFLOW/`.

| File | Role |
|------|------|
| `elphon/dfpt_io.py` | `ph.x` readers of the native vertex: `locate_dfpt_files` (pp.py `save/` or raw `_ph0/` layout), `read_patterns`, `read_dvscf`, `read_irreducible_q`, `read_long_range_tensors`; `read_space_group` (crystal operations, `x -> R x - t`), `read_nscf_plane_wave_setup`. |
| `elphon/plane_waves.py` | Raw plane-wave blocks (`read_plane_wave_block`, no Löwdin step), `periodic_part_on_grid` / `project_on_plane_waves` (FFT box, umklapp gather), `interpolate_to_grid`, `g_sphere`. |
| `elphon/pseudopotential_terms.py` | Bare terms: `LocalPotentialTable` (`vloc_of_g` tables), `bare_local_perturbation` (`compute_dvloc`), `KleinmanBylanderProjectors` (`init_us_2`, `average_pp`, spin-orbit `dvan_so`), `nonlocal_perturbation` (`dvqpsi_us_only`), `real_spherical_harmonics_qe` (`ylmr2`). |
| `elphon/dfpt_vertex.py` | `CoarseVertexSetup` (tables, symmetries, star members, `potential_at`, `matrix_elements`, `pao_vertex_at_q`, `band_vertex_at_q`), `star_members`, `cartesian_perturbation_potentials` (dvscf rotation), `rotate_pao_vertex` / `unfolded_pao_vertices` (irreducible-q unfolding in the PAO basis with self-check). |
| `elphon/coupling_cache.py` | Opt-in disk cache of $g(R_e, R_p)$ (`cache_dir`, `cache_tag`, `basis_tag`). |
| `adaptive_smearing.py` | Velocity-dependent widths of the energy-conserving deltas (grid spacing, state and pair widths, adaptive Fermi level); see [Adaptive smearing](#adaptive-smearing-of-the-deltas). |
| `elphon_native/` | Bridge to the Rust kernel of `indirect_absorption_kernel` (`paoflow_rs.elphon_indirect_absorption`; `PAOFLOW_ELPHON_DISABLE` turns it off). |
| `elphon/qe_elph_io.py` | Readers: `read_epw_epb` (EPW coarse Bloch coupling, multi-pool, gfortran subrecords), `read_epw_ukk` (EPW band bookkeeping: kept bands, window), `read_qe_ahc_gkk`, `read_qe_el_ph_mat`, `el_ph_mat_to_cartesian`, `read_qe_dyn`, plus the older `elph.inp_lambda`/`lambda.in` readers of the property-only route. |
| `elphon/elph_bloch.py` | Core machinery: `read_nscf` (k-points, lattice, Fermi level, crystal symmetries, atom species), `atom_masses` (per-species masses → one per atom), `kq_index_map`, `vertex_pao_R`, `_ws_lattice`, `precompute_dense_electrons` + `lambda_q_dense_ws_fast` (dense-$k$ electron cache + Fermi-surface double delta, shared across q). |
| `elphon/do_pao_eph.py` | **Workflow 1** driver `eliashberg_from_qe_coupling`; per-source vertex builders `vertex_from_epw`, `vertex_from_qe_ahc`, `vertex_from_qe_elphmat`; `load_epw_coupling` (`.ukk` + `.epb` + q in crystal coordinates); `phonon_modes_from_force_constants`. |
| `elphon/do_pao_eph_dense_q.py` | **Workflow 2** driver `eliashberg_dense_q`; `build_g_ReRp` / `g_Re_at_q` (double real-space vertex); `phonon_interp_from_epw` and `phonon_interp_from_dyn` (dense-q phonons with acoustic sum rule); `irreducible_qmesh` / `_crystal_point_group`. |
| `elphon/eph_kq.py` | Property engine: `eliashberg_from_modes` ($\alpha^2F$, $\lambda$, $\omega_{\log}$, $T_c$), `mcmillan_allen_dynes_tc`, `phonon_moments`; shared by every route. |
| `elphon/migdal_eliashberg.py` | Isotropic Migdal–Eliashberg solver on $\alpha^2F$: `solve_imag_iso`, `pade_continuation`, `analytic_continuation_iso`, `gap_edge`, `quasiparticle_dos`, `linearized_max_eigenvalue`; drivers `migdal_eliashberg_iso` / `linearized_eigenvalues`; I/O `a2f_from_npz`, `a2f_from_epw`, `write_me_outputs`. |
| `elphon/fermi_surface_coupling.py` | `FermiSurfacePairCoupling`: irreducible Fermi-surface states and the folded pair coupling $\Lambda_{ab}(\omega)$, filled during the dense-q loop; `write_fs_coupling` / `read_fs_coupling`. |
| `elphon/anisotropic_eliashberg.py` | Anisotropic (FSR) Migdal–Eliashberg solver: `coupling_strength`, `matsubara_kernel`, `solve_imag_aniso`, `linearized_max_eigenvalue_aniso`, `pade_continuation_aniso`, `gap_distribution`; drivers `migdal_eliashberg_aniso` / `linearized_eigenvalues_aniso`; `write_me_aniso_outputs`. |
| `elphon/coupling_screening.py` | Poncé–Resta screening correction of the coupling: `resta_dielectric`, `thomas_fermi_wavevector`, `folded_q_norm`, `CouplingScreening`, `resolve_coupling_screening` (see [Screening correction](#screening-correction-of-the-coupling-poncé-2018)). |
| `elphon/qe_matdyn.py` | WS interpolation of QE force-constant files, used by the property-only route. |
| `gen/epw_inputs.py` | EPW input helpers: `kpoints_card` / `uniform_kpoint_list` (explicit full nscf grid), `write_placeholder_ukk` (EPW without Wannierization), `epw_input`. |
| `inputs/read_QE_xml.py` | `uniform_grid_from_kpoints`: recovers the grid of an explicit `K_POINTS crystal` nscf list, which has no `monkhorst_pack` element. |
| `elphon/basis.py`, `displacements.py`, `do_elphon.py`, `io.py`, `symmetry.py`, `fold.py`, `gkq.py`, `do_gkq.py`, `dvscf_fd.py` | The finite-difference frozen-phonon route; see [Electron-Phonon Coupling](Electron-Phonon-Coupling). |

---

## Required inputs

### DFPT source (native, recommended)

1. **`pw.x` scf**, with symmetry (norm-conserving pseudopotentials).
2. **`ph.x` DFPT** on the coarse q-grid, irreducible q only, with `fildvscf`.
   PAOFLOW reads either the run directory (`_ph0/<prefix>.dvscf1`,
   `_ph0/<prefix>.q_N/`, `_ph0/<prefix>.phsave/patterns.N.xml`, `<prefix>.dynN`) or
   EPW's `pp.py` collection (`save/<prefix>.dvscf_qN`, `<prefix>.dyn_qN`, `<prefix>.phsave/`).
3. **`pw.x` nscf** on the Γ-centred coarse k-grid, `nbnd` above the PAO orbitals,
   in a directory of its own (copy `charge-density.dat` and `data-file-schema.xml`
   of the scf).  Either the **full** grid as an explicit `K_POINTS crystal` list in
   $[0,1)$ (`kpoints_card`; give the k-points with full precision, a list written
   with seven digits limits the agreement with EPW to $10^{-8}$), or the
   **irreducible wedge** with `K_POINTS automatic nk1 nk2 nk3 0 0 0`: PAOFLOW then
   rotates the stored Bloch functions to the full grid by symmetry
   (`unfold_kpoints=True` in `PAOFLOW` / `pao_hamiltonian_from_save`; the example
   drivers pass `stores_irreducible_wedge(save_dir)` and the vertex detects it by
   itself), with the same wavefunction storage savings as
   QE's own symmetry reduction (Si 6³: 16 k instead of 216; Pb 12³ with SOC: 72
   instead of 1728).  The rotated states equal those of a full-grid nscf up to a
   unitary mixing inside degenerate multiplets, to which the projections, the
   Hamiltonian and the vertex are invariant.
4. **PAOFLOW** on that save: `projections` + `projectability` +
   `pao_hamiltonian(expand_wedge=False)` (`PAOFLOW(savedir, unfold_kpoints=True)`
   for an irreducible-wedge nscf), then any dense-q driver with
   `coupling_source='dfpt'` (`source='dfpt'` for the Eliashberg drivers),
   `coupling_dir` = the `ph.x` directory and `save_dir` = the nscf save.
   `basis='auto'` applies the perturbation to the projected PAO functions when
   they are fewer than the bands.  `unfold='irreducible'` computes the vertex at
   the irreducible q only and rotates it to the stars in the PAO basis
   (`basis_shells=data_controller`); it is exact when the nscf band set does not
   cut a degenerate multiplet at any k-point, which the self-check on one star
   member verifies (`unfold_tol`).  `cache_dir` stores $g(R_e, R_p)$ for later runs.

The coarse vertex is distributed over the ranks of each node (one q at a time per
rank), and the band-basis array of EPW never exists: the memory per rank is one
potential on the FFT grid plus the cached plane-wave blocks (`cache_bytes`).

### EPW source (alternative)

The example in `examples/elphon_epw_example` follows this sequence (EPW
tutorial 04, Pb):

1. **`pw.x` scf**, with symmetry.
2. **`ph.x` DFPT** on the coarse q-grid, **irreducible q only** (symmetry on),
   with `fildvscf`. Then EPW's `pp.py` collects `dvscf`, `patterns` and `dyn`
   files into `save/`.
3. **`pw.x` nscf** on the **full** Γ-centred k-grid, given as an explicit
   `K_POINTS crystal` list with coordinates in $[0,1)$ (EPW rejects automatic
   grids). Use `nbnd` > number of PAO orbitals. `PAOFLOW.gen.epw_inputs.kpoints_card`
   writes the list in the order PAOFLOW expects.
4. **`epw.x`** with `elph = .true.`, `epbwrite = .true.`, coarse
   `nk1..3` / `nq1..3`, and **no Wannier functions**: `wannierize = .false.`
   plus a placeholder `prefix.ukk` in the run directory
   (`write_placeholder_ukk`; `epw_input` writes the matching `epw.in`).
   - With `wannierize = .false.`, EPW reads the band bookkeeping only from
     `.ukk`: kept bands, the number of occupied excluded bands, and the window
     flags. Band exclusion (e.g. semicore states) therefore goes into the
     placeholder (`exclude_bands=...`, with `nelec`). `bands_skipped` in
     `epw.in` is only used to write wannier90's `.win`.
   - EPW writes the `.epb` files before its Wannier stage. In EPW 6.0 that
     stage always reads wannier90's `<prefix>.bvec` and `<prefix>.mmn`
     (`vmebloch2wan`, regardless of `vme`), which do not exist without a
     Wannierization. `write_placeholder_ukk` therefore also writes empty stubs
     (zero b-vectors), so the stage completes on empty data. Without them
     `epw.x` exits with an error right after "The .epb files have been correctly
     written", which is harmless for PAOFLOW.
   - A Wannierized EPW run (as in the EPW tutorials) also works and gives the
     same `.epb` content, as long as the outer disentanglement window contains
     every band (no `dis_win_min/max`); otherwise `load_epw_coupling` raises.

   EPW runs one process per pool (`mpirun -np N epw.x -nk N`); the cost of the
   coarse coupling scales with the number of pools and with the kept bands.
5. **PAOFLOW** on the EPW nscf save: `projections` + `projectability` +
   `pao_hamiltonian(expand_wedge=False)`. The explicit list already covers the
   full BZ even though the save keeps the crystal symmetries.

### ph.x sources

1. **`pw.x` nscf** on the full Γ-centred grid (`nosym`, `noinv`, `nbnd` >
   `nawf`), because the vertex needs $A_k$ at every k and k+q in the dump's order.
2. **`ph.x` DFPT** phonons, producing `<prefix>.dyn<iq>`.
3. **The coupling dump**: AHC (`electron_phonon='ahc'`, norm-conserving only)
   or the PAOFLOW-patched `el_ph_mat` (`electron_phonon='interpolated'`, env
   `PAOFLOW_DUMP_ONLY=1`, any pseudopotential).

In both cases, grab `full_projections()[..., ispin]` **before**
`pao_hamiltonian`, which deletes it.

---

## Workflow 1 — coarse-q

`eliashberg_from_qe_coupling` (in `do_pao_eph.py`) interpolates **only the
electronic k-grid**; the phonon q-grid stays at the coarse DFPT resolution.

Per coarse q:
1. Rotate the coupling to the PAO gauge, giving $g_q(R_e)$ (`vertex_from_epw` /
   `vertex_from_qe_ahc` / `vertex_from_qe_elphmat`).
2. Wigner–Seitz interpolate `HRs` and $g_q(R_e)$ to a dense $N_k^3$ grid. The
   electron cache (`precompute_dense_electrons`) is computed **once**; $E(k+q)$
   and its eigenvectors are index shifts of the same grid.
3. Evaluate $\lambda_{q\nu}$ (`lambda_q_dense_ws_fast`), zeroing the $\Gamma$
   acoustic modes (QE convention).
4. Combine all q with `eliashberg_from_modes`.

For `source='epw'`, the q-points and the force constants come from the `.epb`
files (`dyn_paths` is unused), the phonon modes at each coarse q are obtained by
diagonalising $C(q)/\sqrt{M_aM_b}$ (`phonon_modes_from_force_constants`), and
`q_weights=None` selects unit weights over EPW's full coarse grid.

The per-q loop is MPI-parallel; the dense-electron cache is recomputed on every
rank. **Limitation:** $\alpha^2F$ and $\lambda$ are capped by the coarse q-mesh.

---

## Workflow 2 — dense-q (k *and* q interpolation)

`eliashberg_dense_q` (in `do_pao_eph_dense_q.py`) additionally interpolates the
**phonon q-grid**, following the Wannier–Fourier idea of EPW but in the PAO
gauge.

### The double real-space vertex $g(R_e, R_p)$

The half-vertices $g_q(R_e)$ for **every** q of the full coarse grid are
Fourier-transformed over q,

$$
g(R_e,R_p)_{ij,c} = \frac{1}{N_q}\sum_q e^{-2\pi i\,q\cdot R_p}\,g_q(R_e)_{ij,c}
\qquad(\text{`build\_g\_ReRp`}),
$$

and any dense q recovers the half-vertex by a Wigner–Seitz sum,

$$
g_q(R_e) = \sum_{R_p} W_p\, e^{+2\pi i\,q\cdot R_p}\, g(R_e,R_p)
\qquad(\text{`g\_Re\_at\_q`}),
$$

which feeds the same `lambda_q_dense_ws_fast` used by Workflow 1.

The full coarse grid is supplied directly by EPW (it unfolds the irreducible
ph.x q). For the ph.x sources it requires a full-grid (`nosym`) phonon run;
unfolding an irreducible set of PAO vertices is not implemented.

### Dense-q phonons

- **EPW source:** `phonon_interp_from_epw` builds the interpolator from the
  `.epb` force constants ($C(q)$ in Ry/bohr², ASR already applied by EPW).
  `eliashberg_dense_q` does this automatically when `phonon_at_q=None`.
- **ph.x sources:** `phonon_interp_from_dyn` reads the force-constant blocks of
  the `.dyn` files, including every star member listed in each file. Either one
  file per q of a full-grid run or the irreducible-q files of a symmetric run
  works, as long as the grid is covered.

Both share `_phonon_interp_from_force_constants`, which delegates to
`PAOFLOW.phonon.qe_force_constants`. The steps are the Fourier transform to
phonon cells (`real_space_force_constants`), a simple acoustic sum rule, and WS
interpolation and diagonalisation (`phonon_interpolator`). A `min_freq_thz`
guard drops spurious soft modes. `phonon_interp_from_dyn(...,
atom_positions=tau_cryst)` uses atom-pair WS images, as QE `matdyn.x` does. Use
them with several atoms per cell: on Si, single-site images shift the acoustic
branches by about 0.2 THz at general q.

**Stand-alone q2r.x + matdyn.x.** `PAOFLOW.phonon.qe_force_constants` provides
the same chain for phonon dispersions:
- `read_dynamical_matrices(prefix, directory)` reads `<prefix>.dyn0`–`dynN`,
  with ε∞ and Z*.
- `write_qe_force_constants` writes QE's `flfrc` file.
- `phonon_dispersion` follows the high-symmetry path of the Bravais lattice;
  `phonon_density_of_states` gives the DOS.
- `q2r_matdyn` does it all and writes `<fname>_band.dat`, `_band.labels` and
  `_dos.dat` for `GPAO.plot_phonons`.

Validation on Si (ph.x, 3³ q):
- The force constants equal QE `q2r.x` to 10⁻¹⁷.
- The frequencies equal `matdyn.x` (`asr='simple'`) to 2·10⁻⁶ THz along the
  whole path.

Polar materials get a warning: the non-analytic LO–TO term is not included.
See `examples/elphon_example/Si/phonon/q2r_matdyn.py`.

---

## Symmetry reduction of the dense q-grid

`irreducible_qmesh` (with `_crystal_point_group` to filter QE's lattice
holohedry down to the crystal point group) folds the Γ-centred `nq_dense^3`
grid to its irreducible wedge plus time reversal, with star weights. Enable it
with `sym_rots=info['s_cryst']`, `tau_cryst` and `species` from `read_nscf`. A
rank-0 line reports the reduction, e.g.

```
dense-q symmetry: 413 irreducible / 13824 full q  (33.5x fewer)
```

Full-grid and reduced sums agree to the sub-percent level (symmetry-equivalent
q are separate WS interpolations).

---

## Pair-resolved Wigner–Seitz interpolation (several atoms per cell)

Every Fourier interpolation of the dense route sums real-space cells with
Wigner–Seitz (WS) weights. The single-site construction (`_ws_lattice`) picks the
images of a cell $R$ by $|R|$ alone, as if every orbital and atom sat at the
origin. That is exact for one atom per cell (Pb), but with several atoms the
minimal image of a matrix element depends on the separation of its two centres.
Without it, the interpolation is **not symmetric** under the point group.

On MgB₂ (Mg + 2 B, 6³ k / 3³ q coarse grids) the single-site images give:

- dense bands differing between symmetry-equivalent k by up to 83 meV within
  0.3 eV of $E_F$ (16 meV on average), while the coarse 6³ points are exact;
- phonon frequencies differing by up to 1.4 THz (5.6 meV) within a q-star;
- $\lambda_q$ spreading by more than 100% within a q-star.

`_ws_lattice_pairs` keeps a cell for the element $(i,j)$ when
$\pm R + 	au_j - 	au_i$ lies in the WS cell of the supercell (EPW `use_ws`,
wannier90 `use_ws_distance`, QE `matdyn`). It is used for:

| real-space object | pair | sign of $R$ | symmetry after the fix |
|---|---|---|---|
| $H_{ij}(R)$ (`precompute_dense_electrons`) | orbital $i$, orbital $j$ | $-1$ ($H(k) = \sum_R e^{-ik\cdot R}H(R)$) | bands to 0.005 meV |
| vertex $g_{ij,c}(R_e)$ | orbital $i$, orbital $j$ | $+1$ (opposite Fourier convention) | |
| vertex $g_{ij,c}(R_p)$ (`vertex_phonon_ws_weights`) | orbital $i$, displaced atom $\kappa(c)$ | $+1$ | |
| force constants $C_{ab}(R_p)$ (`phonon_interp_from_epw`) | atom $a$, atom $b$ | $+1$ | phonons to $10^{-4}$ THz |

The pair images are switched on by passing the orbital centres,
`orbital_positions=pao_orbital_positions(pf.data_controller, nscf['at'])`, to
`eliashberg_dense_q` (with `tau_cryst`) or `eliashberg_from_qe_coupling`.
Without `orbital_positions` the single-site images are used, so runs on one-atom
cells are unchanged. The MgB₂ example and the `paoflow-gen` EPW driver pass them.

---

## Isotropic Migdal–Eliashberg: gap, Padé, analytic continuation, linearised $T_c$

`elphon/migdal_eliashberg.py` turns the Eliashberg function into
temperature-dependent superconducting properties, following EPW's `liso` /
`limag` / `lpade` / `lacon` / `tc_linear` (EPW tutorial 04). In the isotropic,
Fermi-surface-restricted limit everything depends only on $\alpha^2F(\omega)$,
$\mu^*$, the Matsubara cutoff $\omega_c$ and $T$. The solver therefore runs as a
post-processing step on `eliashberg.npz` (or on EPW's own `<prefix>.a2f`): it is
serial and takes seconds, and needs no new electron–phonon interpolation.

**Imaginary axis** (`solve_imag_iso`). On the fermionic frequencies
$\omega_n = (2n+1)\pi k_BT \le \omega_c$, with
$\lambda(n) = \int 2\omega\,\alpha^2F(\omega)/(\omega^2 + (2\pi n k_BT)^2)\,d\omega$
and the sums restricted to $n' \ge 0$,

$$
Z_n = 1 + \frac{\pi T}{\omega_n}\sum_{n'}\big[\lambda(n-n') - \lambda(n+n'+1)\big]\frac{\omega_{n'}}{R_{n'}},
\qquad
Z_n\Delta_n = \pi T\sum_{n'}\big[\lambda(n-n') + \lambda(n+n'+1) - 2\mu^*\big]\frac{\Delta_{n'}}{R_{n'}},
$$

with $R = \sqrt{\omega^2+\Delta^2}$ and $\mu^*$ applied without rescaling, as in
EPW. The fixed point uses Anderson mixing, EPW's convergence criterion
$\sum|\Delta_{\rm new}-\Delta|/\sum|\Delta_{\rm new}|$, and a warm start from the
previous temperature. A temperature at which the linearised kernel has no
eigenvalue above 1 is in the normal state ($\Delta = 0$).

**Real axis.**

- `pade_continuation`: an N-point Vidberg–Serene Padé approximant through
  `npade`% of the Matsubara points.
- `analytic_continuation_iso`: the iterative Marsiglio–Schossmann–Carbotte
  equations (PRB 37, 4965 (1988)). They are a Matsubara sum with
  $\lambda(\omega - i\omega_m)$ (FFT convolutions) plus a real-axis convolution
  with $\alpha^2F(\nu)[N(\nu) + f(\nu\mp\omega)]$. They start from the Padé
  result.
- The square root $\sqrt{\tilde\omega^2-\phi^2}$ is taken on the retarded
  branch, with $\mathrm{Re} \ge 0$ and $\mathrm{Im} \ge 0$ for $\omega > 0$.
  Negating instead of conjugating a root with $\mathrm{Im}<0$ makes the
  iteration diverge at low $T$.
- `gap_edge` returns the first solution of $\omega = \mathrm{Re}\,\Delta(\omega)$.
- `quasiparticle_dos` gives
  $N_S/N_F = \mathrm{Re}[\omega/\sqrt{\omega^2-\Delta^2(\omega)}]$.

**Linearised kernel** (`linearized_max_eigenvalue`). With the normal-state
$Z_n$, the largest eigenvalue $\rho$ of the symmetrised kernel
$D^{1/2}[\lambda(n-n') + \lambda(n+n'+1) - 2\mu^*]D^{1/2}$, with
$D_n = \pi T/(Z_n\omega_n)$, exceeds 1 below $T_c$ and crosses 1 at $T_c$
(`tc_from_eigenvalues`).

**α²F input** (`a2f_from_npz`). $\alpha^2F$ is rebuilt from the stored
`lambda_qv`, `omega_qv_thz` and `q_weights` with an EPW-like phonon smearing.
`degaussq` defaults to 0.15 meV for the ME equations and 0.5 meV for the
linearised kernel, as in the tutorial. The `a2F` stored in `eliashberg.npz` uses
5% of $\omega_{\max}$. The grid is EPW's: $\omega_j = j\,\omega_{\max}^{\rm ph}\cdot1.1/n_{\rm qstep}$.

| EPW input | PAOFLOW (`me.py` / `me.elphon.py`) | default |
|---|---|---|
| `muc` | `MU_STAR`, `--mu-star` | 0.1 |
| `wscut` | `WSCUT`, `--wscut` (eV) | 0.1 |
| `degaussq` | `DEGAUSSQ`, `--degaussq` (meV) / `DEGAUSSQ_LINEAR` | 0.15 / 0.5 |
| `npade` | `NPADE`, `--npade` (% of Matsubara points) | 90 |
| `nsiter`, `conv_thr_iaxis`, `conv_thr_racon` | `migdal_eliashberg_iso(nsiter=, conv_thr_iaxis=, conv_thr_racon=)` | 500, 1e-4, 1e-4 |
| `temps`, `nstemp` | `TEMPS`, `--temps` | tutorial list (example); `auto` (generator) |
| `lpade`, `lacon`, `tc_linear` | `--no-pade`, `--no-acon`, `--no-linear` | all on |

**Output** (`write_me_outputs`, in `output/me/`, EPW formats and names, energies
in eV):

- `<prefix>.imag_iso_<T>`: $\omega_n$, $Z$, $\Delta$
- `<prefix>.pade_iso_<T>`, `<prefix>.acon_iso_<T>`: $\omega$, Re Z, Im Z, Re Δ, Im Δ
- `<prefix>.qdos_iso_<T>`: $\omega$, $N_S/N_F$
- `gap_vs_T.dat`: $T$, $\Delta(i\omega_0)$, Padé and acon gap edges (meV), $Z(i\omega_0)$
- `max_eigenvalue.dat`: $T$, $\rho$
- `migdal_eliashberg.npz`: all of the above

`GPAO.plot_migdal_eliashberg` (used by `plot_me.py` and `plot.elphon.py`) draws
$\Delta(i\omega_n)$, $Z(i\omega_n)$, Re/Im $\Delta(\omega)$, the
quasiparticle DOS, $\Delta(T)$ and $\rho(T)$.

---

## Anisotropic Migdal–Eliashberg (Fermi-surface restricted)

Two-gap superconductors such as MgB₂ need the anisotropic equations (EPW
`laniso`, `limag`, `lpade`, second part of EPW tutorial 04). Their input is not
$\alpha^2F$ but the coupling of every pair of Fermi-surface states.

**Fermi-surface coupling** (`elphon/fermi_surface_coupling.py`). With
`fs_coupling=True`, `eliashberg_dense_q` accumulates the coupling during its
irreducible-q loop: one extra hook in `lambda_q_dense_ws_fast`, with no new
interpolation and no change to the EPW inputs.

- **States.** Bands within `fsthick_ev` of $E_F$ (EPW `fsthick`), at one
  representative per star of the dense k-grid (`irreducible_mesh`, crystal point
  group plus time reversal). `sym_rots` is required.
- **Folded pair coupling.** For states $a=(n,i)$ and $b=(m,i')$,

$$
\Lambda_{ab}(\omega) = \frac{1}{D_a}\sum_{q\in\mathrm{IBZ}} w_q
  \sum_{k\in\star(i),\,k+q\in\star(i')} \delta(\epsilon_{nk})\,
  \frac{\delta(\epsilon_{m k+q})}{N_q}\sum_\nu \frac{2|g_{mn\nu}(k,q)|^2}{\omega_{q\nu}}
  \,\delta(\omega-\omega_{q\nu}),
\qquad D_a = \sum_{k\in\star(i)}\delta(\epsilon_{nk}).
$$

  Summing irreducible q with their multiplicities $w_q$ is exact because
  $\Lambda_{Sk,Sk+Sq} = \Lambda_{k,k+q}$. The δ-weighted star average makes
  $N_F = \sum_a W_a$ ($W_a = D_a/N_k$) and $\lambda = \sum_a W_a\lambda_a/N_F$
  equal to the isotropic values, where $\lambda_a = \sum_b\int\Lambda_{ab}$ is
  EPW's $\lambda_{n\mathbf k}$.
- **Frequency grid.** $\omega$ is stored on `n_freq_fs` points (default 100) up
  to $1.05\,\omega_{\max}$. Each mode is split linearly between its two
  neighbouring points, which conserves $\lambda$ exactly. Memory is
  $8N^2 n_{\rm freq}$ bytes for $N$ states: 126 states for MgB₂ at 24³ k and
  `fsthick` = 0.2 eV.
- **Output.** `out['fs_coupling']`, saved with `write_fs_coupling` as
  `output/fs_coupling.npz`.

**Solver** (`elphon/anisotropic_eliashberg.py`). With
$\lambda_{ab}(l) = \int\Lambda_{ab}(\omega)\,\omega^2/(\omega^2+\nu_l^2)\,d\omega$
and Coulomb weights $C_{ab}$ (below):

$$
Z_a(j) = 1 + \frac{\pi T}{\omega_j}\sum_{b,j'}[\lambda_{ab}(j-j')-\lambda_{ab}(j+j'+1)]\frac{\omega_{j'}}{R_b(j')},
\qquad
Z_a\Delta_a(j) = \pi T\sum_{b,j'}[\lambda_{ab}(j-j')+\lambda_{ab}(j+j'+1)-2\mu^*C_{ab}]\frac{\Delta_b(j')}{R_b(j')}.
$$

- **Coulomb weights.** $C_{ab} = M_{ab}/N_F$, with
  $M_{ab} = D_a^{-1}\sum_q w_q N_q^{-1}\sum\delta(\epsilon_{nk})\delta(\epsilon_{m\,k+q})$
  accumulated over the same $(k, k+q)$ pairs as $\Lambda_{ab}$, as EPW does. When
  the q-grid is coarser than the k-grid (40³ k / 20³ q in the tutorial), $k+q$
  reaches only a sublattice of the k-grid. A $\mu^*$ spread uniformly over the
  Fermi surface ($C_{ab} = c_b$) then couples sublattices that the phonons do
  not, and the iteration converges to spurious gaps of opposite sign (MgB₂:
  −10 to +11 meV). The pair-restricted weights remove this.
- **Use `nq_dense = nk_dense`.** Even with consistent Coulomb weights, a
  coarser q-grid splits the equations into independent sublattice problems,
  each with its own $T_c$. For MgB₂ at 40³ k / 20³ q (EPW tutorial grids), the
  four symmetry-connected sublattice components gap out at about 30 K (75% of
  the Fermi-surface weight) and 41 K (25%). Both `eliashberg_dense_q` and
  `migdal_eliashberg_aniso` warn in this case.

- **Separable kernel.** `matsubara_kernel` takes the SVD of
  $\omega_\beta^2/(\omega_\beta^2+\nu_l^2)$ on the phonon grid and the bosonic
  frequencies (rank about 20 at 5 K for $\omega_c$ = 0.5 eV), so an iteration
  costs $N^2 r n_\omega$.
- **Fixed point.** Anderson mixing, shared with the isotropic solver
  (`anderson_fixed_point`), with a warm start from the previous temperature. A
  gap below $10^{-7}$ eV is the normal state.
- **Real axis.** Padé state by state (`pade_continuation_aniso`, batched
  `pade_coefficients` / `pade_eval`), gap edges, and the quasiparticle DOS
  $\sum_a c_a\,\mathrm{Re}[\omega/\sqrt{\omega^2-\Delta_a^2(\omega)}]$.
  There is no anisotropic analytic continuation (`lacon`), as in the tutorial.
- **$T_c$.** `linearized_eigenvalues_aniso` gives the leading eigenvalue of the
  linearised equations (ARPACK on the matrix-free operator), and `Tc_gap` comes
  from $\Delta_{\max}^2(T)\to0$.
- **Consistency.** For a single state (or identical states) the equations are
  the isotropic ones with $\Lambda(\omega) = 2\alpha^2F(\omega)/\omega$. The
  solver reproduces `solve_imag_iso` to $10^{-14}$ and
  `linearized_max_eigenvalue` to $10^{-13}$.

**Output** (`write_me_aniso_outputs`, in `output/me_aniso/`, EPW names):

- `<prefix>.lambda_FS`, `<prefix>.lambda_k_pairs`: $\lambda_{n\mathbf k}$ per state, and its distribution
- `<prefix>.imag_aniso_<T>`: $\omega_j$, $\epsilon-E_F$, $Z$, $\Delta$
- `<prefix>.imag_aniso_gap0_<T>`, `<prefix>.pade_aniso_gap0_<T>`: distributions of $\Delta_{n\mathbf k}(i\omega_0)$ and of the Padé gap edges, computed as EPW's `gap_distribution_FS` (`epw_gap_distribution`: 300 bins, half-bin Gaussians) and written in EPW's columns ($T$ + scaled, Δ in meV, $T$, scaled, unscaled), so EPW's gnuplot scripts read them unchanged
- `<prefix>.imag_aniso_gap_FS_<T>`: $\Delta_{n\mathbf k}(i\omega_0)$ per state
- `<prefix>.pade_aniso_<T>`: Fermi-surface averaged $Z(\omega)$, $\Delta(\omega)$
- `<prefix>.qdos_<T>`: $N_S/N_F$
- `gap_vs_T_aniso.dat`, `max_eigenvalue_aniso.dat`, `migdal_eliashberg_aniso.npz`

`GPAO.plot_migdal_eliashberg_aniso` draws the distribution of
$\Delta_{n\mathbf k}(i\omega_0)$ at every temperature as EPW tutorial 04 does
(`plot_gap0.gnu`): filled from $T$ to $T + 3\times10^{-3}\,\rho(\Delta)$
(`distribution_scale`), with the isotropic gap overlaid when given.
It also draws the gap distributions, the quasiparticle DOS, the
$\lambda_{n\mathbf k}$ distribution and $\rho(T)$.

| EPW input | PAOFLOW | default |
|---|---|---|
| `laniso`, `fsthick` | `eliashberg_dense_q(fs_coupling=True, fsthick_ev=)` | off; `fs_window` smearings |
| `degaussw` | `sigmas_ry` | |
| `adapt_smearing` | `adaptive_smearing=` ([adaptive widths](#adaptive-smearing-of-the-deltas) of the Fermi-surface delta, floored at `sigmas_ry`) | off |
| `muc`, `wscut`, `npade` | `migdal_eliashberg_aniso(mu_star=, wscut=, npade=)` | 0.1, 0.1 eV, 90 |
| `nsiter`, `conv_thr_iaxis` | `nsiter=`, `conv_thr_iaxis=` | 500, 1e-4 |
| `tc_linear` | `linearized_eigenvalues_aniso` | |

---

## Phonon-assisted optical absorption

`PAOFLOW.elphon.phonon_assisted_absorption` reuses the dense-q interpolation for
the second-order (photon + phonon) absorption of indirect-gap semiconductors,
EPW's `lindabs` (`indabs.f90`; Noffsinger et al., PRL **108**, 167402 (2012);
EPW tutorial 06). The worked example is `examples/elphon_example/Si`.

For an initial state $i$ at $\mathbf{k}$, a final state $j$ at
$\mathbf{k}+\mathbf{q}$ and a phonon $\nu$, the two time orderings give

$$
S^{a/e}_{\alpha} = \sum_m \frac{g_{jm,\nu}\, v^{\alpha}_{mi}(\mathbf{k})}
{\varepsilon_{m\mathbf{k}} - \varepsilon_{j\mathbf{k}+\mathbf{q}} \pm \omega_{\mathbf{q}\nu} + i\eta}
+ \sum_m \frac{v^{\alpha}_{jm}(\mathbf{k}+\mathbf{q})\, g_{mi,\nu}}
{\varepsilon_{m\mathbf{k}+\mathbf{q}} - \varepsilon_{i\mathbf{k}} \mp \omega_{\mathbf{q}\nu} + i\eta},
$$

for phonon absorption ($a$, upper signs) and emission ($e$), and

$$
\mathrm{Im}\,\varepsilon_{\alpha\alpha}(\omega) = \frac{8\pi^2 g_s}{\Omega\,\omega^2}
\frac{1}{N_k}\sum_{\mathbf{q}} w_{\mathbf{q}} \sum_{ij\nu\mathbf{k}} P^{a/e}\,
\frac{|S^{a/e}_{\alpha}|^2}{2\omega_{\mathbf{q}\nu}}\,
\delta(\varepsilon_{j\mathbf{k}+\mathbf{q}} - \varepsilon_{i\mathbf{k}} - \omega \mp \omega_{\mathbf{q}\nu}),
$$

with $P^a = n f_i(1-f_j) - (n+1)(1-f_i)f_j$ and
$P^e = (n+1) f_i(1-f_j) - n(1-f_i)f_j$, in Rydberg units ($8\pi^2 g_s = 16\pi^2$
for $g_s = 2$, EPW's `cfac`). $g$ is the vertex without $1/\sqrt{2\omega}$, as
in $\lambda$. The kernels transliterate EPW's loops, including the nine $\eta$,
the Gaussian and Lorentzian deltas cut at six smearings, the `fsthick` window
for initial, final and intermediate states, and `eps_acoustic`. The unit tests
compare them with a literal loop transcription of `indabs_main` and `dirabs`.

**Velocity in the gauge of $g$.** The interference of the two paths requires
$g$ and $v$ in the same eigenvector gauge. Both are therefore evaluated on the
dense grid and rotated with the same cached eigenvectors:
- the coupling by `band_vertex` (factored out of `lambda_q_dense_ws_fast`);
- the velocity by `band_velocities`, from `precompute_dense_electrons(...,
  velocities=True)`.

The velocity is the PAO analogue of EPW's `vme = 'wannier'`:

$$
\mathbf{v}_{ij}(\mathbf{k}) = \partial_{\mathbf{k}} H_{ij}(\mathbf{k})
+ i H_{ij}(\mathbf{k})(\boldsymbol{\tau}_j - \boldsymbol{\tau}_i),
$$

i.e. minus PAOFLOW's `dHksp`. With `position_dipoles=<DataController>` the
Löwdin position dipoles of the basis are added, $+\,i[H(\mathbf k), \mathbf d(\mathbf k)]$,
i.e. $i(\varepsilon_n - \varepsilon_m)\, d_{nm}$ in the band basis
(`position-dipoles.md`): this completes the commutator $i[H, \mathbf r]$ of the
PAO Hamiltonian, which already contains the non-local pseudopotential. (The
term $i[V_{\rm NL}, \mathbf r]$ offered until October 2026 double counted it
and was removed, `nonlocal-velocity.md`.) The dipole term is purely
off-diagonal in the band basis:
- **Diagonal and degenerate blocks.** By Hellmann–Feynman these are the group
  velocities $\partial_{\mathbf k}\varepsilon_{n\mathbf k}$ of the PAO
  Hamiltonian and stay untouched.
- **Interband dipoles.** There the dipoles supply the on-site s–p, p–d matrix
  elements that $\partial H/\partial k$ lacks.

All transport quantities therefore use the plain group velocities: α²F_tr, the
plasma frequency and the carrier rates, through `_diagonal_velocities`.

The validation checks are:
- **Exact model.** An exactly solvable two-site tight-binding model checks that
  the whole chain (`vertex_pao_R` → `band_vertex`, `band_velocities` →
  kernel) reproduces the exact gauge-invariant spectrum off the coarse grid.
- **Mutation check.** The test fails if the position term is dropped or
  flipped, or if the vertex is conjugated.
- **Direct term on Si.** The direct term matches `dielectric_tensor` on the
  same Hamiltonian, with and without the position dipoles.

**Driver.** `phonon_assisted_absorption_dense_q(A, HRs, kpts_cryst, bg, at,
alat, cell_volume, epw_dir, qgrid, kgrid, masses_amu=, nelec=, nk_dense=,
nq_dense=, omega_ev=, temps_k=, degauss_ev=, fsthick_ev=, ...)` works in four
stages:
1. It calls `prepare_dense_vertex`, the setup shared with
   `eliashberg_dense_q`: EPW coupling, long-range guard, phonon interpolator,
   node-shared $g(R_e,R_p)$.
2. It caches the band velocities on the dense grid (one copy per node).
3. It loops over (irreducible q, k-block) tasks distributed over MPI ranks.
4. `write_absorption_outputs` writes EPW-format `epsilon2_indabs_<T>K.dat`,
   `epsilon2_indabs_lorenz<T>K.dat`, `epsilon2_dirabs_<T>K.dat`,
   `alpha_<T>K.dat` and `absorption.npz`.

`GPAO.plot_phonon_assisted_absorption` plots them.

| EPW `lindabs` input | PAOFLOW | default |
|---|---|---|
| `nkf`, `nqf` | `nk_dense`, `nq_dense` (`nk_dense % nq_dense == 0`) | 12, 6 |
| `omegamin`, `omegamax`, `omegastep` | `omega_ev=(min, max, step)` | (0.05, 3.0, 0.05) eV |
| `temps` | `temps_k` | 300 K |
| `degaussw` | `degauss_ev` | 0.05 eV |
| `adapt_smearing` | `adaptive_smearing=`, `min_degauss_ev` ([adaptive widths](#adaptive-smearing-of-the-deltas) per transition) | off, 1 meV |
| `fsthick` | `fsthick_ev` | 4 eV |
| `efermi_read`, `fermi_energy` | `fermi_energy_ev` (`None` = mid-gap) | mid-gap |
| `eps_acoustic` | `eps_acoustic_cm` | 0.1 cm⁻¹ |
| `n_r` (tutorial: constant 3.4) | `refractive_index` (constant, `n` on the photon grid, or `(omega, n)` of the `dielectric_tensor` run: `external_refractive_index`) | 3.4; the drivers use n(ω) of `DIRECT_DIELECTRIC_DIR` when given |
| `mp_mesh_k` | `sym_rots` (folds **q**; the polarisation average is exact) | |
| `eig_read`, `scissor` | a scissor (`target_gap_ev` / `scissor_ev`), or an eACBN0 PAO Hamiltonian through `ELECTRONS_SAVE` (energies *and* eigenvectors; the coupling stays PBE in the orbital basis) | PAO (DFT) energies |

Some limitations and conventions apply:
- **Irreducible q.** With `sym_rots`, the dense q-grid is folded to its
  irreducible wedge. Only the polarisation average is invariant, and it is
  returned in all three components. Pass `sym_rots=None` for the resolved
  diagonal.
- **Above the direct gap.** The amplitudes diverge at direct transitions, as
  in EPW. Read the indirect spectrum below the direct gap.
- **Not implemented.** Free-carrier and impurity absorption (`carrier`,
  `ii_g`) and QDPT (`loptabs`).

### Thermal emissivity

`thermal_emissivity(results, thickness_m, angles_deg, ntheta, eta_ev)` turns
the temperature-dependent $\mathrm{Im}\,\varepsilon(\omega, T)$ into the
emissivity of a free-standing slab. By Kirchhoff's law, the emissivity of a
body at uniform temperature equals its absorptance. The run must be made with
`fermi_energy_ev='intrinsic'` and `kk_omega_max_ev`.

For every temperature:
1. **Dielectric function.** $\varepsilon_2 = \varepsilon_2^{dir}$ (all bands,
   smeared by `kk_degauss_ev`) plus $\varepsilon_2^{ind}$, tapered off just
   below the direct gap. $\varepsilon_1$ follows by Kramers–Kronig
   (`response.do_epsilon.kramers_kronig_eps1`, Maclaurin's formula).
2. **Slab absorptance.** It is evaluated with
   `response.do_epsilon.slab_directional_absorptance`: incoherent multiple
   reflections, per polarization,
   $A_\sigma = (1-R_\sigma)(1-\tau)/(1-R_\sigma\tau)$ with
   $\tau = \exp(-2\omega\,\mathrm{Im}\sqrt{\tilde n^2-\sin^2\theta}\,d/c)$.
3. **Averages.** The hemispherical average
   (`spectral_hemispherical_emissivity(..., ene, thickness_m)`) and the
   Planck-weighted total at the **same** temperature
   (`total_hemispherical_emissivity`). The opaque half-space ($1-R$) is
   reported alongside.

Why each ingredient matters:
- **Finite thickness.** Below the direct gap $\kappa \sim 10^{-3}$, so
  $1-R \approx 0.7$ whatever the absorption. Only a finite thickness lets
  $\alpha(T)$ show up in the emissivity.
- **Free carriers.** The blackbody spectrum at 300–1500 K lies mostly below
  the gap. There, the absorption comes from thermally excited carriers: the
  intraband ($i=j$) phonon-assisted terms of the same kernel, weighted by
  $f(1-f)$, i.e. EPW's free-carrier kernel.
- **Charge-neutral $E_F(T)$.** `intrinsic_fermi_levels` sets the carrier
  density. The $\omega$ grid must start near 0.01 eV with a smearing of about
  0.01 eV. `planck_coverage` reports the fraction of the blackbody spectrum
  inside the grid.
- **Kramers–Kronig range.** The PAO spectrum above about 10 eV contains
  transitions into poorly projected (shifted) bands and violates the f-sum rule.
  For Si, $\int\omega\varepsilon_2\,d\omega$ is 4× too large at 20 eV and 85%
  of $\pi\omega_p^2/2$ at 10 eV. Keep `kk_omega_max_ev` near 10 eV and check
  `f_sum_ratio`.
- **Smearing of the KK input.** A weakly smeared, under-sampled direct
  spectrum makes $\varepsilon_1$ oscillate, even below zero, and spoils $R$.
  This is why the wide-range direct term has its own smearing,
  `kk_degauss_ev = 0.1` eV.

**Extended-basis direct term (recommended).** PAOFLOW's optical route uses the
extended basis with the position dipoles. `direct_dielectric=(ω, ε₁, ε₂)`
(`read_direct_dielectric(outdir)` reads the `epsr_*/epsi_*.dat` files of such a
`dielectric_tensor` run) replaces the wide-range Kramers–Kronig term:
- ε₁ is the external one, plus the KK shift of the tapered phonon-assisted
  term;
- ε₂ is the external one above the direct gap, and this run's Gaussian direct
  term plus the phonon-assisted term below it.

The Drude–Lorentz broadening tail of the external ε₂ would otherwise dominate
the sub-gap absorption. The extended run only needs an nscf with more bands than
orbitals; EPW is not rerun. For Si (24³) it gives ε₁(0) = 17.3 and an f-sum
ratio of 0.88.

**Scissor (band-gap correction).** ph.x supports Hubbard U only with atomic
projectors, which barely opens the Si gap, so the e-ph run stays PBE. The
scissor corrects the band energies only:
- **How the shift is set.** `phonon_assisted_absorption_dense_q(...,
  target_gap_ev=..., scissor_velocity=...)` shifts the conduction bands rigidly
  by $\Delta(T) = E_g^{target}(T) - E_g^{DFT}$. Alternatively,
  `scissor_ev=` gives Δ directly.
- **Target gap.** It can be the experimental gap, or the dense-grid gap of an
  eACBN0 (DFT+U+V, ortho-atomic) PAO Hamiltonian (`pao_band_gaps`). It can be
  made temperature dependent with `varshni_gap(E0, alpha, beta)` (Si:
  `SILICON_VARSHNI`).
- **What the shift enters.** It enters the δ functions, the second-order
  denominators, the occupations and the charge-neutral $\mu(T)$.
- **Velocities.** With `scissor_velocity='renormalized'` (default) the
  valence–conduction velocities scale as
  $(\Delta\varepsilon+\Delta)/\Delta\varepsilon$ (Levine–Allan; Del
  Sole–Girlanda). This keeps the length-gauge matrix elements, so the direct
  spectrum moves rigidly. `'ks'` keeps the Kohn–Sham velocities, as EPW's
  `scissor` does.
- **External direct ε.** It gets the same scissor
  (`scissor_shift_dielectric`: ε₂ shifted rigidly, ε₁ corrected by the KK
  difference). With `'ks'` the shifted ε₂ is also multiplied by
  $((\omega-\Delta)/\omega)^2$ (constant momentum matrix elements). For
  Δ < 0 this factor diverges as ω → 0, so it would amplify the broadening
  tail of the external ε₂ below its onset $E_{\rm on}$. Below
  $E_{\rm on}+\Delta$ the factor is therefore held at
  $(E_{\rm on}/(E_{\rm on}+\Delta))^2$ (`dielectric_onset_ev`), and a Δ that
  closes the direct gap raises an error. Before October 18, 2026, the tail was
  amplified: at λ > 4 μm and high T the emissivity of KS runs with Δ < 0
  came out 0.01–0.02 too high (eACBN0 Si, Δ = −0.54 eV at 1073 K).
- **Cost.** Temperatures with equal Δ are processed together. A T-dependent
  scissor repeats only the second-order kernel per temperature.

**Beyond-PBE electrons: an eACBN0 Hamiltonian instead of a scissor.** The
scissor fixes the gap but keeps the PBE eigenvectors and the Levine–Allan
velocity renormalisation. The eACBN0 (DFT+U+V, ortho-atomic) PAO Hamiltonian
of the same crystal can replace the PBE one altogether
(`elphon/pao_electrons.py`; drivers: `ELECTRONS_SAVE` / `--electrons-save`):
- **Why in PAOFLOW.** `ph.x` refuses Hubbard with ortho-atomic projectors and any
  U+V (`phq_readin.f90`), and `epw.x` (up to 6.0) has no Hubbard code at all:
  its `epbwrite` path reaches `commutator_Vhubx_psi` on units it never opens.
  EPW's only beyond-DFT hook, `eig_read`, replaces the eigenvalues and keeps
  the DFT wavefunctions; PAOFLOW discards the `.epb` eigenvalues anyway. The
  dense pipeline, however, takes the coupling (`A`, `.epb`) and `HRs` as
  separate inputs: `prepare_dense_vertex` builds $g$ in the orthonormal PAO
  basis from the EPW save, `precompute_dense_electrons` diagonalises whatever
  `HRs` it is given, and `band_vertex` rotates $g$ with those eigenvectors.
- **What is mixed.** $g_{mn}(\mathbf k,\mathbf q) = V_U^\dagger(\mathbf k+\mathbf q)\,
  g^{\rm PAO,PBE}\,V_U(\mathbf k)$ with the eACBN0 eigenvectors $V_U$, the eACBN0
  energies in every δ function, denominator, occupation and $\mu(T)$, and the
  velocities of the eACBN0 $H(R)$ (plus the position dipoles when requested).
  Δ(0) = 0: no scissor at 0 K. The PAO analogue of `eig_read`, with eigenvectors.
- **What is not.** The Hubbard contribution to the coupling ($dV_{\rm Hub}/du$
  and the response of the occupations; `ph.x` computes it only for DFPT+U with
  atomic projectors), the change of the screening of `dvscf` (beyond the
  [Poncé rescaling](#screening-correction-of-the-coupling-poncé-2018)), and the Hubbard
  term $i[V_U, \mathbf r]$ of the velocity (the intersite V lives in $H(R \ne 0)$
  and is captured by $\partial H/\partial k$; the on-site U lacks the
  intra-atomic dipoles). The projector onto the PBE nscf bands truncates
  $g^{\rm PAO}$: eACBN0 states with weight outside their span see a truncated
  vertex (far above the window only). The f-sum ratio of the eACBN0 and PBE
  `dielectric_tensor` runs (`thermal_emissivity['f_sum_ratio']`) is the
  diagnostic for the missing velocity term.
- **Basis identity.** Same pseudopotentials, structure and atom order, BASIS
  tables, `configuration` and `pthr`; `check_same_pao_basis` compares species,
  positions, lattice, orbital identities and configuration and raises on any
  mismatch; `bnd` and `shift` (the projectability outcome) differ between
  functionals and are reported. The eACBN0 nscf need not be on the EPW k list:
  the H grid is read from `HRs.shape` (`pao_hamiltonian_from_save` infers
  `expand_wedge` from the k set).
- **Temperature dependence.** `varshni_shift(alpha, beta)` is the `scissor_ev`
  $\Delta(T) = -\alpha T^2/(T+\beta)$ on top of the Hamiltonian's own gap. In the
  drivers `--varshni` without a 0 K target now means this relative shift (it
  used to be an error); `--gap-reference-save` and `--electrons-save` are
  mutually exclusive.
- **Direct dielectric function.** `DIRECT_DIELECTRIC_DIR` must then come from a
  `dielectric_tensor` run on the eACBN0 optics save (`optics_eps.py --savedir`).
  `thermal_emissivity(..., direct_scissor=)` sets its shift: `'same'` (default)
  the run's Δ(T); `'thermal'` only Δ(T) − Δ(0) (`results['scissor_0k_ev']`),
  when the external spectrum already has the right 0 K direct gap and a scissor
  (`--target-gap-ev`) corrects the indirect gap only — eACBN0 Si: direct gap
  3.44 eV, indirect 0.22 eV too high, so `'same'` would move the direct
  transitions to 3.22 eV and raise ε₁(0.05 eV, 20 K) from 12.4 to 14.0, while
  `'thermal'` keeps 12.4 and brings n at 0.45–2 μm to −5.5…+4.7 % of experiment;
  `'none'`; or explicit shifts. With `'ks'` velocities the thermal shift
  overestimates dn/dT (×1.1–1.8 for Si), with `'renormalized'` it
  underestimates it (×0.65–0.9). The taper below the direct gap stays at the run's
  scissor-corrected direct gap, where its phonon-assisted term diverges.
  `examples/elphon_example/Si/emissivity_from_absorption.py` recomputes the
  emissivity of a finished run this way, without a new kernel pass;
  `thermal_emissivity` compares the onset of the external $\varepsilon_2$ (5 %
  of its maximum, which lands on the direct gap for PAOFLOW's Si spectrum)
  with the Hamiltonian's direct gap and warns beyond 0.3 eV
  (`check_direct_dielectric_gap`), because only Δ(T) is applied to it.
- **Energy scale.** `HRs` is on the scale of its own save (Fermi level or
  highest occupied level at zero); `FERMI_ENERGY_EV`, and the
  `scattering_rates.npz` of the mobility run for `pf.transport`, refer to the
  electrons save.
- **Provenance.** The drivers add `coupling_save`, `electrons_save`,
  `electrons_dft` (`PBE`, `DFT+U+V`, ...) and `electrons_hubbard` (the U and V
  values of the save, `hubbard_info_from_save`) to `absorption.npz` /
  `carrier_scattering.npz`; `OpticsModel.describe()` of the thermo-optics
  workflow shows them.
- **Hubbard velocity commutator.** Covered by the position dipoles of the PAO
  basis (`position_dipoles=`, see `position-dipoles.md`): $V_U$ sits inside
  $H(R)$, so $i[H, \tau + d]$ contains its commutator.
- **Later stages (not implemented).** The Hubbard contribution to $g$ by finite
  differences of the eACBN0 PAO Hamiltonian on displaced supercells. The
  screening of the coupling can be corrected with the
  [Poncé rescaling](#screening-correction-of-the-coupling-poncé-2018); its
  target should be the measured dielectric constant, not the eACBN0 one (see
  there).

For Si with the Varshni experimental gap (500 µm, 24³/12³, extended-basis ε),
the slab becomes opaque at 700–900 K: ε = 0.003, 0.25 and 0.61 at 500, 700 and
900 K.

Limitations:
- **Metals are not supported.** `thermal_emissivity` raises
  `NotImplementedError` when there is no gap. The intraband (Drude) term, with
  ω_p from the Fermi-surface velocities and 1/τ_tr(T) from a transport α²F,
  is the planned next step. The driver's mid-gap and `'intrinsic'` Fermi
  levels also require a gap.
- Without a scissor, the DFT gap puts the intrinsic carriers, and the rise of
  the emissivity with $T$, at too low a temperature.
- **Lowest photon energies at high T.** Kramers–Kronig misses the free-carrier
  weight below the grid (the Drude part, which would make ε₁ negative). ε₁ at
  the lowest energies is unreliable at the highest temperatures, but the Planck
  weight there is below 1%.
- There is no band-gap renormalisation with $T$, no multiphonon lattice
  absorption, and no doping.

### Metals: transport Eliashberg function, resistivity and infrared optics

`eliashberg_dense_q(..., transport=True, alat=, cell_volume=)`
(driver flag `--transport`) also computes three quantities:
- **λ_tr and α²F_tr.** The double-delta sum of λ is weighted by
  $\mathbf v_{n\mathbf k}\cdot(\mathbf v_{n\mathbf k}-\mathbf v_{m\mathbf k+\mathbf q})/\langle v^2\rangle_{FS}$
  (`lambda_q_dense_ws_fast(..., transport=True)`) and passed through
  `eliashberg_from_modes`.
- **Plasma-frequency tensor.** `omega_plasma2_ev2` and the isotropic
  `omega_plasma_ev` come from `fermi_surface_plasma_frequency`.
- **Band velocities.** They are cached by
  `precompute_dense_electrons(..., diagonal_velocities=True)`.

All of this is stored in `eliashberg.npz` and `alpha2F_tr.dat`. The
scattering-channel layer, [Scattering channels](scattering-channels.md), turns
it into the DC resistivity, Allen's frequency-dependent rate, the extended-Drude
dielectric function and the thermal emissivity of the metal. The generated
`transport.elphon.py` runs that chain.

For Pb (48³/24³) the results are:
- λ_tr = 0.94 and ħω_p = 8.3 eV;
- ρ(300 K) = 16 µΩ cm, against about 21 measured;
- total emissivity 0.037 at 300 K.

The k- and q-grids do not have to be equal for these Fermi-surface averages:
equal grids are required only by the anisotropic Migdal–Eliashberg equations.
Both still have to be converged:
- **k-grid.** It has to resolve the Fermi surface at the chosen smearing (Pb:
  ≥ 48³ at 0.05 eV).
- **q-grid.** For Pb, going from 24³ to 48³ q (at 48³ k) lowers λ_tr by 7%,
  from 0.94 to 0.87.

---

## Carrier transport in semiconductors and insulators (SERTA / MRTA)

α²F_tr is a Fermi-surface average, one rate at E_F. In a semiconductor the
current is carried by the few states within some k_BT of the band edges, so
transport needs the **state-resolved** rate 1/τ_nk(T).
`elphon/carrier_scattering.py` computes it on the dense EPW+PAO interpolation.
It also works for metals, using the states near E_F. The electrons can come
from an eACBN0 PAO Hamiltonian (`ELECTRONS_SAVE`, see the absorption section):
the gap, the masses, the velocities and the carrier statistics are then the
eACBN0 ones, with `scissor_ev=varshni_shift(...)` for the temperature
dependence; the rates still use the PBE coupling in the orbital basis.

**Rates.** The imaginary part of the Fan–Migdal self-energy (SERTA):

$$
\frac{1}{\tau_{n\mathbf k}} = \frac{2\pi}{\hbar}\sum_{m\nu\mathbf q} w_{\mathbf q}|g_{mn\nu}(\mathbf k,\mathbf q)|^2
\big[(n_{\mathbf q\nu}+f_{m\mathbf k+\mathbf q})\,\delta(\varepsilon_{m\mathbf k+\mathbf q}-\varepsilon_{n\mathbf k}-\omega_{\mathbf q\nu})
+ (n_{\mathbf q\nu}+1-f_{m\mathbf k+\mathbf q})\,\delta(\varepsilon_{m\mathbf k+\mathbf q}-\varepsilon_{n\mathbf k}+\omega_{\mathbf q\nu})\big].
$$

- **Coupling.** $|g|^2 = |\tilde g|^2/2\omega$, with $\tilde g$ from
  `band_vertex`.
- **Delta function.** A Gaussian of width `smearing_ev`, cut at 6σ, or with
  `adaptive_smearing=a` one width per final state,
  $a\,|\mathbf v_{m\mathbf k+\mathbf q}|\,\Delta q$ floored at `min_smearing_ev`
  ([adaptive smearing](#adaptive-smearing-of-the-deltas)). Modes below
  `eps_acoustic_cm` are skipped.
- **MRTA.** Each process is weighted by
  $1-\mathbf v_{n\mathbf k}\cdot\mathbf v_{m\mathbf k+\mathbf q}/|\mathbf v_{n\mathbf k}|^2$
  (Claes *et al.* 2022), the state-resolved analogue of α²F_tr. As in EPW, the
  weight is 1 for $|\mathbf v_{n\mathbf k}|^2 \le 10^{-4}$ Ry a.u.
  (≈ 1.1·10⁴ m/s).
- **Velocities.** The group velocities (diagonal of the PAO velocity
  operator, averaged over degenerate subspaces; the dipole term is
  off-diagonal, see above). The work is shared between MPI ranks.

**Symmetry.** The point group (with time reversal) reduces the q-sum to the
**irreducible** dense q-points with star weights. Each k with window states is
then averaged over its star, which is exact for the scalar rates. Writing
$F(\mathbf k,\mathbf q)$ for the summand:

$$
\sum_{\mathbf q}F(\mathbf k,\mathbf q)
=\frac{1}{|G|}\sum_{S\in G}A(S^{-1}\mathbf k),\qquad
A(\mathbf k)=\sum_{\mathbf q\in\mathrm{IBZ}}w_{\mathbf q}F(\mathbf k,\mathbf q).
$$

- **Pruning.** A (k, q) pair is skipped unless k+q has a state within one
  phonon energy plus 6σ of the window. A q-point's vertex is built only when
  some k needs it.
- **Bands.** Only the window bands keep eigenvectors
  (`precompute_dense_electrons(..., keep_bands=)`).
- **Check.** On Si (12³ k / 6³ q), the star-averaged rates agree with a
  full-q run to 2% (median) for the states that carry current. The
  interpolated |g|² is symmetric over stars to about 1%.

**Carrier statistics.** `fermi_levels_for_doping` solves charge neutrality:

$$
\frac{g_s}{N_k}\sum f = N_{el} + N_d\Omega,
$$

with `doping_cm3` > 0 for n type, < 0 for p type and 0 for intrinsic. An
optional scissor (`target_gap_ev`, fixed or Varshni) shifts the conduction
bands for the statistics only. The rates depend only on energy differences
within a band, so the scissor leaves them unchanged.

**Two routes to transport:**
- **(a) `rta_transport` (k-resolved, the reference).**
  $\sigma_{\alpha\beta} = \frac{e^2 g_s}{\Omega N_k}\sum v^\alpha v^\beta\tau_{n\mathbf k}(-\partial f/\partial\varepsilon)$
  over the window states, split into conduction (electrons) and valence
  (holes). It gives n_e and n_h, the mobilities μ = σ/(e n), and the Seebeck
  coefficient from the energy moment of the same sum.
- **(b) `energy_resolved_rates` → `EnergyResolvedRate`.** The
  transport-weighted $\tau_{eff}(E)=\sum v^2\tau\,\delta(E-\varepsilon)/\sum v^2\delta(E-\varepsilon)$
  in 5 meV bins. It is written to `scattering_rates.npz` for
  `pf.transport(scattering_channels=[EnergyResolvedRate.from_npz(...)])`. See
  [Scattering channels](scattering-channels.md) for the band-edge alignment.

`write_mobility_outputs` writes four files:
- `mobility.dat` (T, μ_F, n_e, n_h, SERTA/MRTA mobilities, σ, Seebeck);
- `rates_<T>K.dat`;
- `carrier_scattering.npz`;
- `scattering_rates.npz`.

`GPAO.plot_carrier_mobility` plots μ(T) and 1/τ against the energy from the
band edge.

**Validation.**
- **Si (intrinsic, 300 K, MRTA, 48³ k = q, σ = 10 meV).** The like-for-like
  reference is Poncé *et al.* 2018 (PBE, PBE lattice, no SOC): μ_e = 1343 and
  μ_h = 722 cm²/Vs, converged.

  | DFPT q / H(R) source | μ_e | μ_h |
  |---|---|---|
  | 3³ / EPW 6³ nscf | 1763 | 424 |
  | 3³ / 18³ scf | 1376 | 407 |
  | 6³ / 6³ | 1492 | 440 |
  | 6³ / 18³ | 1167 | 423 |

  - **Band structure.** EPW's 6³ nscf makes the transverse electron mass 10%
    too light. H(R) can come from a denser PAO run in the same basis.
  - **Coupling grid.** The 3³ DFPT q-grid inflates μ_e by 15–20%.
  - **Sampling.** Uniform k/q grids still move μ_h by about 5% per step at
    48–72³.
  - **Open.** The holes are about 40% below the reference.
  - **Not the cause.** The smearing (< 10%) and MRTA vs SERTA (< 3%).
- **Pb (48³ k / 24³ q).** ρ from the MRTA rates is 15.3 µΩ cm at 300 K,
  against 16.2 from α²F_tr.

**Not included:**
- **Long-range couplings.** The Fröhlich term: polar materials raise
  `check_long_range_terms`. The dynamical quadrupoles: in Si-like crystals the
  missing quadrupoles bias the small-q coupling (Brunin *et al.* 2020;
  Jhalani *et al.* 2020).
- **The iterative BTE.** MRTA is usually close to it.
- **Ionized-impurity scattering.** Add a phenomenological channel in route (b).
- **Adaptive smearing.**

---

## Spectral functions and ARPES

`elphon/self_energy.py` turns the coupling into electron self-energies for spectral functions
and the ARPES module (`PAOFLOW.arpes`, see [arpes.md](arpes.md)):

- `eliashberg_self_energy(omega, a2f, w, T)`: the isotropic Engelsberg–Schrieffer
  $\Sigma(\omega, T)$ of $\alpha^2F$ (digamma closed form; $T = 0$ with the logarithmic kernel),
  broadcast over per-state $\alpha^2F_{n\mathbf k}$ for the anisotropic case;
  `mass_enhancement(w, sigma)` returns $\lambda = -\mathrm{Re}\,\Sigma'(0)$ and $Z$;
- `fan_migdal_kernel`: the Fan–Migdal self-energy of a block of states at one q, with the
  Lorentzian width (constant or per intermediate state); its on-shell imaginary part is the
  SERTA sum of `carrier_scattering` with a Lorentzian delta;
- `fan_migdal_self_energy(...)`: the dense-q driver at arbitrary k-points (an ARPES path or
  mesh), sharing the vertex setup of `carrier_scattering_dense_q`; intermediate states
  $\mathbf k+\mathbf q$ from `elph_bloch.bloch_states_at` (the Bloch terms are now kept in the
  dense-electron cache), full dense q-grid split over the MPI ranks, carrier statistics as the
  mobility driver; returns $\Sigma_{n\mathbf k}(\omega, T)$ ``(ntemp, nk, nband, nw)`` relative to
  the Fermi level of each temperature;
- `TabulatedSelfEnergy.from_result(result, kpoints)` feeds it to `PAOFLOW.arpes.simulate`.

## Screening correction of the coupling (Poncé 2018)

DFT underestimates the gap of a semiconductor, so DFPT overscreens the phonon
perturbation: `dvscf`, and with it $g$, is too weak at long wavelength.
Recomputing the phonons with a better screening is prohibitive. Poncé, Margine
and Giustino (PRB 97, 121201(R) (2018)) rescale the matrix elements instead:

$$
g'_{mn\nu}(\mathbf k,\mathbf q) = g_{mn\nu}(\mathbf k,\mathbf q)\,
\frac{\epsilon^{\rm DFT}(|\mathbf q|)}{\epsilon^{\rm target}(|\mathbf q|)} .
$$

Both dielectric functions are Resta's Thomas–Fermi model (PRB 16, 2717 (1977)),
whose only input is the static dielectric constant $\epsilon(0)$:

$$
\epsilon(q) = \frac{q_{TF}^2 + q^2}{q_{TF}^2\,\sin(qR)/(\epsilon(0)\,qR) + q^2},
\qquad \frac{\sinh(q_{TF}R)}{q_{TF}R} = \epsilon(0),
$$

with $q_{TF} = (4k_F/\pi)^{1/2}$ the Thomas–Fermi wave vector of the valence
electrons. The paper neglects local-field effects (a few per cent).

- **Usage.** `screening_eps=` in `phonon_assisted_absorption_dense_q` and
  `carrier_scattering_dense_q`; `SCREENING_EPS` / `--screening-eps` in the
  absorption and mobility drivers, and a paoflow-gen prompt. The DFT constant
  defaults to the isotropic average of the dielectric tensor in the `.epb`
  files, which ph.x computes at Γ for insulators. `screening_eps_dft=` sets it
  when the coupling has none, otherwise the run stops.
- **Implementation** (`elphon/coupling_screening.py`). $|\mathbf q|$ is the
  shortest image $|\mathbf q + \mathbf G|$. The factor multiplies the
  mass-scaled phonon eigenvectors at each dense q, so it costs nothing. It
  depends on $|\mathbf q|$ only, so the irreducible-q reduction stays exact.
  The outputs record `coupling_screening` (`'none'` or `'ponce-resta'`) and the
  `screening_*` constants and factors.
- **Size for Si** (DFPT 12.97, measured 11.94, $q_{TF}$ = 1.100 bohr⁻¹, a =
  10.332 bohr). The factor on $|g|^2$:

  | q | Γ | L | X | Brillouin-zone average |
  |---|---|---|---|---|
  | $|g'|^2/|g|^2$ | 1.180 | 1.031 | 1.019 | 1.051 |

  Small-q (intravalley) scattering is enhanced most. Large-q intervalley
  phonons and the phonon-assisted absorption, which goes through phonons near
  X, change by a few per cent.
- **Effect on the Si absorption** (PBE coupling 12³/6³, dense 12³ k / 6³ q,
  scissor to 1.17 eV, 78 and 296 K). The phonon-assisted ε₂ grows by 3 % at
  the indirect edge and by 4–5 % at 2–3 eV; α below the direct edge by 3.9 %
  (+0.017 dex). The direct term is unchanged. This is a small part of the
  ×0.6–0.8 α deficit of the Si study.
- **Effect on the Si mobility** (6³-q coupling, H(R) from the EPW 12³ nscf,
  48³ k = q, σ = 10 meV, 300 K, MRTA). μ_e goes from 1292 to 1164 cm²/Vs
  (−9.9 %) and μ_h from 432 to 371 (−14.2 %). Poncé *et al.* find −8.8 % and
  −12.4 %. SERTA changes by the same amounts.
- **Which target.** The measured static dielectric constant (Si 11.94, as in
  the paper). It is a macroscopic constant with local fields, like the DFPT
  value. PAOFLOW's `dielectric_tensor` ε₁(0) is an independent-particle value
  on another footing: 17.0 for PBE Si against 12.97 from DFPT. It should not be
  the target, nor should the eACBN0 one (12.3); scaled by the PBE ratio that
  would give 9.4, far below experiment.

## Parallelisation and memory

- **MPI over the q loop** (both workflows), with `load_balancing` and
  `Allreduce(SUM)`. Cap the rank count at the number of q evaluated.
- **Node-shared vertex.** $g(R_e,R_p)$ is allocated once per node
  (`MPI.Win.Allocate_shared`). The node-local rank 0 builds it in place: each
  coarse $g_q(R_e)$ goes to its q-cell, then the q → $R_p$ FFT runs one orbital
  row at a time. There are no full-size temporaries.
- **EPW coupling.** `load_epw_coupling` reads the full `epmatq`
  ($n_{\rm bndep}^2\,N_k\,3n_{\rm at}\,N_q$ complex) on node rank 0
  only. It is released once $g(R_e,R_p)$ is built. The other ranks receive the
  q-points, force constants, Z* and ε by broadcast.
- **Node-shared dense electrons.** `precompute_dense_electrons(...,
  comm=comm)` keeps E and V in node-shared memory. Each rank of the node
  diagonalises its own slice of k, with results identical to the private
  arrays. `release_dense_electrons` frees them.
- **Vertex phases.** These are $e^{2\pi i\mathbf k\cdot\mathbf n}$. They
  are computed per k-block (`vertex_phases`) instead of stored for the whole
  grid, which saved $16\,N_k^3 n_{ws}$ bytes per rank.
- **k+q eigenvectors.** `lambda_q_dense_ws_fast` gathers V(k+q) through an
  index map instead of rolling a copy of V for each q.
- **Absorption kernels.** Both process the transitions in blocks of
  `pair_block`. The direct kernel used to build its delta functions for all
  the pairs of a k-block at once: with the Kramers–Kronig grid up to 10 eV
  this took 6–7 GB per rank.
- **Measured memory (Si emissivity run).** The run was 24³ k / 12³ q,
  7 temperatures, ω to 3 eV, Kramers–Kronig to 10 eV, on 24 ranks:

  | | before | after |
  |---|---|---|
  | node used-memory growth, 3³ DFPT q | 62 GB | 20 GB |
  | per-rank peak RSS, 3³ DFPT q (shared pages counted in every rank) | 7.7 GB | 2.5 GB |
  | node used-memory growth, 6³ DFPT q | 71 GB | 23.5 GB |
  | largest rank, 6³ DFPT q | 17.7 GB | 5.6 GB |

  The results are bitwise identical before and after the change, for the
  absorption, the carrier rates and α²F / α²F_tr. The remaining private
  memory is about 0.85 GB per rank. The shared vertex costs
  $16\,n_{\rm awf}^2\,3n_{\rm at}\,N_k^{\rm coarse}N_q^{\rm coarse}$
  bytes per node: 0.38 GB for Si on 6³ k / 3³ q, 3 GB on 6³ / 6³.
- **Thread/rank balance:** prefer more MPI ranks with 2–4 BLAS threads each.
- **Cost per dense q (phonon-assisted absorption).** Measured for Si with
  12³ k / 6³ q coarse grids, 48³ k, one core per rank and 5 scissor groups
  (Varshni):
  - **Fourier sum** $g(\mathbf k)=\sum_{\mathbf n}e^{2\pi i\mathbf k\cdot\mathbf n}g(\mathbf n)$
    in `band_vertex`: a BLAS `zgemm` over 2191 pair-resolved $R_e$ images,
    about 21 s.
  - **Rotation to bands:** about 8 s.
  - **Second-order kernel:** about 8.6 s.
  - **Vertex phases:** about 3.7 s.
  - **q-sum** $g(R_e,R_p)\to g_q(R_e)$: about 1.5 s.

  That is about 44 s per q in all. Two earlier bottlenecks are fixed:
  - **The q-sum (`g_Re_at_q`)** made one strided pass over the vertex for
    each of the 339 phonon images. It now sums the images of each grid cell
    first and makes one contiguous pass: 20× faster.
  - **The rotation** used chained `einsum`s; it is now batched `matmul`:
    3× faster.

  Both agree with the previous code to 10⁻¹⁵ relative. Two other options were
  measured and rejected:
  - **FFT:** an FFT over the dense k-grid would cut the flops of the Fourier
    sum by about 100×. With Accelerate's `zgemm` (about 100 GFLOP/s per core)
    the wall time is the same, and it needs all k at once (7 GB per q).
  - **Truncating $R_e$:** the PAO-gauge vertex still has 1–2% of its maximum
    at the boundary of the supercell, so a real-space cutoff is not safe.
  - **Pruning k-points without allowed transitions:** these are only about
    2%, because the free-carrier intraband transitions reach almost every k.

---

## Adaptive smearing of the deltas

Every energy-conserving delta of the dense-grid drivers can take a state-dependent
Gaussian width instead of a fixed one, following PAOFLOW's `adaptive_smearing` of
the DOS and of the dielectric function (Yates, Wang, Vanderbilt and Souza, Phys.
Rev. B **75**, 195121 (2007)): the width is the variation of the argument of the
delta across one cell of the grid it is summed over,

$$
\sigma = a\,|\nabla\varepsilon|\,\Delta k,\qquad
\Delta k = \Big(\frac{8\pi^3}{\Omega N}\Big)^{1/3},
$$

with $a \simeq 0.7$ for Gaussians and a floor that keeps the width finite at the
band extrema (EPW's `adapt_smearing` uses the standard deviation of a uniform
distribution over the cell and a 1 meV floor). Coarse grids then get broad,
smooth deltas and fine grids sharp ones without retuning the smearing, so the
grid is converged alone instead of jointly with the smearing. The helpers are in
`PAOFLOW.elphon.adaptive_smearing`; the drivers take `adaptive_smearing=a`
(`None`, the default, keeps the fixed width) and the fixed-width argument
becomes the floor.

| Quantity | delta | summed over | width |
|---|---|---|---|
| scattering rates, `carrier_scattering_dense_q(adaptive_smearing=, min_smearing_ev=)` | $\delta(\varepsilon_{m\mathbf k+\mathbf q}-\varepsilon_{n\mathbf k}\mp\omega_{\mathbf q\nu})$ | q | $a\,|\mathbf v_{m\mathbf k+\mathbf q}|\,\Delta q$ (the phonon group velocity, two orders of magnitude smaller, is neglected) |
| phonon-assisted absorption, `phonon_assisted_absorption_dense_q(adaptive_smearing=, min_degauss_ev=)` | $\delta(\varepsilon_{j\mathbf k+\mathbf q}-\varepsilon_{i\mathbf k}-\omega\mp\omega_{\mathbf q\nu})$ | k and q | $a\sqrt{|\mathbf v_{i\mathbf k}-\mathbf v_{j\mathbf k+\mathbf q}|^2\Delta k^2+|\mathbf v_{j\mathbf k+\mathbf q}|^2\Delta q^2}$; the direct term uses the k part alone |
| Eliashberg function, `eliashberg_dense_q(adaptive_smearing=)` via `precompute_dense_electrons` | $\delta(\varepsilon_{n\mathbf k}-E_F)\,\delta(\varepsilon_{m\mathbf k+\mathbf q}-E_F)$ | k and q | $a\,|\mathbf v_{n\mathbf k}|\,\Delta k$ per state, floored at `sigmas_ry`; the Fermi level, $N(E_F)$ and the Fermi-surface shell use the same widths (`electrons['widths']`) |

The group velocities are the diagonal of the PAO velocity operator averaged over
degenerate subspaces (`_diagonal_velocities`, the same velocities as the MRTA
and transport weights), computed once on the dense grid and shared between the
ranks. With a fixed width the numerics are unchanged: the kernels accept a
scalar or an array of widths, and the Rust absorption kernel takes one width
per pair. Not covered: the coarse-q path (`eliashberg_from_qe_coupling`,
`lambda_q_dense_ws`), the Kramers–Kronig direct term (`kk_degauss_ev`) and the
`fsthick_ev` window of the anisotropic Fermi-surface coupling.

Validation (`tests/unit/elphon/test_adaptive_smearing.py`): a constant array of
widths reproduces the scalar path bit for bit, per-state widths are additive,
and the adaptive $N(E_F)$ of a simple-cubic band on a $14^3$ grid is within 5 %
of a converged $90^3$ reference where a fixed 0.1 meV smearing is off by more
than 50 %. On real data (October 2026): the Pb isotropic λ with the native
DFPT coupling and q = 24³ is 1.068, 1.069 and 1.069 on k = 24³, 48³ and 72³
with a = 0.7 (1.054 and 1.100 for a = 0.35 and 0.2 at 48³), where the fixed
50 meV width gives 1.018, 1.176 and 1.029 (`examples/elphon_example/Pb/README.md`);
the 3C-SiC MRTA electron mobility converges from 48³ on with a = 0.7 (956,
993, 942 cm²/Vs at 48³–72³, experiment 900–1000) where the fixed 10 meV width
was still moving at 60³ (`examples/elphon_example/SiC/REPORT.md`). The SERTA
rate of the polar acoustic term keeps drifting with either smearing (it
diverges logarithmically without free-carrier screening). Mind that a
Fermi-surface width of $a|\mathbf v_F|\Delta k$ is large in a metal (up to
0.25 eV at 72³ in Pb), so the adaptive λ is an average over that energy
window; it is grid-stable, and converges to the sharp Fermi-surface limit as
$1/N$.

---

## Grid consistency rules

| Constraint | Reason |
|---|---|
| nscf k-grid = full Γ-centred grid (`nosym`/`noinv` for ph.x sources; explicit `K_POINTS crystal` list for EPW) | the vertex needs $A_k$ at every k and k+q |
| **coarse q-grid divides the coarse k-grid** in each direction | k+q must be a grid point. `vertex_from_epw` raises otherwise; for the ph.x sources `kq_index_map` silently rounds k+q to the nearest grid point, which takes $A_{k+q}$ at the wrong k (e.g. 189 of 216 q for a $9^3$ k / $6^3$ q setup) |
| `NK_DENSE % NQ_DENSE == 0` (dense-q workflow) | k+q on the dense grid is an integer index roll |
| PAOFLOW `pao_hamiltonian(expand_wedge=False)` on an explicit full list | the list is already the full BZ |

A DFPT q-grid that is too coarse (e.g. $3^3$ for Pb) gives an inflated
$\lambda$ under dense-q interpolation, from Fourier overshoot near $\Gamma$.

---

## The `paoflow-gen elphon` CLI workflow

`paoflow-gen` (see [Input and Script Generators (CLI)](Input-and-Script-Generators-CLI))
writes `main.elphon.py`, `me.elphon.py` and `plot.elphon.py`, plus `me_aniso.elphon.py` for the EPW
source with equal dense k- and q-grids. With the default EPW source it also
writes the QE/EPW inputs in the layout of `examples/elphon_epw_example`, deriving
the scf and nscf from a `pw.x` input of the system:

```
phonon/scf.in  phonon/ph.in                       # pw.x scf + ph.x (irreducible q, fildvscf)
epw/nscf.in    epw/epw.in    epw/write_ukk.py     # nscf on the explicit full k list + epw.x
```

```bash
cd phonon && pw.x -in scf.in && ph.x -in ph.in && python3 <q-e>/EPW/bin/pp.py
cd ../epw && mkdir -p <prefix>.save
cp ../phonon/<prefix>.save/{charge-density.dat,data-file-schema.xml} <prefix>.save/
pw.x -in nscf.in && python3 write_ukk.py && mpirun -np N epw.x -nk N -in epw.in
cd .. && mpirun -np N python main.elphon.py   # PAO interpolation of EPW's coupling -> alpha^2F, lambda, Tc
python me.elphon.py                            # Migdal-Eliashberg gap vs T, linearised-kernel Tc
python me_aniso.elphon.py                      # anisotropic gaps (dense q with nk = nq)
python plot.elphon.py                          # overlays EPW's epw/<prefix>.a2f (dashed) when present
```

The scf and the nscf run in separate directories, so the nscf does not
overwrite the save that ph.x used. Without a `pw.x` input the generator writes
`epw/nscf.kpoints` (the explicit k list) in place of `phonon/scf.in` and
`epw/nscf.in`.

The generator re-prompts until the q-grid divides the k-grid and
`NK_DENSE % NQ_DENSE == 0`. `main.elphon.py` is the same analysis as
`examples/elphon_epw_example/main.py` (options `--coarse-q`, `--nk`, `--nq`,
`--sigma-ev`). Choosing `ahc (Gamma only)` produces the
previous two-phase ph.x script (`inputs`, then `analyse`), with a warning about
its gauge limitation. The `elphmat` choice is disabled in the menu.

---

## Validation

### DFPT source (October 2026)

The native vertex was compared with EPW's `epmatq` on identical inputs, in the
band basis, every k and every q of the full grid:

| system | grids | pseudopotential | agreement |
|---|---|---|---|
| Si (EPW tutorial 06, PBE, 20 Ry) | 6³ k / 3³ q, 12 bands, 24³ FFT | ONCV FR used in a scalar run (`average_pp`) | $5\times10^{-15}$ at Γ and at the second star; $\le 10^{-7}$ elsewhere except $3$–$5\times10^{-4}$ at the umklapp k-points of the two largest stars |
| Pb (EPW tutorial 04, SR) | 6³ k / 6³ q, 16 bands (`exclude_bands = 1:5`), 36³ FFT | ONCV SR | $\le 3\times10^{-9}$ at $G_0 = 0$ points, $\le 2\times10^{-6}$ at umklapp points, all 216 q |
| Pb (fully relativistic, local run) | 4³ k / 2³ q, 30 bands (`exclude_bands = 1:10`), spin-orbit | `PSEUDOS/nc-fr-04` | $\le 4\times10^{-10}$ at $G_0 = 0$, $\le 4\times10^{-7}$ at umklapp points |

The umklapp residuals are on EPW's side: restricting the gathered Fourier
components to EPW's wave-function FFT stick map reduces them tenfold, and EPW's own
$q \leftrightarrow -q$ hermiticity is violated at the same level ($7\times10^{-5}$ on
the Si data) where the native vertex satisfies it to the DFPT convergence
($10^{-6}$).  Through `prepare_dense_vertex`, $g(R_e, R_p)$ agrees to
$8\times10^{-7}$ (Si) and $9\times10^{-9}$ (Pb FR, kept bands) with the EPW source, the
phonons and the dielectric data are identical, and the Si phonon-assisted absorption
at 12³/6³ agrees to $7\times10^{-6}$.  The irreducible-q unfolding reproduces the
direct vertex to $7\times10^{-8}$ on Si with a 20-band projector and fails only where a
band set cuts a degenerate multiplet (12 and 32 bands: up to 0.2 at a few k-points).

### EPW source (October 2026)

Pb, EPW tutorial 04 (ONCV pseudopotential, 6³ coarse k and q, 16 irreducible
ph.x q, `nbnd = 16`, `exclude_bands = 1:5`). EPW's own result on the same coarse
data serves as the reference.

| | PAOFLOW (`source='epw'`) | EPW |
|---|---|---|
| fine grids | 48³ k / 24³ q (413 irreducible q) | 48³ k / 24³ q |
| $\lambda$ | 1.2206 | 1.1515 (tutorial: 1.1584) |
| $\omega_{\log}$ | 4.50 meV | 4.44 meV (tutorial: 4.40) |
| $T_c$ McMillan / Allen–Dynes ($\mu^*=0.1$) | 4.77 / 5.21 K | 4.38 / 4.76 K |
| $N(E_F)$ per spin | 0.2514 /eV | 0.2348 /eV |

Electron smearing is 0.05 eV in both. The remaining 6% in $\lambda$ follows the
7% higher $N(E_F)$ of the 18-orbital PAO bands compared with EPW's 4-function
Wannier bands. At 24³ k / 24³ q, PAOFLOW gives $\lambda = 1.066$.

**Without Wannier functions.** A completely fresh run (new `ph.x` and nscf,
then `epw.x` with `wannierize = .false.`, the placeholder `.ukk` and
`exclude_bands = 1:5`) gives the same PAOFLOW results to all printed digits:
$\lambda = 1.0662$ (24³/24³) and $1.2206$, $\omega_{\log} = 4.500$ meV, $T_c$
4.77 / 5.21 K (48³/24³).

Component checks on the same data:
- **Gauge:** the ±q relation $d^{(-q)}(k+q)=d^{(q)}(k)^\dagger$ holds in the
  PAO gauge to $1\times10^{-6}$ (AHC: 1.41, i.e. random phases).
  $g_q(R_e)$ has 88% of its weight within nearest neighbours for every q
  (AHC, $q\neq0$: 1.7%, identical to random phases).
- **Grid-level exactness:** on the 6³ coarse grid (σ = 0.3 eV), PAOFLOW gives
  $\lambda = 1.337$ against $1.353$ from EPW's `epmatq` and the DFT
  eigenvalues directly. The residual is the projectability factor
  $\sqrt{p_mp_n}$.
- **Phonons:** `phonon_interp_from_epw` reproduces $\mathrm{eig}(C/M)$ at the
  coarse q to $5\times10^{-8}$ THz, and $C/M$ reproduces the ph.x frequencies.

### Migdal–Eliashberg (October 2026)

The PAOFLOW $\alpha^2F$ of Pb, rebuilt with `degaussq` = 0.15 meV (0.5 meV for
the linearised kernel), with $\mu^* = 0.1$, $\omega_c$ = 0.1 eV and the tutorial
temperatures, compared with EPW tutorial 04 on its own $\alpha^2F$:

| | PAOFLOW 48³ k / 24³ q | PAOFLOW 24³ k / 24³ q | EPW tutorial 04 |
|---|---|---|---|
| $\lambda$ of the rebuilt $\alpha^2F$ | 1.200 | 1.059 | 1.158 |
| Matsubara points / iterations at 0.3 K | 616 / 8 | 616 / 8 | 616 / 8 |
| $\Delta(i\omega_0)$, 0.3 K | 1.016 meV | 0.937 meV | 0.923 meV |
| $Z(i\omega_0)$, 0.3 K | 2.110 | 1.997 | 2.071 |
| gap edge, Padé / acon, 0.3 K | 1.037 / 1.037 meV | 0.952 / 0.952 meV | |
| $T_c$, linearised kernel | 5.69 K | 5.42 K | ≈ 5.25 K |
| $T_c$, $\Delta^2(T) \to 0$ | 5.71 K | 5.44 K | |

$(Z_0-1)/\lambda = 0.925$ in both PAOFLOW (48³) and EPW. The larger gap and
$T_c$ therefore follow the larger $\lambda$ and $\omega_{\log}$ of the PAO
$\alpha^2F$, not the solver. For each $\alpha^2F$, the two $T_c$ estimates
agree to 0.5%. The low-$T$ Padé and acon gap edges agree to $10^{-5}$ meV, and
near $T_c$ to about 3%. A full sweep (23 temperatures with both continuations,
plus 25 linearised-kernel temperatures) takes about 3 s.

### MgB₂: pair-resolved interpolation and anisotropic Migdal–Eliashberg (October 2026)

EPW source, 6³ k / 3³ q coarse grids, σ = 0.05 eV, μ* = 0.1, ω_c = 0.5 eV,
`fsthick` = 0.2 eV (`examples/elphon_example/MgB2`).

| | λ | ω_log | $T_c$ AD | $N_F$ | iso. $T_c$ (ME) | aniso. Δ(iω₀), 5 K | aniso. $T_c$ |
|---|---|---|---|---|---|---|---|
| 24³ k / 24³ q, single-site WS | 0.608 | 61.0 meV | 17.3 K | | 19.9 K | | |
| 24³ k / 24³ q | 0.579 | 60.1 meV | 14.7 K | 0.422 | 17.0 K | 1.9–4.9 meV | 24.0 K |
| 32³ k / 32³ q | 0.523 | 62.4 meV | 10.7 K | 0.330 | 12.9 K | 1.1–4.9 meV | 25.9 K |
| 40³ k / 20³ q | 0.578 | 62.3 meV | 15.1 K | 0.368 | 17.6 K | 1.4–7.7 meV | (sublattices) |
| EPW tutorial 04, 40³ k / 20³ q | 0.573 | 61.7 meV | 14.6 K | | | | |

- **Fermi-surface coupling.** Its averages reproduce the isotropic λ and $N_F$
  to $10^{-8}$ on every grid.
- **Solver cost.** It converges in 5–17 iterations per temperature. The 475
  states of the 40³ grid take 42 s for 17 temperatures plus 12 linearised-kernel
  points. The two $T_c$ estimates agree to 0.1–1.4 K.
- **$N_F$ convergence.** With σ = 0.05 eV, $N_F$ reaches 0.357 at 48³–64³ k, so
  the 24³ and 32³ λ (and gaps) are sampling-limited. Converged anisotropic
  values need at least 40³ k with equal q (about 2.5 h on 20 cores).
- **Sublattices at 40³ k / 20³ q.** The equations split into four
  symmetry-connected sublattice components, which lose their gaps at about
  30 K (75% of the Fermi surface) and 41 K (25%); see
  [Anisotropic Migdal–Eliashberg](#anisotropic-migdaleliashberg-fermi-surface-restricted).

### AHC source (historical, before October 2026)

The table previously shown here (Pb, `pb_s.UPF`, 9³ k, AHC; λ = 1.26–1.49 on
6³ q) was produced with **both** Fourier/projection errors listed under
[Theory](#theory). It also used gauge-inconsistent AHC vertices, and a 6³ q-grid
not commensurate with the 9³ k-grid. Its agreement with the literature should
not be taken as validation. On a rerun of that setup (Workflow 1, 216 q, dense
k 18³), λ = 1.46 with neither fix and λ = 3.1 with only the projection fix,
before the Fourier fix existed. The qualitative conclusion that a
3³ DFPT q-grid is too coarse for Pb (dense-q λ inflates instead of converging)
remains a convergence statement, not a code defect.

---

## Practical notes and pitfalls

- **Long-range (polar) terms.** Pass `long_range=LongRangeModel.from_coupling(coupling_dir,
  save_dir, quadrupoles=...)` (`PAOFLOW.elphon.long_range`) to the dense drivers: the
  analytic dipole (Fröhlich) and, with a `quadrupole.fmt` (EPW's format; computed by
  `PAOFLOW.elphon.quadrupoles` from twelve single-q `ph.x` density responses, see the Si
  example's `quadrupoles.py`), quadrupole vertex is subtracted
  from the coarse PAO-gauge vertex and added back at every dense q, and the dipole force
  constants likewise (LO-TO splitting). The phonons must be built with the same model
  (`phonon_at_q=None`, or `phonon_interp_from_dyn(..., long_range=model)`); the driver
  raises otherwise. The model follows EPW's `rgd_blk_epw`/`rgd_blk` term by term
  (`docs/internals/long-range-elphon-plan.md`); `q2r_matdyn` applies the same model to
  the phonon dispersion and DOS (QE's Ewald parameter, `matdyn.x` to 0.01 cm⁻¹). The
  worked case is `examples/elphon_example/SiC` (phonons, the Fröhlich 1/|q| vertex,
  carrier mobility). Without a model the interpolation
  Fourier-transforms the full coupling and force constants, which is exact for metals
  ($Z^* = \varepsilon^\infty = 0$, MgB₂, Pb) but not for insulators:
  `check_long_range_terms` (`do_pao_eph.py`) then
  - raises `NotImplementedError` for polar materials ($|Z^*| > 0.1$), unless
    `allow_missing_long_range=True`;
  - warns for non-polar insulators such as Si ($\varepsilon^\infty$ present),
    whose quadrupole term is missing (also with a dipole-only model).
- **Explicit k lists.** PAOFLOW reads nscf saves without a `monkhorst_pack`
  element by recovering the grid from the list (`uniform_grid_from_kpoints`).
  The list must be the complete grid in `K_POINTS automatic` order (third index
  fastest), as written by `kpoints_card` or wannier90's `kmesh.pl`.
- **Irreducible-wedge nscf** (`K_POINTS automatic`, Γ-centred, no `nosym`):
  `PAOFLOW(..., unfold_kpoints=True)` expands `kpnts`, `kpnts_wght`, `my_eigsmat`
  and `nkpnts` to the full grid before the broadcast and `read_QE_wfc` returns the
  rotated block of every grid point
  (`PAOFLOW.elphon.wavefunction_symmetry.WavefunctionProvider`:
  $c_{Rk}(RG)=e^{2\pi i(\kappa'+m')\cdot t}c_k(G)$, SU(2) of the proper part for
  spinors, $i\sigma_y K$ for time reversal); `read_nscf(save, unfold=True)` gives
  the matching k list, and `CoarseVertexSetup` recognises such a save by itself
  (`unfold_kpoints=False` raises).  Without the option an irreducible-wedge save
  still works for the Hamiltonian through `pao_hamiltonian(expand_wedge=True)`,
  but its projections cover the irreducible points only, which the vertex rejects.
- **`expand_wedge=False`** for the EPW save (see above).
- **Spinor bands (spin-orbit coupling).** Pass `spin_degeneracy=1` to the
  dense drivers (`eliashberg_dense_q`, `eliashberg_from_qe_coupling`,
  `carrier_scattering_dense_q`, `phonon_assisted_absorption_dense_q`; the
  examples set it from the case). It sets the Fermi level of `nelec` electrons
  on the dense grid (`precompute_dense_electrons`), the density of states and
  the plasma frequency. With the default of 2, `nelec` spinor electrons fill
  only `nelec/2` bands: for Pb the Fermi level then sits inside the 5d manifold
  and $\lambda$ vanishes (bug fixed on October 25 2026, when only the plasma
  frequency honoured the value).
- **Hamiltonian from another save.** `phonon_assisted_absorption_dense_q` and
  `carrier_scattering_dense_q` only check that `A` and `HRs` have the same
  orbital count (`check_orbital_dimensions`); the physical requirement (same
  PAO basis on the same crystal) is checked in the drivers with
  `check_same_pao_basis`. Build such Hamiltonians with
  `pao_hamiltonian_from_save`, which reproduces the drivers' call sequence
  bitwise.
- **EPW band window.** Bands excluded with `exclude_bands` are handled through
  `ibndkept`. Packing by an outer disentanglement window (`lwin` false for some
  bands) is not supported.
- **`el_ph_mat` is in the pattern basis**: rotate with `el_ph_mat_to_cartesian`
  using the dump's own `u` matrix.
- **Masses: one per atom.** `masses_amu` of `eliashberg_dense_q` /
  `eliashberg_from_qe_coupling` has one entry per atom of the cell: it sets
  the number of atoms of the EPW record and mass-weights the phonon
  eigenvectors. The generated drivers keep `MASSES_AMU` per species (EPW's
  `amass(i)`, `ATOMIC_SPECIES` order) and expand it with
  `atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names'])`. Before this
  (October 2026), cells with repeated species (e.g. MgB₂: Mg, B, B) failed in
  `read_epw_epb` with a record-size mismatch; the error now names the matching
  number of atoms.
- **Units.** `HRs` and eigenvalues in eV; smearings (`sigmas_ry`) in Ry;
  frequencies in THz; `epmatq` and `ahc_gkk` in Ry/bohr; force constants in
  Ry/bohr²; masses in QE Rydberg units (`AMU_RY`).
- **Environment.** An editable PAOFLOW tree with `mpi4py` on an MPI-3 library
  (OpenMPI/MPICH) for `MPI.Win.Allocate_shared`.

---

## API summary

```python
from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.elph_bloch import read_nscf
from PAOFLOW.elphon.do_pao_eph import eliashberg_from_qe_coupling
from PAOFLOW.elphon.do_pao_eph_dense_q import eliashberg_dense_q

pf = PAOFLOW.PAOFLOW(workpath=..., outputdir='output', savedir='nscf/pb.save')
pf.projections(configuration='standard', basispath=BASIS)
pf.projectability(pthr=0.95)
A = pf.data_controller.full_projections()[:, :, :, 0].copy()   # BEFORE pao_hamiltonian
pf.pao_hamiltonian(expand_wedge=False)                         # explicit full k list
HRs = pf.data_controller.data_arrays['HRs']
info = read_nscf('nscf/pb.save')
ng = (6, 6, 6)
# nscf on the irreducible wedge (K_POINTS automatic 6 6 6 0 0 0) instead of the explicit list:
#   pf = PAOFLOW(savedir='nscf/pb.save', unfold_kpoints=True); ...; info = read_nscf('nscf/pb.save', unfold=True)
# or in one call: pao_hamiltonian_from_save('nscf/pb.save', ..., unfold_kpoints=True)

# Workflow 1 (coarse q from EPW):
out = eliashberg_from_qe_coupling(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    'epw', None, ng, None, source='epw',
    masses_amu=[207.2], nk_dense=48, sigmas_ry=[0.05 / 13.6057], nelec=14, mu_star=0.10,
)

# Native vertex from the ph.x outputs (no EPW): coupling_dir = the ph.x run directory,
# save_dir = the nscf save of the projections; dense-q drivers take coupling_source='dfpt'.
out = eliashberg_dense_q(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    'phonon', (6, 6, 6), None, None, ng, None, source='dfpt', save_dir='nscf/pb.save',
    masses_amu=[207.2], nk_dense=48, nq_dense=24, sigmas_ry=[0.05 / 13.6057], nelec=14,
    sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'], species=info['atom_names'],
)

# Hamiltonian of another save (eACBN0 electrons, PBE coupling):
from PAOFLOW.elphon.pao_electrons import check_same_pao_basis, pao_hamiltonian_from_save
coupling = pao_hamiltonian_from_save('nscf/Si2.save', BASIS, 'standard', 0.95, expand_wedge=False)
electrons = pao_hamiltonian_from_save('acbn0/epw_nscf/Si2.save', BASIS, 'standard', 0.95)
check_same_pao_basis(coupling.data_controller, electrons.data_controller)   # raises on mismatch
# ... phonon_assisted_absorption_dense_q(coupling.projections, electrons.HRs, ...,
#         scissor_ev=varshni_shift(4.73e-4, 636.0))

# Workflow 2 (dense q; coarse q and phonons taken from the .epb files):
out = eliashberg_dense_q(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    'epw', ng, None, None, ng, None, nq_dense=24, source='epw',
    masses_amu=[207.2], nk_dense=48, sigmas_ry=[0.05 / 13.6057], nelec=14, mu_star=0.10,
    sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'], species=info['atom_names'],
)
```

`out` is a dict with `omega`, `a2F`, `lambda`, `lambda_qv`, `omega_qv_thz`,
`omega_log`, `Tc_mcmillan`, `Tc_allen_dynes`, ... (see `eliashberg_from_modes`).

The Migdal–Eliashberg post-processing works on the saved result (rank 0 or a
separate serial run):

```python
import numpy as np
from PAOFLOW.elphon.migdal_eliashberg import (
    a2f_from_npz, linearized_eigenvalues, migdal_eliashberg_iso, write_me_outputs,
)

omega, a2F = a2f_from_npz('output/eliashberg.npz', degaussq_ev=1.5e-4)
res = migdal_eliashberg_iso(omega, a2F, [0.3, 2.1, 4.0, 5.0, 5.5], mu_star=0.1, wscut=0.1)
lin = linearized_eigenvalues(*a2f_from_npz('output/eliashberg.npz', 5.0e-4),
                             np.linspace(0.25, 6.25, 25), mu_star=0.1)
write_me_outputs(res, 'output/me', 'pb', lin)   # EPW-format files + migdal_eliashberg.npz
print(res['gap0_imag'], res['gap_pade'], res['gap_acon'], lin['Tc_linear'])
```

Phonon-assisted absorption on the same inputs (the data controller is passed
for the position dipoles of the velocity operator):

```python
from PAOFLOW.elphon.elph_bloch import atom_masses, pao_orbital_positions
from PAOFLOW.elphon.phonon_assisted_absorption import (
    phonon_assisted_absorption_dense_q, write_absorption_outputs,
)

out = phonon_assisted_absorption_dense_q(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'], info['alat'], info['omega'],
    'epw', (3, 3, 3), (6, 6, 6),
    masses_amu=atom_masses([28.085], info['species'], info['atom_names']), nelec=8,
    nk_dense=12, nq_dense=6, temps_k=[300.0],
    position_dipoles=pf.data_controller,
    sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'], species=info['atom_names'],
    orbital_positions=pao_orbital_positions(pf.data_controller, info['at']),
)
write_absorption_outputs(out, 'output')
```

Carrier scattering rates and mobilities on the same inputs (semiconductors;
equal dense grids, which must be converged together with the smearing):

```python
from PAOFLOW.elphon.carrier_scattering import (
    carrier_scattering_dense_q, rta_transport, write_mobility_outputs,
)

out = carrier_scattering_dense_q(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'], info['alat'], info['omega'],
    'epw', (3, 3, 3), (6, 6, 6),
    masses_amu=atom_masses([28.085], info['species'], info['atom_names']), nelec=8,
    nk_dense=48, nq_dense=48, temps_k=[200.0, 300.0, 400.0], doping_cm3=0.0,
    window_ev=0.3, smearing_ev=0.01,
    screening_eps=11.94,  # optional: Poncé screening correction, measured eps of Si
    sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'], species=info['atom_names'],
    orbital_positions=pao_orbital_positions(pf.data_controller, info['at']),
)
mobility = rta_transport(out, 'mrta')          # mobility_e, mobility_h (cm^2/Vs), sigma, Seebeck
write_mobility_outputs(out, 'output/mobility')  # + scattering_rates.npz for pf.transport
```

The anisotropic equations need the Fermi-surface coupling of a dense-q run with
equal dense grids, and (for several atoms per cell) the orbital centres:

```python
from PAOFLOW.elphon.elph_bloch import pao_orbital_positions
from PAOFLOW.elphon.fermi_surface_coupling import read_fs_coupling, write_fs_coupling
from PAOFLOW.elphon.anisotropic_eliashberg import (
    linearized_eigenvalues_aniso, migdal_eliashberg_aniso, write_me_aniso_outputs,
)

positions = pao_orbital_positions(pf.data_controller, info['at'])  # after projections
out = eliashberg_dense_q(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    'epw', (6, 6, 6), None, None, (6, 6, 6), None, nq_dense=24, source='epw',
    masses_amu=[24.305, 10.811, 10.811], nk_dense=24, sigmas_ry=[0.05 / 13.6057],
    nelec=16, sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'],
    species=info['atom_names'], orbital_positions=positions,
    fs_coupling=True, fsthick_ev=0.2,
)
write_fs_coupling('output/fs_coupling.npz', out.pop('fs_coupling'))

coupling = read_fs_coupling('output/fs_coupling.npz')
res = migdal_eliashberg_aniso(coupling, np.linspace(5, 45, 9), mu_star=0.1, wscut=0.5)
lin = linearized_eigenvalues_aniso(coupling, np.linspace(5, 60, 12), mu_star=0.1, wscut=0.5)
write_me_aniso_outputs(res, coupling, 'output/me_aniso', 'mgb2', lin)
```

---

## References

- L. A. Agapito and M. Bernardi, *Ab initio electron-phonon interactions using
  atomic orbital wave functions*, Phys. Rev. B **97**, 235146 (2018).
- H. Lee *et al.*, *Electron–phonon physics from first principles using the EPW
  code*, npj Comput. Mater. **9**, 156 (2023); S. Poncé, E. R. Margine, C.
  Verdi and F. Giustino, Comput. Phys. Commun. **209**, 116 (2016).
- F. Giustino, *Electron-phonon interactions from first principles*, Rev. Mod.
  Phys. **89**, 015003 (2017).
- J.-M. Lihm and C.-H. Park, Phys. Rev. B **101**, 121102(R) (2020) (QE AHC).
- P. B. Allen and R. C. Dynes, Phys. Rev. B **12**, 905 (1975); W. L.
  McMillan, Phys. Rev. **167**, 331 (1968).
- F. Marsiglio, M. Schossmann and J. P. Carbotte, Phys. Rev. B **37**, 4965
  (1988) (analytic continuation); H. J. Vidberg and J. W. Serene, J. Low Temp.
  Phys. **29**, 179 (1977) (Padé); E. R. Margine and F. Giustino, Phys. Rev. B
  **87**, 024505 (2013) (EPW Migdal–Eliashberg).
- S. Poncé, E. R. Margine and F. Giustino, Phys. Rev. B **97**, 121201(R)
  (2018); S. Poncé, W. Li, S. Reichardt and F. Giustino, Rep. Prog. Phys.
  **83**, 036501 (2020) (ab initio mobilities, SERTA and BTE).
- R. Claes, G. Brunin, M. Giantomassi, G.-M. Rignanese and G. Hautier, Phys.
  Rev. B **106**, 094302 (2022) (momentum-relaxation time approximation).
- G. Brunin *et al.*, Phys. Rev. Lett. **125**, 136601 (2020); V. A. Jhalani
  *et al.*, Phys. Rev. Lett. **125**, 136602 (2020) (dynamical quadrupoles).
- See also [Phonon Module](Phonon-Module) and
  [Input and Script Generators (CLI)](Input-and-Script-Generators-CLI).
