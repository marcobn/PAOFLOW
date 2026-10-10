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
- [Parallelisation and memory](#parallelisation-and-memory)
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

---

## Module map

Paths relative to `src/PAOFLOW/`.

| File | Role |
|------|------|
| `elphon/qe_elph_io.py` | Readers: `read_epw_epb` (EPW coarse Bloch coupling, multi-pool, gfortran subrecords), `read_epw_ukk` (EPW band bookkeeping: kept bands, window), `read_qe_ahc_gkk`, `read_qe_el_ph_mat`, `el_ph_mat_to_cartesian`, `read_qe_dyn`, plus the older `elph.inp_lambda`/`lambda.in` readers of the property-only route. |
| `elphon/elph_bloch.py` | Core machinery: `read_nscf` (k-points, lattice, Fermi level, crystal symmetries, atom species), `atom_masses` (per-species masses → one per atom), `kq_index_map`, `vertex_pao_R`, `_ws_lattice`, `precompute_dense_electrons` + `lambda_q_dense_ws_fast` (dense-$k$ electron cache + Fermi-surface double delta, shared across q). |
| `elphon/do_pao_eph.py` | **Workflow 1** driver `eliashberg_from_qe_coupling`; per-source vertex builders `vertex_from_epw`, `vertex_from_qe_ahc`, `vertex_from_qe_elphmat`; `load_epw_coupling` (`.ukk` + `.epb` + q in crystal coordinates); `phonon_modes_from_force_constants`. |
| `elphon/do_pao_eph_dense_q.py` | **Workflow 2** driver `eliashberg_dense_q`; `build_g_ReRp` / `g_Re_at_q` (double real-space vertex); `phonon_interp_from_epw` and `phonon_interp_from_dyn` (dense-q phonons with acoustic sum rule); `irreducible_qmesh` / `_crystal_point_group`. |
| `elphon/eph_kq.py` | Property engine: `eliashberg_from_modes` ($\alpha^2F$, $\lambda$, $\omega_{\log}$, $T_c$), `mcmillan_allen_dynes_tc`, `phonon_moments`; shared by every route. |
| `elphon/migdal_eliashberg.py` | Isotropic Migdal–Eliashberg solver on $\alpha^2F$: `solve_imag_iso`, `pade_continuation`, `analytic_continuation_iso`, `gap_edge`, `quasiparticle_dos`, `linearized_max_eigenvalue`; drivers `migdal_eliashberg_iso` / `linearized_eigenvalues`; I/O `a2f_from_npz`, `a2f_from_epw`, `write_me_outputs`. |
| `elphon/fermi_surface_coupling.py` | `FermiSurfacePairCoupling`: irreducible Fermi-surface states and the folded pair coupling $\Lambda_{ab}(\omega)$, filled during the dense-q loop; `write_fs_coupling` / `read_fs_coupling`. |
| `elphon/anisotropic_eliashberg.py` | Anisotropic (FSR) Migdal–Eliashberg solver: `coupling_strength`, `matsubara_kernel`, `solve_imag_aniso`, `linearized_max_eigenvalue_aniso`, `pade_continuation_aniso`, `gap_distribution`; drivers `migdal_eliashberg_aniso` / `linearized_eigenvalues_aniso`; `write_me_aniso_outputs`. |
| `elphon/qe_matdyn.py` | WS interpolation of QE force-constant files, used by the property-only route. |
| `gen/epw_inputs.py` | EPW input helpers: `kpoints_card` / `uniform_kpoint_list` (explicit full nscf grid), `write_placeholder_ukk` (EPW without Wannierization), `epw_input`. |
| `inputs/read_QE_xml.py` | `uniform_grid_from_kpoints`: recovers the grid of an explicit `K_POINTS crystal` nscf list, which has no `monkhorst_pack` element. |
| `elphon/basis.py`, `displacements.py`, `do_elphon.py`, `io.py`, `symmetry.py`, `fold.py`, `gkq.py`, `do_gkq.py`, `dvscf_fd.py` | The finite-difference frozen-phonon route; see [Electron-Phonon Coupling](Electron-Phonon-Coupling). |

---

## Required inputs

### EPW source (recommended)

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

Both share `_phonon_interp_from_force_constants`: Fourier transform to phonon
cells, a simple acoustic sum rule, then WS interpolation and diagonalisation.
A `min_freq_thz` guard drops spurious soft modes.

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

i.e. minus PAOFLOW's `dHksp`. The optional non-local pseudopotential term
(`nonlocal_velocity=<DataController>`, norm-conserving only) uses
`build_nonlocal_velocity_kspace` at the dense k-points, with the calibrated sign
of `inject_into_dHksp`.

The validation checks are:
- **Exact model.** An exactly solvable two-site tight-binding model checks that
  the whole chain (`vertex_pao_R` → `band_vertex`, `band_velocities` →
  kernel) reproduces the exact gauge-invariant spectrum off the coarse grid.
- **Mutation check.** The test fails if the position term is dropped or
  flipped, or if the vertex is conjugated.
- **Direct term on Si.** The direct term matches `dielectric_tensor` on the
  same Hamiltonian, both with and without the non-local term.

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
| `fsthick` | `fsthick_ev` | 4 eV |
| `efermi_read`, `fermi_energy` | `fermi_energy_ev` (`None` = mid-gap) | mid-gap |
| `eps_acoustic` | `eps_acoustic_cm` | 0.1 cm⁻¹ |
| `n_r` (tutorial: constant 3.4) | `refractive_index` | 3.4 |
| `mp_mesh_k` | `sym_rots` (folds **q**; the polarisation average is exact) | |
| `eig_read`, `scissor` | not implemented: PAO (DFT) energies | |

Some limitations and conventions apply:
- **Irreducible q.** With `sym_rots`, the dense q-grid is folded to its
  irreducible wedge. Only the polarisation average is invariant, and it is
  returned in all three components. Pass `sym_rots=None` for the resolved
  diagonal.
- **Above the direct gap.** The amplitudes diverge at direct transitions, as
  in EPW. Read the indirect spectrum below the direct gap.
- **Not implemented.** Free-carrier and impurity absorption (`carrier`,
  `ii_g`) and QDPT (`loptabs`).

---

## Parallelisation and memory

- **MPI over the q loop** (both workflows), with `load_balancing` and
  `Allreduce(SUM)`. Cap the rank count at the number of q evaluated.
- **Node-shared vertex.** $g(R_e,R_p)$ is allocated once per node
  (`MPI.Win.Allocate_shared`), built by the node-local rank 0.
- **EPW coupling.** `load_epw_coupling` reads the full `epmatq`
  ($n_{\rm bndep}^2\,N_k\,3n_{\rm at}\,N_q$ complex) on every rank. In the
  dense-q driver it is released once $g(R_e,R_p)$ is built.
- **Thread/rank balance:** prefer more MPI ranks with 2–4 BLAS threads each.

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

- **Long-range (polar) terms are not treated yet.** The interpolation
  Fourier-transforms the full coupling and force constants. There is no dipole
  (Fröhlich) or quadrupole subtract-and-restore step (EPW `lpolar`). That is
  exact for metals, whose `.epb` files carry $Z^* = \varepsilon^\infty = 0$
  (MgB₂, Pb). For EPW couplings, `check_long_range_terms` (`do_pao_eph.py`):
  - raises `NotImplementedError` for polar materials ($|Z^*| > 0.1$), unless
    `allow_missing_long_range=True`;
  - warns for non-polar insulators such as Si ($\varepsilon^\infty$ present),
    whose quadrupole term is missing.

  The implementation plan is the wiki page *Long-range electron–phonon
  interpolation (plan)*.
- **Explicit k lists.** PAOFLOW reads nscf saves without a `monkhorst_pack`
  element by recovering the grid from the list (`uniform_grid_from_kpoints`).
  The list must be the complete grid in `K_POINTS automatic` order (third index
  fastest), as written by `kpoints_card` or wannier90's `kmesh.pl`.
- **`expand_wedge=False`** for the EPW save (see above).
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

pf = PAOFLOW.PAOFLOW(workpath=..., outputdir='output', savedir='epw/pb.save')
pf.projections(configuration='standard', basispath=BASIS)
pf.projectability(pthr=0.95)
A = pf.data_controller.full_projections()[:, :, :, 0].copy()   # BEFORE pao_hamiltonian
pf.pao_hamiltonian(expand_wedge=False)                         # explicit full k list
HRs = pf.data_controller.data_arrays['HRs']
info = read_nscf('epw/pb.save')
ng = (6, 6, 6)

# Workflow 1 (coarse q from EPW):
out = eliashberg_from_qe_coupling(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    'epw', None, ng, None, source='epw',
    masses_amu=[207.2], nk_dense=48, sigmas_ry=[0.05 / 13.6057], nelec=14, mu_star=0.10,
)

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
only for the optional non-local velocity term):

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
    nonlocal_velocity=pf.data_controller,
    sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'], species=info['atom_names'],
    orbital_positions=pao_orbital_positions(pf.data_controller, info['at']),
)
write_absorption_outputs(out, 'output')
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
- See also [Phonon Module](Phonon-Module) and
  [Input and Script Generators (CLI)](Input-and-Script-Generators-CLI).
