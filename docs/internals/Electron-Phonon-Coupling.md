# Electron–Phonon Coupling

This page documents the **PAO-interpolation route** of `PAOFLOW.elphon`: the
production path for computing electron–phonon (el‑ph) coupling and isotropic
Eliashberg superconducting properties ($\alpha^2F$, $\lambda$, $\omega_{\log}$,
$T_c$). The route reads the coarse-grid DFPT coupling computed by Quantum
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
| `elphon/elph_bloch.py` | Core machinery: `read_nscf` (k-points, lattice, Fermi level, crystal symmetries), `kq_index_map`, `vertex_pao_R`, `_ws_lattice`, `precompute_dense_electrons` + `lambda_q_dense_ws_fast` (dense-$k$ electron cache + Fermi-surface double delta, shared across q). |
| `elphon/do_pao_eph.py` | **Workflow 1** driver `eliashberg_from_qe_coupling`; per-source vertex builders `vertex_from_epw`, `vertex_from_qe_ahc`, `vertex_from_qe_elphmat`; `load_epw_coupling` (`.ukk` + `.epb` + q in crystal coordinates); `phonon_modes_from_force_constants`. |
| `elphon/do_pao_eph_dense_q.py` | **Workflow 2** driver `eliashberg_dense_q`; `build_g_ReRp` / `g_Re_at_q` (double real-space vertex); `phonon_interp_from_epw` and `phonon_interp_from_dyn` (dense-q phonons with acoustic sum rule); `irreducible_qmesh` / `_crystal_point_group`. |
| `elphon/eph_kq.py` | Property engine: `eliashberg_from_modes` ($\alpha^2F$, $\lambda$, $\omega_{\log}$, $T_c$), `mcmillan_allen_dynes_tc`, `phonon_moments`; shared by every route. |
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
writes `main.elphon.py` and `plot.elphon.py`. With the default EPW source it also
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
- See also [Phonon Module](Phonon-Module) and
  [Input and Script Generators (CLI)](Input-and-Script-Generators-CLI).
