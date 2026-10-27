# Replacing EPW's `epw1`/`epbwrite` stage with a native DFPT coupling source (design note)

Design note written Oct 24 2026 from a feasibility study (exploration of the PAOFLOW ingest path,
the EPW 6.0 / QE 7.5 Fortran sources and the Rust backend), **implemented the same day**
(phases 0–3 below).  The sections after this one are the original assessment and plan; this
block records what was built and what was learnt.

## Implementation status (Oct 24 2026)

- **Modules**: `elphon/dfpt_io.py`, `elphon/plane_waves.py`, `elphon/pseudopotential_terms.py`,
  `elphon/dfpt_vertex.py`, `elphon/coupling_cache.py`, `elphon_native/`; `prepare_dense_vertex`
  and the three dense drivers take `coupling_source='dfpt'` (`source='dfpt'` in the Eliashberg
  drivers), `save_dir`, `basis`, `unfold`, `basis_shells`, `cache_dir`; `qe_elph_io.write_epw_epb`
  / `write_epw_ukk` are the dev/parity writers; `elph_bloch.pao_vertex_to_cells` is the shared
  tail of `vertex_pao_R`.  The examples default to the DFPT source (`COUPLING_SOURCE`).
- **Parity with EPW** (band basis, every k and q): Si tutorial 6³/3³ at $5\times10^{-15}$ (Γ and
  the second star) to $10^{-7}$, except $3$–$5\times10^{-4}$ at the umklapp k-points of the two
  largest stars; Pb SR 6³/6³ at $\le 2\times10^{-6}$ (216 q); Pb FR (spin-orbit, local 4³/2³ run)
  at $\le 4\times10^{-7}$.  Dense $g(R_e,R_p)$: $8\times10^{-7}$ (Si), $9\times10^{-9}$ (Pb FR).
- **Findings**: (1) `wfc*.dat` stores k in bohr⁻¹ while its Miller indices refer to $2\pi/a$;
  (2) ph.x's `dvscf` holds the Hartree-xc response only (NLCC included), the bare local and KB
  terms are added as EPW does; (3) EPW rotates wavefunctions, PAOFLOW rotates the potential
  (QE `dfile_star`), which also makes spin-orbit trivial (no SU(2)); (4) fully relativistic UPFs in
  scalar runs need QE's `average_pp`; (5) the remaining umklapp residual is EPW's: it shrinks
  tenfold when the gathered components are restricted to EPW's FFT stick map, and EPW's own
  $q\leftrightarrow-q$ hermiticity is violated at that level while the native vertex is consistent
  to the DFPT convergence; (6) EPW uses the exact $k+q$ in the bra projectors, so nscf k-lists
  written with few digits limit the agreement to $10^{-8}$ (the vertex mirrors EPW there);
  (7) the PAO Bloch sums use the cell-phase convention, so the star rotation in the PAO basis
  carries the atom lattice shifts $L_\kappa$ on the orbital indices and no representative phases;
  (8) that unfolding is exact iff the nscf band set does not cut a degenerate multiplet at any k
  (Si: exact to $7\times10^{-8}$ with 20 bands, 0.2 at a few k with 12 or 32 bands), hence the
  mandatory self-check.
- **Rust**: shared `pool.rs` tied to the MPI core budget (`configure_native_threads`, fixes the
  oversubscription), `num-complex` bridging, `elphon_indirect_absorption` (36× faster than the
  NumPy kernel on 24 threads, parity $6\times10^{-16}$).  The fused phases+rotation and
  scatter-add kernels were not needed: the vertex stage is dominated by FFTs and GEMMs.
- **Costs** (Mac Studio, one rank): Si 6³/3³ vertex 3.4 s per q (band basis); Pb SR 6³/6³ 8 s
  per q; irreducible-q unfolding 4.5× faster at |star| ≤ 8.
- **Irreducible-wedge nscf (Oct 25 2026)**: `elphon/wavefunction_symmetry.py` rotates the stored
  Bloch functions of a `K_POINTS automatic` nscf to the full Γ-centred grid (coefficient relabelling
  and phase, SU(2) for spinors, $i\sigma_y K$ for time reversal); `PAOFLOW(unfold_kpoints=True)`,
  `pao_hamiltonian_from_save(unfold_kpoints=True)`, `read_nscf(unfold=True)` and the auto-detecting
  `CoarseVertexSetup` make the rest of the pipeline see a full-grid nscf.  Validation in §"Phase 4
  plans" B: Si 6³ from 16 k, Pb FR 4³ from 8 k with spin-orbit.
- **Not done**: USPP/PAW, LSDA, the dvscf real-space interpolation and the polar
  long-range term (phase 4 follow-ons); the integration asset is built locally
  (`.github/assets_generation/elphon/`) and must be uploaded to a release before the CI job runs.

## Context

PAOFLOW's electron–phonon pipeline (Migdal–Eliashberg, transport, phonon-assisted optics, mobility,
emissivity) takes its coarse-grid coupling from EPW: `epw.x` run with `epbwrite` and no
Wannierisation, then `read_epw_epb` / `read_epw_ukk` (`src/PAOFLOW/elphon/qe_elph_io.py:710,620`).
That is the only EPW dependence left. It costs: an `epw.x` build per QE release and the `.epb`
record layout changing between EPW versions; the `.ukk`/`lwin`/`exclude_bands`/`nimage`
bookkeeping; 38 GB of `.epb` files and a 16–24 GB private array on node rank 0 for Pb 12³/6³
(the OOM class seen on fedora); a serial 40-minute ingest (`prepare_dense_vertex`,
`do_pao_eph_dense_q.py:831`); and hours of `epw.x` wall time per campaign (the FR Pb run is in
progress as this is written). The question was whether porting that stage into PAOFLOW, with an
optional Rust backend, brings gains beyond independence from EPW releases.

## Assessment

### What EPW computes on this path (QE 7.5, `EPW/src/ep_coarse_unfolding.f90`, `ep_coarse.f90`, `dvqpsi.f90`)
g_mn,κα(k,q) = ⟨ψ_m,k+q| ∂V/∂u_κα(q) |ψ_n,k⟩, Ry/bohr, Cartesian displacement index c = 3κ+α,
no 1/√(2Mω). Three terms, all norm-conserving:
1. `dvscf` from ph.x (`fildvscf`): **induced Hartree+xc response only** (NLCC included), lattice-
   periodic part on the dense FFT grid, one record per displacement pattern; EPW contracts with
   conj(u) to the Cartesian basis. The bare ionic term is *not* in it.
2. Bare local term `compute_dvloc`: dvloc(G) = −i·tpiba·(q+G)·u_κ · V^loc_κ(|q+G|) · e^{−i(q+G)·τ_κ}
   (table `vloc_of_g` + analytic Coulomb tail), inverse FFT, multiplied with ψ in real space.
3. Kleinman–Bylander derivative terms (`dvqpsi_us_only3`, D = dvan): projectors at k and k+q,
   ps1/ps2 from becp = ⟨β|ψ⟩ and alphap = ⟨β|i(k+G)|ψ⟩; no radial derivatives.
Star unfolding: EPW does not rotate the potential, it rotates the wavefunctions per star member
(gmapsym, eigv, SU(2) for spinors) and recomputes dvloc per k and re-reads wfc per (q,k).
QE itself has the simpler identity (`PHonon/PH/dfile_star.f90:243-330`): rotate the scalar field
`dvscf` on the FFT grid to Sq (grid permutation + atom/direction relabel + two phases), then use the
true ψ_k, ψ_k+Sq. The port uses the QE identity (no wavefunction rotation, no SU(2)).
Written to `.epb`: nqc, xqc, et, dynq (EPW's own ASR, upper-triangle quirk `io/io.f90:2541`),
epmatq, zstar, epsi. Long-range polar term is not subtracted at this stage.
Minimal path size: ≈3.2k Fortran lines EPW + ≈1.4k QE library; of which the generic plane-wave
algebra (FFT, KB tables, vloc tables, symmetry) PAOFLOW largely has already.
Measured `elphon_wrap` wall (EPW): Pb 6³/6³ 11 bands 94.5 s on 20 pools; Si 6³/3³ 21 s; MgB₂
6³/3³ 243 s; Si 12³/6³ 21 bands 1006 s on 32 pools (≈9 core-hours, epmatq 15.8 GB).

### What PAOFLOW already has (reuse points)
- Wavefunctions: `projection/do_atwfc_proj.py:723 read_QE_wfc` (Miller, evc, npol; applies a Löwdin
  step harmless for NC), G↔r helpers `:612,:647` (Python loop over G, to vectorise), `:91
  radialfft_simpson` (QE `tab_β` form), `:833 calc_ylmg`, `:956 calc_ylmg_so`, `:1634 clebsch_jm_matrix`.
- UPF: `inputs/read_upf.py` (vloc Hartree, β with l and j, dion Hartree, rho_atc; no Q_ij/PAW).
- Symmetry: `inputs/read_QE_xml.py:212-246` (rotations, ft, irt, TR), `elphon/do_pao_eph_dense_q.py:472
  irreducible_mesh`, `elphon/symmetry.py:127 build_dv_symmetry_operators`, `:214
  rotate_dV_under_symmetry` (Wigner-D + atom permutation for a PAO operator), `hamiltonian/pao_sym.py`.
- Phonons natively: `phonon/qe_force_constants.py` (dyn reader, q2r/matdyn equivalents),
  `do_pao_eph_dense_q.py:224 phonon_interp_from_dyn`.
- The drop-in contract: `elph_bloch.py:225 vertex_pao_R` takes any Cartesian band-basis `d(k)`
  from the same nscf as `A` and returns g_q(R_e) = N⁻¹Σ_k e^{−2πik·R} (A_k+q^T d A_k^*);
  `prepare_dense_vertex` writes that into the shared window per q. Everything downstream is untouched.
- Validation data on disk: Si tutorial (`/Users/marco/Workspace/PAOFLOW/validation/EPW_tutorial06_audit/qe/{save,si.save,si.phsave}`,
  23 MB; reference `epw_pbe/si.epb*`), Pb SR (`/Users/marco/Workspace/PAOFLOW/validation/Pb_EPW_tutorial04/phonon/_ph0`
  per-q dvscf + patterns, `pb.dyn0..16`, `epw/pb.epb1..20`, `exclude_bands 1:5`), Pb FR `.epb`
  forthcoming on fedora (`/home/marco/Workspace/Pb_elphon/fr`).

### Gains (honest)
| Item | Today (EPW) | Port | Note |
|---|---|---|---|
| epmatq array | 16·nbndep²·Nk·3nat·Nq B on node rank 0 (15.8 GB Si 12³/6³) | never formed | projected-PAO Bloch functions φ̃_j,k = Σ_n A*_nj ψ_nk give g^PAO = ⟨φ̃_i,k+q\|∂V\|φ̃_j,k⟩ exactly (linearity) |
| `.epb` on disk | 38 GB (Pb/Si 12³/6³) | none; inputs are dvscf (MB) + wfc (10–100 MB) | |
| Ingest | serial node rank 0, ~40 min | q loop over node ranks, minutes | q cells of the shared window are disjoint |
| Vertex-stage core-hours | ≈9 (Si 12³/6³) | phase 1 (full q grid, numpy): comparable; phase 2 (irreducible q + PAO-basis unfolding): ÷\|star\| = ÷13.5 at 6³ plus per-k savings → 10–30× fewer | FFT count per (q,k): (1+3nat)·min(nawf,nbnd) vs EPW 3nat·(1+2·nbndep); Pb FR nawf 18 vs 32 bands gains, Si 'standard' 26 vs 12 uses the band basis (`basis='auto'`) |
| Bookkeeping | `.ukk`, lwin refusal, exclude_bands, pp.py, nimage bug, `.epb` layout per EPW version | none; QE-PHonon formats (stable since 6.x) | |
| Gauge | A and g must come from the same nscf (documented pitfall) | same `wfc*.dat` read by construction | |
| Validation | parity with EPW only | translational sum rule Σ_κ g_mn,κα(k,0) = (E_m−E_n)⟨ψ_m\|∂_α\|ψ_n⟩, hermiticity g_mn(k,q)* = g_nm(k+q,−q), star self-check | exact for NC local+nonlocal+SCF |
| Follow-ons | — | dvscf real-space interpolation (decouple q grid from DFPT), polar long-range term in one place | separate plans |

Does not change: DFPT cost (ph.x stays, `fildvscf` required); dense-grid cost (already BLAS-bound,
`docs/internals/Electron-Phonon-Coupling.md:1083-1150`); polar handling; phonon interpolator.

### Rust (second-order)
- Crate state (`rust/paoflow_rs`): pyo3 0.23 + numpy + rayon, **no complex support, no FFT, no BLAS**;
  thread pools duplicated 3× and **not tied to `core_budget()`** → every MPI rank spawns a pool over all
  cores (live oversubscription bug under `mpirun`). Fixing that is worthwhile regardless of the port.
- Vertex stage: FFT + GEMM bound; Rust (rustfft, fused multiply+FFT+gather, rayon over bands×modes)
  ≤ 2× over a vectorised numpy version. Downstream: `indirect_absorption_kernel` (dense masked deltas,
  8.6 s of 44 s per dense q) and fused phases+rotation (≈12 s) are 20–30 % → ≈1.3× overall;
  `fs_coupling` scatter-adds. zgemm, q-sum, phonons, ME solvers: no gain.

### Limits
Norm-conserving only (USPP/PAW need int3/augmentation terms; `read_QE_wfc`'s Löwdin step also breaks
S-orthonormality); non-magnetic (`nspin_mag = 1`), no LSDA, no gamma_only, no 2D cutoff; Γ-centred nscf
required, either the full grid or its irreducible wedge (`unfold_kpoints`, phase 4B); FFT grid must
be invariant under the symmetries used (QE guarantees it for the symmetries it keeps; assert);
bit-for-bit parity impossible (target ≤1e-10 relative on epmatq with QE-mirrored tables); phonon ASR
differs from EPW's quirk at the ASR level (native dyn reader, documented).

## Architecture (all new modules under `src/PAOFLOW/elphon/`, no `do_*` names)

| Module | Role | Reuse |
|---|---|---|
| `dfpt_io.py` | `locate_dfpt_files` (pp.py `save/` layout and raw `_ph0/` layout), `read_patterns` (u(3nat,3nat), npert, nirr, minus_q), `read_dvscf` (memmap, Fortran order, size check vs nr1x=nr1), `read_irreducible_q`, `SpaceGroup`/`read_space_group`, `read_nscf_plane_wave_setup` (ecut, fft grids, npol, pseudo files), `read_long_range_tensors` | `qe_force_constants.py:58,168`; XML pattern of `read_QE_xml.py:212-246` |
| `plane_waves.py` | `PlaneWaveBlock` (raw coefficients, no Löwdin), `periodic_part_on_grid` (vectorised scatter + ifftn·N), `project_on_plane_waves` (fftn/N gathered at miller−G0), `k_plus_G_cart` | factored from `do_atwfc_proj.py:723-831`, replaces loops at `:612-692` |
| `pseudopotential_terms.py` | `LocalPotentialTable` (QE `vloc_mod.f90:100-213,287-298`: erf split, dq 0.01, analytic tail), `bare_local_perturbation` (3nat, r), `KleinmanBylanderProjectors` (`beta_mod.f90`: tab_β, 4-point Lagrange, D = 2·dion Ry, block-diagonal in (l,m)), `nonlocal_perturbation` (ps1/ps2 terms); SOC `dvan_so` via fcoef in phase 3 | `read_upf.py`, `_two_center.py:56,169`, `do_atwfc_proj.py:91,833,956,1634`; out-of-repo prototype `EPW_opt_audit/compare/pw_velocity_check.py:45-120` |
| `dfpt_vertex.py` | `star_members` (full grid from irreducible q; raises on symmetry mismatch), `cartesian_perturbation_potentials` (conj(u) contraction, `dfile_star` rotation, umklapp phase, TR conjugation), `matrix_elements_at_k` (band or projected-PAO basis), `coarse_coupling_from_dfpt(save_dir, dfpt_dir, projections, kpts_cryst, ng, qgrid, *, basis='auto', tables='qe', comm, on_vertex, band_basis_out)` | `elph_bloch.py:185 kq_index_map`, `:225 vertex_pao_R` tail |
| `coupling_cache.py` (phase 2) | opt-in `cache_dir` cache of g(R_e,R_p): key = sha1(DFPT file names+sizes+patterns sha1, qgrid, ng, rounded masses, A.shape, `cache_tag` = PAO basis signature + pthr + nbnd); `.npy` + provenance json | promotes `Pb_elphon_staging/paoflow/vertex_cache.py` |
| `symmetry.py` (extend, phase 2) | `rotate_vertex_under_symmetry`: bra phase at k+q, ket phase at k, Wigner-D + atom permutation on orbitals, Cartesian rotation + irt + e^{−2πi Sq·rtau} on the mode index, k→Sk scatter, U_inv for improper ops, TR as conjugation + k→−k | `symmetry.py:127,214`, `pao_sym.get_wigner_so:574` (phase 3) |
| `qe_elph_io.py` (extend) | `write_epw_epb`, `write_epw_ukk` (dev/parity tool) | promotes `tests/unit/elphon/synthetic_epw.py:6-43` |
| `src/PAOFLOW/elphon_native/` (phase 2) | Rust bridge, mirrors `epsilon_native/__init__.py` | |

Driver hook: `prepare_dense_vertex` gains `source='dfpt'` + keywords `save_dir, pseudo_dir, basis='auto',
tables='qe', cache_dir=None, cache_tag=None`; the `'epw'/'ahc'/'elphmat'` branches are untouched (numerics
unchanged). New branch: all node ranks read the small data; shared window as today; **round-robin q
slice per node rank**, each rank writes its q cells (`on_vertex`); barrier; q→R_p FFT split over orbital
rows across node ranks (independent rows, identical result to the serial loop); return dict unchanged.
`eliashberg_dense_q`, `phonon_assisted_absorption_dense_q` (`phonon_assisted_absorption.py:1067` hard-codes
`'epw'`), `carrier_scattering_dense_q` (`carrier_scattering.py:444`) get `coupling_source` + the keywords;
`eliashberg_from_qe_coupling` (`do_pao_eph.py:379`) gets a per-q `vertex_from_dfpt` generator. Examples
(`examples/elphon/{Si,Pb,MgB2,SiC}`, `coupling_source='dfpt'` in their `system.py`) use the native
source, with EPW kept as the documented alternative. No `PAOFLOW.py` method
(the e-ph workflows are scripts; `PAOFLOW.electron_phonon` is the finite-difference route).

## Minimal algorithm (conventions)
Units Ry, bohr; k,q,G Cartesian in 2π/alat; τ in alat; FFT: invfft no scale, fwfft 1/N; mode c = 3κ+α.
```
setup: nscf (full Γ-centred grid, file order = order of A), space group (rotations, ft, irt), DFPT files,
       irreducible q from dyn headers, UPF per species (assert NC, nspin_mag = 1), tables vloc/β with
       q_max = sqrt(ecut)·tpiba + margin, dq = 0.01; members = star_members(q_irr) (full grid);
       assert qgrid | ng and FFT grid invariant under every S used.
for q in members (distributed):                                # phase 2: only irreducible q, then rotate g
  u, dv_pat = read_patterns, read_dvscf(iq_irr)                # (3nat, r) pattern basis at q0
  dv = einsum('cv,vxyz->cxyz', conj(u), dv_pat)                 # Cartesian at q0 (ep_coarse.f90:650-660)
  if S ≠ 1 or G ≠ 0 or TR:  dfile_star rotation: strip e^{+2πi q0·τ_κ}, permute grid r → S⁻¹(r − f),
       κ → irt(S,κ), direction in crystal axes, re-add e^{−2πi Sq0·τ_κ'}; umklapp: dv *= e^{−2πi G·r};
       TR: dv = conj(dv)
  dv += bare_local_perturbation(q)                              # NLCC already inside dvscf (addnlcc = F)
  ikq, G0 = kq_index_map(kpts, q, ng)                           # k+q = k' + G0
  for k:  C_k = A_k^* ᵀ c_k (nawf) or c_k (nbnd); C_kq likewise
          u_r = periodic_part_on_grid(C_k); w = dv[:,None]·u_r; dvpsi = project_on_plane_waves(w, miller_kq, G0)
          dvpsi += nonlocal_perturbation(β at k'+G'' (= k+q+G'), becp/alphap at k+G, D)
          M[k] = einsum('mG,cnG->cmn', conj(C_kq), dvpsi)       # (3nat, nb_kq, nb_k)
  gk = M (PAO basis) or A_kq^T M A_k^* (band basis); gR = fftn(scatter(gk, kidx))/N_k   # = vertex_pao_R tail
  on_vertex(iq, q, gR)                                          # shared window cell, or .epb writer with M
```
Nonlocal term: becp_jn = Σ_G β_j*(k+G) c_n(G); alphap_jnα = Σ_G β_j* i·tpiba(k+G)_α c_n; ps1_inα = Σ_j D_ij alphap_jnα;
ps2_inα = −i·tpiba Σ_j D_ij becp_jn; dvpsi_n,3κ+α(G') += Σ_{i∈κ} β_i(k+q+G')[ps1_inα + (k+q+G')_α ps2_inα].

## Phases (when picked up)

**Phase 0 — preparation, no numerical change.** Factor `read_qe_wfc_raw` out of `read_QE_wfc`
(Löwdin step unchanged); vectorise `fft_wfc_G2R`/`fft_allwfc_G2R` (bitwise test); `read_nscf` gains
fft_smooth/ecut/npol; `dfpt_io.read_space_group`; Rust: single `pool.rs` honouring `PAOFLOW_RS_THREADS`,
`parallel_resources.configure_native_threads()` exporting it from `core_budget()` under mpirun,
`num-complex` dependency. Tests: `tests/unit/projection/test_wfc_fft.py`, `tests/unit/elphon/test_dfpt_io.py`,
existing Rust parity suites, `thread_count()` honours the budget.

**Phase 1 — baseline port (NC, scalar, full q grid, `.epb` parity).** Modules above; `.epb` writer;
`prepare_dense_vertex('dfpt')`; `coupling_source` in the three drivers; examples + READMEs.
Unit tests (synthetic, fast): patterns/dvscf round trips in both layouts; `star_members` on a 2³/3³
cubic grid incl. a TR-only star; G0 relabel identity; `project(periodic(c)) == c`; `LocalPotentialTable`
vs direct Simpson 1e-6 and Lagrange at nodes; V_NL hermitian; Σ_κ nonlocal = −[∂_α,V_NL] vs central
differences; **toy plane-wave Hamiltonian** (one atom, Gaussian vloc, one l=0 β, 8³ box, dvscf := 0):
translational sum rule at 1e-10, hermiticity, conj(u) unitarity, grid rotation of a symmetric model
potential vs direct evaluation at Sq, band path == projected-PAO path, `.epb` writer round trip.
Local validation: Si 6³/3³ vs `epw_pbe/si.epb*` (≤1e-10 relative on epmatq; dynq 1e-6, ASR), Pb SR 6³/6³
vs `EPW_Pb/epw/pb.epb*` via ibndkept (exclude_bands), `prepare_dense_vertex('dfpt')` vs `('epw')` on
g(R_e,R_p) ≤1e-8, one `eliashberg_dense_q` λ/ω_log comparison; the two candidate sign conventions of the
grid rotation (S vs S⁻¹, irt direction) behind a private flag until parity selects one.
Integration asset: `tests/integration/elphon/` (conftest with `PAOFLOW_ELPHON_ASSET_*`, `cache_key='elphon'`,
reusing `tests/integration/qe/assets.py`), generator `.github/assets_generation/elphon/` (pw.x scf + ph.x
3³ `fildvscf` + nscf 6³ 12 bands; one epw.x run to extract `reference/epmatq_subset.npz` ≈1 MB, 2 k × all q);
tarball ≈25 MB; `ci.yaml` job mirroring the transport one.
Docs: `docs/internals/Electron-Phonon-Coupling.md` (DFPT source subsection, required inputs with EPW as
alternative, validation numbers, q-parallel fill), example READMEs, `tests/integration/elphon/README.md`,
`CLAUDE.md` folder tree, `docs/internals/testing.md`.

**Phase 2 — performance.** `basis='auto'`; irreducible-q-only mode `unfold='pao'` with
`rotate_vertex_under_symmetry` and a mandatory self-check (one non-identity star member computed
directly, max|Δ|/max|g| < 1e-8, raise otherwise; remedy: more nscf bands when nbnd cuts a degenerate
multiplet); `coupling_cache.py`; Rust in priority (a) `indirect_absorption_kernel` sparse-δ fused loop
(reference loops at `tests/unit/elphon/test_phonon_assisted_absorption.py:238-345`), (b) fused
phases+rotation in `band_vertex`, (d) scatter-adds; (c) `apply_dv` only if profiling shows >30 %
non-FFT/BLAS time in `matrix_elements_at_k`. Parity tests at 1e-12 with the monkeypatch pattern of
`tests/unit/acbn0/test_native_parity.py`; `paoflow-rs.yml` paths gain `src/PAOFLOW/*_native/**`.

**Phase 3 — spin-orbit (npol = 2, non-magnetic).** Spinor blocks; `dvan_so` via fcoef
(`upflib/init_us_1.f90`), j-dependent tab_β, spin-angular functions (`calc_ylmg_so`, `clebsch_jm_matrix`);
scalar dvscf applied to both components; `get_wigner_so` in the vertex rotation. Validate against the
fedora FR Pb `.epb` (copy save/phsave/wfc/.epb locally), spinor sum rule and hermiticity.

**Phase 4 — follow-ons (separate plans).** dvscf real-space interpolation w(r,R_p) (QE `dvscf_q2r.f90`
incl. its long-range dipole subtraction) to decouple the q grid from DFPT; polar long-range term at the
potential level; `paoflow-gen elphon` emitting ph.x-only workflows; retire the `.ukk` helpers.

## Decisions taken (defaults)
`.epb` writer kept as a dev/parity tool; Γ-centred nscf, full grid or irreducible wedge (`unfold_kpoints`); `tables='qe'`
default (`'exact'` available); native ASR on phonons, EPW's quirk not replicated (documented);
FFT-grid symmetry asserted, no fallback; degenerate-multiplet self-check raises; `cache_tag` is the
caller's responsibility (basis signature + pthr + nbnd), never a hash of A; USPP/PAW, LSDA/magnetic,
gamma_only, 2D cutoff, Hubbard-in-dvscf out of scope.

## Verification (for the implementation, when scheduled)
1. `pytest -q tests/unit/elphon tests/unit/projection` (synthetic identities, sum rule, round trips).
2. Local parity scripts on the Si and Pb SR data: epmatq ≤1e-10 relative, g(R_e,R_p) ≤1e-8, λ/ω_log agree.
3. `pytest -m integration tests/integration/elphon` with the new asset; CI job green.
4. Existing `source='epw'` runs bit-identical before/after (examples' Reference data, `tests/unit/elphon/test_epw_io.py`).
5. Phase 2: full-grid vs unfolded vertex ≤1e-8; Rust parity 1e-12; `thread_count()` ≤ `core_budget()` under `mpirun`.
6. Phase 3: FR Pb epmatq parity vs fedora `.epb`; SR vs FR λ sanity.

## Phase 4 plans (Oct 25 2026): ultrasoft/PAW, irreducible-wedge nscf, LSDA

Scoped from the QE 7.5 / EPW 6.0 sources after the norm-conserving port; nothing below is implemented.
Line references are to the QE tree unless a PAOFLOW path is given.

### A. Ultrasoft and PAW pseudopotentials

**What changes in the matrix element.** EPW and ph.x still contract plain wavefunctions,
g = ⟨ψ_{m,k+q}|dvpsi_n⟩ with no S (`EPW/src/ep_coarse.f90:606-608`); every US effect enters
`dvpsi` through the β projectors, and the local part (dvscf + dV_loc on ψ) is unchanged:

1. **D_eff = deeq − ε_nk·qq** replaces D in the moving-projector terms (`PW/src/compute_deff.f90`,
   used at `EPW/src/dvqpsi.f90:289-324`). `deeq_ij = dvan_ij + Ω Re Σ_G conj(Q_ij(G)) e^{iG·τ} V_eff(G)`
   (`PW/src/newd_acc.f90:126-301, 357-363`) couples different (l,m) of one atom; `qq = Ω Q_ij(G=0)`.
   ε_nk is the **absolute** KS eigenvalue of the ket (PAOFLOW's `eigs_ry` is E_F-referenced).  PAW adds
   `ddd_paw` from the one-centre densities (`add_paw_to_deeq`, becsum in `prefix.save/paw.txt`).
2. **int1** = ∫V_eff ∂Q^b_ij/∂τ_bα (q-independent; `dvqpsi.f90:650-660`), enters ps1 of the moving atom.
3. **int2** = ∫[∂V^loc_b/∂τ_bα]_q Q^a_ij (`dvqpsi.f90:617-645`), enters ps1 of **every** augmentation atom a.
4. **int3** = ∫dV^Hxc_q Q^a_ij from the **dense-grid** dvscf (`newdq2`, `dvqpsi.f90:834-907`), applied by
   `adddvscf2` (`:1013-1040`); PAW adds `int3_paw` read from `prefix.dvscf_paw_qN` (one complex
   `(ih,jh,na,is)` record per pattern, `EPW/src/io/io.f90:1820-1895`), the one-centre Hxc change Δ(D¹−D̃¹)
   that ph.x computes with `PAW_dpotential`.

`fildvscf` keeps its meaning (induced Hxc potential, NLCC included, with the augmentation part of the
induced density already inside `drhop` via `addusddens`).  No becsum/dbecsum terms enter the vertex.

**Consequences for the existing identities.** g is no longer Hermitian between q and −q:
g_mn(k,q) − g_nm(k+q,−q)^* = (ε_m − ε_n)⟨ψ_m|∂S/∂u_q|ψ_n⟩, and the translational sum rule becomes
Σ_κ g_mn,κα(k,0) = (ε_m − ε_n)⟨ψ_m|S ∂_α|ψ_n⟩.  Both checks must be generalised.

**The real blocker is the projection metric, not the vertex.** `projwfc.x` and EPW's Amn use
S (`PP/src/projwfc.f90:910-1045`, `EPW/src/pw2wan.f90:1035-1037`); PAOFLOW's internal projector is plain-metric
throughout (`do_atwfc_proj.py:815-825, 1164-1245`): it Löwdin-orthonormalises the bands with ⟨ψ|ψ⟩, which on
US/PAW saves is far from the identity (measured plain norms 0.21–1.63 for GaAs USPP/PAW, 1.00–1.93 for Pt USPP,
off-diagonals up to 0.4), so bands of different energies are mixed before H = A†EA is built.  Any US/PAW work
must first make the internal projector S-consistent: S|ψ⟩ = ψ + Σ β_i q_ij ⟨β_j|ψ⟩, O = ⟨φ|S|φ⟩,
φ̃ = O^{−1/2}φ, A_ni = ⟨φ̃_i|S|ψ_n⟩, no band Löwdin.  The PAO rotation g^PAO = A_{k+q}^T d A_k^* then stays
valid (χ_j = P_S φ̃_j); the `'pao'` basis mode needs a second ket set Σ_n A^*_nj ε_n ψ_n for the −ε·qq term.
This also means the existing EPW-source route was never metric-consistent for US/PAW with internal projections
(all local `.epb` data sets are norm-conserving, so it was never exercised).

**New inputs.** UPF augmentation data (`PP_QIJL` or `PP_QIJ`+`PP_QFCOEF`+`PP_RINNER` via `setqfnew`,
`q_with_l`, `nqf`, `nqlc`, `l_max_rho`, PAW `shape`/`cutoff_r_index`; `read_upf.py` parses none of these),
real Gaunt coefficients (`aainit`) and `tab_qrad`/`qvan2`, the ground-state V_eff = vltot + v_Hxc (from
`pp.x plot_num=1`, 10 significant digits, or a native `v_of_rho` ≈ 500 lines + XC kernels), and for PAW the
`dvscf_paw` files plus a star rotation of int3_paw with Wigner D^{l_i}D^{l_j} (`PAW_dusymmetrize`; `dfile_star`
does not rotate it) and conjugation for −q members.

**Size.** ≈1000 Fortran lines to mirror (EPW ≈400 collinear + QE ≈640) → 400–600 Python lines for collinear
US; +≈320 for noncollinear/SOC US; PAW with `ddd_paw` recomputed natively ≈900 more lines (or read deeq from
a patched QE).  Per-(k,q) cost: extra projector GEMMs only (no new FFTs); per q: `qvan2` at |q+G|, int2, int3
(3nat dense FFTs); US/PAW mainly raise ecutrho.

**Validation.** No US/PAW DFPT or EPW data exist on this machine (every dvscf/.epb set is NC); the
test-suite cases `epw_trev_uspp`/`epw_trev_paw` (SiC) need pseudopotentials not shipped.  Plan: Si with
`Si.pbe-n-rrkjus_psl.1.0.0` (local) at dual 4 with `epbwrite` → epmatq parity; GaAs PAW (local psl UPFs) for
PAW and time-reversal members.  Two apparent EPW bugs limit parity there: with `doublegrid` EPW overwrites the
dense dvscf with its smooth interpolation before `newdq2` (`ep_coarse.f90:118-129`; ph.x does it correctly), and
in the time-reversal branch `int3_paw` is added unconjugated (`dvqpsi.f90:907`).  Reference-free checks: the
S sum rule, the corrected hermiticity, the star self-check, and the projections against `projwfc.x`.

**Order of work.** (1) S-metric internal projector (+ `qqq` already parsed) with a projwfc.x comparison on
the existing US/PAW saves; (2) UPF augmentation parsing, Gaunt, `tab_qrad`/`qvan2`, V_eff reader, deeq/qq;
(3) int1/int2/int3 and D_eff in `nonlocal_perturbation`; (4) US parity on Si rrkjus; (5) PAW: `dvscf_paw`
reader, rotation, `ddd_paw`.  Estimated 2–3 weeks including the projector change.

### B. nscf on the irreducible wedge

**Implemented (Oct 25 2026)** as planned, with three deviations: (i) `rotate_block` does not trust
the umklapp of the map but derives the integer shift from the block's own header k (the stored k of
an automatic grid may carry a different lattice translation than the folded list); (ii) the option
is `unfold_kpoints` everywhere (`PAOFLOW`, `DataController`, `pao_hamiltonian_from_save`,
`CoarseVertexSetup`; `read_nscf(unfold=True)`), and `CoarseVertexSetup` detects an irreducible-wedge
save by itself (`stores_irreducible_wedge`: Γ-centred `monkhorst_pack` element with fewer stored k
than grid points), so the dense-q drivers need no new keyword, and the example and generated
drivers pass `unfold_kpoints=stores_irreducible_wedge(SAVEDIR)` (the Si example ships its nscf
with `K_POINTS automatic`; `kmesh.py --explicit` restores the list EPW needs); (iii) the projections at $Sk$ come from feeding the rotated block
through the unchanged `read_QE_wfc` → `calc_atwfc_k` path (the Löwdin step is now
`loewdin_orthonormalize`).  The `DataController` expansion replaces `kpnts`, `kpnts_wght` (uniform,
same total), `my_eigsmat` and `nkpnts` on the reading rank before the broadcast.

Validation.  V1 (the 16 QE representatives of the 216-point Si save, 36 non-trivial operations incl.
24 with $t=\tfrac14$, 137 umklapps): converged bands agree to $1\times10^{-6}$ inside complete
degenerate groups; the two top stored bands deviate ($9\times10^{-4}$ and 0.96: loosely converged
and a cut multiplet).  V3 (real `6 6 6 0 0 0` nscf, 16 k): blocks $5.5\times10^{-7}$; HRs equal to
the `expand_wedge=True` (pao_sym) route to $4\times10^{-9}$ eV and to the explicit full-grid nscf to
$2\times10^{-3}$ eV (top-band arbitrariness of either nscf); PAO vertex at Γ and at a rotated umklapp
member $9\times10^{-7}$ relative (bands < 10).  V4 (Pb FR `4 4 4 0 0 0`, 8 k, spin-orbit,
`diago_full_acc`): spinor blocks $1.8\times10^{-12}$; time-reversed blocks vs the stored ones at
$-k$ $3.3\times10^{-12}$.  Tests: `tests/unit/elphon/test_wavefunction_symmetry.py` (SU(2)
homomorphism and Pauli covariance, map coverage, QE first-of-star rule, real-space identity
$(O_g\psi)(x)=U\psi(g^{-1}x)$ with umklapp, spinor and time reversal) and
`tests/integration/elphon/test_irreducible_wedge.py` on the asset's `si_ibz.save`.  Scripts:
`Workspace/PAOFLOW/validation/dfpt_vertex_parity/ibz_*.py`, saves `dfpt_port/{si_ibz,pb_fr_ibz}`.

**Formula** (QE convention, operation {R_P|t} on positions `x → R_P x − t` as in `SpaceGroup`; W = R_P^{−T}
acts on crystal k and Miller indices):
κ' = Wκ, m' = Wm, **c'(m') = exp(+2πi (κ'+m')·t) c(m)** (phase after relabelling; equivalently
exp(2πi(κ+m)·R_P^{−1}t) before).  Spinors: c'_σ = phase · Σ_σ' U(R)_σσ' c_σ', U = `find_u(det R · R)`
(`PW/src/divide_class_so.f90:3547-3687`, right-handed axis–angle, global sign irrelevant).  Time reversal
Θ = iσ_y K: k → −k, m → −m, c → c^*, spinor (c↑,c↓) → (c↓^*, −c↑^*).  The sphere |k+G|² ≤ ecut maps onto
itself exactly (edge G at round-off can differ; `gk_sort` uses ≤).  The same construction is QE's
`pw2wannier90 irr_bz` (`rotate_evc`) and EXX (`exx_base.f90:240-290`); EPW never rebuilds k from an IBZ (it
rotates full-grid states only to reuse the dvscf of the irreducible q).

**Gauge.** The rotated state equals QE's own ψ_{Sk} only up to a unitary inside each degenerate group; the
vertex is invariant as long as d and A come from the same rotated states, which is why A_{Sk} must be
**recomputed** from the rotated block (`calc_atwfc_k` with the rotated Miller list and k) rather than rotated.
The same degenerate-cut caveat as `unfold='irreducible'` applies to the top band.

**Design.** New module `elphon/wavefunction_symmetry.py` (~300–350 lines): a map builder (full-grid label →
IBZ point, operation, TR flag, G0; identity first; assert coverage and multiplicities = XML weights × N/2), a
`rotate_block` (relabel, phase, SU(2), TR, then fold to the representative in [0,1) with the umklapp folded
into the Miller list), and a `WavefunctionProvider` holding the IBZ blocks with an LRU cache of rotated ones,
exposing `kpts_cryst`, `block(ik)`, expanded eigenvalues and the `gkspace` for the projections.  Hooks:
`CoarseVertexSetup.__init__/block()/nscf_grid()` (`dfpt_vertex.py`), a provider hook in `read_QE_wfc`/
`calc_proj_k`, a "virtual full-grid nscf" in `DataController` (`kpnts`, `kpnts_wght = 1/N`, expanded
`my_eigsmat`) so that `pao_hamiltonian(expand_wedge=False)` and the vertex share one A, `read_nscf(...,
unfold=True)`, `pao_hamiltonian_from_save(unfold_wavefunctions=True)`.  Requires `K_POINTS automatic`
Γ-centred (explicit IBZ lists have no `monkhorst_pack` element); TR from `noinv`/magnetisation; `nosym`
still unsupported (also breaks the dvscf star).  Trap: `pao_sym` indexes operations as s = R_P^T (k → R^{−1}k);
do not mix with `SpaceGroup`.

**Gain.** Pb 12³ SOC 40 bands: 1728 → 72 k (3.6 GB → 150 MB of wavefunctions, no block-cache thrashing);
the vertex and projection arithmetic over the full grid are unchanged.

**Size and validation.** ≈450 production lines + ≈200 test lines.  V1 without any QE run: choose the 16
representatives of `si.save` (216 k, 48 operations, half with t = ¼) with QE's `kpoint_grid` rule, rebuild all
216 and compare block overlaps O = C_stored^H C_rot (off-group < 1e-6, complete-group singular values 1 ± 1e-6;
the stored k have 8 digits); V2: the 27 PAO vertices from the provider vs the full grid (< 1e-6, complete band
groups); V3: a real IBZ nscf of Si (6 6 6 automatic, 16 k) at ~1e-5; V4: spinors on `dfpt_port/pb_fr` (8
representatives of 64 k).  Estimated one week.

### C. Collinear spin polarisation (LSDA)

**ph.x conventions.** dvscf records are `(r, [up, down])` (`davcio_drho`, `lrdrho` with `nspin_mag = 2`;
`dv_of_drho.f90:87-89, 281-287`); `read_dvscf(nspin_mag=2)` already returns both channels.  In `elphel`
(`PHonon/PH/elphon.f90:455-591`) the dvscf channel of spin σ multiplies the wavefunctions of the same σ
(`apply_dpot` with `current_spin`); the bare local and KB terms are spin independent; NLCC is inside dvscf.
Wavefunctions are `wfcup<ik>.dat` (isk = 1) and `wfcdw<ik>.dat` (isk = 2) with the header `ispin`; the XML
`ks_energies` hold `nbnd_up` up values followed by `nbnd_dw` down values per k (`qexsd_init.f90:1236-1240`),
`nks` is the single-spin count, and there is no `nbnd` element.  Two Fermi levels (`tot_magnetization`) are
refused by ph.x for metals and should be refused here.

**Pre-existing PAOFLOW bug (confirmed, to fix first).** The internal projector opens `wfcdw` for `ispin = 0`
and `wfcup` for `ispin = 1` (`do_atwfc_proj.py:748-751`, copied in `plane_waves.py` and `do_epsilon.py:
1265-1267`), while `read_QE_xml.py:189-198` and the `projwfc` path put spin up in channel 0.  With the internal
projector, `Hks[..., 0]` of a spin-polarised run is A(down)·E(up)·A(down)†.  Invisible for the Pt example
(M ≈ 0); fix = 0 → `wfcup`, 1 → `wfcdw`, plus an assertion on the header spin.  This changes the results of
existing spin-polarised internal-projection runs and needs the user's go-ahead.

**No EPW reference.** EPW refuses LSDA (`EPW/src/readin.f90:1394`, "El-ph and spin not implemented").
Validation comes from ph.x itself: `electron_phonon='simple'` prints the total λ_qν with one joint Fermi level
and DOS (N↑+N↓)/2, λ = (S↑+S↓)/(ω²(N↑+N↓)) (`elphon.f90:1281-1372`; `read_elph_inp_lambda` parses the
file), plus reference-free checks (the Γ sum rule per spin, which fails if the channels are swapped because
V↑ ≠ V↓; the PAO vertex rotated back with the eigenvectors of H_σ at coarse k).  The PAOFLOW-patched ph.x
dump (`qe-7.4.1`) writes both spins in `elphmat` (ik ≤ N up, > N down) but only spin up of `dvind`;
`vertex_from_qe_elphmat` would silently keep spin down (duplicate k labels) and `read_qe_ahc_gkk` fails on
2N k.  AHC works with `skip_upper=.true.`.

**Physics bookkeeping.** Everything is a sum of two independent channels with g_s = 1 sharing one Fermi
level: λ_tot = (N↑λ↑ + N↓λ↓)/(N↑+N↓) (the specific-heat enhancement; PAOFLOW's per-channel λ_σ is normalised
by its own DOS, so a plain sum is wrong), α²F likewise, ρ = (ρ↑^{−1} + ρ↓^{−1})^{−1} and ω_p² = Σ_σ ω²_p,σ for
transport, ε₂ = Σ_σ ε₂,σ with spin-conserving transitions and joint gaps, σ = σ↑ + σ↓ for SERTA/MRTA (plus the
current spin polarisation).  Fermi-surface pair coupling / anisotropic ME is not meaningful for a
ferromagnet (exchange splitting ≫ Δ): guard it.

**Change set.** (0) spin labels (~10 lines); (1) `read_nscf_plane_wave_setup`: `total_magnetization`,
refuse two Fermi energies (~8); (2) `CoarseVertexSetup(spin=0|1)`: wfc file and dvscf channel, raise for
noncolin+domag (~25); (3) `read_nscf`: per-spin eigenvalues, `nspin` (~20); (4) a joint Fermi level helper
(`_fermi_level(concat, σ, nelec, spin_deg=1)`) and a `fermi_levels=` override of `precompute_dense_electrons`
(~50); (5) `prepare_dense_vertex(spin=)` with spin in the cache key, `'elphmat'`/`'ahc'` k-block split,
`'epw'` raising (~40); (6) the four drivers loop over σ with A `(nbnd,nawf,nk,nspin)`, free the shared window
between spins (peak memory unchanged), DOS-weighted combination and per-spin outputs (~310); (7) generated
drivers keep the 4-D A (~30).  Total ≈500–650 production lines + 400–550 test lines; cost ×2 for the vertex,
`g(R_e,R_p)` and the dense electrons.  Validation: spin-degenerate limit (`nspin = 2`, zero starting
magnetisation on Al/Pb must reproduce `nspin = 1`), then bcc Fe with `PSEUDOS/nc-sr-05_pbe_standard_upf/Fe.upf`
(NC, NLCC, 6 projectors; a ≈ 5.42 bohr, M ≈ 2.2 μB, ecut ≈ 90 Ry, 6³–8³ k, 3³ q) against ph.x `'simple'`.
Estimated one to two weeks.

### Recommended order
LSDA step 0 (the spin-label fix) is independent and small; the irreducible-wedge provider is the best
value per line and needs no new DFT data; LSDA next (ph.x reference only); ultrasoft/PAW last, because its
precondition is the S-metric internal projector, which is a change to PAOFLOW's core projection step with
consequences well beyond electron-phonon coupling.
