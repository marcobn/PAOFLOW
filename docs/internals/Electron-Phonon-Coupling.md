# Electron–Phonon Coupling

This page documents the **PAO-interpolation route** of `PAOFLOW.elphon`: the
production path for computing electron–phonon (el‑ph) coupling and isotropic
Eliashberg superconducting properties ($\alpha^2F$, $\lambda$, $\omega_{\log}$,
$T_c$) by reading Quantum ESPRESSO's (QE) **full** DFPT coarse-grid coupling and
interpolating it in the PAOFLOW pseudo-atomic-orbital (PAO) gauge.

---

## Contents

- [Why this route](#why-this-route)
- [Theory](#theory)
- [Module map](#module-map)
- [Required inputs](#required-inputs)
- [Workflow 1 — coarse-q (recommended baseline)](#workflow-1--coarse-q-recommended-baseline)
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

- It reads QE's **full** `el_ph_mat` / AHC coupling dump — bare local, bare
  nonlocal, induced (Hartree+xc), and any NLCC/ultrasoft augmentation — exactly
  as `ph.x` computed it. **No potential reconstruction**, so it works unchanged
  for norm-conserving, ultrasoft, and PAW pseudopotentials.
- The PAO gauge is a **fixed, deterministic atomic-orbital basis**: unlike a
  Wannier-function interpolation (e.g. EPW), no disentanglement, gauge-fixing,
  or manual window selection is required.
- Interpolation to dense grids reuses PAOFLOW's validated Wigner–Seitz
  generalized-Fourier machinery (the same used for `HRs` in every other
  PAOFLOW property).

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
\lambda_{q\nu} = \frac{1}{N(E_F)\,\omega_{q\nu}^2}\,\frac{1}{N_k}\sum_{k}
   \big|g_{q\nu}(k)\big|^2\,\delta(\varepsilon_k - E_F)\,\delta(\varepsilon_{k+q}-E_F),
$$

evaluated with both electronic states pinned to $E_F$ (energies referenced to
$E_F=0$); the only genuine spectral integral in the whole calculation is the
**phonon** frequency integral $2\int \alpha^2F(\omega)/\omega\,d\omega$. $T_c$
follows from the McMillan / Allen–Dynes formula with Coulomb pseudopotential
$\mu^*$.

**The PAO-gauge vertex.** Starting from QE's Bloch-basis Cartesian deformation
potential $d_{mn,\kappa\alpha}(k,q) = \langle m,k{+}q|\partial_{u_{\kappa\alpha}}V|n,k\rangle$
(read directly from the coupling dump — never recomputed), the vertex is
rotated into the PAO gauge with the same projection matrices $A_k$ used to
build `HRs`,

$$
g^{\rm PAO}_{ij,\kappa\alpha}(k,q) = \big[A_{k+q}^\dagger\, d_{\kappa\alpha}(k)\, A_k\big]_{ij},
$$

and Fourier-transformed $k\to R_e$ to a real-space vertex $g(R_e)$ per coarse
$q$ (`vertex_pao_R`). This is the "half-transformed" object at the heart of
both workflows below.

---

## Module map

All paths relative to `src/PAOFLOW/elphon/`.

| File | Role |
|------|------|
| `qe_elph_io.py` | Readers for QE coupling dumps: `read_qe_el_ph_mat` (patched-QE `el_ph_mat`, any pseudopotential), `read_qe_ahc_gkk` (unpatched QE AHC, norm-conserving only), `el_ph_mat_to_cartesian` (pattern→Cartesian rotation), `read_qe_dyn` (`.dyn` frequencies/eigenvectors), plus the older `elph.inp_lambda`/`lambda.in` readers used by the property-only route. |
| `elph_bloch.py` | Core Bloch/PAO-gauge machinery: `read_nscf` (k-points, lattice, Fermi level, **and crystal symmetries** from `data-file-schema.xml`), `kq_index_map`, `vertex_pao_R`, `_ws_lattice` (Wigner–Seitz images/weights), `precompute_dense_electrons` + `lambda_q_dense_ws_fast` (dense-$k$ electron cache + Fermi-surface double delta, shared across q). |
| `do_pao_eph.py` | **Workflow 1** driver: `eliashberg_from_qe_coupling` (coarse-q, MPI over q), `vertex_from_qe_elphmat` / `vertex_from_qe_ahc` (build one coarse-q half-vertex $g_q(R_e)$). |
| `do_pao_eph_dense_q.py` | **Workflow 2** driver: `eliashberg_dense_q`, `build_g_ReRp` / `g_Re_at_q` (double real-space vertex $g(R_e,R_p)$ and its dense-q evaluation), `phonon_interp_from_dyn` (dense-q phonon interpolation directly from `.dyn` files, with acoustic sum rule), `irreducible_qmesh` / `_crystal_point_group` (IBZ symmetry reduction). |
| `eph_kq.py` | Property engine: `eliashberg_from_modes` (builds $\alpha^2F$, $\lambda$, $\omega_{\log}$, $T_c$ from mode-resolved $\lambda_{q\nu}$), `mcmillan_allen_dynes_tc`, `phonon_moments`. Shared by every route, including the property-only and finite-difference routes. |
| `qe_matdyn.py` | Wigner–Seitz interpolation of QE force-constant files (`*.fc`, `a2Fmatdyn.*`) — used by the **property-only** route (reads QE's own already-interpolated coupling; not the route on this page, but the same engine). |
| `basis.py`, `displacements.py`, `do_elphon.py`, `io.py`, `symmetry.py`, `fold.py`, `gkq.py`, `do_gkq.py`, `dvscf_fd.py` | The finite-difference frozen-phonon route; see [Electron-Phonon Coupling](Electron-Phonon-Coupling). |

---

## Required inputs

Both workflows need a **coarse, commensurate** electron k-grid and phonon
q-grid from Quantum ESPRESSO:

1. **`pw.x` nscf** on the **full** (unreduced) Γ-centred k-grid:
   `nosym=.true.`, `noinv=.true.`, `nbnd` > number of PAO orbitals (`nawf`).
   This is a hard requirement — the PAO vertex needs every `k+q` to exist on the
   k-list; QE-side k-grid symmetry reduction is **not** compatible with this
   route (see [Grid consistency rules](#grid-consistency-rules)).
2. **`ph.x` DFPT** phonons on the coarse q-grid, producing `<prefix>.dyn<iq>`
   (frequencies + eigenvectors) for every q.
3. **The coupling dump**, one of:
   - **AHC** (`SOURCE='ahc'`): unpatched QE, `electron_phonon='ahc'` →
     `ahc_dir/ahc_gkk_iq<iq>.bin`. **Norm-conserving pseudopotentials only.**
   - **`el_ph_mat`** (`SOURCE='elphmat'`): a **PAOFLOW-patched** `ph.x`
     (`electron_phonon='interpolated'`, env `PAOFLOW_DUMP_ONLY=1`) →
     `elph_dir/elphmat.<iq>.dat`. Works for **any** pseudopotential (NC,
     ultrasoft, PAW), including NLCC.
4. **PAOFLOW electronic structure**: `projections` + `projectability` +
   `pao_hamiltonian` on the same nscf save, giving the PAO projections `A_k`
   (grab `data_arrays['U'][...,ispin]` **before** `pao_hamiltonian`, which
   deletes it) and the PAO Hamiltonian `HRs`.

---

## Workflow 1 — coarse-q (recommended baseline)

`eliashberg_from_qe_coupling` (in `do_pao_eph.py`) interpolates **only the
electronic k-grid** to a dense mesh; the phonon q-grid stays at the coarse
DFPT resolution (one term per irreducible q, combined with QE star weights).

Per irreducible q:
1. Read the coupling dump and rotate it to the PAO gauge → $g_q(R_e)$
   (`vertex_from_qe_elphmat` / `vertex_from_qe_ahc`).
2. Wigner–Seitz-interpolate `HRs` and $g_q(R_e)$ to a dense $N_k^3$ grid
   (`precompute_dense_electrons`, computed **once** and reused for every q,
   since $E(k+q)$/eigenvectors are index shifts of the same dense grid).
3. Evaluate $\lambda_{q\nu}$ via the Fermi-surface double delta
   (`lambda_q_dense_ws_fast`), zero the $\Gamma$ acoustic blow-up (QE
   convention), and accumulate with the q star weight.
4. Combine all q with `eliashberg_from_modes` → $\alpha^2F$, $\lambda$,
   $\omega_{\log}$, $T_c$.

The per-q loop is MPI-parallel (`comm=` argument); the dense-electron cache is
recomputed redundantly on every rank (cheap relative to the coupling
interpolation).

**Limitation:** because the phonon q-grid is not densified, $\alpha^2F/\lambda$
accuracy is capped by the coarse DFPT q-mesh. A DFPT grid finer than $3^3$
(e.g. $6^3$) is generally required for a metal's $q\to0$ acoustic region to be
adequately sampled — see [Validation](#validation).

---

## Workflow 2 — dense-q (k *and* q interpolation)

`eliashberg_dense_q` (in `do_pao_eph_dense_q.py`) additionally Wigner–Seitz
interpolates the **phonon q-grid**, following the same
Wannier-Fourier-interpolation idea as EPW but in the deterministic PAO gauge
(no wannierization step needed).

### The double real-space vertex $g(R_e, R_p)$

The coarse-q half-vertices $g_q(R_e)$ (from step 1 of Workflow 1, but now for
**every** q of the full coarse grid — symmetry-unfolding of an irreducible set
is not yet automated) are Fourier-transformed over q to phonon-cell space,

$$
g(R_e,R_p)_{ij,c} = \frac{1}{N_q}\sum_q e^{-2\pi i\,q\cdot R_p}\,g_q(R_e)_{ij,c}
\qquad(\text{`build\_g\_ReRp`}),
$$

and any dense q recovers the half-vertex by a Wigner–Seitz sum over $R_p$,

$$
g_q(R_e) = \sum_{R_p} W_p\, e^{+2\pi i\,q\cdot R_p}\, g(R_e,R_p)
\qquad(\text{`g\_Re\_at\_q`}),
$$

which feeds directly into the **unchanged** `lambda_q_dense_ws_fast` used by
Workflow 1 — the electronic side is byte-for-byte identical.

### Dense-q phonons

`phonon_interp_from_dyn` builds the phonon interpolator **directly from the
coarse `.dyn` files** (no `q2r.x`): it reconstructs the full-precision
mass-weighted dynamical matrix $D(q) = \sum_\nu \omega_\nu^2\, e_\nu e_\nu^\dagger$
from each `.dyn`'s diagonalisation block, Fourier-transforms to phonon cells,
applies a **simple acoustic sum rule** (forces $\Gamma$ exactly to zero), and
Wigner–Seitz interpolates/re-diagonalises at any dense q. This intentionally
avoids `q2r.x`, whose star bookkeeping is incompatible with full-grid
(unreduced) `.dyn` dumps such as those written for the AHC path.

> A `min_freq_thz` soft-mode guard exists to drop spurious/imaginary
> interpolated modes; empirically (Pb) it makes little difference once the DFPT
> q-grid is fine enough — see [Validation](#validation).

---

## Symmetry reduction of the dense q-grid

`irreducible_qmesh` (with `_crystal_point_group` to filter QE's reported
lattice holohedry down to the true crystal point group via the atomic basis)
folds the Γ-centred `nq_dense^3` grid to its irreducible wedge (plus time
reversal, $\lambda_q=\lambda_{-q}$), returning representative q-points and
star-multiplicity weights. Because $\lambda_{q\nu}$ is a point-group
invariant, this gives an **identical result** to summing the full grid, at a
fraction of the cost (e.g. ~30× fewer q-evaluations for a cubic $18^3$ mesh).

Enable it by passing `sym_rots=info['s_cryst']` (from `read_nscf`), plus
`tau_cryst`/`species` for the basis filter, to `eliashberg_dense_q`. QE writes
the full lattice holohedry (`nrot`) in `data-file-schema.xml` even when the
nscf itself was run with `nosym=.true.`, so this works without any extra QE
input. A rank-0 diagnostic line reports the reduction, e.g.:

```
dense-q symmetry: 195 irreducible / 5832 full q  (29.9x fewer)
```

In practice the numerical agreement between the full-grid and IBZ-reduced sums
is at the sub-percent level (not bit-identical — symmetry-equivalent q are
computed by independent Wigner–Seitz interpolations, which agree only up to
the interpolation's own discretisation), far below other sources of error in
the method.

---

## Parallelisation and memory

- **MPI over the dense-q loop** (both workflows), distributed with
  `utils.communication.load_balancing` and combined with `Allreduce(SUM)`.
  Cap the rank count at the number of q-points actually evaluated (irreducible
  count, if symmetry is enabled).
- **Node-shared vertex.** The double real-space vertex $g(R_e,R_p)$ can be a
  multi-GB array; `eliashberg_dense_q` allocates it **once per node** via
  `MPI.Win.Allocate_shared` (built only by the node-local rank 0), instead of
  an independent copy per rank — critical to avoid memory overflow when
  running many ranks on a single large-memory node.
- The dense-electron cache (`precompute_dense_electrons`) is still recomputed
  redundantly per rank; it is small relative to the vertex.
- **Thread/rank balance:** because the per-q kernel is small-matrix BLAS work
  (sub-linear OpenMP scaling), prefer more MPI ranks with a modest number of
  BLAS/OMP threads per rank (e.g. 2–4), matched via `OMP_NUM_THREADS` and the
  BLAS-specific thread variable (`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`).
  Pick a rank count that divides the (irreducible) q-count evenly to avoid one
  rank setting the wall time.

---

## Grid consistency rules

| Constraint | Reason |
|---|---|
| nscf k-grid = full, unreduced, Γ-centred (`nosym`, `noinv`) | the PAO vertex needs `k+q` to exist on the k-list |
| `ph.x` q-grid = `.dyn` grid = `QGRID` passed to PAOFLOW | one coupling/`.dyn` per q, index-matched |
| `NK_DENSE % NQ_DENSE == 0` (dense-q workflow) | `k+q` on the dense grid is evaluated as an integer index roll, requiring commensurability |
| coarse k-grid vs. coarse q-grid | in practice the AHC path tolerates `q` not exactly dividing the k-grid (off-grid `k+q` handled by the `A_{k+q}` projection lookup); the `elphmat` path is less exercised in this regime — verify before relying on it |

A DFPT q-grid that is too coarse (e.g. $3^3$ for a metal like Pb) will produce
an **inflated** $\lambda$ under dense-q interpolation, from Fourier-overshoot
of the near-$\Gamma$ acoustic branch and coupling — not a code defect, but a
genuine convergence requirement; see [Validation](#validation).

---

## The `paoflow-gen elphon` CLI workflow

`paoflow-gen` (see [Input and Script Generators (CLI)](Input-and-Script-Generators-CLI))
has a dedicated `elphon` workflow that writes a two-phase `main.elphon.py`:

```bash
python main.elphon.py inputs    # write the ph.x phonon (+ AHC) input templates
# ... run pw.x (scf, nscf) and ph.x (phonon, AHC) here ...
python main.elphon.py analyse   # PAO interpolation -> alpha^2F, lambda, Tc
```

Prompts include the coupling source (`ahc`/`elphmat`), the coarse k/q grids,
`NBND`, atomic masses, `NELEC`, the dense grid size(s), Fermi smearing, and
$\mu^*$. Answering **"Also interpolate the phonon q-grid (dense-q, experimental)?"**
with yes additionally asks for `NQ_DENSE` and routes to
`eliashberg_dense_q` (using `phonon_interp_from_dyn`, no `q2r.x` step); the
crystal symmetries from `read_nscf` are always passed automatically, so the
generated `analyse()` benefits from the IBZ reduction with no further input.
A companion `plot.elphon.py` plots $\alpha^2F(\omega)$ and the cumulative
$\lambda(\omega)$ from `output/eliashberg.npz`. The `analyse` phase is
MPI-parallel: `mpirun -np N python main.elphon.py analyse` (dense-q variant)
or `mpirun -np N python main.elphon.py analyse` (coarse-q variant); both cap
useful `N` at the number of q-points evaluated.

---

## Validation

Elemental fcc Pb (norm-conserving pseudopotential, AHC coupling), nscf
$9^3$ k-grid:

| DFPT q-grid | dense k | dense q | $\lambda$ | $\omega_{\log}$ | $T_c$ (Allen–Dynes) |
|---|---|---|---|---|---|
| $3^3$ (coarse-q only, Workflow 1) | — | — | 1.95 | — | — |
| $3^3$ → dense-q $3^3$ | 18 | 3 | 1.95 (matches coarse) | — | — |
| $3^3$ → dense-q $6^3$ | 18 | 6 | 2.67 (inflated) | 27 K (too low) | 6.4 K |
| $3^3$ → dense-q $18^3$ | 18 | 18 | 12.8 (diverges) | 1.3 K | 1.3 K |
| **$6^3$ → dense-q $6^3$** | 18 | 6 | **1.49** | **60.3 K** | **7.56 K** |
| **$6^3$ → dense-q $18^3$** | 18 | 18 | **1.26** | **61.3 K** | **6.29 K** |

Literature values for Pb: $\lambda\approx1.2$–$1.5$, $\omega_{\log}\approx60$ K,
$T_c\approx6$–$7$ K. The $3^3$ column demonstrates the **coarse-DFPT-grid
failure mode**: densifying q beyond the DFPT resolution amplifies (rather than
converges) $\lambda$, because the near-$\Gamma$ acoustic phonon/coupling
Fourier interpolation overshoots on a mesh that coarse. Full-grid vs.
IBZ-symmetry-reduced sums at $6^3$ agree to $<1\%$ ($\lambda=1.4857$ vs.
$1.4962$), confirming the symmetry reduction.

Diagnostics that were checked and **ruled out** as the cause of the $3^3$
inflation: masking soft/imaginary interpolated phonon modes (their
$\lambda_{q\nu}$ is already $\approx0$); forcing a naive coupling acoustic sum
rule $\sum_{R_p} g(R_e,R_p)=0$ (this made the inflation *worse*). The fix that
worked was simply using an adequately fine DFPT q-grid.

---

## Practical notes and pitfalls

- **`el_ph_mat` is in the pattern (irrep) basis** — always rotate to Cartesian
  with `el_ph_mat_to_cartesian` using the dump's own `u` matrix, not a
  hand-parsed `patterns.xml` (a historical source of sign/gauge bugs; see
  [Elphon module status log](Elphon_module) §6.2 for the fully worked-out
  cautionary tale from the finite-difference-from-`dvscf` attempt).
- **Grab `A_k` before `pao_hamiltonian`.** `pao_hamiltonian()` deletes
  `data_arrays['U']`; copy it immediately after `projectability`.
- **Units.** `HRs`/eigenvalues in eV; QE smearing (`sigmas_ry`) and phonon
  frequencies from `.dyn`/dumps in Ry/THz as documented per function
  (`RY_TO_EV`, `RY_TO_THZ`, `AMU_RY` constants in `elph_bloch.py`).
- **q2r.x is not used** by the dense-q phonon interpolation — it is
  incompatible with full (unreduced) `.dyn` grids as written for the AHC path
  (`q2r.x` errors with `nc already filled`). `phonon_interp_from_dyn` reads
  the `.dyn` force-constant blocks directly instead.
- **Environment.** Run in an environment with an editable PAOFLOW `src` tree
  and `mpi4py` (e.g. `conda run -n work python ...`); `MPI.Win.Allocate_shared`
  requires an MPI-3 build (OpenMPI/MPICH both qualify).

---

## API summary

```python
from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.elph_bloch import read_nscf
from PAOFLOW.elphon.do_pao_eph import eliashberg_from_qe_coupling
from PAOFLOW.elphon.do_pao_eph_dense_q import (
    eliashberg_dense_q, phonon_interp_from_dyn, irreducible_qmesh,
)

pf = PAOFLOW.PAOFLOW(workpath=..., outputdir='output', savedir='<prefix>.save')
pf.projections(configuration='standard', basispath=BASIS)
pf.projectability(pthr=0.90)
A = pf.data_controller.data_arrays['U'][:, :, :, 0].copy()   # BEFORE pao_hamiltonian
pf.pao_hamiltonian()
HRs = pf.data_controller.data_arrays['HRs']
info = read_nscf(SAVEDIR)                                     # includes info['s_cryst']

# Workflow 1 (coarse-q):
out = eliashberg_from_qe_coupling(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    coupling_dir, q_weights, ng, dyn_paths,
    source='ahc', masses_amu=[...], nk_dense=18, sigmas_ry=[0.02],
    nelec=..., mu_star=0.10,
)

# Workflow 2 (dense-q, symmetry-reduced):
phonon_at_q = phonon_interp_from_dyn(dyn_paths_full, qgrid, info['bg'], info['at'])
out = eliashberg_dense_q(
    A, HRs, info['kpts_cryst'], info['bg'], info['at'],
    coupling_dir, qgrid, q_cryst_coarse, dyn_paths_full, ng,
    phonon_at_q, nq_dense=18, source='ahc',
    masses_amu=[...], nk_dense=18, sigmas_ry=[0.02], nelec=..., mu_star=0.10,
    sym_rots=info['s_cryst'], tau_cryst=info['tau_cryst'], species=info['atom_names'],
)
```

`out` in both cases is a dict with `omega`, `a2F`, `lambda`, `lambda_qv`,
`omega_qv_thz`, `omega_log`, `Tc_mcmillan`, `Tc_allen_dynes`, `dos_ef`, ...
(see `eliashberg_from_modes` for the full key list).

---

## References

- L. A. Agapito and M. Bernardi, *Ab initio electron-phonon interactions using
  atomic orbital wave functions*, Phys. Rev. B **97**, 235146 (2018).
- F. Giustino, *Electron-phonon interactions from first principles*, Rev. Mod.
  Phys. **89**, 015003 (2017) (EPW / Wannier-Fourier interpolation, the
  conceptual analogue of this route in the PAO gauge).
- P. Giannozzi *et al.*, *Advanced capabilities for materials modelling with
  Quantum ESPRESSO*, J. Phys.: Condens. Matter **29**, 465901 (2017).
- P. B. Allen and R. C. Dynes, Phys. Rev. B **12**, 905 (1975); W. L.
  McMillan, Phys. Rev. **167**, 331 (1968).
- See also [Phonon Module](Phonon-Module) and
  [Input and Script Generators (CLI)](Input-and-Script-Generators-CLI) for the
  companion harmonic-phonon and CLI-generator documentation.
