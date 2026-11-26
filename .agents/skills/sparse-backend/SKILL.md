---
name: sparse-backend
description: "Sparse role of every public PAOFLOW method, and the design of src/PAOFLOW/sparse/. Use when: adding or changing a public method of PAOFLOW.py (new property, new argument), splitting a dense kernel into per-k functions, touching src/PAOFLOW/sparse/, doubling, or a sparse parity test."
user-invocable: true
---

# Sparse backend

How the sparse backend is built, why, and what every new or changed PAOFLOW property owes it.

## Every public method takes a sparse role

A public method of `PAOFLOW.py` ships with its sparse implementation in the same change. `sparse=True`
routes every method through `sparse/dispatch.py`, so a method without a role is a method that
refuses to run in a sparse job.

1. **Pick the kind** in the table of [Extending to a new property](#extending-to-a-new-property)
   and implement it by the checklist in the module docstring of `sparse/properties/__init__.py`
   (per-k split of the dense kernel, a `MeshProperty`, `@sparse_override`, a parity test).
2. **A new argument of a routed method** goes to its sparse side too: the `MeshProperty.__init__`
   or `SparseEngine` method takes the same name and default. Where the sparse side cannot honour a
   value, it raises `NotImplementedError` there, naming the value, the reason and the way out
   (`bands(interpolation=...)` in `sparse/engine.py` is the pattern).
3. **A method left dense-only** stays unmarked and gets an entry in `DENSE_ONLY_REASONS`
   (`sparse/dispatch.py`), the one place the refusal text lives. The entry names the specific
   obstacle (the dense array or the external code the method needs, e.g. "rotates the dense HRs
   into mirror sectors for Z2Pack") and ends with the way out (`_TO_DENSE`, or the sparse
   alternative). "Not implemented yet" is the obstacle of nothing: write what would have to
   exist.
   `test_every_reason_names_an_obstacle` rejects a bare way out. Before settling for a reason,
   try the cheap routes: `@sparse_shared` for a method that only calls routed methods, and
   `@sparse_base_cell` for an `H(R)` transform (or `modifies=False` for a method that only
   reads `HRs`, like `arpes` and `find_weyl_points`), each checked against the dense result on
   `tests/data/silicon` (the round trip failed for `spinor_basis`; its reason records what
   broke). A shared method that needs what a routed one returns (`doped_transport` reads the
   `mu` of `doping`) works outside `fused()` only: a property hands its return value back
   through `MeshProperty.result`, set in `finalize`.
4. **Say it in the method's docstring** as well: one line in `Notes` (sparse: supported /
   dense-only and why), so the user reads it before the job fails.

Done when `pytest tests/unit/sparse/test_sparse_dispatch.py` is green. It checks that every
public method has a role, every dense-only method has a reason and no routed method has one,
every routed method's sparse side accepts all the dense arguments, and every registered property
is routed. A new sparse property is done when its dense-parity test is green as well.

Related skills: `feature-design` (where the method and its backend go), `docstring` (the per-k
functions split out of a dense kernel are new functions: full docstring and type hints).

## The problem being solved

`doubling_Hamiltonian` grows the orbital dimension `nawf` geometrically
(×2 per doubling) on a fixed R grid. Every dense array downstream scales as
`nawf²` — `HRs`, `Hksp`, `v_k` at 547 MB each and `dHksp`/`pksp` at 1.7 GB
each for a single doubling of example01 — and the dense pipeline OOMs for the
doublings that actually matter. The sparse backend replaces all of them with
one thresholded bond list that grows **linearly** in `nawf` under doubling,
and computes properties from per-k iterative eigensolves.

## Memory contract (the whole point — do not weaken it)

1. **No global dense tensor of shape ~`(nk, nawf, nawf)` or
   `(nawf, nawf, nR)` may ever exist.** No `Hksp`, `dHksp`, `pksp`, stored
   `v_k`, `deltakp2`. If a property "needs" one, redesign the property (see
   the consumer seam below), don't materialize the tensor.
2. **No dense object may be indexed by k.** Per-k dense scratch is
   allowed only when it is freed before the next k-point, and only within
   the explicit bounds in rule 3. This *replaces* the original rule ("no
   `.toarray()` / dense `eigh` anywhere, at any size"); the measurements
   that forced the change, and the bounds that now stand in its place,
   are in **Why the dense branch exists** below. Dense arrays in *tests*
   as references are fine, as before.
3. **Allowed dense objects, exhaustively:**
   - band-diagonal arrays `E_k`, `velkp`, `deltakp` — O(nk·nev), k-scattered;
   - ONE per-k eigenvector block `V (nawf, nev)`, discarded before the next
     k-point;
   - the three per-k gradient blocks `W_l = (dH/dk_l) V`, same shape as
     `V`, discarded with it (they replace recomputing that product inside
     the degeneracy loop);
   - ONE per-k `(nawf, nawf)` scratch inside `solver._solve_dense`, freed
     on return, and only while `nawf <= dense_n_max` (4096 → ≤ 268 MB);
   - when a property needs interband sums over every state
     (`needs = {'full_spectrum'}`: band curvature / Hall transport, the
     anomalous/spin/orbital Hall family, linear response, path Berry
     curvature), the per-k `V` is `(nawf, nawf)` and the per-k matrices built
     from it (`KPoint.pksp` `(3, nawf, nawf)`, operator pairs) live for that
     k-point only; still bounded by `dense_n_max`, above which the pass raises
     naming the Sternheimer follow-up;
   - the base-cell (pre-doubling) input stage: `Hks`/`HRs` at the small
     original `nawf` is inherently dense QE input processing; it is
     converted by `SparseHamiltonian.from_data_controller` (the **single
     sanctioned dense→sparse boundary**) and deleted immediately.
4. Failure is loud: guards raise `NotImplementedError`/`RuntimeError` with
   actionable messages. There is no silent dense fallback — a user who hits
   a wall must know it, not swap 40 GB.

## Why the dense branch exists (amendment to rule 2)

The original rule was absolute: strictly iterative solves, no `.toarray()`
at any size. It was replaced deliberately, not eroded. What changed:

**The rule stopped buying anything in this regime.** scipy picks the Krylov
dimension as `ncv = min(n, max(2k+1, 20))` with `k = nev + guard`. At
`nev = bnd = nawf/2` that **degenerates to `ncv = n`**, so ARPACK allocates
a dense `(n, n)` Arnoldi basis per k-point and does `O(n·ncv²)`
reorthogonalization — the same per-k memory class as a dense `eigh`, at
13–66× the cost (measured at n = 144/576/1152 on matched-density Hermitian
matrices). Because `H` is complex, `eigsh` also delegates to the
*non-symmetric* complex Arnoldi (`znaupd`/`zneupd`), not the cheap 3-term
Lanczos recurrence the old docstring implied.

**Two silent correctness failures, both specific to this workload.** Cell
folding makes exact multiplets of 8–64 bands the normal case, and ARPACK
handles them badly:

- At scipy's default `ncv`, ARPACK returns **fewer copies than the true
  multiplicity** and never raises — measured 15 of 32, 7 of 16, 6 of 8.
  Every band above the multiplet is then shifted. `solve_lowest` now starts
  at `ncv = min(n, max(4k+1, 40))`, which reproduced the exact multiplicity
  in every case measured. **Do not lower it back.**
- Even with all copies present, ARPACK's multiplet vectors are converged as
  eigenvectors but **not orthogonalized against each other** (measured
  singular values 0.89–1.10 across an 8-fold multiplet, projector error
  6e-3). PDOS weights `|V_mn|²` and the `perturb_split` block
  `V_D† (dH/dk) V_D` both assume an orthonormal basis, so this is a small,
  silent, k-dependent error exactly at the degeneracies folding creates.
  `_orthonormalize_degenerate` does a QR per degenerate group (any
  orthonormal basis of the span is an equally valid eigenbasis) using the
  same 5-decimal grouping as `do_eigh.get_degeneracies`.

The dense `evr` branch has neither problem: `zheevr` returns an orthonormal
cluster basis by construction. That is also why `evr` and not `evx` (which
resolves clusters by inverse iteration) or `evd` (no subset support —
measured 1.11–1.31 s vs 0.53 s at n=1152).

**Bounds now in force**, in `solver.select_method`, deterministic in
`(n, nev)` only so it is hoisted out of the k-loop and printed once:

| condition | outcome |
|---|---|
| `method='dense'` / `'arpack'` | honoured verbatim (A/B validation) |
| `nev + guard >= n - 1` | dense (no Krylov room) |
| `nev + guard > dense_ratio*n`, `n <= dense_n_max` | dense |
| `nev + guard > dense_ratio*n`, `n > dense_n_max` | loud `NotImplementedError` naming both exits |
| otherwise | arpack, unchanged |

`dense_ratio` (default 1/8) and `dense_n_max` (default 4096) are `sparse_config`
keys; the defaults are policy values still to be calibrated by
`benchmarks/sparse_scaling`.

**What this costs**: by default the backend is capped at `nawf ≈ 4096`. Past it the
per-k `(n, n)` scratch stops being cheap and `select_method` raises **by
design** — `nx=3` (`nawf=9216`, 1.36 GB/rank, ~270 s/k) needs the
distributed bond list flagged below plus a distributed eigensolver
(ELPA/SLEPc). `nx=2` is the ceiling for this architecture, stated rather
than discovered.

**What it does not change**: rule 1. No `O(nk·nawf²)` tensor is created,
`test_sparse_mesh_parity.py` passes under **both** branches (it is
parametrized over them — at the base cell `auto` picks dense, so without
forcing, the ARPACK path would no longer be covered), and no phase,
folding, `Dnm` or `perturb_split` convention moved.

## Architecture map

```
PAOFLOW.py                the only driver. Each public method has a role
                          (sparse/dispatch.py): @sparse_override (engine method
                          or registered property), @sparse_shared, @sparse_base_cell
                          (densify -> dense body -> re-sparsify), or dense-only
                          with a reason in DENSE_ONLY_REASONS.
sparse/engine.py          SparseEngine: pipeline stages (pao_hamiltonian, doubling,
                          windows, interpolation, bands), the mesh/path pass
                          bookkeeping (_ensure_mesh, _run_path, fused(),
                          run_base_cell), and __getattr__ -> property REGISTRY.
sparse/properties/        one MeshProperty subclass per PAOFLOW property:
                          on_k(KPoint) + finalize(dc), class attributes needs /
                          products / interior / streaming / path / with_dnm /
                          uses_mesh. The module docstring has the "adding a
                          property" checklist. DenseBody = run the dense method
                          after the pass (band-diagonal properties).
sparse/kpoint.py          KPoint: one live k-point (E, V, hk, dhk, vel, delta) with
                          lazily cached degen, pksp, d2hk, delta_all, delta2,
                          project(op) -- shared by every consumer at that k.
sparse/mesh.py            run_mesh: the fused BZ pass. Stores E_k/velkp/deltakp
                          (+ d2Ed2k product) for the window; solves the full
                          spectrum when a consumer needs it.
sparse/bands.py           bands; prepare_path + run_path: the band-path pass
                          (sign=-1, cart=True) for path properties.
sparse/curvature.py       band_curvature_k: the six curvature components of one k-point.
sparse/operators.py       Sj/Lj as lists of CSR matrices (base cell, doubled by
                          doubling_attr_arry), projection_operator.
sparse/hamiltonian.py     SparseHamiltonian: bond list + fixed-pattern CSR plan;
                          assemble_derivatives(order=1|2, with_dnm) gives H, dH/dk
                          and d2H/dk_i dk_j from one product.
sparse/doubling.py        double_axis: O(nnz) index arithmetic replicating the
                          dense doubling kernel bond-for-bond.
sparse/solver.py          solve_lowest: dense/ARPACK dispatch, widened ncv,
                          degenerate-group re-orthonormalization, loud failure.
sparse/bridge.py          sparsify / densify: the single dense<->sparse boundary.
```

Dense kernels are reused through **per-k functions split out of them**
(the dense loop calls the same function, so its output is byte-identical):
`adaptive_widths`, `band_curvature_k_ij`, `velocity_products_k`,
`texture_k`, the `do_Hall` Berry/AC block functions, `eps_accumulate`,
`ree_intra_products_k`, the `linear_response_eqn*` per-k and block sums,
`ipr_k`, `accumulate_density_k`, the `do_topology`/`do_berry_curvature` path
functions, `site_weights_k`, the `do_berry_phase` Wilson-loop steps; and
their reduce/write halves. Band-diagonal dense bodies (Boltzmann stack,
`do_dos_adaptive`, `do_effective_mass`, `do_doping`, `do_fermisurf`) run
unchanged on the stored mesh arrays. Two per-k kernels are sparse-only on purpose:
`momentum_k` (the dense loop uses `momentum_in_eigenbasis` since Nov 2026; the sparse pass
resolves degenerate groups itself, `mesh._band_diagonal_velocities`) and
`sparse/curvature.band_curvature_k` (it assumes that joint eigenbasis).

## Conventions that took real debugging to establish

These are the load-bearing facts. Violating any of them produces results that
are *plausibly wrong* — off by conventions, not by crashes.

- **Bond list lives on the folded R grid** (`folded_R_triples`, components in
  `[-nk/2, nk/2-1]`), matching the dense FFT layout. Doubling operates on
  this raw representation so it stays bond-for-bond identical to the dense
  kernel.
- **Nyquist split at assembly time** (`_nyquist_split` in the plan): every
  bond with a component at `-nk/2` (even grids) is split into ±nk/2 halves.
  This is the bond-list equivalent of `utils.zero_pad` and gives three
  properties at once: (a) H(k) exactly Hermitian at every k, as the
  iterative solver requires; (b) values at original-grid k unchanged
  (provably — the phases coincide); (c) Fourier interpolation to *any*
  finer mesh is exact and free, so `interpolated_hamiltonian` is a pure
  attribute update. Do not "simplify" the split away.
- **Phase sign**: `sign=-1` everywhere, PAOFLOW's `H(k) = sum_R H(R) e^{-2 pi i k.R}` (the
  `fftn` convention of `do_double_grid`/`do_gradient` and, since 15 Nov 2026, of the dense band
  path; `sign=+1` gives `H(-k)`). The band path rotates `kq` to Cartesian with `b_vectors`,
  which `doubling_attr_arry` updates to the doubled cell since Nov 2026. Getting a sign wrong passes every
  *integrated* test (k and −k contributions pair up) and only an
  **index-wise** per-k comparison catches it. That is why
  `test_sparse_mesh_parity.py` exists; keep it.
- **The dense doubling kernel indexes by the negated folded coordinate**
  (`ix = -round(Rx*nk1)` — "the minus sign is due to the Fourier
  transformation"). In true folded coordinates the (0,1) block pulls
  `m = 2M − 1`, not `2M + 1`. The docstring of `doubling.py` records this;
  the parity test pins it.
- **Raw doubled H(R) is slightly non-Hermitian** — the dense kernel maps the
  self-paired Nyquist plane asymmetrically, and the dense pipeline mops it
  up with per-k Hermitizations and one-triangle `eigh` reads. The driver
  calls `hermitize()` once after doubling; bond-level `(B + B†)/2` with
  folded mirroring is *exactly* equal to per-k `(H(k)+H(k)†)/2` at every k.
  Off the original grid this differs from the dense band path (which
  diagonalizes the upper triangle of a non-Hermitian interpolant) by
  O(|H(R)| at the Nyquist shell) ≈ 10 meV here — a convention difference,
  not an error.
- **`nev = attr['bnd']`** on the mesh. Every dense band-diagonal consumer
  slices `[:, :bnd]`, and `do_dos_adaptive` hard-requires `E_k` with ≥ bnd
  bands, so this single choice makes the dense kernels drop in with zero
  edits. (Scale limit: see below.)
- **Gradient coefficient per bond**: `1j * (alat*Rcart_l + Dnm_l) * val`,
  replicating `do_gradient` including the diagonal tight-binding `Dnm`
  correction. `Dnm` lives per bond (`dnm`) and is the doubled cell's
  `tau_n - tau_m`: doubling keeps it on the diagonal blocks and shifts it by
  `-+ a_axis * alat` on the cross-replica ones, as the dense kernel does
  (both zeroed those pairs until 16 Nov 2026, a spurious band-diagonal
  velocity in doubled cells). The dense `Dnm`
  array is deleted at conversion; do not resurrect it (it is O(nawf²)).
- **`KPoint.pksp` is not one basis.** `momentum_k` rotates each degenerate group per
  direction (`perturb_split`), so products of two directions `p^a_nm p^b_mn` taken from it are
  wrong at degenerate k-points (Si orbital magnetization 1e-2 instead of 1e-11). A kernel that
  multiplies two directions takes `V† (dH/dk_l) V` itself (`orbital_moment`) or an operator
  pair through `perturb_split` (Hall, Kubo), as the dense `momentum_in_eigenbasis` does.
- **Position dipoles touch the consumers only.** `-i[H, d]` has no band-diagonal part and
  the dense pipeline takes widths and curvature before adding it, so `run_mesh` computes
  `velkp`, `deltakp` and `d2Ed2k` from the bare `dH/dk` and swaps `kp.dhk` just before
  `on_k`. Base cell only (the tables are those of the DFT cell); the path pass keeps the bare
  gradient, as the dense path kernels do.
- **ARPACK returns shift-inverted eigenpairs in transformed-problem order.**
  Sort *before* truncating the guard pairs (`_sorted_lowest`), or copies of
  a degenerate multiplet silently land in the discarded tail. This exact
  bug produced eV-scale errors that looked like "ARPACK can't do
  degeneracies". It can.

## Gauge zones of the dense conventions (2026-10-04)

Measured while porting the interband properties. They are properties of
the dense kernels, reproduced faithfully, not sparse defects:

- **Window-edge groups.** Since Nov 2026 `get_degeneracies(E, bnd)` keeps a group that reaches
  past `bnd` whole (it dropped it before, leaving its gauge to the eigensolver). Every kernel
  that stores `bnd` bands must take such a group: rotate it whole, and give the second-order
  subspace treatment only to groups entirely below `bnd` (`band_curvature_k_ij`,
  `sparse/curvature.py`). The group usually holds placeholder states at `shift`, and the two
  codes still pick different bases inside it: parity tests leave those k-points out
  (`_nondegenerate` in `test_sparse_mesh_parity.py`; 2 of 4096 on Si 16^3, 113 of 1728 on Fe).
- **The non-projectable block.** Bands above `bnd` sit at `shift`, exactly
  degenerate at every k and never rotated. Kernels summing over all `nawf`
  (Hall, AC conductivity, intra-band REE, linear response) pick up their
  gauge-random velocities (adaptive widths up to 1.6 eV on Fe) and in-block
  Berry pairs (denominator only delta^2). The sparse window kernels
  (conductivity, intra-band REE) sum over `bnd` and match dense truncated
  to the window and to the states below `shift - shift_margin`, which is all the sparse pass
  keeps.
- **Dense path kernels evaluate the raw base-grid HRs.** With one sign of R
  on the folded Nyquist plane that H(k) is non-Hermitian off the mesh (0.2
  eV on Fe 12^3, bands off by 47 meV). The bond list assembles the
  Nyquist-split H(k), which is the zero-padded interpolant; the path tests
  compare against dense kernels run on a hand-padded HRs.
- **Z2 via Pf/sqrt(det)** depends on the sqrt branch, i.e. on the gauge;
  on magnetic Fe even a 1e-13 change of H flips indicators.

## Fused properties and run data

Dense kernels read parameters from the run data (`attr['eminH']`,
`attr['response']`, `arrays['s_tensor']`...), set by the dense method just
before the kernel. Inside `fused()` all properties are set up before any
runs, so the engine records each property's writes
(`MeshProperty.remember_context`) and restores them before every `on_k` and
`finalize` (`activate`). New properties get this for free as long as they
set their run data in `__init__`/`prepare`.

## The environment landmine (read this even if you skip the rest)

numpy < 2.3 under **CPython 3.14** silently corrupts large (> ~256 KB)
function-local arrays: temporary-elision misfires with deferred reference
counts, so `odd = ~even` mutates `even` in place and `y = -x` negates `x`.
Small arrays are unaffected — which is why every unit test can pass while
production-size runs are garbage. `pyproject.toml` pins
`numpy>=2.3.2,<2.5` on 3.14. If results are structurally wrong and the code
looks provably correct, check `numpy.__version__` before doubting the code.

## Extending to a new property

The steps are the checklist in the module docstring of `sparse/properties/__init__.py`. Which
kind of property it is decides the sparse side:

| kind | example | sparse side |
| --- | --- | --- |
| band-diagonal, stored arrays only | transport, effective_mass, doping | `DenseBody` (+ `products`) |
| per-k eigenvector data, window bands | PDOS, textures, conductivity, dielectric | streaming `on_k` |
| interband sums over every state | Hall family, linear response | `needs = {'full_spectrum'}` |
| band path | berry_curvature, topology | `path = True` (`with_dnm` as the dense kernel) |
| base-cell H(R) transform | add_external_fields, cutting | `@sparse_base_cell` |
| base-cell H(R) transform that builds `Sj`/`Lj` | magnetic_field | engine method: operators first, then `run_base_cell` |
| reads the base-cell H(R) at its own k-points | arpes, find_weyl_points | `@sparse_base_cell(modifies=False)` |
| genuinely dense (DFT phonons, Z2Pack) | phonons, mirror_chern_number | `DENSE_ONLY_REASONS` |

Per-k objects that are not k-indexed (`(nk, esize)` scans gathered for a
bxsf, the previous point's occupied block of a Wilson loop) are allowed;
an array of per-k `(nawf, nawf)` matrices is not.

## Dos

- **Mirror the dense kernel's tail exactly** when a property writes files:
  same filenames, same normalization, same `comm.Reduce` placement, same
  rank-0-writes-others-pass-`None` pattern. The visual comparison notebook
  depends on files being drop-in comparable.
- **Keep every k-loop behind `scatter_full` over the dense k-ordering**
  (`n = k + j*nk3 + i*nk2*nk3`, folded values from `get_K_grid_fft_crystal`).
  Consumers then need only a final `Reduce`; per-k results stay
  index-comparable with dense arrays for parity tests.
- **Write the parity test first** (dense pipeline on the *base cell*,
  threshold 0, index-wise comparison) when adding anything that touches
  matrix elements. Base cell is cheap (nawf=18) and runs the real dense
  code in-test.
- **Treat exact degeneracies as gauge-free zones.** At near-exact
  degeneracies (folded bands make them common), per-band diagonal
  quantities are gauge-dependent in dense AND sparse; ARPACK's random start
  re-rolls the gauge every run. Parity tests must split into a strict set
  (no internal gaps < 1e-6) and a bounded set — see
  `test_sparse_mesh_parity.py`. Corollary: the adaptive width `δ ∝ |v|`
  can land near zero at such points, producing isolated 1/δ spikes in
  smeared spectra — in both codes, at different random energies. Known,
  documented in the notebook, not a defect to "fix" in sparse alone.
- **Print what the truncation costs.** `from_data_controller` computes a
  rigorous Gershgorin bound on the eigenvalue shift (`eig_bound` — valid at
  every k, survives doubling unchanged). Keep surfacing it; it is the
  honest answer to "is my threshold OK".
- **Warm-start along paths** (`v0` = previous k's ground vector) and reuse
  the assembly plan (`sph.plan` is built once; per-k work is a phase
  multiply + `add.reduceat`). If you add a new loop and it re-sorts the
  pattern per k, you rebuilt the old attempt's 4× slowdown.

## Don'ts

- **Don't call `doubling_HRs` from sparse code, ever.** The previous
  implementation "went sparse" by wrapping the dense doubling — which is
  precisely the allocation that OOMs. `double_axis` exists; it is
  bond-for-bond identical (tested).
- **Don't copy dense kernels.** Split them: extract the per-k body and
  the reduce/write tail into functions the dense kernel itself calls, and
  prove the dense outputs byte-identical: run every affected dense property
  from a `git worktree` of HEAD and from the working tree (example01 Si and
  spin-orbit example04 Fe, one property per process since some kernels call
  `comm.Abort()`) and `filecmp` every output file, `np.array_equal` every
  `.npz`. This
  *replaces* the earlier "don't touch dense code" rule; flags that change
  dense behaviour are still out.
- **Don't hermitize per k, per assembly, or inside doubling.** Hermitize is
  once, after all doubling, on the bond list. Doubling must stay raw to
  remain dense-parity-testable.
- **Don't "improve" numerical conventions** (phases, folding, `Dnm`,
  `perturb_split` ordering, `eigh` triangle choices) even where
  the dense choice looks arbitrary. Reproducibility against dense is the
  product; physical elegance is not. Document oddities in docstrings
  instead. When `develop` changes a dense convention, the sparse side follows it in the same
  change (the sign, `Dnm` and degenerate-group changes of Nov 2026 all surfaced as sparse test
  failures only at the next rebase).
- **Tolerances bound measured noise, nothing else.** End-to-end validation
  stays visual (the comparison notebook). Parity tests are index-wise and
  tight away from degeneracies; where a dense convention is gauge-sensitive
  they either feed both codes the same eigenvectors (mechanism tests) or
  compare integrated files to a tolerance a few times the *measured*
  difference, quoted in the test.
- **Don't let `nev` creep toward `nawf`** in new features. The V block is
  O(nawf·nev) and the dense scratch O(nawf²); `nev` is what decides
  whether this backend scales. The old enforcement — `solve_lowest`
  raising at `nev + guard >= n-1` — is now a *dispatch trigger* rather
  than an error, so the mechanism no longer punishes creep. The intent
  survives as a log warning (`solver.creep_warning`) whenever
  `nev/nawf > 0.5`, and the automatic window (below) pushes in the other
  direction by sizing `nev` from the property ranges. A new property with
  an energy range must declare it (`MeshProperty.energy_range`), or the
  window cannot cover it.
- **Don't name new test modules with basenames that exist elsewhere**
  (`test_hamiltonian.py` collides with the transport suite — pytest
  imports break). Prefix with `test_sparse_`.

## Known limits and the intended next steps

- **The solve window is derived per pass, never configured.**
  `SparseEngine._resolve_window(props)` runs before every mesh, path and
  band pass (`sparse/window.py` holds the arithmetic). Each property
  declares `energy_range() -> (emin, emax, kind, tmax)`; the margin kinds
  are `'smearing'` (4 x the largest adaptive width, estimated from the
  steepest band at the probe points), `'thermal'` (10 kB tmax) and
  `'none'`. From the bottom: `ehi = min(ptop, max(emax + margin))`; `nev`
  is the projectable count when `ehi` reaches `ptop` (no probe), the given
  `sparse_config['nev']`, or the probe count below `ehi` padded by
  `max(8, 2%)` and **capped at the projectable count** (no k-point has more
  real states; placeholders sit above `ptop`). `interior_window={}` wraps
  the ranges with the same margins; explicit bounds clamp the ranges to the
  window with the thermal margin only.
  Rules that keep it correct:
  - **Only a pass writes `attr['bnd']`** (`_ensure_mesh`, `_run_path`);
    `_resolve_window` never does, and `_execute` restores the mesh pass's
    `bnd` (`_mesh_bnd`) after a path pass with a different count.
  - **`prepare()` runs after the window is known** (`_execute`), because an
    interior clamp needs it; `merge_context` keeps what `prepare()` wrote
    restorable by `activate()` inside `fused()`.
  - **A new window with another `nev` invalidates stored products**
    (`d2Ed2k` of pass 1 no longer matches `E_k` of pass 2).
  - **The probe is collective and cached** until the Hamiltonian changes
    (`run_base_cell` resets it). Probe points are Γ, the zone corners
    (band extrema, for the count) and R3-sequence generic points (band
    velocities vanish on symmetry points; strided mesh points
    underestimated the steepest band by 34% on example01). It densifies
    H(k), so from the bottom above `dense_n_max` it is skipped and `nev`
    falls back to the projectable count.
  - **Doubling after any pass raises**: `doubling_attr_arry` doubles
    `attr['bnd']`, and a pass already sized it for the smaller cell.
  The window is **not** a route back to "strictly sparse": the fraction of
  the spectrum below a fixed `emax` is scale-invariant under folding
  (43/144 = 344/1152 ≈ 0.30), so `nev/nawf` stays put as the cell grows and
  stays above the 1/8 dispatch threshold. Only an interior window changes
  that, and that is a different solver.
- **`sparse.log` is key/value only.** Section titles, `log.field(key,
  value)` lines and one-line `log.warning(...)` (stdout and log, rank 0). No
  sentence explaining the algorithm goes in the log; put the reason in the
  docstring of the function that writes the line. A key appears only when
  it applies to the run. `test_sparse_log.py` enforces the format and a
  list of banned prose phrases.
- **The bond list is replicated per rank** (167 MB at 1e-4 here — fine).
  For very large systems, distribute bonds by rows and turn `assemble_hk`
  into a distributed operator for a distributed-memory solver. That is a
  large step; don't take it implicitly.
- **Threshold semantics**: applied once, at the base cell, in eV, on
  `|H_ij(R)|`; doubling commutes with it (pure rearrangement — tested).
  It does **not** preserve symmetry (Si 12³, 1e-3 alone: 55 meV
  degeneracy splitting); say so wherever it is offered.
  Example01 sweep: 1e-3 → ~160 meV band error, 1e-4 → ~10 meV (at the
  Hermitization-convention floor), 1e-5 → ~9 meV. At the small base cell
  the bond list can exceed the dense array in bytes (44 B/bond); the win
  is that it grows ×2 per doubling while dense grows ×4 — don't panic at
  base-cell stats.
- **`rcut` / `bond_order` semantics** (optional, **default `None`**): a
  *different* truncation axis — bond length
  `|alat·R_cart + tau_i - tau_j|` in Bohr, not matrix-element magnitude.
  **Exclusive with a positive `hopping_threshold`** (the `sparse_config` key;
  `SparseHamiltonian` and the archive still call it `threshold`;
  `config.resolve_threshold`):
  the geometric cut keeps whole atom-pair blocks by a length every
  space-group op preserves, while an element threshold keeps different
  elements of equivalent bonds and splits symmetry-protected degeneracies
  (Si 12³, `bond_order=36`: 0 → 22 meV split with 1e-3 on top, for −38%
  nnz). `threshold=None` therefore resolves to `0.0` with a geometric cut
  and to `1e-3` without one. The cut is folded into the same `keep` mask,
  so the printed `eig_bound` covers it. Like `threshold` it must be
  applied at the base cell: only there are the bond geometry and the orbital map
  well defined, and `double_axis` preserves bond lengths, so the cut carries over. The
  container carries a `_doubled` flag to make that ordering checkable.
  **The mask is symmetrized** over `(i,j,R) → (j,i,−R)`: off the Nyquist
  plane that pairing is automatic, but on the folded Nyquist plane `−R`
  maps onto `R` itself and the two partners have *different* lengths
  (tens of Bohr apart), so an unsymmetrized mask silently breaks
  Hermiticity. `test_sparse_cutoff.py` pins this.
  **An explicit `rcut` is snapped** (`shells.snap_cutoff`) to
  `start + distance_tol` of the outermost shell starting at or below
  `rcut + distance_tol`, so it always sits in a shell gap, as `bond_order`
  does. Without it, a radius on a shell distance keeps an arbitrary,
  rounding-decided part of that shell and breaks symmetry. Shells are
  grouped from the symmetrized `dist` array itself, not from `tau`, so the
  snap also holds in tests whose `Dnm` is unrelated to `tau`. `drop_report`
  carries both `rcut` (applied) and `rcut_requested`.
  **Representable range, enforced by raising** (no warning-only mode): an
  explicit `rcut` past `aliasing_safe_radius` raises (one grid value mixes
  several bonds there; `bond_order` can't get there since shells are only
  counted inside it). Separately, within `|tau_i - tau_j|` of that radius a
  bond can be stored at a longer image than its shortest, because the
  folding box picks `R` alone; `shells.misplaced_bond_length` finds the
  shortest such bond (27-image search, per distinct `Dnm` row) and a cutoff
  of *either* form reaching it raises, naming the max `bond_order`. Si fcc:
  hits the outer 1–3 shells on 3,4,5,7,9³; nothing inside the safe radius on
  even grids ≥ 6³ (12³: first one at 37.3 > 35.5 Bohr). The mask's
  `(j,i,-R)` partner min is exactly the shortest-image rule for the Nyquist
  plane, which is why Nyquist bonds are not counted as misplaced.
  No default value is blessed: the 20 Bohr figure that has been floated
  comes from slot-count geometry, not an accuracy sweep. Calibrate
  against `output/` at `nx=1` before adopting one.
- **LOBPCG / block solvers** could exploit warm-started blocks along the
  mesh, but scipy's LOBPCG needs `nev ≲ n/5` — only viable after the
  energy-window step lands. The solver interface (`solve_lowest`) is the
  seam; add backends there.

## How to verify any change

```bash
python -m pytest tests/unit/sparse -q          # component + parity tests
paoflow-gen --example si_sparse_backend -d /tmp/si_sparse   # the case: QE inputs + main.py (the large-cell tutorial)
cd /tmp/si_sparse                               # pw.x scf/nscf + projwfc.x, or copy tests/data/silicon/silicon.save
mpirun -n 4 python main.py                      # sparse -> output/, dense check of the same H(R) -> output_dense/
mkdir reference && cd reference && ln -s ../silicon.save .
cp <repo>/tests/integration/qe/example01/main.py . && python main.py && cd ..   # clean dense reference -> reference/output/
# open compare_sparse.ipynb, run all cells, judge visually
python main.py                                  # serial must reproduce the MPI files
```
