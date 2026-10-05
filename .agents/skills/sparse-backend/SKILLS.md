# Sparse backend — design handoff

This document explains how the sparse backend is built, why it is built that
way, and what to keep in mind when extending it to properties beyond the
example01 pipeline (bands, DOS/PDOS, Boltzmann transport). Read it before
adding anything.

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
| `nev + guard > n/8`, `n <= 4096` | dense |
| `nev + guard > n/8`, `n > 4096` | loud `NotImplementedError` naming both exits |
| otherwise | arpack, unchanged |

**What this costs**: the backend is capped at `nawf ≈ 4096`. Past it the
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
                          (sign=+1, cart=True) for path properties.
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
`momentum_k`, `adaptive_widths`, `band_curvature_k_ij`, `velocity_products_k`,
`texture_k`, the `do_Hall` Berry/AC block functions, `eps_accumulate`,
`ree_intra_products_k`, the `linear_response_eqn*` per-k and block sums,
`ipr_k`, `accumulate_density_k`, the `do_topology`/`do_berry_curvature` path
functions, `site_weights_k`, the `do_berry_phase` Wilson-loop steps; and
their reduce/write halves. Band-diagonal dense bodies (Boltzmann stack,
`do_dos_adaptive`, `do_effective_mass`, `do_doping`, `do_fermisurf`) run
unchanged on the stored mesh arrays.

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
- **Phase signs**: mesh/property assembly uses `sign=-1` (the `fftn`
  convention of `do_double_grid`/`do_gradient`); the bands path uses
  `sign=+1` with `kq` rotated to Cartesian by `b_vectors` — which doubling
  deliberately does **not** update, so the rotated product reproduces the
  dense band-folding behaviour. Getting a sign wrong passes every
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
  correction. `Dnm` lives per bond (`dnm`), zeroed on cross-replica bonds by
  doubling — exactly the dense `block_diag` semantics. The dense `Dnm`
  array is deleted at conversion; do not resurrect it (it is O(nawf²)).
- **ARPACK returns shift-inverted eigenpairs in transformed-problem order.**
  Sort *before* truncating the guard pairs (`_sorted_lowest`), or copies of
  a degenerate multiplet silently land in the discarded tail. This exact
  bug produced eV-scale errors that looked like "ARPACK can't do
  degeneracies". It can.

## Gauge zones of the dense conventions (2026-10-04)

Measured while porting the interband properties. They are properties of
the dense kernels, reproduced faithfully, not sparse defects:

- **Window-edge pairs.** `get_degeneracies(E, bnd)` drops a degenerate group
  that straddles `bnd`, so its gauge is whatever the eigensolver returned
  (zheevd dense, zheevr sparse). Through `get_degeneracies` on the rotated
  velocities this decides whether bands far below, with velocity zero by
  symmetry, form a "velocity-degenerate" block in `band_curvature_k_ij`:
  O(10%) curvature changes there (64 of 4096 k on Si). The dielectric sums
  agree to 3e-13 once those k are excluded.
- **The non-projectable block.** Bands above `bnd` sit at `shift`, exactly
  degenerate at every k and never rotated. Kernels summing over all `nawf`
  (Hall, AC conductivity, intra-band REE, linear response) pick up their
  gauge-random velocities (adaptive widths up to 1.6 eV on Fe) and in-block
  Berry pairs (denominator only delta^2). The sparse window kernels
  (conductivity, intra-band REE) sum over `bnd` and match dense truncated
  to the window.
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

Follow the checklist in `sparse/properties/__init__.py`: split the dense
kernel into a per-k function and a reduce/write function (byte-identical,
diffed), write a `MeshProperty`, mark the PAOFLOW method
`@sparse_override`, add a parity test. Which kind of property it is:

| kind | example | sparse side |
| --- | --- | --- |
| band-diagonal, stored arrays only | transport, effective_mass, doping | `DenseBody` (+ `products`) |
| per-k eigenvector data, window bands | PDOS, textures, conductivity, dielectric | streaming `on_k` |
| interband sums over every state | Hall family, linear response | `needs = {'full_spectrum'}` |
| band path | berry_curvature, topology | `path = True` (`with_dnm` as the dense kernel) |
| base-cell H(R) transform | add_external_fields, cutting | `@sparse_base_cell` |
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
- **Don't "improve" numerical conventions** (phases, folding, `Dnm`
  zeroing, `perturb_split` ordering, `eigh` triangle choices) even where
  the dense choice looks arbitrary. Reproducibility against dense is the
  product; physical elegance is not. Document oddities in docstrings
  instead.
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
  survives as a loud warning from `describe_method` whenever
  `nev/nawf > 0.5`, and as `energy_window()` pushing in the other
  direction. Neither is automatic: **sizing `nev` is still the caller's
  job.**
- **Don't name new test modules with basenames that exist elsewhere**
  (`test_hamiltonian.py` collides with the transport suite — pytest
  imports break). Prefix with `test_sparse_`.

## Known limits and the intended next steps

- **`nev = bnd = nawf/2` is a milestone choice, not the destiny.**
  `SparsePAOFLOW.energy_window(emin, emax, margin)` now sizes `nev` from
  the property range instead: it probes `count_below(ehi)` (zheevr
  `subset_by_value`, eigenvalues only) at 16 deterministic k-points, pads
  by `max(8, 2%)`, and sets `attr['bnd'] = nev`. Coverage is guarded per
  k-point and reduced once after the loop, so a short window raises with
  the exact `nev` to re-run at rather than truncating silently.
  **It must be called after `doubling_Hamiltonian()`** — `doubling_attr_arry`
  doubles `attr['bnd']` on every call — and the driver raises if the order
  is wrong.
  Two honest caveats. `attr['bnd']` **changes meaning** (from "projectable
  bands × cell multiplier" to "bands inside the window"), so `bands_*.dat`
  gains or loses columns and is not column-comparable across runs with and
  without a window; the downstream math is unaffected (`do_dos_adaptive`'s
  two `bnd` factors cancel, transport slices are `bnd`-independent). And
  the window is **not** a route back to "strictly sparse": the fraction of
  the spectrum below a fixed `emax` is scale-invariant under folding
  (43/144 = 344/1152 ≈ 0.30), so `nev/nawf` stays put as the cell grows and
  stays above the 1/8 dispatch threshold. For a DOS-from-`emin` run the
  dense branch is permanent. Only an *interior* window (shift-invert near
  E_F, transport only) would change that, and that is a different solver.
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
  **Exclusive with a positive `threshold`** (`config.resolve_threshold`):
  the geometric cut keeps whole atom-pair blocks by a length every
  space-group op preserves, while an element threshold keeps different
  elements of equivalent bonds and splits symmetry-protected degeneracies
  (Si 12³, `bond_order=36`: 0 → 22 meV split with 1e-3 on top, for −38%
  nnz). `threshold=None` therefore resolves to `0.0` with a geometric cut
  and to `1e-3` without one. The cut is folded into the same `keep` mask,
  so the printed `eig_bound` covers it. Like `threshold` it must be
  applied at the base cell: `double_axis` zeroes `dnm` on cross-replica
  blocks (deliberately — it replicates `block_diag(Dnm, Dnm)`), after
  which the true bond vector is unrecoverable. The container carries a
  `_doubled` flag to make that ordering checkable.
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
  snap also holds past the aliasing-safe radius and in tests whose `Dnm` is
  unrelated to `tau`. `drop_report` carries both `rcut` (applied) and
  `rcut_requested`.
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
cd examples/qe_examples/example01
python main.py                                  # clean dense reference -> output/
mpirun -n 4 python main_sparse.py               # sparse -> output_sparse/
# open compare_sparse.ipynb, run all cells, judge visually
python main_sparse.py                           # serial must reproduce the MPI files
```
