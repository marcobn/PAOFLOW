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
2. **No `.toarray()` / dense `eigh` anywhere, at any size** (user
   requirement). Solves are strictly iterative (`solver.solve_lowest`:
   ARPACK shift-invert; sparse `splu` is fine — sparse direct ≠ dense).
   Dense arrays in *tests* as references are fine; in the pipeline, never.
3. **Allowed dense objects, exhaustively:**
   - band-diagonal arrays `E_k`, `velkp`, `deltakp` — O(nk·nev), k-scattered;
   - ONE per-k eigenvector block `V (nawf, nev)`, discarded before the next
     k-point;
   - the base-cell (pre-doubling) input stage: `Hks`/`HRs` at the small
     original `nawf` is inherently dense QE input processing; it is
     converted by `SparseHamiltonian.from_data_controller` (the **single
     sanctioned dense→sparse boundary**) and deleted immediately.
4. Failure is loud: guards raise `NotImplementedError`/`RuntimeError` with
   actionable messages. There is no silent dense fallback — a user who hits
   a wall must know it, not swap 40 GB.

## Architecture map

```
SparsePAOFLOW.py          driver; mirrors the dense PAOFLOW method names so an
                          example script is a 2-line diff. Delegates input
                          stages to a wrapped dense PAOFLOW; everything after
                          pao_hamiltonian is sparse. Unknown methods raise
                          NotImplementedError via __getattr__.
sparse/hamiltonian.py     SparseHamiltonian: the bond list (rows, cols, ridx,
                          vals, dnm) + fixed-pattern CSR assembly plan +
                          hermitize(). The only data structure in the backend.
sparse/doubling.py        double_axis: O(nnz) index arithmetic replicating the
                          dense doubling kernel bond-for-bond.
sparse/solver.py          solve_lowest: shift-invert eigsh, retry ladder,
                          loud failure.
sparse/bands.py           k-path eigenvalues (mirrors do_bands scaffolding).
sparse/mesh.py            THE core: one fused pass over the BZ mesh producing
                          E_k / velkp / deltakp and feeding per-k consumers.
sparse/pdos.py            PdosConsumer: the model for every new property.
```

Reused dense code, verbatim (never copied, never modified):
`kpnts_interpolation_mesh`, `get_K_grid_fft(_crystal)`, `get_R_grid_fft`,
`communication.scatter_full/gather_full`, `do_eigh.get_degeneracies`,
`utils.smearing`, `do_dos.do_dos_adaptive`, `do_pdos._build_orbital_prefixes`,
the whole Boltzmann stack (`do_transport`/`do_Boltz_tensors` via the existing
`'velkp' in arrays` branch in `PAOFLOW.transport`), `doubling_attr_arry`,
the `DataController` writers.

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

## The environment landmine (read this even if you skip the rest)

numpy < 2.3 under **CPython 3.14** silently corrupts large (> ~256 KB)
function-local arrays: temporary-elision misfires with deferred reference
counts, so `odd = ~even` mutates `even` in place and `y = -x` negates `x`.
Small arrays are unaffected — which is why every unit test can pass while
production-size runs are garbage. `pyproject.toml` pins
`numpy>=2.3.2,<2.5` on 3.14. If results are structurally wrong and the code
looks provably correct, check `numpy.__version__` before doubting the code.

## Extending to a new property: the decision tree

Ask, in order:

1. **Is the property a function of band-diagonal quantities only**
   (`E_k`, `velkp`, `deltakp`, occupations)?
   → Reuse the dense kernel verbatim, exactly like DOS and transport.
   No sparse code needed at all; just make sure `_ensure_mesh()` ran.
   Examples: Seebeck/σ/κ variants, carrier concentration, Fermi surface
   from `E_k`.

2. **Does it need per-k eigenvector information, consumed additively over
   k** (a BZ sum/integral)?
   → Write a **consumer** (the `PdosConsumer` pattern): a class with
   `on_k(ik, ispin, E, V, vel, delta)` that accumulates into a small
   fixed-size array, and `finalize(dc)` that does `comm.Reduce`,
   normalization, and file writing (mirror the dense kernel's tail
   line-for-line, including filenames and normalization constants).
   Register it in the `run_mesh` call. **Never store V; never return it.**
   Examples: spin texture (needs `Sj` expectation values per k), orbital
   projections, Berry-phase-free spectral functions.

3. **Does it need interband matrix elements** `⟨n|Ô|m⟩` (epsilon, Berry
   curvature, spin Hall, `deltakp2`-style interband smearing)?
   → This is the first genuinely new machinery. The rule: compute the
   `(nev, nev)` interband block **per k inside the consumer** as
   `V† (O_sparse @ V)` — a tall-skinny sparse-matvec block plus a small
   dense `(nev, nev)` product — accumulate the property, discard the block.
   `(nev, nev)` per k is allowed (it is O(nev²), not O(nawf²)); an array of
   them over k is **not**. The energy denominators / smearing come from `E`
   and `delta` already in hand. Degenerate subspaces: reuse the
   `perturb_split` convention (diagonalize the group block of the operator,
   per direction independently — see `mesh.py`).

4. **Does it need the Hamiltonian at perturbed/shifted k** (finite-difference
   Berry curvature, effective mass by curvature)?
   → Assembly is cheap and stateless: call `assemble_hk` at the shifted k
   inside the consumer. Do not build auxiliary grids of stored H(k).

5. **Does it genuinely need the full spectrum or full-BZ eigenvectors at
   once** (exact diagonalization features, some topology invariants)?
   → It does not fit this backend's contract. Say so with a loud
   `NotImplementedError` in the driver stub rather than approximating
   silently. The dense pipeline exists for systems that fit in memory.

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
- **Don't touch dense code.** The backend imports dense kernels; it never
  edits them. If a dense kernel almost-fits, write the sparse counterpart
  next to it (like `pdos.py`), don't add flags to the dense one.
- **Don't hermitize per k, per assembly, or inside doubling.** Hermitize is
  once, after all doubling, on the bond list. Doubling must stay raw to
  remain dense-parity-testable.
- **Don't "improve" numerical conventions** (phases, folding, `Dnm`
  zeroing, `perturb_split` ordering, `eigh` triangle choices) even where
  the dense choice looks arbitrary. Reproducibility against dense is the
  product; physical elegance is not. Document oddities in docstrings
  instead.
- **Don't add tolerance-gated validation.** The user's explicit decision:
  end-to-end validation is *visual*, via
  `examples/qe_examples/example01/compare_sparse.ipynb` only. Unit tests
  guard component correctness (mechanisms, not outputs); keep that split.
- **Don't let `nev` creep toward `nawf`** in new features. ARPACK cost and
  the V block both blow up; the guard in `solve_lowest`
  (`nev + guard >= n-1`) is a hard stop, not a suggestion.
- **Don't name new test modules with basenames that exist elsewhere**
  (`test_hamiltonian.py` collides with the transport suite — pytest
  imports break). Prefix with `test_sparse_`.

## Known limits and the intended next steps

- **`nev = bnd = nawf/2` is a milestone choice, not the destiny.** At scale,
  the per-k V block (`nawf × nev`) and ARPACK workspace are O(nawf²)/2. The
  designed fix: energy-window `nev ≪ nawf` sized from
  `emax + smearing margin`, with a per-k coverage guard (grow that k's
  solve only — never re-solve the whole mesh) and `attr['bnd'] = nev` so
  the dense consumers keep working. The transport driver already checks
  window coverage (`emax` vs the lowest computed top band); generalize that
  pattern.
- **The bond list is replicated per rank** (167 MB at 1e-4 here — fine).
  For very large systems, distribute bonds by rows and turn `assemble_hk`
  into a distributed operator for a distributed-memory solver. That is a
  large step; don't take it implicitly.
- **Threshold semantics**: applied once, at the base cell, in eV, on
  `|H_ij(R)|`; doubling commutes with it (pure rearrangement — tested).
  Example01 sweep: 1e-3 → ~160 meV band error, 1e-4 → ~10 meV (at the
  Hermitization-convention floor), 1e-5 → ~9 meV. At the small base cell
  the bond list can exceed the dense array in bytes (44 B/bond); the win
  is that it grows ×2 per doubling while dense grows ×4 — don't panic at
  base-cell stats.
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
