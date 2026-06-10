# Sparse Migration Hotspots and Status (Current)

This tracker replaces the earlier incremental log and reflects the code as of the current sparse orchestration integration.

## Scope

Primary reference workflow:

- `examples/qe_examples/example01/main.py`
- Run mode: `sparse=True`

Workflow sequence:

1. `pao_hamiltonian`
2. `bands`
3. `interpolated_hamiltonian`
4. `pao_eigh`
5. `gradient_and_momenta`
6. `adaptive_smearing`
7. `dos`
8. `transport`

## Prerequisites

- Sparse eigensolver mode depends on optional Python packages `petsc4py` and
  `slepc4py`.
- In package metadata, the sparse optional dependency group is named `sparse`.
- Recommended install path for these native dependencies is conda-forge:
  `conda install -c conda-forge petsc4py slepc4py`.
- Pip extras (`pip install .[sparse]`) are supported, but may trigger PETSc source builds
  when wheels are unavailable for the active platform/Python version.

## Current Sparse-State Summary

## 1) Sparse orchestration wrapper exists

- Dense mode now calls the existing dense functions directly from `PAOFLOW.py`.
- Sparse mode instantiates only `SparseBackend` in `PAOFLOW.__init__`.
- `defs/sparse/backend.py` now wraps only the sparse-native entry points that are
  actually dispatched from orchestration: sparse `H(k)` build, sparse `H(k)->H(R)`
  conversion, and sparse bands eigensolve.

Sparse mode therefore follows a narrower contract:

- dense-path symmetry adapters were removed,
- sparse-only dispatch remains where orchestration actually needs a sparse-native implementation,
- dense-only entry points continue to enforce their own preconditions directly.

## 2) H(k) and H(R) construction

- Sparse `H(k)` build is implemented and can store distributed local sparse blocks.
- Sparse `H(k) -> H(R)` boundary produces `SparseHRs` directly.
- Dense `HRs` is not stored in sparse mode during this boundary step.
- On a full FFT k-grid, the inverse transform now stays sparse-native at the storage level:
  - sparse `H(k)` blocks are gathered on rank 0,
  - active matrix coordinates are inverse-transformed in bounded FFT batches,
  - each `H(R)` block is sparsified immediately into `SparseHRs`.
- For single-spin wedge expansion, the symmetry expansion to the full FFT grid is now applied blockwise on rank 0 after sparse `H(k)` gather, without materializing a dense global `H(k)` tensor.
- Remaining dense fallback at this boundary:
  - multi-spin wedge expansion still uses the older dense symmetry adapter,
  - `symmetrize=True` still routes through the dense symmetry machinery.

Status:

- Functionally complete for sparse boundary.
- Full-grid `H(k) -> H(R)` no longer needs a dense global `H(k)` tensor.
- Parallelization for boundary accumulation remains rank-0 ownership.

## 3) Bands

- NSCF-grid sparse bands are implemented from `Hks_sparse`.
- Interpolated-path sparse bands are implemented from `SparseHRs`.
- An earlier unused interpolated sparse-bands prototype helper was removed; the
  active interpolated path still uses the temporary dense `SparseHRs -> HRs`
  bridge described below.
- Unsupported modes fail explicitly via flags rather than silent densification.
- Sparse eigensolver controls are now surfaced directly on `PAOFLOW.bands(...)`
  (`sparse_target`, `sparse_nbands`, `sparse_sigma`, `sparse_fermi_energy`,
  `sparse_tol`, `sparse_real_tol`, `sparse_return_eigenvectors`) so defaults are
  visible at the API call site rather than buried in sparse helper `.get(...)`
  fallbacks.

Status:

- Usable for both main sparse band modes.

## 4) Interpolated Hamiltonian (double grid)

- Sparse dispatch in `do_double_grid` exists.
- Sparse path can build enlarged-grid `H(k)` data from `SparseHRs` directly (`do_double_grid_sparse`).
- A dormant helper `SparseHRs.build_sparse_k_blocks(...)` still exists as a possible
  reusable builder for the same distributed `Hks_sparse` contract, namely
  `dict[int, csr_matrix]` keyed by `block_idx = ik * nspin + ispin`, but the current
  orchestration path assembles those local sparse blocks directly inside
  `do_double_grid_sparse`.

Status:

- Feature is available in sparse mode.
- Sparse path avoids dense `HRs` materialization.
- Sparse path now stores distributed interpolated `Hks_sparse` blocks and avoids dense
  `Hksp` materialization.

## 4.5) pao_eigh on interpolated sparse blocks

- `pao_eigh` now has a sparse-native eigensolve branch for workflows where `Hks_sparse`
  exists and dense `Hksp` is absent.
- The sparse branch diagonalizes owned `(k, spin)` sparse blocks locally and writes
  eigensolutions directly into the rank-local scattered-by-k-point layout used
  downstream; it no longer reconstructs dense global `E_k`/`v_k` arrays on rank 0.
- Degeneracy metadata (`degen`) is computed from local `E_k` slices, matching
  the dense-path consumer contract in `do_momentum`.
- The eigensolver uses a sparse full-spectrum backend (SLEPc via `slepc4py`/`petsc4py`)
  with no dense Hamiltonian materialization.

Status:

- Sparse interpolation -> `pao_eigh` avoids dense Hamiltonian bridges and root-side
  dense eigensolution reconstruction, and computes full-spectrum eigenpairs through
  sparse linear algebra kernels.
- Remaining dense boundaries are downstream tensor consumers (for example momentum/transport),
  not the `pao_eigh` Hamiltonian input or eigensolution distribution itself.

## 5) Derivatives and momentum (updated)

Observed current behavior:

- `do_gradient` has a sparse route (`defs/sparse/gradient.py`) that uses `SparseHRs`.
- Sparse gradient dispatch no longer requires `Hksp` FFT-grid dimensions to match the `SparseHRs` storage grid.
- This allows interpolated-grid derivative evaluation directly from `SparseHRs` and the active k-grid, avoiding the previous dense fallback trigger after `interpolated_hamiltonian`.
- In sparse interpolation mode without dense `Hksp`, the main workflow no longer stores either a full global `dHksp` tensor, a rank-local dense `dHksp` slice, or persisted sparse `dHks_sparse` blocks.
- Sparse no-bridge momentum now dispatches to `defs/sparse/momentum.py`, which streams bounded dense derivative batches directly from `SparseHRs` into `perturb_split` and stores only local band-diagonal `velkp` instead of dense `pksp`.
- `band_curvature=True` remains unavailable in this sparse no-bridge path because `do_band_curvature` still requires dense `Hksp`.
- Sparse no-bridge Hall now streams only the requested derivative directions from `SparseHRs`, projects them in-band one local k-point at a time, and accumulates Berry/AC outputs without stored dense `dHksp` or dense local `pksp` tensors.
- Hall AC work no longer triggers a second sparse derivative replay: anomalous Hall now updates Berry and AC accumulators from the same streamed projected operator pair, and spin Hall reuses the same sparse derivative stream while doing only the extra in-band projection that its AC algebra still requires.
- Sparse Hall post-processing now mirrors the dense vectorization pattern more closely: Berry occupation folding is applied energy-by-energy across the full local k-point slice, and AC Kubo denominators are accumulated in vectorized transition chunks instead of one energy at a time.

Status:

- Sparse derivative path is enabled for the sparse interpolated workflow in `main.py` without dense global `Hksp`, dense global `dHksp`, or rank-local dense `dHksp` storage.
- The downstream main workflow now streams derivative batches directly into the sparse momentum step and stores compact `velkp` instead of dense `pksp`.

## 6) Downstream consumers

- `adaptive_smearing` has a strict sparse path that consumes rank-local `velkp`
  directly, avoiding dense `pksp` materialization.
- `dos` uses eigendata and (optionally) adaptive widths.
- `transport` can reuse the compact `velkp` tensor when dense `pksp` is absent.
- `dielectric_tensor` now has a sparse no-bridge path for `from_wfc=None` that
  streams bounded dense band-space momentum matrices from sparse derivative
  batches, accumulates epsilon locally, and never stores dense global `pksp`.
- The top-level `PAOFLOW.transport()` entrypoint now follows that contract explicitly:
  it prefers dense `pksp` when present and otherwise passes the existing local
  `velkp` slice through to `do_transport` without rebuilding dense momentum data.

Status:

- These stages are operational in sparse workflow once `E_k`, `v_k`, and `velkp` are populated.
- Sparse adaptive smearing still emits dense `deltakp2` because Hall-style consumers
  expect the full pairwise band-width tensor, but it avoids the much larger dense
  complex `pksp` tensor.
- The remaining memory profile in the main workflow is now dominated by the dense
  width outputs and the streamed one-block work arrays inside `perturb_split`, not
  sparse Hamiltonian or stored derivative tensors.
- The remaining Hall runtime overhead is now primarily the dense band-basis algebra
  inside `perturb_split` and the size of the compact transition algebra in the AC step,
  rather than redundant sparse Fourier assembly, repeated sparse derivative streaming,
  or Python-level loops over local k-points and energies.

## Operation Support Matrix (Current Practical State)

| Operation                                | Dense path | Sparse path                                                                                                                 |
| ---------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| H(k) build                               | native     | native via `SparseBackend`                                                                                                  |
| H(k) -> H(R)                             | native     | sparse-native inverse FFT to sparse object; dense fallback only for some wedge/symmetry cases via `SparseBackend`           |
| doubling_Hamiltonian                     | native     | native sparse doubling from `SparseHRs` via `SparseBackend`; no dense `HRs` reconstruction                                  |
| Bands (NSCF)                             | native     | native via `SparseBackend`                                                                                                  |
| Bands interpolation                      | native     | temporary dense `HRs` bridge into existing parallel dense kernel                                                            |
| interpolated_hamiltonian                 | native     | sparse-direct path                                                                                                          |
| pao_eigh (interpolated/full-grid)        | native     | native when `Hks_sparse` is present; explicit precondition                                                                  |
| gradient_and_momenta after interpolation | native     | streamed sparse derivative batches plus compact `velkp`                                                                     |
| adaptive_smearing                        | native     | native from local `velkp` without dense `pksp`                                                                              |
| dos                                      | native     | works                                                                                                                       |
| transport                                | native     | works from local `velkp`; no dense `pksp` required                                                                          |
| topology                                 | native     | sparse no-bridge path supports band-path velocities, Berry curvature, and `eff_mass=True`; `spin_Hall`/Z2 remain dense-only |
| ACBN0                                    | native     | unavailable                                                                                                                 |

Note:

- The sparse wrapper is now intentionally limited to the orchestration methods that already need sparse-native dispatch.
- Dense-only methods such as `topology` are no longer described as backend-gated because they bypass the sparse wrapper entirely.
- Sparse doubling currently follows the dense orchestration ownership model: rank 0 applies the doubling update and MPI broadcast distributes updated arrays/attributes. A distributed sparse-doubling algorithm has not been added in this increment.

## Property Classification Beyond `example01`

`sparse=True` is not a blanket switch that automatically makes every later PAOFLOW
property sparse. The current sparse workflow only guarantees sparse-native behavior
through the example01 chain up to eigensolutions, local velocities, adaptive
widths, DoS, and transport. Additional properties fall into three practical
buckets based on the tensors they consume.

### A) Already compatible with the current sparse no-bridge outputs

These consumers work from eigendata and related compact downstream tensors such
as local `velkp` and `deltakp`; they do not require dense `HRs`, dense `pksp`,
or dense `dHksp`.

- `dos` / `pdos`
- `transport`
- `fermi_surface`
- `doping`
- `jdos`
- `ipr`
- `spin_texture` (after `spin_operator`, using `E_k`, `v_k`, and `Sj`)
- `site_projected_bands`

### B) Not automatically sparse: require dense momentum/derivative-style tensors

These consumers are not sparse-native in the current no-bridge workflow because
they still expect dense post-eigensolve tensors. They can in principle be made
to run behind an explicit dense bridge, but that is a separate implementation
choice and not part of the current sparse contract.

- `dielectric_tensor` / epsilon with `from_wfc=None`: sparse no-bridge streamed
  momentum path is available; `from_wfc='internal'` and `from_wfc='external'`
  still materialize dense `pksp`
- `rashba_edelstein`: requires dense diagonal `pksp`
- `spin_Hall`: sparse no-bridge path now streams sparse derivatives directly into the spin-current/Hall contractions; no dense `dHksp` bridge is required
- `anomalous_Hall`: sparse no-bridge path now streams sparse derivatives directly into the Hall contractions; no dense `dHksp` bridge is required
- `effective_mass`: depends on `d2Ed2k`, which currently comes from the dense `band_curvature` path and therefore still requires dense `Hksp`

### C) Not automatically sparse: require dense `HRs` or a dedicated sparse rewrite

These consumers still operate directly on dense real-space Hamiltonian storage
or their own dense Fourier/eigensolve kernels. Supporting them in sparse mode is
not just a matter of reusing `E_k`/`velkp`; they need either an explicit dense
`SparseHRs -> HRs` bridge at entry or a dedicated sparse-native implementation.

- `berry_phase`
- `topology` only for `spin_Hall` / Z2 style branches; Berry and effective-mass path on the interpolated band path now streams sparse derivative and second-derivative operators from `SparseHRs`
- `find_weyl_points`

Practical implication:

- Properties in bucket A should work once the sparse example01 chain has populated
  the expected eigendata on the active grid.
- Properties in bucket B are the next natural compatibility boundary if the goal
  is to extend post-eigensolve sparse workflows without rebuilding dense `HRs`.
- Properties in bucket C need a separate design decision: accept an explicit
  dense `HRs` bridge for those entrypoints, or introduce sparse-native Fourier /
  path-eigensolve implementations for them.

## Recommended Next Increment (Highest Priority)

Characterize the remaining dense downstream outputs after sparse eigensolve, then address the next compatibility boundary.

### Concrete step (single incremental change set)

1. Characterize memory/runtime of the remaining dense downstream outputs (`deltakp2` and streamed one-block `perturb_split` work arrays) on larger interpolated grids and define whether chunking or a sparse-aware projector is needed.
2. Extend sparse no-bridge support to `band_curvature`, or keep making the remaining dense-`Hksp` dependency explicit at its entry point.
3. Decide whether any additional sparse-only orchestration methods justify joining the `SparseBackend` wrapper, or should remain direct entrypoint logic.
4. Update this tracker with:

- whether interpolated-grid sparse derivative path is now always selected,

- sparse-wrapper coverage across orchestration methods,
- MPI ownership decision for derivative work partitioning and momentum tensors.

## Parallelization Notes (Current)

- Sparse `Hks_sparse` ownership is partitioned by contiguous k-point windows with all spins for each owned k-point.
- Sparse bands on NSCF grid use distributed local diagonalization and gather eigensolutions.
- Sparse interpolated bands now reuse the existing dense parallel `do_bands` implementation behind a temporary `SparseHRs -> HRs` bridge. This is an explicit performance-oriented exception because the band-path step is not the dominant sparse-memory hotspot in the main workflow.
- Sparse topology on the band path reuses the same contiguous path ownership as `bands()`: each rank streams only its local `dH/dk` and requested `d2H/dk^2` batches from `SparseHRs`, and only the compact projected outputs are gathered for file writing.

## 7) Topology on sparse no-bridge path

- `PAOFLOW.topology()` now dispatches to `defs/sparse/topology.py` when sparse mode keeps `SparseHRs` and dense `HRs` is absent.
- The sparse topology path reuses `SparseHRs` plus local path eigensolutions to stream bounded dense `dH/dk` batches into band-space momentum matrices, so it no longer fails with `KeyError: 'HRs'` in sparse example04-style workflows.
- `eff_mass=True` is supported through a bounded `SparseHRs.iter_local_d2Hdk2_batches(...)` helper that evaluates only the requested Cartesian second-derivative pair for the local path window.
- Sparse topology effective mass now mirrors the dense band-curvature degeneracy treatment: the second-derivative operator is passed through `perturb_split`, and the interband momentum sum is evaluated in the rotated basis for near-degenerate path states rather than the raw eigensolver basis.
- Sparse topology currently supports the Berry-curvature and effective-mass outputs written by `do_topology`; sparse `spin_Hall` / Z2 handling is still not implemented and remains an explicit compatibility boundary.

Status:

- Sparse band-path topology is operational for the example04-style `Berry=True`, `eff_mass=True`, `spin_Hall=False` workflow.
- Remaining gap is sparse spin-current/TRIM support for the `spin_Hall` / Z2 branch.
- Sparse `pao_eigh` on interpolated/full sparse blocks now keeps `E_k` and `v_k` local in the scattered-by-k-point layout instead of gathering dense eigensolutions on rank 0 and scattering them again.
- Sparse `H(k) -> H(R)` boundary currently gathers sparse `H(k)` blocks to rank 0, builds `SparseHRs` there, and then broadcasts the sparse `H(R)` object plus sparse-dispatch attributes once so later collective sparse stages follow the same branch on every rank.
- The full-grid inverse transform no longer materializes dense global `H(k)`; it runs in bounded coordinate batches on rank 0.
- Single-spin wedge expansion now applies symmetry blockwise on rank 0 after sparse `H(k)` gather, without a dense global `H(k)` tensor.
- Multi-spin wedge expansion and `symmetrize=True` still rely on the dense symmetry adapter.
- Sparse interpolated double-grid computation now broadcasts compact per-spin sparse `H(R)` payloads and assembles only rank-local enlarged-grid `H(k)` blocks on each rank.
- The new interpolated sparse block API is designed for local block assembly on each rank by
  reusing the existing contiguous `(k, spin)` partition; no additional MPI pattern has been
  introduced yet because orchestration has not been rewired.
- Sparse derivative stage is active for interpolated grids; each rank now streams only its local derivative batches using the same contiguous k-point ownership as the scattered eigensolutions.
- Sparse no-bridge momentum now stores only the local band-diagonal velocity tensor
  `velkp` for the main adaptive-smearing/transport workflow; dense `pksp` is not
  materialized on that path.
- The next memory-scaling hotspot in the same workflow is the dense pairwise width
  tensor `deltakp2`, which is still produced for compatibility with Hall-style
  consumers, plus the streamed dense one-block work arrays still used inside `perturb_split`.

## Explicit Open Items

1. Decide whether the remaining dense width output and one-block projector work arrays are sufficient for target system sizes or if chunked/sparse-aware paths are required.
2. Extend or document the sparse no-bridge boundary for `band_curvature` and any other remaining dense-only tensor consumers.

## 7) Hall-style consumers (updated)

- `spin_Hall` and `anomalous_Hall` now have sparse no-bridge implementations in
  `defs/sparse/hall.py`.
- The sparse Hall path reuses `SparseHRs.iter_local_dHdk_batches(...)` and
  `perturb_split(...)` to stream only the two derivative directions required by
  each tensor component.
- No dense global `dHksp` tensor is rebuilt, and no dense local `pksp` tensor is
  cached. Berry and AC contractions are accumulated immediately from the streamed
  band-space operator pairs.
- The AC branch intentionally replays the streamed sparse derivative pass instead
  of storing dense projected operator tensors. This keeps peak memory bounded at
  the cost of a second local contraction pass when `do_ac=True`.

Status:

- Sparse no-bridge Hall is operational for both anomalous and spin Hall outputs.
- Parallelization remains k-point distributed exactly as in the dense path;
  communication is limited to the existing compact spectrum reductions and the
  gathered `Omega(k, E)` field used for `bxsf` output.

3. Define distributed ownership/reduction for constructing/interpolating sparse `H(R)` tensors beyond the current broadcast-plus-local-assembly model.

## Short Rationale For Next Step

The previous derivative fallback trigger after interpolation has been removed from the sparse no-bridge workflow, and the dense-backend symmetry layer has been removed from orchestration. The next practical gap is controlling memory scaling for the remaining dense downstream work arrays and compatibility tensors that are still required by consumer contracts.
