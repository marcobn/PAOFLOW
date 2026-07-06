---
name: sparse-paoflow
description: "Design and implement PAOFLOW sparse workflows. Use when: adding SparsePAOFLOW, src/PAOFLOW/sparse modules, sparse Hamiltonian/eigensolver/DOS/transport paths, avoiding dense materialization, reporting sparsity statistics, or deciding when whole-spectrum properties are feasible."
user-invocable: true
---

# PAOFLOW Sparse Implementation Design

## Scope

Use this skill when designing or implementing a purely sparse PAOFLOW path.

The near-term target is that the workflow in `examples/qe_examples/example01/main.py` can run from start to finish through a sparse driver with no dense manifestation of the sparse data path:

```python
paoflow.read_atomic_proj_QE()
paoflow.projectability()
paoflow.pao_hamiltonian()
paoflow.bands(ibrav=2, nk=2000)
paoflow.interpolated_hamiltonian()
paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing()
paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
paoflow.transport(emin=-12.0, emax=2.2)
paoflow.finish_execution()
```

The sparse implementation must be extensible, maintainable, and minimally disruptive to the existing dense implementation.

---

## Non-Negotiable Sparse Invariants

1. **No dense materialization in sparse workflows.**
   A sparse path must not call `np.asarray`, `.toarray()`, `.todense()`, dense `numpy.linalg` routines, dense `scipy.linalg` routines, or dense copies of sparse matrices unless the method is explicitly restricted to small developer tests and guarded away from production sparse execution.

2. **Do not silently fall back to dense PAOFLOW.**
   Sparse methods must either complete through sparse kernels or raise a clear `NotImplementedError`/domain-specific error that explains which sparse feature is missing and why dense fallback is not allowed.

3. **Sparse data contracts are explicit.**
   New sparse modules should pass typed sparse containers, metadata, and user-facing numerical controls explicitly. Avoid hidden dependence on dense array keys when a sparse object or sparse metadata object would make the contract clearer.

4. **Whole-spectrum requirements must be designed deliberately.**
   Before implementing an observable, classify whether it needs:
   - selected eigenpairs only,
   - energy-window or Fermi-window eigenpairs,
   - stochastic/trace-estimator access,
   - linear solves/Green's functions,
   - or the full spectrum.

   If the full spectrum is mathematically required, document that requirement and provide an explicit feasibility gate. Do not compute a dense full eigensystem as an implementation convenience.

5. **Memory behavior is part of correctness.**
   Treat accidental dense allocation as a correctness bug. Avoid algorithms whose temporary arrays scale as dense `(n_basis, n_basis, n_kpoints, n_spin)` objects in sparse mode.

---

## Architecture Direction

### Preferred Layout

Place new sparse implementation modules under:

```text
src/PAOFLOW/sparse/
```

The user-facing driver may live at `src/PAOFLOW/SparsePAOFLOW.py` so imports stay simple. Keep the driver thin and delegate sparse computation to modules under `src/PAOFLOW/sparse/`.

Use descriptive module names, for example:

- `driver.py` for shared sparse orchestration helpers.
- `containers.py` for sparse Hamiltonian, overlap, eigenpair, and metadata containers.
- `qe_projection.py` for sparse QE projection ingestion and projectability preparation.
- `hamiltonian_builder.py` for sparse PAO Hamiltonian construction.
- `interpolation.py` for sparse real-space/k-space interpolation.
- `eigensolvers.py` for selected-spectrum and energy-window eigensolvers.
- `observables.py` for DOS, adaptive smearing, gradients, momenta, and related observables.
- `transport.py` for sparse transport kernels.
- `stats.py` for sparsity and memory-reporting helpers.

Avoid `do_*` names for new sparse modules.

### Sparse Driver

A separate sparse driver is acceptable and preferred if it reduces cognitive overload:

```python
from PAOFLOW.SparsePAOFLOW import SparsePAOFLOW

paoflow = SparsePAOFLOW(
    savedir="silicon.save",
    outputdir="output",
    smearing="gauss",
    npool=1,
    verbose=True,
)
```

Keep the public method names aligned with `PAOFLOW.PAOFLOW` when the physics action is the same. The example workflow should remain readable as a sequence of high-level physics actions, not sparse implementation steps.

### Dense Architecture Boundary

Minimize changes to existing dense modules. Prefer adding sparse modules and a sparse driver over modifying `PAOFLOW.py`, dense `defs`, or dense data layout.

Allowed dense-side changes are limited to:

- package exports needed to import `SparsePAOFLOW`,
- shared utility hooks that are genuinely backend-agnostic,
- small refactors that remove duplication without changing dense numerical behavior,
- tests or examples that exercise the sparse path.

Do not redesign dense `DataController` or dense computation modules unless it is necessary for clean, maintainable sparse integration and there is no narrower sparse-side alternative.

---

## Data And Type Policy

### Sparse Containers

Use sparse-aware containers instead of overloading dense dictionaries with ambiguous values.

Recommended container fields include:

- sparse matrix format (`csr`, `csc`, `coo`, `bsr`, or `LinearOperator`),
- shape,
- dtype,
- number of nonzeros,
- density,
- estimated memory footprint,
- basis metadata,
- k-point and spin metadata,
- whether the object supports matvec, solve, selected eigensolve, or trace estimation.

Use SciPy sparse matrices and `scipy.sparse.linalg.LinearOperator` where appropriate. Prefer `LinearOperator` for objects whose explicit sparse matrix would still be too large.

### Storage Policy

Store sparse objects only when reused downstream, required for output/restart behavior, or expensive to recompute. Keep one-off intermediates local to the sparse backend function.

Never store both dense and sparse copies of the same object in sparse mode.

### Format Policy

Choose sparse formats based on operation:

- `csr_matrix`/`csr_array` for row slicing, matvec, and iterative eigensolvers.
- `csc_matrix`/`csc_array` for column-oriented solves or factorization workflows.
- `coo_matrix`/`coo_array` for assembly before converting to CSR/CSC.
- `bsr_matrix`/`bsr_array` for natural orbital block structure when blocks are dense and repeated.
- `LinearOperator` when matrix-free evaluation avoids materializing a large sparse object.

Document the chosen format at function boundaries when downstream performance depends on it.

---

## Method Design By Workflow Stage

### `read_atomic_proj_QE()`

Read QE projection data without constructing dense PAOFLOW tensors when sparse mode is active. If the QE input format is intrinsically dense for a small intermediate, isolate that conversion, quantify its size, and fail early when it would exceed sparse-mode memory policy.

### `projectability()`

Compute projectability using sparse reductions or streamed/block processing. Do not require all dense projection amplitudes to be resident simultaneously unless a bounded, documented input-stage exception is unavoidable.

### `pao_hamiltonian()`

Build a sparse PAO Hamiltonian representation directly. Avoid constructing dense `H(k)` or dense real-space Hamiltonian tensors as an intermediate.

### `bands()`

Bands along a path usually require selected eigenvalues at many k-points, not the whole spectrum. Use sparse Hermitian eigensolvers such as `eigsh`, shift-invert where feasible, or matrix-free solvers. Expose controls for band count, energy window, convergence tolerance, and maximum iterations when needed.

If the existing dense method implicitly returns all bands, the sparse API must make the selected-spectrum nature explicit or provide a feasibility gate for full-spectrum requests.

### `interpolated_hamiltonian()`

Interpolate without dense Fourier-transform tensors over all basis pairs. Prefer sparse real-space hopping lists, block-sparse structures, thresholded terms, or matrix-free `H(k)` assembly.

Track whether sparsity is preserved, improved, or degraded by interpolation.

### `pao_eigh()`

Do not implement sparse `pao_eigh()` as a dense full diagonalization. Decide whether the downstream workflow needs:

- all eigenpairs,
- selected eigenpairs near an energy window,
- eigenvalues only,
- moments/traces,
- or Green's-function solves.

For full-spectrum-only dense observables, raise a clear feasibility error or add an explicitly bounded small-system mode.

### `gradient_and_momenta()`

Avoid dense derivative matrices over all bands when possible. Prefer sparse derivatives, matrix elements within selected subspaces, or operator application against selected eigenvectors.

Document any observable that cannot be computed without a full eigensystem.

### `adaptive_smearing()`

Design smearing around available sparse spectral information. If adaptive smearing depends on nearest-band spacing or velocities, compute those quantities only for the selected spectral window required by subsequent observables.

### `dos()`

Do not require full diagonalization by default. Prefer methods such as selected-spectrum accumulation, kernel polynomial method, stochastic trace estimation, contour/Green's-function methods, or energy-window eigensolves.

Make the numerical approximation and convergence controls explicit in the sparse backend.

### `transport()`

Prefer sparse linear solves, recursive Green's functions, block-tridiagonal methods, or `LinearOperator` formulations. Avoid dense inversion and dense Green's-function matrices for large systems.

Expose physical/user-facing controls (`emin`, `emax`, `ne`, `temperature`, `bias`, tolerances) through the driver while hiding sparse solver plumbing inside backend modules.

---

## Sparsity Statistics And Output

Print concise sparsity statistics in the existing PAOFLOW output style when `verbose=True` or equivalent output control is active. Do not overload the output interface with large reports.

Useful statistics include:

- matrix or operator name,
- shape,
- nonzero count,
- density,
- sparse format,
- estimated sparse memory,
- estimated dense memory avoided,
- solver type,
- selected eigenpairs or energy window,
- iteration count and convergence status when available.

Prefer one short line per major sparse object or solver stage, for example:

```text
  Sparse H(k): csr shape=(240000, 240000), nnz=18.4M, density=3.2e-4, mem=294 MB, dense avoided=461 GB
  Sparse bands: eigsh window=[-12.0, 2.2] eV, eigenpairs=96, converged=96/96, iterations=143
```

Centralize formatting in a sparse stats/output helper so future sparse modules report consistently.

---

## Testing And Validation

Every sparse feature should include tests that protect the sparse contract, not only numerical values.

Minimum checks:

- Sparse workflow methods do not call dense fallback code.
- Sparse matrices remain sparse after each major stage.
- `.toarray()`, `.todense()`, dense `numpy.linalg`, and dense `scipy.linalg` are absent from production sparse modules unless explicitly guarded and justified.
- Small-system sparse results agree with dense PAOFLOW within documented tolerances.
- Large synthetic sparse tests validate memory scaling without requiring dense reference arrays.
- Whole-spectrum-only methods raise clear errors when requested outside a feasible small-system mode.

For the initial milestone, add or maintain a sparse example mirroring `examples/qe_examples/example01/main.py` and validate that it reaches `finish_execution()` through sparse modules only.

---

## Implementation Checklist

Before editing code for sparse PAOFLOW, confirm:

- [ ] The public workflow remains close to the existing `main.py` physics sequence.
- [ ] New sparse code is placed under `src/PAOFLOW/sparse/` unless there is a clear reason otherwise.
- [ ] A separate `SparsePAOFLOW` driver is used when it avoids dense-driver complexity.
- [ ] No sparse method silently dispatches to dense PAOFLOW.
- [ ] Each observable is classified by spectral requirement.
- [ ] Full-spectrum requirements are explicit and gated.
- [ ] Sparse object formats, shapes, `nnz`, density, and memory estimates can be reported.
- [ ] Existing dense architecture is unchanged except for minimal exports/shared utilities.
- [ ] Tests cover both numerical agreement on small cases and absence of dense materialization in sparse paths.

---

## Red Flags

Stop and redesign before continuing if a sparse implementation requires:

- a dense Hamiltonian tensor,
- full dense diagonalization,
- dense all-band velocity or momentum tensors,
- hidden conversion from sparse to dense,
- storing dense and sparse versions side by side,
- broad rewrites of `PAOFLOW.py` or dense modules,
- output changes that make dense and sparse user logs difficult to compare,
- or public examples that expose backend sparse plumbing instead of physics actions.
