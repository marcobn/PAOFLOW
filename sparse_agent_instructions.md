# Sparse Matrix Implementation Instructions for the Python Codebase

## Objective

Implement a sparse version of matrix-related functionality for the Python codebase in a way that reduces peak memory usage, avoids unnecessary dense matrix construction, and keeps the rollout safe and manageable. Sparse implementations must consider both memory constraints and parallel execution requirements; when parallelization cannot be preserved or added, the reason must be recorded explicitly and reflected in `sparse_hotspots.md`.

### Test modification policy for this workflow

- Do **not** add new tests.
- Do **not** update existing tests.
- Do **not** extend unit or integration coverage unless the user explicitly requests test changes in that task.

## Core Requirements

## Strict Reuse Rule: Do Not Reimplement Dense Logic Without Justification

Sparse implementations must **not** be introduced as design cleanups, refactors, or stylistic rewrites of existing dense functions.

### When a separate sparse implementation is NOT allowed

Do **not** create a separate sparse implementation if:

- The dense function already operates without constructing large dense matrices
- The dense function’s algorithm is already memory-safe for the target use case
- The logic can be reused directly without introducing dense intermediates
- The change is purely structural, aesthetic, or organizational
- The sparse version would be functionally identical aside from superficial differences

In these cases:

- **Call the existing dense function directly**
- Do **not** wrap it in a sparse-only alias
- Do **not** duplicate it under a new name
- Do **not** rewrite it for style consistency

### Positive requirement: justify every sparse implementation function in `defs/sparse`

Every sparse implementation function added in `defs/sparse` must include a clear justification in its docstring `Notes` section explaining:

- What dense behavior is being avoided (e.g., full matrix allocation)
- Why the dense implementation is not acceptable in this context
- What sparse-native strategy replaces it

If this justification cannot be clearly stated, the sparse implementation function should not exist.

### Reuse-first policy

Before implementing any sparse implementation function, explicitly evaluate:

1. Can the dense function be used directly without causing dense allocation?
2. Can the dense function be used as a subroutine inside a sparse workflow?
3. Can orchestration select the dense path safely even when `sparse=True`?

If the answer to any of the above is **yes**, prefer reuse over reimplementation.

### No “aesthetic sparse” implementations

Sparse functions must not exist for:

- naming symmetry alone
- API symmetry alone
- perceived architectural cleanliness
- minor refactors of dense logic
- “future-proofing” without a concrete memory issue

### Enforcement rule

Any sparse implementation function that:

- mirrors dense logic without eliminating dense allocation, or
- differs only in structure/style but not in memory behavior

is considered **invalid** and should be removed.

### Primary motivation

The sparse method should be implemented because, in some cases, memory may blow up when working with dense matrices. The implementation must therefore prioritize memory efficiency throughout the design.

### Dense reconstruction should be avoided

Whenever possible, do **not** rebuild the dense matrix during sparse workflows. Sparse code should operate directly on sparse representations rather than falling back to dense intermediates.

### Do not build sparse from dense

Do **not** create a sparse matrix from a dense matrix as part of the normal implementation path. That defeats the purpose of the memory constraint and introduces the exact failure mode this work is meant to avoid.

### Incremental development only

Development should happen in **small, incremental, manageable steps**. Each step should be independently reviewable.

### Preserve existing behavior

Avoid touching existing dense functions where possible. Put sparse-native logic in `defs/sparse` and keep the same base function name there (no explicit `_sparse` suffix in that module).

Example pattern:

- `build_matrix(...)` stays unchanged
- `defs/sparse/<module>.py` adds `build_matrix(...)` for sparse-native behavior

### Exception for orchestration layers

The main place where branching is acceptable is orchestration or dispatch code. There, it is acceptable to introduce logic such as:

```python
if sparse:
    return build_matrix_impl(...)
return build_matrix(...)
```

Where `build_matrix_impl` is a local orchestration alias, for example:

```python
from .defs.sparse.my_module import build_matrix as build_matrix_impl
```

Outside of orchestration, avoid retrofitting existing implementations unless there is a compelling reason.

### Avoid descriptive DataController flags

Do **not** add new entries to `data_attributes` or the `DataController` just to describe an internal sparse implementation detail when that detail is already implied by existing arrays, shapes, or control flow.

Examples of flags to avoid unless they are truly required downstream:

- storage-description flags such as `*_storage`
- provenance labels such as `*_source`
- one-off bookkeeping markers that no later code reads

Only add a new runtime attribute when at least one of the following is true:

- downstream code must branch on it for correctness
- restart serialization must preserve it for correctness
- a user-facing configuration or externally consumed report requires it and the value cannot be derived cheaply

If none of those conditions apply, keep the state local to the function or derive it from the existing arrays at the point of use.

## Implementation Principles

### 1. Sparse-first design

New sparse functionality should be designed from the ground up to work from sparse inputs, sparse intermediate representations, or streamed / coordinate-style construction paths.

Preferred patterns include:

- coordinate accumulation (`row`, `col`, `value`)
- CSR/CSC/COO-based construction
- blockwise or streamed assembly
- lazy evaluation where appropriate

Avoid patterns that:

- allocate full dense arrays as intermediates
- convert existing dense outputs into sparse outputs afterward
- rely on temporary dense masks or dense helper matrices unless proven safe

### 2. Parallel structure with dense implementation

Where a dense implementation already exists, the sparse counterpart should mirror:

- naming
- argument ordering
- return semantics, where reasonable
- error handling conventions

This makes adoption, and review substantially easier.

### 3. Small-scope changes

Each pull request or implementation step should do one narrow thing only.

Good step sizes:

1. add sparse data structure helpers
2. add one sparse implementation in `defs/sparse` for one matrix builder
3. add orchestrator dispatch behind `if sparse:`
4. add next sparse implementation

Bad step sizes:

- rewriting the whole matrix stack at once
- mixing sparse migration with unrelated refactors
- changing public APIs broadly in the same change set

### 4. Explicit compatibility boundaries

For every new sparse implementation function in `defs/sparse`, define clearly:

- accepted input types
- output sparse format
- whether format conversions occur
- whether downstream consumers can handle the sparse output directly

If a downstream step cannot yet consume sparse output, do not silently densify deep in the pipeline. Instead:

- document the limitation clearly
- stop at the sparse boundary, or
- add a separate explicit adapter if absolutely necessary

## Coding Standards for All New Sparse Functions

Every new function added for this work must satisfy all of the following:

### Module file naming convention is mandatory

Do **not** create flat module names like `backend.py` or `hr.py` under these packages. The canonical sparse modules already follow this rule: `defs/sparse/backend.py` and `defs/sparse/hr.py`.

Compatibility shims that expose old import paths are **not permitted**. Update every import site directly.

### Type hints are mandatory

All parameters and return values must be type hinted.

### NumPy-style docstrings are mandatory (physicist-oriented)

Each function must include a NumPy-style docstring, but the primary audience is non-software specialists (e.g. physicists) rather than programmers.

Docstrings must prioritize:

- Mathematical meaning over implementation details
- Clarity of inputs/outputs in terms of physical or mathematical objects
- Explicit use of equations wherever helpful
- Array/tensor dimensions and their interpretation

Avoid overly technical software language unless absolutely necessary.

### Required content for docstrings

Each docstring must include:

#### 1. High-level description (conceptual)

Explain what the function computes in plain scientific terms.

#### 2. Parameters (with physical meaning + shape)

For each parameter, include:

- Meaning in mathematical/physical terms
- Expected shape (explicitly)
- Any structure (e.g. symmetry, sparsity pattern)

#### 3. Returns (with interpretation + shape)

Clearly describe:

- What mathematical object is returned
- Its shape
- How it relates to the inputs

#### 4. Notes section (mathematics-first, not code-first)

The `Notes` section must explain the operation using:

- Equations where appropriate
- Mapping between arrays and mathematical objects
- How sparsity arises physically or mathematically
- Why this avoids dense memory usage

#### 5. Optional: dimension summary (strongly encouraged)

When helpful, include a compact summary of dimensions.

### What to avoid

Docstrings should not:

- Focus primarily on data structures (CSR/COO/etc.) without context
- Explain trivial Python behavior
- Read like internal developer notes
- Assume the reader is familiar with implementation details

Each function docstring must include a `Notes` section explaining the math, algorithm, or construction approach implemented by that function.

This section should cover things like:

- what sparse format is being built or consumed
- why the algorithm avoids dense allocation
- mathematical interpretation of the transformation
- complexity or tradeoffs where relevant

## Parallelization requirements for sparse implementations

The dense codebase is designed to run in parallel, and this must remain a first-class consideration for all sparse work. Sparse implementations are being introduced specifically for the large-system cases where dense memory use becomes prohibitive, and these are also the cases where parallel execution is often most important for performance.

### General requirements

- For every new sparse implementation in `defs/sparse`, explicitly evaluate whether the dense counterpart currently relies on parallel execution.
- Preserve parallelization where it is still valid and beneficial for the sparse formulation.
- Do not silently drop parallel execution just because the sparse implementation is new or more complex.
- At the same time, do not force parallelization when it is not mathematically sound, not memory-safe, or unlikely to provide real benefit.

### What to assess for each sparse implementation function

For every sparse function or sparse orchestration path, document and reason about:

1. **Whether parallelization is possible**
   - Can the work be split into independent chunks safely?
   - Are there data dependencies, shared writes, ordering constraints, or reductions that complicate or prevent parallel execution?

2. **Whether parallelization is beneficial**
   - Does the sparse algorithm have enough work per task to amortize overhead?
   - Will parallelization improve runtime for the expected large sparse cases?
   - Could parallelization increase memory pressure enough to undermine the purpose of using sparse methods?

3. **Whether parallelization changes the algorithmic design**
   - Does the sparse version require a different work partitioning strategy than the dense version?
   - Should the algorithm be organized around rows, blocks, entries, graph neighborhoods, or another sparse-native decomposition?
   - Are there sparse formats or data access patterns that make parallelization easier or harder?

### When parallelization is not possible or not appropriate

If a sparse implementation cannot be parallelized, or should not be parallelized, this must be stated explicitly in the code comments/docstrings/Notes section and in `sparse_hotspots.md`.

Acceptable reasons include, for example:

- unavoidable write conflicts or synchronization costs
- strong sequential dependencies in the algorithm
- excessive overhead relative to useful work
- sparse access patterns that destroy locality or introduce too much contention
- memory amplification from per-worker intermediates
- numerical or correctness risks introduced by parallel execution

Do not leave the absence of parallelization unexplained.

### Implementation guidance

- Prefer sparse-native parallel strategies rather than reproducing dense parallel patterns mechanically.
- Avoid designs that rebuild dense intermediates just to recover existing parallel behavior.
- Avoid converting dense outputs to sparse or sparse inputs to dense solely to fit an old parallel implementation.
- When orchestrating, it is acceptable to branch explicitly, for example:
  - `if sparse: ...`
  - `else: ...`
- Keep changes incremental and manageable. Parallelization support for sparse code should also be developed in small steps.

### Documentation requirements for new sparse functions

Every new sparse-related function must:

- live in `defs/sparse` and use the base function name (no `_sparse` suffix in sparse modules) unless it is orchestration logic
- include full type hints
- include NumPy-style docstrings
- include a `Notes` section that explains the math and/or algorithm
- state the parallelization strategy, if any
- state clearly when parallelization is intentionally not used

### Tracker maintenance: sparse_hotspots.md

`sparse_hotspots.md` is the running tracker for sparse development and must be updated whenever relevant.

Update `sparse_hotspots.md` when:

- a new sparse implementation function in `defs/sparse` is added
- a sparse orchestration path is added or changed
- a hotspot is completed
- a hotspot is partially implemented
- a blocker or open design question is discovered
- a decision is made about parallelization support
- it is determined that parallelization is not possible or not worth doing for a given hotspot
- the next recommended implementation step changes

Each relevant tracker update should capture, as applicable:

- what was implemented
- what remains to be done
- whether parallelization was preserved, added, deferred, or ruled out
- why parallelization is or is not feasible
- any memory/performance implications
- the next small incremental step to take

The goal is to keep sparse development easy to resume and to ensure the next steps are always clear.

## Recommended Function Template

```python
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


def build_matrix(
    rows: Sequence[int],
    cols: Sequence[int],
    values: Sequence[float],
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    """Build a sparse matrix directly from coordinate data.

    Parameters
    ----------
    rows : Sequence[int]
        Row indices for nonzero entries.
    cols : Sequence[int]
        Column indices for nonzero entries.
    values : Sequence[float]
        Values associated with each nonzero entry.
    shape : tuple[int, int]
        Shape of the output matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix assembled directly from coordinate inputs.

    Notes
    -----
    This function constructs the sparse matrix directly from coordinate
    triplets without ever materializing a dense intermediate. The algorithm
    corresponds to COO-style accumulation followed by conversion to CSR,
    which is efficient for assembly and suitable for downstream arithmetic.
    This avoids the memory blow-up risk that would occur if a dense matrix
    were allocated first and sparsified afterward.
    """
    coo = sparse.coo_matrix((values, (rows, cols)), shape=shape)
    return coo.tocsr()
```

## Required Development Strategy

### Phase 1: Identify dense hotspots

Start by identifying the parts of the codebase where dense matrix construction is most likely to cause memory blowups.

For each hotspot, document:

- function name
- current dense output shape
- expected sparsity pattern
- whether the matrix is assembled directly or derived from other structures
- whether downstream code can already accept sparse matrices

### Phase 2: Introduce sparse builders, not conversions

For each hotspot, implement a direct sparse builder rather than a dense builder followed by conversion.

Preferred question:

- "How do we generate the nonzero structure directly?"

Avoid this question:

- "How do we convert the current dense matrix to sparse later?"

### Phase 3: Add orchestration switch

Only after the sparse implementation in `defs/sparse` is validated should orchestration code be updated to dispatch between dense and sparse paths.

Example:

```python

def assemble_system(..., sparse: bool = False):
    from .defs.sparse.system import assemble_system as assemble_system_impl

    if sparse:
        return assemble_system_impl(...)
    return assemble_system(...)
```

This keeps the behavioral diff isolated and easy to review.

### Phase 4: Expand coverage gradually

After one sparse twin is stable, continue function-by-function. Do not attempt a full migration in one pass.

## Review Checklist

Before merging any sparse implementation step, confirm all of the following:

- The implementation does not construct a dense matrix internally unless explicitly justified and documented.
- The implementation does not convert dense output to sparse as its main strategy.
- Existing functions remain untouched except for limited orchestration logic where needed.
- Sparse behavior is implemented in `defs/sparse` with base-name functions (no `_sparse` suffix required in sparse modules).
- Any new `data_attributes` or `DataController` flags have a concrete downstream reader or restart-correctness requirement; write-only descriptive flags were not added.
- All new functions are fully type hinted.
- All new functions use NumPy-style docstrings.
- Every new function has a `Notes` section explaining the math or algorithm.
- The step is small enough to review safely.
- Ensure that the sparse related functions are placed in defs/sparse/ with appropriate naming conventions.

## Practical Guidance for Claude Code or Any Coding Agent

When implementing this work, follow these instructions strictly:

1. Do not rewrite existing dense functions unless the change is confined to orchestration.
2. Add sparse implementations under `defs/sparse` using base function names instead of mutating existing dense logic.
3. Build sparse matrices directly from sparse-relevant inputs or coordinate data.
4. Never allocate a dense matrix just to convert it into sparse form.
5. Avoid hidden densification in helper functions.
6. Add type hints and NumPy-style docstrings to every new function.
7. Include a `Notes` section in every docstring explaining the math or algorithm.
8. Call out any downstream limitation that still requires dense behavior rather than silently converting.
9. Prefer explicitness, memory safety, and incremental delivery over clever refactoring.
10. Do not add descriptive `data_attributes` flags unless downstream code or restart correctness actually depends on them.

## Requested Guidance to Include Verbatim in Spirit

The following project intent should govern the implementation:

- I want to implement the sparse method because in some cases, memory may blow up.
- Therefore, whenever possible I want to avoid rebuilding the dense matrix.
- I also don't want to create a sparse matrix from a dense matrix since that would defeat the purpose of memory constraints.
- I want the development to happen in small incremental manageable steps.
- I would like to avoid touching existing dense functions and instead add sparse implementations in `defs/sparse` using base names.
- Except in the case of orchestrating where one can then use something like `if sparse: ...`.
- I want every new function to be typehinted, have NumPy-style docstrings, and a `Notes` section that explains the math or algorithm being implemented in the function.

## Suggested Prompt Stub for Implementation Work

```text
Implement the sparse counterpart for the targeted matrix-building path.

Requirements:
- Do not modify the existing dense function except in orchestration if needed.
- Add a sparse implementation function in `defs/sparse` using the base function name (no `_sparse` suffix in sparse modules).
- Do not construct a dense matrix as an intermediate.
- Do not create a sparse matrix by converting from a dense matrix.
- Keep the change small and manageable.
- Add full type hints.
- Use NumPy-style docstrings.
- Include a Notes section explaining the math or algorithm.
- Prefer direct sparse construction using coordinate, CSR, CSC, or other sparse-native approaches.
- DO NOT add any tests for now.
```
