# ACBN0 Module

The ACBN0 module automatically determines Hubbard U (on-site) and V (intersite) correction parameters for DFT+U calculations using a self-consistent approach.

**References:**
- Agapito et al. (2015) — ACBN0 formulas for U and J
- Lee & Son (2020) — eACBN0 extension to intersite V parameters

## Architecture

The module uses two execution contexts:

**Driver (rank-0 Python):** Orchestrates workflows, manages file I/O, and launches QE codes. Free of `mpi4py` dependencies.

**Worker (independent MPI process):** Computes Hartree energies via auto-generated stubs.

Communication between driver and worker occurs exclusively through pickle files.

## Core Classes

**`_HartreeKernel`** — Base MPI-aware class providing Coulomb integral computation via `PAOFLOW.defs.pyints.contr_coulomb` and MPI scatter/allgather utilities for distributed calculations.

**`ACBN0_Hartree`** — Calculates on-site U/J numerators using 8-fold symmetry reduction on ERI tensors.

**`eACBN0_Hartree`** — Extends to intersite V calculations with 4-fold symmetry optimization.

**`ACBN0` Driver** — Manages the self-consistent U loop, orchestrating DFT → PAOFLOW → Hartree evaluation cycles.

**`eACBN0`** — Subclass adding pair enumeration, pair density-matrix construction, and joint U+V optimization.

## Computational Flow

Each self-consistent iteration:

1. QE SCF/NSCF calculations with current `HUBBARD` card
2. Projector augmented-wave projection via `projwfc.x`
3. PAOFLOW bandstructure computation — H(k), S(k)
4. Density-matrix construction and Hartree energy evaluation
5. Convergence check with optional mixing

## Implementation Details

**Type discipline:** On-site U values must remain real Python `float` throughout — complex numbers cause QE parser failure. The intersite V path permits complex values (which carry Bloch phases).

**Performance optimizations (P1–P6):**
- P6: 8-fold symmetry on d-shell U evaluation → 5.9× speedup
- P5: Eigendecomposition caching across V-pair loops
- P2: Vectorized density-matrix operations

**Validation:** Reproduces published benchmarks (Si, ZnO) with ortho-atomic projection.

## Physical Applicability

ACBN0 works well for FM insulators and half-metals. It overestimates U in itinerant FM metals where metallic screening dominates. Near Stoner instabilities, fixed-U scanning is preferable to self-consistent determination.
