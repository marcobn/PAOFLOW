# ACBN0 Module

The ACBN0 module takes care of automatically determining Hubbard U (on-site) and V (intersite) correction parameters for DFT+U calculations. Rather than guessing or fitting U by hand, it computes them self-consistently from the electronic structure.

**Key references:**
- Agapito et al. (2015) — the ACBN0 formulas for U and J
- Lee & Son (2020) — the eACBN0 extension to intersite V parameters

## Architecture

The module runs across two execution contexts to keep things clean:

**Driver (rank-0 Python):** Orchestrates the workflow, manages file I/O, and launches QE codes. Deliberately kept free of `mpi4py` dependencies.

**Worker (independent MPI process):** Handles the heavy lifting — Hartree energy computations via auto-generated stubs.

Driver and worker communicate exclusively through pickle files, so there's no MPI entanglement in the orchestration layer.

## Core Classes

**`_HartreeKernel`** — The base MPI-aware class. Provides Coulomb integral computation via `PAOFLOW.defs.pyints.contr_coulomb` and scatter/allgather utilities for distributed calculations.

**`ACBN0_Hartree`** — Computes on-site U/J numerators using 8-fold symmetry reduction on ERI tensors.

**`eACBN0_Hartree`** — Extends the above to intersite V calculations with 4-fold symmetry optimization.

**`ACBN0` Driver** — Manages the self-consistent U loop, coordinating the DFT → PAOFLOW → Hartree evaluation cycle.

**`eACBN0`** — Subclass that adds pair enumeration, pair density-matrix construction, and joint U+V optimization.

## How It Works

Each self-consistent iteration goes through these steps:

1. QE SCF/NSCF calculation with the current `HUBBARD` card
2. Projector augmented-wave projection via `projwfc.x`
3. PAOFLOW bandstructure computation — H(k) and S(k)
4. Density-matrix construction and Hartree energy evaluation
5. Convergence check, with optional mixing

## Things Worth Knowing

**Keep U values as real floats.** On-site U values must remain real Python `float` throughout the workflow — passing complex numbers will break the QE input parser. The intersite V path does permit complex values (they carry Bloch phases), but on-site U does not.

**Performance optimizations (P1–P6):**
- P6 applies 8-fold symmetry to d-shell U evaluation, giving a 5.9× speedup
- P5 caches eigendecompositions across V-pair loops
- P2 vectorizes density-matrix operations

**Validation:** The module reproduces published benchmarks on Si and ZnO with ortho-atomic projection.

## When to Use It (and When Not To)

ACBN0 works well for FM insulators and half-metals. For itinerant FM metals, metallic screening tends to cause U overestimation. Near Stoner instabilities, fixed-U scanning is a more reliable approach than self-consistent determination.
