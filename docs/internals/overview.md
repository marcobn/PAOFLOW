# Project Overview

PAOFLOW is an open-source Python package for post-processing Density Functional Theory (DFT) calculations. It processes Kohn–Sham eigenstates from Quantum ESPRESSO or VASP to construct tight-binding Hamiltonians in a Projected Atomic Orbital (PAO) basis, enabling computation of electronic, transport, and spectroscopic properties including band structures, density of states, Berry curvature effects, and more — without empirical parameters.

This wiki serves as developer-facing documentation covering installation notes, contribution guidelines, module overviews, and internal workflow descriptions. For tutorials and API reference, see the [ReadTheDocs site](https://paoflow.readthedocs.io).

## Design Philosophy

PAOFLOW is built around a few core principles:

**PAO basis construction.** Starting from a converged DFT calculation, PAOFLOW projects Bloch wavefunctions onto atomic orbital bases to produce compact, _ab initio_ tight-binding Hamiltonians suitable for large-scale property calculations.

**Single source of truth.** Each module owns one well-defined responsibility. Shared state passes through the `DataController` dictionary, which every function documents explicitly — listing the specific keys it adds or modifies.

**Minimal external coupling.** The driver (rank-0 Python process) and MPI worker processes communicate exclusively through pickle files, keeping the orchestration layer free of MPI dependencies.

**Testability first.** New functionality must be accompanied by integration tests using low-accuracy settings (reduced grids, k-points) for speed, plus reference output files for regression comparison. See [Testing](testing.md).
