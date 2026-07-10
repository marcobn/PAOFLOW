# Project Overview

PAOFLOW is an open-source Python package for post-processing Density Functional Theory (DFT) calculations. It takes Kohn–Sham eigenstates from Quantum ESPRESSO or VASP and constructs tight-binding Hamiltonians in a Projected Atomic Orbital (PAO) basis — giving you a compact, _ab initio_ starting point for computing electronic, transport, and spectroscopic properties like band structures, density of states, and Berry curvature effects, all without empirical parameters.

This section is the developer-facing side of the documentation: installation notes, contribution guidelines, module overviews, and internal workflow descriptions.

## Design Philosophy

A few core ideas shape how PAOFLOW is built and maintained:

**PAO basis construction.** Everything starts from a converged DFT calculation. PAOFLOW projects Bloch wavefunctions onto atomic orbital bases to produce compact, _ab initio_ tight-binding Hamiltonians that are efficient enough for large-scale property calculations.

**Single source of truth.** Each module owns one well-defined responsibility. Shared state travels through the `DataController` dictionary, and every function that touches it documents exactly which keys it reads or modifies — so you always know where data comes from and where it ends up.

**Minimal external coupling.** The driver (rank-0 Python process) and MPI worker processes talk to each other exclusively through pickle files, keeping the orchestration layer simple and free of MPI dependencies.

**Testability first.** New functionality comes with integration tests. We keep them fast by using low-accuracy settings (reduced grids and k-points) and staging large inputs as release assets rather than committing them to the repository. See [Testing](testing.md) for the full picture.
