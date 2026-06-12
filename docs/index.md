# PAOFLOW

**PAOFLOW** is an open-source Python framework for constructing and operating on *ab initio* tight-binding Hamiltonians built from the projection of DFT wavefunctions onto atomic orbital (PAO) bases. Starting from a converged Quantum ESPRESSO or VASP calculation, PAOFLOW delivers a compact Hamiltonian that drives a wide range of electronic, topological, optical, and transport property calculations — without empirical parameters.

---

## What is PAOFLOW?

PAOFLOW reads the output of a plane-wave DFT code, projects the Bloch eigenstates onto a compact PAO basis, and exposes a high-level Python API for property calculations. Because the PAO Hamiltonian is orders of magnitude cheaper to diagonalize than the full DFT problem, PAOFLOW enables dense **k**-point sampling and fine spectral resolution at low computational cost.

Key capabilities include:

| Domain | What PAOFLOW computes |
|---|---|
| **Electronic structure** | Band structures, DOS (total & projected), Fermi surfaces |
| **Optical & dielectric** | Complex dielectric tensor ε(ω), optical conductivity, JDOS |
| **Transport** | Electrical conductivity, Seebeck coefficient, electronic thermal conductivity |
| **Topology** | Berry curvature, anomalous Hall conductivity, Z₂ invariants, surface states |
| **Spin & magnetism** | Spin Hall conductivity, spin texture, SOC Hamiltonians |
| **Model Hamiltonians** | Slater–Koster TB models, Kane–Mele, custom lattice models |
| **ACBN0** | Self-consistent Hubbard U and U+V |

---

## Who is this for?

PAOFLOW is aimed at **computational materials scientists** who:

- Use Quantum ESPRESSO or VASP for their DFT calculations and want post-processing beyond the standard tools
- Need band structures, transport coefficients, or topological invariants on dense k-meshes without re-running expensive DFT
- Work on high-throughput screening campaigns and need a scriptable, MPI-parallel post-processing layer
- Want to prototype tight-binding models grounded in first-principles data

Basic familiarity with DFT concepts (plane-wave basis, pseudopotentials, Brillouin zone sampling) and Python scripting is assumed.

---

## Start here

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Get PAOFLOW installed in a few commands.

- :material-rocket-launch: **[Quickstart](quickstart.md)**

    Run your first PAOFLOW workflow on a Silicon example in minutes.

</div>

---

## Main documentation sections

| Section | Contents |
|---|---|
| [Installation](installation.md) | pip install, conda environment, optional dependencies, verification |
| [Quickstart](quickstart.md) | Step-by-step first workflow using the Silicon QE example |
| [Examples](examples/index.md) | Overview of all available QE and VASP example workflows |
| [API Reference](api/index.md) | Full Python API generated from source docstrings |

---

## License & citation

PAOFLOW is distributed under the **GNU General Public License v3**.
Copyright 2016–2026 Marco Buongiorno Nardelli and the PAOFLOW Development Team.

If you use PAOFLOW in published work, please cite:

> F.T. Cerasoli *et al.*, *Advanced modeling of materials with PAOFLOW 2.0*, Comp. Mat. Sci. **200**, 110828 (2021).

> M. Buongiorno Nardelli *et al.*, *PAOFLOW: A utility to construct and operate on ab initio Hamiltonians…*, Comp. Mat. Sci. **143**, 462 (2018).

See the [repository README](https://github.com/marcobn/PAOFLOW#license--citation) for the full citation list.
