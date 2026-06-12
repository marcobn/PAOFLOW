# API Reference

The PAOFLOW API is a single high-level class — `PAOFLOW.PAOFLOW.PAOFLOW` — that encapsulates the full post-processing workflow. All methods are called sequentially on the same instance.

---

## Package layout

```
PAOFLOW/                  ← top-level package (src/PAOFLOW/)
├── PAOFLOW.py            ← main class: PAOFLOW.PAOFLOW.PAOFLOW
├── DataController.py     ← central data store
├── boltzmann/            ← Boltzmann transport machinery
├── hamiltonian/          ← Hamiltonian construction and symmetrisation
├── inputs/               ← QE/VASP XML readers
├── models/               ← TB model Hamiltonians, Slater–Koster
├── projection/           ← PAO projection utilities
├── graphics/             ← Optional plotting helpers
├── basis_gen/            ← Pseudopotential basis generation
└── gen/                  ← CLI driver generators
```

---

## Import path

```python
from PAOFLOW import PAOFLOW          # imports the module PAOFLOW.PAOFLOW
pf = PAOFLOW.PAOFLOW(savedir=...)    # instantiates the class
```

---

## Pages in this section

- [PAOFLOW class](paoflow.md) — constructor, all public methods, and key attributes
