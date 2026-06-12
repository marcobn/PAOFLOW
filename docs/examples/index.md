# Examples

PAOFLOW ships a set of ready-to-run example workflows in the [`examples/`](https://github.com/marcobn/PAOFLOW/tree/master/examples) directory of the repository. All examples are **Python scripts** — no notebooks are required to get started.

---

## How the examples are organised

```
examples/
├── qe_examples/        # Quantum ESPRESSO-based workflows (main focus of this page)
├── vasp_examples/      # VASP-based workflows
├── acbn0_examples/     # ACBN0 Hubbard U calculations
├── TBmodel_examples/   # Tight-binding model Hamiltonians
├── transport_examples/ # Landauer–Büttiker quantum transport
└── plot_examples/      # Post-processing and plotting scripts
```

---

## Running an example

Each QE example directory is self-contained. It includes:

- A pre-computed `*.save/` directory with the QE output (you do **not** need to run QE to try the examples)
- A `main.py` driver script that runs PAOFLOW on that output
- A `Reference/` directory with expected output for validation

To run any example:

```bash
cd examples/qe_examples/example01
python main.py
```

For parallel execution:

```bash
mpirun -np <num_cores> python main.py
```

Alternatively, the legacy XML-input style (PAOFLOW v1 compatibility) uses the shared `main.py` at the top of `qe_examples/`:

```bash
python ../main.py ./ inputfile.xml
```

---

## Example categories

| Category | Location | Description |
|---|---|---|
| QE examples | `qe_examples/` | Full workflows from QE output — see [QE Examples](qe-examples.md) |
| VASP examples | `vasp_examples/` | Workflows from VASP OUTCAR/vasprun.xml |
| ACBN0 | `acbn0_examples/` | Hubbard U/V for MgO, ZnO, MnO |
| TB models | `TBmodel_examples/` | Slater–Koster and Kane–Mele model Hamiltonians |
| Transport | `transport_examples/` | Green's function / Landauer–Büttiker conductance |
| Plotting | `plot_examples/` | Standalone scripts for band, DOS, and transport plots |

---

## Notebook-based tutorials

A Jupyter notebook tutorial (`PAOFLOW-tutorial/PAO-tutorial.ipynb`) is available in the repository for a guided introduction. Full notebook-based documentation will be added in a future milestone.
