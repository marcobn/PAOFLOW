# Installation

## Prerequisites

PAOFLOW requires **Python 3.10 or later**.
[Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) are the recommended Python distributions.

### MPI

`mpi4py` is a required dependency. Installing it via conda/mamba is recommended to ensure it links against your system MPI libraries:

```bash
conda install mpi4py
```

If a compatible MPI implementation (OpenMPI or MPICH) is already present on your system, pip also works:

```bash
pip install mpi4py
```

---

## Installing from PyPI

```bash
pip install PAOFLOW
```

---

## Manual installation

From the root of the repository:

```bash
pip install .
```

Without administrator privileges:

```bash
pip install --user .
```

---

## Optional dependencies

All optional dependency groups are declared in `pyproject.toml`. Install them by appending the group name in brackets.

| Extra | Contents | Install command |
|---|---|---|
| `weyl_search` | z2pack, tbmodels | `pip install PAOFLOW[weyl_search]` |
| `graphics` | vtk, matplotlib | `pip install PAOFLOW[graphics]` |
| `transport` | pyyaml, psutil, pydantic | `pip install PAOFLOW[transport]` |
| `pyskeaf` | scikit-image, shapely, joblib, pandas | `pip install PAOFLOW[pyskeaf]` |
| `fast` | numba | `pip install PAOFLOW[fast]` |
| `dev` | pre-commit, pytest | `pip install PAOFLOW[dev]` |

Multiple extras can be combined:

```bash
pip install PAOFLOW[pyskeaf,fast]
```

---

## pyskeaf CLI

When the `pyskeaf` extra is installed, the `pyskeaf` command-line tool becomes available:

```bash
pyskeaf --help
```

---

## Editable (development) install

To reflect source-code changes without reinstalling:

```bash
pip install -e .[dev]
```

> **Note:** an interpreter restart (or Jupyter kernel restart) is required for changes to take effect even in editable mode.
