# Installation

## Requirements

- Python **3.10** or newer
- A working Quantum ESPRESSO or VASP installation (for DFT input generation; not required to import PAOFLOW itself)
- Core Python dependencies are installed automatically: `numpy`, `scipy`, `mpi4py`, `pandas`

---

## Recommended: pip install

The simplest way to install PAOFLOW is from PyPI:

```bash
pip install PAOFLOW
```

This installs PAOFLOW and its mandatory runtime dependencies.

### Optional dependency groups

PAOFLOW ships several optional extras. Install only what you need:

```bash
# Matplotlib + VTK plotting support
pip install "PAOFLOW[graphics]"

# Boltzmann transport extras (pydantic, pyyaml, psutil)
pip install "PAOFLOW[transport]"

# Fermi surface orbit analysis (pyskeaf)
pip install "PAOFLOW[pyskeaf]"

# Weyl point search (z2pack, tbmodels)
pip install "PAOFLOW[weyl_search]"

# JIT compilation via Numba (speed-up for dense k-grids)
pip install "PAOFLOW[fast]"

# Sparse matrix solvers (requires PETSc/SLEPc system libraries)
pip install "PAOFLOW[sparse]"

# All common extras at once
pip install "PAOFLOW[graphics,transport,pyskeaf,fast]"
```

---

## Developer installation

To work on PAOFLOW itself, or to run the latest unreleased code from the repository:

```bash
git clone https://github.com/marcobn/PAOFLOW.git
cd PAOFLOW
pip install -e ".[dev,graphics,transport]"
```

The `-e` flag installs the package in _editable_ mode — changes to `src/PAOFLOW/` take effect immediately without reinstalling. The `dev` extra adds `pre-commit` and `pytest`.

---

## Conda / virtual environment (recommended for HPC)

It is strongly recommended to work inside an isolated environment, especially on shared HPC clusters where MPI libraries must match the system's MPI installation.

```bash
conda create -n paoflow python=3.12
conda activate paoflow

# Install MPI-compatible mpi4py via conda (links against the system MPI)
conda install -c conda-forge mpi4py

# Then install PAOFLOW from PyPI
pip install "PAOFLOW[graphics,transport]"
```

:::{warning} mpi4py and system MPI
On HPC systems, `pip install mpi4py` may build against the wrong MPI library.
Use `conda install -c conda-forge mpi4py` or load the correct MPI module and
install mpi4py from source to ensure ABI compatibility.
:::

---

## Verifying the installation

Open a Python interpreter and run:

```python
import PAOFLOW
print(PAOFLOW.__version__)
```

You should see the current version string (e.g. `2.9.3`).

---
