# Development Environment

## Requirements

- Python 3.10 or newer
- Linux / POSIX environment
- A working MPI installation (required by `mpi4py`)

Conda-based distributions (Miniforge, Miniconda) are recommended for managing the environment.

## Installation for Developers

Clone the repository and install in editable mode with the `dev` extras:

```bash
conda create -n paoflow-dev python=3.10
conda activate paoflow-dev
conda install mpi4py
pip install -e .[dev]
```

The `dev` extra installs `pre-commit` and `pytest`. Install the git hooks immediately after:

```bash
pre-commit install
```

To verify the setup:

```bash
python -c "import PAOFLOW; print('PAOFLOW import OK')"
pytest -q tests/unit
```

## Optional Dependency Groups

| Extra | Dependencies | Use case |
|-------|--------------|----------|
| `weyl_search` | z2pack, tbmodels | Topological invariants |
| `graphics` | vtk, matplotlib | Visualization |
| `transport` | pyyaml, psutil, pydantic | Transport calculations |
| `pyskeaf` | scikit-image, shapely, joblib, pandas | Fermi surface |
| `fast` | numba | JIT-accelerated paths |
| `dev` | pre-commit, pytest | Development tools |

Combine extras as needed:

```bash
pip install -e .[graphics,transport,fast,dev]
```

## Code Formatting

PAOFLOW uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks apply formatting automatically before each commit.

Run hooks manually on all files:

```bash
pre-commit run --all-files
```

To clean and reset:

```bash
pre-commit clean
pre-commit run --all-files
```

**VSCode:** Install the Ruff extension and enable format-on-save in `settings.json` for continuous formatting.

## Troubleshooting

**`mpi4py` issues:** Prefer Conda installation (`conda install mpi4py`) over pip to ensure MPI library compatibility.

**Wrong Python environment:** Use `python -m pip install -e .[dev]` to guarantee consistency between `pip` and `python`.

**Editable install stale after code changes:** Restart the Python interpreter or notebook kernel after changes — editable installs do not hot-reload.
