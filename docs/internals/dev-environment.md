# Development Environment

Getting set up to work on PAOFLOW should only take a few minutes. Here's everything you need.

## Requirements

- Python 3.10 or newer
- Linux / POSIX environment
- A working MPI installation (needed by `mpi4py`)

We recommend a Conda-based distribution like Miniforge or Miniconda for managing your environment — it makes installing `mpi4py` much smoother.

## Setting Up

Clone the repository and install in editable mode with the `dev` extras:

```bash
conda create -n paoflow-dev python=3.10
conda activate paoflow-dev
conda install mpi4py
pip install -e .[dev]
```

Then install the pre-commit hooks right away — they keep formatting consistent automatically:

```bash
pre-commit install
```

To confirm everything is working:

```bash
python -c "import PAOFLOW; print('PAOFLOW import OK')"
pytest -q tests/unit
```

## Optional Dependency Groups

Install only what you need:

| Extra | Dependencies | Use case |
|-------|--------------|----------|
| `weyl_search` | z2pack, tbmodels | Topological invariants |
| `graphics` | vtk, matplotlib | Visualization |
| `transport` | pyyaml, psutil, pydantic | Transport calculations |
| `pyskeaf` | scikit-image, shapely, joblib, pandas | Fermi surface |
| `fast` | numba | JIT-accelerated paths |
| `dev` | pre-commit, pytest | Development tools |

You can combine extras freely:

```bash
pip install -e .[graphics,transport,fast,dev]
```

## Code Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The pre-commit hooks take care of this for you automatically on each commit, so you shouldn't need to think about it much. If you ever want to run them manually:

```bash
pre-commit run --all-files
```

And to reset if something gets into a weird state:

```bash
pre-commit clean
pre-commit run --all-files
```

If you use VSCode, the Ruff extension with format-on-save gives you continuous feedback as you write.

## Troubleshooting

**`mpi4py` issues:** Install it through Conda (`conda install mpi4py`) rather than pip — this ensures it links against the right MPI libraries.

**Wrong Python environment:** Use `python -m pip install -e .[dev]` to make sure `pip` and `python` are from the same environment.

**Stale editable install:** If your code changes don't seem to take effect, restart the Python interpreter or notebook kernel — editable installs don't hot-reload.
