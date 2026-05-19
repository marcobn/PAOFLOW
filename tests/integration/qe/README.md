# QE integration tests (asset-bundled)

This folder contains integration tests based on Quantum ESPRESSO (QE) examples.
The goal is to run PAOFLOW in CI **without requiring QE at runtime** by using a
pre-generated asset bundle that contains:

- QE `*.save/` directories (needed by `read_atomic_proj_QE()`)
- PAOFLOW `Reference/` directories ( `*.dat` outputs)

The pytest suite copies each job into a temporary sandbox, overlays `*.save/` and
`Reference/` from the asset bundle, runs `python main.py`, then compares
`output/*.dat` against `Reference/*.dat`.

## What is a “job”?

A **job** is any directory under an `example*` folder that contains a `main.py`.

Job discovery is implemented in [jobs.py](jobs.py).

## Local-first workflow (recommended while developing)

You can validate the asset-bundle infrastructure locally before publishing anything
to GitHub Releases.

### 1) Generate `*.save/` and `Reference/` (requires QE)

Use the bash runner to run QE and then PAOFLOW, and to trim the `*.save/` folders
to a minimal size:

```bash
# From repository root
cd .github/assets_generation/qe

# Run QE + PAOFLOW for all examples (creates *.save and Reference)
QE_BIN=/path/to/qe/bin ./job.sh --all
```

Optionally, build the asset tarball automatically after a successful run:

```bash
QE_BIN=/path/to/qe/bin ./job.sh --all --build-assets
```

To override the tarball output location:

```bash
QE_BIN=/path/to/qe/bin ./job.sh --all --build-assets --assets-out ./_assets/qe_test_assets_dev.tar.gz
```

Notes:

- `job.sh` will run QE in any folder containing `*.in` and PAOFLOW in any folder
  containing `main.py`.
- QE launch can be controlled via `PARALLEL_EXEC`:
  - serial (default): unset `PARALLEL_EXEC`
  - MPI: `PARALLEL_EXEC="mpirun -np 8"`
  - SLURM: `PARALLEL_EXEC="srun -n 8"`
- After a successful PAOFLOW run, it moves the PAOFLOW output directory into
  `Reference/` inside the same job folder.
- It also cleans up QE outputs and trims `*.save/` directories to keep only `*.xml`
  and `*.UPF` (which are required by PAOFLOW to read projections).

### 2) Build a local asset tarball

Create a tarball containing only `Reference/` and `*.save/` directories, preserving
their paths relative to this directory:

```bash
# From repository root
mkdir -p tests/integration/qe/_assets
python .github/assets_generation/qe/build_assets.py \
  --qe-root tests/integration/qe \
  --out tests/integration/qe/_assets/qe_test_assets_dev.tar.gz
```

The builder is [.github/assets_generation/qe/build_assets.py](../../../.github/assets_generation/qe/build_assets.py).

### 3) Run pytest using the local tarball

```bash
pytest -q tests/integration/qe/test_qe_examples.py \
  --qe-assets-archive tests/integration/qe/_assets/qe_test_assets_dev.tar.gz
```

By default, assets are overlaid into the sandbox via symlinks (fast). To use a
copy instead:

```bash
pytest -q tests/integration/qe/test_qe_examples.py \
  --qe-assets-archive tests/integration/qe/_assets/qe_test_assets_dev.tar.gz \
  --qe-assets-link copy
```

## Asset configuration knobs

Assets are required. If assets are not configured, pytest exits with a usage
error explaining which CLI options or environment variables to set.

You can configure assets via CLI flags or environment variables.

### CLI flags

- `--qe-assets-archive PATH` local tarball path
- `--qe-assets-url URL` download tarball from URL (future/CI usage)
- `--qe-assets-sha256 SHA256` expected checksum (recommended)
- `--qe-assets-version VERSION` cache label (used for naming downloads)
- `--qe-assets-link symlink|copy` overlay strategy

### Environment variables

- `PAOFLOW_QE_ASSET_ARCHIVE` local tarball path
- `PAOFLOW_QE_ASSET_URL` download URL
- `PAOFLOW_QE_ASSET_SHA256` expected checksum
- `PAOFLOW_QE_ASSET_VERSION` cache label

Extraction is cached under `${XDG_CACHE_HOME:-~/.cache}/paoflow/qe-assets/`.
Implementation lives in [assets.py](assets.py).

## QE execution

Pytest does **not** run QE. QE is expected to be run on HPC resources via
[.github/assets_generation/qe/job.sh](../../../.github/assets_generation/qe/job.sh) to generate `*.save/` and `Reference/`, which are then packaged
into the asset tarball.

## How outputs and comparisons work

- PAOFLOW is executed as `python main.py` in the job directory.
- The output directory is inferred by a simple regex that looks for
  `outputdir = '...'` in `main.py`. If not found, it defaults to `output`.
- Comparison is done by [compare.py](compare.py):
  - only `*.dat` files are compared
  - the list of output `*.dat` filenames must match the reference list exactly
  - data is compared column-wise (ignoring the first column), with a default tolerance
  - per-file comparison plots (`output vs reference` + `|delta|`) are written under
    each sandbox job folder at `_compare_plots/*.png` for visual verification

## File guide

Recommended execution order:

1. Generate QE `*.save/` + PAOFLOW `Reference/` on HPC: [.github/assets_generation/qe/submit.sh](../../../.github/assets_generation/qe/submit.sh) (SLURM) or [.github/assets_generation/qe/job.sh](../../../.github/assets_generation/qe/job.sh) (direct)
2. Package `*.save/` + `Reference/` into a tarball: [.github/assets_generation/qe/build_assets.py](../../../.github/assets_generation/qe/build_assets.py)
3. Run pytest using the tarball (PAOFLOW-only): [test_qe_examples.py](test_qe_examples.py)
4. Internals used by pytest: [assets.py](assets.py), [runner.py](runner.py), [jobs.py](jobs.py), [compare.py](compare.py), [conftest.py](conftest.py)

- [.github/assets_generation/qe/job.sh](../../../.github/assets_generation/qe/job.sh): generate `*.save/` (QE) and `Reference/` (PAOFLOW) in-place; trims large QE artifacts
- [.github/assets_generation/qe/submit.sh](../../../.github/assets_generation/qe/submit.sh): SLURM wrapper for `job.sh`
- [.github/assets_generation/qe/build_assets.py](../../../.github/assets_generation/qe/build_assets.py): package `Reference/` + `*.save/` into a tar.gz (local-first)
- [assets.py](assets.py): resolve/download/verify/extract the asset tarball into a cache
- [jobs.py](jobs.py): discover runnable jobs (any directory with `main.py`)
- [runner.py](runner.py): sandbox runner; overlays assets; runs PAOFLOW
- [test_qe_examples.py](test_qe_examples.py): pytest entrypoint for QE integration tests
- [compare.py](compare.py): output-vs-reference `*.dat` comparison logic
- [conftest.py](conftest.py): pytest options and fixtures (asset flags)
