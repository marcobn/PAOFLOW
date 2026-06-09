# QE integration tests (asset-bundled)

This folder contains integration tests based on Quantum ESPRESSO (QE) examples.
The goal is to run PAOFLOW in CI **without requiring QE at runtime** by using a
pair of pre-generated asset bundles that contain:

- QE `*.save/` directories (needed by `read_atomic_proj_QE()`)
- PAOFLOW `Reference/` directories (`*.dat` outputs)

The pytest suite copies each job into a temporary sandbox, overlays `*.save/`
from the QE asset bundle and `Reference/` from the PAOFLOW asset bundle, runs
`python main.py`, then compares `output/*.dat` against `Reference/*.dat`.

## What is a “job”?

A **job** is any directory under an `example*` folder that contains a `main.py`.

Job discovery is implemented in [jobs.py](jobs.py).

## Local-first workflow (recommended while developing)

You can validate the asset-bundle infrastructure locally before publishing anything
to GitHub Releases.

### 1) Generate `*.save/` and staged `Reference/` assets (requires QE)

Use the bash runner to run QE, generate PAOFLOW outputs, and stage the test
reference data:

```bash
# From repository root
QE_BIN=/path/to/qe/bin .github/assets_generation/qe/create_assets.sh --all
```

This creates:

- trimmed QE `*.save/` folders under `examples/qe_examples/...`
- staged PAOFLOW test outputs under `tests/integration/qe/_assets/staging/...`

Notes:

- `create_assets.sh` runs QE in any folder containing `*.in` and PAOFLOW in any
  folder containing `main.py`.
- QE launch can be controlled via `PARALLEL_EXEC`:
  - serial (default): unset `PARALLEL_EXEC`
  - MPI: `PARALLEL_EXEC="mpirun -np 8"`
  - SLURM: `PARALLEL_EXEC="srun -n 8"`
- `create_assets.sh` writes PAOFLOW example outputs to in-place `Reference/`
  folders only when `--paoflow-examples` is selected. The test workflow only
  needs the staged test outputs created by `--paoflow-test`.
- staged test outputs are stored directly under each job path in
  `tests/integration/qe/_assets/staging/...`; `build_tar.sh` repacks them into
  `Reference/` directories inside `paoflow_assets.tar.gz` for the test runner.
- After a successful QE run, the workflow trims `*.save/` directories to keep
  only the files needed by PAOFLOW.
- `submit.sh` still exits nonzero if any example fails, but it now runs
  `build_tar.sh` first so `paoflow_assets.tar.gz` is created from successful
  staged jobs whenever possible.

### 2) Build local asset tarballs

Create the QE and PAOFLOW test tarballs from the generated assets:

```bash
# From repository root
.github/assets_generation/qe/build_tar.sh --all
```

By default this writes:

- `.github/assets_generation/qe/_assets/qe_assets.tar.gz`
- `.github/assets_generation/qe/_assets/paoflow_assets.tar.gz`

To override the output locations:

```bash
.github/assets_generation/qe/build_tar.sh \
  --qe-assets-out .github/assets_generation/qe/_assets/qe_assets_dev.tar.gz \
  --paoflow-assets-out .github/assets_generation/qe/_assets/paoflow_assets_dev.tar.gz
```

To delete staged PAOFLOW test outputs after a successful Reference tar build:

```bash
.github/assets_generation/qe/build_tar.sh --paoflow-test --clean-paoflow-test-staging
```

Cleanup is opt-in and only removes staging subdirectories for the selected
examples. If you also pass `--examples`, unrelated staged outputs are left in
place.

`build_tar.sh` uses [.github/assets_generation/qe/build_assets.py](../../../.github/assets_generation/qe/build_assets.py)
internally for the QE `*.save/` tarball. That Python helper packages QE assets only.

### 3) Upload tarballs to a GitHub release

After building the local tarballs, upload them with:

```bash
# From repository root
.github/assets_generation/qe/upload_release_assets.sh integration-assets-v1
```

The upload helper resolves `.github/assets_generation/qe/_assets` relative to
its own script location, so it works from any clone path without editing the
script. You can override the defaults when needed:

```bash
ASSET_DIR=/path/to/qe/_assets \
REPO=owner/repo \
.github/assets_generation/qe/upload_release_assets.sh integration-assets-v1
```

### 4) Run pytest using the local tarballs

Point pytest at both archives:

```bash
pytest -q tests/integration/qe/test_qe_examples.py \
  --qe-assets-archive .github/assets_generation/qe/_assets/qe_assets.tar.gz \
  --reference-assets-archive .github/assets_generation/qe/_assets/paoflow_assets.tar.gz
```

By default, assets are overlaid into the sandbox via symlinks (fast). To use a
copy instead:

```bash
pytest -q tests/integration/qe/test_qe_examples.py \
  --qe-assets-archive .github/assets_generation/qe/_assets/qe_assets.tar.gz \
  --reference-assets-archive .github/assets_generation/qe/_assets/paoflow_assets.tar.gz \
  --qe-assets-link copy
```

## Asset configuration knobs

Assets are required. If assets are not configured, pytest exits with a usage
error explaining which CLI options or environment variables to set.

You can configure the QE savedir assets and the Reference assets via CLI flags
or environment variables.

### CLI flags

- `--qe-assets-archive PATH` local QE asset tarball path
- `--qe-assets-url URL` QE asset tarball URL
- `--qe-assets-sha256 SHA256` expected QE checksum (recommended)
- `--qe-assets-version VERSION` QE cache label
- `--reference-assets-archive PATH` local Reference asset tarball path
- `--reference-assets-url URL` Reference asset tarball URL
- `--reference-assets-sha256 SHA256` expected Reference checksum
- `--reference-assets-version VERSION` Reference cache label
- `--qe-assets-link symlink|copy` overlay strategy

### Environment variables

- `PAOFLOW_QE_ASSET_ARCHIVE` local QE asset tarball path
- `PAOFLOW_QE_ASSET_URL` QE asset tarball URL
- `PAOFLOW_QE_ASSET_SHA256` expected QE checksum
- `PAOFLOW_QE_ASSET_VERSION` QE cache label
- `PAOFLOW_REFERENCE_ASSET_ARCHIVE` local Reference asset tarball path
- `PAOFLOW_REFERENCE_ASSET_URL` Reference asset tarball URL
- `PAOFLOW_REFERENCE_ASSET_SHA256` expected Reference checksum
- `PAOFLOW_REFERENCE_ASSET_VERSION` Reference cache label

Extraction is cached under `${XDG_CACHE_HOME:-~/.cache}/paoflow/qe-assets/`.
Implementation lives in [assets.py](assets.py).

## QE execution

Pytest does **not** run QE. QE is expected to be run on HPC resources via
[.github/assets_generation/qe/create_assets.sh](../../../.github/assets_generation/qe/create_assets.sh)
or [.github/assets_generation/qe/submit.sh](../../../.github/assets_generation/qe/submit.sh)
to generate `*.save/` and staged `Reference/` assets, which are then packaged
into separate tarballs.

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

1. Generate QE `*.save/` + staged PAOFLOW `Reference/` assets on HPC: [.github/assets_generation/qe/submit.sh](../../../.github/assets_generation/qe/submit.sh) (SLURM) or [.github/assets_generation/qe/create_assets.sh](../../../.github/assets_generation/qe/create_assets.sh) (direct)
2. Package QE and Reference assets into separate tarballs: [.github/assets_generation/qe/build_tar.sh](../../../.github/assets_generation/qe/build_tar.sh)
3. Optionally upload both tarballs to a GitHub release: [.github/assets_generation/qe/upload_release_assets.sh](../../../.github/assets_generation/qe/upload_release_assets.sh)
4. Run pytest using both tarballs (PAOFLOW-only): [test_qe_examples.py](test_qe_examples.py)
5. Internals used by pytest: [assets.py](assets.py), [runner.py](runner.py), [jobs.py](jobs.py), [compare.py](compare.py), [conftest.py](conftest.py)

- [.github/assets_generation/qe/create_assets.sh](../../../.github/assets_generation/qe/create_assets.sh): generate `*.save/` (QE) and staged `Reference/` assets for tests; trims large QE artifacts
- [.github/assets_generation/qe/submit.sh](../../../.github/assets_generation/qe/submit.sh): SLURM wrapper for `create_assets.sh` and `build_tar.sh`
- [.github/assets_generation/qe/build_tar.sh](../../../.github/assets_generation/qe/build_tar.sh): package QE and Reference assets into separate tar.gz files
- [.github/assets_generation/qe/build_assets.py](../../../.github/assets_generation/qe/build_assets.py): package QE `*.save/` directories into a tar.gz
- [.github/assets_generation/qe/upload_release_assets.sh](../../../.github/assets_generation/qe/upload_release_assets.sh): upload `qe_assets.tar.gz` and `paoflow_assets.tar.gz` from the repo-relative `_assets/` directory; override with `ASSET_DIR` and `REPO` if needed
- [assets.py](assets.py): resolve/download/verify/extract the asset tarball into a cache
- [jobs.py](jobs.py): discover runnable jobs (any directory with `main.py`)
- [runner.py](runner.py): sandbox runner; overlays QE and Reference assets; runs PAOFLOW
- [test_qe_examples.py](test_qe_examples.py): pytest entrypoint for QE integration tests
- [compare.py](compare.py): output-vs-reference `*.dat` comparison logic
- [conftest.py](conftest.py): pytest options and fixtures (asset flags)
