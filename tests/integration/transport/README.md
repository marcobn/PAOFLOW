# Transport integration tests (asset-bundled)

This folder contains integration tests based on PAOFLOW transport examples.
The goal is to run transport integration checks in CI using pre-generated test
assets, without requiring QE at test runtime.

Transport jobs require:

- `*.save` data (used by `read_atomic_proj_QE()`)
- `Reference/*.dat` outputs for regression comparison

The pytest suite copies each job into a temporary sandbox, optionally overlays
`*.save` and `Reference` from an asset bundle, runs the transport Python
entrypoint(s), then compares `output/paoflow/*.dat` against `Reference/*.dat`.

## What is a “job”?

A **job** is any directory under an `example*` folder that contains one of:

- `main.py`
- `main_conductor.py`
- `main_current.py`

Job discovery is implemented in [jobs.py](jobs.py).

## Local-first workflow

### 1) Generate savedirs and staged Reference outputs

Use the transport asset-generation wrapper on a machine that has QE and the
required Python environment available:

```bash
# From repository root
.github/assets_generation/transport/create_assets.sh --all
```

Useful selectors:

```bash
.github/assets_generation/transport/create_assets.sh --all --examples example01
.github/assets_generation/transport/create_assets.sh --qe --skip-qe-if-save-exists
```

The script stages test-side `Reference/` outputs under
`tests/integration/transport/_assets/staging/` by default.

Transport savedirs are discovered recursively and the workflow intentionally
keeps `*wfc*` files inside `*.save/`.

### 2) Build a local combined asset tarball

Package example `*.save/` directories together with the staged test
`Reference/` outputs into one tarball:

```bash
# From repository root
.github/assets_generation/transport/build_tar.sh --all
```

By default this writes:

```bash
.github/assets_generation/transport/_assets/transport_test_assets.tar.gz
```

Upload the tarball to an existing GitHub release with:

```bash
# From repository root
.github/assets_generation/transport/upload_release_assets.sh integration-assets-v1
```

The upload helper resolves `.github/assets_generation/transport/_assets`
relative to its own script location, so it works from any clone path without
editing the script. You can override the defaults when needed:

```bash
ASSET_DIR=/path/to/transport/_assets \
REPO=owner/repo \
.github/assets_generation/transport/upload_release_assets.sh integration-assets-v1
```

### 3) Build directly from existing folders

Create a tarball containing `Reference/` and `*.save` directories for all
discovered jobs, preserving their paths relative to this directory:

```bash
# From repository root
mkdir -p examples/transport_examples/_assets
python .github/assets_generation/transport/build_assets.py \
  --transport-root examples/transport_examples \
  --reference-root tests/integration/transport/_assets/staging \
  --out examples/transport_examples/_assets/transport_test_assets_dev.tar.gz
```

The builder is [.github/assets_generation/transport/build_assets.py](../../../.github/assets_generation/transport/build_assets.py).

When generating assets through [.github/assets_generation/transport/create_assets.sh](../../../.github/assets_generation/transport/create_assets.sh), QE launch can be controlled via
`PARALLEL_EXEC`:

- serial (default): unset `PARALLEL_EXEC`
- MPI: `PARALLEL_EXEC="mpirun -np 8"`
- SLURM: `PARALLEL_EXEC="srun -n 8"`

### 4) Run pytest using the local tarball

```bash
pytest -q tests/integration/transport/test_transport_examples.py \
  --transport-assets-archive examples/transport_examples/_assets/transport_test_assets_dev.tar.gz
```

By default, assets are overlaid into the sandbox via symlinks (fast). To use a
copy instead:

```bash
pytest -q tests/integration/transport/test_transport_examples.py \
  --transport-assets-archive examples/transport_examples/_assets/transport_test_assets_dev.tar.gz \
  --transport-assets-link copy
```

## Asset configuration knobs

Assets are required. If assets are not configured, pytest exits with a usage
error explaining which CLI options or environment variables to set.

You can configure assets via CLI flags or environment variables.

### CLI flags

- `--transport-assets-archive PATH` local tarball path
- `--transport-assets-url URL` download tarball from URL
- `--transport-assets-sha256 SHA256` expected checksum
- `--transport-assets-version VERSION` cache label
- `--transport-assets-link symlink|copy` overlay strategy

### Environment variables

- `PAOFLOW_TRANSPORT_ASSET_ARCHIVE` local tarball path
- `PAOFLOW_TRANSPORT_ASSET_URL` download URL
- `PAOFLOW_TRANSPORT_ASSET_SHA256` expected checksum
- `PAOFLOW_TRANSPORT_ASSET_VERSION` cache label

Extraction is cached under `${XDG_CACHE_HOME:-~/.cache}/paoflow/transport-assets/`.
Implementation lives in [assets.py](assets.py).

## File guide

- [.github/assets_generation/transport/create_assets.sh](../../../.github/assets_generation/transport/create_assets.sh): run QE and PAOFLOW to populate transport savedirs and staged test References
- [.github/assets_generation/transport/build_tar.sh](../../../.github/assets_generation/transport/build_tar.sh): build the combined `transport_test_assets.tar.gz`
- [.github/assets_generation/transport/build_assets.py](../../../.github/assets_generation/transport/build_assets.py): package staged `Reference/` + discovered `*.save` into tar.gz
- [.github/assets_generation/transport/job.sh](../../../.github/assets_generation/transport/job.sh): convenience wrapper for `create_assets.sh` + `build_tar.sh`
- [.github/assets_generation/transport/upload_release_assets.sh](../../../.github/assets_generation/transport/upload_release_assets.sh): upload `transport_test_assets.tar.gz` to a GitHub release
- [assets.py](assets.py): resolve/download/verify/extract asset tarball into cache
- [jobs.py](jobs.py): discover runnable transport jobs
- [runner.py](runner.py): sandbox runner; overlays assets; runs transport scripts
- [test_transport_examples.py](test_transport_examples.py): pytest integration entrypoint
- [compare.py](compare.py): output-vs-reference `*.dat` comparison logic with per-file plot outputs under `_compare_plots`
- [conftest.py](conftest.py): pytest options and asset fixtures
