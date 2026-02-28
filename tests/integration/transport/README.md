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

### 1) Build a local asset tarball

Create a tarball containing `Reference/` and `*.save` directories for all
discovered jobs, preserving their paths relative to this directory:

```bash
# From repository root
mkdir -p tests/integration/transport/_assets
python -m tests.integration.transport.build_assets \
  --out tests/integration/transport/_assets/transport_test_assets_dev.tar.gz
```

The builder is [build_assets.py](build_assets.py).

When generating assets through [job.sh](job.sh), QE launch can be controlled via
`PARALLEL_EXEC`:

- serial (default): unset `PARALLEL_EXEC`
- MPI: `PARALLEL_EXEC="mpirun -np 8"`
- SLURM: `PARALLEL_EXEC="srun -n 8"`

### 2) Run pytest using the local tarball

```bash
pytest -q tests/integration/transport/test_transport_examples.py \
  --transport-assets-archive tests/integration/transport/_assets/transport_test_assets_dev.tar.gz
```

By default, assets are overlaid into the sandbox via symlinks (fast). To use a
copy instead:

```bash
pytest -q tests/integration/transport/test_transport_examples.py \
  --transport-assets-archive tests/integration/transport/_assets/transport_test_assets_dev.tar.gz \
  --transport-assets-link copy
```

## Asset configuration knobs

Assets are optional. If assets are not configured and the working tree does not
contain `Reference/` and `*.save`, tests skip with a clear message.

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

- [build_assets.py](build_assets.py): package `Reference/` + `*.save` into tar.gz
- [assets.py](assets.py): resolve/download/verify/extract asset tarball into cache
- [jobs.py](jobs.py): discover runnable transport jobs
- [runner.py](runner.py): sandbox runner; overlays assets; runs transport scripts
- [test_transport_examples.py](test_transport_examples.py): pytest integration entrypoint
- [compare.py](compare.py): output-vs-reference `*.dat` comparison logic
- [conftest.py](conftest.py): pytest options and asset fixtures
