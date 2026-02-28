# TBmodel integration tests (asset-bundled)

This folder contains integration tests based on PAOFLOW tight-binding models.
The goal is to run TBmodel integration checks in CI using pre-generated test
assets.

TBmodel jobs require:

- `*.dat` outputs for regression comparison

The pytest suite copies each job into a temporary sandbox, runs the TBmodel
script, then compares output `*.dat` against the asset bundle.

## What is a "job"?

A **job** is any top-level Python script in this directory that is not part of
the test harness (for example: `graphene.py`, `kane_mele.py`, `slater_koster.py`).

Job discovery is implemented in [jobs.py](jobs.py).

## Local-first workflow

### 1) Generate outputs

Run the TBmodel scripts to generate `*.dat` outputs in place:

```bash
# From repository root
cd tests/integration/TBmodel
./job.sh --build-assets
```

When generating assets through [job.sh](job.sh), you can control launch with
`PARALLEL_EXEC`:

- serial (default): unset `PARALLEL_EXEC`
- MPI: `PARALLEL_EXEC="mpirun -np 8"`
- SLURM: `PARALLEL_EXEC="srun -n 8"`

### 2) Build a local asset tarball

Create a tarball containing output `*.dat` files for all discovered jobs:

```bash
# From repository root
mkdir -p tests/integration/TBmodel/_assets
python -m tests.integration.TBmodel.build_assets \
  --out tests/integration/TBmodel/_assets/tbmodel_test_assets_dev.tar.gz
```

The builder is [build_assets.py](build_assets.py).

### 3) Run pytest using the local tarball

```bash
pytest -q tests/integration/TBmodel/test_tbmodel_examples.py \
  --tbmodel-assets-archive tests/integration/TBmodel/_assets/tbmodel_test_assets_dev.tar.gz
```

By default, assets are overlaid into the sandbox via symlinks (fast). To use a
copy instead:

```bash
pytest -q tests/integration/TBmodel/test_tbmodel_examples.py \
  --tbmodel-assets-archive tests/integration/TBmodel/_assets/tbmodel_test_assets_dev.tar.gz \
  --tbmodel-assets-link copy
```

## Asset configuration knobs

Assets are optional. If assets are not configured, tests skip with a clear message.

You can configure assets via CLI flags or environment variables.

### CLI flags

- `--tbmodel-assets-archive PATH` local tarball path
- `--tbmodel-assets-url URL` download tarball from URL
- `--tbmodel-assets-sha256 SHA256` expected checksum
- `--tbmodel-assets-version VERSION` cache label
- `--tbmodel-assets-link symlink|copy` overlay strategy

### Environment variables

- `PAOFLOW_TBMODEL_ASSET_ARCHIVE` local tarball path
- `PAOFLOW_TBMODEL_ASSET_URL` download URL
- `PAOFLOW_TBMODEL_ASSET_SHA256` expected checksum
- `PAOFLOW_TBMODEL_ASSET_VERSION` cache label

Extraction is cached under `${XDG_CACHE_HOME:-~/.cache}/paoflow/tbmodel-assets/`.
Implementation lives in [assets.py](assets.py).

## File guide

- [job.sh](job.sh): run TBmodel scripts to generate `*.dat` outputs
- [submit.sh](submit.sh): SLURM wrapper for `job.sh`
- [build_assets.py](build_assets.py): package output `*.dat` files into a tar.gz
- [assets.py](assets.py): resolve/download/verify/extract the asset tarball
- [jobs.py](jobs.py): discover runnable TBmodel scripts
- [runner.py](runner.py): sandbox runner; overlays assets; runs TBmodel scripts
- [test_tbmodel_examples.py](test_tbmodel_examples.py): pytest entrypoint
- [compare.py](compare.py): output-vs-reference `*.dat` comparison logic
- [conftest.py](conftest.py): pytest options and asset fixtures
