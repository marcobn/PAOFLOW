# Release and Maintenance

## Release Workflow

Releases follow the branch model described in [Contribution Guidelines](contributing.md#release-process). Before a release goes out, the QE integration test assets need to be generated and uploaded — this page walks through that process.

## QE Test Asset Generation

The test asset pipeline manages a combined archive (`qe_test_assets.tar.gz`) that contains everything the integration tests need:

- Trimmed QE `.save` directories
- PAOFLOW reference outputs for regression testing
- `BASIS` directories for internal-basis examples

### Key scripts

| Script | What it does |
|--------|-------------|
| `create_assets.sh` | Runs QE calculations and PAOFLOW analyses to generate raw outputs |
| `build_tar.sh` | Packages everything into the archive (supports `--repack` for partial updates) |
| `build_assets.py` | Internal helper used by the packaging logic |
| `submit.sh` | SLURM wrapper for running on HPC systems |
| `upload_release_assets.sh` | Uploads the archive and checksums to a GitHub release |

All scripts live under `.github/assets_generation/`.

### Stage 1: Generate raw outputs

Run `create_assets.sh` to execute the QE calculations and PAOFLOW analyses. This produces the `.save` directories and staged outputs that go into the archive.

One thing to watch out for: running without flags does a full generation, while `--all` performs QE + test staging only and skips example references. It's an easy distinction to miss, so double-check which mode you need before kicking off a long run.

### Stage 2: Build the archive

```bash
bash build_tar.sh
```

If you only need to update a subset of the assets, use repack mode instead of regenerating everything from scratch:

1. Download the current archive.
2. Unpack it under `.github/assets_generation/qe/_assets/`.
3. Replace the files you want to update.
4. Run `bash build_tar.sh --repack`.

### Stage 3: Publish the release

```bash
bash upload_release_assets.sh
```

This uploads the archive and its SHA256 checksum to the GitHub release. After publishing, update `.github/workflows/ci.yaml` with the new release tag, download URL, and checksum. If CI still points to the old release, it will keep downloading the previous assets (or fail) — so this step is easy to forget but important not to.

## CI/CD

CI runs the full test suite on every pull request targeting `develop`. Integration tests download the QE assets from whichever release is specified in `.github/workflows/ci.yaml`.

See [Testing](testing.md) for guidance on interpreting CI failures and deciding whether to fix code, regenerate references, or adjust comparison thresholds.
