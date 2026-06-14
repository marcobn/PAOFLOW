# Release and Maintenance

## Release Workflow

Releases follow the branch model described in [Contribution Guidelines](contributing.md#release-process). Before a release can be published, the QE integration test assets must be generated and uploaded.

## QE Test Asset Generation

The test asset pipeline manages a combined archive (`qe_test_assets.tar.gz`) containing:

- Trimmed QE `.save` directories
- PAOFLOW reference outputs for regression testing
- Required `BASIS` directories for internal-basis examples

### Key Scripts

| Script | Purpose |
|--------|---------|
| `create_assets.sh` | Generates raw QE and PAOFLOW outputs |
| `build_tar.sh` | Packages outputs into the archive (supports `--repack`) |
| `build_assets.py` | Internal helper for packaging logic |
| `submit.sh` | SLURM wrapper for HPC environments |
| `upload_release_assets.sh` | Publishes archive and checksums to a GitHub release |

All scripts live under `.github/assets_generation/`.

### Stage 1: Raw output generation

Run `create_assets.sh` to execute QE calculations and PAOFLOW analyses, producing `.save` directories and staged outputs.

**Important:** Running without flags performs full generation. Running with `--all` performs QE + test staging only, skipping example references. This distinction is easy to miss — double-check which mode you need.

### Stage 2: Archive assembly

```bash
bash build_tar.sh
```

For partial updates (replacing only some assets), use repack mode:

1. Download the current archive.
2. Unpack it under `.github/assets_generation/qe/_assets/`.
3. Replace the needed files.
4. Run `bash build_tar.sh --repack`.

### Stage 3: Publishing the release

```bash
bash upload_release_assets.sh
```

This uploads the archive and its SHA256 checksum to the GitHub release.

After publishing, update `.github/workflows/ci.yaml` with the new release tag, download URL, and SHA256 checksum. If CI still points to the old release or old checksum, the QE integration job will continue to download the previous assets or fail.

## CI/CD Processes

CI runs the full test suite on pull requests targeting `develop`. Integration tests download the QE assets from the release specified in `.github/workflows/ci.yaml`.

See [Testing](testing.md) for how to interpret CI failures and adjust reference data or thresholds.
