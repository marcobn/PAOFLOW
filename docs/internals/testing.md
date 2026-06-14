# Testing

## Overview

PAOFLOW has two test categories:

**Unit tests** (`tests/unit/`) — fast, isolated tests for individual functions or classes. File names start with `test_`, functions named `test_<function_name>()`.

**Integration tests** (`tests/integration/`) — end-to-end tests that run a minimal workflow and compare output against known-good reference data. Use lower accuracy settings (reduced grid size, k-points) to keep CI fast. Large prerequisites are staged as release assets rather than committed to the repository.

## Running Tests

```bash
# All tests
pytest -q tests

# All unit tests
pytest -q tests/unit

# Specific unit test file
pytest -q tests/unit/transport/calculators/test_current.py::test_build_bias_grid_linspace

# All integration tests for a family
pytest -q tests/integration/transport

# Specific integration example
pytest -q tests/integration/transport -k example01
```

Remove `-q` for verbose output.

## Integration Test Patterns

There are three established patterns. Use the pattern that matches your example type — do not invent a new one unless genuinely necessary.

**Pattern A: Split assets** — for examples requiring both expensive runtime inputs (e.g., QE `.save` directories) and separate reference outputs for regression comparison. One archive for runtime inputs, one for reference outputs. Typical for QE and VASP-based examples.

**Pattern B: Combined assets** — when runtime inputs and reference outputs naturally belong together. One archive containing both. Typical for transport examples.

**Pattern C: Reference-only assets** — for self-contained Python-only examples that run from committed source files alone. One archive with only expected output files.

For new families (VASP, ACBN0, etc.), copy an existing pattern rather than creating a new layout.

## Adding Integration Tests for a New Example Family

Recommended directory layout:

```
tests/integration/<family>/
  __init__.py
  README.md
  assets.py
  conftest.py
  jobs.py
  runner.py
  compare.py
  test_<family>_examples.py
  example01/
  example02/
```

An integration test answers: "Can this example still run successfully and produce numerically consistent results in an isolated environment?" It should:

- Run only the minimum workflow needed.
- Avoid expensive external calculations during CI.
- Compare generated outputs against known-good reference data.
- Stage large prerequisites as release assets (see [Release Workflow](release.md)).

## What To Do If Integration Tests Fail

Failures fall into three categories: real behavioral regression, missing or wrong test assets, and small numerical differences between machines. Distinguish these before modifying code, references, or thresholds.

### Reading the comparison plots

Failed tests generate comparison plots showing reference data, newly produced output, and absolute differences.

- **GitHub Actions:** Download the `qe-comparison-plots` artifact.
- **Local runs:** Plots appear in the `_compare_plots` sandbox directory; the failure message prints the path.

**Signs of real regression:**
- Output and reference curves have clearly different shapes
- Peaks move significantly
- Missing or unexpected output files
- Different array sizes
- Large differences across broad regions

**Signs of machine-dependent numerical drift:**
- Curves visually overlap almost everywhere
- Small, smooth difference plots
- Tiny deviations near steep features or zero crossings
- Same code passes on one machine but fails on another

### Current comparison logic

The integration comparison checks `*.dat` files with numeric columns against references. The first column (x-axis) is excluded; remaining columns use tolerance-based comparison. The current tolerance is `0.01`, hardcoded in suite-specific test files. File lists and data shapes must match exactly.

### Resolving failures

**Option 1: Regenerate references** — when the new output is correct and the existing references are outdated.

**Option 2: Adjust the threshold** — only when output is scientifically equivalent to the reference, the mismatch is machine-dependent, and relaxing the threshold will not hide regressions. Make the minimum increase needed. Document the change in the pull request.

**Do not adjust thresholds** if physics appears visually different, files are missing or added unexpectedly, array shapes differ, or failures followed a logic change.

### Debugging checklist

1. Download or open comparison plots.
2. Read failure messages carefully.
3. Confirm file status and shape matching.
4. Assess curve qualitative differences.
5. Classify: regression, stale assets, or numerical drift.
6. Fix code, regenerate references, or adjust tolerance accordingly.
