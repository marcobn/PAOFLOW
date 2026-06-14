# Testing

Testing is a first-class part of contributing to PAOFLOW. Good tests protect everyone's work — they catch regressions early and give reviewers confidence that new code behaves as intended.

## Test Categories

**Unit tests** (`tests/unit/`) are fast and isolated, targeting individual functions or classes. File names start with `test_` and test functions follow the pattern `test_<function_name>()`.

**Integration tests** (`tests/integration/`) run a minimal end-to-end workflow and compare output against known-good reference data. We keep them fast by using lower accuracy settings (reduced grid sizes and k-points) and staging large prerequisites as release assets rather than committing them to the repository.

## Running Tests

```bash
# All tests
pytest -q tests

# All unit tests
pytest -q tests/unit

# A specific unit test file
pytest -q tests/unit/transport/calculators/test_current.py::test_build_bias_grid_linspace

# All integration tests for a family
pytest -q tests/integration/transport

# A specific integration example
pytest -q tests/integration/transport -k example01
```

Drop the `-q` flag if you'd like more detailed output.

## Integration Test Patterns

We have three established patterns for integration tests. When adding a new family, pick the one that fits rather than inventing something new — consistency makes the test suite much easier to navigate.

**Pattern A: Split assets** — for examples that need both expensive runtime inputs (QE `.save` directories, binary artifacts) and separate reference outputs for comparison. One archive per purpose. Typical for QE and VASP examples.

**Pattern B: Combined assets** — when runtime inputs and reference outputs naturally belong together in a single archive. Typical for transport examples.

**Pattern C: Reference-only assets** — for self-contained Python-only examples that run from committed source files. One archive with just the expected output files.

## Adding Integration Tests for a New Example Family

Here's a recommended layout to start from:

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

A good integration test answers the question: "Can this example still run successfully and produce numerically consistent results in an isolated environment?" To that end, it should:

- Run the minimum workflow needed to exercise the feature.
- Avoid expensive external calculations during CI.
- Compare generated outputs against known-good references.
- Pull large inputs from release assets rather than the repository (see [Release Workflow](release.md)).

## What To Do If Integration Tests Fail

When a test fails, the first step is figuring out *why* before making any changes. Failures usually fall into one of three buckets: a real regression, missing or wrong test assets, or small numerical differences between machines.

### Reading the comparison plots

Every failed test generates comparison plots showing reference data, fresh output, and the absolute difference — these are the fastest way to understand what's happening.

- **In GitHub Actions:** Download the `qe-comparison-plots` artifact.
- **Locally:** Plots appear in `_compare_plots/`; the failure message prints the path.

**Signs of a real regression:**
- Output and reference curves have clearly different shapes
- Peaks shift significantly
- Unexpected output files appear or expected ones go missing
- Array sizes differ
- Large differences spread across broad regions

**Signs of machine-dependent numerical drift:**
- Curves overlap almost everywhere visually
- Differences are small and smooth
- Tiny deviations near steep features or zero crossings
- The same code passes on one machine but fails on another

### Resolving the failure

**Option 1: Regenerate references** — the right choice when the new output is correct and the existing references are outdated.

**Option 2: Adjust the threshold** — acceptable when the output is scientifically equivalent, the mismatch is clearly machine-dependent, and relaxing the threshold won't mask real regressions. The current tolerance is `0.01`, hardcoded in the suite-specific test files. Make the smallest increase that fixes the failure, and document the reason in your pull request.

**Please don't adjust thresholds** if physics looks visibly different, files are unexpectedly missing or added, array shapes differ, or the failure followed a logic change. These are signs of a real problem worth investigating.

### Debugging checklist

1. Open the comparison plots.
2. Read the failure message carefully.
3. Confirm file counts and array shapes match.
4. Assess whether curves are qualitatively different or just numerically close.
5. Classify the failure: regression, stale assets, or numerical drift.
6. Fix code, regenerate references, or adjust tolerance as appropriate.
