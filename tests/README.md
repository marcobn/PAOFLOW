# Tests

This folder contains unit and integration tests for PAOFLOW.

## Layout

- unit/: fast, isolated tests of small units
- integration/: run smaller, modified versions of examples
- integration/examples/: example-derived inputs and scripts used by integration tests
- fixtures/: shared test data
- utils/: shared test helpers

## Running tests

- All tests: pytest
- Unit only: pytest -m unit
- Integration only: pytest -m integration

## Integration tests

### QE examples

QE integration tests live under tests/integration/qe and mirror the example layout
from examples/qe*examples. Each example folder contains QE input files, main.py, and
Reference data. The test runs each example in a sandbox, runs PAOFLOW, and compares
output/*.dat against Reference/\_.dat using the same tolerance as the legacy
check_test.py script.

Run PAOFLOW only (assumes a \*.save directory exists):

```bash
pytest -m integration tests/integration/qe
```

QE executables are intentionally not supported via pytest. Use the asset-bundle
workflow described in tests/integration/qe/README.md to generate `*.save/` and
`Reference/` on an HPC system and run PAOFLOW-only in CI.

### Transport examples

Transport integration tests live under tests/integration/transport and are based on
trimmed transport examples. Each example is copied into a sandbox, asset-bundle
content is overlaid into the job directory, the example scripts are executed, and
output/paoflow/_.dat is compared to Reference/_.dat.

Transport integration tests require a combined asset tarball containing both
`*.save/` and `Reference/` data. Use the asset-bundle workflow described in
tests/integration/transport/README.md to generate and publish that archive.

Run transport integration tests:

```bash
pytest -m integration tests/integration/transport
```

#### More examples

Example: To run all integration tests for transport, execute the following command from the repository root:

```bash
pytest -q tests/integration/transport
```

To run a specific example test:

```bash
pytest -q tests/integration/transport -k example01
```

To run all unit tests:

```bash
pytest -q tests/unit
```

To run a unit test for a specific function (replace `test_module.py` and `test_function` with the actual file and test name):

```bash
pytest -q tests/unit/transport/calculators/test_current.py::test_build_bias_grid_linspace
```

Remove `-q` for more verbose pytest output.
