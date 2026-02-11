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
