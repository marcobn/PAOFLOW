# Contribution Guidelines

## Branch Model

PAOFLOW uses a Gitflow-inspired workflow with two long-lived branches:

| Branch | Role |
|--------|------|
| `master` | Stable, released code — advances only when `develop` is promoted and tagged |
| `develop` | Integration branch for ongoing work — may be temporarily unstable |

Feature branches follow the naming convention `develop/<feature_name>` and merge back via pull request with at least one approval required.

Bug fixes follow the same path as features through `develop` — there is no separate hotfix branch.

## Pull Request Process

1. Branch from `develop` using the `develop/<feature_name>` convention.
2. Open or comment on a GitHub Issue describing the proposed solution before beginning work — see [Connect](../about/connect.md) for the contribution model.
3. Implement the change. Ensure pre-commit hooks pass (`pre-commit run --all-files`).
4. Write tests. See [Testing](testing.md) for requirements.
5. Open a pull request targeting `develop`. Include a reference to the approved issue.
6. At least one core developer approval is required before merge.

## Coding Standards

PAOFLOW follows these principles for readable, maintainable code:

**Single responsibility.** Each function accomplishes one task.

**Descriptive names.** Functions and variables use clear, unambiguous names.

**Documentation.** Every function and class requires a docstring. See [Docstrings](docstrings.md) for the full style guide.

**DataController keys.** Any function that reads from or writes to the `DataController` dictionary must document the specific keys it accesses or modifies in its docstring.

**Formatting.** Ruff enforces code style automatically via pre-commit hooks. See [Development Environment](dev-environment.md#code-formatting) for setup.

**Tests.** New workflows require example scripts in `examples/` and integration tests in `tests/integration/`. See [Testing](testing.md).

## Review Expectations

Pull requests are reviewed for:

- Correctness relative to the approved issue description
- Adherence to the docstring and formatting standards
- Test coverage — new functionality must include integration tests with reference outputs
- Absence of duplicate documentation — prefer links to the canonical source

Unapproved large contributions may not be accepted. Early discussion in an Issue avoids duplicated effort and conflicting implementations.

## Release Process

On a scheduled basis:

1. `develop` passes the standardized test suite.
2. `develop` merges into `master`.
3. The merge is tagged (e.g., `v2.9.3`).
4. A GitHub Release is created from that tag, including QE test assets.

See [Release Workflow](release.md) for the full asset-generation and publication procedure.
