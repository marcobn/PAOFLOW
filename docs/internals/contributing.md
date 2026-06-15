# Contribution Guidelines

Welcome, and thank you for wanting to contribute to PAOFLOW! This page covers how the project is organized, how changes get merged, and what we look for in contributions.

## Branch Model

We use a Gitflow-inspired workflow with two long-lived branches:

| Branch    | Role                                                                        |
| --------- | --------------------------------------------------------------------------- |
| `master`  | Stable, released code — only advances when `develop` is promoted and tagged |
| `develop` | The integration branch for ongoing work — may be temporarily unstable       |

When you start work on something new, branch from `develop` using the naming convention `develop/<feature_name>`. Bug fixes follow the same path as features — there's no separate hotfix branch to worry about.

Pull requests to `develop` need at least one approval before they merge.

## Pull Request Process

1. Branch from `develop` using `develop/<feature_name>`.
2. Before writing any code, open or comment on a GitHub Issue describing what you'd like to do — see [Contact](../about/connect.md) for more on the discussion-first model.
3. Implement your change. Run `pre-commit run --all-files` to make sure formatting is tidy.
4. Add tests. Take a look at [Testing](testing.md) to see what's expected.
5. Open a pull request targeting `develop` and link it to the approved issue.
6. A core developer will review and merge once it's ready.

## Coding Standards

We care a lot about code being readable and maintainable — not just correct. Here's what that means in practice:

**Single responsibility.** Each function does one thing. If a function is doing several things at once, it's usually a sign it should be split up.

**Descriptive names.** Good names make code self-documenting. Choose names that clearly express what a function does or what a variable holds.

**Docstrings.** Every function and class needs one. See [Docstrings](docstrings.md) for our style guide — we follow NumPy docstring format with sections for Parameters, Returns, Notes, and Examples where applicable.

**DataController documentation.** If your function reads from or writes to the `DataController` dictionary, list the specific keys it accesses or modifies in the docstring. This makes the data flow traceable.

**Formatting.** Ruff handles this automatically through pre-commit hooks. See [Development Environment](dev-environment.md#code-formatting) for setup.

**Tests.** New workflows need example scripts in `examples/` and integration tests in `tests/integration/`. See [Testing](testing.md).

## Review Expectations

When reviewing pull requests, we look at:

- Correctness relative to the approved issue
- Docstring and formatting standards
- Test coverage — new functionality needs integration tests with reference outputs
- Documentation — if your change adds something users or developers need to know about, please document it
- Avoiding duplication — link to the canonical source rather than restating the same information

We try to give feedback that's constructive and specific. If something isn't quite right, we'll explain why and work with you to get it there.

## Release Process

On a scheduled basis, `develop` gets promoted to `master`:

1. `develop` passes the full test suite.
2. `develop` merges into `master`.
3. The merge is tagged (e.g., `v2.9.3`).
4. A GitHub Release is created from that tag, including QE test assets.

See [Release Workflow](release.md) for the full asset-generation and publication procedure.
