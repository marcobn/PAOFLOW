# Developer's Guide

## Pre-commit Hooks

This repository uses **pre-commit** to enforce basic code hygiene and consistent Python linting/formatting _before code is committed_.
The goal is to catch obvious issues early, reduce review noise, and keep formatting debates out of pull requests.

### Requirements

- **pre-commit ≥ 2.20.0**
- Python hooks run using **python3**

Install pre-commit (preferably inside your virtual or conda environment):

```bash
python3 -m pip install -U pre-commit
```

Verify installation:

```bash
pre-commit --version
```

### One-time Setup

After cloning the repository, install the git hooks:

```bash
pre-commit install
```

This ensures the checks run automatically on every `git commit`.

(Optional) Update hook versions periodically:

```bash
pre-commit autoupdate
```

### Excluded Paths

Pre-commit hooks do **not** run on the following paths:

- `BASIS/`
- `examples/`

These directories typically contain generated output, third-party code, or example material where strict enforcement is unnecessary or counterproductive.

### Hooks Executed on Commit

#### General Repository Hygiene

Provided by `pre-commit/pre-commit-hooks`:

- **Remove trailing whitespace**
  Strips whitespace at the end of lines.
- **End-of-file fixer**
  Ensures files end with a single newline.
- **YAML / JSON / TOML validation**
  Fails commits containing invalid configuration or data files.
- **Merge conflict detection**
  Prevents committing unresolved conflict markers.
- **Line ending normalization (LF)**
  Converts mixed or Windows-style line endings to LF for cross-platform consistency.
- **Large file check (max 1 MB)**
  Prevents accidentally committing large files. Large artifacts should live outside the repository (e.g. LFS, releases, artifact storage).
- **Double-quote string fixer (Python)**
  Normalizes string quoting where safe to reduce stylistic churn.

#### Python Linting and Formatting (Ruff)

Python code is checked using **Ruff**, which provides fast linting and formatting.

- **Ruff linting**
  - Automatically applies safe fixes.
  - If fixes are applied, the hook fails so changes must be reviewed and re-staged.
- **Ruff formatting**
  - Applies a consistent formatter to Python files.
  - Formatting rules can be agreed upon by the team or based on reasonable defaults offered by Ruff.

### Pre-commit in CI (Main Branch Enforcement)

All commits that target the **main branch** run the same pre-commit hooks in CI.

This means:

- Any commit or pull request that does not pass pre-commit **will fail CI**
- Formatting or linting issues caught locally will also be caught in the main branch checks

To avoid unnecessary CI failures and noisy pull request iterations, developers are **strongly encouraged** to run pre-commit locally and fix any issues _before_ opening a pull request.

#### Recommended Workflow

In most cases, committing code “just works”:

```bash
git commit
```

If a hook modifies files, the commit will fail. Re-stage and commit again:

```bash
git status
git add -A
git commit
```

## Running Hooks Manually

Run all hooks against all files (recommended after rebases or large refactors):

```bash
pre-commit run --all-files
```

Run individual hooks:

```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```

Running pre-commit locally makes failures easier to diagnose and keeps pull requests focused on code changes, not formatting or hygiene fixes.

## Troubleshooting

### `pre-commit` command not found

Ensure it is installed in the active environment.

### Broken hook environments

Reset and re-run:

```bash
pre-commit clean
pre-commit run --all-files
```
