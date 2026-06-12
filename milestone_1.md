# Claude Prompt: PAOFLOW Documentation Milestone 1

You are helping set up the first controlled documentation milestone for the PAOFLOW codebase.

Repository:

```text
https://github.com/marcobn/PAOFLOW/tree/develop
```

Package/import name:

```text
PAOFLOW
```

Target audience:

```text
Computational materials scientists who want to install PAOFLOW, run a first workflow, understand the QE examples, and browse the Python API.
```

Documentation goal:

Create a **minimal, working Read the Docs documentation skeleton** inspired by the organization and readability of the Pydantic docs, but do **not** attempt to build the full website yet.

Use:

```text
MkDocs + Material for MkDocs + mkdocstrings
```

Do not use Sphinx for this milestone. Do not convert examples to notebooks yet. Do not introduce MyST/Jupyter notebook infrastructure yet. That will be a later milestone.

---

## Strict scope for this milestone

Only create or modify the files needed for the first documentation skeleton:

```text
mkdocs.yml
.readthedocs.yaml
docs/index.md
docs/installation.md
docs/quickstart.md
docs/examples/index.md
docs/examples/qe-examples.md
docs/api/index.md
docs/api/paoflow.md
```

If supporting dependency files must be changed, propose the minimal change first and explain why. Do not edit unrelated package code.

---

## What the milestone must contain

### 1. `mkdocs.yml`

Create a clean MkDocs Material configuration with:

- Site name: `PAOFLOW`
- Repository link: `https://github.com/marcobn/PAOFLOW`
- Material theme
- Search enabled
- Navigation for:
  - Home
  - Installation
  - Quickstart
  - Examples
  - API Reference
- `mkdocstrings` configured for Python API documentation
- Useful Material features such as navigation tabs, code copy, and search suggestions

Keep the configuration simple. Do not add advanced plugins unless they are required for this milestone.

### 2. `.readthedocs.yaml`

Create a minimal Read the Docs config that builds the MkDocs site.

Use a modern Python version supported by Read the Docs.

Install documentation dependencies in the simplest robust way. If the repository already has an appropriate docs requirements file, use it. If not, propose creating one, but do not create a complex dependency structure.

### 3. Landing page: `docs/index.md`

Create a Pydantic-inspired but PAOFLOW-specific landing page with:

- Short one-paragraph description of PAOFLOW
- A clear “What is PAOFLOW?” section
- A “Who is this for?” section aimed at computational materials scientists
- A “Start here” section linking to installation and quickstart
- A “Main documentation sections” overview

Do not overclaim features. If something is not obvious from the repository, mark it as a placeholder or keep it general.

### 4. Installation page: `docs/installation.md`

Create a practical installation page with:

- Recommended installation path
- Developer installation path
- How to verify the installation
- A short troubleshooting section

Do not invent installation commands if they are not supported by the repository. Inspect the repository files first, especially README, pyproject/config files, setup files, and existing install notes.

### 5. Quickstart page: `docs/quickstart.md`

Create a first quickstart page that:

- Uses one simple existing PAOFLOW example from the repository
- Explains what the user will run
- Shows the command or script entry point
- Explains the expected output at a high level

Do not rewrite all examples. Pick one suitable existing example and keep the quickstart focused.

### 6. Examples overview

Create:

```text
docs/examples/index.md
docs/examples/qe-examples.md
```

The examples pages should:

- Explain that examples currently exist as Python scripts
- List the main QE examples found in the repository
- Briefly describe how users should approach them
- Mention that notebook-based tutorials may be added later, but do not implement notebooks now

### 7. API reference

Create:

```text
docs/api/index.md
docs/api/paoflow.md
```

Use `mkdocstrings` to document the top-level PAOFLOW API.

Start small. Do not attempt to document every submodule manually. The goal is only to prove that API documentation generation works.

Example style:

```markdown
::: PAOFLOW
```

If the actual import path differs, inspect the repository and use the correct import path.

---

## Important constraints

- Do not redesign the entire website.
- Do not convert examples to notebooks.
- Do not add MyST yet.
- Do not add a blog.
- Do not add versioned docs yet.
- Do not add complex CI workflows yet.
- Do not change package source code unless absolutely required for docs import to work.
- Do not invent unsupported installation instructions.
- Do not document APIs that cannot currently be imported.
- Prefer placeholders over false claims.
- Keep all pages concise and useful.

---

## Expected final response

After making changes, provide:

1. A list of files created or modified.
2. The exact command to build the docs locally.
3. The exact command to serve the docs locally.
4. Any unresolved issues or assumptions.
5. A short recommendation for Milestone 2.

Do not proceed to Milestone 2.
