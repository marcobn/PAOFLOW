# Milestone 1 — Documentation Skeleton Summary

## Files created

| File | Role |
|---|---|
| `docs/requirements.txt` | Docs build dependencies (`mkdocs`, `mkdocs-material`, `mkdocstrings[python]`) |
| `mkdocs.yml` | MkDocs Material config — navigation, theme, mkdocstrings with `paths: [src]` |
| `.readthedocs.yaml` | RTD build config (ubuntu-22.04, Python 3.12, installs `docs/requirements.txt`) |
| `docs/index.md` | Landing page — what PAOFLOW is, who it's for, start-here links, capability table |
| `docs/installation.md` | pip, conda/HPC, developer install, verification, troubleshooting |
| `docs/quickstart.md` | Walkthrough of `example01` (Silicon) — annotated `main.py`, expected outputs |
| `docs/examples/index.md` | Examples overview — directory layout, how to run, categories table |
| `docs/examples/qe-examples.md` | Table of all 16 QE examples with materials and physics highlights |
| `docs/api/index.md` | API overview — package layout, correct import path |
| `docs/api/paoflow.md` | mkdocstrings directive that auto-generates docs from source docstrings |

---

## Local build & serve

```bash
mkdocs build --strict          # one-shot build → site/
mkdocs serve                   # live-reload dev server at http://127.0.0.1:8000
```

---

## Unresolved issues / assumptions

1. **`:::PAOFLOW.PAOFLOW` renders the module, not the class.** mkdocstrings documents everything in `PAOFLOW/PAOFLOW.py`. To document only the class, change the directive in `docs/api/paoflow.md` to `:::PAOFLOW.PAOFLOW.PAOFLOW`. Check the rendered output and decide.

2. **example15 and example16 are undocumented** in the existing `examples/README`. The QE examples page marks them as "see directory". Fill these in once their contents are clear.

3. **`docs/index.md` uses Material's `grid cards` feature.** Renders correctly with MkDocs Material ≥ 9.x; falls back gracefully on older versions.

4. **No `INSTALL.md` exists** — `README.md` links to one that is missing. The installation page was written from `pyproject.toml` and the README. Cross-check if `INSTALL.md` is added later.

---

## Recommendation for Milestone 2

Convert the top two or three QE examples (Si, Al, Fe SOC) into **MyST-NB executed notebooks** so the docs site renders actual band-structure plots inline. This requires:

- Adding `myst-nb` or `mkdocs-jupyter` to `docs/requirements.txt`
- Enabling the plugin in `mkdocs.yml`
- Converting `example01/main.py` → `docs/examples/example01-silicon.ipynb` with narrative cells and embedded output figures

This is a natural next step once the skeleton is validated on Read the Docs.
