# Thermo-optics agentic workflow

This workflow compares PAOFLOW's temperature-dependent optical response with experiments, for any material: absorption (direct and phonon-assisted edges), free-carrier absorption, refractive index, reflectance and slab emissivity. The method itself is not tied to optics: another observable (transport, mobility, spectra, phonons) needs only a new model adapter (`templates/model_adapter_template.py`).

The experimental data can come in any form:
- tables (CSV, XLSX, paper tables);
- free text;
- PDF figures, whether vector-drawn or scanned;
- slide decks;
- images.

An AI agent turns each source into one canonical dataset with provenance, evaluates the model at the measured conditions, and quotes the agreement against explicit baselines. It then diagnoses the discrepancies and writes publication figures and a report. You steer it with the prompts. The library is `PAOFLOW.agentic.thermo_optics`.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[thermo_optics]"` from the repository root (PyMuPDF for PDFs, matplotlib, pillow). Slide decks (`.pptx`) are read with the standard library.
- **Have a finished calculation:** its output folder, which stays read-only, and its driver script.
- **Have the experimental sources** in any form, and know their provenance: raw, transcribed, or replotted in a later paper.

## Start a study

```bash
cd PAOFLOW/agentic/thermo_optics
mkdir -p projects/<study>/sources     # or let the agent work next to the calculation (see below)
claude                                # or another agent; AGENTS.md / CLAUDE.md load automatically
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. The agent will:
1. inventory the model output and the sources;
2. propose a **physics map**: which observable of which source is compared with which model quantity, at which conditions, and with which baseline;
3. report back with questions before digitising anything.

Continue with the prompts in `prompts/README.md`. To resume in a later session, use `prompts/13_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: operating rules, Stages 0–8, ingestion by form, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) from the Si absorption/emissivity study |
| `prompts/` | kickoff; ingestion of tables, text, vector PDF, raster, decks; alignment; comparison; diagnosis; figures; report; review; resume; new model run |
| `templates/` | the three study scripts, a model-adapter skeleton, a dataset CSV, a report outline |
| `projects/` | your studies, one folder per study (ignored by git) |

## Worked example

`EPW_Si_large/EPW_Si/thermo_optics/` is a study on the user's workstation (not bundled). It compares PAOFLOW phonon-assisted absorption and thermal emissivity of Si (`../output`) with experiments from five figures in three papers:
- Noffsinger 2012: vector PDF, and an embedded JPEG with open circles;
- Fan 1956: scanned line;
- Sato 1967: scanned emissivity curves with dashed branches, and dots on fitted lines.

The study contains `digitize_sources.py`, `analysis.py`, `plot_figures.py`, `REPORT.md`, `checks/` and `figures/`. Its methods are summarised in `AGENT_WORKFLOW.md` and its pitfalls in `LESSONS_LEARNED.md`.
