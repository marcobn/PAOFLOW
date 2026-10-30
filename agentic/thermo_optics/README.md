# Thermo-optics agentic workflow

This workflow compares PAOFLOW's temperature-dependent optical response with experiments, for any material: absorption (direct and phonon-assisted edges), free-carrier absorption, refractive index, reflectance and slab emissivity. The model comes from an electron-phonon **system folder** of the `examples/elphon/<System>` layout (`system.py` + the `optics/` workflow of `PAOFLOW.elphon.workflows`); the adapter `OpticsModel` opens it read-only. The method itself is not tied to these observables: the sibling workflows `agentic/transport/` (mobilities, resistivity, emissivity of metals), `agentic/superconductivity/` (α²F, Tc, gaps) and `agentic/arpes/` (band maps, kinks, linewidths) reuse this folder's toolkit (`datasets`, `units`, `ingest`, `digitize`, `metrics`, `plotting`) with their own adapters, and another observable needs only a new adapter (`templates/model_adapter_template.py`).

The experimental data can come in any form:
- tables (CSV, XLSX, paper tables);
- free text;
- PDF figures, whether vector-drawn or scanned;
- slide decks;
- images.

An AI agent turns each source into one canonical dataset with provenance, evaluates the model at the measured conditions, and quotes the agreement against explicit baselines. It then diagnoses the discrepancies and writes publication figures and a report. You steer it with the prompts. The library is `PAOFLOW.agentic.thermo_optics`.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[thermo_optics]"` from the repository root (PyMuPDF for PDFs, matplotlib, pillow). Slide decks (`.pptx`) are read with the standard library.
- **Have a finished calculation:** an electron-phonon system folder (`examples/elphon/<System>` layout, written by `paoflow-gen elphon` or copied from the examples) with the runs of `optics/absorption.py` (`--emissivity --direct-dielectric ../qe/optics/output` for the slab emissivity) in `optics/output*/`. The folder stays read-only; `OpticsModel.from_system(folder)` reads it and records its `system.py` for the report. Carrier transport (`transport/mobility.py`) is the subject of `agentic/transport/`.
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
| `templates/` | the three study scripts, a model-adapter skeleton (the built-in adapter is `OpticsModel`; `MobilityModel`, `ResistivityModel`, `EliashbergModel`, `GapModel` and `ArpesModel` live in the sibling workflows), a dataset CSV, a report outline |
| `projects/` | your studies, one folder per study (ignored by git) |

## Worked example

`Workspace/PAOFLOW/production/Si_optics_emissivity/study/thermo_optics/` is a study on the user's workstation (not bundled). It compares PAOFLOW phonon-assisted absorption and thermal emissivity of Si (the converged run `../runs/production`: PBE + scissor to the Varshni gap, position dipoles, Poncé screening, 60³/30³ dense grids, 26 temperatures; an earlier eACBN0 phase lives in `../../coarse_data/thermo_optics/`) with experiments from five figures in three papers:
- Noffsinger 2012: vector PDF, and an embedded JPEG with open circles;
- Fan 1956: scanned line;
- Sato 1967: scanned emissivity curves with dashed branches, and dots on fitted lines.

The study contains `digitize_sources.py`, `analysis.py`, `plot_figures.py`, `REPORT.md`, `checks/` and `figures/`. Its methods are summarised in `AGENT_WORKFLOW.md` and its pitfalls in `LESSONS_LEARNED.md`.
