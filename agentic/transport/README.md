# Transport agentic workflow

This workflow compares PAOFLOW's electron-phonon transport with experiments, for any material:

- **semiconductors**: the phonon-limited drift mobilities of electrons and holes (SERTA and MRTA), carrier densities, conductivity and Seebeck coefficient versus temperature and doping;
- **metals**: the resistivity ρ(T) from the transport Eliashberg function α²F_tr with residual (impurity) and electron-electron channels, λ_tr, the plasma frequency, the extended-Drude infrared optics and the thermal emissivity of the metal;
- **the Boltzmann bridge**: the same scattering channels, plus impurity channels fitted to a sample, inside `PAOFLOW.transport(scattering_channels=...)`.

The model comes from an electron-phonon **system folder** of the `examples/elphon/<System>` layout (`system.py` + the `transport/` and `superconductivity/` workflows of `PAOFLOW.elphon.workflows`); the adapters `MobilityModel` and `ResistivityModel` of `PAOFLOW.agentic.transport` open it read-only. The experimental data can come in any form (tables, free text, PDF figures, slide decks, images): they are ingested with the shared toolkit of the thermo-optics workflow (`PAOFLOW.agentic.thermo_optics`: `datasets`, `units`, `ingest`, `digitize`, `metrics`, `plotting`).

An AI agent turns each source into one canonical dataset with provenance, evaluates the model at the measured conditions (temperature, doping, residual resistivity, drift versus Hall mobility), quotes the agreement against explicit baselines (published calculations, a second measurement, Bloch-Grüneisen or power-law fits), diagnoses the discrepancies and writes publication figures and a report. You steer it with the prompts.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[thermo_optics]"` from the repository root (matplotlib, pillow, PyMuPDF for PDF sources).
- **Have a finished calculation:** a system folder with the runs of `transport/mobility.py` (semiconductors: `transport/<tag>/mobility.dat`, `carrier_scattering.npz`, `scattering_rates.npz`) and/or `superconductivity/eliashberg.py --transport` followed by `transport/resistivity.py` (metals: `superconductivity/output/eliashberg.npz`, `transport/<tag>/resistivity.dat`, `emissivity.npz`). The folder stays read-only.
- **Have the experimental sources** in any form, and know their provenance (raw, transcribed, digitised, replotted) and the sample: doping, purity (residual resistivity or RRR), measurement method (Hall, time-of-flight, four-probe).

## Start a study

```bash
cd PAOFLOW/agentic/transport
mkdir -p projects/<study>/sources     # or let the agent work next to the calculation
claude                                # or another agent; AGENTS.md / CLAUDE.md load automatically
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. The agent will:
1. inventory the model output (`describe()` of the adapters) and the sources;
2. propose a **physics map**: which observable of which source is compared with which model quantity, at which conditions, with which baseline;
3. report back with questions before ingesting anything.

Continue with the prompts in `prompts/README.md`. To resume in a later session, use `prompts/11_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: the model pipeline, operating rules, Stages 0–8, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) from the Si mobility and Pb resistivity studies |
| `prompts/` | kickoff; physics map; data from tables, text and figures; condition alignment; comparison; Boltzmann channels; diagnosis; figures; report; review; resume; new run |
| `templates/` | the study scripts (`analysis_template.py`, `plot_template.py`), a dataset CSV, a report outline |
| `projects/` | your studies, one folder per study (ignored by git) |

## Worked examples (on the user's workstation, not bundled)

- **Si mobility** (`Workspace/PAOFLOW/production/Si_mobility_quadrupoles/`): phonon-limited electron and hole mobilities at 300 K on 60³, 72³ and 84³ dense grids (12³ nscf / 6³ DFPT q, 10 meV smearing) against the published calculation of Poncé *et al.* 2018 (1343 / 722 cm²/Vs, EPW) and the measured drift mobilities (≈ 1400 / 470–500 cm²/Vs); the table is in `examples/elphon/Si/README.md` §5.
- **Pb resistivity** (`Workspace/PAOFLOW/production/Pb_superconductivity_transport/`, `examples/elphon/Pb/README.md` §5): ρ(T) from α²F_tr (λ_tr = 0.94, ħω_p = 8.34 eV at 48³/24³) against the measured 21 µΩ cm at 300 K, the Bloch-Grüneisen baseline and the convergence of λ_tr with the grids and the smearing.
