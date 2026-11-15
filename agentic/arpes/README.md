# ARPES agentic workflow

This workflow compares PAOFLOW's simulated angle-resolved photoemission (ARPES) with measured spectra, for any material: the band maps along a cut with the photoemission matrix elements of the PAO orbitals, the dispersion dressed by the electron-phonon self-energy (the kink, the mass enhancement λ, the renormalised Fermi velocity), the self-energy Σ(ω, T) itself (isotropic from α²F, anisotropic from the Fermi-surface coupling, or the band-resolved Fan-Migdal form), the momentum widths and the phonon linewidths of every state versus temperature.

The model comes from an electron-phonon **system folder** of the `examples/elphon/<System>` layout (`system.py` + the `arpes/` workflow of `PAOFLOW.elphon.workflows`, on top of `superconductivity/`); the adapter `ArpesModel` of `PAOFLOW.agentic.arpes` opens it read-only and extracts what ARPES extracts: MDC dispersions, k_F, v_F, λ, Re/Im Σ, kink energies, linewidths. The experimental data can come in any form: dispersion tables or digitised curves, self-energies extracted by the authors, λ values, linewidths versus temperature, and **band-map images**, which the same MDC analysis reads through an axis calibration (`PAOFLOW.agentic.arpes.images`). Tables, text, PDF figures and decks are ingested with the shared toolkit of the thermo-optics workflow (`PAOFLOW.agentic.thermo_optics`).

An AI agent turns each source into one canonical dataset with provenance, evaluates the model at the measured conditions (temperature, photon energy and polarisation, resolution, cut, with the dispersions aligned at k_F and the bare band defined the same way on both sides), quotes the agreement against explicit baselines (the authors' own analysis, a tunnelling α²F, a Debye-model self-energy, a second measurement), diagnoses the discrepancies and writes publication figures and a report. You steer it with the prompts.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[thermo_optics]"` from the repository root (matplotlib, pillow, PyMuPDF for PDF sources).
- **Have a finished calculation:** a system folder with `superconductivity/eliashberg.py` (and `--fs-coupling` for the anisotropic self-energy) and `arpes/arpes.py` run: `arpes/output/{arpes_bare.npz, arpes_<kind>_<T>K.npz, sigma_<kind>_<T>K.dat, self_energy_fan_migdal.npz, arpes.png}`. The folder stays read-only.
- **Have the experimental sources** in any form, and know the measurement conditions: photon energy, polarisation, temperature, energy and momentum resolution, the cut (direction, k_z), surface preparation, and how the authors defined the bare band when they quote Σ or λ.

## Start a study

```bash
cd PAOFLOW/agentic/arpes
mkdir -p projects/<study>/sources     # or let the agent work next to the calculation
claude                                # or another agent; AGENTS.md / CLAUDE.md load automatically
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. The agent will:
1. inventory the model output (`ArpesModel.describe()`) and the sources;
2. propose a **physics map**: which observable of which source is compared with which model quantity, how the dispersions are aligned (k_F, E_F offset) and the bare band defined, with which baseline;
3. report back with questions before ingesting anything.

Continue with the prompts in `prompts/README.md`. To resume in a later session, use `prompts/11_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: the model pipeline, operating rules, Stages 0–8, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) from the Pb, MgB₂ and Si ARPES examples |
| `prompts/` | kickoff; physics map; data from tables and text; band-map images; self-energy extraction; condition alignment; comparison; diagnosis; figures; report; review; resume; new run |
| `templates/` | the study scripts (`analysis_template.py`, `plot_template.py`), a dataset CSV, a report outline |
| `projects/` | your studies, one folder per study (ignored by git) |

## Worked examples (in the repository examples)

- **Pb, the phonon kink** (`examples/elphon/Pb/README.md` §6): the band map across the Γ–X Fermi crossing of band 6 dressed with the isotropic Σ from α²F (λ = 1.11 from the slope, Z = 0.48, Im Σ plateau 8.7 meV beyond the phonon energies at 10 K, Re Σ extrema at ±7 meV), and why the state-resolved Fan-Migdal form is structureless at affordable q-grids.
- **MgB₂, the σ-band kink** (`examples/elphon/MgB2/README.md` §6): the anisotropic self-energy of every Fermi-surface state (λ_nk from 0.33 on the π sheets to 0.99 on the σ sheets) against the isotropic average.
- **Si, phonon linewidths** (`examples/elphon/Si/README.md` §6): the Fan-Migdal −Im Σ_nk(ε_nk) of the valence and conduction states at 10 / 300 / 600 K (0.6–105 meV), their growth with temperature and distance from the band edges, and the matrix-element weights of the s- and p-like bands at He I.
