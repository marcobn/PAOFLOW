# Superconductivity agentic workflow

This workflow compares PAOFLOW's electron-phonon superconductivity with experiments, for any conventional superconductor:

- the **Eliashberg function** α²F(ω) (shape, peak positions, the cumulative coupling λ(ω)) against tunnelling-inverted spectra (McMillan–Rowell), and its moments λ, ω_log, ω₂;
- the **critical temperature** from McMillan, Allen–Dynes, the isotropic and the anisotropic Migdal–Eliashberg equations against the measured Tc, with the sensitivity to the one free parameter μ*;
- the **gap**: Δ(T), the gap edges and the ratio 2Δ(0)/k_BTc against tunnelling, ARPES, STM or μSR data; the quasiparticle density of states against the tunnelling conductance;
- **multi-gap** metals: the distribution of Δ_nk and λ_nk over the Fermi surface, its peaks and sheet averages (MgB₂: σ and π gaps) against point-contact and ARPES gaps;
- **mass enhancement**: λ against the specific-heat and de Haas–van Alphen enhancements (1 + λ; the latter through the `agentic/skeaf` workflow) and against the ARPES kink (`agentic/arpes`).

The model comes from an electron-phonon **system folder** of the `examples/elphon/<System>` layout (`system.py` + the `superconductivity/` workflow of `PAOFLOW.elphon.workflows`: `eliashberg.py`, `me.py`, `me_aniso.py`); the adapters `EliashbergModel`, `GapModel` and `AnisotropicGapModel` of `PAOFLOW.agentic.superconductivity` open it read-only, and `baselines` provides the reference formulas (moments, McMillan and Allen–Dynes Tc and their inversion, the BCS gap and its universal Δ(T), the strong-coupling gap ratio). The experimental data can come in any form (tables, text, PDF figures, decks, images) and are ingested with the shared toolkit of the thermo-optics workflow (`PAOFLOW.agentic.thermo_optics`).

An AI agent turns each source into one canonical dataset with provenance, evaluates the model at the measured conditions (temperature, μ*, the same definition of the gap), quotes the agreement against explicit baselines (published calculations such as EPW on the same grids, the BCS curve, a second measurement, the Tc-inverted λ), diagnoses the discrepancies (grids, smearing, μ*, anisotropy, spin-orbit, anharmonicity) and writes publication figures and a report. You steer it with the prompts.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[thermo_optics]"` from the repository root (matplotlib, pillow, PyMuPDF for PDF sources).
- **Have a finished calculation:** a system folder with `superconductivity/eliashberg.py` run (`superconductivity/output/eliashberg.npz`, `alpha2F.dat`; with `--fs-coupling` also `fs_coupling.npz`), and as needed `me.py` (`output/me/migdal_eliashberg.npz`, `gap_vs_T.dat`) and `me_aniso.py` (`output/me_aniso/migdal_eliashberg_aniso.npz`, `gap_vs_T_aniso.dat`). The folder stays read-only.
- **Have the experimental sources** in any form, and know the measurement: method (tunnelling, ARPES, STM, point contact, specific heat, μSR), sample (single crystal, film, purity), temperature, and how the gap is defined (Δ(iω₀)-like order parameter versus the spectral gap edge, FWHM of a coherence peak, BCS fit).

## Start a study

```bash
cd PAOFLOW/agentic/superconductivity
mkdir -p projects/<study>/sources     # or let the agent work next to the calculation
claude                                # or another agent; AGENTS.md / CLAUDE.md load automatically
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. The agent will:
1. inventory the model output (`describe()` of the adapters) and the sources;
2. propose a **physics map**: which observable of which source is compared with which model quantity, with which definition and μ*, against which baseline;
3. report back with questions before ingesting anything.

Continue with the prompts in `prompts/README.md`. To resume in a later session, use `prompts/11_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: the model pipeline, operating rules, Stages 0–8, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) from the Pb and MgB₂ examples and the EPW comparisons |
| `prompts/` | kickoff; physics map; data from tables, text and figures; condition alignment; comparison; diagnosis; figures; report; review; resume; new run |
| `templates/` | the study scripts (`analysis_template.py`, `plot_template.py`), a dataset CSV, a report outline |
| `projects/` | your studies, one folder per study (ignored by git) |

## Worked examples (in the repository examples)

- **Pb** (`examples/elphon/Pb/README.md`): λ = 1.22, ω_log = 4.50 meV, Tc = 4.77 / 5.21 K (McMillan / Allen–Dynes, μ* = 0.1) at 48³/24³ against EPW (1.15, 4.44 meV, 4.38 / 4.76 K) and the measured Tc = 7.2 K, λ ≈ 1.55 (tunnelling); isotropic Migdal–Eliashberg Δ(iω₀) = 1.016 meV at 0.3 K, Tc 5.7 K; (Z₀ − 1)/λ = 0.925 in both codes.
- **MgB₂** (`examples/elphon/MgB2/README.md`): λ = 0.58, ω_log = 62 meV, Tc = 14.7 K (Allen–Dynes) reproducing EPW to 1 % on the tutorial grids; the anisotropic equations give two gaps (π ≈ 2 meV, σ ≈ 5–8 meV at 5 K) and about twice the isotropic Tc (measured 39 K, gaps ≈ 2 and 7 meV); the equal-grid rule and the k-convergence of N_F.
