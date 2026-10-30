# Non-linear optics agentic workflow

This workflow compares PAOFLOW's second-order optical responses with experiments and published
calculations, for any non-centrosymmetric crystal: the SHG susceptibility χ^{(2)}(−2ω; ω, ω)
(static values, the d coefficients, spectra of |χ|, Re χ, Im χ), the shift current σ^{abc}(ω)
(bulk photovoltaic effect), the circular and magnetic injection currents, and the magnetic shift
current of T-broken crystals, in 3D or as sheet responses of 2D crystals.

The model is an output folder of `PAOFLOW.nonlinear_optics` (`examples/nlo_examples/<System>`
layout: QE scf/nscf inputs, `main.py`, `output/`); the adapter `NonlinearModel` of
`PAOFLOW.agentic.nlo` opens it read-only and gives spectra, tensors, static values, peaks,
onsets, the point-group constraints (`symmetry`: forbidden components, relations between the
allowed ones, the symmetry error of the run), the Kleinman check and the unit conversions
(χ ↔ d, sheet ↔ effective 3D). Experimental data (SHG coefficients, |χ(ω)| spectra, photocurrent
spectra) in any form are ingested with the shared toolkit of the thermo-optics workflow
(`PAOFLOW.agentic.thermo_optics`: `datasets`, `units`, `ingest`, `digitize`, `metrics`, `plotting`).

An AI agent turns each source into one canonical dataset with provenance, evaluates the model at
the measured conditions (photon energy, polarisation geometry, the component and convention of
the quoted coefficient, scissor to the measured gap), quotes the agreement against explicit
baselines (published calculations at the same level of theory, Miller's rule or Kleinman-symmetric
estimates, the spread between measurements) and against the symmetry checks of the run, diagnoses
the discrepancies (k-convergence, broadening, scissor, degeneracy tolerance, the symmetry of the
PAO Hamiltonian, basis projectability, excitons and local fields outside the model) and writes
publication figures and a report. You steer it with the prompts.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[thermo_optics]"` (matplotlib, pillow,
  PyMuPDF for PDF sources).
- **Have a finished calculation:** an `examples/nlo_examples/<System>`-like folder with the
  `output*/` of `main.py` (`shift_`, `shiftm_`, `injectc_`, `injectm_`, `shgr_`, `shgi_`,
  `shga_<abc>.dat`), its `main.py` (grid, η, scissor, components, `twoD`) and the QE inputs
  (ultrasoft / PAW inputs need the augmented projections of October 2026 or later: `LESSONS_LEARNED.md` #2). The folder stays read-only.
- **Have the experimental sources** and know the conventions: which coefficient is quoted
  (d_ijk = χ_ijk/2 in pm/V, or |χ|), at which photon energy, with which reference material, which
  component and crystal orientation; for photocurrents the illumination geometry and whether the
  quoted conductivity is 3D or a sheet value divided by a thickness.

## Start a study

```bash
cd PAOFLOW/agentic/nlo
mkdir -p projects/<study>/sources
claude
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. Continue with
`prompts/README.md`; resume with `prompts/11_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: the model pipeline, operating rules, Stages 0–8, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) from the GaAs, GeS and graphene validation |
| `prompts/` | kickoff; physics map; data from tables/text and figures; symmetry audit; condition alignment; comparison; diagnosis; figures; report; review; resume; new run |
| `templates/` | `analysis_template.py`, `plot_template.py`, a dataset CSV, a report outline |
| `projects/` | your studies (ignored by git) |

## Worked example

`examples/nlo_examples/REPORT.md` (in the repository): GaAs χ_xyz and σ_xyz against the
low-frequency SHG measurements and published calculations, monolayer GeS against the shift-current
literature, gapped graphene against the D3h selection rules; the symmetry and tolerance diagnostics
behind the defaults of `nonlinear_optics`.
