# SSCHA agentic workflow: anharmonic phonons at temperature

This workflow runs and interprets stochastic self-consistent harmonic approximation (SSCHA) campaigns with
PAOFLOW, for any crystal (metal or insulator) on any machine (a workstation, a Slurm cluster, or anything else
through the manual launcher). Typical questions:

- Is a lattice that is unstable in the harmonic approximation stabilised by anharmonicity at temperature T?
- Above which temperature does it become stable (the stability temperature T_s), and how does that compare with
  the measured transition, eutectoid or decomposition temperature?
- What are the anharmonic phonon dispersions at T, and the temperature-dependent force constants that the thermal
  (special-displacement) cells and the electron-phonon coupling need?

The engine is PAOFLOW's SSCHA interface (`PAOFLOW.sscha_setup`, `sscha_population`, `sscha_minimize`,
`sscha_hessian`, `sscha_force_constants`, `sscha_status`; python-sscha and CellConstructor underneath), driven by
`main.sscha.py` drivers that `paoflow-gen` writes (workflow `sscha`). An AI agent plans the campaign (supercell,
k-grid, electrons, populations, temperatures), generates and launches the drivers, monitors the runs through the
read-only adapter `SSCHARun` of `PAOFLOW.agentic.sscha`, diagnoses convergence problems, brackets the stability
temperature, compares it with experiment, hands the force constants to the downstream workflows and writes
figures and a report. You steer it with the prompts.

Nothing here depends on a particular machine or system: the launcher is a setting of the driver, the scheduler
directives are plain lines you give once, and the state of every run lives in its own directory (`state.json`),
so a campaign can be continued from any session and moved between machines by copying the run directories.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[sscha]"` (python-sscha, CellConstructor, ASE; a Fortran
  compiler and BLAS/LAPACK are needed: `docs/internals/sscha.md`, "Installation"). On a cluster, install it in
  the environment the jobs load.
- **Have a converged scf of the primitive cell** (the `<prefix>.save` the drivers read: cutoffs, pseudopotentials,
  occupations). For metals at high temperature decide the electronic temperature of the force runs
  (`LESSONS_LEARNED.md` #4).
- **Harmonic phonons:** a `ph.x` q-grid run that contains the q-points of the supercell (a 2x2x2 cell needs a grid
  that is a multiple of 2), or let the driver's `harmonic` phase run it.
- **The experimental reference**, if the study compares with one: transition, eutectoid or decomposition
  temperature with its source and uncertainty; measured phonon frequencies at temperature (inelastic neutron or
  X-ray scattering, Raman).

## Start a study

```bash
cd PAOFLOW/agentic/sscha
mkdir -p projects/<study>
claude
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. Continue with `prompts/README.md`; resume
with `prompts/11_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: the engine, operating rules, Stages 0–8, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) from the δ-TaN campaign and the implementation tests |
| `prompts/` | kickoff; plan; harmonic start; drivers and launch; monitoring; diagnosis; Hessian and stability; temperature scan; downstream; figures; report; review; resume |
| `templates/` | `campaign_template.py` (drivers of a scan), `analysis_template.py` (tables, stability temperature), `plot_template.py` (figures), `report_template.md` |
| `projects/` | your studies (ignored by git) |

## Worked example

δ-TaN (rocksalt), the prototype of this workflow: harmonically unstable at X (−264 cm⁻¹ with cold electrons,
−223 cm⁻¹ with Fermi–Dirac electrons at 2800 K), stabilised by anharmonicity at 2800 and 4500 K (2×2×2 cell, 6³
k-grid, populations of 100–200 configurations, 800 for the Hessian); the stability-temperature scan at 1000, 1500
and 2000 K against the measured δ-TaN eutectoid. Production folders and report:
`Workspace/PAOFLOW/production/XTa_project/` (outside the repository).
