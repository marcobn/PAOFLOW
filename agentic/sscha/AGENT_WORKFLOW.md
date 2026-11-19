# SSCHA workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) running **anharmonic phonon campaigns at temperature**
with PAOFLOW's SSCHA interface, for **any crystal** on **any machine**: deciding whether a harmonically unstable
lattice is stabilised at T, bracketing the **stability temperature** and comparing it with experiment, and
producing the **temperature-dependent force constants** that the thermal (special-displacement) cells and the
electron-phonon coupling consume. Read `LESSONS_LEARNED.md` once before starting.

A study has three scripts, each started from `templates/`:

| Script | Reads | Writes |
|---|---|---|
| `campaign.py` (from `campaign_template.py`) | the settings of the campaign | one `main.sscha.py` per machine / run folder (through `build_sscha_script`) |
| `analysis.py` | the run directories (read-only, `SSCHARun`), `data/*.csv` | `tables/runs.csv`, `tables/scan.csv`, `tables/stability.md`, `tables/diagnostics.md` |
| `plot_figures.py` | the run directories and `tables/` | `figures/*.pdf` + `*.png` |

```python
from PAOFLOW.agentic.sscha import SSCHARun, find_runs, scan_table, stability_temperature, next_temperature
from PAOFLOW.agentic.sscha import dispersion, grid_minimum
from PAOFLOW.agentic.sscha.plotting import plot_convergence, plot_scan, plot_dispersions, publication_style, save_figure
from PAOFLOW.gen.sscha_driver import build_sscha_script
```

## The engine

| step | PAOFLOW call (driver phase) | what happens | cost |
|---|---|---|---|
| harmonic start | `ph.x` on a q-grid containing the supercell q (`harmonic`) | the starting matrices and the harmonic support of the interpolation | one DFPT run |
| setup | `sscha_setup(T, supercell, harmonic_dyn, ...)` | the q of the supercell grid; imaginary modes made positive (or `start_from` a converged run); `state.json` | seconds |
| population | `sscha_population()` | N configurations from the current Gaussian; pw.x inputs with the parent's cutoffs and pseudopotentials and the chosen electrons | seconds |
| forces | pw.x on every `cfg_XXXX.in` (launcher) | `cfg_XXXX.out`; a population is used when 95 % finished | N × one supercell scf: the whole cost |
| minimisation | `sscha_minimize()` | free energy minimised over the auxiliary force constants with reweighting until the Kong–Liu ratio drops; `pop<p>/summary.json` | minutes |
| Hessian | `sscha_population()` after convergence, pw.x, `sscha_hessian()` | the physical phonons (free-energy Hessian, bubble term) and the stability verdict on the supercell grid | one population (larger) |
| export | `sscha_force_constants('auxiliary'|'hessian')` | anharmonic correction of the supercell grid added to the harmonic matrices of the ph.x grid; QE matrices and `.fc` | seconds |
| next step | `sscha_status()` (`status`) | `generate`, `forces`, `minimize`, `hessian`, `export`, `done` or `failed` | — |

`main.sscha.py run` chains all of this per temperature: `local` runs pw.x itself to the end; `slurm` submits each
population as a job array and a step job (afterany) that minimises and submits the next population; `manual` writes
the inputs and stops (run them with anything, call `run` again). `run` is idempotent: it continues each temperature
where its `state.json` says, and resubmits the missing configurations of a population.

Outputs per temperature (`output/sscha_T<T>/`): `pop<p>/` (inputs, outputs, `dyn*`, `summary.json`), `hessian/`
(`dyn*`, `summary.json`: `stable`, softest mode per q), `force_constants_auxiliary/` and `_hessian/` (QE matrices on
the ph.x grid), `<prefix>_sscha_<kind>.fc`.

---

## 0. Operating rules (always)

1. **Cost first.** Before any launch state configurations × cost of one supercell scf × expected populations
   (4–6 populations + a Hessian population of 2–4× the size are typical) and ask.
2. **One campaign, its own folders.** Never touch the run directories or jobs of another campaign; never cancel
   jobs you did not submit.
3. **State from files.** `SSCHARun(d).status()` / `main.sscha.py status`; log greps give false positives.
4. **The Hessian decides stability**, on a fine q-mesh (`grid_minimum`, `diagnostics(fine_mesh=...)`); the
   auxiliary frequencies are positive by construction.
5. **Settings with every number**: supercell, k-grid, electrons, populations, Hessian size, harmonic support.
6. **Provenance on every experimental number** and what it measures (transition vs eutectoid vs decomposition;
   heating vs cooling; composition and its range).
7. **Scratch and permanent storage**: on clusters run in scratch; after every finished temperature copy
   `state.json`, `pop*/summary.json`, `pop*/dyn*`, `hessian/`, `force_constants_*/`, `*.fc` to permanent storage.
8. **Correct yourself explicitly** (dated note in `REPORT.md`).
9. **PAOFLOW conventions**: Python arguments, no config files, tests and NumPy docstrings for library changes.

---

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: system, question, harmonic instabilities, machine, budget | harmonic `ph.x`, `grid_minimum`, `dispersion` | inventory, harmonic dispersion |
| 1 | Plan: supercell, k-grid, electrons, populations, temperatures, launcher | (reasoning), cost estimate | `PLAN.md` |
| 2 | Drivers and launch | `build_sscha_script`, `main.sscha.py harmonic/run` | run folders, jobs |
| 3 | Monitoring | `SSCHARun.status/describe`, `main.sscha.py status` | progress notes |
| 4 | Diagnosis of convergence | `SSCHARun.diagnostics`, `plot_convergence` | decisions (larger populations, restart) |
| 5 | Hessian and stability at each T | `hessian`, `lowest_hessian`, `grid_minimum`, `dispersion` | verdict per T |
| 6 | Temperature scan: T_s and experiment | `scan_table`, `stability_temperature`, `next_temperature` | `tables/scan.csv`, `tables/stability.md` |
| 7 | Downstream: thermal cells, electron-phonon | `special_displacement(force_constants_file='sscha')`, `ElphonSystem(phonon_source='sscha')` | handed-over paths |
| 8 | Figures and report | `plotting.*`, `templates/report_template.md` | `figures/`, `REPORT.md` |

## Stage 0: intake

1. **System and question.** Crystal, phase, composition; metal or insulator; the question (stable at T? T_s? force
   constants for which downstream step?).
2. **Harmonic picture.** Read the harmonic `ph.x` run (or ask for it): `grid_minimum(harmonic_dyn, mesh)` and the
   dispersion along a standard path; where are the instabilities (q, mode character), how deep (cm⁻¹)? Note the
   electrons of that run (cold smearing or Fermi–Dirac at T).
3. **Machine and budget.** Cores per configuration, wall time per scf, queue, scratch and permanent paths, the
   environment line that loads QE and PAOFLOW in a job.
4. **Experiment.** Temperatures (with what they mean), measured phonons at T, sources and uncertainties.
5. Report the inventory and the questions before planning.

## Stage 1: plan

- **Supercell.** Its q-grid must contain the instabilities (a 2×2×2 fcc cell has Γ, X and L only). Larger cells
  sample more q but cost N³ more per configuration and need more configurations. Start with the smallest cell that
  contains the unstable q; check the others on the fine mesh after the run (Stage 5).
- **k-grid** of the supercell: the parent's k-density (the default divides the scf grid by the supercell);
  metals need the same density as the harmonic run.
- **Electrons.** Parent occupations, or Fermi–Dirac at the ionic temperature (`ELECTRONIC_TEMPERATURE = 'ionic'`)
  when the instability is tied to the Fermi surface (metals, high T; `LESSONS_LEARNED.md` #4). The harmonic start
  can stay cold: it only seeds the Gaussian.
- **Populations.** 100 configurations for cells of ~16 atoms, growing (100, 100, 200, 200, 300, ...); the Hessian
  population 2–4× the last. Budget: `len(POPULATION_SIZES)` populations before the run is declared failed.
- **Temperatures.** For T_s: two temperatures that bracket it (one unstable, one stable), then refine with
  `next_temperature` until the bracket is narrower than the target (100 K). Warm starts (`START_FROM`) from the
  nearest converged temperature save populations.
- **Launcher.** `local` (workstation), `slurm` (cluster), `manual` (anything else).
- Write `PLAN.md` with the cost estimate; confirm with the user.

## Stage 2: drivers and launch

1. `campaign.py` writes one `main.sscha.py` per run folder with `build_sscha_script(cfg)` (or run `paoflow-gen`
   interactively). Check the constants of the written driver.
2. `python main.sscha.py harmonic` if `HARMONIC_DYN` does not exist (it runs or submits `ph.x`).
3. `python main.sscha.py run` (all temperatures) or `--temperature T`. On Slurm the first population and its step job
   are submitted; the chain continues alone. Record job ids in `PLAN.md`.

## Stage 3: monitoring

- `python main.sscha.py status` or, without PAOFLOW's save, `SSCHARun(d).describe()` for every run directory
  (`find_runs(root)`).
- Per population: configurations finished, free energy and error, Kong–Liu ratio, convergence, lowest auxiliary
  frequencies. A stuck population: `failed_configurations()` and the pw.x outputs.
- Wake-ups: poll at the cadence of one population (minutes to hours), not faster.

## Stage 4: diagnosis

`SSCHARun.diagnostics()` codes and what to do:

| code | meaning | action |
|---|---|---|
| `kong_liu_every_population` | every minimisation stopped on the Kong–Liu ratio | larger populations or smaller `MIN_STEP_DYN` |
| `free_energy_drift`, `frequencies_moving` | still minimising | continue (more populations) |
| `failed_configurations` | > 5 % of pw.x runs without forces | scf convergence (`MIXING_BETA`, `CONV_THR`), wall time, memory; `run` resubmits |
| `not_converged` | population budget exhausted | extend `POPULATION_SIZES` and `run` again (the state continues) |
| `unstable` | the Hessian is unstable at T | T is below T_s (or the supercell is too small); a data point of the scan |
| `unstable_between_grid_points` | stable on the supercell grid, unstable on the fine mesh | larger supercell, or a harmonic support that captures the instability |
| `cold_electrons` | high T with the parent's occupations | consider Fermi–Dirac at T for metals |

A run that converges in its first population (warm or lucky start) is not wrong, but check that its error bars are
sane and rely on the Hessian population.

## Stage 5: Hessian and stability

- `SSCHARun.hessian()`: `stable` and the softest mode per supercell q. Then `grid_minimum(force_constants('hessian'),
  mesh)` on a mesh 3–4× the supercell grid, and the dispersion along the standard path against the harmonic one.
- The verdict at T: stable / unstable, with the softest mode (q, cm⁻¹) and the settings. Small negative values at Γ
  acoustic or within the Hessian's stochastic noise (a few cm⁻¹) are not instabilities: say so and compare with the
  population error.

## Stage 6: temperature scan and experiment

- `rows = scan_table(find_runs(root), fine_mesh)`; `stability_temperature(T, softest)` → `t_stability` (bracket
  interpolation of ω|ω|), the local fit (`t_fit`, `slope`, `rms`) and `extrapolated`. Quote the bracket, not only the
  number. `next_temperature` proposes the next run until the bracket is below the target width.
- Compare with experiment as an inequality or a range: the SSCHA gives the temperature above which the *harmonic
  instability is lifted* at fixed volume; a measured eutectoid or transition temperature also involves competing
  phases, composition and thermal expansion. State the differences (fixed cell, electrons, supercell, functional).

## Stage 7: downstream

- Thermal cells at T: `special_displacement([T], force_constants_file='sscha')` after `sscha_force_constants
  ('auxiliary')` (the auxiliary matrices are the Gaussian of the ensemble at T).
- Electron-phonon with anharmonic phonons: `ElphonSystem(..., phonon_source='sscha', sscha_dynamical_matrices=
  '<run>/force_constants_hessian/<prefix>.dyn')` on the same q-grid as the coupling (the harmonic DFPT vertex is
  kept; frequencies and polarisations follow T).
- Hand over paths, not copies, and record which run (T, settings) each downstream result used.

## Stage 8: figures and report

Figures: convergence per run (`plot_convergence`), the scan with T_s and the experimental reference (`plot_scan`),
dispersions at the computed temperatures against the harmonic one (`plot_dispersions`, ≤ 5 temperatures per
panel). Report from `templates/report_template.md`: bottom line first, settings table, per-T verdicts, T_s with the
bracket and the comparison, limitations, reproduce block.

## Decision tree

1. Harmonic run stable everywhere? → SSCHA only for the temperature dependence (force constants), no T_s.
2. Unstable q not in the supercell grid? → larger supercell before anything else.
3. Run fails to converge? → Stage 4 table.
4. Hessian unstable at the highest affordable T? → report "not stabilised up to T" (a bound), check the electrons
   and the supercell; do not lower thresholds to force a verdict.
5. Hessian stable at all computed T? → go lower until a bracket appears, or report "stable down to T" (a bound).
