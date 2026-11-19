# Agentic campaigns: orchestrating and watching runs on several machines

`PAOFLOW.agentic.campaign` serves the agentic workflow `agentic/campaign/`: an AI agent plans a computational
campaign (runs × stages × machines, cost), checks every input before launch, runs a pilot, launches the runs with the
campaign's own scripts (`paoflow-gen` drivers, production stage scripts), watches them, and interjects when something
goes wrong: convergence, errors in the files, memory or performance failures, budget, storage. The playbook, lessons
learned, prompts and templates live in `agentic/campaign/`; this package holds the parts that can be tested.

The package **reads** schedulers and files; it never submits or cancels a job. Actions are taken by the agent, within
the authority attached to each finding.

## Modules

| module | role |
|---|---|
| `machines` | `Machine(name, host, scheduler, environment, scratch, permanent, max_cores, billing_granularity, charge_per_core_hour, charge_unit, scratch_purge_days, user_id)`: `shell(script)` locally (`bash -s`) or through ssh, `accounting(since, prefixes)` (sacct), `queue()` (squeue), `read_files(directory, globs, grep)` (age, size and matching lines of many files in one round trip); `MachineUnreachable` for a failed ssh; `local_machine()` |
| `scheduler` | `JobRecord`, `parse_sacct` (steps folded into their job: largest `MaxRSS`), `parse_squeue`, `slurm_seconds`, `billed_cores(cores, granularity)`, `balanced_pools(kpoints, cores, cores_per_pool, granularity)`, `core_hours` |
| `pw_output` | `parse_pw_output` → `PwOutput` (k-points, bands, atoms, FFT box, symmetry operations, density from file, cell, `ecutrho`, scf accuracies and times, convergence, `JOB DONE`, max_seconds, errors; `seconds_per_iteration`, `largest_rise`, `iterations_to_converge(conv_thr)`, `stagnating`); `parse_pw_input` → `PwInput` (k-grid, `nosym`, atoms, cutoff, mixing, starting potential); `error_signatures(text)` over `ERROR_SIGNATURES`; `PW_OUTPUT_PATTERN` and `PW_INPUT_PATTERN` for remote `grep -E` |
| `cells` | `fft_box`, `lll_reduce`, `reduced_basis`, `fft_inflation`, `time_reversal_kpoints`, `shortest_superlattice_vector`, `grid_is_invariant`, `choose_kgrid(cell, min_points, floor_bohr)`, `reduce_cell_input(text, nosym, damp_mixing, start_from_density, floor_bohr)` |
| `preflight` | `preflight_pw_input(text, run, displaced, metallic, interpolated, qgrid, cores, cores_per_pool, billing_granularity, floor_bohr)` |
| `diagnosis` | `ACTIONS` (code → severity, authority, action), `Finding`, `findings_for_job`, `findings_for_pw`, `findings_for_log`, `findings_for_budget`, `findings_for_storage`, `sort_findings` |
| `campaign` | `CampaignRun`, `RunState`, `Campaign(name, runs, study_dir, since, approved_budget)`: `status()`, `diagnose()`, `spent()`, `describe()`, `write_status()` (`STATUS.md`), `journal()` (`JOURNAL.md`) |
| `watcher` | `watch(campaign, interval_seconds, max_hours, stop_at, unreachable_polls, once)`, `load_campaign(path)`: polls until new findings, completion, a lost machine or a time-out; `.watch_reported` keeps reported findings from being raised twice |
| `watch` | the command line, `python -m PAOFLOW.agentic.campaign.watch <campaign.py> [--interval S] [--once]` (not imported by the package, so `python -m` runs it cleanly) |

## Design choices

- **Authority with every finding.** `ACTIONS` gives each code a severity, an action and an authority: `report`,
  `act` (the agent may act without asking: cancel a job that provably cannot finish, a hung job after a rank failed,
  an orphaned dependent; resubmit once after a node failure; restart after max_seconds; copy results before a purge)
  or `ask` (any change of settings, plan or budget). The playbook's tables are the same table.
- **`cannot_finish` is conservative.** It needs at least six iterations and a projected time over twice the time left
  (log-linear fit of the last eight scf accuracies); between one and two times it is `may_not_finish` (ask).
- **Unknown is not empty.** A failed ssh raises `MachineUnreachable`; the run is reported `unknown`, and the watcher
  ends only after several unreachable polls.
- **One round trip per call.** `read_files` sends one shell script that globs, stats and greps, so a remote poll of a
  run costs two to four ssh calls (with a ControlMaster socket, a fraction of a second each).
- **Python, not configuration files.** A campaign is a `campaign.py` that builds `Machine` and `CampaignRun` objects.
- **Preflight from experience.** Each check reproduces a failure of the TaC–HfC campaigns: k-points inflated by a
  grid that the lattice group does not preserve, FFT boxes inflated by skewed supercell bases, charge sloshing of
  large metallic cells, two-point axes in interpolated Hamiltonians, q-grids that do not divide the coupling grid,
  pool layouts and core requests that waste billed cores. `reduce_cell_input` reproduces, byte for byte, the inputs of
  the reduced-basis rerun of the Ta₀.₈Hf₀.₂C cells.

## Tests

`tests/unit/agentic/campaign/test_campaign_agentic.py` uses filtered lines of real Ta₀.₈Hf₀.₂C pw.x outputs (a cell
that timed out: 81 k-points, 192³ FFT box, sloshing; a sloshing rhombohedral cell; the converged reduced-basis
reference) and real `sacct` records with steps. It checks the parsers, the four causes found for the timed-out cell
(`charge_sloshing`, `cannot_finish`, `kpoint_inflation`, `fft_inflation`), the cell rewrite (same atoms, compact
FFT box, k-grid rules), the preflight checks, the budget findings, a campaign on a local machine (status, diagnosis,
report, journal) and the watcher (events and its command line).
