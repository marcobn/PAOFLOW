# Campaign workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) that **orchestrates a computational campaign on several
machines**: many runs (DFT, phonons, electron–phonon, thermal cells, optics, ...) chained in stages, on clusters with
a scheduler and on workstations, over days. The agent plans the campaign, checks every input before launch, runs a
pilot, launches with the campaign's own scripts, **watches** it, and **interjects** when something goes wrong:
convergence, errors in the files, memory or performance failures, budget, storage. Every intervention has an
explicit **authority**: report, act and report, or ask. Read `LESSONS_LEARNED.md` once before starting.

The workflow does not replace the scripts that build and submit the runs (`paoflow-gen` drivers, the stage scripts of
a production workflow, a restage script); it decides *when* to use them, checks *what* they will do, and watches
*what happened*.

A study has these files, in `projects/<study>/`:

| File | Written by | Content |
|---|---|---|
| `PLAN.md` | the agent, from `templates/plan_template.md`, approved by the user | runs × stages × machines, resources, cost, approved budget, monitoring cadence, storage, authority overrides |
| `campaign.py` | the agent, from `templates/campaign_template.py` | machines and runs in Python: `CAMPAIGN = Campaign(...)` |
| `STATUS.md` | the watcher, every poll | one row per run, spend, findings |
| `JOURNAL.md` | the agent, `CAMPAIGN.journal(...)` | every action, with its evidence and the finding it answers (append-only) |
| `REPORT.md` | the agent, from `templates/report_template.md` | results, cost against plan, incidents, lessons |

```python
from PAOFLOW.agentic.campaign import Campaign, CampaignRun, Machine, local_machine
from PAOFLOW.agentic.campaign import preflight_pw_input, reduce_cell_input, choose_kgrid, balanced_pools
from PAOFLOW.agentic.campaign import parse_pw_output, error_signatures, ACTIONS
```

## The engine (library)

| piece | what it does |
|---|---|
| `Machine(name, host, scheduler, environment, billing_granularity, charge_per_core_hour, charge_unit, scratch_purge_days, ...)` | runs shell scripts locally or through ssh (one round trip per call); `accounting(since, prefixes)` (sacct), `queue()` (squeue), `read_files(dir, globs, grep)`; a failed ssh raises `MachineUnreachable`: **unknown, never an empty queue** |
| `CampaignRun(name, machine, directory, job_prefix, pw_outputs, logs, results, conv_thr, reference_seconds_per_iteration)` | one run: its jobs (Slurm names starting with `job_prefix`), the pw.x outputs to follow, the logs to scan, the files that mean "done" |
| `Campaign(name, runs, study_dir, since, approved_budget)` | `status()` (one poll), `diagnose()` (findings, worst first), `spent()` (billed core hours and budget units per machine), `describe()` / `write_status()` (`STATUS.md`), `journal(action, evidence, findings)` (`JOURNAL.md`) |
| `parse_pw_output` / `parse_pw_input` | state of a pw.x run: k-points, FFT box, symmetry, scf accuracy and time per iteration, convergence, density from file, errors; settings of the input |
| `error_signatures(text)` | failures in any log: Davidson/Cholesky, scf not converged, max_seconds, QE errors, MPI count overflow and aborts, Python tracebacks, out of memory, segmentation faults, time limit, node failure, q-grid/k-grid, k-points not accounted, missing files |
| `preflight_pw_input(text, run, displaced, metallic, interpolated, qgrid, cores, cores_per_pool, billing_granularity)` | checks before launch (FFT inflation, k-point inflation, two-point axes, coarse grids, q-grid/k-grid, plain mixing of large metals, pool layout, billing rounding) |
| `reduce_cell_input(text, nosym, damp_mixing, start_from_density, floor_bohr)` | rewrites a pw.x input in the reduced basis of its lattice (same atoms), with a k-grid of ≥ 3 points per axis and a shortest superlattice vector ≥ `floor_bohr`, `nosym`, damped mixing |
| `choose_kgrid`, `time_reversal_kpoints`, `shortest_superlattice_vector`, `fft_box`, `fft_inflation`, `balanced_pools`, `billed_cores` | the arithmetic behind the checks |
| `python -m PAOFLOW.agentic.campaign.watch <study>/campaign.py [--interval S] [--once]` | the background watcher: polls, rewrites `STATUS.md`, exits with the **new** findings, `done`, `unreachable` or `timeout` |
| `ACTIONS` | code → (severity, authority, action): the table of section 6, in code |

---

## 0. Operating rules (always)

1. **Budget first.** Before any launch: runs × stages × measured (or pilot) time × billed cores, against the approved
   budget. Nothing outside `PLAN.md` without approval. Billed cores, not requested: shared nodes bill in steps
   (24-core steps on Snellius fat nodes).
2. **Pilot before bulk.** One representative job of the most expensive kind first; re-estimate from its measured time
   per iteration and iteration count. The campaign that skipped this spent six times its plan.
3. **Preflight every input.** `preflight_pw_input` on each run before submission; fix what it finds (or record why
   not) before the pilot.
4. **One campaign, its own jobs.** Never cancel, resubmit, edit or delete jobs, files or monitors of another campaign
   or session. Shared workstations: read and update the machine's row in the locks file before running.
5. **State from the scheduler and the files**, through the parsers (`Campaign.status`), not from ad-hoc log greps; an
   unreachable machine is *unknown*.
6. **Interject within your authority** (section 6) and **journal every action** (`CAMPAIGN.journal`). Report `act`
   actions at once; for `ask` actions propose options with their cost and wait.
7. **Code versions.** Record the commit of the code on every machine in `PLAN.md`. Remote clones are updated only by
   `git pull` of pushed commits, never by copying files, and **not while a campaign that depends on them runs**. After
   a code update, recompute one finished case and compare before mixing old and new results.
8. **Storage.** Runs in scratch; results to permanent storage after every stage; watch the purge age.
9. **Never commit** (the user commits). Write only in the study folder, the campaign's run directories and the
   workflow scripts the user designated.
10. **Correct yourself explicitly**: a dated note in `REPORT.md` and `JOURNAL.md`.
11. **PAOFLOW conventions**: Python arguments, no config files; library changes with tests and NumPy docstrings.

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: goal, systems, machines, budget, other campaigns | `Machine.shell`, locks file | inventory |
| 1 | Plan: runs × stages × machines, resources, cost | `balanced_pools`, `billed_cores`, past timings | `PLAN.md` (approved) |
| 2 | Preflight: every input checked | `preflight_pw_input`, `reduce_cell_input`, `choose_kgrid`, dry runs | fixed inputs, preflight table |
| 3 | Pilot: one representative job | the campaign's scripts, `Campaign.status` | measured time, re-estimate, go / no-go |
| 4 | Launch and watch | the campaign's scripts, `campaign.py`, the watcher | job ids, running watcher |
| 5 | Monitor | watcher (background), `--once` on demand | `STATUS.md` |
| 6 | Interject | `diagnose`, `ACTIONS`, journal | actions within authority, proposals |
| 7 | Collect and account | permanent storage, `spent`, the domain workflow | results, cost against plan |
| 8 | Report and lessons | `templates/report_template.md` | `REPORT.md`, new lessons |

## Stage 0: intake

1. **Goal and systems**: what is computed, for which systems, to what end (a figure, a number against experiment, a
   paper); which domain workflow will analyse the results (`thermo_optics`, `transport`, `superconductivity`, `sscha`,
   ...).
2. **Machines.** For each: access (ssh alias, control socket), scheduler, partitions, cores per node, memory per core,
   billing granularity and unit, the budget and who shares it, scratch and permanent paths, purge age, the environment
   line of a job, the code versions installed (QE build, PAOFLOW commit), limits on shared workstations (ranks, `nice`).
   Run `Machine.shell('...')` to check what you are told.
3. **Other campaigns.** What else runs on these machines and accounts (other sessions, other users of a shared budget);
   the locks file; what you must not touch.
4. **Existing scripts and data**: the production workflow (stage scripts, submit script, results saver), drivers,
   previous runs to reuse (saves, force constants), and the timings of similar runs.
5. Report the inventory and your questions. Launch nothing.

## Stage 1: plan

- **Runs and stages.** One row per run: machine, stage chain (scf → phonons → ... → post), dependencies, inputs.
- **Resources per stage.** Cores (a multiple of the billing granularity; pools balanced over k-points:
  `balanced_pools`), memory, time limit with a margin over the expected time (≥ 1.5×, but not 3× "to be safe": a stuck
  job burns the whole limit), `max_seconds` of pw.x below the limit for long runs so that a restart is possible.
- **Cost.** Expected time × billed cores × charge per core hour, per stage and in total; the source of every timing
  (a pilot, a similar finished run). State the uncertainty. If no timing exists, the plan *is* the pilot.
- **Monitoring cadence** (≈ the time of the shortest stage, 10–20 min for hour-long jobs) and **storage** (what is
  copied to permanent storage after which stage).
- Write `PLAN.md`; ask for approval of the budget; record the approved number as `approved_budget` of the campaign.

## Stage 2: preflight

For every input of the plan:

1. `preflight_pw_input(text, run, displaced=..., metallic=..., interpolated=..., qgrid=..., cores=...,
   billing_granularity=...)`. Each finding is a mistake of an earlier campaign (`LESSONS_LEARNED.md`); fix it before the
   pilot (`reduce_cell_input` for FFT and k-point inflation; ≥ 3 points per axis for interpolated Hamiltonians; damped
   mixing for large metallic cells; a coupling k-grid that the q-grid divides).
2. Dry-run the scripts that write and submit the runs (most have a mode that prints the plan without moving anything)
   and read what they would do.
3. Record the preflight table (run, findings, fix) in `PLAN.md`.

## Stage 3: pilot

- Submit one representative job of the most expensive kind (the largest cell, the slowest stage).
- Watch its first iterations: time per iteration, convergence trend (no rise above the first estimate, steady
  decrease), memory. After a few iterations report the measured numbers and the re-estimated cost.
- Go / no-go with the user. If the pilot fails, it is a diagnosis case (section 6), not a reason to launch the rest.

## Stage 4: launch and watch

1. Launch with the campaign's own scripts, stage chains with `afterok` dependencies; record job ids in `JOURNAL.md`.
2. Write `campaign.py`: one `CampaignRun` per run with its `job_prefix` (name every job of a run with a common
   prefix), the pw.x outputs to follow, the logs to scan, the result files. Test it once:
   `python -m PAOFLOW.agentic.campaign.watch campaign.py --once`.
3. Start the watcher in the background (`--interval` per the plan). Keep the watcher in the study folder or call it
   through the library (scripts in `/tmp` vanish with a restart). One watcher per campaign; after editing a running
   watcher script, stop and restart it (a running shell keeps the old code).

## Stage 5: monitor

- The watcher exits when a **new** finding of severity `error` or `warning` appears (findings already reported are in
  `.watch_reported`), when every run is done, when a machine stays unreachable for three polls, or after `--max-hours`.
  Its exit is the notification: read `STATUS.md`, act (section 6), restart it.
- On request, `--once` gives the current status without waiting.
- A machine that does not answer is *unknown*; say so, retry, and tell the user if it lasts.

## Stage 6: interject

On a watcher event:

1. **Verify.** Re-read the evidence from the files (`Campaign.status()`, the output and the log tail), not from the
   watcher's line alone.
2. **Decide by authority** (`ACTIONS[code]`):

| authority | meaning | examples |
|---|---|---|
| `report` | tell the user; no action | `run_complete`, `slow_iterations`, `pool_layout`, `long_pending`, `budget_warning`, `machine_unreachable` |
| `act` | do it without asking, report at once, journal | `cannot_finish` (cancel: its own convergence and timing cannot reach the end within the limit), `output_stalled_after_error` (cancel a hung job after a rank failed), `dependency_never_satisfied` (cancel the orphan), `job_node_failure` (resubmit the identical job once), `max_seconds` (restart a run that stopped cleanly, within the plan), `scratch_purge_risk` (copy results now) |
| `ask` | propose options with cost, wait | every change of physics or numerical settings (mixing, grids, basis, smearing, bands), resubmissions after a timeout, a failure or an out-of-memory, anything outside the plan or the budget, code updates, anything on another campaign |

3. **Act or propose.** For `act`: do it, journal it (`CAMPAIGN.journal(action, evidence, findings)`), report it.
   For `ask`: write the options (fix, cost, what it changes in the results) and wait; never relaunch on your own a run
   whose settings you would change.
4. **Fix the cause, not the symptom.** A timeout is a symptom: find the cause (k-points, FFT box, sloshing, pools) in
   the output before any resubmission. A resubmission identical to a failed job is allowed only for node failures.
5. Restart the watcher.

### Diagnosis table (codes of `ACTIONS`)

| code | evidence | likely cause | fix (authority) |
|---|---|---|---|
| `cannot_finish` / `may_not_finish` | scf trend × time per iteration beyond the time left (> 2× / > 1×) | slow iterations (below) or sloshing | cancel (act) / prepare the fix (ask) |
| `charge_sloshing` | accuracy rises > 10× above the first estimate | plain mixing, large metallic cell | `local-TF`, β 0.2–0.3, 12 vectors; start from a converged density of the same cell (ask) |
| `scf_stagnating`, `scf_not_converged` | no progress over 6 iterations / stopped | mixing, smearing, bands | as above (ask) |
| `kpoint_inflation` | no symmetry, k-points > time-reversal count of the grid | grid not invariant under the lattice group | `nosym = .true.` or an invariant grid (ask) |
| `fft_inflation` | FFT box ≥ 2× the reduced basis | skewed supercell basis | `reduce_cell_input` (ask) |
| `output_stalled`, `output_stalled_after_error` | output unchanged for 3 iterations (≥ 30 min) while running | a rank died, the others wait in a collective | cancel if a failure is in a log (act), else investigate (ask) |
| `job_timeout` | Slurm TIMEOUT | any of the above | fix the cause, re-estimate (ask) |
| `job_out_of_memory`, `out_of_memory` | OOM state or messages | ranks × memory per rank | fewer ranks per node, more memory, smaller buffers (ask) |
| `job_node_failure` | NODE_FAIL / BOOT_FAIL | hardware | resubmit once (act) |
| `dependency_never_satisfied` | pending with that reason | parent failed or cancelled | cancel the orphan (act); resubmit the chain later (ask) |
| `davidson_cholesky` | "problems computing cholesky" | ill-conditioned overlap, many bands | `diagonalization = 'cg'` (ask) |
| `mpi_count_overflow` | MPI_ERR_ARG / COUNT / TRUNCATE | an array beyond 2³¹ elements | PAOFLOW with chunked collectives or a smaller grid (ask) |
| `python_exception`, `mpi_abort`, `segmentation_fault`, `qe_error` | the message | input, environment, code | read it, fix, resubmit the stage (ask) |
| `qgrid_does_not_divide_kgrid` | the message | grids of the system description | a multiple of the q-grid (ask) |
| `missing_file`, `kpoints_not_accounted` | the message | a stage input missing; a k-list not covering the grid | provide the input; check nosym/noinv (ask) |
| `budget_warning`, `budget_exceeded` | spend ≥ 80 % / ≥ 100 % of the approved budget | — | report / stop launching and ask |
| `scratch_purge_risk` | files near the purge age | — | copy to permanent storage (act) |

## Stage 7: collect and account

- Copy results to permanent storage (the campaign's saver, or `rsync` without `--delete`), check sizes or checksums.
- Spend: `CAMPAIGN.spent()` per machine against the plan and the approved budget; account for lost work (failed,
  cancelled, timed-out jobs) separately. A shared-budget report from the provider may lag behind `sacct`.
- Hand the results to the domain workflow that analyses them (paths, not copies; which runs and settings).

## Stage 8: report and lessons

`REPORT.md` from `templates/report_template.md`: bottom line, results (with their settings), cost against plan,
incidents (finding → action → outcome, from `JOURNAL.md`), open points. New failure modes become rows of
`LESSONS_LEARNED.md` (propose the edit) and, when they can be detected, a code of `ACTIONS` with a test (propose the
library change).

## Decision tree

1. No timing for the most expensive run? → the pilot is the next step, not the campaign.
2. Preflight finding on an input? → fix before the pilot, or record why it is accepted.
3. Pilot slower than planned by > 1.5×? → find why in its output (k-points, FFT box, pools, sloshing) before launching
   the rest; re-estimate and ask.
4. A job that cannot finish? → cancel (act), find the cause, propose the fix with its cost (ask).
5. Budget at 80 %? → report with the estimate to finish; at 100 % → stop launching, ask.
6. A code update needed mid-campaign? → wait for the running jobs, or ask; after the update recompute one finished case
   and compare.
