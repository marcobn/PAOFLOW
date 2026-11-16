# Agent instructions: campaign workflow

You are orchestrating a computational campaign on several machines with PAOFLOW and Quantum ESPRESSO: plan, check,
pilot, launch, watch, and interject when something goes wrong (convergence, errors in the files, memory or
performance failures, budget, storage), within an explicit authority.

1. **Read before acting:** `AGENT_WORKFLOW.md` and `LESSONS_LEARNED.md`. Follow them.
2. **Use the library** `PAOFLOW.agentic.campaign`:
   - `Machine(name, host, scheduler, environment, billing_granularity, charge_per_core_hour, charge_unit,
     scratch_purge_days, user_id)`, `local_machine()`; `Machine.shell`, `accounting`, `queue`, `read_files`;
     `MachineUnreachable` (unknown, never an empty queue).
   - `CampaignRun(name, machine, directory, job_prefix, pw_outputs, logs, results, conv_thr,
     reference_seconds_per_iteration)`, `Campaign(name, runs, study_dir, since, approved_budget)`: `status()`,
     `diagnose()`, `spent()`, `describe()`, `write_status()`, `journal(action, evidence, findings)`.
   - `preflight_pw_input`, `reduce_cell_input`, `choose_kgrid`, `time_reversal_kpoints`, `fft_inflation`,
     `balanced_pools`, `billed_cores`; `parse_pw_output`, `parse_pw_input`, `error_signatures`; `ACTIONS`.
   - The watcher: `python -m PAOFLOW.agentic.campaign.watch projects/<study>/campaign.py [--interval S] [--once]`, run
     in the background; its exit is your notification.
   - The campaign's own scripts (drivers, stage scripts, submit and restage scripts) build and submit the runs.
   Every study has `PLAN.md`, `campaign.py` (from `templates/`), `STATUS.md`, `JOURNAL.md`, `REPORT.md`. Inputs are
   Python arguments; no config files.
3. **Where to write:** the study folder, the campaign's run directories and the scripts the user designated; never
   into another campaign's jobs, run directories, monitors or locks rows; do not modify `src/PAOFLOW/` or this folder
   unless asked (propose library changes with a test).
4. **Non-negotiable:**
   - state the cost (billed cores × expected time × charge) and get the budget approved before any launch; a pilot
     before the bulk; nothing outside `PLAN.md` without approval;
   - `preflight_pw_input` on every input before submission;
   - act only within the authority of `ACTIONS` (report / act / ask); journal every action with its evidence; report
     `act` actions at once; for `ask`, propose options with their cost and wait;
   - read states through the parsers and the scheduler; an unreachable machine is unknown;
   - never cancel, resubmit or delete jobs and files of another campaign or session; respect the machine locks and
     core limits;
   - never commit; remote clones only by `git pull`, never during a campaign that uses them; record code versions;
   - results to permanent storage after every stage;
   - quote every number with its file; correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`; do what they ask and report the **Check** line.
