# Campaign workflow: orchestrating and watching computational campaigns on several machines

This workflow runs **computational campaigns** with PAOFLOW and Quantum ESPRESSO across machines (clusters with
Slurm, workstations without a scheduler): many runs, chained in stages, over days. An AI agent plans the campaign,
checks every input before launch, runs a pilot, launches the runs with the campaign's own scripts, **watches** them,
and **interjects** when something goes wrong: an scf that will not converge in time, charge sloshing, inflated
k-points or FFT boxes, errors in the output files, memory or performance failures, a hung MPI job, a budget running
out, scratch about to be purged. Typical uses:

- a production chain of many systems or configurations on a cluster (phonons, electron–phonon, thermal cells, optics);
- a series of runs on several workstations and a cluster at once;
- any long campaign that should not need a person to read logs every hour.

The agent works through `PAOFLOW.agentic.campaign`, which reads schedulers and files and never submits or cancels
anything itself. Every intervention has an explicit authority (`AGENT_WORKFLOW.md`, section 6):

- **report**: tell you, no action;
- **act**: cancel a job that provably cannot finish, a hung job after a rank failed, or an orphaned dependent;
  resubmit once after a node failure; restart a run that stopped cleanly at `max_seconds`; copy results before a
  scratch purge. These are done at once, journaled and reported;
- **ask**: everything that changes physics or numerical settings, spends budget outside the plan, or touches another
  campaign. The agent proposes options with their cost and waits.

Nothing here depends on a particular machine: machines are Python objects of the study's `campaign.py` (ssh alias,
scheduler, billing, scratch purge), and the runs are described by where their jobs and files are.

## Before you start

- **PAOFLOW installed** where the agent runs (the watcher imports the library); ssh access to every machine without a
  password prompt (keys, a ControlMaster socket).
- **The scripts that build and submit the runs**: `paoflow-gen` drivers, a production workflow (stage scripts, a
  submit script, a results saver), restage scripts. The workflow uses them; it does not replace them.
- **A budget and its owner** (core hours or the provider's units, who else spends it), and the machine rules: core
  limits on shared workstations, a locks file if several sessions share machines.
- **Timings of similar runs**, if any (the pilot replaces them otherwise).

## Start a campaign

```bash
cd PAOFLOW/agentic/campaign
mkdir -p projects/<study>
claude
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. Continue with `prompts/README.md`; resume with
`prompts/11_resume_session.md`.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: library, operating rules, stages 0–8, authority and diagnosis tables, decision tree |
| `LESSONS_LEARNED.md` | pitfalls (symptom → cause → fix) of the TaC, δ-TaN, HfC and Ta–Hf–C campaigns |
| `prompts/` | kickoff; inventory; plan; preflight; pilot; launch and watch; status; interject; change request; collect and account; report; review; resume |
| `templates/` | `campaign_template.py` (machines and runs), `stage_template.sbatch` (a Slurm stage with the guards), `plan_template.md`, `report_template.md` |
| `projects/` | your studies (ignored by git) |

## Worked example

The Ta–Hf–C emissivity campaigns on Snellius (November 2026): five POCC configurations of Ta₀.₈Hf₀.₂C at four
temperatures (special-displacement cells of 80 atoms), whose first round ran into the 36 h limit (inflated k-points,
skewed FFT boxes, charge sloshing); the reduced-basis rerun (28 cells, 6.9 k SBU); the Ta₀.₅Hf₀.₅C and Ta₀.₂Hf₀.₈C
series (14 runs). Production folders, reports and the original scripts: `Workspace/PAOFLOW/production/XTa_project/`
(outside the repository: `snellius_workflow/`, `HfTaC/zg_reduce/`, `TaHfC_series/`).
