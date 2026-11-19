# Studies

One folder per campaign, e.g. `projects/TaHfC_series/`. Each holds `PLAN.md` (approved plan and budget),
`campaign.py` (machines and runs: `CAMPAIGN = Campaign(...)`), `STATUS.md` (written by the watcher), `JOURNAL.md`
(every action of the agent), `REPORT.md`, and `.watch_reported` (findings the watcher has already reported). The run
directories live on the machines (scratch on clusters) and are referenced by path; results go to permanent storage
after every stage. Everything here except this README is ignored by git.
