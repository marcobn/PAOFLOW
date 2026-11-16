# Prompts to start and steer the campaign workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent opened in `agentic/campaign/`.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new campaign (intake: goal, machines, budget, other campaigns) |
| 01 | `01_inventory.md` | check the machines: access, scheduler, billing, scratch, code versions, load |
| 02 | `02_plan.md` | runs × stages × machines, resources, cost, monitoring, storage → `PLAN.md` |
| 03 | `03_preflight.md` | check every input before launch and fix what the checks find |
| 04 | `04_pilot.md` | one representative job; measured timing; re-estimate; go / no-go |
| 05 | `05_launch_and_watch.md` | launch the approved runs, write `campaign.py`, start the watcher |
| 06 | `06_status.md` | where every run stands (one poll) |
| 07 | `07_interject.md` | a watcher event: verify, act within authority, propose the rest |
| 08 | `08_change_request.md` | a change of settings, plan or budget, with options and cost |
| 09 | `09_collect_and_account.md` | results to permanent storage, spend against plan, hand-over |
| 10 | `10_report_and_review.md` | the report, the lessons, and a re-test of the claims |
| 11 | `11_resume_session.md` | continue in a new session |

**Control phrases**: "Ask before spending more than <N> <units>.", "You may cancel jobs that cannot finish without
asking.", "Do not resubmit anything without asking.", "Write only to <folder>.", "Poll every <N> minutes.", "Quote every
number with the file it comes from."
