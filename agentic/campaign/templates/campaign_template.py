#!/usr/bin/env python3
"""Machines and runs of a campaign (template: copy to projects/<study>/campaign.py and fill in).

    python campaign.py                                                   # one poll: STATUS.md and the findings
    python -m PAOFLOW.agentic.campaign.watch campaign.py --interval 900  # the background watcher
    python -m PAOFLOW.agentic.campaign.watch campaign.py --once          # one poll from the command line

The watcher imports this file and uses CAMPAIGN. Playbook: PAOFLOW/agentic/campaign/AGENT_WORKFLOW.md (stage 4).
Every value is a Python argument; nothing is read from config files.
"""

import os

from PAOFLOW.agentic.campaign import Campaign, CampaignRun, Machine, local_machine

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ machines
# A Slurm cluster reached through an ssh alias (ControlMaster recommended). Billing: the provider's unit per core
# hour and the step in which shared-node jobs are billed (Snellius fat nodes: 1.5 SBU, 24 cores).
CLUSTER = Machine(
    name='<cluster>',
    host='<ssh alias>',
    scheduler='slurm',
    environment='source ~/<environment script> > /dev/null 2>&1',  # loads QE and PAOFLOW in every command
    scratch='/scratch/<user>/<campaign>',
    permanent='~/<campaign>_results',
    billing_granularity=24,
    charge_per_core_hour=1.5,
    charge_unit='SBU',
    scratch_purge_days=14,
    user_id='<numeric uid>',  # cancellations by other ids are reported as external
)

# A workstation without a scheduler (runs started directly, e.g. with nohup); shared machines: a core limit.
WORKSTATION = local_machine('<workstation>', max_cores=16)

# ------------------------------------------------------------------ runs
# One CampaignRun per run directory. Name every job of a run with a common prefix (the job_prefix), list the pw.x
# outputs to follow (their inputs are the files with the same stem and .in), the logs to scan for errors, and the
# files whose presence means the run is done.
RUN_ROOT = '/scratch/<user>/<campaign>'
RUN_NAMES = ['<run 1>', '<run 2>']

RUNS = [
    CampaignRun(
        name=name,
        machine=CLUSTER,
        directory=os.path.join(RUN_ROOT, name),
        job_prefix='<campaign tag>_%s_' % name,
        pw_outputs=('<stage dir>/*.out',),
        logs=('jobs/logs/*.out', '<stage dir>/*.log'),
        results=('<output dir>/<result file>',),
        conv_thr=1e-8,
        reference_seconds_per_iteration=None,  # from the pilot, once measured
    )
    for name in RUN_NAMES
]

CAMPAIGN = Campaign(
    name='<campaign name>',
    runs=RUNS,
    study_dir=HERE,
    since='<YYYY-MM-DD>',  # start of the accounting window (machine time zone)
    approved_budget=None,  # the budget approved in PLAN.md, in the charge unit
)

if __name__ == '__main__':
    states = CAMPAIGN.status()
    findings = CAMPAIGN.diagnose(states)
    print(open(CAMPAIGN.write_status(states, findings)).read())
