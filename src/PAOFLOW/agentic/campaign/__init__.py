"""Orchestrating and watching computational campaigns on several machines.

Serves the repository workflow ``agentic/campaign/`` (playbook, lessons learned, prompts, templates): an agent plans
a campaign (runs × stages × machines, cost, preflight checks), launches it with the campaign's own scripts, watches
it, and interjects when something goes wrong (convergence, errors in the files, memory or performance failures),
within an explicit authority: report, act and report, or ask. The package reads schedulers and files and never
submits or cancels anything itself.

* :mod:`~PAOFLOW.agentic.campaign.machines` -- :class:`~PAOFLOW.agentic.campaign.machines.Machine`: this computer
  or an ssh host, with or without Slurm; accounting, queue and files in one round trip; an unreachable machine is
  *unknown*, never an empty queue;
* :mod:`~PAOFLOW.agentic.campaign.scheduler` -- Slurm records (``sacct``, ``squeue``), billing granularity, pool
  layouts, core hours;
* :mod:`~PAOFLOW.agentic.campaign.pw_output` -- the state of a pw.x run from its output (convergence trajectory,
  time per iteration, k-points, FFT box, symmetry, errors) and the error signatures of any log;
* :mod:`~PAOFLOW.agentic.campaign.cells` -- FFT boxes, reduced bases, k-grids by their shortest superlattice
  vector, and the rewrite of a pw.x input in the reduced basis of its lattice;
* :mod:`~PAOFLOW.agentic.campaign.preflight` -- checks of an input before launch;
* :mod:`~PAOFLOW.agentic.campaign.diagnosis` -- findings with evidence, action and authority (:data:`ACTIONS`);
* :mod:`~PAOFLOW.agentic.campaign.campaign` -- :class:`Campaign` and :class:`CampaignRun`: status, diagnosis,
  spend, ``STATUS.md`` and the ``JOURNAL.md`` of actions;
* :mod:`~PAOFLOW.agentic.campaign.watcher` -- the background watcher; its command line is
  ``python -m PAOFLOW.agentic.campaign.watch``.
"""

from .campaign import Campaign, CampaignRun, RunState
from .cells import (
    choose_kgrid,
    fft_box,
    fft_inflation,
    reduce_cell_input,
    reduced_basis,
    shortest_superlattice_vector,
    time_reversal_kpoints,
)
from .diagnosis import ACTIONS, Finding, sort_findings
from .machines import Machine, MachineUnreachable, local_machine
from .preflight import preflight_pw_input
from .pw_output import PwInput, PwOutput, error_signatures, parse_pw_input, parse_pw_output
from .scheduler import (
    JobRecord,
    balanced_pools,
    billed_cores,
    core_hours,
    parse_sacct,
    parse_squeue,
)
from .watcher import load_campaign, watch

__all__ = [
    'ACTIONS',
    'Campaign',
    'CampaignRun',
    'Finding',
    'JobRecord',
    'Machine',
    'MachineUnreachable',
    'PwInput',
    'PwOutput',
    'RunState',
    'balanced_pools',
    'billed_cores',
    'choose_kgrid',
    'core_hours',
    'error_signatures',
    'fft_box',
    'fft_inflation',
    'load_campaign',
    'local_machine',
    'parse_pw_input',
    'parse_pw_output',
    'parse_sacct',
    'parse_squeue',
    'preflight_pw_input',
    'reduce_cell_input',
    'reduced_basis',
    'shortest_superlattice_vector',
    'sort_findings',
    'time_reversal_kpoints',
    'watch',
]
