"""Watch a campaign until something needs the agent: new findings, all runs done, or a machine lost.

The command line is :mod:`~PAOFLOW.agentic.campaign.watch`.

Run in the background by the agent (its exit is the notification)::

    python -m PAOFLOW.agentic.campaign.watch projects/<study>/campaign.py --interval 900
    python -m PAOFLOW.agentic.campaign.watch projects/<study>/campaign.py --once      # one poll, print, exit

The study's ``campaign.py`` defines ``CAMPAIGN`` (:class:`~.campaign.Campaign`). Every poll rewrites
``STATUS.md``; findings already reported are listed in ``.watch_reported`` of the study folder and are not
reported again (delete the file to see them all again).
"""

from __future__ import annotations

import importlib.util
import os
import time

from .campaign import Campaign
from .diagnosis import Finding


def load_campaign(path: str) -> Campaign:
    """The ``CAMPAIGN`` object defined by a study's ``campaign.py``."""
    spec = importlib.util.spec_from_file_location('campaign_definition', os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, 'CAMPAIGN', None), Campaign):
        raise SystemExit('%s does not define CAMPAIGN = Campaign(...)' % path)
    return module.CAMPAIGN


def _reported(study_dir: str) -> set[str]:
    path = os.path.join(study_dir, '.watch_reported')
    if not os.path.exists(path):
        return set()
    return {line.strip() for line in open(path) if line.strip()}


def _remember(study_dir: str, findings: list[Finding]) -> None:
    with open(os.path.join(study_dir, '.watch_reported'), 'a') as handle:
        handle.writelines('%s\n' % f.key for f in findings)


def watch(
    campaign: Campaign,
    interval_seconds: float = 900.0,
    max_hours: float = 72.0,
    stop_at: tuple[str, ...] = ('error', 'warning'),
    unreachable_polls: int = 3,
    once: bool = False,
) -> tuple[str, list[Finding]]:
    """Poll the campaign until an event and return it.

    Parameters
    ----------
    campaign : Campaign
        The campaign.
    interval_seconds : float
        Time between polls (match the slowest stage: minutes for short jobs, 10-20 min for hours-long ones).
    max_hours : float
        Give up after this long (the agent restarts the watch).
    stop_at : tuple of str
        Severities that end the watch when a new finding has one.
    unreachable_polls : int
        Consecutive polls with an unreachable machine before that is an event.
    once : bool
        Poll once and return.

    Returns
    -------
    event : str
        ``'findings'``, ``'done'``, ``'unreachable'``, ``'timeout'`` or ``'poll'`` (with ``once``).
    findings : list of Finding
        The new findings (all findings with ``once``).
    """
    deadline = time.time() + max_hours * 3600.0
    unreachable_count = 0
    while True:
        states = campaign.status()
        findings = campaign.diagnose(states)
        campaign.write_status(states, findings)
        if once:
            return 'poll', findings
        if any(not s.reachable for s in states):
            unreachable_count += 1
            if unreachable_count >= unreachable_polls:
                return 'unreachable', [f for f in findings if f.code == 'machine_unreachable']
        else:
            unreachable_count = 0
        seen = _reported(campaign.study_dir)
        new = [
            f
            for f in findings
            if f.severity in stop_at and f.key not in seen and f.code != 'machine_unreachable'
        ]
        if new:
            _remember(campaign.study_dir, new)
            return 'findings', new
        if states and all(s.done for s in states):
            return 'done', [f for f in findings if f.code == 'run_complete']
        if time.time() + interval_seconds > deadline:
            return 'timeout', []
        time.sleep(interval_seconds)
