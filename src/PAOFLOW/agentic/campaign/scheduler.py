"""Slurm accounting and queue records: parsing, billing and core hours.

The functions read the text that ``sacct`` and ``squeue`` print with the formats of :data:`SACCT_FORMAT` and
:data:`SQUEUE_FORMAT`, so they can be tested without a cluster and used on any machine that runs Slurm.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

SACCT_FORMAT = 'JobID,JobName%80,State%40,Elapsed,Timelimit,AllocCPUS,MaxRSS,ExitCode,Reason%40'
"""``sacct --format`` of :func:`parse_sacct` (with ``--parsable2 --noheader``)."""

SQUEUE_FORMAT = '%i|%j|%T|%M|%l|%C|%R'
"""``squeue -o`` format of :func:`parse_squeue` (with ``--noheader``)."""

FAILURE_STATES = (
    'FAILED',
    'TIMEOUT',
    'OUT_OF_MEMORY',
    'NODE_FAIL',
    'BOOT_FAIL',
    'DEADLINE',
    'PREEMPTED',
)
"""Terminal states of a job that did not finish its work."""

TERMINAL_STATES = FAILURE_STATES + ('COMPLETED', 'CANCELLED')
"""States after which a job holds no resources."""


@dataclass
class JobRecord:
    """One Slurm job (or array task) with its accounting.

    Attributes
    ----------
    job_id : str
        Job id, ``<array>_<task>`` for array tasks.
    name : str
        Job name.
    state : str
        First word of the Slurm state (``RUNNING``, ``COMPLETED``, ``CANCELLED``, ...).
    state_detail : str
        The full state text (``CANCELLED by 79346``).
    elapsed_seconds : float
        Elapsed wall time.
    time_limit_seconds : float or None
        Wall-time limit (None if unlimited or unknown).
    cores : int
        Allocated (billed) cores.
    max_rss_bytes : float or None
        Largest resident memory of any step (``sacct`` steps), None when not reported.
    exit_code : str
        ``<code>:<signal>``.
    reason : str
        Pending reason (``Priority``, ``Dependency``, ``DependencyNeverSatisfied``, ...) or ``''``.
    """

    job_id: str
    name: str
    state: str
    state_detail: str = ''
    elapsed_seconds: float = 0.0
    time_limit_seconds: float | None = None
    cores: int = 0
    max_rss_bytes: float | None = None
    exit_code: str = ''
    reason: str = ''
    steps: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        """True once the job holds no resources."""
        return self.state in TERMINAL_STATES

    @property
    def failed(self) -> bool:
        """True for the terminal states in which the work did not finish."""
        return self.state in FAILURE_STATES

    @property
    def remaining_seconds(self) -> float | None:
        """Wall time left before the limit (None without a limit)."""
        if self.time_limit_seconds is None:
            return None
        return max(self.time_limit_seconds - self.elapsed_seconds, 0.0)

    @property
    def core_hours(self) -> float:
        """Elapsed time times the allocated cores, in core hours."""
        return self.elapsed_seconds * self.cores / 3600.0


def slurm_seconds(text: str) -> float | None:
    """Seconds of a Slurm duration (``[D-]HH:MM:SS``, ``MM:SS``, ``MM:SS.mmm``); None for ``UNLIMITED`` or empty.

    Parameters
    ----------
    text : str
        Duration as printed by ``sacct`` or ``squeue``.

    Returns
    -------
    float or None
        The duration in seconds.
    """
    text = text.strip()
    if not text or text.upper() in ('UNLIMITED', 'INVALID', 'NOT_SET', 'PARTITION_LIMIT'):
        return None
    days = 0
    if '-' in text:
        day_text, text = text.split('-', 1)
        days = int(day_text)
    parts = [float(part) for part in text.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts[-3:]
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def memory_bytes(text: str) -> float | None:
    """Bytes of a Slurm memory field (``123456K``, ``1.5G``, ``800M``); None when empty."""
    match = re.match(r'^\s*([\d.]+)\s*([KMGTP]?)', text or '')
    if not match:
        return None
    scale = {'': 1.0, 'K': 1024.0, 'M': 1024.0**2, 'G': 1024.0**3, 'T': 1024.0**4, 'P': 1024.0**5}
    return float(match.group(1)) * scale[match.group(2)]


def parse_sacct(text: str) -> list[JobRecord]:
    """Jobs of ``sacct --parsable2 --noheader --format=`` :data:`SACCT_FORMAT` output, steps folded into their job.

    Parameters
    ----------
    text : str
        The ``sacct`` output, one line per job or step (``<id>.batch``, ``<id>.0``, ...).

    Returns
    -------
    list of JobRecord
        One record per job or array task, in the order of the output; ``max_rss_bytes`` is the largest of its steps.
    """
    jobs: dict[str, JobRecord] = {}
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) < 8 or not fields[0].strip():
            continue
        job_id, name, state_text, elapsed, limit, cores, max_rss, exit_code = (
            f.strip() for f in fields[:8]
        )
        reason = fields[8].strip() if len(fields) > 8 else ''
        if '.' in job_id:  # a step: fold its memory into the job
            parent = jobs.get(job_id.split('.', 1)[0])
            if parent is not None:
                rss = memory_bytes(max_rss)
                if rss is not None:
                    parent.max_rss_bytes = max(parent.max_rss_bytes or 0.0, rss)
                parent.steps.append(job_id)
            continue
        state = state_text.split()[0] if state_text else ''
        jobs[job_id] = JobRecord(
            job_id=job_id,
            name=name,
            state=state,
            state_detail=state_text,
            elapsed_seconds=slurm_seconds(elapsed) or 0.0,
            time_limit_seconds=slurm_seconds(limit),
            cores=int(cores) if cores.isdigit() else 0,
            max_rss_bytes=memory_bytes(max_rss),
            exit_code=exit_code,
            reason='' if reason in ('None', '') else reason,
        )
    return list(jobs.values())


def parse_squeue(text: str) -> list[JobRecord]:
    """Jobs of ``squeue --noheader -o`` :data:`SQUEUE_FORMAT` output (queued and running only).

    Parameters
    ----------
    text : str
        The ``squeue`` output.

    Returns
    -------
    list of JobRecord
        One record per line; ``reason`` holds the pending reason or the node list.
    """
    records = []
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) < 7:
            continue
        job_id, name, state, elapsed, limit, cores, reason = (f.strip() for f in fields[:7])
        records.append(JobRecord(
            job_id=job_id, name=name, state=state, state_detail=state,
            elapsed_seconds=slurm_seconds(elapsed) or 0.0, time_limit_seconds=slurm_seconds(limit),
            cores=int(cores) if cores.isdigit() else 0, reason=reason.strip('()'),
        ))  # fmt: skip
    return records


def billed_cores(requested_cores: int, granularity: int = 1) -> int:
    """Cores charged for a shared-node job: the request rounded up to the billing granularity.

    Parameters
    ----------
    requested_cores : int
        Cores asked for (``-n`` times ``--cpus-per-task``).
    granularity : int
        Billing unit in cores (Snellius fat nodes: 24, one eighth of a node).

    Returns
    -------
    int
        Billed cores.
    """
    if granularity <= 1:
        return requested_cores
    return int(math.ceil(requested_cores / granularity) * granularity)


def balanced_pools(kpoints: int, cores: int, cores_per_pool: int = 6, granularity: int = 1) -> dict:
    """Pools and cores for a pw.x run whose k-points are spread over pools, with the billing taken into account.

    The time of an scf iteration is set by the pool with the most k-points; billed cores times that load measure
    the cost. The function compares the request with one pool per k-point.

    Parameters
    ----------
    kpoints : int
        Irreducible k-points of the run.
    cores : int
        Requested cores.
    cores_per_pool : int
        Cores of one pool.
    granularity : int
        Billing unit in cores.

    Returns
    -------
    dict
        ``pools``, ``kpoints_per_pool`` (the largest), ``billed``, ``cost`` (billed × k-points per pool) for the
        request, and the same for one pool per k-point under ``one_pool_per_kpoint``.
    """

    def layout(total_cores: int) -> dict:
        pools = max(1, min(kpoints, total_cores // cores_per_pool))
        load = int(math.ceil(kpoints / pools))
        billed = billed_cores(pools * cores_per_pool, granularity)
        return {'pools': pools, 'cores': pools * cores_per_pool, 'kpoints_per_pool': load, 'billed': billed,
                'cost': billed * load}  # fmt: skip

    requested = layout(cores)
    requested['one_pool_per_kpoint'] = layout(kpoints * cores_per_pool)
    return requested


def core_hours(records: list[JobRecord], granularity: int = 1) -> float:
    """Billed core hours of a list of jobs (allocated cores rounded up to the billing granularity).

    Parameters
    ----------
    records : list of JobRecord
        Jobs from :func:`parse_sacct`.
    granularity : int
        Billing unit in cores.

    Returns
    -------
    float
        Sum of elapsed hours times billed cores.
    """
    return sum(r.elapsed_seconds / 3600.0 * billed_cores(r.cores, granularity) for r in records)
