"""Machines of a campaign: this computer or an ssh host, with or without Slurm.

A :class:`Machine` runs shell scripts (``bash -s`` locally or through ``ssh``) and reads what a monitor needs in one
round trip per call: the Slurm accounting and queue, and the matching lines, age and size of many files. A failed
ssh raises :class:`MachineUnreachable`: the state of that machine is *unknown*, never an empty queue.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field

from .scheduler import SACCT_FORMAT, SQUEUE_FORMAT, JobRecord, parse_sacct, parse_squeue


class MachineUnreachable(RuntimeError):
    """The machine did not answer (ssh failure or time-out): its state is unknown."""


@dataclass
class RemoteFile:
    """A file seen on a machine: path, age, size and the lines that matched the requested pattern."""

    path: str
    age_seconds: float
    size_bytes: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The matching lines as one text."""
        return '\n'.join(self.lines)


@dataclass
class Machine:
    """A computer that runs part of a campaign.

    Attributes
    ----------
    name : str
        Label used in reports.
    host : str or None
        ssh destination (an alias of ``~/.ssh/config`` with ControlMaster is best); None for this computer.
    scheduler : str
        ``'slurm'`` or ``'none'`` (processes started directly).
    environment : str
        Shell line run before every command (modules, conda, the QE and PAOFLOW environment).
    scratch, permanent : str
        Run area and backed-up storage.
    max_cores : int or None
        Cores the campaign may use at once (workstations shared with other sessions).
    billing_granularity : int
        Cores are billed in multiples of this (Snellius fat nodes: 24).
    charge_per_core_hour : float
        Budget units per core hour (Snellius fat nodes: 1.5 SBU).
    charge_unit : str
        Name of the budget unit.
    scratch_purge_days : int or None
        Age at which the scheduler purges scratch files.
    user_id : str
        Numeric user id (cancellations by other ids are reported as external).
    ssh_options : tuple of str
        Options of every ssh call.
    """

    name: str
    host: str | None = None
    scheduler: str = 'slurm'
    environment: str = ''
    scratch: str = ''
    permanent: str = ''
    max_cores: int | None = None
    billing_granularity: int = 1
    charge_per_core_hour: float = 1.0
    charge_unit: str = 'core-h'
    scratch_purge_days: int | None = None
    user_id: str = ''
    ssh_options: tuple[str, ...] = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20')

    def shell(self, script: str, timeout: float = 300.0) -> str:
        """Run a bash script on the machine and return its standard output.

        Parameters
        ----------
        script : str
            Bash script (``environment`` is run first).
        timeout : float
            Seconds before giving up.

        Returns
        -------
        str
            Standard output.

        Raises
        ------
        MachineUnreachable
            ssh failed (exit status 255) or the call timed out.
        """
        full = (self.environment + '\n' if self.environment else '') + script
        command = (
            ['bash', '-s']
            if self.host is None
            else ['ssh', *self.ssh_options, self.host, 'bash -s']
        )
        try:
            result = subprocess.run(
                command, input=full, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as error:
            raise MachineUnreachable(
                '%s: no answer within %.0f s' % (self.name, timeout)
            ) from error
        if self.host is not None and result.returncode == 255:
            raise MachineUnreachable(
                '%s: ssh failed: %s' % (self.name, result.stderr.strip()[-300:])
            )
        return result.stdout

    # ------------------------------------------------------------------ scheduler
    def accounting(self, since: str, name_prefixes: tuple[str, ...] = ()) -> list[JobRecord]:
        """Jobs of the user since a date (``sacct``), optionally only those whose name starts with a prefix.

        Parameters
        ----------
        since : str
            Start of the window, ``YYYY-MM-DD[THH:MM]`` in the machine's time zone.
        name_prefixes : tuple of str
            Keep only jobs whose names start with one of these.

        Returns
        -------
        list of JobRecord
            Empty on machines without Slurm.
        """
        if self.scheduler != 'slurm':
            return []
        text = self.shell('sacct -u "$USER" -S %s --parsable2 --noheader --format=%s'
                          % (shlex.quote(since), SACCT_FORMAT))  # fmt: skip
        records = parse_sacct(text)
        if name_prefixes:
            records = [r for r in records if r.name.startswith(tuple(name_prefixes))]
        return records

    def queue(self) -> list[JobRecord]:
        """Queued and running jobs of the user (``squeue``); empty on machines without Slurm."""
        if self.scheduler != 'slurm':
            return []
        return parse_squeue(
            self.shell('squeue -u "$USER" --noheader -o %s' % shlex.quote(SQUEUE_FORMAT))
        )

    # ------------------------------------------------------------------ files
    def read_files(
        self,
        directory: str,
        patterns: tuple[str, ...],
        grep: str | None = None,
        max_lines: int = 4000,
    ) -> dict[str, RemoteFile]:
        """Files matching shell globs in a directory, with age, size and the lines matching ``grep``.

        Parameters
        ----------
        directory : str
            Base directory of the globs.
        patterns : tuple of str
            Shell globs relative to ``directory`` (or absolute paths).
        grep : str, optional
            ``grep -E`` pattern of the lines to return (None: no lines, only age and size).
        max_lines : int
            Last matching lines kept per file.

        Returns
        -------
        dict
            ``path -> RemoteFile``.
        """
        if not patterns:
            return {}
        globs = ' '.join(
            p if p.startswith('/') else '%s/%s' % (shlex.quote(directory), p) for p in patterns
        )
        grep_line = (
            ('grep -E -- %s "$f" | tail -n %d' % (shlex.quote(grep), max_lines)) if grep else ':'
        )
        script = (
            'now=$(date +%%s)\n'
            'for f in %s; do\n'
            '  [ -f "$f" ] || continue\n'
            '  m=$(stat -c %%Y "$f" 2>/dev/null || stat -f %%m "$f")\n'
            '  s=$(stat -c %%s "$f" 2>/dev/null || stat -f %%z "$f")\n'
            '  echo "@@@FILE|$f|$((now - m))|$s"\n'
            '  %s\n'
            'done\n'
        ) % (globs, grep_line)
        files: dict[str, RemoteFile] = {}
        current = None
        for line in self.shell(script).splitlines():
            if line.startswith('@@@FILE|'):
                _, path, age, size = line.split('|', 3)
                current = files[path] = RemoteFile(path, float(age), int(size))
            elif current is not None:
                current.lines.append(line)
        return files

    def describe(self) -> str:
        """One line: name, host, scheduler and budget unit."""
        return '%s (%s, %s; %s per core hour × %g, billed in %d-core steps)' % (
            self.name, self.host or 'this computer', self.scheduler, self.charge_unit, self.charge_per_core_hour,
            self.billing_granularity)  # fmt: skip


def local_machine(
    name: str = 'local', max_cores: int | None = None, environment: str = ''
) -> Machine:
    """This computer, without a scheduler (processes started directly)."""
    return Machine(
        name=name,
        host=None,
        scheduler='none',
        max_cores=max_cores or os.cpu_count(),
        environment=environment,
    )
