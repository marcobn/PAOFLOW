"""A campaign: runs on several machines, their state read from the scheduler and the files, diagnosed and journaled.

The campaign is described in Python (the study's ``campaign.py``, from ``agentic/campaign/templates/``)::

    snellius = Machine('snellius', host='snellius', billing_granularity=24, charge_per_core_hour=1.5,
                       charge_unit='SBU', scratch_purge_days=14)
    CAMPAIGN = Campaign('TaHfC series', study_dir=HERE, since='2026-11-15', approved_budget=7000, runs=[
        CampaignRun('x50_111_T2800', snellius, '/scratch-shared/me/TaX/TaHfC_x50_111_T2800',
                    job_prefix='tax_TaHfC_x50_111_T2800_', pw_outputs=('thermal/zg/seed0/*.out',),
                    logs=('jobs/logs/*.out',), results=('thermal/output/emissivity_zg.dat',)),
        ...])

and read with ``CAMPAIGN.status()``, ``CAMPAIGN.diagnose()``, ``CAMPAIGN.describe()``; actions go to
``CAMPAIGN.journal(...)``.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field

from .diagnosis import (
    Finding,
    findings_for_budget,
    findings_for_job,
    findings_for_log,
    findings_for_pw,
    findings_for_storage,
    sort_findings,
)
from .machines import Machine, MachineUnreachable
from .pw_output import (
    ERROR_SIGNATURES,
    PW_INPUT_PATTERN,
    PW_OUTPUT_PATTERN,
    error_signatures,
    parse_pw_input,
    parse_pw_output,
)
from .scheduler import JobRecord, billed_cores

ERROR_PATTERN = '|'.join('(%s)' % pattern for pattern in ERROR_SIGNATURES.values())


@dataclass
class CampaignRun:
    """One run of a campaign: a directory on a machine, its jobs and the files that tell its state.

    Attributes
    ----------
    name : str
        Short label.
    machine : Machine
        Where it runs.
    directory : str
        Run directory (globs below are relative to it).
    job_prefix : str or None
        Slurm jobs whose names start with this belong to the run.
    pw_outputs : tuple of str
        Globs of the pw.x outputs to follow (the input is the file with the same stem and ``.in``).
    logs : tuple of str
        Globs of job and driver logs to scan for error signatures.
    results : tuple of str
        Globs of the files whose presence means the run is done.
    conv_thr : float
        scf threshold of its pw.x runs (Ry).
    reference_seconds_per_iteration : float or None
        Expected time per scf iteration (from a pilot).
    """

    name: str
    machine: Machine
    directory: str
    job_prefix: str | None = None
    pw_outputs: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    conv_thr: float = 1e-8
    reference_seconds_per_iteration: float | None = None


@dataclass
class RunState:
    """What a poll found for one run."""

    run: CampaignRun
    reachable: bool = True
    jobs: list[JobRecord] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)  # path -> (PwOutput, PwInput or None, age seconds)
    log_errors: dict = field(default_factory=dict)  # path -> [(code, line)]
    results: list[str] = field(default_factory=list)
    oldest_age_days: float | None = None
    error: str = ''

    @property
    def active_jobs(self) -> list[JobRecord]:
        """Jobs still pending or running."""
        return [job for job in self.jobs if not job.finished]

    @property
    def done(self) -> bool:
        """All result files present and no job left in the queue."""
        return (
            bool(self.run.results)
            and len(self.results) >= len(self.run.results)
            and not self.active_jobs
        )


class Campaign:
    """A set of runs on several machines, with an approved budget, a study folder and a journal.

    Parameters
    ----------
    name : str
        Name of the campaign.
    runs : list of CampaignRun
        Its runs.
    study_dir : str
        Folder of the study (``PLAN.md``, ``STATUS.md``, ``JOURNAL.md``, ``REPORT.md``).
    since : str
        Start of the accounting window (``YYYY-MM-DD[THH:MM]``).
    approved_budget : float, optional
        Budget approved by the user, in the charge unit of the machines (or core hours).
    pending_limit_hours : float
        Pending jobs older than this are reported.
    """

    def __init__(
        self,
        name: str,
        runs: list[CampaignRun],
        study_dir: str,
        since: str,
        approved_budget: float | None = None,
        pending_limit_hours: float = 24.0,
    ) -> None:
        self.name = name
        self.runs = list(runs)
        self.study_dir = study_dir
        self.since = since
        self.approved_budget = approved_budget
        self.pending_limit_hours = pending_limit_hours

    @property
    def machines(self) -> list[Machine]:
        """Distinct machines of the runs, in order of appearance."""
        seen: dict[str, Machine] = {}
        for run in self.runs:
            seen.setdefault(run.machine.name, run.machine)
        return list(seen.values())

    # ------------------------------------------------------------------ state
    def status(self) -> list[RunState]:
        """Poll every machine once for its jobs and every run for its files.

        Returns
        -------
        list of RunState
            One per run; ``reachable`` is False (and ``error`` set) when its machine did not answer.
        """
        jobs_by_machine: dict[str, list[JobRecord] | None] = {}
        for machine in self.machines:
            prefixes = tuple(
                r.job_prefix for r in self.runs if r.machine.name == machine.name and r.job_prefix
            )
            try:
                jobs_by_machine[machine.name] = machine.accounting(self.since, prefixes)
            except MachineUnreachable:
                jobs_by_machine[machine.name] = None
        states = []
        for run in self.runs:
            state = RunState(run)
            jobs = jobs_by_machine.get(run.machine.name)
            if jobs is None:
                state.reachable, state.error = False, '%s did not answer' % run.machine.name
                states.append(state)
                continue
            state.jobs = [
                job for job in jobs if run.job_prefix and job.name.startswith(run.job_prefix)
            ]
            try:
                self._read_files(run, state)
            except MachineUnreachable as error:
                state.reachable, state.error = False, str(error)
            states.append(state)
        return states

    @staticmethod
    def _read_files(run: CampaignRun, state: RunState) -> None:
        machine = run.machine
        outputs = machine.read_files(run.directory, run.pw_outputs, PW_OUTPUT_PATTERN)
        inputs = machine.read_files(
            run.directory, tuple(os.path.splitext(p)[0] + '.in' for p in outputs), PW_INPUT_PATTERN
        )
        for path, remote in outputs.items():
            source = inputs.get(os.path.splitext(path)[0] + '.in')
            state.outputs[path] = (parse_pw_output(remote.text), parse_pw_input(source.text) if source else None,
                                   remote.age_seconds)  # fmt: skip
        for path, remote in machine.read_files(
            run.directory, run.logs, ERROR_PATTERN, max_lines=200
        ).items():
            found = error_signatures(remote.text)
            if found:
                state.log_errors[path] = found
        results = machine.read_files(run.directory, run.results)
        state.results = sorted(results)
        ages = [f.age_seconds for f in list(outputs.values()) + list(results.values())]
        state.oldest_age_days = max(ages) / 86400.0 if ages else None

    # ------------------------------------------------------------------ diagnosis
    def spent(self, states: list[RunState] | None = None) -> dict[str, tuple[float, float, str]]:
        """Billed core hours and budget units per machine, from the jobs of the runs.

        Returns
        -------
        dict
            ``machine -> (core hours, charge, unit)``.
        """
        states = states if states is not None else self.status()
        totals: dict[str, tuple[float, float, str]] = {}
        for state in states:
            machine = state.run.machine
            hours = sum(job.elapsed_seconds / 3600.0 * billed_cores(job.cores, machine.billing_granularity)
                        for job in state.jobs)  # fmt: skip
            old = totals.get(machine.name, (0.0, 0.0, machine.charge_unit))
            totals[machine.name] = (
                old[0] + hours,
                old[1] + hours * machine.charge_per_core_hour,
                machine.charge_unit,
            )
        return totals

    def diagnose(self, states: list[RunState] | None = None) -> list[Finding]:
        """Findings of every run and of the campaign, worst first.

        Parameters
        ----------
        states : list of RunState, optional
            A poll from :meth:`status` (polled now if omitted).

        Returns
        -------
        list of Finding
        """
        states = states if states is not None else self.status()
        findings: list[Finding] = []
        for state in states:
            run = state.run
            if not state.reachable:
                findings.append(Finding('machine_unreachable', run.name, state.error))
                continue
            for job in state.jobs:
                findings += findings_for_job(
                    job, run.name, self.pending_limit_hours, run.machine.user_id
                )
            running = [job for job in state.jobs if job.state == 'RUNNING']
            for path, (output, settings, age) in state.outputs.items():
                job = running[0] if len(running) == 1 else None
                findings += findings_for_pw(output, run.name, os.path.relpath(path, run.directory), job, settings,
                                            run.conv_thr, age, run.reference_seconds_per_iteration)  # fmt: skip
            log_failure = False
            for path, found in state.log_errors.items():
                findings += findings_for_log(found, run.name, os.path.relpath(path, run.directory))
                log_failure = log_failure or any(code in ('python_exception', 'mpi_abort', 'mpi_count_overflow')
                                                 for code, _ in found)  # fmt: skip
            if (
                log_failure
            ):  # a rank failed and the others wait in a collective: the stall is a hang
                findings = [
                    Finding('output_stalled_after_error', f.run, f.evidence)
                    if f.run == run.name and f.code == 'output_stalled' else f
                    for f in findings
                ]  # fmt: skip
            findings += findings_for_storage(
                state.oldest_age_days, run.machine.scratch_purge_days, run.name
            )
            if state.done:
                findings.append(
                    Finding('run_complete', run.name, '%d result file(s)' % len(state.results))
                )
        charge = sum(value[1] for value in self.spent(states).values())
        findings += findings_for_budget(charge, self.approved_budget, self._unit())
        return sort_findings(findings)

    def _unit(self) -> str:
        units = {m.charge_unit for m in self.machines}
        return units.pop() if len(units) == 1 else 'units'

    # ------------------------------------------------------------------ reports
    def describe(
        self, states: list[RunState] | None = None, findings: list[Finding] | None = None
    ) -> str:
        """Markdown status: one table row per run, the spend, the findings.

        Returns
        -------
        str
            The text of ``STATUS.md``.
        """
        states = states if states is not None else self.status()
        findings = findings if findings is not None else self.diagnose(states)
        now = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        lines = ['# %s: status %s' % (self.name, now), '',
                 '| run | machine | jobs | scf | results |', '|---|---|---|---|---|']  # fmt: skip
        for state in states:
            run = state.run
            if not state.reachable:
                lines.append(
                    '| %s | %s | unknown | unknown | unknown |' % (run.name, run.machine.name)
                )
                continue
            counts: dict[str, int] = {}
            for job in state.jobs:
                counts[job.state] = counts.get(job.state, 0) + 1
            jobs = ', '.join('%d %s' % (n, s.lower()) for s, n in sorted(counts.items())) or '–'
            scf = []
            for path, (output, _, _) in state.outputs.items():
                tag = os.path.basename(path)
                if output.finished:
                    scf.append('%s done (%d it)' % (tag, output.iterations))
                elif output.accuracies:
                    scf.append(
                        '%s it %d, %.1e Ry' % (tag, output.iterations, output.accuracies[-1])
                    )
            results = '%d/%d' % (len(state.results), len(run.results)) if run.results else '–'
            lines.append(
                '| %s | %s | %s | %s | %s |'
                % (run.name, run.machine.name, jobs, '; '.join(scf) or '–', results)
            )
        lines += ['', '**Spend:** ' + '; '.join('%s %.0f core-h = %.0f %s' % (name, hours, charge, unit)
                                                for name, (hours, charge, unit) in self.spent(states).items())]  # fmt: skip
        if self.approved_budget:
            lines[-1] += ' (approved %.0f %s)' % (self.approved_budget, self._unit())
        lines += ['', '## Findings', ''] + ([f.line() for f in findings] or ['none'])
        return '\n'.join(lines) + '\n'

    def write_status(
        self, states: list[RunState] | None = None, findings: list[Finding] | None = None
    ) -> str:
        """Write ``STATUS.md`` in the study folder and return its path."""
        path = os.path.join(self.study_dir, 'STATUS.md')
        with open(path, 'w') as handle:
            handle.write(self.describe(states, findings))
        return path

    def journal(self, action: str, evidence: str = '', findings: list[Finding] | tuple = ()) -> str:
        """Append a dated entry to ``JOURNAL.md`` in the study folder (every action of the agent goes there).

        Parameters
        ----------
        action : str
            What was done (commands, job ids).
        evidence : str
            Why (numbers, files).
        findings : list of Finding
            Findings the action answers.

        Returns
        -------
        str
            Path of the journal.
        """
        path = os.path.join(self.study_dir, 'JOURNAL.md')
        stamp = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        entry = ['', '## %s' % stamp, '', '- **Action:** %s' % action]
        if evidence:
            entry.append('- **Evidence:** %s' % evidence)
        entry += ['- **Finding:** `%s` %s (%s)' % (f.code, f.run, f.authority) for f in findings]
        new = not os.path.exists(path)
        with open(path, 'a') as handle:
            if new:
                handle.write('# %s: journal of actions\n' % self.name)
            handle.write('\n'.join(entry) + '\n')
        return path
