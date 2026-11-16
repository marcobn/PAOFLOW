"""Findings of a campaign: what went wrong, the evidence, the action, and who may take it.

Every finding carries a ``code`` from :data:`ACTIONS`, which fixes its severity, its *authority* and the action:

* ``report``: tell the user; nothing to do yet;
* ``act``: the agent may act without asking (cancel a job that provably cannot finish or is orphaned, resubmit a
  job once after a node failure, copy results before a scratch purge), then reports at once and writes the
  action to the journal;
* ``ask``: the action changes physics or numerical settings, launches work outside the approved plan or spends
  budget: propose it with its cost and wait.

The rules encode the failures of the TaC, TaN, HfC and TaC–HfC production campaigns (``agentic/campaign/
LESSONS_LEARNED.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .pw_output import PwInput, PwOutput
from .scheduler import JobRecord

SEVERITY_ORDER = {'error': 0, 'warning': 1, 'info': 2}

ACTIONS: dict[str, tuple[str, str, str]] = {
    # code: (severity, authority, action)
    'cannot_finish': (
        'error',
        'act',
        'cancel the job: by its own convergence and timing it cannot finish within its '
        'time limit (journal the evidence); then find the cause before any resubmission',
    ),
    'may_not_finish': (
        'warning',
        'ask',
        'the projected end is after the time limit: let it run and prepare the fix, '
        'or cancel now if the fix is certain',
    ),
    'charge_sloshing': (
        'warning',
        'ask',
        "damp the mixing (mixing_mode = 'local-TF', mixing_beta 0.2-0.3, "
        "mixing_ndim 12) and start from a converged density of the same cell (startingpot = 'file')",
    ),
    'scf_stagnating': (
        'warning',
        'ask',
        'the scf accuracy stopped improving: damp the mixing, check smearing and bands',
    ),
    'kpoint_inflation': (
        'warning',
        'ask',
        'the cell has no symmetry but more k-points than its grid reduced by time '
        'reversal: the grid is not invariant under the lattice group; use nosym = .true. or an '
        'invariant grid',
    ),
    'fft_inflation': (
        'warning',
        'ask',
        'the FFT box is much larger than the volume needs (skewed basis): rewrite the '
        'cell in a reduced basis of the same lattice (cells.reduce_cell_input)',
    ),
    'output_stalled': (
        'error',
        'ask',
        'the output has not changed for several iterations while the job runs: a rank '
        'may have died (check every log); cancel if a failure is found',
    ),
    'output_stalled_after_error': (
        'error',
        'act',
        'a rank failed and the job is hanging in a collective: cancel it '
        'and fix the error (run MPI drivers through run_main or python -m mpi4py)',
    ),
    'slow_iterations': (
        'info',
        'report',
        'iterations are slower than the reference: check k-points, FFT box, pool '
        'layout and node sharing',
    ),
    'job_timeout': (
        'error',
        'ask',
        'find why it was slow (k-points, FFT box, sloshing, pool layout) and fix that before '
        'resubmitting; re-estimate the cost',
    ),
    'job_out_of_memory': (
        'error',
        'ask',
        'more memory per task (fewer ranks per node, more nodes, --mem) or smaller '
        'buffers (bands, FFT box, interpolation grid)',
    ),
    'job_node_failure': (
        'warning',
        'act',
        'resubmit the identical job once; ask if it fails again',
    ),
    'job_failed': (
        'error',
        'ask',
        'read the end of the job log and the program output, fix the cause, resubmit the stage',
    ),
    'job_cancelled_externally': (
        'warning',
        'report',
        'cancelled by another user or the system: ask before resubmitting',
    ),
    'dependency_never_satisfied': (
        'error',
        'act',
        'cancel the orphaned job (its parent failed or was cancelled); '
        'resubmitting the chain from the failed stage is a separate decision',
    ),
    'long_pending': (
        'info',
        'report',
        'queue priority or resources: a core count that fits the free nodes, another '
        'partition or another machine may start sooner (ask before changing)',
    ),
    'scf_not_converged': (
        'error',
        'ask',
        'the scf stopped without convergence: damp the mixing, raise electron_maxstep, '
        'check the smearing',
    ),
    'max_seconds': (
        'info',
        'act',
        'the run stopped cleanly at max_seconds: resubmit with restart_mode = '
        "'restart' (same work, within the approved plan)",
    ),
    'davidson_cholesky': (
        'error',
        'ask',
        'Davidson failed (overlap matrix not positive definite): diagonalization = '
        "'cg' (QE 7.5 has no ppcg), or more bands / fewer random starting wavefunctions",
    ),
    'bands_not_converged': (
        'warning',
        'report',
        'some bands did not converge in the diagonalisation: harmless if '
        'rare, otherwise diago_thr_init, more bands or cg',
    ),
    'mpi_count_overflow': (
        'error',
        'ask',
        'an array beyond the 32-bit MPI count: a PAOFLOW with chunked collectives, '
        'or a smaller interpolation grid',
    ),
    'mpi_abort': (
        'error',
        'ask',
        'a rank aborted: read the first error of any rank, fix it, resubmit the stage',
    ),
    'python_exception': (
        'error',
        'ask',
        'read the traceback, fix the driver or its inputs, resubmit the stage',
    ),
    'out_of_memory': ('error', 'ask', 'more memory per task or smaller buffers'),
    'segmentation_fault': (
        'error',
        'ask',
        'crash in a compiled code: memory, stack size (ulimit -s), a broken build or '
        'a corrupted input',
    ),
    'time_limit': ('error', 'ask', 'see job_timeout'),
    'node_failure': ('warning', 'act', 'see job_node_failure'),
    'qgrid_does_not_divide_kgrid': (
        'error',
        'ask',
        'make the coupling k-grid a multiple of the q-grid (it only enters '
        'the system description when no nscf runs)',
    ),
    'kpoints_not_accounted': (
        'error',
        'ask',
        'the k-point list does not cover the grid: check nosym/noinv and the grid of the save',
    ),
    'missing_file': (
        'error',
        'ask',
        'an input of the stage is missing (save, force constants, markers): link or '
        'produce it, then resubmit the stage',
    ),
    'qe_error': ('error', 'ask', 'read the error message of the program'),
    'budget_warning': (
        'warning',
        'report',
        'most of the approved budget is spent: report the spend and the estimate to finish',
    ),
    'budget_exceeded': (
        'error',
        'ask',
        'the approved budget is spent: launch nothing more; report the spend and the '
        'estimate to finish',
    ),
    'scratch_purge_risk': ('warning', 'act', 'copy the results to permanent storage now'),
    'machine_unreachable': (
        'warning',
        'report',
        'state unknown (not an empty queue); retry, and tell the user if it persists',
    ),
    'two_point_axis': (
        'error',
        'ask',
        'at least three k-points per axis for any run whose Hamiltonian is Fourier '
        'interpolated (PAOFLOW splits the Nyquist term)',
    ),
    'coarse_kgrid': (
        'info',
        'report',
        'the shortest superlattice vector of the k-grid is below the floor of the campaign',
    ),
    'plain_mixing_large_cell': (
        'warning',
        'ask',
        'large metallic cell with plain mixing at a large beta: '
        "mixing_mode = 'local-TF', mixing_beta 0.3",
    ),
    'pool_layout': (
        'info',
        'report',
        'a pool layout with fewer k-points per pool (or one pool per k-point) costs less '
        'per iteration',
    ),
    'billing_rounding': (
        'info',
        'report',
        'the core request is rounded up by the billing granularity: use the cores you pay for',
    ),
    'run_complete': ('info', 'report', 'collect the results'),
}


@dataclass
class Finding:
    """One problem (or event) of a run, with its evidence and the action.

    Attributes
    ----------
    code : str
        Key of :data:`ACTIONS`.
    run : str
        Name of the run (or ``campaign``).
    evidence : str
        What was observed, with numbers and file names.
    severity, authority, action : str
        From :data:`ACTIONS` unless given.
    """

    code: str
    run: str
    evidence: str
    severity: str = ''
    authority: str = ''
    action: str = ''

    def __post_init__(self) -> None:
        severity, authority, action = ACTIONS.get(self.code, ('warning', 'ask', 'investigate'))
        self.severity = self.severity or severity
        self.authority = self.authority or authority
        self.action = self.action or action

    @property
    def key(self) -> str:
        """``run:code``, the identity of the finding across polls."""
        return '%s:%s' % (self.run, self.code)

    def line(self) -> str:
        """One Markdown line: severity, run, code, evidence, authority and action."""
        return '- **%s** `%s` %s: %s → *%s*: %s' % (
            self.severity, self.run, self.code, self.evidence, self.authority, self.action)  # fmt: skip


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Findings worst first (error, warning, info), then by run."""
    return sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.run, f.code))


def findings_for_job(
    record: JobRecord, run: str, pending_limit_hours: float = 24.0, own_user_id: str = ''
) -> list[Finding]:
    """Findings from the scheduler state of one job.

    Parameters
    ----------
    record : JobRecord
        The job.
    run : str
        Run it belongs to.
    pending_limit_hours : float
        Pending longer than this (by its elapsed field) is reported.
    own_user_id : str
        Numeric user id; a cancellation by another id is external.

    Returns
    -------
    list of Finding
    """
    label = '%s (%s)' % (record.job_id, record.name)
    if record.state == 'TIMEOUT':
        return [
            Finding(
                'job_timeout',
                run,
                '%s reached its limit after %.1f h' % (label, record.elapsed_seconds / 3600),
            )
        ]
    if record.state == 'OUT_OF_MEMORY':
        rss = record.max_rss_bytes / 1024**3 if record.max_rss_bytes else float('nan')
        return [Finding('job_out_of_memory', run, '%s, largest step %.1f GB' % (label, rss))]
    if record.state in ('NODE_FAIL', 'BOOT_FAIL'):
        return [Finding('job_node_failure', run, '%s %s' % (label, record.state))]
    if record.state in ('FAILED', 'DEADLINE', 'PREEMPTED'):
        return [
            Finding('job_failed', run, '%s %s, exit %s' % (label, record.state, record.exit_code))
        ]
    if record.state == 'CANCELLED' and own_user_id and 'by ' in record.state_detail:
        if record.state_detail.split('by ', 1)[1].strip() != own_user_id:
            return [
                Finding('job_cancelled_externally', run, '%s %s' % (label, record.state_detail))
            ]
    if record.state == 'PENDING':
        if 'DependencyNeverSatisfied' in record.reason:
            return [
                Finding(
                    'dependency_never_satisfied',
                    run,
                    '%s pending, DependencyNeverSatisfied' % label,
                )
            ]
        if record.elapsed_seconds == 0 and record.reason in ('Priority', 'Resources'):
            return []
    return []


def findings_for_pw(
    output: PwOutput,
    run: str,
    output_name: str,
    job: JobRecord | None = None,
    settings: PwInput | None = None,
    conv_thr: float = 1e-8,
    output_age_seconds: float | None = None,
    reference_seconds_per_iteration: float | None = None,
    stall_iterations: float = 3.0,
) -> list[Finding]:
    """Findings from a pw.x output (and the job running it).

    Parameters
    ----------
    output : PwOutput
        Parsed output (:func:`~.pw_output.parse_pw_output`).
    run, output_name : str
        Run and file, for the evidence.
    job : JobRecord, optional
        The job running the output (enables the time-limit projection and the stall check).
    settings : PwInput, optional
        The input of the run (enables the k-point check).
    conv_thr : float
        scf threshold of the run (Ry).
    output_age_seconds : float, optional
        Seconds since the output was last written.
    reference_seconds_per_iteration : float, optional
        Expected time per scf iteration (e.g. from a pilot); slower runs are reported.
    stall_iterations : float
        An output unchanged for this many iterations (at least 30 min) while the job runs is stalled.

    Returns
    -------
    list of Finding
    """
    findings: list[Finding] = []
    where = '%s/%s' % (run, output_name)
    accuracies = ', '.join('%.1e' % a for a in output.accuracies[-4:])
    seconds = output.seconds_per_iteration()
    running = job is not None and job.state == 'RUNNING'

    if output.converged is False:
        findings.append(
            Finding(
                'scf_not_converged',
                run,
                '%s: no convergence after %d iterations' % (where, output.iterations),
            )
        )
    if output.max_seconds_reached:
        findings.append(Finding('max_seconds', run, '%s stopped at max_seconds' % where))
    if output.largest_rise() > 10.0 and not output.converged:
        findings.append(Finding('charge_sloshing', run, '%s: accuracy rose %.0f-fold above the first estimate; last %s Ry'
                                % (where, output.largest_rise(), accuracies)))  # fmt: skip
    elif output.stagnating(conv_thr=conv_thr) and not output.converged:
        findings.append(Finding('scf_stagnating', run, '%s: no progress over the last 6 iterations; last %s Ry'
                                % (where, accuracies)))  # fmt: skip

    if running and output.converged is None and seconds and job.remaining_seconds is not None:
        remaining_iterations = output.iterations_to_converge(conv_thr)
        if remaining_iterations is not None:
            needed = (
                remaining_iterations * seconds if math.isfinite(remaining_iterations) else math.inf
            )
            evidence = ('%s: iteration %d, last %s Ry, %.0f s/iteration, %s more needed (%.1f h), %.1f h left'
                        % (where, output.iterations, accuracies, seconds,
                           'no trend' if not math.isfinite(remaining_iterations) else '%.0f' % remaining_iterations,
                           needed / 3600 if math.isfinite(needed) else float('inf'), job.remaining_seconds / 3600))  # fmt: skip
            if needed > 2.0 * job.remaining_seconds and output.iterations >= 6:
                findings.append(Finding('cannot_finish', run, evidence))
            elif needed > job.remaining_seconds:
                findings.append(Finding('may_not_finish', run, evidence))

    if running and output_age_seconds is not None and output.iterations > 0 and not output.finished:
        limit = max(stall_iterations * (seconds or 600.0), 1800.0)
        if output_age_seconds > limit:
            findings.append(Finding('output_stalled', run, '%s unchanged for %.0f min (%.0f s/iteration)'
                                    % (where, output_age_seconds / 60, seconds or float('nan'))))  # fmt: skip

    if (
        settings is not None
        and settings.kgrid
        and output.symmetry_operations == 0
        and not settings.nosym
    ):
        from .cells import time_reversal_kpoints

        expected = time_reversal_kpoints(settings.kgrid)
        if output.kpoints and output.kpoints > expected:
            findings.append(Finding('kpoint_inflation', run, '%s: %d k-points for the %s grid without symmetry '
                                    '(%d with time reversal alone)' % (where, output.kpoints, 'x'.join(map(str, settings.kgrid)), expected)))  # fmt: skip

    if output.cell_bohr is not None and output.ecutrho:
        from .cells import fft_box, fft_inflation, reduced_basis

        ratio = fft_inflation(output.cell_bohr, output.ecutrho)
        if ratio > 2.0:
            reduced = fft_box(reduced_basis(output.cell_bohr)[0], output.ecutrho)
            findings.append(Finding('fft_inflation', run, '%s: FFT box %s, %.1f times the %s of a reduced basis'
                                    % (where, output.fft_dimensions, ratio, tuple(reduced))))  # fmt: skip

    if (
        seconds
        and reference_seconds_per_iteration
        and seconds > 1.5 * reference_seconds_per_iteration
    ):
        findings.append(Finding('slow_iterations', run, '%s: %.0f s/iteration against %.0f expected'
                                % (where, seconds, reference_seconds_per_iteration)))  # fmt: skip
    for error in output.errors:
        code = (
            'davidson_cholesky'
            if 'cholesky' in error or 'positive definite' in error
            else 'qe_error'
        )
        findings.append(Finding(code, run, '%s: %s' % (where, error)))
    return findings


def findings_for_log(signatures: list[tuple[str, str]], run: str, log_name: str) -> list[Finding]:
    """Findings from the error signatures of a log (:func:`~.pw_output.error_signatures`)."""
    return [Finding(code, run, '%s: %s' % (log_name, line)) for code, line in signatures]


def findings_for_budget(
    spent: float, approved: float | None, unit: str = 'core-h', warn_fraction: float = 0.8
) -> list[Finding]:
    """Findings of the spend against the approved budget of the campaign."""
    if not approved:
        return []
    fraction = spent / approved
    evidence = '%.0f of %.0f %s approved (%.0f %%)' % (spent, approved, unit, 100 * fraction)
    if fraction >= 1.0:
        return [Finding('budget_exceeded', 'campaign', evidence)]
    if fraction >= warn_fraction:
        return [Finding('budget_warning', 'campaign', evidence)]
    return []


def findings_for_storage(
    oldest_age_days: float | None, purge_days: int | None, run: str, margin_days: float = 3.0
) -> list[Finding]:
    """A finding when the oldest file of a run on scratch nears the purge age of the machine."""
    if oldest_age_days is None or not purge_days:
        return []
    if oldest_age_days >= purge_days - margin_days:
        return [Finding('scratch_purge_risk', run, 'files %.1f days old, scratch purged after %d days'
                        % (oldest_age_days, purge_days))]  # fmt: skip
    return []
