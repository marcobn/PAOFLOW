"""Unit tests of PAOFLOW.agentic.campaign: scheduler records, pw.x outputs, cells, preflight, diagnosis, campaign, watch.

The data are filtered lines (``PW_OUTPUT_PATTERN``) of real Ta0.8Hf0.2C supercell runs on Snellius (November 2026):

* ``hftac_p2_t2800_timeout``: 80 atoms, skewed basis (FFT 192^3), 4x4x3 grid without nosym (81 k-points), plain
  mixing at beta 0.7 (sloshing), timed out at 36 h;
* ``hftac_p1_t300_sloshing``: rhombohedral 80-atom cell (FFT 288^3), accuracy 25 -> 9300 Ry in four iterations;
* ``hftac_p2_t2800_reduced_ref``: the same lattice in a reduced basis (FFT 72x100x192), nosym, damped mixing:
  converged in 11 iterations;
* ``sacct_steps.txt``: accounting with steps (a timeout, a cancellation, a completed job).
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import pytest

from PAOFLOW.agentic.campaign import (
    Campaign,
    CampaignRun,
    Finding,
    balanced_pools,
    billed_cores,
    choose_kgrid,
    error_signatures,
    fft_box,
    fft_inflation,
    local_machine,
    parse_pw_input,
    parse_pw_output,
    parse_sacct,
    parse_squeue,
    preflight_pw_input,
    reduce_cell_input,
    reduced_basis,
    shortest_superlattice_vector,
    time_reversal_kpoints,
    watch,
)
from PAOFLOW.agentic.campaign.diagnosis import (
    ACTIONS,
    findings_for_budget,
    findings_for_job,
    findings_for_pw,
)
from PAOFLOW.agentic.campaign.scheduler import JobRecord, slurm_seconds

pytestmark = pytest.mark.unit

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _read(name: str) -> str:
    return open(os.path.join(DATA, name)).read()


# ---------------------------------------------------------------------------------------------------- scheduler
def test_slurm_durations_and_billing():
    assert slurm_seconds('1-12:00:05') == 129605
    assert slurm_seconds('01:22:33') == 4953
    assert slurm_seconds('20:26') == 1226
    assert slurm_seconds('UNLIMITED') is None
    assert (
        billed_cores(60, 24) == 72 and billed_cores(114, 24) == 120 and billed_cores(48, 24) == 48
    )
    layout = balanced_pools(kpoints=19, cores=60, cores_per_pool=6, granularity=24)
    assert (layout['pools'], layout['kpoints_per_pool'], layout['billed'], layout['cost']) == (
        10,
        2,
        72,
        144,
    )
    best = layout['one_pool_per_kpoint']
    assert (best['pools'], best['cores'], best['billed'], best['cost']) == (19, 114, 120, 120)


def test_sacct_folds_steps_into_jobs():
    records = parse_sacct(_read('sacct_steps.txt'))
    assert [r.job_id for r in records] == ['29060735_2', '29095642_2', '29144365_1']
    timeout, cancelled, completed = records
    assert timeout.state == 'TIMEOUT' and timeout.failed and timeout.finished
    assert timeout.max_rss_bytes == pytest.approx(6339021 * 1024)
    assert cancelled.state == 'CANCELLED' and cancelled.state_detail == 'CANCELLED by 79346'
    assert completed.cores == 72 and completed.core_hours == pytest.approx(4953 * 72 / 3600)
    queue = parse_squeue('29274939_1|tax_x_zgr_ref|RUNNING|36:21|12:00:00|120|fcn73\n'
                         '29274940_[2]|tax_x_zgr_cells|PENDING|0:00|12:00:00|120|(Dependency)\n')  # fmt: skip
    assert queue[0].remaining_seconds == pytest.approx(12 * 3600 - 36 * 60 - 21)
    assert queue[1].reason == 'Dependency'


def test_job_findings():
    timeout, cancelled, completed = parse_sacct(_read('sacct_steps.txt'))
    assert [f.code for f in findings_for_job(timeout, 'P2')] == ['job_timeout']
    assert findings_for_job(cancelled, 'P1', own_user_id='79346') == []
    assert [f.code for f in findings_for_job(cancelled, 'P1', own_user_id='1')] == [
        'job_cancelled_externally'
    ]
    assert findings_for_job(completed, 'P2') == []
    orphan = JobRecord('9', 'tax_x_zg_post', 'PENDING', reason='DependencyNeverSatisfied')
    finding = findings_for_job(orphan, 'x')[0]
    assert finding.code == 'dependency_never_satisfied' and finding.authority == 'act'


# ---------------------------------------------------------------------------------------------------- pw.x
def test_pw_output_of_the_timed_out_cell():
    output = parse_pw_output(_read('hftac_p2_t2800_timeout.out'))
    assert (output.kpoints, output.fft_dimensions, output.symmetry_operations) == (
        81,
        (192, 192, 192),
        0,
    )
    assert output.converged is None and not output.finished and output.iterations == 25
    assert output.ecutrho == 328.0 and output.cell_bohr.shape == (3, 3)
    assert 4000 < output.seconds_per_iteration() < 5000
    assert output.largest_rise() > 100
    assert fft_inflation(output.cell_bohr, output.ecutrho) == pytest.approx(5.13, abs=0.01)
    settings = parse_pw_input(_read('hftac_p2_t2800_timeout.in'))
    assert (
        settings.kgrid == (4, 4, 3)
        and not settings.nosym
        and settings.mixing_beta == 0.7
        and settings.atoms == 80
    )


def test_pw_output_of_the_reduced_basis_reference():
    output = parse_pw_output(_read('hftac_p2_t2800_reduced_ref.out'))
    assert output.finished and output.converged and output.iterations == 11
    assert output.kpoints == 19 and output.fft_dimensions == (72, 100, 192)
    assert output.largest_rise() == 1.0 and output.iterations_to_converge() == 0.0
    assert output.wall_seconds == pytest.approx(4925.79)
    assert fft_inflation(output.cell_bohr, output.ecutrho) == pytest.approx(1.0)


def test_findings_of_the_timed_out_cell_name_its_four_causes():
    output = parse_pw_output(_read('hftac_p2_t2800_timeout.out'))
    settings = parse_pw_input(_read('hftac_p2_t2800_timeout.in'))
    job = JobRecord('1', 'cells', 'RUNNING', elapsed_seconds=120000, time_limit_seconds=129600)
    codes = {
        f.code: f
        for f in findings_for_pw(output, 'P2', 'cell.out', job, settings, output_age_seconds=600)
    }
    assert set(codes) == {'charge_sloshing', 'cannot_finish', 'kpoint_inflation', 'fft_inflation'}
    assert (
        codes['cannot_finish'].authority == 'act' and codes['kpoint_inflation'].authority == 'ask'
    )
    assert (
        '81 k-points' in codes['kpoint_inflation'].evidence
        and '26 with time reversal' in codes['kpoint_inflation'].evidence
    )
    # 12.9 h still needed with 8.3 h left: a warning to prepare the fix, not a cancellation (needs > 2x the time left)
    relaxed = JobRecord(
        '1', 'cells', 'RUNNING', elapsed_seconds=1000, time_limit_seconds=1000 + 30000
    )
    codes = {f.code for f in findings_for_pw(output, 'P2', 'cell.out', relaxed, settings)}
    assert 'cannot_finish' not in codes and 'may_not_finish' in codes
    # a stalled output while the job runs
    stalled = findings_for_pw(output, 'P2', 'cell.out', job, settings, output_age_seconds=6 * 3600)
    assert 'output_stalled' in {f.code for f in stalled}


def test_sloshing_cell_and_clean_reference():
    sloshing = parse_pw_output(_read('hftac_p1_t300_sloshing.out'))
    assert sloshing.largest_rise() > 300 and fft_inflation(
        sloshing.cell_bohr, sloshing.ecutrho
    ) == pytest.approx(16.0, abs=0.1)
    clean = parse_pw_output(_read('hftac_p2_t2800_reduced_ref.out'))
    assert findings_for_pw(clean, 'P2', 'ref.out', None, None) == []


def test_error_signatures():
    log = textwrap.dedent("""
         Error in routine cdiaghg (155):
         problems computing cholesky
        Traceback (most recent call last):
        ValueError: the q-grid (3, 3, 2) must divide the k-grid (8, 8, 8)
        MPI_ERR_ARG: invalid argument of some other kind
        slurmstepd: error: *** JOB 123 ON fcn1 CANCELLED AT 2026-11-12T10:00:00 DUE TO TIME LIMIT ***
    """)  # fmt: skip
    codes = [code for code, _ in error_signatures(log)]
    for code in ('davidson_cholesky', 'qe_error', 'python_exception', 'qgrid_does_not_divide_kgrid',
                 'mpi_count_overflow', 'time_limit'):  # fmt: skip
        assert code in codes and code in ACTIONS
    assert error_signatures('convergence has been achieved in 11 iterations\nJOB DONE.') == []


# ---------------------------------------------------------------------------------------------------- cells
def _skewed_input(
    nat_line: str = 'nat = 2', mixing: str = '', kgrid: str = '4 4 4 0 0 0', nosym: str = ''
) -> str:
    """Rock-salt cell (a = 8.5 bohr) in a skewed basis (a1, a2 + 2 a1, a3 + 3 a2) of its fcc lattice."""
    fcc = 8.5 / 2 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    skewed = np.array([fcc[0], fcc[1] + 2 * fcc[0], fcc[2] + 3 * fcc[1]])
    cart = np.array([[0.0, 0.0, 0.0], [4.25, 0.0, 0.0]])
    frac = cart @ np.linalg.inv(skewed)
    frac -= np.floor(frac)
    cell = '\n'.join('  %.10f %.10f %.10f' % tuple(v) for v in skewed)
    positions = '\n'.join('  %s %.10f %.10f %.10f' % (s, *f) for s, f in zip(('Ta', 'C'), frac))
    return textwrap.dedent("""\
        &control
          calculation = 'scf'
        /
        &system
          ibrav = 0, %s, ntyp = 2, ecutwfc = 82, ecutrho = 328 %s
        /
        &electrons
          conv_thr = 1d-8 %s
        /
        CELL_PARAMETERS bohr
        %s
        ATOMIC_SPECIES
          Ta 180.95 Ta.upf
          C 12.011 C.upf
        ATOMIC_POSITIONS crystal
        %s
        K_POINTS automatic
          %s
        """) % (nat_line, nosym, mixing, cell, positions, kgrid)  # fmt: skip


def test_reduced_basis_keeps_the_lattice_and_shrinks_the_fft_box():
    from PAOFLOW.agentic.campaign.cells import read_cell, read_positions

    text = _skewed_input()
    cell = read_cell(text)
    reduced, transform = reduced_basis(cell)
    assert abs(round(np.linalg.det(transform))) == 1
    assert np.allclose(np.sort(np.linalg.norm(reduced, axis=1)), 8.5 / np.sqrt(2))
    assert fft_inflation(cell, 328.0) > 4 and fft_box(reduced, 328.0) == [
        36,
        36,
        36,
    ]  # 2 x 17 + 1 -> 36
    assert time_reversal_kpoints((4, 4, 3)) == 26 and time_reversal_kpoints((4, 3, 3)) == 19
    assert time_reversal_kpoints((3, 3, 3)) == 14
    # n x n x n grids are the same k-set in any basis
    assert shortest_superlattice_vector(cell, (4, 4, 4)) == pytest.approx(
        shortest_superlattice_vector(reduced, (4, 4, 4))
    )

    new_text, summary = reduce_cell_input(text, nosym=True, damp_mixing=True, floor_bohr=40.0)
    new_cell = read_cell(new_text)
    species, frac = read_positions(new_text)
    assert species == ['Ta', 'C'] and abs(np.linalg.det(new_cell)) == pytest.approx(
        abs(np.linalg.det(cell))
    )
    distance = (frac[1] - frac[0]) @ new_cell
    images = [np.linalg.norm(distance + np.array(n) @ new_cell) for n in np.ndindex(3, 3, 3)]
    images += [np.linalg.norm(distance - np.array(n) @ new_cell) for n in np.ndindex(3, 3, 3)]
    assert min(images) == pytest.approx(4.25)  # Ta-C nearest neighbours kept
    settings = parse_pw_input(new_text)
    assert settings.nosym and settings.mixing_mode == 'local-tf' and settings.mixing_beta == 0.3
    assert min(summary['kgrid']) >= 3 and summary['shortest_vector_bohr'] >= 40.0
    assert summary['fft_new'] == [36, 36, 36]


def test_choose_kgrid_respects_points_and_floor():
    cell = 8.5 / 2 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    choice = choose_kgrid(cell, min_points=3, floor_bohr=48.0)
    assert min(choice['kgrid']) >= 3 and choice['shortest_vector_bohr'] >= 48.0
    assert choice['kpoints'] == time_reversal_kpoints(choice['kgrid'])


# ---------------------------------------------------------------------------------------------------- preflight
def test_preflight_catches_the_campaign_mistakes():
    text = _skewed_input(nat_line='nat = 80', kgrid='8 8 2 0 0 0')
    codes = {f.code: f for f in preflight_pw_input(text, 'cell', displaced=True, interpolated=True, qgrid=(3, 3, 2),
                                                   cores=60, billing_granularity=24)}  # fmt: skip
    assert {'fft_inflation', 'two_point_axis', 'qgrid_does_not_divide_kgrid', 'billing_rounding',
            'plain_mixing_large_cell'} <= set(codes)  # fmt: skip
    assert codes['two_point_axis'].severity == 'error'
    reduced, _ = reduce_cell_input(_skewed_input(), floor_bohr=40.0)
    assert not ({'fft_inflation', 'two_point_axis', 'plain_mixing_large_cell', 'kpoint_inflation'}
                & {f.code for f in preflight_pw_input(reduced, 'cell', displaced=True, interpolated=True)})  # fmt: skip


def test_budget_findings():
    assert findings_for_budget(500, None) == []
    assert findings_for_budget(500, 1000) == []
    assert findings_for_budget(850, 1000, 'SBU')[0].code == 'budget_warning'
    assert findings_for_budget(1100, 1000, 'SBU')[0].code == 'budget_exceeded'


# ---------------------------------------------------------------------------------------------------- campaign
def _study(tmp_path) -> Campaign:
    run_dir = tmp_path / 'runs' / 'P2_T2800'
    (run_dir / 'cells').mkdir(parents=True)
    (run_dir / 'logs').mkdir()
    (run_dir / 'output').mkdir()
    (run_dir / 'cells' / 'cell.out').write_text(_read('hftac_p2_t2800_timeout.out'))
    (run_dir / 'cells' / 'cell.in').write_text(_read('hftac_p2_t2800_timeout.in'))
    (run_dir / 'cells' / 'ref.out').write_text(_read('hftac_p2_t2800_reduced_ref.out'))
    (run_dir / 'logs' / 'post.out').write_text('Traceback (most recent call last):\nMemoryError\n')
    (run_dir / 'output' / 'emissivity.dat').write_text('# done\n')
    study = tmp_path / 'study'
    study.mkdir()
    machine = local_machine('this-computer', max_cores=4)
    run = CampaignRun('P2_T2800', machine, str(run_dir), pw_outputs=('cells/*.out',), logs=('logs/*.out',),
                      results=('output/emissivity.dat',))  # fmt: skip
    return Campaign('test campaign', [run], str(study), since='2026-11-01')


def test_campaign_status_diagnosis_report_and_journal(tmp_path):
    campaign = _study(tmp_path)
    states = campaign.status()
    assert (
        len(states) == 1 and states[0].reachable and len(states[0].outputs) == 2 and states[0].done
    )
    codes = {f.code for f in campaign.diagnose(states)}
    assert {'kpoint_inflation', 'fft_inflation', 'charge_sloshing', 'python_exception', 'out_of_memory',
            'run_complete'} <= codes  # fmt: skip
    text = campaign.describe(states)
    assert 'P2_T2800' in text and 'cell.out it 25' in text and 'ref.out done (11 it)' in text
    path = campaign.journal(
        'cancelled job 1',
        evidence='projection',
        findings=[Finding('cannot_finish', 'P2_T2800', 'x')],
    )
    assert 'cannot_finish' in open(path).read()


def test_watch_reports_new_findings_once_then_the_end(tmp_path):
    campaign = _study(tmp_path)
    event, findings = watch(campaign, once=True)
    assert event == 'poll' and os.path.exists(os.path.join(campaign.study_dir, 'STATUS.md'))
    event, findings = watch(campaign, interval_seconds=0.0, stop_at=('error', 'warning'))
    assert (
        event == 'findings'
        and findings
        and all(f.severity in ('error', 'warning') for f in findings)
    )
    event, findings = watch(campaign, interval_seconds=0.0, stop_at=('error', 'warning'))
    assert event == 'done'


def test_watch_command_line(tmp_path, capsys):
    from PAOFLOW.agentic.campaign.watch import main

    campaign = _study(tmp_path)
    definition = tmp_path / 'campaign.py'
    run = campaign.runs[0]
    definition.write_text(textwrap.dedent("""
        from PAOFLOW.agentic.campaign import Campaign, CampaignRun, local_machine
        CAMPAIGN = Campaign('cli', [CampaignRun('P2_T2800', local_machine(), %r, pw_outputs=('cells/*.out',))],
                            %r, since='2026-11-01')
    """) % (run.directory, campaign.study_dir))  # fmt: skip
    assert main([str(definition), '--once']) == 0
    out = capsys.readouterr().out
    assert 'CAMPAIGN cli: POLL' in out and 'kpoint_inflation' in out
