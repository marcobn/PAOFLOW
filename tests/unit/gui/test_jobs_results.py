"""Run manager (whitelist, ranks, logs, cancel) and results endpoints."""

from __future__ import annotations

import os
import shutil
import time

import pytest

from conftest import EXAMPLE


def _wait(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body, _ = client.get('/api/jobs/log?id=%s' % job_id)
        if body['status'] != 'running':
            return body
        time.sleep(0.2)
    raise AssertionError('job still running')


def test_python_driver_runs_and_logs(client, gui_root):
    (gui_root / 'Si' / 'hello.py').write_text(
        "import os\nprint('hello', os.environ['OMP_NUM_THREADS'])\n"
    )
    step = {'label': 'hello', 'argv': ['python', 'hello.py'], 'cwd': 'Si', 'mpi': False}
    status, job, _ = client.post('/api/jobs/launch', {'step': step, 'ranks': 1})
    assert status == 200 and job['status'] == 'running'
    log = _wait(client, job['id'])
    assert log['status'] == 'finished' and 'hello 1' in log['text']
    status, listing, _ = client.get('/api/jobs')
    assert listing['jobs'][0]['id'] == job['id'] and listing['jobs'][0]['returncode'] == 0


def test_failed_and_cancelled_jobs(client, gui_root):
    (gui_root / 'Si' / 'fail.py').write_text('raise SystemExit(3)\n')
    (gui_root / 'Si' / 'sleep.py').write_text('import time\ntime.sleep(60)\n')
    _, job, _ = client.post(
        '/api/jobs/launch', {'step': {'argv': ['python', 'fail.py'], 'cwd': 'Si'}}
    )
    assert _wait(client, job['id'])['status'] == 'failed'
    _, job, _ = client.post(
        '/api/jobs/launch', {'step': {'argv': ['python', 'sleep.py'], 'cwd': 'Si'}}
    )
    status, cancelled, _ = client.post('/api/jobs/cancel', {'id': job['id']})
    assert status == 200 and cancelled['status'] == 'cancelled'
    assert _wait(client, job['id'])['status'] == 'cancelled'


@pytest.mark.parametrize(
    'argv, cwd',
    [
        (['rm', '-rf', '/'], 'Si'),
        (['python', '-c', 'print(1)'], 'Si'),
        (['python', 'missing.py'], 'Si'),
        (['python', '../../x.py'], 'Si'),
        (['bash', 'evil.sh;ls'], 'Si'),
        (['python', 'hello.py', '$(id)'], 'Si'),
        (['python', 'hello.py'], '../..'),
    ],
)
def test_whitelist_refuses_other_commands(client, gui_root, argv, cwd):
    (gui_root / 'Si' / 'hello.py').write_text('print(1)\n')
    status, body, _ = client.post('/api/jobs/launch', {'step': {'argv': argv, 'cwd': cwd}})
    assert status in (400, 403), body


def test_rank_cap_and_interactive_steps(make_client, gui_root):
    client = make_client(max_ranks=2)
    (gui_root / 'Si' / 'hello.py').write_text('print(1)\n')
    step = {'argv': ['python', 'hello.py'], 'cwd': 'Si', 'mpi': True}
    status, body, _ = client.post('/api/jobs/launch', {'step': step, 'ranks': 3})
    assert status == 400 and 'between 1 and 2' in body['error']
    status, body, _ = client.post(
        '/api/jobs/launch', {'step': dict(step, interactive=True), 'ranks': 1}
    )
    assert status == 400


def test_read_only_session_refuses_runs(make_client, gui_root):
    client = make_client(read_only=True)
    (gui_root / 'Si' / 'hello.py').write_text('print(1)\n')
    status, _, _ = client.post(
        '/api/jobs/launch', {'step': {'argv': ['python', 'hello.py'], 'cwd': 'Si'}}
    )
    assert status == 403


def test_results_scan_figure_compare_and_export(client, gui_root):
    shutil.copytree(EXAMPLE / 'Reference', gui_root / 'Si' / 'output')
    shutil.copytree(EXAMPLE / 'Reference', gui_root / 'Si' / 'output_copy')
    status, body, _ = client.post('/api/results/scan', {'path': 'Si'})
    assert status == 200
    bands = [d for d in body['datasets'] if d['kind'] == 'bands']
    assert len(bands) == 2
    status, figure, _ = client.post('/api/results/figure', {'path': 'Si', 'id': bands[0]['id']})
    assert status == 200 and figure['figure']['data'][0]['line']['color'] == 'series-1'
    assert [t for t in figure['figure']['layout']['xaxis']['ticktext']][0] == 'Γ'
    window = {'window': [-5, 5]}
    status, figure, _ = client.post(
        '/api/results/figure', {'path': 'Si', 'id': bands[0]['id'], 'options': window}
    )
    assert figure['figure']['layout']['yaxis']['range'] == [-5, 5]
    items = [{'path': 'Si', 'id': b['id'], 'label': b['folder']} for b in bands]
    status, compared, _ = client.post('/api/results/compare', {'items': items})
    assert status == 200 and [t['line']['color'] for t in compared['figure']['data']] == [
        'series-1',
        'series-2',
    ]
    status, csv, _ = client.post(
        '/api/results/export', {'path': 'Si', 'id': bands[0]['id'], 'format': 'csv'}
    )
    assert status == 200 and csv['text'].startswith('# Band structure')
    status, pdf, response = client.request(
        'POST', '/api/results/export', {'path': 'Si', 'id': bands[0]['id'], 'format': 'pdf'}
    )
    assert (
        status == 200
        and response.getheader('Content-Type') == 'application/pdf'
        and pdf[:4] == b'%PDF'
    )
    assert client.post('/api/results/scan', {'path': '../'})[0] == 403


def test_results_experiment_overlay(client, gui_root):
    shutil.copytree(EXAMPLE / 'Reference', gui_root / 'Si' / 'output')
    (gui_root / 'Si' / 'exp.csv').write_text('# E, dos\n-2,0.1\n-1,0.2\n0,0.0\n')
    body = client.post('/api/results/scan', {'path': 'Si'})[1]
    dos = next(d for d in body['datasets'] if d['kind'] == 'dos')
    payload = {
        'path': 'Si',
        'id': dos['id'],
        'experiment': {'file': 'Si/exp.csv', 'x_column': 0, 'y_column': 1},
    }
    status, figure, _ = client.post('/api/results/figure', payload)
    assert status == 200 and figure['figure']['data'][-1]['mode'] == 'markers'
    assert figure['metrics']['points'] == 3


def _fake_qe(tmp_path, monkeypatch, name='pw.x'):
    bin_dir = tmp_path / 'qe_bin'
    bin_dir.mkdir(exist_ok=True)
    program = bin_dir / name
    program.write_text('#!/bin/sh\necho "Program PWSCF (stand-in) $@"\necho "JOB DONE."\n')
    program.chmod(0o755)
    monkeypatch.setenv('PATH', '%s:%s' % (bin_dir, os.environ['PATH']))
    return program


def test_qe_program_writes_its_output_file(client, gui_root, tmp_path, monkeypatch):
    _fake_qe(tmp_path, monkeypatch)
    (gui_root / 'Si' / 'Si2.scf.in').write_text('&control\n/\n')
    step = {
        'label': 'scf',
        'argv': ['pw.x', '-in', 'Si2.scf.in', '-nk', '2'],
        'cwd': 'Si',
        'stdout': 'Si2.scf.out',
    }
    status, job, _ = client.post('/api/jobs/launch', {'step': step})
    assert status == 200, job
    log = _wait(client, job['id'])
    assert log['status'] == 'finished' and 'JOB DONE.' in log['text']
    output = (gui_root / 'Si' / 'Si2.scf.out').read_text()
    assert output.startswith('Program PWSCF (stand-in) -in Si2.scf.in -nk 2')
    assert job['log'] == 'Si/Si2.scf.out'


@pytest.mark.parametrize(
    'argv, stdout, message',
    [
        (['pw.x', '-in', 'missing.in'], None, 'not found'),
        (['pw.x', '-in', 'Si2.scf.in', '-outdir', '/tmp'], None, 'unsupported QE option'),
        (['pw.x', '-nk', 'x', '-in', 'Si2.scf.in'], None, 'integer'),
        (['pw.x', '-nk', '2'], None, 'input file'),
        (['pw.x', '-in', 'Si2.scf.in'], '../escape.out', 'output file'),
    ],
)
def test_qe_arguments_are_checked(client, gui_root, tmp_path, monkeypatch, argv, stdout, message):
    _fake_qe(tmp_path, monkeypatch)
    (gui_root / 'Si' / 'Si2.scf.in').write_text('&control\n/\n')
    step = {'argv': argv, 'cwd': 'Si', 'stdout': stdout}
    status, body, _ = client.post('/api/jobs/launch', {'step': step})
    assert status == 400 and message in body['error']


def test_missing_qe_program_is_explained(make_client, gui_root, monkeypatch):
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    client = make_client(qe_search=False)  # never a real QE of the machine
    (gui_root / 'Si' / 'Si2.scf.in').write_text('&control\n/\n')
    status, body, _ = client.post(
        '/api/jobs/launch', {'step': {'argv': ['ph.x', '-in', 'Si2.scf.in'], 'cwd': 'Si'}}
    )
    assert status == 400 and 'no Quantum ESPRESSO installation found' in body['error']
    assert '--qe-bin' in body['error']


def _qe_build(folder, version, mpi=True, programs=('pw.x',)):
    """A stand-in QE build whose programs print their version and arguments."""
    (folder / 'bin').mkdir(parents=True)
    (folder / 'include').mkdir()
    (folder / 'include' / 'qe_version.h').write_text("version_number = '%s'\n" % version)
    (folder / 'make.inc').write_text('DFLAGS = -D__FFTW%s\n' % (' -D__MPI' if mpi else ''))
    for name in programs:
        program = folder / 'bin' / name
        program.write_text(
            '#!/bin/sh\necho "Program %s v.%s $@"\necho "JOB DONE."\n' % (name, version)
        )
        program.chmod(0o755)
    return folder / 'bin'


def test_runs_use_the_chosen_qe_build(make_client, gui_root, tmp_path, monkeypatch):
    builds = tmp_path / 'Programs'
    default = _qe_build(builds / 'qe-7.4.1', '7.4.1')
    newer = _qe_build(builds / 'qe-7.5', '7.5')
    serial = _qe_build(builds / 'qe-6.8', '6.8', mpi=False)
    monkeypatch.setenv('PATH', '%s:/usr/bin:/bin' % default)
    client = make_client(qe_search=False, max_ranks=4)
    status, body, _ = client.get('/api/qe')
    assert status == 200 and body['default'] == str(default)
    assert [q['version'] for q in body['installations']] == ['7.5', '7.4.1', '6.8']
    assert client.get('/api/session')[1]['features']['pw'] is True
    (gui_root / 'Si' / 'Si2.scf.in').write_text('&control\n/\n')
    step = {'argv': ['pw.x', '-in', 'Si2.scf.in'], 'cwd': 'Si', 'stdout': 'Si2.scf.out'}
    for qe_bin, version in ((None, '7.4.1'), (str(newer), '7.5')):
        status, job, _ = client.post('/api/jobs/launch', {'step': step, 'qe_bin': qe_bin})
        assert status == 200, job
        assert _wait(client, job['id'])['status'] == 'finished'
        assert ('v.%s -in Si2.scf.in' % version) in (gui_root / 'Si' / 'Si2.scf.out').read_text()
    # The chosen build goes first on the PATH of any run (scripts and drivers that call QE).
    (gui_root / 'Si' / 'main.py').write_text(
        "import os\nprint('PATH0', os.environ['PATH'].split(os.pathsep)[0])\n"
    )
    status, job, _ = client.post(
        '/api/jobs/launch',
        {'step': {'argv': ['python', 'main.py'], 'cwd': 'Si'}, 'qe_bin': str(newer)},
    )
    assert status == 200 and 'PATH0 %s' % newer in _wait(client, job['id'])['text']
    errors = [
        ({'step': step, 'qe_bin': str(tmp_path)}, 'not one of the Quantum ESPRESSO installations'),
        ({'step': dict(step, mpi=True), 'ranks': 2, 'qe_bin': str(serial)}, 'serial build'),
        (
            {'step': dict(step, argv=['ph.x', '-in', 'Si2.scf.in']), 'qe_bin': str(newer)},
            'ph.x is not in',
        ),
    ]
    for payload, message in errors:
        status, body, _ = client.post('/api/jobs/launch', payload)
        assert status == 400 and message in body['error'], body
