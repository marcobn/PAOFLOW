"""Generator endpoints: forms, previews identical to the CLI, writing with confirmation."""

from __future__ import annotations

import types

from PAOFLOW.gen import paoflow_driver as pd
from PAOFLOW.gen.outputs import workflow_outputs
from PAOFLOW.gen.questions import collect


def _resolve(client, workflow, answers=None, options=None, workdir='Si'):
    status, body, _ = client.post(
        '/api/gen/resolve',
        {
            'workflow': workflow,
            'workdir': workdir,
            'answers': answers or {},
            'options': options or {},
        },
    )
    assert status == 200, body
    return body


def test_workflows_listed(client):
    names = [w['name'] for w in client.get('/api/gen/workflows')[1]['workflows']]
    assert names == ['acbn0', 'run', 'phonon', 'elphon', 'sscha', 'qe_input', 'basis']


def test_detection_of_the_example(client):
    status, body, _ = client.get('/api/workspace/detect?path=Si')
    assert status == 200 and body['saves'] == ['silicon.save'] and body['pw_inputs'] == ['scf.in']
    assert body['save_summary']['formula'] == 'Si2' and not body['save_summary']['metal']
    assert body['suggestions'][0]['workflow'] == 'run'


def test_run_form_needs_properties(client):
    body = _resolve(client, 'run')
    assert not body['complete'] and 'properties' in body['errors']
    items = {i['key']: i for i in body['items'] if i['type'] == 'question'}
    assert items['savedir']['value'] == 'silicon.save' and items['savedir']['is_default']
    assert items['nk']['visible'] is False  # no band structure selected yet


def test_gui_preview_is_the_cli_output(client, gui_root, monkeypatch):
    answers = {'properties': ['bands', 'dos', 'transport'], 'npool': '2', 'emax': '3.5'}
    body = _resolve(client, 'run', answers)
    assert body['complete']
    gui = {f['path']: f['text'] for f in body['files']}
    replies = {'Enter a comma': '1 2 3', 'npool': '2', '  EMAX': '3.5'}

    def answer(prompt):
        return next((value for key, value in replies.items() if prompt.startswith(key)), '')

    monkeypatch.setattr(pd, '_input', answer)
    args = types.SimpleNamespace(savedir=None, prefix=None, upf=None, out=None)
    cfg = collect('run', args, str(gui_root / 'Si'))
    cli = {f.path: f.content() for f in workflow_outputs('run', cfg)}
    assert cli.keys() == gui.keys()
    assert cli == gui


def test_write_conflict_backup_and_keep(client, gui_root):
    payload = {'workflow': 'run', 'workdir': 'Si', 'answers': {'properties': ['bands']}}
    status, body, _ = client.post('/api/gen/write', payload)
    assert status == 200 and body['written'] == ['Si/main.py', 'Si/plot.py']
    changed = dict(payload, answers={'properties': ['bands'], 'npool': '4'})
    status, body, _ = client.post('/api/gen/write', changed)
    assert status == 409 and body['conflicts'] == ['main.py']
    status, body, _ = client.post('/api/gen/write', dict(changed, keep=['main.py']))
    assert status == 200 and body['kept'] == ['Si/main.py'] and body['unchanged'] == ['Si/plot.py']
    assert 'NPOOL = 1' in (gui_root / 'Si' / 'main.py').read_text()
    status, body, _ = client.post('/api/gen/write', dict(changed, overwrite=['main.py']))
    assert status == 200 and body['written'] == ['Si/main.py'] and len(body['backups']) == 1
    assert 'NPOOL = 4' in (gui_root / 'Si' / 'main.py').read_text()
    assert 'NPOOL = 1' in (gui_root / body['backups'][0]).read_text()


def test_elphon_system_folder_and_plan(client, gui_root):
    body = _resolve(client, 'elphon', {'workflows': ['arpes']}, {'folder': 'Si/system'})
    assert body['complete'] and body['target'].endswith('Si/system')
    paths = [f['path'] for f in body['files']]
    assert paths[0] == 'system.py' and 'arpes/arpes.py' in paths
    commands = [step['command'] for step in body['plan']]
    assert (
        commands[0] == '(cd qe && bash run_qe.sh)' and 'python arpes/arpes.py --no-plot' in commands
    )
    status, written, _ = client.post(
        '/api/gen/write',
        {
            'workflow': 'elphon',
            'workdir': 'Si',
            'answers': {'workflows': ['arpes']},
            'options': {'folder': 'Si/system'},
        },
    )
    assert status == 200 and (gui_root / 'Si' / 'system' / 'cache').is_dir()
    assert (gui_root / 'Si' / 'system' / 'qe' / 'run_qe.sh').stat().st_mode & 0o111


def test_gui_only_settings_change_the_file_only_when_set(client):
    base = {'launcher': 'manual'}
    default = {f['path']: f['text'] for f in _resolve(client, 'sscha', base)['files']}
    same = {
        f['path']: f['text']
        for f in _resolve(client, 'sscha', dict(base, min_step_dyn='0.05'))['files']
    }
    changed = _resolve(client, 'sscha', dict(base, min_step_dyn='0.1'))['files'][0]['text']
    assert same == default and 'MIN_STEP_DYN = 0.1' in changed


def test_validation_messages_reach_the_form(client):
    body = _resolve(client, 'elphon', {'kgrid_axis': '6', 'qgrid_axis': '4'})
    assert body['errors']['qgrid_axis'] == 'The q-grid must divide the k-grid (6).'
    body = _resolve(client, 'run', {'properties': ['bands'], 'npool': 'two'})
    assert body['errors']['npool'].startswith('not an integer')


def test_basis_command(client):
    status, body, _ = client.post(
        '/api/gen/basis/command',
        {'workdir': 'Si', 'pseudo_dir': '.', 'out': 'BASIS_PS', 'preset': 'standard'},
    )
    assert status == 200 and body['argv'] == [
        'paoflow-genbasis-ps',
        '--pseudo-dir',
        '.',
        '--out',
        'BASIS_PS',
        '--preset',
        'standard',
    ]
    assert client.post('/api/gen/basis/command', {'workdir': 'Si', 'out': '../../x'})[0] == 403


def test_qe_input_preview_and_write(client, gui_root, monkeypatch):
    from PAOFLOW.gen.qe_input import cli

    record = types.SimpleNamespace(
        compound='Si2',
        natoms=2,
        species=['Si'],
        metallic=False,
        magnetic=False,
        dimensionality='3D',
        kpoints=(8, 8, 8),
        source='aflow',
        spacegroup=227,
    )
    monkeypatch.setattr(cli, 'generate_qe_input', lambda **kw: (record, '&control\n/\n'))
    payload = {'identifier': 'aflow:abc', 'pseudo_dir': '.', 'workdir': 'Si'}
    status, body, _ = client.post('/api/gen/qe-input/preview', payload)
    assert status == 200 and body['filename'] == 'Si2.scf.in' and body['status'] == 'new'
    status, body, _ = client.post('/api/gen/qe-input/write', payload)
    assert status == 200 and (gui_root / 'Si' / 'Si2.scf.in').read_text() == '&control\n/\n'
