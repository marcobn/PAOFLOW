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
    assert names == [
        'acbn0',
        'run',
        'tb_model',
        'phonon',
        'elphon',
        'sscha',
        'pyskeaf',
        'qe_input',
        'basis',
    ]


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
    qe_step = body['plan'][0]
    assert qe_step['qe_launch'] and qe_step['ranks_env'] == 'NP'  # ranks, QE build, pw.x options
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
    run, setup = body['plan']
    assert (
        run['argv'] == ['pw.x', '-in', 'Si2.scf.in']
        and run['stdout'] == 'Si2.scf.out'
        and run['mpi']
    )
    assert run['command'] == 'pw.x -in Si2.scf.in > Si2.scf.out'
    assert setup['action'] == 'setup:run' and setup['manual'] and body['folder'] == 'Si'


def test_band_path_preview_endpoint(client):
    rows = [['gG', '0', '0', '0'], ['X', '0', '0.5', '0.5']]
    status, body, _ = client.post(
        '/api/gen/bandpath',
        {'workdir': 'Si', 'savedir': 'silicon.save', 'band_path': 'gG-X', 'points': rows},
    )
    assert (
        status == 200
        and body['length'] > 0
        and body['figure']['data'][1]['line']['color'] == 'series-1'
    )
    status, body, _ = client.post(
        '/api/gen/bandpath',
        {'workdir': 'Si', 'savedir': 'silicon.save', 'band_path': 'gG-X', 'points': [['gG', 'a']]},
    )
    assert 'error' in body


def test_run_plan_of_an_input_on_disk(client, gui_root):
    """An input written earlier (or by paoflow-gen-qe) gets the pw.x plan of a freshly written one."""
    (gui_root / 'Si' / 'gaas.scf.in').write_text("&control\n  calculation = 'scf'\n/\n")
    status, body, _ = client.post('/api/gen/qe-input/plan', {'path': 'Si/gaas.scf.in'})
    assert status == 200 and body['folder'] == 'Si' and body['file'] == 'Si/gaas.scf.in'
    run, setup = body['plan']
    assert run['argv'] == ['pw.x', '-in', 'gaas.scf.in'] and run['stdout'] == 'gaas.scf.out'
    assert setup['action'] == 'setup:run'
    status, body, _ = client.post('/api/gen/qe-input/plan', {'path': 'Si/silicon.save'})
    assert status == 404
    (gui_root / 'Si' / 'notes.in').write_text('not a namelist\n')
    status, body, _ = client.post('/api/gen/qe-input/plan', {'path': 'Si/notes.in'})
    assert status == 400 and 'not a pw.x input' in body['error']
    status, body, _ = client.post('/api/gen/qe-input/plan', {'path': '../outside.in'})
    assert status == 403


def test_session_asks_for_a_restart_after_a_code_change(client):
    status, body, _ = client.get('/api/session')
    assert status == 200 and body['restart_needed'] is False
    client.app.started = 0.0  # every module is newer than the start of this server
    status, body, _ = client.get('/api/session')
    assert body['restart_needed'] is True


def test_save_directory_without_a_save_is_flagged(client):
    """A Save directory that holds no save is warned about (not an error) with the found saves."""
    payload = {'workflow': 'run', 'workdir': 'Si', 'answers': {'properties': ['bands']}}
    items = {
        i['key']: i for i in client.post('/api/gen/resolve', payload)[1]['items'] if 'key' in i
    }
    assert items['savedir']['value'] == 'silicon.save' and 'warning' not in items['savedir']
    assert items['savedir']['widget'] == 'folder'
    payload['answers']['savedir'] = 'gaas.save'
    status, body, _ = client.post('/api/gen/resolve', payload)
    items = {i['key']: i for i in body['items'] if 'key' in i}
    assert status == 200 and 'gaas.save does not exist' in items['savedir']['warning']
    assert items['savedir']['suggestions'] == ['silicon.save']


def test_edited_files_are_previewed_and_written(client, gui_root):
    """Edits of the previewed files replace the generated text (status, diff and write)."""
    payload = {'workflow': 'run', 'workdir': 'Si', 'answers': {'properties': ['bands']}}
    status, body, _ = client.post('/api/gen/resolve', dict(payload, edits={'main.py': 'x = 1\n'}))
    main = next(f for f in body['files'] if f['path'] == 'main.py')
    assert main['edited'] and main['text'] == 'x = 1\n' and 'PAOFLOW' in main['generated']
    assert not next(f for f in body['files'] if f['path'] == 'plot.py')['edited']
    edits = {'main.py': 'x = 1\n', 'gone.py': 'y = 2\n'}
    status, body, _ = client.post('/api/gen/write', dict(payload, edits=edits))
    assert status == 200 and (gui_root / 'Si' / 'main.py').read_text() == 'x = 1\n'
    assert not (gui_root / 'Si' / 'gone.py').exists()  # never outside the files of the form
    status, body, _ = client.post('/api/gen/resolve', dict(payload, edits=edits))
    assert body['unused_edits'] == ['gone.py']
    assert next(f for f in body['files'] if f['path'] == 'main.py')['status'] == 'identical'
    status, body, _ = client.post('/api/gen/resolve', dict(payload, edits=['main.py']))
    assert status == 400


def test_one_plot_switch_for_every_workflow(client, gui_root):
    """The Output switch (options.plot) decides the plot scripts of every workflow; without it,
    the defaults of paoflow-gen (PLOT_DEFAULTS); the form shows no second plot question."""
    from PAOFLOW.gen.outputs import PLOT_DEFAULTS

    answers = {
        'run': {'properties': ['bands']},
        'acbn0': {'hubbard': [['Si-3p', '0.5']]},
        'phonon': {'raman_method': 'static'},
        'sscha': {},
        'pyskeaf': {},
        'elphon': {'workflows': ['transport']},
    }
    workflows = {w['name']: w for w in client.get('/api/gen/workflows')[1]['workflows']}
    for name, given in answers.items():
        assert workflows[name]['plot_default'] == PLOT_DEFAULTS[name]
        payload = {'workflow': name, 'workdir': 'Si', 'answers': given}
        if name == 'elphon':
            payload['options'] = {'folder': 'Si/system'}
        results = {}
        for plot in (None, True, False):
            body = dict(payload)
            if plot is not None:
                body['options'] = dict(payload.get('options', {}), plot=plot)
            resolved = client.post('/api/gen/resolve', body)[1]
            assert resolved['complete'], (name, resolved['errors'])
            plots = [f['path'] for f in resolved['files'] if 'plot' in f['path'].split('/')[-1]]
            results[plot] = plots
            shown = [i for i in resolved['items'] if i.get('key') == 'plot' and i['visible']]
            assert not shown, name  # no plot question in the form
        assert results[True] and not results[False], (name, results)
        assert bool(results[None]) == PLOT_DEFAULTS[name], (name, results)
        figures = [s for s in resolved['plan'] if s['label'].startswith('Figures')]
        assert not figures, (name, figures)  # no step for a script that is not written


def test_the_icon_of_paoflow_org(client):
    from pathlib import Path

    icon = (
        Path(__file__).resolve().parents[3]
        / 'docs'
        / '_static'
        / 'images'
        / 'logos'
        / 'favicon.ico'
    )
    for path in ('/favicon.ico', '/static/favicon.ico'):
        status, body, response = client.get(path)
        assert status == 200 and body == icon.read_bytes()
        assert 'icon' in response.getheader('Content-Type')


def test_tb_model_run_needs_no_save(client, gui_root):
    """A property run on a tight-binding model: suggested where there is no QE data, its band
    path drawn in the zone of the model's lattice, and its files written without a save."""
    (gui_root / 'model').mkdir()
    detection = client.get('/api/workspace/detect?path=model')[1]
    assert 'tb_model' in [s['workflow'] for s in detection['suggestions']]
    answers = {'tb_model': 'kane_mele', 'properties': ['bands', 'nonlinear']}
    status, preview, _ = client.post(
        '/api/gen/bandpath',
        {
            'workdir': 'model',
            'workflow': 'tb_model',
            'answers': answers,
            'band_path': 'gG-M-K-gG',
            'points': [['gG', 0, 0, 0], ['M', 0.5, 0, 0], ['K', 1 / 3, 1 / 3, 0]],
        },
    )
    assert status == 200 and 'figure' in preview and preview['length'] > 0
    status, body, _ = client.post(
        '/api/gen/resolve', {'workflow': 'tb_model', 'workdir': 'model', 'answers': answers}
    )
    assert status == 200 and not body['errors']
    assert [f['path'] for f in body['files']] == ['main.py', 'plot.py']
    status, written, _ = client.post(
        '/api/gen/write', {'workflow': 'tb_model', 'workdir': 'model', 'answers': answers}
    )
    assert status == 200 and (gui_root / 'model' / 'main.py').is_file()
    assert 'kane_mele_model(**MODEL_PARAMETERS)' in (gui_root / 'model' / 'main.py').read_text()
    assert written['plan'][0]['argv'] == ['python', 'main.py']
