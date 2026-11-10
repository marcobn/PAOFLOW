"""Read-only folders (--allow-read, PAOFLOW's PSEUDOS and BASIS): readable, never written."""

from __future__ import annotations

import os
import shutil
import types

import pytest
from conftest import EXAMPLE

from PAOFLOW.gui import security
from PAOFLOW.gui.server import paoflow_data_folders


def test_sandbox_read_roots(tmp_path):
    root = tmp_path / 'root'
    pseudos = tmp_path / 'pseudos'
    secret = tmp_path / 'secret'
    for folder in (root, pseudos / 'sub', secret):
        folder.mkdir(parents=True)
    box = security.Sandbox(root, read_roots=[pseudos, root / 'inside', tmp_path / 'missing'])
    assert box.read_roots == [pseudos.resolve()]
    assert box.resolve_readable(str(pseudos / 'sub')) == (pseudos / 'sub').resolve()
    with pytest.raises(PermissionError, match='read-only'):
        box.resolve(str(pseudos / 'sub'))
    with pytest.raises(PermissionError, match='--allow-read'):
        box.resolve_readable(str(secret))
    os.symlink(secret, pseudos / 'escape')
    assert not box.readable(pseudos / 'escape')
    assert box.relative(pseudos / 'sub') == str((pseudos / 'sub').resolve())
    assert box.relative(root) == '.'
    assert box.place(pseudos / 'sub') == pseudos.resolve()


def test_listing_and_session(make_client, tmp_path):
    pseudos = tmp_path / 'pseudos'
    (pseudos / 'nc').mkdir(parents=True)
    (pseudos / 'Si.upf').write_text('<UPF version="2.0.1"></UPF>\n')
    client = make_client(read_roots=[pseudos])
    status, session, _ = client.get('/api/session')
    assert str(pseudos.resolve()) in session['read_roots']
    status, listing, _ = client.get('/api/fs/list?path=%s' % pseudos)
    assert status == 200 and listing['place']['writable'] is False
    assert listing['parent'] == str(tmp_path.resolve())  # the picker may go up (browse anywhere)
    assert [e['name'] for e in listing['entries']] == ['nc', 'Si.upf'] and listing['entries'][1][
        'tags'
    ] == ['upf']
    status, listing, _ = client.get('/api/fs/list?path=%s' % (pseudos / 'nc'))
    assert listing['parent'] == str(pseudos.resolve())
    status, root_listing, _ = client.get('/api/fs/list')
    assert root_listing['place'] == {
        'path': '.',
        'label': 'root',
        'writable': True,
        'outside': False,
    }
    status, elsewhere, _ = client.get('/api/fs/list?path=%s' % tmp_path)
    assert status == 200 and elsewhere['place']['outside']  # listed, not opened


def test_paoflow_pseudos_are_readable_by_default(client):
    folders = paoflow_data_folders()
    if not folders:
        pytest.skip('no PSEUDOS/BASIS next to this PAOFLOW')
    status, session, _ = client.get('/api/session')
    assert set(map(str, folders)) <= set(session['read_roots'])
    assert client.get('/api/fs/list?path=%s' % folders[0])[0] == 200


def test_qe_input_reads_a_read_only_pseudo_folder(make_client, tmp_path, monkeypatch):
    from PAOFLOW.gen.qe_input import cli

    pseudos = tmp_path / 'pseudos'
    pseudos.mkdir()
    seen = {}

    def fake(**kw):
        seen.update(kw)
        record = types.SimpleNamespace(compound='Si2', natoms=2, species=['Si'], metallic=False, magnetic=False,
                                       dimensionality='3D', kpoints=None, source='aflow', spacegroup=None)  # fmt: skip
        return record, '&control\n/\n'

    monkeypatch.setattr(cli, 'generate_qe_input', fake)
    client = make_client(read_roots=[pseudos])
    payload = {'identifier': 'aflow:x', 'workdir': 'Si', 'pseudo_dir': str(pseudos)}
    status, body, _ = client.post('/api/gen/qe-input/preview', payload)
    assert status == 200 and seen['pseudo_dir'] == str(pseudos.resolve())
    other = tmp_path / 'other'
    other.mkdir()
    status, body, _ = client.post('/api/gen/qe-input/preview', dict(payload, pseudo_dir=str(other)))
    assert status == 403 and '--allow-read' in body['error']
    status, body, _ = client.post('/api/gen/qe-input/write', dict(payload, workdir=str(pseudos)))
    assert status == 403


def test_basis_command_and_results_from_read_roots(make_client, tmp_path):
    shared = tmp_path / 'shared'
    shutil.copytree(EXAMPLE / 'Reference', shared / 'output')
    client = make_client(read_roots=[shared])
    status, body, _ = client.post(
        '/api/gen/basis/command', {'workdir': 'Si', 'pseudo_dir': str(shared), 'out': 'BASIS_PS'}
    )
    assert status == 200 and body['argv'][2] == str(shared)
    status, body, _ = client.post(
        '/api/gen/basis/command', {'workdir': 'Si', 'pseudo_dir': '.', 'out': str(shared / 'b')}
    )
    assert status == 403
    status, scan, _ = client.post('/api/results/scan', {'path': str(shared)})
    assert status == 200 and scan['folder'] == str(shared.resolve()) and scan['datasets']
    assert not scan['writable']  # read-only: Setup and Runs do not follow it
    bands = next(d for d in scan['datasets'] if d['kind'] == 'bands')
    status, figure, _ = client.post('/api/results/figure', {'path': str(shared), 'id': bands['id']})
    assert status == 200 and figure['figure']['data']


def test_path_answers_must_be_readable(make_client, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'scf.in').write_text("&control\n  prefix = 'x'\n/\n")
    client = make_client()
    payload = {
        'workflow': 'elphon',
        'workdir': 'Si',
        'answers': {'pw_template': str(outside / 'scf.in')},
    }
    status, body, _ = client.post('/api/gen/resolve', payload)
    assert status == 403
    client = make_client(read_roots=[outside])
    status, body, _ = client.post('/api/gen/resolve', payload)
    assert status == 200
    status, body, _ = client.post(
        '/api/gen/resolve',
        {'workflow': 'run', 'workdir': 'Si', 'answers': {'savedir': '../../../../etc'}},
    )
    assert status == 403


def test_sandbox_opens_folders_for_writing_or_reading(tmp_path):
    from PAOFLOW.gui.security import Sandbox

    root, work, data = tmp_path / 'root', tmp_path / 'work', tmp_path / 'data'
    for folder in (root, work, data):
        folder.mkdir()
    sandbox = Sandbox(root)
    with pytest.raises(PermissionError, match='open it'):
        sandbox.resolve(str(work))
    sandbox.open(work, writable=True)
    sandbox.open(data, writable=False)
    assert sandbox.resolve(str(work / 'new')) == (work / 'new').resolve()
    assert sandbox.resolve_readable(str(data)) == data.resolve()
    with pytest.raises(PermissionError, match='read-only'):
        sandbox.resolve(str(data / 'x'))
    assert (
        sandbox.writable(work)
        and not sandbox.writable(data)
        and sandbox.place(work / 'a') == work.resolve()
    )
    assert sandbox.relative(work) == str(work.resolve())  # absolute outside the root
    with pytest.raises(NotADirectoryError):
        sandbox.open(tmp_path / 'missing')


def test_picker_browses_the_computer_and_opens_folders(tmp_path):
    from PAOFLOW.gui.server import GuiApp, start_in_thread

    from conftest import Client

    root, work = tmp_path / 'root', tmp_path / 'elsewhere' / 'Si'
    locked = tmp_path / 'elsewhere' / 'locked'
    root.mkdir()
    work.mkdir(parents=True)
    (work / 'scf.in').write_text('&control\n/\n')
    servers = []

    def client(**options):
        app = GuiApp(root, default_read_roots=False, **options)
        server, _ = start_in_thread(app)
        servers.append(server)
        return Client(app)

    try:
        free = client()
        status, listing, _ = free.get('/api/fs/list?path=%s' % (tmp_path / 'elsewhere'))
        assert (
            status == 200
            and listing['place']['outside']
            and listing['parent'] == str(tmp_path.resolve())
        )
        assert [e['name'] for e in listing['entries']] == ['Si']
        assert free.get('/api/workspace/detect?path=%s' % work)[0] == 403  # listed, not opened
        status, opened, _ = free.post('/api/fs/open', {'path': str(work), 'writable': True})
        assert (
            status == 200
            and opened['writable']
            and opened['opened'][0]['path'] == str(work.resolve())
        )
        status, detection, _ = free.get('/api/workspace/detect?path=%s' % work)
        assert status == 200 and detection['pw_inputs'] == ['scf.in']
        session = free.get('/api/session')[1]
        assert session['home'] and not session['fixed_root'] and session['opened'][0]['writable']
        # remembered: a restarted server has the folder again
        again = client()
        assert again.get('/api/workspace/detect?path=%s' % work)[0] == 200
        # --read-only: an opened folder is read only; --fixed-root: no browsing, no opening
        reader = client(read_only=True)
        other = tmp_path / 'elsewhere'
        assert not reader.post('/api/fs/open', {'path': str(other), 'writable': True})[1][
            'writable'
        ]
        fixed = client(fixed_root=True)
        assert fixed.get('/api/fs/list?path=%s' % (tmp_path / 'elsewhere'))[0] == 403
        status, body, _ = fixed.post('/api/fs/open', {'path': str(work)})
        assert status == 403 and '--fixed-root' in body['error']
        assert fixed.get('/api/fs/list?path=.')[1]['parent'] is None  # no way above the root
        # an entry that cannot be inspected (system folders) does not fail the listing
        locked.mkdir()
        (locked / 'inner').mkdir()
        locked.chmod(0)
        status, listing, _ = free.get('/api/fs/list?path=%s' % (tmp_path / 'elsewhere'))
        assert status == 200 and 'Si' in [e['name'] for e in listing['entries']]
    finally:
        if locked.exists():
            locked.chmod(0o755)
        for server in servers:
            server.shutdown()
            server.server_close()
