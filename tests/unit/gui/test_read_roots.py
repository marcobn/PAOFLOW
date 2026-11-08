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
    assert status == 200 and listing['parent'] is None and listing['place']['writable'] is False
    assert [e['name'] for e in listing['entries']] == ['nc', 'Si.upf'] and listing['entries'][1][
        'tags'
    ] == ['upf']
    status, listing, _ = client.get('/api/fs/list?path=%s' % (pseudos / 'nc'))
    assert listing['parent'] == str(pseudos.resolve())
    status, root_listing, _ = client.get('/api/fs/list')
    assert root_listing['place'] == {'path': '.', 'label': 'root', 'writable': True}
    assert client.get('/api/fs/list?path=%s' % tmp_path)[0] == 403


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
