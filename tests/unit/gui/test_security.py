"""Session token, Host/Origin checks, the path sandbox and read-only sessions."""

from __future__ import annotations

import os

import pytest

from PAOFLOW.gui import security


def test_requests_need_the_token(client):
    assert client.get('/api/session', token=False)[0] == 403
    status, _, _ = client.get('/api/session', headers={'X-PAOFLOW-Token': 'wrong'}, token=False)
    assert status == 403
    status, body, _ = client.get('/api/session')
    assert status == 200 and body['read_only'] is False


def test_token_in_the_url_becomes_a_cookie(client):
    status, _, response = client.get('/?token=%s' % client.app.token, token=False)
    assert status == 303 and response.getheader('Location') == '/'
    cookie = response.getheader('Set-Cookie')
    assert 'HttpOnly' in cookie and 'SameSite=Strict' in cookie
    value = cookie.split(';')[0]
    status, _, _ = client.get('/api/session', token=False, headers={'Cookie': value})
    assert status == 200
    assert client.get('/?token=nope', token=False)[0] == 403


def test_index_without_token_explains_how_to_open(client):
    status, body, response = client.get('/', token=False)
    assert status == 403 and b'?token=' in body


def test_foreign_host_and_origin_are_refused(client):
    assert client.get('/api/session', headers={'Host': 'evil.example:80'})[0] == 403
    status, _, _ = client.post('/api/gen/resolve', {}, headers={'Origin': 'http://evil.example'})
    assert status == 403
    origin = 'http://127.0.0.1:%d' % client.app.port
    status, _, _ = client.post(
        '/api/gen/resolve', {'workflow': 'run', 'workdir': 'Si'}, headers={'Origin': origin}
    )
    assert status == 200


def test_security_headers(client):
    _, _, response = client.get('/api/session')
    assert "default-src 'self'" in response.getheader('Content-Security-Policy')
    assert response.getheader('X-Content-Type-Options') == 'nosniff'


def test_paths_stay_inside_the_root(client, make_client, gui_root, tmp_path):
    assert client.get('/api/fs/list?path=..')[0] == 403  # relative paths stay in the root
    # The folder picker lists the computer (absolute paths) but nothing is read or written
    # there until the folder is opened (/api/fs/open); --fixed-root does not even list.
    status, listing, _ = client.get('/api/fs/list?path=/etc')
    assert status == 200 and listing['place']['outside']
    assert client.get('/api/fs/read?path=/etc/hosts')[0] == 403
    assert client.get('/api/workspace/detect?path=/etc')[0] == 403
    assert make_client(fixed_root=True).get('/api/fs/list?path=/etc')[0] == 403
    outside = tmp_path / 'outside'
    outside.mkdir()
    os.symlink(outside, gui_root / 'link')
    assert client.get('/api/fs/list?path=link')[0] == 403
    names = [e['name'] for e in client.get('/api/fs/list')[1]['entries']]
    assert 'link' not in names and 'Si' in names


def test_static_files_and_traversal(client):
    status, body, response = client.get('/static/js/app.js')
    assert status == 200 and response.getheader('Content-Type').startswith('text/javascript')
    assert client.get('/static/../server.py')[0] == 404
    assert client.get('/static/%2e%2e/server.py')[0] == 404


def test_read_only_session_refuses_writes(make_client):
    client = make_client(read_only=True)
    status, body, _ = client.post(
        '/api/gen/write', {'workflow': 'run', 'workdir': 'Si', 'answers': {'properties': ['bands']}}
    )
    assert status == 403 and 'read-only' in body['error']


def test_sandbox_unit(tmp_path):
    box = security.Sandbox(tmp_path)
    assert box.resolve(None) == tmp_path.resolve()
    assert box.resolve('a/../b') == (tmp_path / 'b').resolve()
    with pytest.raises(PermissionError):
        box.resolve('../x')
    assert security.host_allowed('localhost:8765', 8765) and not security.host_allowed(
        'localhost:1', 8765
    )
    assert security.origin_allowed(None, 8765) and not security.origin_allowed('null', 8765)


def test_wrong_token_and_wrong_port_explain_the_tunnel(client):
    """The two ways an SSH tunnel goes wrong get a page that says what to do."""
    port = client.app.port
    status, body, response = client.get('/?token=not-this-one', token=False)
    page = body.decode() if isinstance(body, bytes) else str(body)
    assert status == 403 and response.getheader('Content-Type').startswith('text/html')
    assert "not this session's token" in page and 'port %d' % port in page
    assert 'ExitOnForwardFailure' in page and str(client.app.sandbox.root) not in page
    status, body, _ = client.get('/api/session?token=not-this-one', token=False)
    assert status == 403 and body['error'] == 'wrong token'  # the API keeps its JSON answer
    status, body, response = client.get('/', headers={'Host': '127.0.0.1:%d' % (port + 1)})
    page = body.decode() if isinstance(body, bytes) else str(body)
    assert status == 403 and 'same port on both ends' in page
    assert '-L %d:127.0.0.1:%d' % (port, port) in page
    status, body, _ = client.get('/', headers={'Host': 'evil.example:%d' % port})
    assert status == 403 and body == {'error': 'unexpected Host header'}  # no page for others


def test_startup_message_gives_the_tunnel(tmp_path, monkeypatch, capsys):
    from PAOFLOW.gui import cli

    class Server:
        url = 'http://127.0.0.1:8780/?token=t'

        def __init__(self, app, port, verbose):
            app.port = port

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr('PAOFLOW.gui.server.GuiServer', Server)
    assert cli.main(['--root', str(tmp_path), '--port', '8780', '--no-browser']) == 0
    out = capsys.readouterr().out
    assert 'ssh -N -o ExitOnForwardFailure=yes -L 8780:127.0.0.1:8780' in out
    assert 'the same port on both ends' in out
