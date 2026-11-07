"""A GUI server on an ephemeral port over a temporary root, and a small HTTP client."""

from __future__ import annotations

import http.client
import json
import shutil
from pathlib import Path

import pytest

from PAOFLOW.gui.server import GuiApp, start_in_thread

EXAMPLE = Path(__file__).resolve().parents[3] / 'examples' / 'qe_examples' / 'example01'


class Client:
    """Requests with the session token header (or none), the Host and Origin the browser sends."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def request(self, method, path, body=None, token=True, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.app.port, timeout=30)
        sent = {'Host': '127.0.0.1:%d' % self.app.port}
        if token:
            sent['X-PAOFLOW-Token'] = self.app.token
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            sent['Content-Type'] = 'application/json'
        sent.update(headers or {})
        connection.request(method, path, body=data, headers=sent)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        try:
            content = json.loads(payload) if payload else None
        except ValueError:
            content = payload
        return response.status, content, response

    def get(self, path, **kw):
        return self.request('GET', path, **kw)

    def post(self, path, body=None, **kw):
        return self.request('POST', path, body=body or {}, **kw)


@pytest.fixture()
def gui_root(tmp_path):
    root = tmp_path / 'root'
    work = root / 'Si'
    work.mkdir(parents=True)
    shutil.copytree(EXAMPLE / 'silicon.save', work / 'silicon.save')
    shutil.copy(EXAMPLE / 'scf.in', work / 'scf.in')
    shutil.copy(EXAMPLE / 'Si.pbe-tm-new-gipaw-v2.1.UPF', work / 'Si.pbe-tm-new-gipaw-v2.1.UPF')
    return root


@pytest.fixture()
def make_client(gui_root):
    servers = []

    def make(**options):
        app = GuiApp(gui_root, **options)
        server, _ = start_in_thread(app)
        servers.append(server)
        return Client(app)

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def client(make_client):
    return make_client()
