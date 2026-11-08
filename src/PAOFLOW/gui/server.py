"""The local HTTP server of the PAOFLOW Studio (standard library only).

:class:`GuiApp` holds the session (root, token, options) and the routes; the API modules
(:mod:`PAOFLOW.gui.workspace`, :mod:`PAOFLOW.gui.generator_api`, ...) register their endpoints on
it.  :class:`GuiServer` is a threaded ``http.server`` bound to the loopback interface whose
handler checks the Host header, the session token and, for POST requests, the Origin before a
route runs.  Responses are JSON, static files, or a byte stream (downloads); the browser polls the
logs of the runs and the live results.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlencode, urlsplit

from . import security

STATIC_DIR = Path(__file__).resolve().parent / 'static'
MAX_BODY_BYTES = 20 * 1024 * 1024


class HttpError(Exception):
    """An error answered with ``status`` and a JSON ``{"error": message}`` body.

    Parameters
    ----------
    status : int
        HTTP status.
    message : str
        Message shown by the GUI.
    details : dict, optional
        Extra fields of the JSON body.
    """

    def __init__(self, status: int, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or {}


@dataclass
class Request:
    """An HTTP request as the routes see it.

    Parameters
    ----------
    method : str
        ``GET`` or ``POST``.
    path : str
        URL path.
    query : dict
        Query parameters (last value of each).
    headers : Mapping
        Request headers.
    body : bytes
        Request body.
    """

    method: str
    path: str
    query: dict[str, str]
    headers: Mapping[str, str]
    body: bytes = b''

    def json(self) -> Any:
        """The body parsed as JSON (``{}`` when empty).

        Returns
        -------
        Any

        Raises
        ------
        HttpError
            400 when the body is not JSON.
        """
        if not self.body:
            return {}
        try:
            return json.loads(self.body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpError(400, 'the request body is not JSON: %s' % exc) from None


@dataclass
class Response:
    """An HTTP response.

    Parameters
    ----------
    status : int
        HTTP status.
    body : bytes or iterable of bytes
        Content, or chunks written as they come (event streams).
    content_type : str
        MIME type.
    headers : dict
        Extra headers.
    """

    status: int = 200
    body: bytes | Iterable[bytes] = b''
    content_type: str = 'application/json; charset=utf-8'
    headers: dict[str, str] = field(default_factory=dict)


def json_response(data: Any, status: int = 200) -> Response:
    """A JSON response (no caching).

    Parameters
    ----------
    data : Any
        JSON-serialisable content.
    status : int, optional
        HTTP status.

    Returns
    -------
    Response
    """
    body = json.dumps(finite(data), default=_json_default, allow_nan=False).encode('utf-8')
    return Response(status=status, body=body, headers={'Cache-Control': 'no-store'})


def finite(data: Any) -> Any:
    """``data`` with every non-finite float replaced by ``None`` (strict JSON).

    Parameters
    ----------
    data : Any
        Nested dicts, lists, tuples and scalars.

    Returns
    -------
    Any
    """
    import math

    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {key: finite(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [finite(value) for value in data]
    if hasattr(data, 'tolist') and hasattr(data, 'dtype'):  # numpy arrays and scalars
        return finite(data.tolist())
    return data


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError('%r is not JSON serialisable' % type(value).__name__)


Route = Callable[[Request], Response]


class GuiApp:
    """Session state and routes of the GUI.

    Parameters
    ----------
    root : str or Path
        Folder the browser may read and write under.
    token : str, optional
        Session token (random by default).
    read_only : bool, optional
        Refuse writes and runs.
    max_ranks : int, optional
        Cap of the MPI ranks of a run (default: the core budget of the machine).
    start : str, optional
        Folder (relative to the root) the GUI opens in.
    read_roots : sequence of str or Path, optional
        Extra folders the browser may read but never write (``--allow-read``).
    default_read_roots : bool, optional
        Add PAOFLOW's own ``PSEUDOS`` and ``BASIS`` folders when they exist (default).
    """

    def __init__(
        self,
        root: str | Path,
        token: str | None = None,
        read_only: bool = False,
        max_ranks: int | None = None,
        start: str | None = None,
        read_roots: Sequence[str | Path] = (),
        default_read_roots: bool = True,
    ) -> None:
        folders = list(read_roots) + (paoflow_data_folders() if default_read_roots else [])
        self.sandbox = security.Sandbox(root, read_roots=folders)
        self.token = token or security.new_token()
        self.read_only = read_only
        self.max_ranks = max_ranks
        self.start = start or '.'
        self.port = 0
        self.lock = threading.Lock()
        self._routes: dict[tuple[str, str], Route] = {}
        self._prefix_routes: list[tuple[str, str, Route]] = []
        self.services: dict[str, Any] = {}
        from . import generator_api, jobs, results_api, workspace

        for module in (workspace, generator_api, results_api, jobs):
            module.register(self)

    # ----------------------------------------------------------------------------- routes
    def route(self, method: str, path: str, prefix: bool = False) -> Callable[[Route], Route]:
        """Decorator registering a route.

        Parameters
        ----------
        method : str
            ``GET`` or ``POST``.
        path : str
            Exact path, or a prefix when ``prefix`` is true (``/api/jobs/``).
        prefix : bool, optional
            Match every path that starts with ``path``.

        Returns
        -------
        callable
        """

        def register(function: Route) -> Route:
            if prefix:
                self._prefix_routes.append((method, path, function))
            else:
                self._routes[(method, path)] = function
            return function

        return register

    def require_writable(self) -> None:
        """Refuse the request in a read-only session.

        Raises
        ------
        HttpError
            403 in read-only mode.
        """
        if self.read_only:
            raise HttpError(403, 'this GUI session is read-only (paoflow-studio --read-only)')

    def resolve(self, path: str | None) -> Path:
        """A browser path resolved inside the root.

        Parameters
        ----------
        path : str or None
            Relative (to the root) or absolute path.

        Returns
        -------
        Path

        Raises
        ------
        HttpError
            403 when the path leaves the root.
        """
        try:
            return self.sandbox.resolve(path)
        except PermissionError as exc:
            raise HttpError(403, str(exc)) from None

    def resolve_readable(self, path: str | None) -> Path:
        """A browser path resolved inside the root or a read-only folder (for reading only).

        Parameters
        ----------
        path : str or None
            Relative (to the root) or absolute path.

        Returns
        -------
        Path

        Raises
        ------
        HttpError
            403 when the path is outside the root and every read-only folder.
        """
        try:
            return self.sandbox.resolve_readable(path)
        except PermissionError as exc:
            raise HttpError(403, str(exc)) from None

    def dispatch(self, request: Request) -> Response:
        """Run the route of a request (static files, API, or 404).

        Parameters
        ----------
        request : Request
            The request (already authorised).

        Returns
        -------
        Response
        """
        route = self._routes.get((request.method, request.path))
        if route is None:
            for method, prefix, function in self._prefix_routes:
                if method == request.method and request.path.startswith(prefix):
                    route = function
                    break
        if route is None and request.method == 'GET':
            return self.static(request.path)
        if route is None:
            raise HttpError(404, 'no route %s %s' % (request.method, request.path))
        return route(request)

    def static(self, path: str) -> Response:
        """A file of the GUI front end, or the Plotly bundle of the installed ``plotly``.

        Parameters
        ----------
        path : str
            URL path (``/``, ``/static/...``, ``/vendor/plotly.min.js``).

        Returns
        -------
        Response
        """
        if path in ('/', '/index.html'):
            target = STATIC_DIR / 'index.html'
        elif path == '/vendor/plotly.min.js':
            target = plotly_bundle()
            if target is None:
                raise HttpError(404, 'plotly is not installed: pip install "PAOFLOW[gui]"')
        elif path.startswith('/static/'):
            target = (STATIC_DIR / path[len('/static/') :]).resolve()
            if not target.is_relative_to(STATIC_DIR):
                raise HttpError(404, 'not found')
        else:
            raise HttpError(404, 'not found: %s' % path)
        if not target.is_file():
            raise HttpError(404, 'not found: %s' % path)
        content_type = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        if content_type.startswith('text/') or content_type.endswith('javascript'):
            content_type += '; charset=utf-8'
        return Response(
            body=target.read_bytes(),
            content_type=content_type,
            headers={'Cache-Control': 'no-cache'},
        )


def paoflow_data_folders() -> list[Path]:
    """PAOFLOW's own ``PSEUDOS`` and ``BASIS`` folders (source checkouts and editable installs).

    Returns
    -------
    list of Path
        The folders that exist.
    """
    package = Path(__file__).resolve().parents[1]
    candidates = [
        package.parents[1] / 'PSEUDOS',
        package.parents[1] / 'BASIS',
        package / 'PSEUDOS',
        package / 'BASIS',
    ]
    return [folder for folder in candidates if folder.is_dir()]


def plotly_bundle() -> Path | None:
    """``plotly.min.js`` shipped with the installed ``plotly`` package, if any.

    Returns
    -------
    Path or None
    """
    try:
        import plotly
    except ImportError:
        return None
    bundle = Path(plotly.__file__).resolve().parent / 'package_data' / 'plotly.min.js'
    return bundle if bundle.is_file() else None


mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('image/svg+xml', '.svg')

UNAUTHORISED_PAGE = b"""<!doctype html><meta charset="utf-8"><title>PAOFLOW Studio</title>
<body style="font-family:system-ui;margin:3em;max-width:40em">
<h1>PAOFLOW Studio</h1><p>This session needs its access token. Open the URL that
<code>paoflow-studio</code> printed in the terminal (it ends with <code>?token=...</code>).</p>
</body>"""


class GuiRequestHandler(BaseHTTPRequestHandler):
    """Request handler: Host, token and Origin checks, then :meth:`GuiApp.dispatch`."""

    server_version = 'paoflow-studio'
    sys_version = ''
    protocol_version = 'HTTP/1.1'

    @property
    def app(self) -> GuiApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if getattr(self.server, 'verbose', False):
            sys.stderr.write('paoflow-studio: %s\n' % (format % args))

    def do_GET(self) -> None:  # noqa: N802
        self._handle('GET')

    def do_POST(self) -> None:  # noqa: N802
        self._handle('POST')

    def _send(self, response: Response) -> None:
        streaming = not isinstance(response.body, (bytes, bytearray))
        self.send_response(response.status)
        self.send_header('Content-Type', response.content_type)
        for name, value in {**security.SECURITY_HEADERS, **response.headers}.items():
            self.send_header(name, value)
        if streaming:
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                for chunk in response.body:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
        else:
            self.send_header('Content-Length', str(len(response.body)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(response.body)

    def _error(self, status: int, message: str, details: dict | None = None) -> None:
        self._send(json_response({'error': message, **(details or {})}, status=status))

    def _handle(self, method: str) -> None:
        app = self.app
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        if not security.host_allowed(self.headers.get('Host'), app.port):
            self._error(403, 'unexpected Host header')
            return
        given = query.pop('token', None)
        if given is not None and method == 'GET':
            if not security.same_token(given, app.token):
                self._error(403, 'wrong token')
                return
            location = url.path + ('?' + urlencode(query) if query else '')
            self._send(
                Response(
                    status=303,
                    headers={
                        'Location': location,
                        'Set-Cookie': security.session_cookie(app.token),
                    },
                    content_type='text/plain; charset=utf-8',
                )
            )
            return
        presented = self.headers.get(security.TOKEN_HEADER) or security.cookie_token(
            self.headers.get('Cookie')
        )
        if not security.same_token(presented, app.token):
            if method == 'GET' and url.path in ('/', '/index.html'):
                self._send(
                    Response(
                        status=403, body=UNAUTHORISED_PAGE, content_type='text/html; charset=utf-8'
                    )
                )
            else:
                self._error(403, 'missing or wrong session token')
            return
        if method == 'POST' and not security.origin_allowed(self.headers.get('Origin'), app.port):
            self._error(403, 'cross-origin request refused')
            return
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY_BYTES:
            self._error(413, 'request body too large')
            return
        body = self.rfile.read(length) if length else b''
        request = Request(method, url.path, query, self.headers, body)
        try:
            response = app.dispatch(request)
        except HttpError as exc:
            self._error(exc.status, exc.message, exc.details)
            return
        except Exception as exc:  # report, keep serving
            import traceback

            traceback.print_exc()
            self._error(500, '%s: %s' % (type(exc).__name__, exc))
            return
        self._send(response)


class GuiServer(ThreadingHTTPServer):
    """Threaded HTTP server of a :class:`GuiApp` on the loopback interface.

    Parameters
    ----------
    app : GuiApp
        The application.
    port : int, optional
        Port (0: any free port).
    host : str, optional
        Loopback address to bind.
    verbose : bool, optional
        Log every request on stderr.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, app: GuiApp, port: int = 0, host: str = '127.0.0.1', verbose: bool = False):
        if host not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('the GUI binds to the loopback interface only (use an SSH tunnel)')
        super().__init__((host, port), GuiRequestHandler)
        self.app = app
        self.verbose = verbose
        app.port = self.server_address[1]

    @property
    def url(self) -> str:
        """URL of the GUI with the session token."""
        return 'http://127.0.0.1:%d/?token=%s' % (self.app.port, self.app.token)


def start_in_thread(app: GuiApp, port: int = 0) -> tuple[GuiServer, threading.Thread]:
    """Start a server in a daemon thread (tests, notebooks).

    Parameters
    ----------
    app : GuiApp
        The application.
    port : int, optional
        Port (0: any free port).

    Returns
    -------
    server : GuiServer
    thread : threading.Thread
    """
    server = GuiServer(app, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def environment_features() -> dict[str, bool]:
    """Optional packages the GUI can use.

    Returns
    -------
    dict
        ``plotly``, ``matplotlib``, ``skimage`` (Fermi surfaces), ``mpi`` (mpirun on the PATH).
    """
    import importlib.util
    import shutil

    return {
        'plotly': plotly_bundle() is not None,
        'matplotlib': importlib.util.find_spec('matplotlib') is not None,
        'skimage': importlib.util.find_spec('skimage') is not None,
        'mpi': shutil.which('mpirun') is not None or shutil.which('mpiexec') is not None,
        'pw': shutil.which('pw.x') is not None,
    }


def cpu_budget() -> int:
    """Cores this process may use (the PAOFLOW core budget, else the CPU count).

    Returns
    -------
    int
    """
    try:
        from ..utils.parallel_resources import core_budget

        return max(1, int(core_budget()))
    except Exception:  # pragma: no cover - fallback outside MPI-aware setups
        return max(1, os.cpu_count() or 1)
