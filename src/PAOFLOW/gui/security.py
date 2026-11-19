"""Security of the local GUI server: session token, Host/Origin checks and the path sandbox.

The server binds to the loopback interface only, but on a shared machine (an HPC login node)
every user can reach ``127.0.0.1:<port>``, and any web page open in the user's browser can try
cross-site requests or DNS rebinding.  Hence:

* a random token, given once in the URL and then kept in an ``HttpOnly; SameSite=Strict`` cookie,
  is required on every request;
* the ``Host`` header must name the loopback address and the server port, and state-changing
  requests must come from the same origin;
* every path the browser sends is resolved under the server root (symbolic links included), and
  anything outside it is refused.
"""

from __future__ import annotations

import hmac
import secrets
from pathlib import Path
from typing import Sequence

COOKIE_NAME = 'paoflow_studio_token'
TOKEN_HEADER = 'X-PAOFLOW-Token'
LOOPBACK_NAMES = ('127.0.0.1', 'localhost', '[::1]')

#: Headers sent with every response.
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Content-Security-Policy': (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; connect-src 'self'; font-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
}


def new_token() -> str:
    """A random session token (256 bits, URL safe).

    Returns
    -------
    str
    """
    return secrets.token_urlsafe(32)


def same_token(given: str | None, expected: str) -> bool:
    """Constant-time comparison of a presented token with the session token.

    Parameters
    ----------
    given : str or None
        Token of the request.
    expected : str
        Session token.

    Returns
    -------
    bool
    """
    return bool(given) and hmac.compare_digest(given.encode(), expected.encode())


def cookie_token(cookie_header: str | None) -> str | None:
    """The session token of a ``Cookie`` header.

    Parameters
    ----------
    cookie_header : str or None
        The header value.

    Returns
    -------
    str or None
    """
    if not cookie_header:
        return None
    for part in cookie_header.split(';'):
        name, _, value = part.strip().partition('=')
        if name == COOKIE_NAME:
            return value
    return None


def session_cookie(token: str) -> str:
    """``Set-Cookie`` value of the session token.

    Parameters
    ----------
    token : str
        The session token.

    Returns
    -------
    str
    """
    return '%s=%s; HttpOnly; SameSite=Strict; Path=/' % (COOKIE_NAME, token)


def allowed_hosts(port: int) -> set[str]:
    """``Host`` header values accepted by a server on ``port``.

    Parameters
    ----------
    port : int
        The server port.

    Returns
    -------
    set of str
    """
    return {'%s:%d' % (name, port) for name in LOOPBACK_NAMES}


def host_allowed(host_header: str | None, port: int) -> bool:
    """Whether the ``Host`` header names this loopback server (DNS-rebinding guard).

    Parameters
    ----------
    host_header : str or None
        The header value.
    port : int
        The server port.

    Returns
    -------
    bool
    """
    return (host_header or '').strip().lower() in allowed_hosts(port)


def origin_allowed(origin_header: str | None, port: int) -> bool:
    """Whether a state-changing request comes from the GUI itself.

    Parameters
    ----------
    origin_header : str or None
        The ``Origin`` header (browsers send it on POST); absent for non-browser clients.
    port : int
        The server port.

    Returns
    -------
    bool
    """
    if origin_header is None:
        return True
    return origin_header.strip().lower() in {'http://' + host for host in allowed_hosts(port)}


class Sandbox:
    """Resolve browser paths: writes under the server root, reads also under read-only folders.

    Parameters
    ----------
    root : str or Path
        The root folder (``paoflow-studio --root``): read and written.
    read_roots : sequence of str or Path, optional
        Extra folders that are read only (``paoflow-studio --allow-read``, PAOFLOW's own
        ``PSEUDOS`` and ``BASIS``).

    Notes
    -----
    Folders opened from the browser's folder picker (:meth:`open`, unless the server runs with
    ``--fixed-root``) join the root (``write_roots``) or the read-only folders.
    """

    def __init__(self, root: str | Path, read_roots: Sequence[str | Path] = ()) -> None:
        self.root = Path(root).expanduser().resolve()
        self.write_roots: list[Path] = []
        roots: list[Path] = []
        for folder in read_roots:
            path = Path(folder).expanduser().resolve()
            if (
                path.is_dir()
                and path != self.root
                and not path.is_relative_to(self.root)
                and path not in roots
            ):
                roots.append(path)
        self.read_roots = roots

    @staticmethod
    def _candidate(root: Path, path: str | Path | None) -> tuple[str, Path]:
        text = '' if path is None else str(path).strip()
        candidate = Path(text).expanduser() if text else root
        if not candidate.is_absolute():
            candidate = root / candidate
        return text, candidate.resolve()

    @staticmethod
    def _inside(path: Path, folder: Path) -> bool:
        return path == folder or path.is_relative_to(folder)

    def _writable_roots(self) -> list[Path]:
        return [self.root, *self.write_roots]

    def writable(self, path: str | Path) -> bool:
        """Whether a path is inside the root or a folder opened for writing.

        Parameters
        ----------
        path : str or Path
            Absolute path, or relative to the root.

        Returns
        -------
        bool
        """
        _text, resolved = self._candidate(self.root, path)
        return any(self._inside(resolved, folder) for folder in self._writable_roots())

    def open(self, path: str | Path, writable: bool = True) -> Path:
        """Add a folder chosen in the browser to the folders of the session.

        Parameters
        ----------
        path : str or Path
            Absolute path of an existing folder.
        writable : bool, optional
            Read and write it (a working folder), or read it only.

        Returns
        -------
        Path
            The resolved folder.

        Raises
        ------
        NotADirectoryError
            If the path is not a folder.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError('%s is not a folder' % path)
        if self.writable(resolved):
            return resolved
        if writable:
            self.write_roots.append(resolved)
            self.read_roots = [f for f in self.read_roots if f != resolved]
        elif not any(self._inside(resolved, folder) for folder in self.read_roots):
            self.read_roots.append(resolved)
        return resolved

    def resolve(self, path: str | Path | None) -> Path:
        """Absolute path of ``path`` inside the root (relative to it, or absolute): where writes go.

        Parameters
        ----------
        path : str or Path or None
            Path from the browser; blank means the root.

        Returns
        -------
        Path

        Raises
        ------
        PermissionError
            If the path (after resolving symbolic links and ``..``) leaves the root.
        """
        text, resolved = self._candidate(self.root, path)
        if not any(self._inside(resolved, folder) for folder in self._writable_roots()):
            readable = any(self._inside(resolved, folder) for folder in self.read_roots)
            raise PermissionError(
                '%s is %s'
                % (
                    text,
                    'read-only (an --allow-read folder, or one opened for reading)'
                    if readable
                    else 'outside the GUI root %s (open it with Change… or Browse…)' % self.root,
                )
            )
        return resolved

    def resolve_readable(self, path: str | Path | None) -> Path:
        """Absolute path of ``path`` inside the root or a read-only folder (reads only).

        Parameters
        ----------
        path : str or Path or None
            Path from the browser: relative to the root, or absolute.

        Returns
        -------
        Path

        Raises
        ------
        PermissionError
            If the path is outside the root and every read-only folder.
        """
        text, resolved = self._candidate(self.root, path)
        if any(
            self._inside(resolved, folder) for folder in [*self._writable_roots(), *self.read_roots]
        ):
            return resolved
        raise PermissionError(
            '%s is outside the GUI root %s and the read-only folders; start paoflow-studio with '
            '--allow-read <folder> to read it' % (text, self.root)
        )

    def readable(self, path: str | Path) -> bool:
        """Whether a path may be read (inside the root or a read-only folder).

        Parameters
        ----------
        path : str or Path
            Absolute path, or relative to the root.

        Returns
        -------
        bool
        """
        try:
            self.resolve_readable(path)
        except (PermissionError, OSError, ValueError):
            return False
        return True

    def place(self, path: str | Path) -> Path:
        """The root or read-only folder that contains ``path``.

        Parameters
        ----------
        path : str or Path
            A resolved readable path.

        Returns
        -------
        Path
        """
        resolved = Path(path).resolve()
        for folder in [self.root, *self.write_roots, *self.read_roots]:
            if self._inside(resolved, folder):
                return folder
        raise PermissionError('%s is not readable' % path)

    def relative(self, path: str | Path) -> str:
        """Path relative to the root (``'.'`` for the root itself); absolute outside the root.

        Parameters
        ----------
        path : str or Path
            An absolute path inside the root or a read-only folder.

        Returns
        -------
        str
        """
        resolved = Path(path).resolve()
        if not self._inside(resolved, self.root):
            return str(resolved)
        relative = resolved.relative_to(self.root)
        return str(relative) if str(relative) != '.' else '.'
