"""``paoflow-studio``: the browser interface of PAOFLOW (setup, runs and results).

Usage::

    paoflow-studio                         # serve the current folder, open the browser
    paoflow-studio --root ~/runs --port 8765 --no-browser
    paoflow-studio plot Si/output          # open the Results tab on a folder
    paoflow-studio report Si/output -o Si_report.html   # static HTML report, no server

On a remote machine run it with ``--no-browser`` and tunnel the port::

    ssh -N -L 8765:127.0.0.1:8765 <host>

then open the printed URL (it carries the session token) in the local browser.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser


def _serve(args: argparse.Namespace, start: str | None, tab: str) -> int:
    from .server import GuiApp, GuiServer

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        sys.stderr.write('paoflow-studio: %s is not a folder\n' % root)
        return 1
    if start is not None:
        start_path = os.path.abspath(os.path.expanduser(start))
        if not os.path.exists(start_path) and not os.path.isabs(os.path.expanduser(start)):
            start_path = os.path.join(root, start)  # relative to the root
        if not (start_path == root or start_path.startswith(root + os.sep)):
            sys.stderr.write('paoflow-studio: %s is outside the root %s\n' % (start_path, root))
            return 1
        start = os.path.relpath(start_path, root)
    app = GuiApp(
        root,
        token=args.token,
        read_only=args.read_only,
        max_ranks=args.max_ranks,
        start=start,
    )
    server = None
    last_error = None
    for port in [args.port] if args.port == 0 else range(args.port, args.port + 20):
        try:
            server = GuiServer(app, port=port, verbose=args.verbose)
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        sys.stderr.write('paoflow-studio: no free port from %d: %s\n' % (args.port, last_error))
        return 1
    url = server.url + '#' + tab
    print('PAOFLOW Studio serving %s%s' % (root, ' (read-only)' if args.read_only else ''))
    print('  open:   %s' % url)
    print(
        '  remote: ssh -N -L %d:127.0.0.1:%d <this host>, then open the URL above'
        % (app.port, app.port)
    )
    print('  stop:   Ctrl-C')
    sys.stdout.flush()
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\npaoflow-studio stopped')
    finally:
        server.server_close()
        jobs = app.services.get('jobs')
        if jobs is not None:
            jobs.detach()
    return 0


def _report(args: argparse.Namespace) -> int:
    try:
        from ..results.report import write_report
    except ImportError as exc:  # pragma: no cover - results layer missing
        sys.stderr.write('paoflow-studio report: %s\n' % exc)
        return 1
    path = write_report(args.folder, args.output, title=args.title)
    print('Wrote %s' % path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser of ``paoflow-studio``.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='paoflow-studio',
        description='Browser interface of PAOFLOW: generate drivers, run them, plot the results.',
    )
    parser.add_argument(
        '--root', default='.', help='folder the GUI may read and write (default: .)'
    )
    parser.add_argument(
        '--port', type=int, default=8765, help='first port to try (0: any free port)'
    )
    parser.add_argument('--no-browser', action='store_true', help='do not open a browser')
    parser.add_argument(
        '--read-only', action='store_true', help='browse and plot only: no writes, no runs'
    )
    parser.add_argument('--max-ranks', type=int, default=None, help='cap of the MPI ranks of a run')
    parser.add_argument('--token', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--verbose', action='store_true', help='log every request')
    commands = parser.add_subparsers(dest='command')
    setup = commands.add_parser('setup', help='open the Setup tab (default)')
    setup.add_argument('folder', nargs='?', default=None, help='working folder inside the root')
    plot = commands.add_parser('plot', help='open the Results tab on a folder')
    plot.add_argument('folder', help='output folder (or a run folder) inside the root')
    report = commands.add_parser(
        'report', help='write a static HTML report of a folder (no server)'
    )
    report.add_argument('folder', help='output folder (or a run folder)')
    report.add_argument('-o', '--output', default='paoflow_report.html', help='HTML file to write')
    report.add_argument('--title', default=None, help='title of the report')
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``paoflow-studio``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (``sys.argv[1:]`` by default).

    Returns
    -------
    int
        Exit status.
    """
    args = build_parser().parse_args(argv)
    if args.command == 'report':
        return _report(args)
    if args.command == 'plot':
        return _serve(args, args.folder, 'results')
    return _serve(args, getattr(args, 'folder', None), 'setup')


if __name__ == '__main__':
    raise SystemExit(main())
