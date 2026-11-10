"""Run a plotting script without a display and keep its figures as files.

``python -m PAOFLOW.graphics.figure_capture --out DIR script.py [arguments]`` runs ``script.py``
as ``__main__``, with its arguments, on matplotlib's Agg backend. Every figure the script shows
(``plt.show()``) or saves (``savefig``) is also written to ``DIR``, as PNG for display and PDF for
print. ``DIR/figures.json`` lists the figures in order and is rewritten after each one, so a
viewer can show them while the script runs. Prompts (``input()``) take their defaults.

The Plots tab of ``paoflow-studio`` runs the generated plot scripts (``plot.py``,
``plot.acbn0.py``, ``superconductivity/plot_me.py``, ...) this way. The same command works on a
machine without a display, such as a cluster login node.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import re
import runpy
import sys
import time
from pathlib import Path

MANIFEST = 'figures.json'
FORMATS = ('png', 'pdf')
DPI = 150


def capture_folder(script: str) -> str:
    """Folder of the captured figures of a script, relative to the folder it runs in.

    Parameters
    ----------
    script : str
        The script, relative to the folder it runs in (``plot.py``,
        ``superconductivity/plot_me.py``).

    Returns
    -------
    str
        ``figures/<name>``, the name being the script path without ``.py`` and with ``/`` as
        ``_`` (``figures/plot``, ``figures/superconductivity_plot_me``).
    """
    stem = script[:-3] if script.endswith('.py') else script
    return 'figures/' + re.sub(r'[^\w.-]+', '_', stem).strip('_.')


def _slug(text: str) -> str:
    return re.sub(r'\W+', '_', text).strip('_')[:60] or 'figure'


def figure_title(figure) -> str:
    """Title of a figure: its suptitle, or the title of its first axes that has one.

    Parameters
    ----------
    figure : matplotlib.figure.Figure

    Returns
    -------
    str
        Empty when the figure has no title.
    """
    suptitle = getattr(figure, '_suptitle', None)
    if suptitle is not None and suptitle.get_text().strip():
        return suptitle.get_text().strip()
    for axes in figure.get_axes():
        for location in ('center', 'left', 'right'):
            title = axes.get_title(location).strip()
            if title:
                return title
    return ''


class FigureCapture:
    """The figures of one run: files in a folder and their manifest.

    Parameters
    ----------
    folder : Path
        Where the files go (created).
    formats : tuple of str, optional
        File formats of each figure; the first is the one a viewer shows.
    dpi : int, optional
        Resolution of the raster formats.
    script : str, optional
        The script, for the manifest.
    arguments : list of str, optional
        Its arguments, for the manifest.
    """

    def __init__(
        self,
        folder: Path,
        formats: tuple[str, ...] = FORMATS,
        dpi: int = DPI,
        script: str = '',
        arguments: list[str] | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.formats = tuple(formats)
        self.dpi = dpi
        self.script = script
        self.arguments = list(arguments or [])
        self.started = time.time()
        self.entries: list[dict] = []
        self._captured: set[int] = set()
        self._savefig = None  # the original Figure.savefig, set by install()

    # ------------------------------------------------------------------------ files
    def clear(self) -> None:
        """Remove the files of the previous run (only those its manifest lists)."""
        manifest = self.folder / MANIFEST
        try:
            previous = json.loads(manifest.read_text())
        except (OSError, ValueError):
            previous = {}
        for entry in previous.get('figures', []):
            for name in entry.get('files', {}).values():
                path = self.folder / str(name)
                if path.parent == self.folder:
                    path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)

    def write_manifest(self, status: int | None = None) -> None:
        """Write ``figures.json`` (``status`` None while the script runs).

        Parameters
        ----------
        status : int or None, optional
            Exit status of the script, once it has ended.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {
            'script': self.script,
            'arguments': self.arguments,
            'started': self.started,
            'ended': None if status is None else time.time(),
            'status': status,
            'figures': self.entries,
        }
        temporary = self.folder / (MANIFEST + '.tmp')
        temporary.write_text(json.dumps(data, indent=1))
        temporary.replace(self.folder / MANIFEST)

    # ------------------------------------------------------------------------ figures
    def capture(self, figure, target: str | None = None) -> None:
        """Write one figure in every format and add it to the manifest.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
        target : str, optional
            The file the script saved it to, if it did.
        """
        number = len(self.entries) + 1
        title = (
            figure_title(figure) or (Path(target).stem if target else '') or 'Figure %d' % number
        )
        stem = '%02d_%s' % (number, _slug(title))
        self.folder.mkdir(parents=True, exist_ok=True)
        files = {}
        save = self._savefig or type(figure).savefig
        for fmt in self.formats:
            name = '%s.%s' % (stem, fmt)
            save(figure, self.folder / name, format=fmt, dpi=self.dpi, bbox_inches='tight')
            files[fmt] = name
        # The tight bounding box restores the figure after drawing, which marks it changed; it
        # is changed again only when the script draws on it.
        figure.stale = False
        self._captured.add(id(figure))
        self.entries.append({'title': title, 'files': files, 'saved_as': target})
        self.write_manifest()

    def _is_new(self, figure) -> bool:
        # A figure drawn again unchanged (saved twice, or saved then shown) is not a new figure.
        return id(figure) not in self._captured or bool(figure.stale)

    def capture_open_figures(self, close: bool = True) -> None:
        """Capture the open figures not captured yet (``plt.show()``, the end of the script).

        Parameters
        ----------
        close : bool, optional
            Close them afterwards, as a blocking ``show()`` does when its windows close.
        """
        import matplotlib.pyplot as plt

        for number in plt.get_fignums():
            figure = plt.figure(number)
            if figure.get_axes() and self._is_new(figure):
                self.capture(figure)
        if close:
            plt.close('all')

    def install(self) -> None:
        """Route ``plt.show``, ``Figure.savefig``, ``matplotlib.use`` and ``input`` here."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        capture = self
        original = Figure.savefig
        self._savefig = original

        def savefig(figure, fname, *args, **kwargs):
            new = capture._is_new(figure)
            result = original(figure, fname, *args, **kwargs)
            if new:
                target = os.fspath(fname) if isinstance(fname, (str, os.PathLike)) else None
                capture.capture(figure, target)
            # Saving restores the size and colours of the figure, which marks it changed: what
            # was saved is what was captured.
            figure.stale = False
            return result

        def show(*args, **kwargs):
            capture.capture_open_figures()

        def use(*args, **kwargs):
            pass  # keep Agg: there is no display

        def answer_default(prompt=''):
            print('%s(default)' % prompt, flush=True)
            return ''

        Figure.savefig = savefig
        plt.show = show
        matplotlib.use = use
        builtins.input = answer_default


def main(argv: list[str] | None = None) -> int:
    """Command line: run a plotting script and keep its figures.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (``sys.argv[1:]`` when None).

    Returns
    -------
    int
        Exit status of the script.
    """
    parser = argparse.ArgumentParser(
        prog='python -m PAOFLOW.graphics.figure_capture',
        description='Run a plotting script without a display; its figures go to --out as PNG '
        'and PDF files, listed in figures.json.',
    )
    parser.add_argument('--out', required=True, help='folder of the figures')
    parser.add_argument(
        '--formats', default=','.join(FORMATS), help='comma-separated formats (png,pdf)'
    )
    parser.add_argument('--dpi', type=int, default=DPI, help='resolution of the PNG files')
    parser.add_argument('script', help='the plotting script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='arguments of the script')
    args = parser.parse_args(argv)

    os.environ['MPLBACKEND'] = 'Agg'
    script = os.path.abspath(args.script)
    formats = tuple(fmt.strip() for fmt in args.formats.split(',') if fmt.strip())
    capture = FigureCapture(
        Path(args.out), formats, args.dpi, script=args.script, arguments=args.arguments
    )
    capture.clear()
    capture.install()
    capture.write_manifest()
    sys.argv = [script, *args.arguments]
    sys.path.insert(0, os.path.dirname(script))
    status = 1
    try:
        runpy.run_path(script, run_name='__main__')
        status = 0
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            status = exc.code or 0
        else:
            print(exc.code, file=sys.stderr)
            status = 1
    finally:
        capture.capture_open_figures()
        capture.write_manifest(status)
        print(
            '%d figure%s in %s'
            % (len(capture.entries), '' if len(capture.entries) == 1 else 's', capture.folder),
            flush=True,
        )
    return status


if __name__ == '__main__':
    sys.exit(main())
