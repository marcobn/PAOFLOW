"""Browser interface of PAOFLOW (``paoflow-studio``): Setup, Runs and Results tabs.

The server (:mod:`PAOFLOW.gui.server`) uses the standard library only and binds to the loopback
interface; the front end (``static/``) is plain JavaScript with Plotly.js served from the
installed ``plotly`` package (``pip install "PAOFLOW[gui]"``).  The physics stays in PAOFLOW: the
Setup tab drives the ``paoflow-gen`` questions and builders (:mod:`PAOFLOW.gen.questions`), the
Results tab the readers and figure specifications of :mod:`PAOFLOW.results`.
"""

from __future__ import annotations

__all__ = ['main']


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``paoflow-studio`` (:func:`PAOFLOW.gui.cli.main`)."""
    from .cli import main as cli_main

    return cli_main(argv)
