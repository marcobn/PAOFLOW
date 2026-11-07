"""Declarative questions of the ``paoflow-gen`` workflows (shared by the CLI and the browser GUI).

Each workflow is a :class:`~PAOFLOW.gen.questions.model.Workflow`: an ordered list of questions and
notes plus the ``assemble`` function that turns the answers into the configuration the script
builders of :mod:`PAOFLOW.gen.paoflow_driver`, :mod:`PAOFLOW.gen.elphon_system` and
:mod:`PAOFLOW.gen.sscha_driver` read.  :func:`collect` is the interactive CLI of one workflow.
"""

from __future__ import annotations

from typing import Any

from .acbn0 import ACBN0
from .elphon import ELPHON
from .model import Answers, Context, Note, Question, Workflow
from .phonon import PHONON
from .prompt import ask_items
from .run import RUN
from .sscha import SSCHA

#: The workflows of ``paoflow-gen``, in the order of its first prompt.
WORKFLOWS: dict[str, Workflow] = {
    workflow.name: workflow for workflow in (ACBN0, RUN, PHONON, ELPHON, SSCHA)
}


def collect(name: str, args: Any, workdir: str) -> dict | None:
    """Ask the questions of a workflow on the terminal and assemble its configuration.

    Parameters
    ----------
    name : str
        Workflow name (a key of :data:`WORKFLOWS`).
    args : argparse.Namespace or any object
        Command-line options (``savedir``, ``prefix``, ``upf``, ``out`` overrides).
    workdir : str
        Directory scanned for the Quantum ESPRESSO artefacts.

    Returns
    -------
    dict or None
        The configuration, or ``None`` when the answers leave nothing to generate.
    """
    workflow = WORKFLOWS[name]
    context = Context.from_args(args, workdir, workflow.overrides)
    answers = ask_items(workflow.items, context)
    if answers is None:
        return None
    return workflow.assemble(answers, context)


__all__ = [
    'Answers',
    'Context',
    'Note',
    'Question',
    'WORKFLOWS',
    'Workflow',
    'ask_items',
    'collect',
]
