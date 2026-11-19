"""Questions shared by the ``run``, ``acbn0`` and ``phonon`` workflows (the QE run and the basis)."""

from __future__ import annotations

from .model import Answers, Context, Question, comma_list

SYSTEM = 'System'


def detected_savedir(context: Context) -> str | None:
    """The ``<prefix>.save`` found in the working directory (first, sorted).

    Parameters
    ----------
    context : Context
        The session context.

    Returns
    -------
    str or None
    """
    from .. import paoflow_driver as pd

    return context.cached('savedir', lambda: pd.detect_savedir(context.workdir))


def detected_upfs(context: Context) -> list[str]:
    """The pseudopotentials of the save (or of the working directory).

    Parameters
    ----------
    context : Context
        The session context.

    Returns
    -------
    list of str
    """
    from .. import paoflow_driver as pd

    return context.cached(
        'upfs', lambda: pd.detect_upfs(context.workdir, detected_savedir(context))
    )


def detected_prefix(context: Context) -> str:
    """The QE prefix (from the save name, else ``scf.in`` / ``nscf.in``, else ``pwscf``).

    Parameters
    ----------
    context : Context
        The session context.

    Returns
    -------
    str
    """
    from .. import paoflow_driver as pd

    return context.cached(
        'prefix', lambda: pd.detect_prefix(context.workdir, detected_savedir(context))
    )


def save_summary(answers: Answers, context: Context) -> dict | None:
    """Summary of the answered save (``None`` when it has no readable ``data-file-schema.xml``).

    Parameters
    ----------
    answers : dict
        Answers so far (``savedir``).
    context : Context
        The session context.

    Returns
    -------
    dict or None
        :func:`PAOFLOW.gen.system_summary.read_save_summary`.
    """
    import os

    from ..system_summary import read_save_summary

    savedir = answers.get('savedir')
    if not savedir:
        return None
    path = os.path.join(context.workdir, savedir)
    return context.cached('save:' + path, lambda: read_save_summary(path))


def from_save(key: str, fallback):
    """Default taken from the save summary (``fallback`` without a readable save).

    Parameters
    ----------
    key : str
        Key of :func:`PAOFLOW.gen.system_summary.read_save_summary`.
    fallback : Any
        The default without a save.

    Returns
    -------
    callable
        ``default(answers, context)``.
    """

    def default(answers: Answers, context: Context):
        summary = save_summary(answers, context)
        return summary[key] if summary is not None else fallback

    return default


def _default_upfs(answers: Answers, context: Context) -> str:
    upfs = detected_upfs(context)
    return ', '.join(upfs) if upfs else 'pseudo.UPF'


COMMON_ITEMS = (
    Question(
        'savedir',
        'Save directory (<prefix>.save)',
        default=lambda a, c: detected_savedir(c) or 'pwscf.save',
        override='savedir',
        label='Save directory',
        help='The <prefix>.save folder of the scf/nscf run, relative to the working directory.',
        section=SYSTEM,
        widget='folder',
    ),
    Question(
        'prefix',
        'Prefix',
        default=lambda a, c: detected_prefix(c),
        override='prefix',
        section=SYSTEM,
    ),
    Question(
        'upfs',
        'Pseudopotential (UPF) file(s), comma-separated',
        default=_default_upfs,
        parse=comma_list,
        override='upf',
        label='Pseudopotentials',
        help='One UPF per species; the pseudo-atom basis is generated from them.',
        section=SYSTEM,
    ),
    Question(
        'basisdir',
        'Basis directory name',
        default='BASIS_PS',
        help='Folder of the pseudo-atom basis tables (paoflow-genbasis-ps), created if missing.',
        section=SYSTEM,
    ),
    Question(
        'ibrav',
        'ibrav (0 = read cell; needs band path for bands)',
        kind='int',
        default=lambda a, c: (save_summary(a, c) or {}).get('bravais_index') or 0,
        label='ibrav',
        help='QE Bravais-lattice index of the default band path (from the save); 0 reads the '
        'cell and needs an explicit band path.',
        section=SYSTEM,
    ),
    Question(
        'is_2d',
        'Is this a 2D system (slab with vacuum along c)?',
        kind='bool',
        default=from_save('two_dimensional', False),
        label='2D system (vacuum along c)',
        help='Default from the cell of the save: a slab when the atoms fill less than half of '
        'the height along c.',
        section=SYSTEM,
    ),
    Question(
        'projector',
        'Projector source (internal = PAOFLOW atomic projection; projwfc = QE projwfc.x)',
        kind='choice',
        choices=['internal', 'projwfc'],
        default='internal',
        label='Projector',
        help='internal: PAOFLOW projects on the pseudo-atom basis; projwfc: QE projwfc.x '
        'projections (needed by eACBN0 for the true overlaps).',
        section=SYSTEM,
    ),
)

COMMON_OVERRIDES = ('savedir', 'prefix', 'upf')


def assemble_common(answers: Answers, context: Context) -> dict:
    """Configuration of the common questions.

    Parameters
    ----------
    answers : dict
        Answers of :data:`COMMON_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
        ``savedir``, ``prefix``, ``upfs``, ``basisdir``, ``ibrav``, ``is_2d``, ``projector``,
        ``outputdir`` and ``workdir``.
    """
    return {
        'savedir': answers['savedir'],
        'prefix': answers['prefix'],
        'upfs': answers['upfs'],
        'basisdir': answers['basisdir'],
        'ibrav': answers['ibrav'],
        'is_2d': answers['is_2d'],
        'projector': answers['projector'],
        'outputdir': 'output',
        'workdir': context.workdir,
    }
