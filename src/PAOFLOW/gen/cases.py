"""Example cases (``examples/cases``): list them, plan their runs and copy one into a working folder.

A case is a folder with the inputs of a calculation (Quantum ESPRESSO, VASP or a model Hamiltonian),
a ``README.md`` and a ``case.json``:

==================  =====================================================================================
key                 content
==================  =====================================================================================
``title``           one line
``summary``         a few sentences: the system and what PAOFLOW computes
``workflow``        the generator workflow (``run``, ``landauer``, ``acbn0``, ...), or ``scripts`` for a
                    case whose drivers come with it (models, analyses, multi-stage folders)
``answers``         the answers of its form, as the Setup tab of ``paoflow-studio`` keeps them (not
                    used by ``scripts`` cases)
``pseudopotentials`` the UPF files, relative to the ``PSEUDOS`` folder (``library/Si....UPF``)
``pseudo_dir``      folder of the case the pseudopotentials are copied to (default ``.``)
``qe``              the QE steps in order: ``{"program": "pw.x", "input": "scf.in", "label": ...}``,
                    with an optional ``cwd`` (folder of the step, relative to the case)
``steps``           the PAOFLOW steps of the case, after the QE ones and the generated drivers:
                    ``{"label": ..., "command": ["python", "main.py"], "cwd": ".", "mpi": false,
                    "interactive": false, "note": ...}`` (required for ``scripts`` cases); a script that
                    runs Quantum ESPRESSO itself (``bash run_qe.sh``) adds ``"ranks_env": "NP"`` and
                    ``"qe_launch": true`` (the Runs tab passes the ranks and the QE build)
``tags``            keywords
``source``          the folder of the former example
==================  =====================================================================================

The pseudopotentials live once in ``PSEUDOS/library``; :func:`copy_case` copies them next to the
inputs (``pseudo_dir = './'`` in the inputs, or the ``pseudo_dir`` folder of the case). Cases come with a source checkout (or an editable install): a wheel
of PAOFLOW has no ``examples`` folder, and :func:`cases_root` returns None.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .outputs import RunStep

CASE_FILE = 'case.json'

#: Workflow of a case whose drivers come with it: its run plan is the list of its ``steps``.
SCRIPTS = 'scripts'

#: Entries of a case folder that a copy leaves behind (local runs, generated basis tables and kept
#: reference outputs).
NOT_COPIED = ('Reference', 'output', '__pycache__', 'BASIS_PS', '.ipynb_checkpoints')


@dataclass
class Case:
    """One example case.

    Parameters
    ----------
    name : str
        Folder name.
    folder : Path
        The case folder.
    title, summary, workflow : str
        From ``case.json``.
    answers : dict
        Answers of the workflow form.
    pseudopotentials : list of str
        UPF files relative to ``PSEUDOS``.
    qe : list of dict
        QE steps (``program``, ``input``, ``label``, optional ``cwd``).
    tags : list of str
        Keywords.
    source : str
        Folder of the former example.
    steps : list of dict
        PAOFLOW steps (``label``, ``command``, optional ``cwd``, ``mpi``, ``interactive``,
        ``note``).
    pseudo_dir : str
        Folder of the case that receives the pseudopotentials.
    """

    name: str
    folder: Path
    title: str
    summary: str
    workflow: str
    answers: dict[str, Any]
    pseudopotentials: list[str] = field(default_factory=list)
    qe: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = ''
    steps: list[dict[str, Any]] = field(default_factory=list)
    pseudo_dir: str = '.'

    @property
    def scripts(self) -> bool:
        """True for a case whose drivers come with it (``workflow == 'scripts'``)."""
        return self.workflow == SCRIPTS

    def to_json(self) -> dict[str, Any]:
        """The case for the browser (``folder`` left out)."""
        return {
            'name': self.name,
            'title': self.title,
            'summary': self.summary,
            'workflow': self.workflow,
            'answers': self.answers,
            'pseudopotentials': [Path(p).name for p in self.pseudopotentials],
            'qe': self.qe,
            'tags': self.tags,
            'source': self.source,
            'steps': self.steps,
            'scripts': self.scripts,
        }


def _repository_folder(*parts: str) -> Path | None:
    """``<checkout>/<parts>`` of a source checkout or an editable install, if it exists."""
    package = Path(__file__).resolve().parents[1]  # .../src/PAOFLOW
    for base in (package.parents[1], package):
        candidate = base.joinpath(*parts)
        if candidate.is_dir():
            return candidate
    return None


def cases_root() -> Path | None:
    """``examples/cases`` of the checkout, or None (an installed wheel has no examples)."""
    return _repository_folder('examples', 'cases')


def pseudopotentials_root() -> Path | None:
    """``PSEUDOS`` of the checkout, or None."""
    return _repository_folder('PSEUDOS')


def load_case(folder: Path) -> Case:
    """Read the ``case.json`` of ``folder``.

    Parameters
    ----------
    folder : Path
        A case folder.

    Returns
    -------
    Case

    Raises
    ------
    ValueError
        If ``case.json`` misses a required key (``answers`` for a generator workflow, ``steps``
        for a ``scripts`` case), or a step has no ``label`` or ``command``.
    """
    data = json.loads((folder / CASE_FILE).read_text(encoding='utf-8'))
    required = ['title', 'summary', 'workflow']
    required.append('steps' if data.get('workflow') == SCRIPTS else 'answers')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError('%s: %s misses %s' % (folder.name, CASE_FILE, ', '.join(missing)))
    for step in data.get('steps', []):
        if not step.get('label') or not step.get('command'):
            raise ValueError('%s: every step needs a label and a command' % folder.name)
    return Case(
        name=folder.name,
        folder=folder,
        title=data['title'],
        summary=data['summary'],
        workflow=data['workflow'],
        answers=dict(data.get('answers', {})),
        pseudopotentials=list(data.get('pseudopotentials', [])),
        qe=list(data.get('qe', [])),
        tags=list(data.get('tags', [])),
        source=data.get('source', ''),
        steps=list(data.get('steps', [])),
        pseudo_dir=str(data.get('pseudo_dir', '.')),
    )


def list_cases(root: Path | None = None) -> list[Case]:
    """Every case under ``root`` (default :func:`cases_root`), by name.

    Parameters
    ----------
    root : Path, optional
        Folder of the cases.

    Returns
    -------
    list of Case
        Empty without a cases folder.
    """
    root = root if root is not None else cases_root()
    if root is None or not root.is_dir():
        return []
    return [
        load_case(folder) for folder in sorted(root.iterdir()) if (folder / CASE_FILE).is_file()
    ]


def find_case(name: str, root: Path | None = None) -> Case:
    """The case called ``name``.

    Raises
    ------
    KeyError
        For an unknown name (the message lists the known ones).
    """
    cases = {case.name: case for case in list_cases(root)}
    if name not in cases:
        known = ', '.join(sorted(cases)) or 'none (no examples/cases folder)'
        raise KeyError('no example case %r; the cases are: %s' % (name, known))
    return cases[name]


def case_plan(case: Case) -> list[RunStep]:
    """The runs of a copied case: its QE steps, the PAOFLOW drivers of its workflow, its steps.

    Parameters
    ----------
    case : Case

    Returns
    -------
    list of RunStep
        ``pw.x`` / ``projwfc.x`` runs (MPI, output ``<input stem>.out``, in the ``cwd`` of the
        step); for a generator workflow, a manual step that opens the workflow in the Setup tab;
        then the ``steps`` of the case (``python`` runs with the interpreter of the GUI).
    """
    steps = [
        RunStep(
            '%s (%s)' % (step.get('label') or step['input'], step['program']),
            (step['program'], '-in', step['input']),
            cwd=step.get('cwd', '.'),
            mpi=True,
            stdout=step['input'][:-3] + '.out' if step['input'].endswith('.in') else None,
            note='%s must be on the PATH of paoflow-studio' % step['program'],
        )
        for step in case.qe
    ]
    if not case.scripts:
        steps.append(
            RunStep(
                'PAOFLOW drivers (%s workflow, answers of the case)' % case.workflow,
                (),
                manual=True,
                action='setup:%s' % case.workflow,
                note='Setup tab: the form is filled in; write the files, then run them here',
            )
        )
    for step in case.steps:
        command = step['command']
        argv = tuple(command.split() if isinstance(command, str) else command)
        steps.append(
            RunStep(
                step['label'],
                argv,
                cwd=step.get('cwd', '.'),
                mpi=bool(step.get('mpi', False)),
                ranks_env=step.get('ranks_env'),
                interactive=bool(step.get('interactive', False)),
                note=step.get('note', ''),
                qe_launch=bool(step.get('qe_launch', False)),
            )
        )
    return steps


def case_files(case: Case) -> list[Path]:
    """Files of the case folder that a copy takes (inputs, README, scripts), relative to it."""
    files = []
    for path in sorted(case.folder.rglob('*')):
        relative = path.relative_to(case.folder)
        if path.is_dir() or relative.parts[0] in NOT_COPIED or relative.name == CASE_FILE:
            continue
        if any(part.endswith('.save') for part in relative.parts):
            continue
        files.append(relative)
    return files


def copy_case(
    case: Case, destination: Path, overwrite: bool = False, pseudos: Path | None = None
) -> list[Path]:
    """Copy a case into ``destination``: its inputs and README, and its pseudopotentials.

    Parameters
    ----------
    case : Case
        The case.
    destination : Path
        Target folder (created).
    overwrite : bool, optional
        Replace files that exist and differ.
    pseudos : Path, optional
        The ``PSEUDOS`` folder (default :func:`pseudopotentials_root`).

    Returns
    -------
    list of Path
        The files written, relative to ``destination``.

    Raises
    ------
    FileExistsError
        Without ``overwrite``, if files of the case exist in ``destination`` with another content
        (``args[1]``: their relative paths); nothing is written then.
    FileNotFoundError
        If a pseudopotential is missing from ``PSEUDOS``.
    """
    pseudos = pseudos if pseudos is not None else pseudopotentials_root()
    sources = [(case.folder / relative, relative) for relative in case_files(case)]
    for upf in case.pseudopotentials:
        source = (pseudos / upf) if pseudos is not None else None
        if source is None or not source.is_file():
            raise FileNotFoundError(
                'pseudopotential %s of the case %s not found' % (upf, case.name)
            )
        sources.append((source, Path(case.pseudo_dir) / source.name))
    conflicts = [
        str(relative)
        for source, relative in sources
        if (destination / relative).exists()
        and (destination / relative).read_bytes() != source.read_bytes()
    ]
    if conflicts and not overwrite:
        raise FileExistsError('files of the case exist in %s' % destination, conflicts)
    written = []
    for source, relative in sources:
        target = destination / relative
        if target.exists() and target.read_bytes() == source.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        written.append(relative)
    return written
