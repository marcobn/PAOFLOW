"""Local run manager of the GUI (decision D2: runs on the machine that serves the GUI).

Runs are the steps of a generated workflow's run plan (:func:`PAOFLOW.gen.outputs.run_plan`).
A step becomes a process only through :func:`build_command`, which accepts ``python`` on a driver
file inside the root, ``bash`` on a shell script inside the root, and ``paoflow-genbasis-ps``;
arguments are plain tokens and nothing goes through a shell.  MPI steps get ``mpirun -np N`` with
``N`` capped by the session (``--max-ranks``, default the core budget), and every process runs
with one thread per rank (``OMP/OPENBLAS/MKL/VECLIB_NUM_THREADS=1``), niced, in its own session
so it survives the GUI.  Job records and logs live in ``<root>/.paoflow-studio/jobs/``; a restarted
GUI shows the earlier jobs and follows the ones still running.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .server import HttpError, Request, Response, json_response

if TYPE_CHECKING:
    from .server import GuiApp

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')
#: Quantum ESPRESSO programs the GUI may start (from the installation chosen in the Runs tab, see
#: :mod:`PAOFLOW.gui.qe_installations`; by default the one on the PATH of the server).
QE_PROGRAMS = (
    'pw.x',
    'ph.x',
    'projwfc.x',
    'pp.x',
    'q2r.x',
    'matdyn.x',
    'dos.x',
    'bands.x',
    'epw.x',
    'hp.x',
    'dynmat.x',
)
QE_INPUT_FLAGS = ('-in', '-inp', '-input', '-i')
QE_PARALLEL_FLAGS = (
    '-nk',
    '-npool',
    '-npools',
    '-ni',
    '-nimage',
    '-nd',
    '-ndiag',
    '-nt',
    '-ntg',
    '-nb',
    '-nbgrp',
)
OUTPUT_NAME = re.compile(r'^[\w.-]+$')
SAFE_TOKEN = re.compile(r'^[\w.,:=+@%/-]+$')
SHELL_SCRIPT = re.compile(r'^[\w./-]+\.sh$')
THREAD_VARIABLES = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)
MAX_LOG_CHUNK = 256 * 1024


@dataclass
class Job:
    """A launched step.

    Parameters
    ----------
    id : str
        Identifier.
    label : str
        What the step does.
    argv : list of str
        The command that was executed.
    cwd : str
        Folder, relative to the root.
    ranks : int
        MPI ranks (1 without MPI).
    status : str
        ``running``, ``finished``, ``failed``, ``cancelled`` or ``lost``.
    pid : int
        Process id (process group leader).
    started, ended : float
        Times (``ended`` 0 while running).
    returncode : int or None
        Exit status.
    log : str
        Log file, relative to the root.
    """

    id: str
    label: str
    argv: list[str]
    cwd: str
    ranks: int = 1
    status: str = 'running'
    pid: int = 0
    started: float = field(default_factory=time.time)
    ended: float = 0.0
    returncode: int | None = None
    log: str = ''

    def to_json(self) -> dict:
        data = asdict(self)
        data['command'] = ' '.join(self.argv)
        data['elapsed'] = (self.ended or time.time()) - self.started
        return data


def _python_executable() -> str:
    return sys.executable or shutil.which('python3') or 'python3'


def build_command(
    app: 'GuiApp', step: dict, ranks: int, qe_bin: str | None = None
) -> tuple[list[str], Path, dict[str, str]]:
    """The argument list, folder and environment of a plan step (validated).

    Parameters
    ----------
    app : GuiApp
        The application (root, rank cap, Quantum ESPRESSO installations).
    step : dict
        ``argv``, ``cwd`` (relative to the root), ``mpi`` and ``ranks_env``.
    ranks : int
        Requested MPI ranks.
    qe_bin : str, optional
        ``bin`` folder of the Quantum ESPRESSO build to use (one of
        :meth:`GuiApp.qe_installations`). A QE program runs from it, and it goes first on the PATH
        of the run, so shell scripts and drivers that call QE use it too. QE programs use the
        default build when it is not given.

    Returns
    -------
    argv : list of str
    cwd : Path
    env : dict

    Raises
    ------
    HttpError
        400 for a step outside the whitelist or invalid arguments.
    """
    argv = [str(token) for token in step.get('argv') or []]
    if not argv:
        raise HttpError(400, 'this step has no command (do it by hand)')
    cwd = app.resolve(step.get('cwd') or '.')
    if not cwd.is_dir():
        raise HttpError(
            400, 'the folder %s does not exist (write the files first)' % step.get('cwd')
        )
    for token in argv[1:]:
        if not SAFE_TOKEN.match(token):
            raise HttpError(400, 'unsupported argument %r' % token)
    program = argv[0]
    qe = app.qe_installation(qe_bin) if qe_bin or program in QE_PROGRAMS else None
    if program == 'python':
        if len(argv) < 2 or not argv[1].endswith('.py'):
            raise HttpError(400, 'python runs a driver file')
        script = app.resolve(str(cwd / argv[1]))
        if not script.is_file():
            raise HttpError(400, '%s not found (write the files first)' % argv[1])
        command = [_python_executable(), '-u', *argv[1:]]
    elif program == 'bash':
        if len(argv) != 2 or not SHELL_SCRIPT.match(argv[1]):
            raise HttpError(400, 'bash runs one generated shell script')
        script = app.resolve(str(cwd / argv[1]))
        if not script.is_file():
            raise HttpError(400, '%s not found' % argv[1])
        command = ['bash', argv[1]]
    elif program == 'paoflow-genbasis-ps':
        command = [_python_executable(), '-u', '-m', 'PAOFLOW.basis_gen.cli', *argv[1:]]
    elif program in QE_PROGRAMS:
        command = [_qe_executable(qe, program), *_qe_arguments(app, cwd, argv[1:])]
    else:
        raise HttpError(
            400,
            'the GUI runs python drivers, generated shell scripts, paoflow-genbasis-ps and the QE '
            'programs (%s) only' % ', '.join(QE_PROGRAMS),
        )
    cap = app.max_ranks or _core_budget()
    try:
        ranks = int(ranks)
    except (TypeError, ValueError):
        raise HttpError(400, 'ranks must be an integer') from None
    if not 1 <= ranks <= cap:
        raise HttpError(400, 'ranks must be between 1 and %d (paoflow-studio --max-ranks)' % cap)
    env = dict(os.environ)
    for name in THREAD_VARIABLES:
        env[name] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    if qe is not None:
        env['PATH'] = str(qe.bin) + os.pathsep + env.get('PATH', '')
    if program in QE_PROGRAMS and qe.mpi is False and ranks > 1:
        raise HttpError(
            400, 'QE %s in %s is a serial build (no MPI): run it with 1 rank' % (qe.version, qe.bin)
        )
    if step.get('mpi') and ranks > 1:
        own = qe.mpirun if qe is not None and program in QE_PROGRAMS else None
        launcher = str(own) if own else shutil.which('mpirun') or shutil.which('mpiexec')
        if launcher is None:
            raise HttpError(400, 'mpirun is not on the PATH: run with 1 rank')
        command = [launcher, '-np', str(ranks), *command]
    if step.get('ranks_env'):
        if not re.match(r'^[A-Z_][A-Z0-9_]*$', step['ranks_env']):
            raise HttpError(400, 'invalid environment variable')
        env[step['ranks_env']] = str(ranks)
    if shutil.which('nice'):
        command = ['nice', '-n', '10', *command]
    return command, cwd, env


def _qe_executable(qe, program: str) -> str:
    """Absolute path of a QE program in the chosen installation."""
    executable = qe.bin / program
    if not (executable.is_file() and os.access(executable, os.X_OK)):
        raise HttpError(
            400,
            '%s is not in %s (QE %s): choose another Quantum ESPRESSO installation in the Runs tab'
            % (program, qe.bin, qe.version or 'version ?'),
        )
    return str(executable)


def _qe_arguments(app: 'GuiApp', cwd: Path, arguments: list[str]) -> list[str]:
    """Validated arguments of a QE program: one input file inside the folder and parallel flags."""
    checked: list[str] = []
    seen_input = False
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        value = arguments[index + 1] if index + 1 < len(arguments) else None
        if value is None:
            raise HttpError(400, '%s needs a value' % flag)
        if flag in QE_INPUT_FLAGS:
            path = app.resolve(str(cwd / value))
            if not path.is_file():
                raise HttpError(400, '%s not found (write the input first)' % value)
            seen_input = True
        elif flag in QE_PARALLEL_FLAGS:
            if not value.isdigit():
                raise HttpError(400, '%s takes an integer' % flag)
        else:
            raise HttpError(
                400, 'unsupported QE option %r (input file and parallel flags only)' % flag
            )
        checked += [flag, value]
        index += 2
    if not seen_input:
        raise HttpError(400, 'a QE run needs its input file (-in <file>)')
    return checked


def _core_budget() -> int:
    from .server import cpu_budget

    return cpu_budget()


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobManager:
    """Launch, follow and cancel local jobs; keep their records on disk.

    Parameters
    ----------
    app : GuiApp
        The application.
    """

    def __init__(self, app: 'GuiApp') -> None:
        self.app = app
        self.folder = app.sandbox.root / '.paoflow-studio' / 'jobs'
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self._load()

    # ------------------------------------------------------------------------ records
    def _records(self) -> Path:
        return self.folder / 'jobs.json'

    def _load(self) -> None:
        try:
            records = json.loads(self._records().read_text())
        except (OSError, ValueError):
            return
        for record in records:
            record.pop('command', None)
            record.pop('elapsed', None)
            job = Job(**record)
            if job.status == 'running' and not _alive(job.pid):
                job.status = 'lost' if job.returncode is None else job.status
                job.ended = job.ended or time.time()
            self.jobs[job.id] = job
        for job in self.jobs.values():
            if job.status == 'running':
                threading.Thread(target=self._follow_detached, args=(job,), daemon=True).start()

    def _save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        records = [job.to_json() for job in self.jobs.values()]
        temporary = self._records().with_suffix('.tmp')
        temporary.write_text(json.dumps(records, indent=1))
        temporary.replace(self._records())

    # ------------------------------------------------------------------------ actions
    def launch(self, step: dict, ranks: int = 1, qe_bin: str | None = None) -> Job:
        """Start a plan step.

        Parameters
        ----------
        step : dict
            Plan step (``label``, ``argv``, ``cwd``, ``mpi``, ``ranks_env``).
        ranks : int, optional
            MPI ranks.
        qe_bin : str, optional
            ``bin`` folder of the Quantum ESPRESSO build (see :func:`build_command`).

        Returns
        -------
        Job
        """
        self.app.require_writable()
        if step.get('interactive') or step.get('manual'):
            raise HttpError(400, 'this step needs a terminal or a manual action')
        command, cwd, env = build_command(self.app, step, ranks, qe_bin)
        identifier = time.strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(2)
        self.folder.mkdir(parents=True, exist_ok=True)
        output = step.get('stdout')
        if output:
            # The program's standard output is its result file (pw.x > scf.out): write it there,
            # without a header, and follow it as the log.
            if not OUTPUT_NAME.match(str(output)):
                raise HttpError(400, 'invalid output file name %r' % output)
            log_path = self.app.resolve(str(cwd / str(output)))
        else:
            log_path = self.folder / ('%s.log' % identifier)
        with open(log_path, 'wb') as log:
            if not output:
                log.write(('$ %s\n# in %s\n\n' % (' '.join(command), cwd)).encode())
                log.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        job = Job(
            id=identifier,
            label=str(step.get('label') or command[-1]),
            argv=command,
            cwd=self.app.sandbox.relative(cwd),
            ranks=int(ranks),
            pid=process.pid,
            log=self.app.sandbox.relative(log_path),
        )
        with self.lock:
            self.jobs[identifier] = job
            self.processes[identifier] = process
            self._save()
        threading.Thread(target=self._wait, args=(job, process), daemon=True).start()
        return job

    def _finish(self, job: Job, returncode: int | None) -> None:
        with self.lock:
            job.returncode = returncode
            job.ended = time.time()
            if job.status != 'cancelled':
                job.status = (
                    'finished' if returncode == 0 else ('lost' if returncode is None else 'failed')
                )
            self.processes.pop(job.id, None)
            self._save()

    def _wait(self, job: Job, process: subprocess.Popen) -> None:
        self._finish(job, process.wait())

    def _follow_detached(self, job: Job) -> None:
        while _alive(job.pid):
            time.sleep(2.0)
        self._finish(job, None)

    def cancel(self, identifier: str) -> Job:
        """Stop a running job (SIGTERM to its process group, SIGKILL after 10 s).

        Parameters
        ----------
        identifier : str
            Job id.

        Returns
        -------
        Job
        """
        job = self.get(identifier)
        if job.status != 'running':
            return job
        job.status = 'cancelled'
        try:
            os.killpg(job.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        def kill_later() -> None:
            time.sleep(10.0)
            if _alive(job.pid):
                try:
                    os.killpg(job.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        threading.Thread(target=kill_later, daemon=True).start()
        with self.lock:
            self._save()
        return job

    def get(self, identifier: str) -> Job:
        """A job by id.

        Parameters
        ----------
        identifier : str
            Job id.

        Returns
        -------
        Job
        """
        job = self.jobs.get(identifier)
        if job is None:
            raise HttpError(404, 'no job %s' % identifier)
        return job

    def log(self, identifier: str, offset: int = 0) -> dict:
        """The log of a job from a byte offset.

        Parameters
        ----------
        identifier : str
            Job id.
        offset : int, optional
            Bytes already read.

        Returns
        -------
        dict
            ``text``, ``offset`` (new), ``status``, ``job``.
        """
        job = self.get(identifier)
        path = self.app.resolve(job.log)
        text = ''
        try:
            with open(path, 'rb') as handle:
                handle.seek(max(0, offset))
                data = handle.read(MAX_LOG_CHUNK)
                offset = handle.tell()
            text = ANSI_ESCAPE.sub('', data.decode('utf-8', errors='replace'))
        except OSError:
            pass
        return {'text': text, 'offset': offset, 'status': job.status, 'job': job.to_json()}

    def detach(self) -> None:
        """Leave the running jobs running when the GUI stops (they have their own session)."""
        with self.lock:
            self._save()


def register(app: 'GuiApp') -> None:
    """Register the run endpoints.

    Parameters
    ----------
    app : GuiApp
        The application.
    """
    manager = JobManager(app)
    app.services['jobs'] = manager

    @app.route('GET', '/api/jobs')
    def jobs(request: Request) -> Response:
        ordered = sorted(manager.jobs.values(), key=lambda job: -job.started)
        return json_response(
            {
                'jobs': [job.to_json() for job in ordered],
                'max_ranks': app.max_ranks or _core_budget(),
            }
        )

    @app.route('POST', '/api/jobs/launch')
    def launch(request: Request) -> Response:
        payload = request.json()
        job = manager.launch(
            payload.get('step') or {}, payload.get('ranks', 1), payload.get('qe_bin') or None
        )
        return json_response(job.to_json())

    @app.route('GET', '/api/qe')
    def qe_installations(request: Request) -> Response:
        from .qe_installations import default_installation

        installations = app.qe_installations(rescan=bool(request.query.get('rescan')))
        default = default_installation(installations)
        return json_response(
            {
                'installations': [installation.to_json() for installation in installations],
                'default': str(default.bin) if default is not None else None,
            }
        )

    @app.route('POST', '/api/jobs/cancel')
    def cancel(request: Request) -> Response:
        return json_response(manager.cancel(request.json().get('id', '')).to_json())

    @app.route('GET', '/api/jobs/log')
    def log(request: Request) -> Response:
        try:
            offset = int(request.query.get('offset', 0))
        except ValueError:
            offset = 0
        return json_response(manager.log(request.query.get('id', ''), offset))
