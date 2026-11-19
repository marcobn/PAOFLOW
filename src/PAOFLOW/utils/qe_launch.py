"""How the generated drivers start Quantum ESPRESSO: their defaults, or the settings of a run.

The drivers that run QE themselves (``main.acbn0.py``, ``main.phonon.py``, ``main.raman.py``,
``main.sscha.py`` with the local launcher, ``qe/run_qe.sh`` of an electron-phonon system) keep
their launch commands as editable defaults. These variables take precedence, so that one place
sets the parallel run, the Runs tab of ``paoflow-studio`` (or a shell)::

    PAOFLOW_MPI_QE       MPI prefix of the QE programs ('mpirun -np 8'; empty: serial)
    PAOFLOW_QE_BIN       bin folder of the Quantum ESPRESSO build
    PAOFLOW_QE_OPTIONS   runtime options of pw.x / ph.x ('-nk 4 -nd 1')
    PAOFLOW_MPI_HARTREE  MPI prefix of the ACBN0 Hartree step (default: PAOFLOW_MPI_QE)
    PAOFLOW_MPI_PY       MPI prefix of the ACBN0 PAOFLOW step
    PAOFLOW_PYTHON_BIN   bin folder of the python that has PAOFLOW (ACBN0)

``paoflow-studio`` sets them from the ranks, the Quantum ESPRESSO build and the options of the
step (:func:`PAOFLOW.gui.jobs.build_command`); in a terminal::

    PAOFLOW_MPI_QE='mpirun -np 8' PAOFLOW_QE_OPTIONS='-nk 4' python main.phonon.py run
"""

from __future__ import annotations

import os

MPI_QE = 'PAOFLOW_MPI_QE'
QE_BIN = 'PAOFLOW_QE_BIN'
QE_OPTIONS = 'PAOFLOW_QE_OPTIONS'
MPI_HARTREE = 'PAOFLOW_MPI_HARTREE'
MPI_PY = 'PAOFLOW_MPI_PY'
PYTHON_BIN = 'PAOFLOW_PYTHON_BIN'

#: Every variable of the launch contract.
VARIABLES = (MPI_QE, QE_BIN, QE_OPTIONS, MPI_HARTREE, MPI_PY, PYTHON_BIN)


def mpi_prefix(default: str, variable: str = MPI_QE) -> str:
    """The MPI prefix of a parallel step: the variable when set (even empty: serial), else ``default``.

    Parameters
    ----------
    default : str
        The driver's own setting (``'mpirun -np 4'``).
    variable : str, optional
        :data:`MPI_QE` (QE programs, default), :data:`MPI_HARTREE` or :data:`MPI_PY`.  The
        Hartree step follows :data:`MPI_QE` when only that one is set (the ranks of the run).

    Returns
    -------
    str
    """
    if variable in os.environ:
        return os.environ[variable]
    if variable == MPI_HARTREE and MPI_QE in os.environ:
        return os.environ[MPI_QE]
    return default


def qe_bin(default: str = '') -> str:
    """The bin folder of Quantum ESPRESSO: :data:`QE_BIN` when set, else ``default``.

    Parameters
    ----------
    default : str, optional
        The driver's own setting (empty: the programs on the PATH).

    Returns
    -------
    str
    """
    return os.environ.get(QE_BIN, default)


def qe_options(default: str = '') -> str:
    """The runtime options of pw.x / ph.x: :data:`QE_OPTIONS` when set, else ``default``.

    Parameters
    ----------
    default : str, optional
        The driver's own setting.

    Returns
    -------
    str
    """
    return os.environ.get(QE_OPTIONS, default)


def python_bin(default: str = '') -> str:
    """The bin folder of the python that has PAOFLOW: :data:`PYTHON_BIN` when set, else ``default``.

    Parameters
    ----------
    default : str, optional
        The driver's own setting.

    Returns
    -------
    str
    """
    return os.environ.get(PYTHON_BIN, default)


def qe_command(program: str, default: str) -> str:
    """Command of a QE program: the build and the options of the run, else the driver's default.

    Parameters
    ----------
    program : str
        ``'pw.x'``, ``'ph.x'``, ...
    default : str
        The driver's own command, which may carry options (``'pw.x -npool 4'``).

    Returns
    -------
    str
        ``<PAOFLOW_QE_BIN>/<program> <PAOFLOW_QE_OPTIONS>`` when either variable is set (the run
        decides the build and the options), else ``default``.
    """
    folder = os.environ.get(QE_BIN)
    options = os.environ.get(QE_OPTIONS)
    if folder is None and options is None:
        return default
    executable = os.path.join(folder, program) if folder else program
    return ' '.join(part for part in (executable, options) if part)
