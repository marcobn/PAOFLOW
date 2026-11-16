"""pw.x outputs and job logs: the state of an scf run and the error signatures of a failed one.

:data:`PW_OUTPUT_PATTERN` selects the lines :func:`parse_pw_output` needs, so a remote machine can send a few
kilobytes (``grep -E``) instead of a whole output. :func:`error_signatures` scans any log (pw.x, ph.x, Slurm,
MPI, Python) for the failures met in production campaigns.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

PW_OUTPUT_PATTERN = (
    r'number of k points|FFT dimensions|number of Kohn-Sham states|number of atoms/cell|iteration #'
    r'|total cpu time spent|estimated scf accuracy|convergence has been achieved|convergence NOT achieved'
    r'|JOB DONE|Sym\. Ops|No symmetry found|initial density is read|PWSCF +:|lattice parameter \(alat\)'
    r'|a\([123]\) = |charge density cutoff|Error in routine|cholesky|not positive definite'
    r'|Maximum CPU time exceeded|too many bands are not converged'
)
"""``grep -E`` pattern of the pw.x output lines read by :func:`parse_pw_output`."""

PW_INPUT_PATTERN = r'K_POINTS|^ *[0-9]+ +[0-9]+ +[0-9]+ +[01] +[01] +[01] *$|nosym|nat *=|ecutrho|mixing_|startingpot'
"""``grep -E`` pattern of the pw.x input lines read by :func:`parse_pw_input`."""

ERROR_SIGNATURES = {
    'davidson_cholesky': r'problems computing cholesky|S matrix not positive definite',
    'bands_not_converged': r'too many bands are not converged',
    'scf_not_converged': r'convergence NOT achieved',
    'max_seconds': r'Maximum CPU time exceeded',
    'qe_error': r'Error in routine',
    'mpi_count_overflow': r'MPI_ERR_(ARG|COUNT|TRUNCATE)|Message truncated',
    'mpi_abort': r'MPI_ABORT was invoked|application called MPI_Abort',
    'python_exception': r'Traceback \(most recent call last\)',
    'out_of_memory': r'oom[-_ ]kill|Out Of Memory|OUT_OF_MEMORY|std::bad_alloc|MemoryError|[Cc]annot allocate memory',
    'segmentation_fault': r'Segmentation fault|SIGSEGV|signal 11 |Bus error',
    'time_limit': r'DUE TO TIME LIMIT',
    'node_failure': r'DUE TO NODE FAILURE|NODE_FAIL',
    'qgrid_does_not_divide_kgrid': r'q-grid .* must divide the k-grid',
    'kpoints_not_accounted': r'NOT ALL KPOINTS ACCOUNTED FOR',
    'missing_file': r'No such file or directory|FileNotFoundError',
}
"""Regular expressions of the failures seen in production runs, by code (see :mod:`~.diagnosis`)."""


@dataclass
class PwOutput:
    """State of one pw.x run read from its output.

    Attributes
    ----------
    finished : bool
        ``JOB DONE`` printed.
    converged : bool or None
        True after ``convergence has been achieved``, False after ``convergence NOT achieved``, None while running.
    accuracies : list of float
        Estimated scf accuracy (Ry) after each iteration.
    cpu_seconds : list of float
        ``total cpu time spent up to now`` after each iteration (wall time of the master process in practice).
    kpoints, bands, atoms : int or None
        Irreducible k-points, Kohn-Sham states and atoms of the run.
    fft_dimensions : tuple of int or None
        Dense FFT box.
    symmetry_operations : int or None
        Crystal symmetry operations used (0 for ``No symmetry found``).
    density_from_file : bool
        The run started from a density on file (``startingpot = 'file'``).
    cell_bohr : numpy.ndarray or None
        Lattice vectors (rows, bohr).
    ecutrho : float or None
        Charge-density cutoff (Ry).
    wall_seconds : float or None
        Final wall time (``PWSCF : ... WALL``).
    errors : list of str
        Error lines (``Error in routine`` and the message after it).
    """

    finished: bool = False
    converged: bool | None = None
    accuracies: list[float] = field(default_factory=list)
    cpu_seconds: list[float] = field(default_factory=list)
    kpoints: int | None = None
    bands: int | None = None
    atoms: int | None = None
    fft_dimensions: tuple[int, int, int] | None = None
    symmetry_operations: int | None = None
    density_from_file: bool = False
    cell_bohr: np.ndarray | None = None
    ecutrho: float | None = None
    wall_seconds: float | None = None
    max_seconds_reached: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        """scf iterations with an accuracy estimate."""
        return len(self.accuracies)

    def seconds_per_iteration(self, last: int = 5) -> float | None:
        """Mean time of the last ``last`` scf iterations (the first iteration, with its long diagonalisation, excluded).

        Parameters
        ----------
        last : int
            Number of iterations averaged.

        Returns
        -------
        float or None
            Seconds per iteration, None before the third iteration.
        """
        times = self.cpu_seconds
        if len(times) < 3:
            return None
        steps = np.diff(times[1:])
        return float(np.mean(steps[-last:]))

    def largest_rise(self) -> float:
        """Largest ratio of a later scf accuracy to the first one (1 if it never rises).

        A well-mixed scf decreases from the first estimate on; a metallic supercell that sloshes jumps by orders of
        magnitude in the first iterations (Ta0.8Hf0.2C, 80 atoms, plain mixing at beta 0.7: 25 -> 1000 -> 9300 Ry).
        """
        if len(self.accuracies) < 2:
            return 1.0
        return max(1.0, max(self.accuracies[1:]) / self.accuracies[0])

    def iterations_to_converge(self, conv_thr: float = 1e-8, fit_last: int = 8) -> float | None:
        """Further scf iterations to reach ``conv_thr`` by the log-linear trend of the last iterations.

        Parameters
        ----------
        conv_thr : float
            scf convergence threshold of the run (Ry).
        fit_last : int
            Iterations used in the fit of :math:`\\log_{10}` of the accuracy against the iteration number.

        Returns
        -------
        float or None
            0 if converged, ``inf`` if the trend does not decrease, None with fewer than four iterations.

        Notes
        -----
        The accuracy of a well-mixed scf decreases roughly geometrically, so a straight line in
        :math:`\\log_{10}(\\text{accuracy})` extrapolated to :math:`\\log_{10}(\\text{conv\\_thr})` estimates the
        iterations left. Sloshing runs bounce and give a flat or rising line.
        """
        if self.converged:
            return 0.0
        values = [a for a in self.accuracies if a > 0]
        if len(values) < 4:
            return None
        tail = np.log10(values[-fit_last:])
        slope, intercept = np.polyfit(np.arange(len(tail)), tail, 1)
        if slope >= -1e-3:
            return math.inf
        current = slope * (len(tail) - 1) + intercept
        return max(0.0, (math.log10(conv_thr) - current) / slope)

    def stagnating(self, window: int = 6, conv_thr: float = 1e-8) -> bool:
        """True if the best accuracy of the last ``window`` iterations is no better than that of the previous window."""
        values = self.accuracies
        if len(values) < 2 * window or min(values) < 100 * conv_thr:
            return False
        return min(values[-window:]) >= 0.5 * min(values[-2 * window : -window])


def _wall_seconds(text: str) -> float | None:
    """Seconds of a pw.x clock string (``4925.79s``, ``1h22m``, ``2d 3h 5m``)."""
    total, found = 0.0, False
    for value, unit in re.findall(r'([\d.]+)\s*([dhms])', text):
        total += float(value) * {'d': 86400, 'h': 3600, 'm': 60, 's': 1}[unit]
        found = True
    return total if found else None


def parse_pw_output(text: str) -> PwOutput:
    """State of a pw.x run from its output (the whole file or the lines of :data:`PW_OUTPUT_PATTERN`).

    Parameters
    ----------
    text : str
        pw.x output text.

    Returns
    -------
    PwOutput
        Progress, size and errors of the run.
    """
    state = PwOutput()
    alat = None
    vectors: dict[int, list[float]] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if 'JOB DONE' in line:
            state.finished = True
        elif 'convergence has been achieved' in line:
            state.converged = True
        elif 'convergence NOT achieved' in line:
            state.converged = False
        elif 'estimated scf accuracy' in line:
            match = re.search(r'<\s*([-+\d.EeDd]+)\s*Ry', line)
            if match:
                state.accuracies.append(float(match.group(1).replace('D', 'E').replace('d', 'e')))
        elif 'total cpu time spent up to now is' in line:
            match = re.search(r'is\s+([\d.]+)\s*secs', line)
            if match:
                state.cpu_seconds.append(float(match.group(1)))
        elif 'number of k points=' in line and state.kpoints is None:
            state.kpoints = int(re.search(r'=\s*(\d+)', line).group(1))
        elif 'FFT dimensions' in line and state.fft_dimensions is None:
            dims = re.findall(r'\d+', line.split('FFT dimensions')[1])
            state.fft_dimensions = tuple(int(d) for d in dims[:3])
        elif 'number of Kohn-Sham states' in line and state.bands is None:
            state.bands = int(re.search(r'=\s*(\d+)', line).group(1))
        elif 'number of atoms/cell' in line and state.atoms is None:
            state.atoms = int(re.search(r'=\s*(\d+)', line).group(1))
        elif 'No symmetry found' in line:
            state.symmetry_operations = 0
        elif 'Sym. Ops' in line and state.symmetry_operations is None:
            match = re.search(r'(\d+)\s+Sym\. Ops', line)
            if match:
                state.symmetry_operations = int(match.group(1))
        elif 'initial density is read' in line:
            state.density_from_file = True
        elif 'Maximum CPU time exceeded' in line:
            state.max_seconds_reached = True
        elif re.search(r'PWSCF\s+:', line):
            state.wall_seconds = _wall_seconds(line.split('CPU')[-1])
        elif 'lattice parameter (alat)' in line:
            alat = float(re.search(r'=\s*([\d.]+)', line).group(1))
        elif re.search(r'a\(([123])\) = \(', line):
            number = int(re.search(r'a\(([123])\)', line).group(1))
            vectors[number] = [
                float(v) for v in re.findall(r'-?\d+\.\d+', line.split('=', 1)[1])[:3]
            ]
        elif 'charge density cutoff' in line and state.ecutrho is None:
            state.ecutrho = float(re.search(r'=\s*([\d.]+)', line).group(1))
        elif 'Error in routine' in line:
            message = lines[index + 1].strip() if index + 1 < len(lines) else ''
            state.errors.append('%s: %s' % (line.strip(), message))
    if alat is not None and len(vectors) == 3:
        state.cell_bohr = np.array([vectors[1], vectors[2], vectors[3]]) * alat
    return state


@dataclass
class PwInput:
    """The settings of a pw.x input that the checks need."""

    kgrid: tuple[int, int, int] | None = None
    kshift: tuple[int, int, int] | None = None
    nosym: bool = False
    atoms: int | None = None
    ecutrho: float | None = None
    mixing_beta: float | None = None
    mixing_mode: str | None = None
    density_from_file: bool = False


def parse_pw_input(text: str) -> PwInput:
    """Settings of a pw.x input (the whole file or the lines of :data:`PW_INPUT_PATTERN`).

    Parameters
    ----------
    text : str
        pw.x input text.

    Returns
    -------
    PwInput
        k-grid (automatic), symmetry switch, atoms, cutoff, mixing, starting potential.
    """
    settings = PwInput()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        lower = line.lower()
        if 'k_points' in lower and 'automatic' in lower and index + 1 < len(lines):
            numbers = [int(v) for v in lines[index + 1].split()[:6]]
            settings.kgrid, settings.kshift = tuple(numbers[:3]), tuple(numbers[3:6])
        for key, value in re.findall(r'([a-z_]+)\s*=\s*([^,\s]+)', lower):
            value = value.strip('\'"')
            if key == 'nosym':
                settings.nosym = value.startswith('.t')
            elif key == 'nat':
                settings.atoms = int(value)
            elif key == 'ecutrho':
                settings.ecutrho = float(value.replace('d', 'e'))
            elif key == 'mixing_beta':
                settings.mixing_beta = float(value.replace('d', 'e'))
            elif key == 'mixing_mode':
                settings.mixing_mode = value
            elif key == 'startingpot':
                settings.density_from_file = value == 'file'
    return settings


def error_signatures(text: str) -> list[tuple[str, str]]:
    """Failures found in a log, as ``(code, first matching line)`` pairs in the order of :data:`ERROR_SIGNATURES`.

    Parameters
    ----------
    text : str
        Any log: pw.x / ph.x output, Slurm ``.out`` file, PAOFLOW driver log.

    Returns
    -------
    list of tuple
        One entry per code found.
    """
    found = []
    for code, pattern in ERROR_SIGNATURES.items():
        match = re.search(r'^.*(%s).*$' % pattern, text, flags=re.M)
        if match:
            found.append((code, match.group(0).strip()[:300]))
    return found
