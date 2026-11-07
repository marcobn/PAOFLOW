"""Anharmonic phonons from the stochastic self-consistent harmonic approximation (python-sscha).

Procedural backend of the ``PAOFLOW.sscha_*`` methods: the SSCHA (Monacelli, Errea, Calandra and Mauri,
J. Phys.: Condens. Matter 33, 363001 (2021)) minimises the free energy of the ions at temperature T over trial
Gaussian (harmonic) density matrices, sampling ionic configurations of a supercell whose forces come from pw.x.
python-sscha and CellConstructor are the engine (optional extra ``sscha``); this module connects them to PAOFLOW:

- the starting matrices from a ``ph.x`` q-grid run (the q of the supercell grid, imaginary modes made positive);
- the pw.x inputs of every configuration written by PAOFLOW's supercell writer, so the force runs carry the
  occupations, cutoffs and pseudopotentials of the parent run (Fermi-Dirac at k_B T for hot electrons);
- the forces of a population read from the pw.x outputs, the minimisation and the free-energy Hessian;
- the anharmonic correction (SSCHA minus harmonic on the supercell grid) interpolated onto the harmonic q-grid and
  written as QE dynamical matrices and real-space force constants (``q2r.x`` format, without ``q2r.x``).

CellConstructor works in Angstrom (structures), Ry / bohr^2 (force constants) and Ry (energies, masses in Ry units);
QE dynamical-matrix files are read and written in their own units.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

#: Ry -> cm^-1 (frequencies of CellConstructor, which returns them in Ry).
RY_TO_CM1 = 109737.31568
#: Atomic mass unit in the Rydberg mass units of QE and CellConstructor.
AMU_RY = 911.444243
#: Frequencies below this (cm^-1) count as unstable (0.5 meV).
UNSTABLE_CM1 = -4.03


def require_sscha() -> tuple[Any, Any]:
    """Import CellConstructor and python-sscha, with an explicit message when they are missing.

    Returns
    -------
    tuple
        ``(cellconstructor.Phonons module, sscha package)`` with ``sscha.Ensemble`` and
        ``sscha.SchaMinimizer`` imported.

    Raises
    ------
    ImportError
        If python-sscha or CellConstructor is not installed (``pip install PAOFLOW[sscha]``).
    """
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # optional aiida backend warning of python-sscha
            import cellconstructor.Phonons as cc_phonons
            import sscha
            import sscha.Ensemble  # noqa: F401
            import sscha.SchaMinimizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'the SSCHA methods need python-sscha and CellConstructor: pip install "PAOFLOW[sscha]" '
            '(Fortran compiler and BLAS/LAPACK required, see docs/internals/sscha.md)'
        ) from exc
    return cc_phonons, sscha


def dynamical_matrix_files(prefix: str) -> list[str]:
    """The QE dynamical-matrix files ``<prefix>1 ... <prefix>N`` (one per irreducible star), in order.

    Parameters
    ----------
    prefix : str
        Path prefix, e.g. ``'qe/phonon/tan.dyn'`` (``<prefix>0`` is the q-grid index).

    Returns
    -------
    list of str
        Existing files ``<prefix>1``, ``<prefix>2``, ... up to the first missing index.
    """
    paths, index = [], 1
    while os.path.isfile('%s%d' % (prefix, index)):
        paths.append('%s%d' % (prefix, index))
        index += 1
    return paths


def supercell_qpoint_files(harmonic_prefix: str, supercell: Sequence[int]) -> list[str]:
    """Dynamical-matrix files of a ``ph.x`` grid run whose q-points are commensurate with a supercell.

    Parameters
    ----------
    harmonic_prefix : str
        ``fildyn`` path prefix of the ``ph.x`` run (``<prefix>0`` must list the grid).
    supercell : sequence of int
        Diagonal supercell ``(n1, n2, n3)`` of the SSCHA.

    Returns
    -------
    list of str
        The files whose (first) q has crystal coordinates ``q . a_i`` multiple of ``1 / n_i``, Gamma first.

    Raises
    ------
    ValueError
        If no file is commensurate or the Gamma file is missing.

    Notes
    -----
    Each file holds a full star; with the stars of the commensurate q the selected files cover the supercell grid,
    the set CellConstructor reads for a supercell ``n1 x n2 x n3``.
    """
    from .qe_force_constants import read_dynamical_matrices, read_dynamical_matrix_file

    directory, stem = _fildyn_stem(harmonic_prefix)
    lattice = read_dynamical_matrices(stem, directory)['at']
    sc = np.asarray(supercell, dtype=float)
    gamma, others = [], []
    for path in dynamical_matrix_files(harmonic_prefix):
        q_cart = np.asarray(read_dynamical_matrix_file(path)['q_cart'][0], dtype=float)
        q_cryst = lattice @ q_cart
        if np.allclose(q_cryst * sc, np.round(q_cryst * sc), atol=1.0e-6):
            (gamma if np.allclose(q_cart, 0.0) else others).append(path)
    if not gamma:
        raise ValueError('no Gamma dynamical matrix in %s' % harmonic_prefix)
    return gamma + others


def _fildyn_stem(prefix: str) -> tuple[str, str]:
    """``(directory, stem)`` of a ``<stem>.dyn`` path prefix, as :func:`read_dynamical_matrices` expects."""
    directory, name = os.path.split(os.path.abspath(prefix))
    if not name.endswith('.dyn'):
        raise ValueError(
            'the dynamical-matrix prefix must end in .dyn (QE fildyn convention): %s' % prefix
        )
    return directory, name[:-4]


def starting_matrices(
    harmonic_prefix: str, supercell: Sequence[int], start_dir: str, positive_definite: bool = True
) -> tuple[Any, NDArray[np.float64], NDArray[np.float64]]:
    """Harmonic matrices of the supercell q-points as the SSCHA starting point.

    Parameters
    ----------
    harmonic_prefix : str
        ``fildyn`` path prefix of the ``ph.x`` grid run.
    supercell : sequence of int
        Diagonal supercell of the SSCHA.
    start_dir : str
        Directory receiving the copies ``dyn1 ... dynN`` of the selected files (created).
    positive_definite : bool, optional
        Make imaginary frequencies positive (CellConstructor ``ForcePositiveDefinite``), required to start from an
        unstable harmonic lattice.

    Returns
    -------
    matrices : cellconstructor.Phonons.Phonons
        Symmetrised (and positive-definite) starting matrices.
    harmonic_cm1 : ndarray
        Supercell frequencies of the harmonic matrices (cm^-1, imaginary as negative), sorted.
    start_cm1 : ndarray
        Supercell frequencies of the starting matrices (cm^-1), sorted.

    Notes
    -----
    Writes ``start_dir/dyn*``.
    """
    cc_phonons, _ = require_sscha()
    os.makedirs(start_dir, exist_ok=True)
    files = supercell_qpoint_files(harmonic_prefix, supercell)
    for index, path in enumerate(files, start=1):
        shutil.copy(path, os.path.join(start_dir, 'dyn%d' % index))
    matrices = cc_phonons.Phonons(os.path.join(start_dir, 'dyn'), len(files))
    harmonic_cm1 = np.sort(matrices.DiagonalizeSupercell()[0] * RY_TO_CM1)
    matrices.Symmetrize()
    if positive_definite:
        matrices.ForcePositiveDefinite()
    start_cm1 = np.sort(matrices.DiagonalizeSupercell()[0] * RY_TO_CM1)
    return matrices, harmonic_cm1, start_cm1


def configuration_input(
    data_controller,
    structure,
    prefix: str,
    pp_dir: str,
    kgrid: Sequence[int],
    conv_thr: float = 1.0e-10,
    mixing_beta: float = 0.3,
    electronic_temperature: float | None = None,
) -> str:
    """pw.x scf input (with forces) of one SSCHA configuration, written by PAOFLOW's supercell writer.

    Parameters
    ----------
    data_controller : DataController
        The parent run: occupations (``qe_smearing``, ``qe_degauss``), cutoffs, spin and pseudopotentials.
    structure : cellconstructor.Structure.Structure
        The displaced supercell (Angstrom, CellConstructor atom order).
    prefix : str
        pw.x ``prefix`` of the configuration (its ``outdir`` is ``./tmp_<prefix>``).
    pp_dir : str
        ``pseudo_dir``.
    kgrid : sequence of int
        Monkhorst-Pack grid of the supercell (Gamma-centred).
    conv_thr, mixing_beta : float, optional
        Self-consistency threshold (Ry) and mixing of the configuration.
    electronic_temperature : float, optional
        Fermi-Dirac electrons at this temperature (K) instead of the occupations of the parent run
        (hot electrons: the electronic temperature of the ensemble).

    Returns
    -------
    str
        The input text: ``ibrav = 0`` with ``CELL_PARAMETERS`` and ``ATOMIC_POSITIONS`` in Angstrom, in the atom
        order of the ensemble (the displacements are read back against it).
    """
    from .io import _atomic_species_block, _namelists, _pp_filenames, fermi_dirac_lines
    from .structure import _element_symbol

    symbols = [_element_symbol(str(name)) for name in structure.atoms]
    masses_amu = np.array([structure.masses[name] / AMU_RY for name in structure.atoms])

    class _Cell:  # the attributes the supercell writer reads
        def __init__(self) -> None:
            self.symbols, self.masses = symbols, masses_amu

        def __len__(self) -> int:
            return len(symbols)

    cell = _Cell()
    occupations = (
        None if electronic_temperature is None else fermi_dirac_lines(electronic_temperature)
    )
    text = _namelists(
        data_controller, cell, prefix, pp_dir, ibrav=0, conv_thr=conv_thr, mixing_beta=mixing_beta,
        occupations=occupations,
    )  # fmt: skip
    lines = [
        text,
        _atomic_species_block(cell, _pp_filenames(data_controller)),
        'CELL_PARAMETERS angstrom',
    ]
    lines += [
        '  %18.12f %18.12f %18.12f' % tuple(row)
        for row in np.asarray(structure.unit_cell, dtype=float)
    ]
    lines.append('ATOMIC_POSITIONS angstrom')
    lines += [
        '  %-3s %18.12f %18.12f %18.12f' % (sym, *xyz)
        for sym, xyz in zip(symbols, np.asarray(structure.coords))
    ]
    lines.append('K_POINTS automatic')
    lines.append('  %d %d %d 0 0 0' % tuple(int(n) for n in kgrid))
    return '\n'.join(lines) + '\n'


def generate_population(
    data_controller,
    matrices,
    temperature: float,
    size: int,
    population_dir: str,
    prefix: str,
    pp_dir: str,
    kgrid: Sequence[int],
    conv_thr: float = 1.0e-10,
    mixing_beta: float = 0.3,
    electronic_temperature: float | None = None,
) -> list[str]:
    """Configurations of one SSCHA population and their pw.x inputs.

    Parameters
    ----------
    data_controller : DataController
        The parent run (see :func:`configuration_input`).
    matrices : cellconstructor.Phonons.Phonons
        Current SSCHA (auxiliary) matrices: the Gaussian the configurations are drawn from.
    temperature : float
        Ionic temperature (K).
    size : int
        Number of configurations (even: python-sscha draws +/- pairs).
    population_dir : str
        Directory of the population (created; old inputs and outputs removed).
    prefix : str
        Base pw.x prefix; configuration ``i`` uses ``<prefix>_<i:04d>``.
    pp_dir : str
        ``pseudo_dir`` of the force runs.
    kgrid : sequence of int
        Supercell k-grid.
    conv_thr, mixing_beta : float, optional
        Self-consistency settings of the force runs.
    electronic_temperature : float, optional
        Fermi-Dirac electrons at this temperature (K) in the force runs (default: the parent's occupations).

    Returns
    -------
    list of str
        The input files ``population_dir/cfg_XXXX.in``.
    """
    _, sscha = require_sscha()
    if size % 2:
        size += 1
    ensemble = sscha.Ensemble.Ensemble(matrices, float(temperature), matrices.GetSupercell())
    ensemble.generate(int(size))
    os.makedirs(population_dir, exist_ok=True)
    for old in glob.glob(os.path.join(population_dir, 'cfg_*')):
        os.remove(old) if os.path.isfile(old) else shutil.rmtree(old)
    paths = []
    for index, structure in enumerate(ensemble.structures):
        path = os.path.join(population_dir, 'cfg_%04d.in' % index)
        with open(path, 'w') as handle:
            handle.write(configuration_input(
                data_controller, structure, '%s_%04d' % (prefix, index), pp_dir, kgrid, conv_thr, mixing_beta,
                electronic_temperature,
            ))  # fmt: skip
        paths.append(path)
    return paths


def completed_outputs(population_dir: str) -> tuple[list[str], int]:
    """pw.x outputs of a population that finished with forces.

    Parameters
    ----------
    population_dir : str
        Directory with ``cfg_XXXX.in`` and their outputs ``cfg_XXXX.out``.

    Returns
    -------
    outputs : list of str
        Outputs that contain ``JOB DONE`` and the total force.
    n_inputs : int
        Number of configurations of the population.
    """
    n_inputs = len(glob.glob(os.path.join(population_dir, 'cfg_*.in')))
    done = []
    for path in sorted(glob.glob(os.path.join(population_dir, 'cfg_*.out'))):
        if _finished(path):
            done.append(path)
    return done, n_inputs


def missing_configurations(population_dir: str) -> list[str]:
    """Inputs of a population whose pw.x output is absent or unfinished.

    Parameters
    ----------
    population_dir : str
        Directory with ``cfg_XXXX.in`` and their outputs ``cfg_XXXX.out``.

    Returns
    -------
    list of str
        The ``cfg_XXXX.in`` files still to run (sorted), what a launcher submits or a restart reruns.
    """
    return [
        path
        for path in sorted(glob.glob(os.path.join(population_dir, 'cfg_*.in')))
        if not _finished(path[:-3] + '.out')
    ]


def run_status(sscha_dir: str, min_fraction: float = 0.95) -> dict[str, Any]:
    """State of an SSCHA run read from its directory, and the next step.

    Parameters
    ----------
    sscha_dir : str
        Directory of the run (``state.json`` written by ``PAOFLOW.sscha_setup`` and the population folders).
    min_fraction : float, optional
        Fraction of finished configurations a population needs before it is minimised.

    Returns
    -------
    dict
        ``stage`` (``'generate'``, ``'forces'``, ``'minimize'``, ``'hessian'``, ``'export'``, ``'done'`` or
        ``'failed'``: not converged within ``population_sizes``), ``population``, ``inputs`` and ``missing`` (pw.x
        inputs of the pending population and those without a finished output), ``converged``, ``stable`` (``None``
        before the Hessian), ``exported``, ``temperature``, ``electronic_temperature``, ``supercell``,
        ``sscha_dir`` and ``history`` (``summary.json`` of every minimised population).

    Raises
    ------
    FileNotFoundError
        If ``sscha_dir`` has no ``state.json``.
    """
    import json

    with open(os.path.join(sscha_dir, 'state.json')) as handle:
        state = json.load(handle)
    generated, minimized = int(state['generated']), int(state['minimized'])
    inputs: list[str] = []
    missing: list[str] = []
    exported = list(state.get('exported', []))
    budget = len(state.get('population_sizes') or []) or np.inf
    if state.get('hessian_done'):  # the Hessian population is never "minimised": test this first
        stage = 'done' if {'auxiliary', 'hessian'} <= set(exported) else 'export'
    elif generated > minimized:
        population_dir = os.path.join(sscha_dir, 'pop%d' % generated)
        inputs = sorted(glob.glob(os.path.join(population_dir, 'cfg_*.in')))
        missing = missing_configurations(population_dir)
        if len(inputs) - len(missing) < min_fraction * max(len(inputs), 1):
            stage = 'forces'
        else:
            stage = 'hessian' if state.get('hessian_population') == generated else 'minimize'
    elif state.get('converged'):
        stage = 'generate'
    else:
        stage = 'failed' if minimized >= budget else 'generate'
    history = []
    for population in range(1, minimized + 1):
        path = os.path.join(sscha_dir, 'pop%d' % population, 'summary.json')
        if os.path.isfile(path):
            with open(path) as handle:
                history.append(json.load(handle))
    return dict(
        stage=stage, population=generated, inputs=inputs, missing=missing, converged=bool(state.get('converged')),
        stable=state.get('stable'), exported=exported, temperature=state.get('temperature'),
        electronic_temperature=state.get('electronic_temperature'), supercell=state.get('supercell'),
        sscha_dir=os.path.abspath(sscha_dir), history=history,
    )  # fmt: skip


def _finished(output: str) -> bool:
    """True when a pw.x output reached ``JOB DONE`` and printed the total force."""
    if not os.path.isfile(output):
        return False
    text = open(output, errors='ignore').read()
    return 'JOB DONE' in text and 'Total force' in text


def load_population(matrices, temperature: float, population_dir: str, min_fraction: float = 0.95):
    """Ensemble of a computed population, energies and forces read from the pw.x outputs.

    Parameters
    ----------
    matrices : cellconstructor.Phonons.Phonons
        The matrices the population was generated from.
    temperature : float
        Ionic temperature (K).
    population_dir : str
        Directory of the population.
    min_fraction : float, optional
        Smallest accepted fraction of finished configurations.

    Returns
    -------
    sscha.Ensemble.Ensemble
        The loaded ensemble (energies in Ry, forces in Ry / Angstrom).

    Raises
    ------
    RuntimeError
        If fewer than ``min_fraction`` of the configurations finished.

    Notes
    -----
    The finished outputs are linked into ``population_dir/computed/`` and read by python-sscha
    (``load_from_calculator_output``, ASE's pw.x parser); each output carries its own structure, so the order of the
    files is irrelevant.
    """
    _, sscha = require_sscha()
    outputs, n_inputs = completed_outputs(population_dir)
    if len(outputs) < min_fraction * max(n_inputs, 1):
        raise RuntimeError(
            '%s: %d of %d configurations finished' % (population_dir, len(outputs), n_inputs)
        )
    computed = os.path.join(population_dir, 'computed')
    shutil.rmtree(computed, ignore_errors=True)
    os.makedirs(computed)
    for path in outputs:
        os.symlink(
            os.path.abspath(path), os.path.join(computed, os.path.basename(path)[:-4] + '.pwo')
        )
    ensemble = sscha.Ensemble.Ensemble(matrices, float(temperature), matrices.GetSupercell())
    ensemble.load_from_calculator_output(computed, out_ext='.pwo')
    return ensemble


def minimise(
    ensemble,
    min_step_dyn: float = 0.05,
    kong_liu_ratio: float = 0.5,
    meaningful_factor: float = 1.0e-3,
) -> dict:
    """SSCHA minimisation of the free energy on one ensemble.

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        Computed population (:func:`load_population`).
    min_step_dyn : float, optional
        Step of the dynamical-matrix minimisation.
    kong_liu_ratio : float, optional
        Smallest effective-sample ratio before a new population is needed.
    meaningful_factor : float, optional
        Gradient threshold relative to its stochastic error.

    Returns
    -------
    dict
        ``matrices`` (CellConstructor Phonons), ``free_energy_ry``, ``free_energy_error_ry``, ``converged``,
        ``frequencies_cm1`` (supercell frequencies of the new matrices, sorted) and ``kong_liu`` (final
        effective-sample ratio :math:`N_\\mathrm{eff}/N`: a minimisation that stopped on ``kong_liu_ratio``
        needs a new population, one that stops on it every time needs larger populations or a smaller step).

    Notes
    -----
    The free energy is minimised with respect to the auxiliary force constants :math:`\\Phi` (the centroids are kept
    at the high-symmetry positions by symmetry),

    .. math::

        \\mathcal F[\\Phi] = \\langle V - \\mathcal V_\\Phi\\rangle_{\\rho_\\Phi} + F_\\Phi ,

    with the averages over the configurations reweighted from the population's Gaussian to the current one until the
    Kong-Liu effective sample size drops below ``kong_liu_ratio``.
    """
    _, sscha = require_sscha()
    minimizer = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
    minimizer.min_step_dyn = float(min_step_dyn)
    minimizer.kong_liu_ratio = float(kong_liu_ratio)
    minimizer.meaningful_factor = float(meaningful_factor)
    minimizer.init()
    minimizer.run(verbose=0)
    minimizer.finalize(verbose=0)
    free_energy, error = minimizer.get_free_energy(return_error=True)
    return dict(
        matrices=minimizer.dyn,
        free_energy_ry=float(free_energy),
        free_energy_error_ry=float(error),
        converged=bool(minimizer.is_converged()),
        frequencies_cm1=np.sort(minimizer.dyn.DiagonalizeSupercell()[0] * RY_TO_CM1),
        kong_liu=float(ensemble.get_effective_sample_size() / max(ensemble.N, 1)),
    )


def free_energy_hessian(ensemble, include_v4: bool = False):
    """Free-energy Hessian (the physical static phonons) of a converged ensemble.

    Parameters
    ----------
    ensemble : sscha.Ensemble.Ensemble
        Population generated from the converged matrices.
    include_v4 : bool, optional
        Include the fourth-order term (more expensive; the bubble term is usually enough).

    Returns
    -------
    cellconstructor.Phonons.Phonons
        Dynamical matrices of the Hessian; imaginary frequencies signal a dynamical instability at this temperature.

    Notes
    -----
    .. math::

        \\frac{\\partial^2 F}{\\partial R_a \\partial R_b} = \\Phi_{ab} + \\overset{(3)}{\\Phi}_{acd}\\,
        \\Lambda_{cdef}\\,[1 - \\overset{(4)}{\\Phi}\\Lambda]^{-1}_{efgh}\\,\\overset{(3)}{\\Phi}_{ghb},

    where :math:`\\Lambda` is built from the auxiliary frequencies and occupations; ``include_v4=False`` drops the
    fourth-order kernel.
    """
    require_sscha()
    return ensemble.get_free_energy_hessian(include_v4=bool(include_v4))


def lowest_frequencies(matrices) -> list[tuple[NDArray[np.float64], float]]:
    """Lowest mode at every q of a set of matrices (the three Gamma translations skipped).

    Parameters
    ----------
    matrices : cellconstructor.Phonons.Phonons
        Dynamical matrices on a q-grid.

    Returns
    -------
    list of (q, frequency)
        ``q`` as stored by CellConstructor (Cartesian, 2 pi / Angstrom) and the lowest frequency in cm^-1
        (imaginary as negative).
    """
    rows = []
    for iq, q in enumerate(matrices.q_tot):
        frequencies = np.sort(matrices.DyagDinQ(iq)[0] * RY_TO_CM1)
        rows.append(
            (
                np.asarray(q, dtype=float),
                float(frequencies[3] if np.allclose(q, 0.0) else frequencies[0]),
            )
        )
    return rows


def interpolate_correction(
    sscha_matrices,
    harmonic_coarse,
    harmonic_fine,
    supercell: Sequence[int],
    fine_grid: Sequence[int],
):
    """The anharmonic correction of the supercell grid added to harmonic matrices on a finer q-grid.

    Parameters
    ----------
    sscha_matrices : cellconstructor.Phonons.Phonons
        SSCHA (auxiliary) or free-energy-Hessian matrices on the supercell grid.
    harmonic_coarse : cellconstructor.Phonons.Phonons
        Harmonic matrices on the same supercell grid (the starting matrices before the positive-definite fix).
    harmonic_fine : cellconstructor.Phonons.Phonons
        Harmonic matrices on the fine grid (the ``ph.x`` run).
    supercell, fine_grid : sequence of int
        The two grids.

    Returns
    -------
    cellconstructor.Phonons.Phonons
        Matrices on the fine grid.

    Notes
    -----
    .. math::

        D(\\mathbf q) = D_{\\rm harm}(\\mathbf q) + \\mathcal I\\big[D_{\\rm SSCHA} - D_{\\rm harm}\\big](\\mathbf q),

    with :math:`\\mathcal I` the real-space Fourier interpolation from the supercell grid; exact at the supercell q,
    and the harmonic long-range part is kept on the fine grid.
    """
    require_sscha()
    return sscha_matrices.Interpolate(
        tuple(int(n) for n in supercell), tuple(int(n) for n in fine_grid),
        support_dyn_coarse=harmonic_coarse, support_dyn_fine=harmonic_fine,
    )  # fmt: skip


def write_force_constants(
    matrices, prefix: str, grid: Sequence[int], fc_path: str
) -> tuple[str, str]:
    """Write matrices as QE dynamical-matrix files (with the ``dyn0`` index) and real-space force constants.

    Parameters
    ----------
    matrices : cellconstructor.Phonons.Phonons
        Dynamical matrices on ``grid``.
    prefix : str
        Output prefix of the dynamical-matrix files: ``<prefix>0 ... <prefix>N``.
    grid : sequence of int
        Their q-grid.
    fc_path : str
        Output force-constant file (QE ``flfrc`` format, as ``q2r.x`` writes).

    Returns
    -------
    tuple of str
        ``(prefix, fc_path)``.

    Notes
    -----
    The real-space force constants come from PAOFLOW's own ``q2r`` (:mod:`PAOFLOW.phonon.qe_force_constants`), no
    QE executable is needed.
    """
    from .qe_force_constants import (
        read_dynamical_matrices,
        read_dynamical_matrix_file,
        real_space_force_constants,
        write_qe_force_constants,
    )

    _fildyn_stem(prefix)  # validate the name before writing
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    for old in glob.glob(prefix + '[0-9]*'):
        os.remove(old)
    matrices.save_qe(prefix)
    files = dynamical_matrix_files(prefix)
    with open(prefix + '0', 'w') as handle:
        handle.write('%4d%4d%4d\n%4d\n' % (*tuple(int(n) for n in grid), len(files)))
        for path in files:
            handle.write(
                '  %22.15E  %22.15E  %22.15E\n'
                % tuple(read_dynamical_matrix_file(path)['q_cart'][0])
            )
    directory, stem = _fildyn_stem(prefix)
    dynamical = read_dynamical_matrices(stem, directory)
    write_qe_force_constants(
        fc_path, dynamical, real_space_force_constants(dynamical['force_constants_q'])
    )
    return prefix, fc_path
