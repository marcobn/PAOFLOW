r"""Kubo-Bastin linear response on the interpolated k-grid (driver of :meth:`PAOFLOW.kubo_response`).

For every requested component the band-basis matrices of the response and field operators
(:mod:`.kubo_operators`) are built one k-point at a time and contracted with the kernels of
:mod:`.kubo_kernels`; each rank keeps only the sums over its own k-points (one array of chemical
potentials per term), reduced over MPI at the end. Memory: no ``(nk, nb, nb)`` array is formed.

Terms
-----
``total``
    The full Kubo-Bastin response at constant broadening, ``surface + sea``.
``surface``
    Fermi-surface term (Bonbien-Manchon :math:`\tilde\sigma^{I}`), symmetric part of
    :math:`A_{nm}B_{mn}`; tutorial03 Eq. 1 at :math:`T = 0`.
``sea``
    Fermi-sea term (Bonbien-Manchon :math:`\tilde\sigma^{II}`), antisymmetric part.
``surface_intraband``, ``surface_interband``
    The :math:`n = m` and :math:`n \ne m` parts of ``surface``.
``intrinsic``
    The :math:`\eta\to0` limit of ``sea``: :math:`2\sum_n f_n\sum_m\mathrm{Im}[A_{nm}B_{mn}]/
    (\varepsilon_{nm}^2+\delta^2)` with the run's smearing (``T = 0``) or Fermi-Dirac
    occupations; minus the Berry-curvature sums of ``do_Hall`` and tutorial03 Eq. 3.

Sign convention: the Kubo-Bastin formula :math:`K(A, \\hbar v)` with :math:`\\hbar v = \\partial H/\\partial k`
and the electron charge :math:`q = -|e|`: :math:`\\sigma = q^2K` for the charge current (Yao et al. 2004),
:math:`(q/|e|)K` for every other observable (Guo et al. 2008). A magnet's response is that of the computed
spin configuration; see ``docs/internals/kubo-bastin.md`` for the older routes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from ..utils.smearing import intgaussian, intmetpax
from . import kubo_kernels as kernels
from .kubo_operators import (
    OBSERVABLES,
    band_matrices,
    component_label,
    parse_component,
    physical_bands,
    unit_factor,
)

if TYPE_CHECKING:
    from ..DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#: Boltzmann constant (eV/K).
K_BOLTZMANN_EV = 8.617333262e-5

TERMS = ('total', 'surface', 'sea', 'surface_intraband', 'surface_interband', 'intrinsic')

#: The sign convention, one header line of every kubo_* output file.
CONVENTION = (
    'sign: Kubo formula with the electron charge q = -|e| (q^2 for the charge current; q for all other '
    'observables); magnets: the computed spin configuration (QE starting_magnetization > 0 = spins '
    'along +z = moment along -z)'
)

#: Below kT = THERMAL_THRESHOLD * eta the exact terms are evaluated at T = 0.
THERMAL_THRESHOLD = 0.01

#: k-points per kernel call: block * nmu * nb and block * nb^2 stay below this (16 bytes each,
#: a few such arrays at a time).
BLOCK_ELEMENTS = 1_000_000
_EXACT = {'total', 'surface', 'sea', 'surface_intraband', 'surface_interband'}


def occupation_sum(
    energies: NDArray[np.float64],
    widths: NDArray[np.float64] | None,
    omega: NDArray[np.float64],
    mu: NDArray[np.float64],
    kT: float,
    smearing: str | None,
) -> NDArray[np.float64]:
    """:math:`\\sum_{k,n} f_{nk}(\\mu)\\,\\Omega_{nk}` over a block of k-points, every ``mu``.

    Parameters
    ----------
    energies, omega : ndarray, shape ``(nk, nb)``
        Band energies (eV) and the band-resolved quantity.
    widths : ndarray, shape ``(nk, nb)``, or None
        Smearing widths (eV) of the same bands; read at ``kT = 0`` with a smearing.
    mu : ndarray, shape ``(nmu,)``
    kT : float
        Temperature (eV); 0 uses ``smearing``.
    smearing : {'gauss', 'm-p', None}
        Occupations at ``kT = 0``: Gaussian or Methfessel-Paxton with ``widths``, a step otherwise.

    Returns
    -------
    ndarray, shape ``(nmu,)``
    """
    total = np.zeros(mu.size)
    if widths is not None:
        # an adaptive width of exactly zero (a band with zero velocity) would give 0/0
        widths = np.where(widths > 0.0, widths, 1.0e-8)
    for i, m in enumerate(mu):
        if kT > 0.0:
            occupation = 0.5 * (1.0 - np.tanh(0.5 * (energies - m) / kT))
        elif smearing == 'gauss':
            occupation = intgaussian(energies, m, widths)
        elif smearing == 'm-p':
            occupation = intmetpax(energies, m, widths)
        else:
            occupation = 0.5 * (1.0 - np.sign(energies - m))
        total[i] = np.sum(occupation * omega)
    return total


def _occupation_sum(
    data_controller: 'DataController', omega: NDArray, mu: NDArray, kT: float
) -> NDArray[np.float64]:
    """:func:`occupation_sum` over the local k-points and spins.

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k`` and ``deltakp`` (smearing widths at ``T = 0``).
    omega : ndarray, shape ``(nk_local, nb, nspin)``
        Band-resolved quantity.
    mu : ndarray, shape ``(nmu,)``
    kT : float
        Temperature (eV); 0 uses the run's smearing (Gaussian or Methfessel-Paxton with the adaptive
        widths, a step otherwise).

    Returns
    -------
    ndarray, shape ``(nmu,)``
    """
    arry, attr = data_controller.data_dicts()
    total = np.zeros(mu.size)
    smearing = attr.get('smearing') if 'deltakp' in arry else None
    for ispin in range(omega.shape[2]):
        nb = omega.shape[1]
        energies = arry['E_k'][:, :nb, ispin]
        widths = arry['deltakp'][:, :nb, ispin] if smearing else None
        total += occupation_sum(energies, widths, omega[:, :, ispin], mu, kT, smearing)
    return total


def bastin_setup(
    data_controller: 'DataController',
    observable: str,
    mu: NDArray[np.float64],
    eta: float,
    terms: Sequence[str],
    temperature: float,
    two_dimensional: bool,
    nspin: int,
) -> tuple[NDArray[np.float64], float, bool, bool, float]:
    """Checks, energy grid and unit factor of :func:`kubo_bastin_response`.

    Parameters
    ----------
    data_controller : DataController
        Reads ``dftSO``, ``nkpnts`` and the cell attributes.
    observable, mu, eta, terms, temperature, two_dimensional
        As :func:`kubo_bastin_response`.
    nspin : int
        Number of spin channels.

    Returns
    -------
    grid : ndarray
        Energies the exact kernels are evaluated on: ``mu``, or the grid of the thermal average.
    kT : float
        Temperature (eV).
    exact : bool
        Whether any exact (constant-broadening) term is asked for.
    thermal : bool
        Whether the exact terms are thermally averaged afterwards.
    scale : float
        Unit factor, spin multiplicity and ``1 / nkpnts``.

    Raises
    ------
    ValueError
        Unknown term, non-positive ``eta`` or a spin observable without a spinor basis.
    """
    attr = data_controller.data_attributes
    unknown = set(terms) - set(TERMS)
    if unknown:
        raise ValueError(f'unknown terms {sorted(unknown)}; choose from {TERMS}')
    if eta <= 0.0:
        raise ValueError('eta must be positive')
    spec = OBSERVABLES[observable]
    if spec.base == 'Sj' and not attr['dftSO']:
        raise ValueError(f'{observable} needs a spinor basis (spin-orbit or adhoc_SO)')
    kT = K_BOLTZMANN_EV * temperature
    exact = bool(_EXACT & set(terms))
    # kT <= eta/100: the thermal smearing of the exact terms is a Sommerfeld correction of relative
    # size ~ (pi kT / eta)^2 / 6 < 2e-4, below the k-grid error; the T = 0 kernels are used.
    thermal = exact and kT > THERMAL_THRESHOLD * eta
    grid = kernels.thermal_grid(mu, eta, kT) if thermal else mu
    spin_multiplicity = 1.0 if (attr['dftSO'] or nspin == 2) else 2.0
    scale = spin_multiplicity * unit_factor(data_controller, observable, two_dimensional)
    scale /= float(attr['nkpnts'])
    return grid, kT, exact, thermal, scale


def bastin_finish(
    local: dict[str, NDArray[np.float64]],
    grid: NDArray[np.float64],
    mu: NDArray[np.float64],
    kT: float,
    thermal: bool,
    terms: Sequence[str],
    scale: float,
) -> dict[str, NDArray[np.float64]]:
    """Reduce the local k-sums of one component over MPI and assemble the terms.

    Parameters
    ----------
    local : dict
        This rank's sums ``'surface'``, ``'intraband'``, ``'sea'`` (on ``grid``) and
        ``'intrinsic'`` (on ``mu``).
    grid, mu, kT, thermal, scale
        From :func:`bastin_setup`.
    terms : sequence of str
        Subset of :data:`TERMS`.

    Returns
    -------
    dict
        ``{'mu': mu, term: values, ...}`` on every rank, in the unit of the observable.
    """
    summed = {}
    for name, values in local.items():
        total = np.zeros_like(values)
        comm.Allreduce(np.ascontiguousarray(values), total, op=MPI.SUM)
        summed[name] = total
    if thermal:
        for name in ('surface', 'intraband', 'sea'):
            summed[name] = kernels.thermal_average(grid, summed[name], mu, kT)
    available = {
        'surface': summed['surface'],
        'sea': summed['sea'],
        'total': summed['surface'] + summed['sea'],
        'surface_intraband': summed['intraband'],
        'surface_interband': summed['surface'] - summed['intraband'],
        'intrinsic': summed['intrinsic'],
    }
    result = {'mu': mu.copy()}
    result.update({name: scale * available[name] for name in terms})
    return result


def kubo_bastin_response(
    data_controller: 'DataController',
    observable: str,
    components: Sequence[str | Sequence[int]],
    mu: NDArray[np.float64],
    eta: float,
    terms: Sequence[str] = ('total', 'surface', 'sea'),
    temperature: float = 0.0,
    regulariser: float = 0.0,
    projector: NDArray[np.complex128] | None = None,
    operator: NDArray[np.complex128] | None = None,
    two_dimensional: bool = False,
    bands: int | None = None,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Kubo-Bastin response of ``observable`` to an electric field, every term and component.

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k``, ``v_k``, ``dHksp``, ``degen``, ``deltakp`` (intrinsic term at ``T = 0``),
        ``Sj``/``Lj`` and the cell attributes.
    observable : str
        Key of :data:`.kubo_operators.OBSERVABLES`.
    components : sequence
        Components, e.g. ``['xy']`` or ``[(2, 0, 1)]`` (see :func:`.kubo_operators.parse_component`).
    mu : ndarray, shape ``(nmu,)``
        Chemical potentials (eV, the energy zero of the run).
    eta : float
        Constant broadening :math:`\\eta = \\hbar/2\\tau` (eV), > 0.
    terms : sequence of str
        Subset of :data:`TERMS`.
    temperature : float
        Electronic temperature (K). At 0 the exact kernels use a sharp Fermi level and the intrinsic
        term the run's smearing; below :math:`k_BT = 0.01\\eta` the exact kernels stay at
        :math:`T = 0` (Sommerfeld correction < 2e-4).
    regulariser : float
        :math:`\\delta` of the intrinsic term (eV).
    projector : ndarray, shape ``(nawf, nawf)``, optional
        Orbital projector of the response operator.
    operator : ndarray, shape ``(nawf, nawf)``, optional
        Matrix of ``observable='custom'``.
    two_dimensional : bool
        Sheet values (e^2/hbar) for the currents.
    bands : int, optional
        Number of bands in the sums (:func:`.kubo_operators.physical_bands`; default the bands below the
        projectability shift of a DFT run, every band of a model).

    Returns
    -------
    dict
        ``{label: {'mu': mu, term: values, ...}}`` on every rank, in the unit of the observable.

    Raises
    ------
    ValueError
        Unknown term, non-positive ``eta`` or a spin observable without a spinor basis.
    """
    arry, attr = data_controller.data_dicts()
    mu = np.asarray(mu, dtype=float)
    nk_local, _, nspin = arry['E_k'].shape
    grid, kT, exact, thermal, scale = bastin_setup(
        data_controller, observable, mu, eta, terms, temperature, two_dimensional, nspin
    )

    results = {}
    for component in components:
        indices = parse_component(observable, component)
        local = {name: np.zeros(grid.size) for name in ('surface', 'intraband', 'sea')}
        nb = physical_bands(data_controller, bands)
        omega = np.zeros((nk_local, nb, nspin)) if 'intrinsic' in terms else None
        block = max(1, min(BLOCK_ELEMENTS // (grid.size * nb), BLOCK_ELEMENTS // (nb * nb)))
        for ispin in range(nspin):
            for start in range(0, nk_local, block):
                ks = range(start, min(start + block, nk_local))
                pairs = np.empty((len(ks), nb, nb), dtype=complex)
                for i, ik in enumerate(ks):
                    response, field = band_matrices(
                        data_controller, observable, indices, ik, ispin, projector, operator
                    )
                    response, field = response[:nb, :nb], field[:nb, :nb]
                    pairs[i] = response * field.T
                    if omega is not None:
                        omega[ik, :, ispin] = -kernels.berry_curvature(
                            arry['E_k'][ik, :nb, ispin], response, field, regulariser, 1.0e-12
                        )
                if exact:
                    energies = arry['E_k'][start : start + len(ks), :nb, ispin]
                    surface, intraband = kernels.surface_kernel(energies, pairs, grid, eta)
                    local['surface'] += surface
                    local['intraband'] += intraband
                    local['sea'] += kernels.sea_kernel(energies, pairs, grid, eta)
        local['intrinsic'] = (
            _occupation_sum(data_controller, omega, mu, kT)
            if omega is not None
            else np.zeros(mu.size)
        )
        label = component_label(observable, indices)
        results[label] = bastin_finish(local, grid, mu, kT, thermal, terms, scale)
    return results


def write_kubo_response(
    data_controller: 'DataController',
    observable: str,
    results: dict[str, dict[str, NDArray[np.float64]]],
    two_dimensional: bool,
    header: str = '',
) -> list[str]:
    """Write one file per component, ``kubo_<observable>_<label>.dat``: mu and one column per term.

    Parameters
    ----------
    data_controller : DataController
        Output directory ``opath``.
    observable : str
    results : dict
        From :func:`kubo_bastin_response`.
    two_dimensional : bool
        Selects the unit written in the header.
    header : str
        Extra header line (settings).

    Returns
    -------
    list of str
        The file names (written by rank 0).
    """
    attr = data_controller.data_attributes
    spec = OBSERVABLES[observable]
    unit = spec.sheet_unit if two_dimensional else spec.unit
    names = []
    for label, data in results.items():
        terms = [k for k in data if k != 'mu']
        fname = f'kubo_{observable}_{label}.dat'
        names.append(fname)
        if rank == 0:
            table = np.column_stack([data['mu']] + [data[t] for t in terms])
            lines = [f'Kubo-Bastin {observable} {label}, unit {unit}', CONVENTION]
            if header:
                lines.append(header)
            lines.append('  '.join(['mu (eV)'] + terms))
            np.savetxt(
                os.path.join(attr['opath'], fname), table, fmt='%.10e', header='\n'.join(lines)
            )
    comm.Barrier()
    return names
