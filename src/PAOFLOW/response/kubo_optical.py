r"""Frequency-dependent (optical) Kubo response on the interpolated k-grid.

Driver of :meth:`PAOFLOW.kubo_optical_response`: the Kubo-Greenwood response of any observable of
:mod:`.kubo_operators` to an electric field of frequency :math:`\omega` at one chemical potential,

.. math::

    \chi_{AB}(\omega) = s_q\,\Big[-i\sum_{k}\sum_{\varepsilon_{mn}\ne0}
        \frac{(f_n-f_m)\,A_{nm}(\hbar v_b)_{mn}}{\varepsilon_{mn}(\varepsilon_{mn}-\hbar\omega-i\eta)}
        + \frac{i}{\hbar\omega+2i\eta}\sum_k\sum_{\varepsilon_{mn}=0}(-f'_n)\,A_{nm}(\hbar v_b)_{mn}\Big],

interband transitions (length gauge) plus the Drude term of the degenerate blocks, with the unit
factors and the charge sign :math:`s_q` of :func:`.kubo_operators.unit_factor` (the convention of
:meth:`PAOFLOW.kubo_response`). At :math:`\omega\to0`, :math:`\eta\to0` the interband Hall part is the
``intrinsic`` term and the intraband part the Boltzmann limit of ``surface_intraband``.

The k-points are streamed in blocks (band matrices from :func:`.kubo_operators.band_matrices`), the
sums reduced over MPI at the end; memory is independent of the number of k-points.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from ..utils.smearing import gaussian, intgaussian, intmetpax, metpax
from . import kubo_kernels as kernels
from .kubo_bastin import BLOCK_ELEMENTS, CONVENTION, K_BOLTZMANN_EV
from .kubo_operators import (
    OBSERVABLES,
    band_matrices,
    component_label,
    parse_component,
    unit_factor,
)

if TYPE_CHECKING:
    from ..DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#: Width (eV) that replaces an adaptive smearing width of exactly zero in the occupations.
MIN_WIDTH = 1.0e-8

OPTICAL_TERMS = ('total', 'interband', 'intraband')


def band_occupations(
    data_controller: 'DataController',
    energies: NDArray[np.float64],
    widths: NDArray[np.float64] | None,
    mu: float,
    kT: float,
    eta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Occupations :math:`f_n` and Fermi window :math:`-f'_n` of a block of bands.

    Parameters
    ----------
    data_controller : DataController
        Reads the smearing of the run.
    energies : ndarray
        Band energies (eV).
    widths : ndarray or None
        Adaptive smearing widths ``deltakp`` of the same bands.
    mu : float
        Chemical potential (eV).
    kT : float
        Temperature (eV); > 0 gives Fermi-Dirac.
    eta : float
        Broadening (eV): the Lorentzian Fermi window when the run has no smearing widths.

    Returns
    -------
    occupations, fermi_window : ndarray
        Same shape as ``energies``.
    """
    smearing = data_controller.data_attributes.get('smearing') if widths is not None else None
    if widths is not None:
        # a band with zero velocity (e.g. a flat or decoupled band) has zero adaptive width: 0/0 otherwise
        widths = np.where(widths > 0.0, widths, MIN_WIDTH)
    if kT > 0.0:
        occupation = 0.5 * (1.0 - np.tanh(0.5 * (energies - mu) / kT))
        return occupation, occupation * (1.0 - occupation) / kT
    if smearing == 'gauss':
        return intgaussian(energies, mu, widths), gaussian(energies, mu, widths)
    if smearing == 'm-p':
        return intmetpax(energies, mu, widths), metpax(energies, mu, widths)
    occupation = 0.5 * (1.0 - np.sign(energies - mu))
    return occupation, (eta / np.pi) / ((energies - mu) ** 2 + eta**2)


def kubo_optical_response(
    data_controller: 'DataController',
    observable: str,
    components: Sequence[str | Sequence[int]],
    omegas: NDArray[np.float64],
    mu: float,
    eta: float,
    terms: Sequence[str] = OPTICAL_TERMS,
    temperature: float = 0.0,
    degenerate_threshold: float = 1.0e-4,
    projector: NDArray[np.complex128] | None = None,
    operator: NDArray[np.complex128] | None = None,
    two_dimensional: bool = False,
) -> dict[str, dict[str, NDArray]]:
    """Optical response of ``observable`` to an electric field, every term and component.

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k``, ``v_k``, ``dHksp``, ``degen``, ``deltakp`` and ``Sj``/``Lj``.
    observable : str
        Key of :data:`.kubo_operators.OBSERVABLES`.
    components : sequence
        Components, e.g. ``['xx', 'xy']`` (see :func:`.kubo_operators.parse_component`).
    omegas : ndarray, shape ``(nw,)``
        Photon energies (eV).
    mu : float
        Chemical potential (eV, the energy zero of the run).
    eta : float
        Broadening (eV), > 0.
    terms : sequence of str
        Subset of :data:`OPTICAL_TERMS`.
    temperature : float
        Electronic temperature (K); 0 uses the run's smearing.
    degenerate_threshold : float
        Energy below which two bands count as degenerate (eV).
    projector, operator : ndarray, optional
        Orbital projector of the response operator; matrix of ``observable='custom'``.
    two_dimensional : bool
        Sheet values (e^2/hbar) for the currents.

    Returns
    -------
    dict
        ``{label: {'omega': omegas, term: complex values}}`` on every rank, in the unit of the
        observable.

    Raises
    ------
    ValueError
        Unknown term, non-positive ``eta`` or a spin observable without a spinor basis.
    """
    arry, attr = data_controller.data_dicts()
    unknown = set(terms) - set(OPTICAL_TERMS)
    if unknown:
        raise ValueError(f'unknown terms {sorted(unknown)}; choose from {OPTICAL_TERMS}')
    if eta <= 0.0:
        raise ValueError('eta must be positive')
    if OBSERVABLES[observable].base == 'Sj' and not attr['dftSO']:
        raise ValueError(f'{observable} needs a spinor basis (spin-orbit or adhoc_SO)')
    omegas = np.asarray(omegas, dtype=float)
    kT = K_BOLTZMANN_EV * temperature
    nk_local, nb, nspin = arry['E_k'].shape
    spin_multiplicity = 1.0 if (attr['dftSO'] or nspin == 2) else 2.0
    scale = spin_multiplicity * unit_factor(data_controller, observable, two_dimensional)
    scale /= float(attr['nkpnts'])
    block = max(1, BLOCK_ELEMENTS // (nb * nb))

    results = {}
    for component in components:
        indices = parse_component(observable, component)
        interband = np.zeros(omegas.size, dtype=complex)
        intraband = np.zeros(omegas.size, dtype=complex)
        for ispin in range(nspin):
            for start in range(0, nk_local, block):
                stop = min(start + block, nk_local)
                pairs = np.empty((stop - start, nb, nb), dtype=complex)
                for i, ik in enumerate(range(start, stop)):
                    response, field = band_matrices(
                        data_controller, observable, indices, ik, ispin, projector, operator
                    )
                    pairs[i] = response * field.T
                energies = arry['E_k'][start:stop, :, ispin]
                widths = arry['deltakp'][start:stop, :, ispin] if 'deltakp' in arry else None
                occupation, window = band_occupations(
                    data_controller, energies, widths, mu, kT, eta
                )
                inter, intra = kernels.optical_kernel(
                    energies, pairs, occupation, window, omegas, eta, degenerate_threshold
                )
                interband += inter
                intraband += intra
        summed = {}
        for name, values in (('interband', interband), ('intraband', intraband)):
            real, imag = np.zeros(omegas.size), np.zeros(omegas.size)
            comm.Allreduce(np.ascontiguousarray(values.real), real, op=MPI.SUM)
            comm.Allreduce(np.ascontiguousarray(values.imag), imag, op=MPI.SUM)
            summed[name] = scale * (real + 1j * imag)
        summed['total'] = summed['interband'] + summed['intraband']
        label = component_label(observable, indices)
        results[label] = {'omega': omegas.copy()}
        results[label].update({name: summed[name] for name in terms})
    return results


def write_kubo_optical_response(
    data_controller: 'DataController',
    observable: str,
    results: dict[str, dict[str, NDArray]],
    two_dimensional: bool,
    header: str = '',
) -> list[str]:
    """Write ``kubo_optical_<observable>_<label>.dat``: omega, then Re and Im of every term.

    Parameters
    ----------
    data_controller : DataController
        Output directory ``opath``.
    observable : str
    results : dict
        From :func:`kubo_optical_response`.
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
        terms = [k for k in data if k != 'omega']
        fname = f'kubo_optical_{observable}_{label}.dat'
        names.append(fname)
        if rank == 0:
            columns = [data['omega']]
            titles = ['omega (eV)']
            for term in terms:
                columns += [data[term].real, data[term].imag]
                titles += [f'Re {term}', f'Im {term}']
            lines = [f'Kubo-Greenwood {observable} {label}, unit {unit}', CONVENTION]
            if header:
                lines.append(header)
            lines.append('  '.join(titles))
            np.savetxt(
                os.path.join(attr['opath'], fname),
                np.column_stack(columns),
                fmt='%.10e',
                header='\n'.join(lines),
            )
    comm.Barrier()
    return names
