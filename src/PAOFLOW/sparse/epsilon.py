from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from ..smearing import gaussian, intgaussian, intmetpax, metpax
from .operators import iter_projected_operators

if TYPE_CHECKING:
    from ..DataController import DataController


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def use_dielectric_tensor(
    data_controller: DataController,
    from_wfc: str | None,
) -> bool:
    """Decide whether the dielectric tensor can be evaluated in sparse mode.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with the arrays and attributes of the active run.
    from_wfc : str | None
        Dipole-evaluation mode requested by ``PAOFLOW.dielectric_tensor``.

    Returns
    -------
    bool
        ``True`` when the sparse no-bridge dielectric workflow is physically and
        algorithmically valid.

    Notes
    -----
    The sparse dielectric path is currently valid only when dipole matrix
    elements are obtained from Hamiltonian derivatives, that is,
    ``from_wfc=None``. The wavefunction-based modes still rely on dense
    momentum-like tensors, so they remain on the dense implementation.
    """
    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None
    return (
        bool(attributes.get('sparse', False))
        and from_wfc is None
        and 'SparseHRs' in arrays
        and 'Hksp' not in arrays
        and 'pksp' not in arrays
    )


def _dielectric_prefactor(attributes: dict, from_wfc: str | None) -> float:
    """Return the global prefactor multiplying the dielectric spectrum.

    Parameters
    ----------
    attributes : dict
        Runtime scalar attributes for the active calculation.
    from_wfc : str | None
        Dipole-evaluation mode.

    Returns
    -------
    float
        Overall prefactor applied after the k-point sums have been accumulated.

    Notes
    -----
    The sparse routine uses the same physical normalization as the dense one.
    Only the construction of the dipole matrix elements changes; the final
    dielectric formula and its prefactor are kept identical.
    """
    from ..constants import BOHR_RADIUS_ANGS, ELECTRONVOLT_SI, RYTOEV

    if from_wfc is None:
        return (
            ELECTRONVOLT_SI
            * (1e10)
            / (8.8541878188e-12)
            * BOHR_RADIUS_ANGS**2
            / attributes['nkpnts']
            / (attributes['omega'] * BOHR_RADIUS_ANGS**3)
        )
    if from_wfc == 'external' or from_wfc == 'internal':
        return (
            2
            * (np.pi / attributes['alat']) ** 2
            * RYTOEV**3
            * 64.0
            * np.pi
            / (attributes['omega'] * attributes['nkpnts'])
        )
    raise Exception('ERROR: no dipole mode specified')


def _prepare_occupations(
    arrays: dict,
    attributes: dict,
    ispin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, float, float, int, int]:
    """Prepare occupations and thresholds for one spin channel.

    Parameters
    ----------
    arrays : dict
        Runtime arrays containing eigenvalues and, when present, adaptive
        widths.
    attributes : dict
        Runtime scalar attributes controlling smearing and occupations.
    ispin : int
        Spin-channel index.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None, float, float, float, int, int]
        ``(Ek, fn, fnf, th0, th1, fac, bndmax, spin_factor)`` for the local
        k-point slice of the chosen spin channel.

    Notes
    -----
    The dielectric response depends on band occupations and, for metallic terms,
    on the derivative of the occupation function. This helper preserves the same
    occupation model as the dense implementation, including adaptive widths when
    they are available. The sparse path changes only how the matrix elements are
    supplied, not how occupations are evaluated.
    """
    bndmax = int(attributes['bnd'])
    ek = np.ascontiguousarray(arrays['E_k'][:, :bndmax, ispin])

    smearing = attributes.get('smearing')
    degauss: float | np.ndarray = attributes['degauss']
    spin_factor = 2 if int(attributes['nspin']) == 1 else 1
    ef = 1.0e-9

    if smearing is None or attributes['insulator']:
        fn = spin_factor * (ek <= ef)
    else:
        if 'deltakp' in arrays:
            degauss = np.ascontiguousarray(arrays['deltakp'][:, :bndmax, ispin])

        if smearing == 'gauss':
            fn = spin_factor * intgaussian(ek, ef, degauss)
        else:
            fn = spin_factor * intmetpax(ek, ef, degauss)

    if attributes['insulator']:
        fnf = None
    elif smearing == 'gauss':
        fnf = spin_factor * gaussian(ek, ef, degauss)
    elif smearing == 'm-p':
        fnf = spin_factor * metpax(ek, ef, degauss)
    else:
        fnf = spin_factor * gaussian(ek, ef, degauss)

    th0 = 1.0e-3 * spin_factor
    th1 = 0.5e-4 * spin_factor
    fac = 1 if attributes['dftSO'] else 2
    return (ek, fn, fnf, th0, th1, fac, bndmax, spin_factor)


def _compute_inverse_epsilon(ene: np.ndarray, epsi: np.ndarray) -> np.ndarray:
    """Build the integrated inverse dielectric spectrum.

    Parameters
    ----------
    ene : numpy.ndarray
        Energy mesh, shape ``(nenergy,)``.
    epsi : numpy.ndarray
        Imaginary part of the dielectric function on the same mesh.

    Returns
    -------
    numpy.ndarray
        Integrated inverse dielectric spectrum on the same mesh.

    Notes
    -----
    This quantity is obtained from the same discrete energy integral used in the
    legacy dense code. The sparse implementation keeps that numerical convention
    unchanged so only the Hamiltonian-derivative handling, not the spectral
    post-processing, differs from the dense workflow.
    """
    ieps = np.zeros_like(epsi)
    if ene.size > 3:
        delta_e = ene[3] - ene[2]
    elif ene.size > 1:
        delta_e = ene[1] - ene[0]
    else:
        delta_e = 0.0

    for i in range(ene.size):
        ieps[i] = (
            1.0
            + (2.0 / np.pi) * np.sum(ene[1:] * epsi[1:] / (ene[i] ** 2 + ene[1:] ** 2)) * delta_e
        )

    return ieps


def do_dielectric_tensor(
    data_controller: DataController,
    ene: np.ndarray,
    from_wfc: str | None,
) -> None:
    """Compute the dielectric spectrum from streamed sparse derivatives.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse no-bridge eigendata and sparse ``H(R)``.
    ene : numpy.ndarray
        Energy mesh on which the dielectric spectra are accumulated.
    from_wfc : str | None
        Dipole-evaluation mode. Only ``None`` is supported here.

    Returns
    -------
    None
        Writes the same dielectric output files as the dense path.

    Notes
    -----
    The dielectric tensor depends on band-to-band matrix elements of the
    Hamiltonian derivatives. In dense mode these are often stored as the full
    tensor ``pksp(k, l, n, m, s)``, which can dominate memory use. This sparse
    version never stores that tensor. Instead it evaluates the needed projected
    matrices for one small k-point batch at a time, accumulates the spectral
    contribution immediately, and discards the temporary matrices.

    Parallelization strategy:
        Each rank processes only its own local k-point window, matching the
        distributed eigenvalue and eigenvector layout. Only the final compact
        spectral arrays are reduced with MPI.
    """
    from ..constants import LL

    if from_wfc is not None:
        raise NotImplementedError(
            'Sparse dielectric_tensor currently supports only from_wfc=None. '
            'Wavefunction-based dipole modes still use the dense pksp path.'
        )

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None
    if not use_dielectric_tensor(data_controller, from_wfc):
        raise RuntimeError(
            'Sparse dielectric_tensor requires sparse no-bridge input: SparseHRs '
            'must exist, dense Hksp must be absent, and from_wfc must be None.'
        )

    energies = np.array(ene, dtype=float, copy=True)
    if energies.size > 0 and energies[0] == 0.0:
        energies[0] = 0.00001

    d_tensor = np.asarray(arrays['d_tensor'], dtype=int)
    ncomponents = int(d_tensor.shape[0])
    nspin = int(attributes['nspin'])
    esize = int(energies.size)
    intersmear = float(attributes['delta'])
    intrasmear = float(attributes['intrasmear'])
    factor = _dielectric_prefactor(attributes, from_wfc)
    requested_directions = np.unique(d_tensor.reshape(-1)).astype(int)

    occupation_data = [_prepare_occupations(arrays, attributes, ispin) for ispin in range(nspin)]

    epsi_local = np.zeros((ncomponents, nspin, esize), dtype=float)
    epsr_local = np.zeros_like(epsi_local)

    for ik_local, ispin, momentum_by_direction in iter_projected_operators(
        data_controller,
        requested_directions,
        band_count=int(attributes['bnd']),
    ):
        ek_spin, fn_spin, fnf_spin, th0, th1, fac, bndmax, spin_factor = occupation_data[ispin]
        ek_row = ek_spin[ik_local]
        fn_row = fn_spin[ik_local]

        for component_index, (ipol, jpol) in enumerate(d_tensor):
            momentum_i = momentum_by_direction[int(ipol)]
            momentum_j = momentum_by_direction[int(jpol)]

            for iband2 in range(bndmax):
                for iband1 in range(bndmax):
                    if iband1 == iband2:
                        continue

                    e_diff_nm = ek_row[iband2] - ek_row[iband1]
                    f_nm = fn_row[iband2] - fn_row[iband1]
                    if np.abs(f_nm) > th0 and fn_row[iband1] > th1 and fn_row[iband2] < spin_factor:
                        pksp2 = np.real(momentum_i[iband1, iband2] * momentum_j[iband2, iband1])
                        denominator = (
                            (e_diff_nm**2 - energies**2) ** 2 + intersmear**2 * energies**2
                        ) * e_diff_nm
                        epsi_local[component_index, ispin] += (
                            fac * pksp2 * intersmear * energies * fn_row[iband1] / denominator
                        )
                        epsr_local[component_index, ispin] += (
                            fac
                            * pksp2
                            * (e_diff_nm**2 - energies**2)
                            * fn_row[iband1]
                            / denominator
                        )

            if fnf_spin is None:
                continue

            fnf_row = fnf_spin[ik_local]
            denominator_metal = energies**4 + intrasmear**2 * energies**2
            for iband1 in range(bndmax):
                pksp2 = np.real(momentum_i[iband1, iband1] * momentum_j[iband1, iband1])
                epsi_local[component_index, ispin] += (
                    pksp2 * intrasmear * energies * fnf_row[iband1] / denominator_metal
                )
                epsr_local[component_index, ispin] -= (
                    pksp2 * fnf_row[iband1] * energies**2 / denominator_metal
                )

    epsi_global = np.zeros_like(epsi_local)
    epsr_global = np.zeros_like(epsr_local)
    comm.Allreduce(epsi_local, epsi_global, op=MPI.SUM)
    comm.Allreduce(epsr_local, epsr_global, op=MPI.SUM)

    epsi_global *= factor
    epsr_global *= factor

    if nspin == 1:
        for component_index, (ipol, jpol) in enumerate(d_tensor):
            epsi = epsi_global[component_index, 0]
            epsr = float(ipol == jpol) + epsr_global[component_index, 0]
            eels = epsi / (epsi**2 + epsr**2)
            ieps = _compute_inverse_epsilon(energies, epsi)
            indices = (LL[int(ipol)], LL[int(jpol)])
            for spectrum, tag in ((epsi, 'epsi'), (epsr, 'epsr'), (eels, 'eels'), (ieps, 'ieps')):
                filename = '%s_%s%s.dat' % ((tag,) + indices)
                data_controller.write_file_row_col(filename, energies, spectrum)

            if rank == 0 and int(ipol) == int(jpol):
                renorm = np.sqrt((2.0 / np.pi) * np.trapezoid(epsi * energies, x=energies))
                component = LL[int(ipol)] + LL[int(jpol)]
                print('Component', component, ', plasmon frequency = ', renorm, 'eV')
        return

    for component_index, (ipol, jpol) in enumerate(d_tensor):
        epsi_0 = epsi_global[component_index, 0]
        epsr_0 = float(ipol == jpol) + epsr_global[component_index, 0]
        eels_0 = epsi_0 / (epsi_0**2 + epsr_0**2)
        ieps_0 = _compute_inverse_epsilon(energies, epsi_0)

        epsi_1 = epsi_global[component_index, 1]
        epsr_1 = float(ipol == jpol) + epsr_global[component_index, 1]
        eels_1 = epsi_1 / (epsi_1**2 + epsr_1**2)
        ieps_1 = _compute_inverse_epsilon(energies, epsi_1)

        indices_0 = (LL[int(ipol)], LL[int(jpol)], 0)
        for spectrum, tag in (
            (epsi_0, 'epsi'),
            (epsr_0, 'epsr'),
            (eels_0, 'eels'),
            (ieps_0, 'ieps'),
        ):
            filename = '%s_%s%s_%d.dat' % ((tag,) + indices_0)
            data_controller.write_file_row_col(filename, energies, spectrum)

        indices_1 = (LL[int(ipol)], LL[int(jpol)], 1)
        for spectrum, tag in (
            (epsi_1, 'epsi'),
            (epsr_1, 'epsr'),
            (eels_1, 'eels'),
            (ieps_1, 'ieps'),
        ):
            filename = '%s_%s%s_%d.dat' % ((tag,) + indices_1)
            data_controller.write_file_row_col(filename, energies, spectrum)

        if rank == 0 and int(ipol) == int(jpol):
            epsi_total = epsi_0 + epsi_1
            renorm = np.sqrt((2.0 / np.pi) * np.trapezoid(epsi_total * energies, x=energies))
            component = LL[int(ipol)] + LL[int(jpol)]
            print('Component', component, ', plasmon frequency = ', renorm, 'eV')
