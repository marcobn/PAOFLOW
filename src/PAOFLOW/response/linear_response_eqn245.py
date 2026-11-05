from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from ..DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from .kubo_units import conductivity_conversion, sheet_unit
from ..utils.perturb_split import perturb_split
from ..utils.communication import reduce_full
from ..utils.smearing import gaussian, intgaussian


bohr_to_cm = 5.29177249e-9


def calc_chi(data_controller):
    arry, attr = data_controller.data_dicts()
    if attr['response'] == 'ree' or attr['response'] == 'shc':
        if attr['dftSO'] == False:
            if rank == 0:
                print('Relativistic calculation with SO required')
                comm.Abort()
            comm.Barrier()
    nk, nbnd, nspin = arry['E_k'].shape

    if attr['response'] == 'shc' and attr['full_chi2'] and attr['t_odd'] == False:
        for tensor in arry['shc_tensor']:
            spol, ipol = tensor[0], tensor[2]
            if rank == 0:
                start_time = time.time()
            jdHksp = do_spin_current(data_controller=data_controller, tensor=tensor)
            oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            for ik in range(jdHksp.shape[0]):
                for ispin in range(jdHksp.shape[3]):
                    oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = perturb_split(
                        jdHksp[ik, :, :, ispin],
                        arry['dHksp'][ik, ipol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            jdHksp = None
            calc_chi2(
                data_controller=data_controller,
                tensor=tensor,
                prop='shc',
                oper_matrix1=oper_matrix1,
                oper_matrix2=oper_matrix2,
            )
            oper_matrix1 = oper_matrix2 = None
            jdHksp = None
            if rank == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 60.0
                print(
                    f'EVEN SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.'
                )

    if attr['response'] == 'shc' and attr['t_odd'] == True:
        if attr['intraband'] or attr['interband']:
            for tensor in arry['shc_tensor']:
                spol, ipol = tensor[0], tensor[2]
                if rank == 0:
                    start_time = time.time()
                jdHksp = do_spin_current(data_controller=data_controller, tensor=tensor)
                oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                for ik in range(jdHksp.shape[0]):
                    for ispin in range(jdHksp.shape[3]):
                        oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = (
                            perturb_split(
                                jdHksp[ik, :, :, ispin],
                                arry['dHksp'][ik, ipol, :, :, ispin],
                                arry['v_k'][ik, :, :, ispin],
                                arry['degen'][ispin][ik],
                            )
                        )
                jdHksp = None
                if attr['intraband']:
                    fermi_surf(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='shc',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                if attr['interband']:
                    fermi_sea(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='shc',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                oper_matrix1 = oper_matrix2 = None
                jdHksp = None
                if rank == 0:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 60.0
                    print(
                        f'ODD SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.'
                    )
        else:
            pass

    if attr['response'] == 'ree' and attr['full_chi2'] and attr['t_odd'] == True:
        for tensor in arry['ree_tensor']:
            if rank == 0:
                start_time = time.time()
            spol, ipol = tensor[0], tensor[1]
            oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            for ik in range(nk):
                for ispin in range(nspin):
                    oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = perturb_split(
                        arry['Sj'][spol, :, :],
                        arry['dHksp'][ik, ipol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            calc_chi2(
                data_controller=data_controller,
                tensor=tensor,
                prop='ree',
                oper_matrix1=oper_matrix1,
                oper_matrix2=oper_matrix2,
            )
            oper_matrix1 = oper_matrix2 = None
            if rank == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 60.0
                print(f'ODD REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

    if attr['response'] == 'ree' and attr['t_odd'] == False:
        if attr['intraband'] or attr['interband']:
            for tensor in arry['ree_tensor']:
                if rank == 0:
                    start_time = time.time()
                spol, ipol = tensor[0], tensor[1]
                oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                for ik in range(nk):
                    for ispin in range(nspin):
                        oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = (
                            perturb_split(
                                arry['Sj'][spol, :, :],
                                arry['dHksp'][ik, ipol, :, :, ispin],
                                arry['v_k'][ik, :, :, ispin],
                                arry['degen'][ispin][ik],
                            )
                        )
                if attr['intraband']:
                    fermi_surf(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='ree',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                if attr['interband']:
                    fermi_sea(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='ree',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                oper_matrix1 = oper_matrix2 = None
                if rank == 0:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 60.0
                    print(f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

    if attr['response'] == 'ahc' and attr['full_chi2']:
        for tensor in arry['ree_tensor']:
            if rank == 0:
                start_time = time.time()
            cpol, ipol = tensor[0], tensor[1]
            oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            for ik in range(nk):
                for ispin in range(nspin):
                    oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = perturb_split(
                        arry['dHksp'][ik, cpol, :, :, ispin],
                        arry['dHksp'][ik, ipol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            calc_chi2(
                data_controller=data_controller,
                tensor=tensor,
                prop='ahc',
                oper_matrix1=oper_matrix1,
                oper_matrix2=oper_matrix2,
            )
            oper_matrix1 = oper_matrix2 = None
            if rank == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 60.0
                print(f'AHC [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

    if attr['response'] == 'cond':
        if attr['intraband'] or attr['interband']:
            for tensor in arry['ree_tensor']:
                if rank == 0:
                    start_time = time.time()
                cpol, ipol = tensor[0], tensor[1]
                oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                for ik in range(nk):
                    for ispin in range(nspin):
                        oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = (
                            perturb_split(
                                arry['dHksp'][ik, cpol, :, :, ispin],
                                arry['dHksp'][ik, ipol, :, :, ispin],
                                arry['v_k'][ik, :, :, ispin],
                                arry['degen'][ispin][ik],
                            )
                        )
                if attr['intraband']:
                    fermi_surf(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='cond',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                if attr['interband']:
                    fermi_sea(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='cond',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                oper_matrix1 = oper_matrix2 = None
                if rank == 0:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 60.0
                    print(f'Cond [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')


def calc_chi2(
    data_controller: 'DataController' = None,
    tensor: np.ndarray | None = None,
    prop: str | None = None,
    oper_matrix1: np.ndarray | None = None,
    oper_matrix2: np.ndarray | None = None,
) -> None:
    """Fermi-sea term at finite broadening, tutorial03 Eq. (2).

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k``, ``deltakp`` and the settings of :meth:`PAOFLOW.linear_response`.
    tensor : ndarray of int
        ``(spin, current, field)`` for 'shc', ``(response, field)`` otherwise.
    prop : {'shc', 'ree', 'ahc'}
    oper_matrix1, oper_matrix2 : ndarray, shape ``(nk, nbnd, nbnd, nspin)``, complex
        Response and field operators in the band basis (:func:`perturb_split`).

    Returns
    -------
    None
        Writes ``SHC_EVEN_*``, ``REE_ODD_*`` or ``AHC_*`` files.

    Notes
    -----
    .. math::

        \\chi = \\pm\\frac{2}{N_k}\\sum_{k,n\\ne m} f_n\\,\\mathrm{Im}[A_{nm}B_{mn}]
        \\frac{\\gamma^2-\\varepsilon_{nm}^2}{(\\varepsilon_{nm}^2+\\gamma^2)^2},

    the Kubo-Bastin Fermi sea with the occupation taken at the band energy (sharp in the level
    :math:`n`, Lorentzian in :math:`m`). The sign is + for 'shc' and 'ree' and - for 'ahc': as
    :math:`\\gamma\\to 0` the 'shc'/'ree' values tend to the Berry-curvature form (Eq. 3) and the
    'ahc' value to minus it, i.e. ``full_chi2`` flips the sign of the AHC (kept for compatibility;
    see ``docs/internals/kubo-bastin.md``).
    """
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    ene = eqn245_energy_grid(attr)
    gamma = attr['gamma']
    for ispin in eqn245_spins(prop, nspin):
        aux1 = np.zeros((nk, nbnd), dtype=float)
        for ik in range(nk):
            aux1[ik, :] = chi2_k(
                oper_matrix1[ik, :, :, ispin],
                oper_matrix2[ik, :, :, ispin],
                arry['E_k'][ik, :, ispin],
                gamma,
                prop,
            )
        local_response = occupied_sum(
            arry['E_k'][:, :, ispin], arry['deltakp'][:, :, ispin], aux1, ene
        )
        aux1 = None
        eqn245_finish(data_controller, local_response, ene, 'chi2', prop, tensor, ispin)

    oper_matrix1 = None
    oper_matrix2 = None


def fermi_surf(
    data_controller: 'DataController' = None,
    tensor: np.ndarray | None = None,
    prop: str | None = None,
    oper_matrix1: np.ndarray | None = None,
    oper_matrix2: np.ndarray | None = None,
) -> None:
    """Intraband part of the Fermi-surface term, tutorial03 Eq. (4).

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k``, ``deltakp`` and the settings of :meth:`PAOFLOW.linear_response`.
    tensor : ndarray of int
        ``(spin, current, field)`` for 'shc', ``(response, field)`` otherwise.
    prop : {'shc', 'ree', 'cond'}
    oper_matrix1, oper_matrix2 : ndarray, shape ``(nk, nbnd, nbnd, nspin)``, complex
        Response and field operators in the band basis.

    Returns
    -------
    None
        Writes ``SHC_ODD_*``, ``REE_EVEN_*`` or ``Cond_*`` files ('cond': one per spin channel when
        ``nspin = 2``).

    Notes
    -----
    .. math::

        \\chi = \\frac{1}{2\\gamma N_k}\\sum_{k,n}\\mathrm{Re}[A_{nn}B_{nn}]\\,
        \\delta(\\mu-\\varepsilon_n),

    the small-:math:`\\gamma` limit of :math:`\\pi\\sum_n A_{nn}B_{nn}L_n(\\mu)^2`
    (:math:`\\int\\pi L^2 = 1/2\\gamma`), i.e. the Boltzmann form with :math:`\\tau=\\hbar/2\\gamma`;
    the delta function is the adaptive Gaussian. 'shc' and 'ree' carry an extra minus sign.
    """
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    ene = eqn245_energy_grid(attr)
    for ispin in eqn245_spins(prop, nspin):
        numerator = np.zeros((nk, nbnd), dtype=float)
        for ik in range(nk):
            numerator[ik, :] = surf_k(
                oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin], prop
            )
        local_response = surface_sum(
            arry['E_k'][:, :, ispin], arry['deltakp'][:, :, ispin], numerator, ene
        )
        numerator = None
        eqn245_finish(data_controller, local_response, ene, 'surf', prop, tensor, ispin)

    oper_matrix1 = None
    oper_matrix2 = None


def fermi_sea(
    data_controller: 'DataController' = None,
    tensor: np.ndarray | None = None,
    prop: str | None = None,
    oper_matrix1: np.ndarray | None = None,
    oper_matrix2: np.ndarray | None = None,
) -> None:
    """Interband part of the Fermi-surface term, tutorial03 Eq. (5).

    The name is historical: the quantity stands for the interband (:math:`n\\ne m`) part of the
    Fermi-surface term (Eq. 1), written as an occupation-weighted sum; it is not the Kubo-Bastin Fermi
    sea.

    Parameters
    ----------
    data_controller : DataController
        Reads ``E_k``, ``deltakp`` and the settings of :meth:`PAOFLOW.linear_response`.
    tensor : ndarray of int
        ``(spin, current, field)`` for 'shc', ``(response, field)`` otherwise.
    prop : {'shc', 'ree', 'cond'}
    oper_matrix1, oper_matrix2 : ndarray, shape ``(nk, nbnd, nbnd, nspin)``, complex
        Response and field operators in the band basis.

    Returns
    -------
    None
        Writes ``SHC_ODD_*``, ``REE_EVEN_*`` or ``Cond_*`` files.

    Notes
    -----
    .. math::

        \\chi = \\pm\\frac{2}{N_k}\\sum_{k,n\\ne m} f_n\\,\\mathrm{Re}[A_{nm}B_{mn}]
        \\frac{\\gamma\\,(\\varepsilon_m-\\varepsilon_n)}{((\\varepsilon_m-\\varepsilon_n)^2+\\gamma^2)^2}
        = \\pm\\frac{\\pi}{N_k}\\sum_{k,n\\ne m} f_n\\,\\mathrm{Re}[A_{nm}B_{mn}]\\,
        L_m'(\\varepsilon_n),

    with + for 'shc'/'ree' and - for 'cond'. The exact interband Fermi-surface term at constant
    broadening, :math:`\\pi\\sum_{n\\ne m}\\mathrm{Re}[A_{nm}B_{mn}]\\int(-f')L_nL_m`, is
    :meth:`PAOFLOW.kubo_response` with ``term='interband'``.
    """
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    ene = eqn245_energy_grid(attr)
    gamma = attr['gamma']
    for ispin in eqn245_spins(prop, nspin):
        aux_odd = np.zeros((nk, nbnd), dtype=float)
        for ik in range(nk):
            aux_odd[ik, :] = sea_k(
                oper_matrix1[ik, :, :, ispin],
                oper_matrix2[ik, :, :, ispin],
                arry['E_k'][ik, :, ispin],
                gamma,
                prop,
            )
        local_response = occupied_sum(
            arry['E_k'][:, :, ispin], arry['deltakp'][:, :, ispin], aux_odd, ene
        )
        aux_odd = None
        eqn245_finish(data_controller, local_response, ene, 'sea', prop, tensor, ispin)

    oper_matrix1 = None
    oper_matrix2 = None


# Broadening of the energy denominators of equations (2), (4) and (5)
_DELTAB = {'chi2': 0.001, 'surf': 0.0001, 'sea': 0.001}


def eqn245_energy_grid(attr):
    """Energy grid of equations (2), (4), (5), with ``emaxH`` clipped to ``shift``."""
    return np.linspace(attr['eminH'], min(attr['shift'], attr['emaxH']), attr['esize'])


def eqn245_spins(prop, nspin):
    """Spin channels summed: SHC and REE read spin 0 only, conductivity and AHC each one."""
    return [0] if prop in ('shc', 'ree') else list(range(nspin))


def chi2_k(op1, op2, E, gamma, prop):
    """Equation (2) band vector at one k-point and spin.

    Returns
    -------
    np.ndarray, shape ``(nbnd,)``
        :math:`\\pm 2 \\sum_{m \\ne n} \\mathrm{Im}[O_{1,nm} O_{2,mn}]
        (\\gamma^2 - E_{nm}^2) / ((E_{nm}^2 + \\gamma^2)^2 + \\delta^2)`,
        ``+`` for SHC/REE, ``-`` for AHC.
    """
    deltab = _DELTAB['chi2']
    sign = 2.0 if prop in ('shc', 'ree') else -2.0
    E_nm = (E - E[:, None]) ** 2
    aux = op1 * op2.T
    np.fill_diagonal(aux, 0.0)
    aux = sign * np.imag(aux) * (gamma**2 - E_nm)
    aux /= (E_nm + gamma**2) ** 2 + deltab**2
    return np.sum(aux, axis=1)


def surf_k(op1, op2, prop):
    """Equation (4) intraband numerator :math:`\\mathrm{Re}[O_{1,nn} O_{2,nn}]` at one k-point and spin."""
    numerator = np.real(np.diagonal(op1) * np.diagonal(op2))
    if prop in ('shc', 'ree'):
        numerator = 0.5 * (numerator + np.conj(numerator.T))
    return numerator


def sea_k(op1, op2, E, gamma, prop):
    """Equation (5) band vector at one k-point and spin (``+2`` SHC/REE, ``-2`` conductivity)."""
    deltab = _DELTAB['sea']
    sign = 2.0 if prop in ('shc', 'ree') else -2.0
    E_nm = E - E[:, None]
    aux = op1 * op2.T
    np.fill_diagonal(aux, 0.0)
    aux = sign * np.real(aux) * gamma * E_nm
    aux /= (E_nm**2 + gamma**2) ** 2 + deltab**2
    return np.sum(aux, axis=1)


def occupied_sum(E_k, deltakp, aux, ene):
    """Fermi-sea sum :math:`\\sum_{kn} a_{kn} f(E_{kn})` over a block of k-points, per energy.

    The dense kernel passes its whole k-slice, the sparse backend one
    k-point at a time; ``E_k``, ``deltakp`` and ``aux`` are ``(nk, nbnd)``.
    """
    local_response = np.zeros(len(ene), dtype=float)
    for ie in range(len(ene)):
        smear = intgaussian(E_k, ene[ie], deltakp)
        local_response[ie] = np.sum(aux * smear)
        smear = None
    return local_response


def surface_sum(E_k, deltakp, numerator, ene):
    """Fermi-surface sum :math:`\\sum_{kn} a_{kn} \\delta(E_{kn} - E)` over a block of k-points."""
    local_response = np.zeros(len(ene), dtype=float)
    numerator_flat = numerator.ravel()
    for ie in range(len(ene)):
        smear = gaussian(E_k, ene[ie], deltakp)
        local_response[ie] = np.dot(numerator_flat, smear.ravel())
        smear = None
    return local_response


def eqn245_finish(data_controller, local_response, ene, kind, prop, tensor, ispin):
    """Reduce, normalize and write one component of equation (2), (4) or (5).

    Parameters
    ----------
    local_response : np.ndarray, shape ``(esize,)``
        This rank's block sums for spin ``ispin``.
    kind : {'chi2', 'surf', 'sea'}
        Equation (2), (4) or (5).
    prop : {'shc', 'ree', 'ahc', 'cond'}
    """
    arry, attr = data_controller.data_dicts()
    nspin = arry['E_k'].shape[2]
    gamma = attr['gamma']
    deltab = _DELTAB[kind]

    cgs_conv = conductivity_conversion(data_controller, attr['twoD'])

    # reduce_full gives the sum over ALL k-points on rank 0.
    aux_full = reduce_full(local_response, sroot=0)
    local_response = None
    if rank == 0:
        # Since reduce_full has already summed over k,now divide by total k-points.
        aux_full /= attr['nkpnts']
        if prop in ('shc', 'ahc', 'cond'):
            aux_full *= cgs_conv
        if prop == 'ree':
            aux_full *= bohr_to_cm
        if kind == 'surf':
            if prop in ('shc', 'ree'):
                aux_full /= (-2.0 * gamma) + deltab
            else:
                aux_full /= (2.0 * gamma) + deltab

    xzy = ['x', 'y', 'z']
    even = kind == 'chi2'
    if prop == 'shc':
        spol, jpol, ipol = (tensor[0], tensor[1], tensor[2])
        fname = f'SHC_{"EVEN" if even else "ODD"}_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Chi [(hbar/e)(S/cm)]'
    elif prop == 'ree':
        spol, ipol = (tensor[0], tensor[1])
        fname = f'REE_{"ODD" if even else "EVEN"}_{xzy[spol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Chi [hbar*(cm/V)]'
    else:
        cpol, ipol = (tensor[0], tensor[1])
        stem = 'AHC' if prop == 'ahc' else 'Cond'
        if nspin == 1:
            fname = f'{stem}_{xzy[cpol]}{xzy[ipol]}.dat'
        else:
            fname = f'{stem}_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'AH Conductivity [S/cm]' if prop == 'ahc' else 'Conductivity [S/cm]'
    if attr['twoD']:
        unit2 = sheet_unit(unit2)
    data_controller.write_file_row_col_units(fname, ene, aux_full, unit1, unit2)
    aux_full = None


def do_spin_current(data_controller=None, tensor=None) -> np.ndarray:
    """Spin current {S_spin, dH_current}/2 of ``tensor = (spin, current, ...)`` at every local k."""
    from .do_Hall import do_spin_current as spin_current_operator

    return spin_current_operator(data_controller, tensor[0], tensor[1])
