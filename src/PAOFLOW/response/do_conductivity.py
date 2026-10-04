#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def velocity_products_k(dH_i, dH_j, v_k: np.ndarray, degen: list) -> np.ndarray:
    """Band-diagonal :math:`v_i v_j` of :func:`do_conductivity` at one k-point.

    Parameters
    ----------
    dH_i, dH_j : np.ndarray or scipy.sparse matrix, shape ``(nawf, nawf)``
        ``dH/dk`` along the two Cartesian directions, PAO basis.
    v_k : np.ndarray, shape ``(nawf, m)``
        Eigenvectors at this k-point (columns).
    degen : list of np.ndarray
        Degenerate band groups at this k-point.

    Returns
    -------
    np.ndarray, shape ``(m,)``, complex
        :math:`\\langle n|\\partial_i H|n\\rangle \\langle n|\\partial_j H|n\\rangle`
        with both projected in the basis that diagonalizes ``dH_i`` inside
        each degenerate group (:func:`perturb_split`).  For ``i != j`` this
        is not ``velkp_i * velkp_j`` at a degeneracy, where ``velkp_j`` is
        resolved by ``dH_j`` instead.
    """
    from ..utils.perturb_split import perturb_split

    op1, op2 = perturb_split(dH_i, dH_j, v_k, degen)
    return np.diagonal(op1 * op2)


def conductivity_from_products(
    data_controller,
    products: np.ndarray,
    emin: float,
    emax: float,
    ne: int,
    delta: float,
    ipol: int,
    jpol: int,
    ispin: int,
) -> None:
    """Smear, reduce and write one component of the conductivity.

    Parameters
    ----------
    data_controller : DataController
        Supplies ``E_k`` and ``deltakp`` (this rank's k-slice, the bands of
        ``products``) and the run attributes; performs the file write.
    products : np.ndarray, shape ``(nk_local, nbands)``
        :func:`velocity_products_k` of every local k-point, spin ``ispin``.
    emin, emax, ne : float, float, int
        Energy grid; ``emax`` is clipped to ``attr['shift']``.
    delta : float
        Fixed Gaussian width, used without adaptive smearing.
    ipol, jpol, ispin : int
        Tensor component and spin channel, for the file name.

    Returns
    -------
    None
        Writes ``cond_<i><j>_<ispin>.dat`` (collective call).
    """
    from ..utils.constants import LL
    from ..utils.smearing import gaussian, metpax

    arrays, attributes = data_controller.data_dicts()

    # Conductivity Calculation with Gaussian Smearing
    emax = np.amin(np.array([attributes['shift'], emax]))
    ene = np.linspace(emin, emax, ne)

    nbands = products.shape[1]
    E_k = np.real(arrays['E_k'][:, :nbands, ispin])
    condaux = np.zeros((ne), dtype=float)

    if attributes['smearing'] != None:
        taux = np.zeros((arrays['deltakp'].shape[0], nbands), dtype=float)
        deltakp = arrays['deltakp'][:, :nbands, ispin]

    for n in range(ne):
        # Adaptive Gaussian Smearing
        if attributes['smearing'] == 'gauss':
            taux = gaussian(ene[n], E_k, deltakp)
        # Adaptive M-P smearing
        elif attributes['smearing'] == 'm-p':
            taux = metpax(ene[n], E_k, deltakp)
        elif attributes['smearing'] == None:
            taux = np.exp(-(((ene[n] - E_k[:, :]) / delta) ** 2)) / np.sqrt(np.pi)

        condaux[n] += np.sum(np.real(taux * products))

    cond = np.zeros((ne), dtype=float) if rank == 0 else None

    comm.Reduce(condaux, cond, op=MPI.SUM)
    condaux = None

    if rank == 0:
        if attributes['smearing'] == None:
            cond /= (float(attributes['nkpnts'])) * delta  # Chech the normalization factor
            # cond /= ((float(attributes['nkpnts']))*np.sqrt(np.pi)*delta)
        else:
            cond /= float(attributes['nkpnts'])

    cart_indices = (str(LL[ipol]), str(LL[jpol]), str(ispin))

    fcond = 'cond_%s%s_%s.dat' % cart_indices
    data_controller.write_file_row_col(fcond, ene, cond)


def do_conductivity(data_controller, emin, emax, ne, delta, ipol, jpol):
    """Band-diagonal conductivity component ``ipol, jpol`` on the dense k-slice.

    Notes
    -----
    Per k-point the work is :func:`velocity_products_k`; the smearing,
    reduction and file write are :func:`conductivity_from_products`, which
    the sparse backend calls with products it streams from the mesh pass.
    """
    arrays, attributes = data_controller.data_dicts()

    nspin = attributes['nspin']
    nktot = arrays['pksp'].shape[0]  # attributes['nkpnts']
    nawf = attributes['nawf']

    for ispin in range(nspin):
        products = np.empty((nktot, nawf), dtype=complex)
        for ik in range(nktot):
            products[ik] = velocity_products_k(
                arrays['dHksp'][ik, ipol, :, :, ispin],
                arrays['dHksp'][ik, jpol, :, :, ispin],
                arrays['v_k'][ik, :, :, ispin],
                arrays['degen'][ispin][ik],
            )
        conductivity_from_products(
            data_controller, products, emin, emax, ne, delta, ipol, jpol, ispin
        )
