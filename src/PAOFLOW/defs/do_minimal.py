
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Block diagonalization of Hermitian matrices
"""L S Cederbaum, J Schirmer and H-D Meyer
J. Phys. A: Mat. Gen. 22, 2427 (1989)

The formula for the block diagonal part of the S matrix is from:
B.N. Parlett, The symmetric eigenvalue problem (SIAM's Classics in Applied Mathematics, 1998)
pag. 45, sec. 3.1.1 remark n. 5."""


def do_minimal(data_controller, first_band):
    """Remove low-energy bands from the Hamiltonian via block diagonalisation.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``Hksp`` (shape ``(nkpnts, nawf, nawf, nspin)``).
        Required attributes: ``nawf``, ``nspin``, ``nkpnts``.
        Also reads ``basis`` from ``data_arrays``.
    first_band : int
        Index of the first band to retain.  Bands with indices
        ``0, 1, ..., first_band-1`` are projected out.

    Returns
    -------
    None
        Modifies ``data_controller.data_arrays`` and
        ``data_controller.data_attributes`` in place:

        - ``Hksp`` : np.ndarray, shape ``(nkpnts, nawf-first_band,
          nawf-first_band, nspin)`` — the reduced Hamiltonian after
          projecting out the lowest ``first_band`` bands.
        - ``Dnm`` : np.ndarray, shape ``(nawf-first_band, nawf-first_band, 3)``
          — position-difference matrix for the retained orbitals.

        Updates attribute: ``nawf = nawf - first_band``.

    Notes
    -----
    The block diagonalisation follows the algorithm of Cederbaum, Schirmer,
    and Meyer (J. Phys. A **22**, 2427, 1989).  At each k-point the full
    Hamiltonian is diagonalised, its eigenvector matrix is partitioned into
    a :math:`(2 \\times 2)` block structure, and a similarity transformation
    :math:`T` is constructed such that :math:`T^\\dagger H T` is block-diagonal.
    The lower-right ``(nawf - first_band) x (nawf - first_band)`` block is
    retained as the reduced Hamiltonian.

    Small random perturbations (amplitude :math:`10^{-4}`) are added to the
    block matrices to break accidental degeneracies before inversion.
    """
    # this eliminates bands at the bottom

    arry, attr = data_controller.data_dicts()

    import numpy.random as rd
    from numpy import linalg as npl

    # bnd = attr['bnd']
    bnd = first_band
    nawf = attr['nawf']
    nspin = attr['nspin']
    nkpnts = attr['nkpnts']
    basis = arry['basis']

    # arry['Hksp'] = np.reshape(arry['Hksp'],(nawf,nawf,nkpnts,nspin))

    Hks = np.zeros((nkpnts, nawf - bnd, nawf - bnd, nspin), dtype=complex)
    for ik in range(nkpnts):
        for ispin in range(nspin):
            Sbd = np.zeros((nawf, nawf), dtype=complex)
            Sbdi = np.zeros((nawf, nawf), dtype=complex)
            S = sv = np.zeros((nawf, nawf), dtype=complex)
            e = se = np.zeros(nawf, dtype=float)
            e, S = npl.eigh(arry['Hksp'][ik, :, :, ispin])
            S11 = S[:bnd, :bnd] + 1.0 * rd.random(bnd) / 10000.0
            S21 = S[:bnd, bnd:] + 1.0 * rd.random(nawf - bnd) / 10000.0
            S12 = S21.T
            S22 = S[bnd:, bnd:] + 1.0 * rd.random(nawf - bnd) / 10000.0
            S22 = S22 + S21.T.dot(np.dot(npl.inv(S11), S12.T))
            Sbd[:bnd, :bnd] = 0.5 * (S11 + np.conj(S11.T))
            Sbd[bnd:, bnd:] = 0.5 * (S22 + np.conj(S22.T))
            Sbdi = npl.inv(np.dot(Sbd, np.conj(Sbd.T)))
            se, sv = npl.eigh(Sbdi)
            se = np.sqrt(se + 0.0j) * np.identity(nawf, dtype=complex)
            Sbdi = sv.dot(se).dot(np.conj(sv).T)
            T = S.dot(np.conj(Sbd.T)).dot(Sbdi)
            Hbd = np.conj(T.T).dot(np.dot(arry['Hksp'][ik, :, :, ispin], T))
            # Hks[:,:,ik,ispin] = 0.5*(Hbd[:bnd,:bnd]+np.conj(Hbd[:bnd,:bnd].T))
            # Hks[:,:,ik,ispin] = 0.5*(Hbd[first_band:,first_band:]+np.conj(Hbd[first_band:,first_band:].T))
            Hks[ik, :, :, ispin] = Hbd[first_band:, first_band:]

    arry['Hksp'] = Hks  # np.reshape(Hks,(nkpnts,nawf-bnd,nawf-bnd,nspin))
    attr['nawf'] = nawf - bnd
    # ashape = (attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin'])
    # arry['Hks'] = np.reshape(arry['Hks'], ashape)
    arry['Dnm'] = np.empty((attr['nawf'], attr['nawf'], 3))
    for i in range(3):
        for n in range(attr['nawf']):
            for m in range(attr['nawf']):
                arry['Dnm'][n, m, i] = (
                    basis[n + first_band]['tau'][i] - basis[m + first_band]['tau'][i]
                )
