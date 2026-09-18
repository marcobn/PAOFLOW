import numpy as np
from mpi4py import MPI

from ..spectrum.do_eigh import get_degeneracies
from ..utils.communication import gather_scatter
from ..utils.perturb_split import perturb_split

# initialize parallel execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scipy import fftpack as FFT


def do_d2Hd2k_ij(Hksp, dHksp, Dnm, Rfft, alat, npool, v_kp, bnd, degen):
    """Compute the six unique second-order derivatives of the k-space Hamiltonian.

    Parameters
    ----------
    Hksp : np.ndarray, shape ``(snawf, nk1, nk2, nk3, nspin)``
        Distributed real-space Hamiltonian in the FFT representation, where
        each element already contains the factor :math:`i \\cdot a_{\\text{lat}}`
        (i.e. ``HR * 1j * alat``).
    dHksp : np.ndarray, shape ``(nkpnts, 3, nawf, nawf, nspin)``
        Gradient of the k-space Hamiltonian.  Its ``i`` component at each
        k-point is passed to :func:`perturb_split` as the perturbation that
        defines the rotated basis (see Notes).
    Dnm : np.ndarray, shape ``(snawf, 3)``
        Cartesian factors, one per distributed real-space element ``n``,
        entering the three additional terms of the second derivative
        (see Notes).
    Rfft : np.ndarray, shape ``(nk1, nk2, nk3, 3)``
        Real-space grid vectors used as multiplication factors in FFT-based
        gradient computation.
    alat : float
        Lattice constant in Bohr radii, used to scale the real-space Hamiltonian
        prior to the FFT.
    npool : int
        Number of MPI pools used for the :func:`gather_scatter` redistribution.
    v_kp : np.ndarray, shape ``(nkpnts, nawf, bnd, nspin)``
        Bloch eigenvectors (columns are eigenstates) at each k-point,
        distributed over pools.
    bnd : int
        Number of bands for which the curvature is computed.
    degen : list
        Nested list of degenerate subspace indices, one entry per spin channel
        and k-point, as produced by the eigenvalue solver.

    Returns
    -------
    M_ij : np.ndarray, shape ``(6, nkpnts, bnd, nspin)``
        Diagonal matrix elements
        :math:`\\langle n | d^2H / dk_i dk_j | n \\rangle` in the Bloch
        eigenstate basis for the six unique ``ij`` pairs
        ``xx, yy, zz, xy, xz, yz``.
    vel_degen : list
        Nested list of shape ``[6][nspin][nkpnts]`` holding the degenerate
        subspaces of the band velocity along ``i``, obtained by applying
        :func:`get_degeneracies` to the diagonal of ``dH/dk_i`` in the
        rotated basis.  Each entry is a list of index groups, one group per
        degenerate subspace.
    degen_M : list
        Nested list of shape ``[6][nspin][nkpnts]`` holding, for each
        subspace listed in ``vel_degen``, the corresponding diagonal block
        of :math:`d^2H / dk_i dk_j` in the rotated basis, i.e. an
        ``np.ndarray`` of shape ``(ul - ll, ul - ll)``.
    dvec_list : list
        Nested list of shape ``[6][nspin][nkpnts]`` containing the modified
        eigenvector arrays after degenerate-subspace diagonalisation (used
        by :func:`do_band_curvature` to compute the perturbative correction).
        Each element is either an empty array or an ``np.ndarray`` of shape
        ``(nawf, nawf)``.

    Notes
    -----
    For each of the six unique Cartesian index pairs :math:`(i, j)`, the
    second derivative is obtained via the FFT convolution

    .. math::

        d^2 H(\\mathbf{k}) / dk_i dk_j
            = \\mathcal{F}\\left[ R_i R_j \\cdot H(\\mathbf{R}) \\cdot i \\cdot a_{\\text{lat}} \\right]
            + D_i D_j \\cdot \\mathcal{F}\\left[ H(\\mathbf{R}) \\cdot i / a_{\\text{lat}} \\right]
            + D_i \\cdot \\mathcal{F}\\left[ R_j \\cdot H(\\mathbf{R}) \\cdot i \\right]
            + D_j \\cdot \\mathcal{F}\\left[ R_i \\cdot H(\\mathbf{R}) \\cdot i \\right]

    where :math:`R_i` is the *i*-th Cartesian component of the real-space
    lattice vector grid ``Rfft`` and :math:`D_i` is ``Dnm[:, i]``.  The four
    terms are the expansion of
    :math:`\\mathcal{F}[(a_{\\text{lat}} R_i + D_i)(a_{\\text{lat}} R_j + D_j)
    \\cdot H(\\mathbf{R}) \\cdot i / a_{\\text{lat}}]`.  The result is then
    projected onto the Bloch eigenstate basis using :func:`perturb_split` to
    correctly handle degenerate bands.  The basis is built from the first
    argument of :func:`perturb_split`, which is ``dH/dk_i``, so degenerate
    bands are resolved by the velocity along ``i`` rather than by the second
    derivative itself; that rotated velocity is reused to build ``vel_degen``
    and ``degen_M``.
    """
    # ----------------------
    # Compute the gradient of the k-space Hamiltonian
    # ----------------------
    Rfft = np.transpose(Rfft, (3, 0, 1, 2))

    num_n, nk1, nk2, nk3, nspin = Hksp.shape

    _, nk1, nk2, nk3, nspin = Hksp.shape

    M_ij = np.zeros((6, v_kp.shape[0], bnd, v_kp.shape[3]), dtype=float, order='C')
    ij_ind = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]], dtype=int)

    comm.Barrier()
    ########################################
    ### real space grid replaces k space ###
    ########################################
    # c1=c2=0
    # for ik in range(len(degen[0])):
    #     if len(degen[0][ik]) != 0:
    #         c1+=1
    #     else:
    #         c2+=1
    # print(c1+c2,c1,c2)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    num_n = Hksp.shape[0]

    dvec_list = []
    vel_degen = []
    degen_M = []

    for ij in range(M_ij.shape[0]):
        dir_tmp = []
        vel_degen_by_ij = []
        degen_M_by_ij = []
        d2Hksp = None
        d2Hksp = np.zeros((num_n, nk1, nk2, nk3, nspin), dtype=complex, order='C')

        ipol = ij_ind[ij][0]
        jpol = ij_ind[ij][1]

        RIJ = Rfft[ipol] * Rfft[jpol]

        for ispin in range(d2Hksp.shape[4]):
            for n in range(d2Hksp.shape[0]):
                # because of the way this is coded...Hksp is actually HR*1.0j*alat
                d2Hksp[n, :, :, :, ispin] = (
                    FFT.fftn(RIJ * Hksp[n, :, :, :, ispin] * 1.0j * alat)
                    + Dnm[n, ipol] * Dnm[n, jpol] * FFT.fftn(Hksp[n, :, :, :, ispin] * 1.0j / alat)
                    + Dnm[n, ipol] * FFT.fftn(Rfft[jpol] * Hksp[n, :, :, :, ispin] * 1.0j)
                    + Dnm[n, jpol] * FFT.fftn(Rfft[ipol] * Hksp[n, :, :, :, ispin] * 1.0j)
                )

        #############################################################################################
        #############################################################################################
        #############################################################################################

        # gather the arrays into flattened dHk
        d2Hksp = np.reshape(d2Hksp, (num_n, nk1 * nk2 * nk3, nspin), order='C')
        d2Hksp = gather_scatter(d2Hksp, 1, npool)
        nawf = int(np.sqrt(d2Hksp.shape[0]))

        d2Hksp = np.reshape(d2Hksp, (nawf, nawf, d2Hksp.shape[1], nspin), order='C')

        tksp = np.zeros_like(d2Hksp)

        # find non-degenerate set of psi(k) for d2H/d2k_ij
        for ispin in range(tksp.shape[3]):
            isp_tmp = []
            vel_degen_by_spin = []
            for ik in range(tksp.shape[2]):
                # we save dvec so that it can be used when calculating the second term in d2E/d2k
                v_aux, tksp[:, :, ik, ispin], dvec = perturb_split(
                    dHksp[ik, ipol, :, :, ispin],
                    d2Hksp[:, :, ik, ispin],
                    v_kp[ik, :, :, ispin],
                    degen[ispin][ik],
                    return_v_k=True,
                )

                vel_degen_by_kp = get_degeneracies(
                    v_aux.diagonal().reshape((1, len(v_aux.diagonal()), 1)), bnd
                )

                # vel_degen_by_spin.append(vel_degen_by_kp[vel_degen_by_kp == degen[ispin][ik]])
                vel_degen_by_spin.append(vel_degen_by_kp[0][0])

                isp_tmp.append(dvec)
            vel_degen_by_ij.append(vel_degen_by_spin)
            dir_tmp.append(isp_tmp)
        vel_degen.append(vel_degen_by_ij)
        dvec_list.append(dir_tmp)

        # get the value for d2H/d2k
        for ispin in range(tksp.shape[3]):
            for n in range(bnd):
                M_ij[ij, :, n, ispin] = tksp[n, n, :, ispin].real

        for ispin in range(tksp.shape[3]):
            degen_M_by_spin = []
            for ik in range(tksp.shape[2]):
                degen_M_by_kp = []
                for i in range(len(vel_degen[ij][ispin][ik])):
                    # degenerate subspace indices upper and lower lim
                    ll = vel_degen[ij][ispin][ik][i][0]
                    ul = vel_degen[ij][ispin][ik][i][-1] + 1

                    degen_M_by_kp.append(tksp[ll:ul, ll:ul, ik, ispin])
                degen_M_by_spin.append(degen_M_by_kp)
            degen_M_by_ij.append(degen_M_by_spin)
        degen_M.append(degen_M_by_ij)
        comm.Barrier()

    d2Hksp = None

    return M_ij, vel_degen, degen_M, dvec_list
