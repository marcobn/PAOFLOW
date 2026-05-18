import numpy as np
from mpi4py import MPI

from .communication import gather_scatter
from .perturb_split import perturb_split

# initialize parallel execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scipy import fftpack as FFT


def do_d2Hd2k_ij(Hksp, Rfft, alat, npool, v_kp, bnd, degen):
    """Compute the six unique second-order derivatives of the k-space Hamiltonian.

    Parameters
    ----------
    Hksp : np.ndarray, shape ``(snawf, nk1, nk2, nk3, nspin)``
        Distributed real-space Hamiltonian in the FFT representation, where
        each element already contains the factor :math:`i \\cdot a_{\\text{lat}}`
        (i.e. ``HR * 1j * alat``).
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

    where :math:`R_i` is the *i*-th Cartesian component of the real-space
    lattice vector grid ``Rfft``.  The result is then projected onto the
    Bloch eigenstate basis using :func:`perturb_split` to correctly handle
    degenerate bands.
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

    for ij in range(M_ij.shape[0]):
        dir_tmp = []
        d2Hksp = None
        d2Hksp = np.zeros((num_n, nk1, nk2, nk3, nspin), dtype=complex, order='C')

        ipol = ij_ind[ij][0]
        jpol = ij_ind[ij][1]

        RIJ = Rfft[ipol] * Rfft[jpol]

        for ispin in range(d2Hksp.shape[4]):
            for n in range(d2Hksp.shape[0]):
                # because of the way this is coded...Hksp is actually HR*1.0j*alat
                d2Hksp[n, :, :, :, ispin] = FFT.fftn(RIJ * Hksp[n, :, :, :, ispin] * 1.0j * alat)

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
            for ik in range(tksp.shape[2]):
                # we save dvec so that it can be used when calculating the second term in d2E/d2k
                tksp[:, :, ik, ispin], _, dvec = perturb_split(
                    d2Hksp[:, :, ik, ispin],
                    d2Hksp[:, :, ik, ispin],
                    v_kp[ik, :, :, ispin],
                    degen[ispin][ik],
                    return_v_k=True,
                )

                isp_tmp.append(dvec)
            dir_tmp.append(isp_tmp)
        dvec_list.append(dir_tmp)

        # get the value for d2H/d2k
        for ispin in range(d2Hksp.shape[3]):
            for n in range(bnd):
                M_ij[ij, :, n, ispin] = tksp[n, n, :, ispin].real

        comm.Barrier()

    d2Hksp = None

    return M_ij, dvec_list
