import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


### Reformat
def build_Hks(data_controller):
    """Construct the PAO Hamiltonian :math:`H(\\mathbf{k})` from projected DFT eigenstates.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``my_eigsmat`` (shape ``(nbnds, nkpnts, nspin)``).
        The projections are read through
        :meth:`~PAOFLOW.DataController.DataController.local_projections`.
        Required attributes: ``bnd``, ``nawf``, ``nspin``, ``nkpnts``,
        ``npool``, ``shift``, ``shift_type``.

    Returns
    -------
    np.ndarray or None
        On rank 0, the PAO Hamiltonian in k-space with shape
        ``(nawf, nawf, nkpnts, nspin)``; ``None`` on all other ranks.

    Notes
    -----
    The PAO Hamiltonian is built at each k-point and spin channel via

    .. math::

        H(\\mathbf{k}) = A_c \\varepsilon_c A_c^\\dagger
            + \\eta \\bigl(\\mathbf{1} - A_c (A_c^\\dagger A_c)^{-1} A_c^\\dagger\\bigr)

    where :math:`A_c` contains the ``bnd`` lowest normalised eigenvectors of
    the DFT overlap matrix and :math:`\\varepsilon_c` the corresponding
    eigenvalues, following the shift schemes of Buongiorno Nardelli *et al.*
    PRB 2013 (``shift_type=0``) and PRB 2016 (``shift_type=1``), or no shift
    (``shift_type=2``).  Hermiticity is enforced after each k-point.

    The k-point loop is distributed over MPI ranks, so each rank builds and
    stores only its own ``nkpnts / size`` share before the result is gathered
    on rank 0.
    """
    from scipy import linalg as spl

    from ..utils.communication import gather_full, scatter_full

    arrays, attributes = data_controller.data_dicts()

    bnd = attributes['bnd']
    nawf = attributes['nawf']
    eta = attributes['shift']
    nspin = attributes['nspin']
    nkpnts = attributes['nkpnts']
    npool = attributes['npool']
    shift_type = attributes['shift_type']
    # Local (per-k) projectability threshold.  Bands whose squared PAO
    # projection at a given k falls below this value are excluded from the
    # construction at that k-point even if they pass the global ``pthr``
    # criterion that determines ``bnd``.  This guards against pathological
    # gauge mixings in degenerate subspaces (e.g. at high-symmetry points
    # such as Gamma) where one of the otherwise "good" bands can have a
    # vanishing PAO content and would otherwise be artificially renormalised
    # to unit norm, corrupting H(k) at that point.
    pthr_local = attributes.get('pthr_local', 0.5 * attributes.get('pthr', 0.95))

    U_local = data_controller.local_projections()
    my_eigsmat = arrays['my_eigsmat']

    # Same partition as U_local; needed to index the (replicated) eigenvalues.
    k_local = scatter_full(np.arange(nkpnts, dtype=int), npool)

    Hks_local = np.zeros((k_local.size, nawf, nawf, nspin), dtype=complex)

    for ikl in range(k_local.size):
        ik = int(k_local[ikl])
        for ispin in range(nspin):
            my_eigs = my_eigsmat[:, ik, ispin]

            # Building the Hamiltonian matrix
            UU = np.transpose(
                U_local[ikl, :, :, ispin]
            )  # transpose of U. Now the columns of UU are the eigenvector of length nawf
            proj_k = np.real(np.sum(np.conj(UU) * UU, axis=0))
            # Avoid division by zero for vanishing projections.
            safe_proj = np.where(proj_k > 0.0, proj_k, 1.0)
            norms = 1.0 / np.sqrt(safe_proj)
            UU[:, :nawf] = UU[:, :nawf] * norms[:nawf]

            # Select bands that (i) are below the energy shift and
            # (ii) have a sizable local projectability.  Bands failing the
            # local projectability test are dropped at this k-point only.
            sel = [n for n in range(bnd) if my_eigs[n] <= eta and proj_k[n] > pthr_local]
            bnd_ik = len(sel)
            if bnd_ik == 0:
                print('No Eigenvalues in the selected energy range')
                comm.Abort()
            ac = UU[:, sel]  # filtering: per-k projectability + energy cutoff
            ee1 = np.diag(my_eigs[sel])
            if shift_type == 0:
                # option 1 (PRB 2013)
                hk = ac.dot(ee1).dot(np.conj(ac).T) + eta * (
                    np.identity(nawf) - ac.dot(np.conj(ac).T)
                )

            elif shift_type == 1:
                # option 2 (PRB 2016)
                aux_p = spl.inv(np.dot(np.conj(ac).T, ac))
                hk = ac.dot(ee1).dot(np.conj(ac).T) + eta * (
                    np.identity(nawf) - ac.dot(aux_p).dot(np.conj(ac).T)
                )

            elif shift_type == 2:
                # no shift
                hk = ac.dot(ee1).dot(np.conj(ac).T)

            else:
                if rank == 0:
                    print("'shift_type' Not Recognized")
                comm.Abort()

            # Enforce Hermiticity (just in case...)
            Hks_local[ikl, :, :, ispin] = 0.5 * (hk + np.conj(hk.T))

    Hks = gather_full(Hks_local, npool)
    if rank != 0:
        return None
    return np.ascontiguousarray(np.transpose(Hks, axes=(1, 2, 0, 3)))


def do_build_pao_hamiltonian(data_controller):
    """Build the PAO Hamiltonian and optionally handle symmetry expansion and non-orthogonality.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``U``, ``my_eigsmat``.
        Required attributes: ``bnd``, ``nawf``, ``nspin``, ``nkpnts``,
        ``shift``, ``shift_type``, ``expand_wedge``, ``acbn0``,
        ``nk1``, ``nk2``, ``nk3``.

    Returns
    -------
    None
        Adds or updates the following entries in ``data_controller.data_arrays``
        and ``data_controller.data_attributes``:

        - ``Hks`` : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``,
          complex — the PAO Hamiltonian in k-space.

        Updates attribute: ``nkpnts = nk1 * nk2 * nk3``.

    Notes
    -----
    Calls :func:`build_Hks` to construct :math:`H(\\mathbf{k})`.  When
    ``expand_wedge`` is ``True``, the symmetry-reduced wedge is expanded to
    the full Brillouin zone via :func:`open_grid_wrapper`.  When ``acbn0``
    is ``True``, the non-orthogonal correction :math:`H \\to S^{1/2} H S^{1/2}`
    is applied via :func:`do_non_ortho` on rank 0 and the result is written
    to disk with :meth:`DataController.write_Hk_acbn0`; execution then exits.
    """
    # ------------------------------
    # Building the PAO Hamiltonian
    # ------------------------------
    arry, attr = data_controller.data_dicts()

    arry['Hks'] = build_Hks(data_controller)

    ashape = (attr['nawf'], attr['nawf'], attr['nk1'], attr['nk2'], attr['nk3'], attr['nspin'])

    if attr['expand_wedge']:
        from .pao_sym import open_grid_wrapper

        # open_grid_wrapper reads Hks on every rank, so undo the rank-0-only
        # layout build_Hks returns.  The IBZ wedge is small compared with the
        # full grid, so this replication is cheap.
        data_controller.broadcast_single_array('Hks')

        # Expands both Hks and (when present) Sks from the IBZ wedge
        # to the full BZ.  Both transform identically under the point
        # group: A(Rk) = U(R,k) A(k) U(R,k)^†.
        open_grid_wrapper(data_controller)

    attr['nkpnts'] = np.prod(ashape[2:5])

    # NOTE: Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to
    # apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
    # acbn0 flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
    # acbn0 flag == 1 - makes H orthogonal (rotated basis)

    #  Reshape Hks to (nawf, nawf, nk1, nk2, nk3, nspin) for the IFFT.
    #  When expand_wedge=False (full BZ already provided by QE, nosym=noinv=.t.),
    #  open_grid_wrapper is skipped so the reshape must be done unconditionally.
    #  Skip the reshape if Hks is still on a reduced (IBZ) grid, as is the
    #  case for the ACBN0 path (which only needs Hks at the IBZ k-points and
    #  exits before any FFT to real space).
    #  Also skip when acbn0=True: do_non_ortho (called below) needs the flat
    #  4-D form (nawf, nawf, nkpnts, nspin); the 6-D reshape would break it
    #  whenever nkpnts == nk1*nk2*nk3 (nosym=noinv=.true. grids, or when
    #  expand_wedge has already filled the full BZ for eACBN0).
    if rank == 0 and not attr.get('acbn0', False) and arry['Hks'].size == int(np.prod(ashape)):
        arry['Hks'] = np.reshape(arry['Hks'], ashape)

        # Shift the Fermi energy to zero
        # tshape = (ashape[0], ashape[1], nkpnts, ashape[5])
        # Ef = E_Fermi(arry['Hks'].reshape(tshape), data_controller)
        # Ef = 0
        # dinds = np.diag_indices(attr['nawf'])
        # arry['Hks'][dinds[0], dinds[1]] -= Ef

    if attr['acbn0']:
        import sys

        if rank == 0:
            from ..projection.do_non_ortho import do_non_ortho

            # This is needed for consistency of the ordering of the matrix elements
            # Important in ACBN0 file writing
            arry['Sks'] = np.transpose(arry['Sks'], (1, 0, 2))

            # tshape = (ashape[0], ashape[1], nkpnts, ashape[5])
            arry['Hks'] = do_non_ortho(arry['Hks'], arry['Sks'])

            data_controller.write_Hk_acbn0()
        sys.exit(0)
        comm.Barrier()


def do_Hks_to_HRs(data_controller):
    """Transform the k-space PAO Hamiltonian to real space via inverse FFT.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``Hks`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``).
        Optional array: ``Sks`` (shape ``(nawf, nawf, nk1, nk2, nk3)``) —
        read and transformed if present.

    Returns
    -------
    None
        Adds the following entries to ``data_controller.data_arrays``:

        - ``HRs`` : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``,
          complex — the real-space Hamiltonian, broadcast to all MPI ranks.
        - ``SRs`` : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3)`` — the
          real-space overlap matrix (only when ``Sks`` is available).

    Notes
    -----
    Only MPI rank 0 performs the inverse FFT.  The result is broadcast via
    :meth:`DataController.broadcast_single_array` so that all ranks carry
    an identical copy of ``HRs``.
    """
    from scipy import fftpack as FFT

    arry, attr = data_controller.data_dicts()

    # ----------------------------------------------------------
    # Define the Hamiltonian and overlap matrix in real space:
    #   HRs and SRs (noinv and nosym = True in pw.x)
    # ----------------------------------------------------------
    if rank == 0:
        # Original k grid to R grid
        arry['HRs'] = FFT.ifftn(arry['Hks'], axes=[2, 3, 4])
