def do_non_ortho(Hks, Sks):
    """Apply a non-orthogonality transformation to the PAO Hamiltonian.

    Parameters
    ----------
    Hks : np.ndarray, shape ``(nawf, nawf, nkpnts, nspin)``
        Orthogonal PAO Hamiltonian in k-space (as produced by
        ``projwfc`` or the PAO builder).
    Sks : np.ndarray, shape ``(nawf, nawf, nkpnts)`` or larger
        Overlap matrix in k-space.  Only the first ``nawf x nawf`` block
        is used.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nkpnts, nspin)``
        Symmetrically orthogonalised Hamiltonian
        :math:`S^{1/2} H S^{1/2}` for each k-point and spin channel.

    Notes
    -----
    When the PAO basis is non-orthogonal, the PAO Hamiltonian obtained from
    ``projwfc`` must be transformed to an orthogonal representation before
    diagonalisation.  The transformation is

    .. math::

        \\tilde{H}(\\mathbf{k}) = S^{1/2}(\\mathbf{k})\\, H(\\mathbf{k})\\,
                                    S^{1/2}(\\mathbf{k})

    where :math:`S^{1/2}` is the matrix square root of the overlap matrix,
    computed via :func:`scipy.linalg.sqrtm` for each k-point independently.
    The eigenvalues of :math:`\\tilde{H}` are identical to those of the
    generalised eigenvalue problem :math:`H \\mathbf{v} = E S \\mathbf{v}`.
    """
    import numpy as np
    from scipy import linalg as spl

    # Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(1/2)*Hks*Sks^(1/2)

    nawf, _, nkpnts, nspin = Hks.shape
    S2k = np.zeros((nawf, nawf, nkpnts), dtype=complex)
    for ik in range(nkpnts):
        S2k[:, :, ik] = spl.sqrtm(Sks[:nawf, :nawf, ik])

    Hks_no = np.zeros((nawf, nawf, nkpnts, nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            Hks_no[:, :, ik, ispin] = np.dot(S2k[:, :, ik], Hks[:, :, ik, ispin]).dot(S2k[:, :, ik])

    return Hks_no
