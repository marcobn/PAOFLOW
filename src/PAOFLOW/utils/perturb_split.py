def perturb_split(rot_op1, rot_op2, v_k, degen, return_v_k=False):
    """Project two operators onto the Bloch eigenstate basis with degenerate-subspace diagonalisation.

    Parameters
    ----------
    rot_op1 : np.ndarray, shape ``(nawf, nawf)``
        First operator matrix in the original PAO basis.  The degenerate
        subspaces are diagonalised with respect to this operator.
    rot_op2 : np.ndarray, shape ``(nawf, nawf)``
        Second operator matrix in the original PAO basis.  Projected using
        the eigenvectors obtained by diagonalising ``rot_op1``.
    v_k : np.ndarray, shape ``(nawf, bnd)``
        Bloch eigenvector matrix at a single k-point (columns are eigenstates).
    degen : list of array_like
        List of degenerate subspace index sets at this k-point.  Each element
        is an array of indices corresponding to a degenerate manifold.  An
        empty list indicates no degeneracies.
    return_v_k : bool, optional
        If ``True``, also return the modified eigenvector matrix after
        degenerate-subspace rotations.  Default is ``False``.

    Returns
    -------
    op1 : np.ndarray, shape ``(nawf, nawf)``
        ``rot_op1`` projected onto the (modified) Bloch eigenstate basis.
    op2 : np.ndarray, shape ``(nawf, nawf)``
        ``rot_op2`` projected onto the (modified) Bloch eigenstate basis.
    v_k_temp : np.ndarray, shape ``(nawf, bnd)``
        Modified eigenvector matrix (returned only when ``return_v_k=True``).

    Notes
    -----
    When no degenerate subspaces are present (``len(degen) == 0``), the
    projection is a straightforward unitary transformation:

    .. math::

        O = V^\\dagger A V

    where :math:`V` = ``v_k`` and :math:`A` is either operator.

    For each degenerate manifold :math:`\\mathcal{D}` (indices ``ll`` to
    ``ul``), the function diagonalises ``op1`` restricted to that block:

    .. math::

        \\text{eigh}\\left( O_1[\\mathcal{D}, \\mathcal{D}] \\right)
        \\rightarrow \\text{eigenvalues},\\; W_{\\mathcal{D}}

    and applies the rotation :math:`V_{\\mathcal{D}} \\leftarrow V_{\\mathcal{D}} W_{\\mathcal{D}}`
    to lift the degeneracy.  Both operators are then projected using the
    updated eigenvectors.  This approach is used throughout PAOFLOW to obtain
    well-defined momentum and curvature matrix elements at degenerate k-points.
    """
    import numpy as np
    from scipy import linalg as LAN

    op1 = np.dot(np.conj(v_k.T), np.dot(rot_op1, v_k))
    if len(degen) == 0:
        op2 = np.dot(np.conj(v_k.T), np.dot(rot_op2, v_k))
        if return_v_k:
            return op1, op2, np.array([[]])
        else:
            return op1, op2

    v_k_temp = np.copy(v_k)

    for i in range(len(degen)):
        # degenerate subspace indices upper and lower lim
        ll = degen[i][0]
        ul = degen[i][-1] + 1

        # diagonalize in degenerate subspace
        vals, weight = LAN.eigh(op1[ll:ul, ll:ul])

        # linear combination of eigenvectors of H that diagonalize
        v_k_temp[:, ll:ul] = np.dot(v_k_temp[:, ll:ul], weight)

    # return new operator in non degenerate basis
    op1 = np.dot(np.conj(v_k_temp.T), np.dot(rot_op1, v_k_temp))
    op2 = np.dot(np.conj(v_k_temp.T), np.dot(rot_op2, v_k_temp))

    if return_v_k:
        return (op1, op2, v_k_temp)
    else:
        return (op1, op2)
