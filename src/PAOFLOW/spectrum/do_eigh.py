import numpy as np
from numpy import linalg as npl


def get_degeneracies(E_k, bnd):
    """Detect degenerate eigenvalue groups for all k-points and spin channels.

    Parameters
    ----------
    E_k : np.ndarray, shape ``(nkpnts, nawf, nspin)``
        Eigenvalues in eV.
    bnd : int
        Only bands with indices ``< bnd`` are considered.

    Returns
    -------
    list of list of list of np.ndarray
        ``all_degen[ispin][ik]`` is a list of 1-D integer arrays, each
        containing the band indices that belong to a degenerate group at
        k-point ``ik`` and spin channel ``ispin``.  Groups involving at
        least two bands are reported; singletons are omitted.

    Notes
    -----
    Degeneracy is determined by rounding eigenvalues to 5 decimal places
    and comparing adjacent values with a relative tolerance of
    :math:`10^{-5}` eV.
    """
    all_degen = []

    E_k_round = np.around(E_k, decimals=5)

    for ispin in range(E_k_round.shape[2]):
        by_spin = []
        for ik in range(E_k_round.shape[0]):
            by_kp = []
            eV = np.unique(
                E_k_round[ik, :, ispin][:-1][
                    np.isclose(
                        E_k_round[ik, :, ispin][1:], E_k_round[ik, :, ispin][:-1], atol=1.0e-5
                    )
                ]
            )

            for i in range(len(eV)):
                inds = np.where(np.isclose(E_k_round[ik, :, ispin], eV[i], atol=1.0e-5))[0]

                if len(inds) > 1 and np.all(inds < bnd):
                    by_kp.append(inds)

            by_spin.append(by_kp)

        all_degen.append(by_spin)

    return all_degen


def do_pao_eigh(data_controller):
    """Diagonalise the PAO Hamiltonian on the local k-point subset.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``Hksp`` (shape ``(snktot, nawf, nawf, nspin)``) —
        the Hamiltonian on the local k-point slice.
        Required attribute: ``bnd``.

    Returns
    -------
    None
        Adds the following entries to ``data_controller.data_arrays``:

        - ``E_k`` : np.ndarray, shape ``(snktot, nawf, nspin)`` — eigenvalues
          in ascending order (eV).
        - ``v_k`` : np.ndarray, shape ``(snktot, nawf, nawf, nspin)``, complex
          — corresponding eigenvectors (columns).
        - ``degen`` : nested list — degenerate band groups, as returned by
          :func:`get_degeneracies` restricted to the first ``bnd`` bands.
    """
    from numpy.linalg import eigh

    arrays, attributes = data_controller.data_dicts()

    snktot, nawf, _, nspin = arrays['Hksp'].shape

    arrays['E_k'] = np.zeros((snktot, nawf, nspin), dtype=float)
    arrays['v_k'] = np.zeros((snktot, nawf, nawf, nspin), dtype=complex)

    for ispin in range(nspin):
        # Batched Hermitian diagonalisation over the whole k-slice: one LAPACK
        # call per spin instead of a per-k Python loop. eigh broadcasts over the
        # leading axis, eigenvalues ascending, eigenvectors as columns.
        E, V = eigh(arrays['Hksp'][:, :, :, ispin], UPLO='U')
        arrays['E_k'][:, :, ispin] = E
        arrays['v_k'][:, :, :, ispin] = V

    # arrays['degen'] = get_degeneracies(arrays['E_k'], attributes['nawf'])
    arrays['degen'] = get_degeneracies(arrays['E_k'], attributes['bnd'])


def do_eigh_calc(HRaux, SRaux, kq, R, read_S):
    """Diagonalise the Hamiltonian on an arbitrary k-mesh via Fourier interpolation.

    Parameters
    ----------
    HRaux : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``, complex
        Real-space PAO Hamiltonian.
    SRaux : np.ndarray or None
        Real-space overlap matrix (shape ``(nawf, nawf, nk1, nk2, nk3)``),
        or ``None`` when ``read_S`` is ``False``.
    kq : np.ndarray, shape ``(nkpi, 3)``
        k-points in Cartesian coordinates.
    R : np.ndarray, shape ``(nrtot, 3)``
        Real-space lattice vectors.
    read_S : bool
        When ``True``, the generalised eigenvalue problem
        :math:`H v = \\varepsilon S v` is solved.

    Returns
    -------
    E_kp : np.ndarray, shape ``(nkpi, nawf, nspin)``
        Band eigenvalues (eV).
    v_kp : np.ndarray, shape ``(nkpi, nawf, nawf, nspin)``, complex
        Corresponding eigenvectors.
    """
    # Compute bands on a selected mesh in the BZ

    nkpi = kq.shape[0]
    nawf = HRaux.shape[0]
    nspin = HRaux.shape[-1]

    Hks_int = band_loop_H(HRaux, kq, R)

    if read_S:
        Sks_int = band_loop_S(SRaux, kq, R)

    E_kp = np.empty((nkpi, nawf, nspin), dtype=float)
    v_kp = np.empty((nkpi, nawf, nawf, nspin), dtype=complex)

    if read_S:
        # Generalised problem H v = e S v, batched over k via Cholesky S = L Lᴴ:
        # M = L⁻¹ H L⁻ᴴ is Hermitian, eigh(M) gives e and y; v = L⁻ᴴ y. This
        # mirrors scipy.linalg.eigh(H, S) for every k in a few LAPACK calls.
        S = np.moveaxis(Sks_int, 2, 0)  # (nkpi, nawf, nawf)
        L = npl.cholesky(S)
        Linv = npl.inv(L)
        for ispin in range(nspin):
            H = np.moveaxis(Hks_int[:, :, :, ispin], 2, 0)  # (nkpi, nawf, nawf)
            M = Linv @ H @ np.conj(np.transpose(Linv, (0, 2, 1)))
            E, y = npl.eigh(M, UPLO='U')
            v = np.conj(np.transpose(Linv, (0, 2, 1))) @ y
            E_kp[:, :, ispin] = E
            v_kp[:, :, :, ispin] = v
    else:
        for ispin in range(nspin):
            H = np.moveaxis(Hks_int[:, :, :, ispin], 2, 0)  # (nkpi, nawf, nawf)
            E, V = npl.eigh(H, UPLO='U')
            E_kp[:, :, ispin] = E
            v_kp[:, :, :, ispin] = V

    return (E_kp, v_kp)


### R_wght assumed to be 1
def band_loop_H(HRaux, kq, R):
    """Evaluate the Bloch Hamiltonian :math:`H(\\mathbf{k})` on a set of k-points.

    Parameters
    ----------
    HRaux : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``, complex
        Real-space Hamiltonian.
    kq : np.ndarray, shape ``(nkpi, 3)``
        k-points in Cartesian coordinates.
    R : np.ndarray, shape ``(nrtot, 3)``
        Real-space lattice vectors in units of :math:`1/a_{\\rm lat}`.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nkpi, nspin)``, complex
        :math:`H(\\mathbf{k})` evaluated at each requested k-point.
    """
    nkpi = kq.shape[0]
    nawf, _, nk1, nk2, nk3, nspin = HRaux.shape
    auxh = np.empty((nawf, nawf, nkpi, nspin), dtype=complex)
    HRaux = np.reshape(HRaux, (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')

    for ik in range(nkpi):
        for ispin in range(nspin):
            auxh[:, :, ik, ispin] = np.sum(
                HRaux[:, :, :, ispin] * np.exp(2.0j * np.pi * kq[ik, :].dot(R[:, :].T)), axis=2
            )

    return auxh


def band_loop_S(SRaux, kq, R):
    """Evaluate the overlap matrix :math:`S(\\mathbf{k})` on a set of k-points.

    Parameters
    ----------
    SRaux : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3)``, complex
        Real-space overlap matrix.
    kq : np.ndarray, shape ``(nkpi, 3)``
        k-points in Cartesian coordinates.
    R : np.ndarray, shape ``(nrtot, 3)``
        Real-space lattice vectors in units of :math:`1/a_{\\rm lat}`.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nkpi)``, complex
        :math:`S(\\mathbf{k})` evaluated at each requested k-point.
    """
    nkpi = kq.shape[0]
    nawf, _, nk1, nk2, nk3 = SRaux.shape
    auxs = np.empty((nawf, nawf, nkpi), dtype=complex)
    SRaux = np.reshape(SRaux, (nawf, nawf, nk1 * nk2 * nk3), order='C')

    for ik in range(nkpi):
        auxs[:, :, ik] = np.sum(
            SRaux[:, :, :] * np.exp(2.0j * np.pi * kq[ik, :].dot(R[:, :].T)), axis=2
        )

    return auxs
