def spinor(l, j, m, spin):
    """Compute a numerical Clebsch-Gordan coefficient for a spinor.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    j : float
        Total angular momentum quantum number (:math:`j = l \\pm 1/2`).
    m : int
        Projection of the total angular momentum along z
        (:math:`m_j - 1/2` for the down component).
    spin : int
        Selects the spin-up (``0``) or spin-down (``1``) coefficient.

    Returns
    -------
    float
        The Clebsch-Gordan coefficient
        :math:`\\langle l, m_l; 1/2, \\pm 1/2 \\mid j, m_j \\rangle`.

    Notes
    -----
    The function aborts via ``quit()`` when an incompatible combination of
    ``j``, ``l``, or ``spin`` is supplied, printing a diagnostic message on
    rank 0.
    """
    import numpy as np
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    if spin != 0 and spin != 1:
        if rank == 0:
            print('Spinor - spin direction unknown')
        quit()
    if m < -l - 1 or m > l:
        if rank == 0:
            print('Spinor - m not allowed')
        quit()

    denom = 1.0 / (2.0 * l + 1.0)
    if abs(j - l - 0.5) < 1.0e-8:
        if spin == 0:
            spinor = np.sqrt((l + m + 1.0) * denom)
        if spin == 1:
            spinor = np.sqrt((l - m) * denom)
    elif abs(j - l + 0.5) < 1.0e-8:
        if m < -l + 1:
            spinor = 0.0
        else:
            if spin == 0:
                spinor = np.sqrt((l - m + 1.0) * denom)
            if spin == 1:
                spinor = -np.sqrt((l + m) * denom)
    else:
        if rank == 0:
            print('Spinor - j and l not compatible')
        quit()

    return spinor


def clebsch_gordan(nawf, sh_l, sh_j, spol):
    """Build the spin operator matrix in the coupled :math:`\\lvert j, m_j, l, s \\rangle` basis.

    Parameters
    ----------
    nawf : int
        Total number of atomic wave-function projectors (= 2 * number of
        spinor components).
    sh_l : list of int
        Orbital angular momentum quantum numbers :math:`l` for each shell.
    sh_j : list of float
        Total angular momentum quantum numbers :math:`j` for each shell,
        used to detect descending/ascending ordering in the pseudopotential.
    spol : int
        Spin polarisation component: ``0`` for x, ``1`` for y, ``2`` for z.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf)``, complex
        The spin operator :math:`S^{spol}` expressed in the
        :math:`\\lvert j, m_j, l, s \\rangle` basis.

    Notes
    -----
    The transformation matrices :math:`U_l` are built by contracting
    Clebsch-Gordan coefficients (from :func:`spinor`) for angular momenta
    :math:`l = 0, 1, 2, 3`.  The full basis-change matrix :math:`T_n` is
    assembled block-diagonally over all shells.  The spin operator is then

    .. math::

        S_j = T_n \\, S_l \\, T_n^\\top

    where :math:`S_l` is the Pauli spin operator in the
    :math:`\\lvert l, m_l, s, s_z \\rangle` basis.  Raises ``ValueError`` if the
    shell occupancies implied by ``sh_l`` do not sum to ``nawf``.
    """
    import numpy as np

    # Transformation matrices from the | l m s s_z > basis to the
    # | j mj l s > basis in the l-subspace
    #
    l = 0
    Ul0 = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)), dtype=float)
    Ul0[0, 1] = 1
    Ul0[1, 0] = 1

    l = 1
    Ul1 = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)), dtype=float)
    j = l - 0.5
    for m1 in range(1, 2 * l + 1):
        m = m1 - l
        Ul1[m1 - 1, 2 * (m1 - 1) + 1 - 1] = spinor(l, j, m, 0)
        Ul1[m1 - 1, 2 * (m1 - 1) + 4 - 1] = spinor(l, j, m, 1)
    j = l + 0.5
    for m1 in range(1, 2 * l + 2 + 1):
        m = m1 - l - 2
        if m1 == 1:
            Ul1[m1 + 2 * l - 1, 2 * (m1 - 1) + 2 - 1] = spinor(l, j, m, 1)
        elif m1 == 2 * l + 2:
            Ul1[m1 + 2 * l - 1, 2 * (m1 - 1) - 1 - 1] = spinor(l, j, m, 0)
        else:
            Ul1[m1 + 2 * l - 1, 2 * (m1 - 1) - 1 - 1] = spinor(l, j, m, 0)
            Ul1[m1 + 2 * l - 1, 2 * (m1 - 1) + 2 - 1] = spinor(l, j, m, 1)

    l = 2
    Ul2 = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)), dtype=float)
    j = l - 0.5
    for m1 in range(1, 2 * l + 1):
        m = m1 - l
        Ul2[m1 - 1, 2 * (m1 - 1) + 1 - 1] = spinor(l, j, m, 0)
        Ul2[m1 - 1, 2 * (m1 - 1) + 4 - 1] = spinor(l, j, m, 1)
    j = l + 0.5
    for m1 in range(1, 2 * l + 2 + 1):
        m = m1 - l - 2
        if m1 == 1:
            Ul2[m1 + 2 * l - 1, 2 * (m1 - 1) + 2 - 1] = spinor(l, j, m, 1)
        elif m1 == 2 * l + 2:
            Ul2[m1 + 2 * l - 1, 2 * (m1 - 1) - 1 - 1] = spinor(l, j, m, 0)
        else:
            Ul2[m1 + 2 * l - 1, 2 * (m1 - 1) - 1 - 1] = spinor(l, j, m, 0)
            Ul2[m1 + 2 * l - 1, 2 * (m1 - 1) + 2 - 1] = spinor(l, j, m, 1)

    l = 3
    Ul3 = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)), dtype=float)
    j = l - 0.5
    for m1 in range(1, 2 * l + 1):
        m = m1 - l
        Ul3[m1 - 1, 2 * (m1 - 1) + 1 - 1] = spinor(l, j, m, 0)
        Ul3[m1 - 1, 2 * (m1 - 1) + 4 - 1] = spinor(l, j, m, 1)
    j = l + 0.5
    for m1 in range(1, 2 * l + 2 + 1):
        m = m1 - l - 2
        if m1 == 1:
            Ul3[m1 + 2 * l - 1, 2 * (m1 - 1) + 2 - 1] = spinor(l, j, m, 1)
        elif m1 == 2 * l + 2:
            Ul3[m1 + 2 * l - 1, 2 * (m1 - 1) - 1 - 1] = spinor(l, j, m, 0)
        else:
            Ul3[m1 + 2 * l - 1, 2 * (m1 - 1) - 1 - 1] = spinor(l, j, m, 0)
            Ul3[m1 + 2 * l - 1, 2 * (m1 - 1) + 2 - 1] = spinor(l, j, m, 1)

    Ul = [Ul0, Ul1, Ul2, Ul3]

    Ul1_alt = np.roll(np.roll(Ul1, -2, axis=0), -2, axis=1)
    Ul2_alt = np.roll(np.roll(Ul2, -4, axis=0), -4, axis=1)
    Ul3_alt = np.roll(np.roll(Ul3, -6, axis=0), -6, axis=1)

    Ul_alt = [Ul0, Ul1_alt, Ul2_alt, Ul3_alt]

    # Build the full transformation matrix

    occ = [2, 6, 10, 14]

    ntot = 0
    for n in range(len(sh_l)):
        ntot += occ[sh_l[n]]
    if ntot != nawf:
        raise ValueError("Wrong number of shells when creating 'Sj'")

    Tn = np.zeros((ntot, ntot), dtype=float)

    n = 0
    for l in range(len(sh_l)):
        if sh_l[l] == 0:
            Tn[n : n + occ[sh_l[l]], n : n + occ[sh_l[l]]] = Ul[sh_l[l]]
        else:
            if (sh_l[l] - sh_j[l]) > 0.0:
                # the case when j are sorted in ascending order in the pseudo
                Tn[n : n + occ[sh_l[l]], n : n + occ[sh_l[l]]] = Ul[sh_l[l]]
            else:
                # the case when j are sorted in descending order in the pseudo
                Tn[n : n + occ[sh_l[l]], n : n + occ[sh_l[l]]] = Ul_alt[sh_l[l]]

        n += occ[sh_l[l]]

    # Pauli matrices (x,y,z)
    sP = 0.5 * np.array(
        [[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1.0j], [1.0j, 0.0]], [[1.0, 0.0], [0.0, -1.0]]]
    )

    # Spin operator matrix  in the basis of |l,m,s,s_z>
    Sl = np.zeros((nawf, nawf), dtype=complex)
    for i in range(0, nawf, 2):
        Sl[i, i] = sP[spol][0, 0]
        Sl[i, i + 1] = sP[spol][0, 1]
        Sl[i + 1, i] = sP[spol][1, 0]
        Sl[i + 1, i + 1] = sP[spol][1, 1]

    # Spin operator matrix  in the basis of |j,m_j,l,s>
    Sj = np.zeros((nawf, nawf), dtype=complex)
    Sj = np.dot(Tn, Sl).dot(Tn.T)

    return Sj
