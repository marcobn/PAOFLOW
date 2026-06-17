"""Rotate a fully-relativistic Hamiltonian from the J basis to the lm basis.

Fully-relativistic QE calculations (``dftSO=True``) express the Hamiltonian in
the coupled total-angular-momentum basis ``|j, m_j>``.  This module rotates the
PAOFLOW native real-space Hamiltonian into the QE real-spherical-harmonic (lm)
basis, in ``global_spin_block`` ordering -- all spin-up lm orbitals first, then
all spin-down lm orbitals -- which matches the layout produced by
``PAOFLOW.adhoc_spin_orbit``.  After the rotation the orbital character (s/p/d,
m) is well defined, so the orbital-resolved band routines
(:func:`orbital_projected_bands`, :func:`species_orbital_projected_bands`,
:func:`character_projected_bands`) can be used.

Conventions follow Quantum ESPRESSO / ``projwfc.x``:

* p orbitals:  ``p_z, p_x, p_y``
* d orbitals:  ``d_z2, d_zx, d_zy, d_x2-y2, d_xy``

The per-shell transformation matrices below are taken from a validated
notebook (J basis -> QE real harmonics, ``global_spin_block`` order) and have
been checked for unitarity.  Only s, p and d shells (l = 0, 1, 2) are
implemented.
"""

import numpy as np

# Spectroscopic label per angular momentum, for synthesised basis labels.
_L_LETTER = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

# QE / projwfc.x real spherical-harmonic order within each l shell (the lm,
# "after" ordering).  p: pz, px, py.  d: dz2, dzx, dzy, dx2-y2, dxy.
ORBITALS_QE = {0: ['s'], 1: ['pz', 'px', 'py'], 2: ['dz2', 'dzx', 'dzy', 'dx2-y2', 'dxy']}


def n_real_orbitals_l(l):
    """Number of real (lm) orbitals for angular momentum ``l``."""
    if l == 0:
        return 1
    if l == 1:
        return 3
    if l == 2:
        return 5
    raise NotImplementedError('Only l = 0, 1, 2 (s, p, d) are implemented.')


def shell_dimension(l):
    """Spinful dimension of one angular-momentum shell, 2*(2l+1)."""
    return 2 * n_real_orbitals_l(l)


def spin_block_to_orbital_spin(U, norb):
    """Reorder rows from ``[all up][all down]`` to ``[orb1 up, orb1 down, ...]``."""
    perm = []
    for iorb in range(norb):
        perm.append(iorb)
        perm.append(iorb + norb)
    return U[perm, :]


def U_s_QE(output_order='spin_block'):
    """l = 0 block, J basis -> QE real basis.  Columns |1/2,-1/2>, |1/2,+1/2>."""
    return np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )


def U_p_QE(output_order='spin_block'):
    """l = 1 block, J basis -> QE real basis.  QE order p_z, p_x, p_y."""
    sqrt = np.sqrt
    i = 1j

    U = np.array(
        [
            [0, -1 / sqrt(3), 0, 0, sqrt(2 / 3), 0],
            [-1 / sqrt(3), 0, 0, 1 / sqrt(6), 0, -1 / sqrt(2)],
            [i / sqrt(3), 0, 0, -i / sqrt(6), 0, -i / sqrt(2)],
            [1 / sqrt(3), 0, 0, sqrt(2 / 3), 0, 0],
            [0, -1 / sqrt(3), 1 / sqrt(2), 0, -1 / sqrt(6), 0],
            [0, -i / sqrt(3), -i / sqrt(2), 0, -i / sqrt(6), 0],
        ],
        dtype=complex,
    )

    if output_order == 'spin_block':
        return U
    if output_order == 'orbital_spin':
        return spin_block_to_orbital_spin(U, norb=3)
    raise ValueError("output_order must be 'spin_block' or 'orbital_spin'.")


def U_d_QE(output_order='spin_block'):
    """l = 2 block, J basis -> QE real basis.  QE order d_z2, d_zx, d_zy, d_x2-y2, d_xy."""
    sqrt = np.sqrt
    i = 1j

    U = np.array(
        [
            [0, 0, -sqrt(2 / 5), 0, 0, 0, 0, sqrt(3 / 5), 0, 0],
            [0, -sqrt(3 / 10), 0, 1 / sqrt(10), 0, 0, 1 / sqrt(5), 0, -sqrt(2 / 5), 0],
            [0, i * sqrt(3 / 10), 0, i / sqrt(10), 0, 0, -i / sqrt(5), 0, -i * sqrt(2 / 5), 0],
            [-sqrt(2 / 5), 0, 0, 0, 0, 1 / sqrt(10), 0, 0, 0, 1 / sqrt(2)],
            [i * sqrt(2 / 5), 0, 0, 0, 0, -i / sqrt(10), 0, 0, 0, i / sqrt(2)],
            [0, sqrt(2 / 5), 0, 0, 0, 0, sqrt(3 / 5), 0, 0, 0],
            [1 / sqrt(10), 0, -sqrt(3 / 10), 0, 0, sqrt(2 / 5), 0, -1 / sqrt(5), 0, 0],
            [-i / sqrt(10), 0, -i * sqrt(3 / 10), 0, 0, -i * sqrt(2 / 5), 0, -i / sqrt(5), 0, 0],
            [0, 0, 0, sqrt(2 / 5), 1 / sqrt(2), 0, 0, 0, 1 / sqrt(10), 0],
            [0, 0, 0, i * sqrt(2 / 5), -i / sqrt(2), 0, 0, 0, i / sqrt(10), 0],
        ],
        dtype=complex,
    )

    if output_order == 'spin_block':
        return U
    if output_order == 'orbital_spin':
        return spin_block_to_orbital_spin(U, norb=5)
    raise ValueError("output_order must be 'spin_block' or 'orbital_spin'.")


def U_l_QE(l, output_order='spin_block'):
    """Per-shell J -> QE real transformation matrix for l = 0, 1, or 2."""
    if l == 0:
        return U_s_QE(output_order=output_order)
    if l == 1:
        return U_p_QE(output_order=output_order)
    if l == 2:
        return U_d_QE(output_order=output_order)
    raise NotImplementedError('Only s, p, d blocks are implemented: l = 0, 1, 2.')


def block_diag(mats):
    """Block-diagonal constructor (complex), without scipy."""
    nrow = sum(M.shape[0] for M in mats)
    ncol = sum(M.shape[1] for M in mats)
    out = np.zeros((nrow, ncol), dtype=complex)
    r0 = c0 = 0
    for M in mats:
        r1 = r0 + M.shape[0]
        c1 = c0 + M.shape[1]
        out[r0:r1, c0:c1] = M
        r0, c0 = r1, c1
    return out


def global_spin_block_permutation(shells):
    """Row permutation from shell-by-shell spin blocks to global spin blocks.

    Maps row order ``[shell1 up, shell1 down, shell2 up, shell2 down, ...]`` to
    ``[all up orbitals, all down orbitals]``.  Acts on the rows of the
    block-diagonal ``U``; the columns stay in the original PAOFLOW/J order.
    """
    up_indices = []
    down_indices = []
    offset = 0
    for l in shells:
        n = n_real_orbitals_l(l)
        up_indices.extend(range(offset, offset + n))
        down_indices.extend(range(offset + n, offset + 2 * n))
        offset += 2 * n
    return np.array(up_indices + down_indices, dtype=int)


def build_U_spd(shells, output_order='global_spin_block', check_unitary=True):
    """Build the full J -> lm transformation matrix for a flat shell list.

    Parameters
    ----------
    shells : list of int
        Angular momenta in PAOFLOW/QE block order (0=s, 1=p, 2=d), one entry
        per (n, l) shell, concatenated over atoms in the Hamiltonian's atom
        order.
    output_order : str
        'global_spin_block' (default; all up orbitals then all down orbitals,
        matching ``adhoc_spin_orbit``), 'spin_block', or 'orbital_spin'.
    check_unitary : bool
        If True, raise when ``U`` is not unitary to 1e-12.

    Returns
    -------
    U : np.ndarray, shape ``(2N, 2N)``, complex
        Satisfies ``H_lm = U H_J U^dagger``.
    """
    if output_order == 'global_spin_block':
        blocks = [U_l_QE(l, output_order='spin_block') for l in shells]
        U = block_diag(blocks)
        U = U[global_spin_block_permutation(shells), :]
    elif output_order in ('spin_block', 'orbital_spin'):
        blocks = [U_l_QE(l, output_order=output_order) for l in shells]
        U = block_diag(blocks)
    else:
        raise ValueError(
            "output_order must be 'global_spin_block', 'spin_block', or "
            "'orbital_spin'."
        )

    if check_unitary:
        err = np.max(np.abs(U.conj().T @ U - np.eye(U.shape[1], dtype=complex)))
        if err > 1e-12:
            raise ValueError(
                'J -> lm transformation matrix is not unitary. '
                'max |U^dag U - I| = %g' % err
            )
    return U


def _collapse_jshells(jshells):
    """Collapse a fully-relativistic (j-resolved) shell list to ``(n,l)`` shells.

    Quantum ESPRESSO lists the atomic wavefunctions of a relativistic
    pseudopotential per ``(n, l, j)`` shell, so each ``l > 0`` shell appears as
    two consecutive entries with the same ``l`` (``j = l-1/2`` then
    ``j = l+1/2``); ``l = 0`` has only ``j = 1/2``.  The notebook's per-shell
    ``U`` blocks combine both ``j`` channels of one ``(n, l)`` shell into a
    single ``2(2l+1)`` block, so the two consecutive ``l > 0`` entries must be
    paired back into one.

    Example: ``[0, 0, 1, 1, 2, 2]`` (Mo) -> ``[0, 0, 1, 2]``;
    ``[0, 1, 1]`` (S) -> ``[0, 1]``.
    """
    out = []
    i = 0
    n = len(jshells)
    while i < n:
        l = int(jshells[i])
        out.append(l)
        if l > 0 and i + 1 < n and int(jshells[i + 1]) == l:
            i += 2  # consume the j = l+1/2 partner of this (n, l) shell
        else:
            i += 1
    return out


def _flat_shells(arry):
    """Flat ``(n,l)`` shell list over the Hamiltonian's atom order.

    Built from the j-resolved ``arry['shells']`` by collapsing each l>0 j-pair
    (see :func:`_collapse_jshells`).
    """
    shells = []
    for atom in arry['atoms']:
        shells.extend(_collapse_jshells(arry['shells'][atom]))
    return shells


def _build_lm_basis(arry):
    """Per-orbital lm basis (one spin block) matching ``global_spin_block`` order.

    Returns a list of ``len == N`` records ``{'atom', 'tau', 'l', 'm', 'label'}``
    ordered atom-by-atom, shell-by-shell, ``m = 1..2l+1`` (QE real-harmonic
    order).  ``nawf == 2N`` afterwards, so :func:`build_orbital_map` tiles this
    into the ``[up][down]`` layout to line up with ``v_k``.
    """
    basis = []
    for ia, atom in enumerate(arry['atoms']):
        tau = arry['tau'][ia]
        seen = {}
        for l in _collapse_jshells(arry['shells'][atom]):
            l = int(l)
            # Disambiguate repeated l on the same atom (e.g. two s shells).
            n = seen.get(l, 0)
            seen[l] = n + 1
            label = _L_LETTER.get(l, 'l%d' % l)
            if n > 0:
                label = '%s%d' % (label, n + 1)
            for m in range(1, 2 * l + 2):
                basis.append(
                    {'atom': atom, 'tau': tau, 'l': l, 'm': m, 'label': label}
                )
    return basis


def _fmt_half(two_x):
    """Format a half-integer given as its numerator over 2 (e.g. 3 -> '3/2')."""
    return str(two_x // 2) if two_x % 2 == 0 else '%d/2' % two_x


def j_basis_labels(data_controller):
    """Per-column labels of the Hamiltonian **before** the transform (J basis).

    Each entry is ``'<site>_<l>_j<j>_mj<mj>'`` in the column order assumed for
    the fully-relativistic ``HRs``: atom-major, shell-by-shell, and within each
    ``(n,l)`` shell the ``j = l-1/2`` block (``2l`` states) followed by the
    ``j = l+1/2`` block (``2l+2`` states), each with ``m_j`` ascending. Length
    ``== nawf``. Site is ``'<species><site-index>'`` to disambiguate equivalent
    atoms.
    """
    arry, _ = data_controller.data_dicts()
    labels = []
    for ia, atom in enumerate(arry['atoms']):
        site = '%s%d' % (atom, ia)
        seen = {}
        for l in _collapse_jshells(arry['shells'][atom]):
            n = seen.get(l, 0)
            seen[l] = n + 1
            tag = _L_LETTER.get(l, 'l%d' % l) + ('' if n == 0 else '#%d' % (n + 1))
            j2_list = [1] if l == 0 else [2 * l - 1, 2 * l + 1]  # 2j per j block
            for j2 in j2_list:
                for mj2 in range(-j2, j2 + 1, 2):  # m_j ascending
                    labels.append(
                        '%s_%s_j%s_mj%s' % (site, tag, _fmt_half(j2), _fmt_half(mj2))
                    )
    return labels


def lm_basis_labels(data_controller):
    """Per-column labels of the Hamiltonian **after** the transform (lm basis).

    QE real spherical harmonics in ``global_spin_block`` order (the
    ``adhoc_soc`` layout): all spin-up lm orbitals first, then all spin-down,
    each block atom-major and shell-by-shell in :data:`ORBITALS_QE` order.
    Length ``== nawf``. Site is ``'<species><site-index>'``.
    """
    arry, _ = data_controller.data_dicts()
    spinless = []
    for ia, atom in enumerate(arry['atoms']):
        site = '%s%d' % (atom, ia)
        seen = {}
        for l in _collapse_jshells(arry['shells'][atom]):
            n = seen.get(l, 0)
            seen[l] = n + 1
            suff = '' if n == 0 else '#%d' % (n + 1)
            spinless.extend('%s_%s%s' % (site, orb, suff) for orb in ORBITALS_QE[l])
    return [s + '_up' for s in spinless] + [s + '_down' for s in spinless]


def j_to_lm_hamiltonian(data_controller, shells=None, check_unitary=True):
    """Rotate the real-space Hamiltonian from the J basis to the lm basis.

    Transforms ``arry['HRs']`` in place to the QE real-harmonic (lm) basis in
    ``global_spin_block`` ordering, refreshes ``arry['Hks']`` to stay
    consistent, and rebuilds ``arry['basis']`` as the lm spatial orbitals so the
    orbital-resolved band routines interpret the new basis correctly.

    Parameters
    ----------
    data_controller : DataController
        Must provide ``arry['HRs']`` (shape ``(nawf, nawf, nk1, nk2, nk3,
        nspin)``), and -- when ``shells`` is None -- ``arry['atoms']`` and
        ``arry['shells']`` (dict species -> list of l).
    shells : list of int, optional
        Flat shell list (0=s, 1=p, 2=d) in the Hamiltonian's atom/shell order.
        Defaults to the list built from ``arry['atoms']``/``arry['shells']``.
    check_unitary : bool
        Forwarded to :func:`build_U_spd`.

    Returns
    -------
    np.ndarray
        The transformation matrix ``U`` used (shape ``(nawf, nawf)``).
    """
    arry, attr = data_controller.data_dicts()

    if 'HRs' not in arry:
        raise Exception(
            "j_to_lm_hamiltonian requires the real-space Hamiltonian 'HRs'; "
            "run 'pao_hamiltonian' first."
        )

    HRs = np.asarray(arry['HRs'])
    if HRs.ndim != 6 or HRs.shape[0] != HRs.shape[1]:
        raise Exception(
            'j_to_lm_hamiltonian: expected HRs shape '
            '(nawf, nawf, nk1, nk2, nk3, nspin); got %s.' % (HRs.shape,)
        )

    if shells is None:
        shells = _flat_shells(arry)

    U = build_U_spd(shells, output_order='global_spin_block', check_unitary=check_unitary)

    if U.shape[0] != HRs.shape[0]:
        raise Exception(
            'j_to_lm_hamiltonian: the shell list implies a Hamiltonian '
            'dimension of %d, but HRs has dimension %d.\n'
            'shells = %s\n'
            'This usually means the atom/shell order differs from the '
            'Hamiltonian, the pseudopotential has extra radial channels, or a '
            "shell with l > 2 is present. Pass an explicit 'shells' list to "
            'match the Hamiltonian ordering.' % (U.shape[0], HRs.shape[0], shells)
        )

    # Rotate only the orbital axes: H_lm[a,b,...] = sum_ij U[a,i] H_J[i,j,...] U*[b,j].
    H_lm = np.einsum('ai,ij...,bj->ab...', U, HRs.astype(complex), U.conj(), optimize=True)

    arry['HRs'] = H_lm
    arry['Hks'] = np.fft.fftn(H_lm, axes=(2, 3, 4))

    # The basis is now lm; rebuild it (length N) so the orbital-resolved
    # routines tile it to the [up][down] layout via the nawf == 2N detection.
    arry['basis'] = _build_lm_basis(arry)

    return U
