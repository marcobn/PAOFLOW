"""Fold the supercell PAO Hamiltonian derivative to the primitive cell (P2).

The finite-difference derivative ``dV`` lives in the *supercell* PAO basis over
the supercell real-space grid.  Because the supercell is an exact tiling of the
primitive cell, every supercell orbital maps to ``(primitive orbital, cell
translation)`` and every supercell lattice vector combines with the sub-cell
translations to a primitive lattice vector.  Displacing the reference-cell atom
``kappa`` (translation ``T = 0``) therefore gives, after re-indexing,

    dV_prim_{ij}(R_p) = d<0, i | H | R_p, j> / du_{kappa, 0},

the primitive real-space electron-phonon derivative on the (denser) primitive
grid ``N_p = diag(S) * N_supercell`` -- i.e. back on the original unit-cell
k/R grid.  The atom mapping is done in **Cartesian coordinates**, so it is
robust to any difference in lattice orientation or atom ordering between codes.

Only diagonal supercell matrices are handled here (the common isotropic /
anisotropic-diagonal case); a general matrix would need a Smith-normal-form
enumeration of the sub-cells.
"""

import numpy as np


def supercell_atom_translations(phonon, tol=1.0e-4):
    """Map each supercell atom to its primitive atom and cell translation.

    Returns
    -------
    s2p : ndarray, shape (natom_sc,)
        Primitive-atom index (0..nprim-1) of every supercell atom.
    translations : ndarray, shape (natom_sc, 3), int
        Integer primitive-lattice translation of every supercell atom.
    """
    prim_cell = np.asarray(phonon.primitive.cell, dtype=float)
    prim_pos = np.asarray(phonon.primitive.positions, dtype=float)
    sc_pos = np.asarray(phonon.supercell.positions, dtype=float)
    natom_sc = len(sc_pos)
    nprim = len(prim_pos)
    inv_prim = np.linalg.inv(prim_cell)

    s2p = np.empty(natom_sc, dtype=int)
    translations = np.zeros((natom_sc, 3), dtype=int)
    for a in range(natom_sc):
        for p in range(nprim):
            frac = (sc_pos[a] - prim_pos[p]) @ inv_prim
            if np.allclose(frac, np.round(frac), atol=tol):
                s2p[a] = p
                translations[a] = np.round(frac).astype(int)
                break
        else:
            raise ValueError('Supercell atom %d has no primitive image (tol=%g).' % (a, tol))
    return s2p, translations


def _require_diagonal(supercell_matrix):
    S = np.asarray(supercell_matrix, dtype=int)
    if S.shape != (3, 3) or np.count_nonzero(S - np.diag(np.diagonal(S))) != 0:
        raise NotImplementedError('fold_dV_to_primitive currently supports diagonal supercells.')
    return np.diagonal(S).astype(int)


def _fft_integers(n):
    """FFT-ordered integer lattice indices [0, 1, ..., n//2, -(n-1)//2, ..., -1]."""
    return np.fft.fftfreq(n, d=1.0 / n).astype(int)


def fold_dV_to_primitive(dV_sc, s2p, translations, naw_per_atom, supercell_matrix):
    """Re-index a supercell derivative into the primitive electron-phonon tensor.

    Displacing atom ``kappa`` in supercell cell 0 and reading the supercell
    matrix element ``<T_A, i | H | R_sc + T_B, j>`` gives, by translational
    invariance, the primitive real-space electron-phonon tensor

        g_{ij}(R_e, R_p) = d<0, i | H | R_e, j> / du_{kappa, R_p},

    with the electron hopping ``R_e = S . R_sc + T_B - T_A`` (primitive grid) and
    the phonon cell ``R_p = -T_A`` (reduced to the sub-cell / commensurate q
    grid).  Using every bra cell ``T_A`` (not only the reference cell) recovers
    the full ``R_p`` dependence, i.e. the coupling at all ``|det S|``
    commensurate q-points.

    Parameters
    ----------
    dV_sc : ndarray
        ``(nawf_sc, nawf_sc, n1, n2, n3, nspin)`` supercell derivative
        (FFT-ordered real-space axes).
    s2p, translations : ndarray
        Output of :func:`supercell_atom_translations`.
    naw_per_atom : array_like, shape (natom_sc,)
        PAO orbitals on each supercell atom (same order as ``dV_sc``).
    supercell_matrix : array_like
        Diagonal ``(3, 3)`` supercell matrix.

    Returns
    -------
    ndarray
        ``(nawf_prim, nawf_prim, N1e, N2e, N3e, s1, s2, s3, nspin)`` electron-
        phonon tensor: ``R_e`` on the primitive grid (``Nie = S_ii * n_i``) and
        ``R_p`` on the sub-cell grid (``si = S_ii``).
    """
    dV_sc = np.asarray(dV_sc)
    naw = np.asarray(naw_per_atom, dtype=int)
    diag = _require_diagonal(supercell_matrix)

    n1, n2, n3, nspin = dV_sc.shape[2], dV_sc.shape[3], dV_sc.shape[4], dV_sc.shape[5]
    Ne = (int(diag[0] * n1), int(diag[1] * n2), int(diag[2] * n3))
    s1, s2, s3 = int(diag[0]), int(diag[1]), int(diag[2])

    orb_start = np.concatenate(([0], np.cumsum(naw)[:-1]))
    if orb_start[-1] + naw[-1] != dV_sc.shape[0]:
        raise ValueError('naw_per_atom does not sum to the supercell orbital count.')

    # Primitive orbital layout from the reference-cell (T == 0) atoms.
    nprim = int(s2p.max()) + 1
    naw_prim_atom = np.zeros(nprim, dtype=int)
    for a in range(len(naw)):
        if np.all(translations[a] == 0):
            naw_prim_atom[s2p[a]] = naw[a]
    if np.any(naw_prim_atom == 0):
        raise ValueError('Not every primitive atom has a reference-cell (T=0) image.')
    prim_orb_start = np.concatenate(([0], np.cumsum(naw_prim_atom)[:-1]))
    nawf_prim = int(naw_prim_atom.sum())

    g = np.zeros(
        (nawf_prim, nawf_prim, Ne[0], Ne[1], Ne[2], s1, s2, s3, nspin),
        dtype=dV_sc.dtype,
    )

    rsc = (_fft_integers(n1), _fft_integers(n2), _fft_integers(n3))

    for a in range(len(naw)):  # bra atom (any cell)
        pa = s2p[a]
        A0, nb = orb_start[a], naw[a]
        i0 = prim_orb_start[pa]
        Ta = translations[a]
        rp = ((-Ta) % diag).astype(int)  # phonon-cell sub-cell index
        for b in range(len(naw)):  # ket atom
            pb = s2p[b]
            B0, nk = orb_start[b], naw[b]
            j0 = prim_orb_start[pb]
            Tb = translations[b]
            re1 = (diag[0] * rsc[0] + Tb[0] - Ta[0]) % Ne[0]
            re2 = (diag[1] * rsc[1] + Tb[1] - Ta[1]) % Ne[1]
            re3 = (diag[2] * rsc[2] + Tb[2] - Ta[2]) % Ne[2]
            block = dV_sc[A0 : A0 + nb, B0 : B0 + nk]  # (nb, nk, n1, n2, n3, nspin)
            g[
                i0 : i0 + nb,
                j0 : j0 + nk,
                re1[:, None, None],
                re2[None, :, None],
                re3[None, None, :],
                rp[0],
                rp[1],
                rp[2],
                :,
            ] = block
    return g


def supercell_naw(phonon, configuration, pp_filenames, pp_dir):
    """PAO orbital count per supercell atom for a basis configuration."""
    import os

    from ..phonon.structure import _element_symbol
    from .basis import species_pao_orbitals

    cache = {}
    naw = []
    for sym in phonon.supercell.symbols:
        elem = _element_symbol(str(sym))
        if elem not in cache:
            cache[elem] = species_pao_orbitals(
                os.path.join(pp_dir, pp_filenames[elem]), configuration=configuration
            )
        naw.append(cache[elem])
    return np.asarray(naw, dtype=int)
