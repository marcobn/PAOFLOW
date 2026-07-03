"""Symmetry expansion of the electron-phonon derivative (P2, step B).

A single symmetry-reduced displacement gives the Hamiltonian derivative along
*one* direction ``d`` for the displaced atom.  The full Cartesian derivative
tensor ``dH/du_{kappa,alpha}`` (alpha = x, y, z) is recovered by applying the
site-symmetry operations of the atom: each rotation ``W`` maps the measured
response along ``d`` to the response along ``W.d`` (with the PAO orbitals rotated
by the Wigner-D representation and the lattice vectors permuted).  Collecting the
responses along the star ``{W.d}`` and solving the linear system

    response(n) = sum_alpha (n)_alpha * dH/du_{kappa,alpha}

for every generated direction ``n`` yields the three Cartesian derivatives
(exactly as phonopy symmetrises its force constants).

This module provides the convention-free core -- the Cartesian least-squares
solve, the site-symmetry selection and the fractional->Cartesian rotation bridge
that feeds :func:`PAOFLOW.hamiltonian.pao_sym.get_wigner`.  The PAO Wigner-D
rotation of the ``dV`` tensor itself is layered on top and validated against
explicitly computed Cartesian displacements.
"""

import numpy as np


def frac_to_cartesian_rotation(w_frac, a_vectors):
    """Convert a fractional rotation ``W`` to its Cartesian form.

    With lattice vectors as the *rows* of ``a_vectors`` (``A``), a fractional
    rotation acting on fractional coordinates ``f`` corresponds to the Cartesian
    rotation ``R = A^T W A^{-T}`` acting on Cartesian coordinates ``r = A^T f``.

    Parameters
    ----------
    w_frac : array_like, shape (3, 3)
        Rotation in fractional (crystal) coordinates.
    a_vectors : array_like, shape (3, 3)
        Lattice vectors as rows (any consistent length unit).

    Returns
    -------
    ndarray, shape (3, 3)
        The Cartesian rotation matrix.
    """
    A = np.asarray(a_vectors, dtype=float)
    W = np.asarray(w_frac, dtype=float)
    AT = A.T
    return AT @ W @ np.linalg.inv(AT)


def site_symmetry_rotations(sym_rot, shifts, frac_positions, atom_index, tol=1.0e-4):
    """Indices of the space-group operations that leave ``atom_index`` fixed.

    Parameters
    ----------
    sym_rot : array_like, shape (nsym, 3, 3)
        Fractional rotations.
    shifts : array_like, shape (nsym, 3)
        Fractional translations (may be ``None`` -> treated as zero).
    frac_positions : array_like, shape (natom, 3)
        Atomic positions in fractional coordinates.
    atom_index : int
        Atom whose site symmetry is requested.
    tol : float
        Fractional tolerance for the "maps to itself" test.

    Returns
    -------
    ndarray
        Integer indices into ``sym_rot`` of the operations fixing the atom.
    """
    sym_rot = np.asarray(sym_rot, dtype=float)
    pos = np.asarray(frac_positions, dtype=float)
    if shifts is None:
        shifts = np.zeros((sym_rot.shape[0], 3))
    shifts = np.asarray(shifts, dtype=float)

    r = pos[atom_index]
    keep = []
    for i, (W, t) in enumerate(zip(sym_rot, shifts)):
        image = W @ r + t
        diff = image - r
        diff -= np.round(diff)  # wrap into (-0.5, 0.5]
        if np.all(np.abs(diff) < tol):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def cartesian_derivatives_from_directional(directions, responses, rcond=None):
    """Least-squares Cartesian derivatives from directional responses.

    Solves ``response(n) = sum_alpha n_alpha D_alpha`` for the three Cartesian
    derivative tensors ``D_alpha`` given a set of unit directions ``n`` and their
    response tensors (all of identical shape).  At least three independent
    directions are required.

    Parameters
    ----------
    directions : array_like, shape (ndir, 3)
        Displacement directions (need not be normalised; the magnitude scales
        the corresponding response).
    responses : array_like, shape (ndir, *tensor_shape)
        Response tensor for each direction.
    rcond : float, optional
        Cutoff passed to :func:`numpy.linalg.lstsq`.

    Returns
    -------
    ndarray, shape (3, *tensor_shape)
        The Cartesian derivative tensors ``[D_x, D_y, D_z]``.
    """
    dirs = np.asarray(directions, dtype=float)
    resp = np.asarray(responses)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError('directions must have shape (ndir, 3).')
    if resp.shape[0] != dirs.shape[0]:
        raise ValueError('directions and responses must share the leading axis.')
    if np.linalg.matrix_rank(dirs, tol=1.0e-8) < 3:
        raise ValueError('need at least three independent directions to solve for x, y, z.')

    tensor_shape = resp.shape[1:]
    rhs = resp.reshape(dirs.shape[0], -1)
    sol, *_ = np.linalg.lstsq(dirs, rhs, rcond=rcond)  # (3, M)
    return sol.reshape((3,) + tensor_shape)


def build_dv_symmetry_operators(
    a_vectors, tau, alat, sym_rot, equiv_atom, shells_per_atom, atom_labels, ngrid
):
    """Prepare the crystal-symmetry operator set that rotates a supercell PAO
    operator (``H`` or ``dV``) exactly as :mod:`PAOFLOW.hamiltonian.pao_sym`.

    The returned operators act on a supercell-basis operator ``M(k)`` (Bloch
    transform ``M(k) = sum_R M(R) exp(+2 pi i k.R)``) via

    .. math::

        M(W k) = U_k \\, M(k) \\, U_k^{\\dagger} \\;[\\times U_{inv}]

    with ``U_k = U * exp(-2 pi i (shift[a_index] . k))``.  This convention has
    been validated to reproduce ``H(Wk)`` to machine precision for every space
    group operation of the Al 2x2x2 supercell.

    Parameters
    ----------
    a_vectors : (3, 3) array
        Lattice vectors as rows, in ``alat`` units (supercell cell).
    tau : (natom, 3) array
        Atomic positions in Cartesian (Bohr) coordinates.
    alat : float
        Lattice parameter (Bohr) used to normalise ``tau``.
    sym_rot : (nsym, 3, 3) array
        Fractional (crystal) rotation matrices.
    equiv_atom : (nsym, natom) int array
        ``equiv_atom[isym, i]`` is the atom that block ``i`` draws from under the
        operation (the ``map_equiv_atoms`` convention).
    shells_per_atom : dict
        Mapping ``atom_label -> list of orbital l`` (PAOFLOW ``arry['shells']``).
    atom_labels : sequence
        Per-atom species labels (``arry['atoms']``).
    ngrid : tuple(int, int, int)
        The supercell ``R``/``k`` grid sizes ``(n1, n2, n3)``.

    Returns
    -------
    dict
        ``{'symop', 'symop_cart', 'U', 'phase', 'a_index', 'inv_flag',
        'U_inv', 'equiv_atom', 'ngrid'}``.
    """
    import scipy.linalg as _LA

    from ..hamiltonian import pao_sym as ps

    a_vectors = np.asarray(a_vectors, dtype=float)
    tau = np.asarray(tau, dtype=float)
    equiv_atom = np.asarray(equiv_atom, dtype=int)

    symop = ps.correct_roundoff(np.asarray(sym_rot, dtype=float))
    inv_a = _LA.inv(a_vectors)
    symop_cart = np.array([inv_a @ symop[i] @ a_vectors for i in range(symop.shape[0])])
    symop_cart = ps.correct_roundoff(symop_cart, incl_hex=True, atol=1.0e-6)

    atom_pos = np.around(ps.correct_roundoff((tau / alat) @ inv_a), 6)

    shells = []
    a_index = []
    for i, lab in enumerate(atom_labels):
        ash = list(shells_per_atom[lab])
        shells += ash
        a_index += [i] * int(np.sum([2 * n + 1 for n in ash]))
    shells = np.array(shells)
    a_index = np.array(a_index)

    wigner, inv_flag = ps.get_wigner(symop_cart)
    wigner = ps.convert_wigner_d(wigner)
    U = ps.build_U_matrix(wigner, shells)
    U = ps.add_U_wyc(U, ps.map_equiv_atoms(a_index, equiv_atom))
    U_inv = ps.get_inv_op(shells)
    phase = ps.get_phase_shifts(atom_pos, symop, equiv_atom)

    return {
        'symop': symop,
        'symop_cart': symop_cart,
        'U': U,
        'phase': phase,
        'a_index': a_index,
        'inv_flag': np.asarray(inv_flag, dtype=int),
        'U_inv': U_inv,
        'equiv_atom': equiv_atom,
        'ngrid': tuple(int(x) for x in ngrid),
    }


def rotate_dV_under_symmetry(dV, isym, ops, chunk_bytes=128 * 1024 * 1024):
    """Rotate a supercell ``dV`` operator by space-group operation ``isym``.

    ``dV`` is the electronic response (a Hermitian-structured PAO operator) to
    displacing one supercell atom ``kappa`` along a direction ``d``.  The result
    is the response of the *image* atom ``equiv_atom[isym, kappa]`` displaced
    along ``symop_cart[isym] . d`` -- i.e. the caller relabels the displaced atom
    and direction accordingly.

    Parameters
    ----------
    dV : ndarray, shape (nawf, nawf, n1, n2, n3, nspin)
        Supercell real-space derivative.
    isym : int
        Index of the operation in ``ops``.
    ops : dict
        Operator set from :func:`build_dv_symmetry_operators`.
    chunk_bytes : int, optional
        Approximate memory budget (bytes) for the per-chunk working arrays of the
        change-of-basis step.  The ``(k, spin)`` batch is processed in blocks so
        peak memory stays bounded to roughly ``dVk`` + ``out`` + one chunk,
        rather than several ``nawf**2 * N_tot`` copies (many GB for large
        supercells).  Larger values trade memory for slightly larger GEMMs.

    Returns
    -------
    ndarray
        Rotated ``dV`` of identical shape.
    """
    dV = np.asarray(dV)
    nawf = dV.shape[0]
    n1, n2, n3 = ops['ngrid']
    ntot = n1 * n2 * n3
    nspin = dV.shape[5]

    U = ops['U'][isym]
    Ud = np.conj(U.T)
    W = np.asarray(ops['symop'][isym])
    shift = ops['phase'][isym]  # (natom, 3)
    a_index = ops['a_index']
    inv_flag = bool(ops['inv_flag'][isym])
    U_inv = ops['U_inv']

    kf1 = np.fft.fftfreq(n1, 1.0 / n1).astype(int)
    kf2 = np.fft.fftfreq(n2, 1.0 / n2).astype(int)
    kf3 = np.fft.fftfreq(n3, 1.0 / n3).astype(int)

    # Bloch transform (e^{+2 pi i k.R}); matches the eliashberg convention.
    dVk = np.fft.ifftn(dV, axes=(2, 3, 4)) * ntot

    # Integer k-frequencies on the grid (C order matching the reshape below).
    KF1, KF2, KF3 = np.meshgrid(kf1, kf2, kf3, indexing='ij')
    kfrac = np.stack([KF1.ravel() / n1, KF2.ravel() / n2, KF3.ravel() / n3], axis=0)  # (3, nk)

    # Per-orbital phase exp(-2 pi i shift.k) for every k at once: (nawf, nk).
    shift_orb = shift[a_index]  # (nawf, 3)
    ph = np.exp(-2.0j * np.pi * (shift_orb @ kfrac))  # (nawf, nk)

    # Scatter map k -> k' = (W k) (a permutation of the flat k-index).
    newf = W @ np.stack([KF1.ravel(), KF2.ravel(), KF3.ravel()], axis=0)  # (3, nk)
    j1 = np.rint(newf[0]).astype(int) % n1
    j2 = np.rint(newf[1]).astype(int) % n2
    j3 = np.rint(newf[2]).astype(int) % n3
    tgt = (j1 * n2 + j2) * n3 + j3  # target flat k-index

    # U_k M U_k^dagger  with  U_k = U diag(ph)  reduces to
    #   U [ dVk * outer(ph, conj(ph)) ] U^dagger,
    # i.e. an elementwise phase then a single k-independent change of basis.
    # The batch (k, spin) is processed in chunks so the two contractions are
    # single BLAS-sized GEMMs (batched matmul over 1000s of tiny matrices is
    # latency bound), while peak memory stays bounded to ~dVk + out + one chunk
    # -- a fully vectorised version would hold several nawf^2 * N_tot copies,
    # which is many GB for large supercells.
    dVk_flat = dVk.reshape(nawf, nawf, ntot, nspin)
    out = np.empty_like(dVk_flat)

    per_k = nawf * nawf * max(nspin, 1) * dVk.dtype.itemsize
    chunk = int(np.clip(chunk_bytes // max(per_k, 1), 1, ntot))
    Uinv_b = U_inv[:, :, None, None] if inv_flag else None
    for c0 in range(0, ntot, chunk):
        c1 = min(c0 + chunk, ntot)
        nc = c1 - c0
        nbc = nc * nspin
        Mc = (
            dVk_flat[:, :, c0:c1, :] * ph[:, None, c0:c1, None] * np.conj(ph)[None, :, c0:c1, None]
        ).reshape(nawf, nawf, nbc)
        # T[p, j] = sum_i U[p, i] Mc[i, j];  THP[p, q] = sum_j T[p, j] conj(U[q, j]).
        Tc = (U @ Mc.reshape(nawf, nawf * nbc)).reshape(nawf, nawf, nbc)
        Hc = (Tc.transpose(0, 2, 1).reshape(nawf * nbc, nawf) @ Ud).reshape(nawf, nbc, nawf)
        Hc = Hc.transpose(0, 2, 1).reshape(nawf, nawf, nc, nspin)
        if inv_flag:
            Hc = Hc * Uinv_b
        out[:, :, tgt[c0:c1], :] = Hc

    del dVk, dVk_flat  # free the k-space input before the final transform
    out = out.reshape(nawf, nawf, n1, n2, n3, nspin)
    dV_rot = np.fft.fftn(out, axes=(2, 3, 4)) / ntot
    if not np.iscomplexobj(dV):
        dV_rot = dV_rot.real
    return dV_rot


def expand_directional_responses(directional, ops, tol=1.0e-4):
    """Symmetry-expand a reduced set of directional ``dV`` responses.

    A symmetry-reduced calculation (``displacement_mode='symmetry'``) measures
    the Hamiltonian derivative along a *single* direction per inequivalent atom.
    For each measured entry the site-symmetry operations of the displaced atom
    are applied to generate the star ``{W.d}`` of directions with their rotated
    ``dV`` tensors, so that every displaced atom acquires at least three linearly
    independent directions -- the input the Cartesian least-squares solve
    (:func:`cartesian_derivatives_from_directional`) requires.

    Parameters
    ----------
    directional : list[dict]
        Measured responses ``{sc_atom, displacement, dV}`` (Cartesian
        ``displacement``; supercell-basis ``dV``).
    ops : dict
        Operator set from :func:`build_dv_symmetry_operators` (of the reference
        supercell).
    tol : float
        Tolerance for treating two (anti-)parallel directions as duplicates.

    Returns
    -------
    list[dict]
        Expanded list of ``{sc_atom, displacement, dV}`` entries.
    """
    symop_cart = ops['symop_cart']
    equiv = ops['equiv_atom']
    nsym = symop_cart.shape[0]

    expanded = []
    for entry in directional:
        k0 = int(entry['sc_atom'])
        d0 = np.asarray(entry['displacement'], dtype=float)
        dV0 = np.asarray(entry['dV'])
        seen = []
        for isym in range(nsym):
            if int(equiv[isym, k0]) != k0:
                continue  # site-symmetry operations only
            d_new = symop_cart[isym] @ d0
            norm = np.linalg.norm(d_new)
            if norm < 1.0e-12:
                continue
            u = d_new / norm
            if any(np.allclose(u, s, atol=tol) or np.allclose(u, -s, atol=tol) for s in seen):
                continue
            seen.append(u)
            dV_new = rotate_dV_under_symmetry(dV0, isym, ops)
            expanded.append({'sc_atom': k0, 'displacement': d_new.tolist(), 'dV': dV_new})
    return expanded
