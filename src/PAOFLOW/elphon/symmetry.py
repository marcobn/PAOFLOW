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
