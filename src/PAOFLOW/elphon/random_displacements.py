r"""Random all-atom displacements and the local least-squares vertex.

Around a reference cell with little or no symmetry (a thermally displaced
special-displacement cell, an alloy supercell, a molecular-dynamics snapshot) the
Cartesian finite-difference route needs ``6 N`` displaced supercells: one ``+/-``
pair per atom and direction.  This module replaces them by ``M`` random
configurations in which *every* atom is displaced at once,

.. math::

    u^m_{\kappa} = d\, n^m_{\kappa}, \qquad |n^m_{\kappa}| = 1, \qquad m = 1 \dots M,

each with its mirror image :math:`-u^m`.  The antisymmetric difference of the PAO
Hamiltonian in the fixed reference basis,

.. math::

    D_m = \tfrac{1}{2}\,[H(+u^m) - H(-u^m)]
        = \sum_{\kappa\alpha} u^m_{\kappa\alpha}\, \partial_{\kappa\alpha} V + O(u^3),

is linear in the unknown derivatives, so every matrix element of
:math:`\partial_{\kappa\alpha} V` follows from a least-squares fit over the ``M``
configurations.  Locality keeps the fit small: the element between an orbital on
atom ``A`` and an orbital on atom ``B`` responds only to the atoms within
``cutoff`` of ``A`` or ``B`` (periodic minimum image), so it is fitted against
:math:`3\,|N(A) \cup N(B)|` unknowns instead of :math:`3N`.  The normal equations

.. math::

    G = \sum_m u^m (u^m)^T, \qquad B = \sum_m u^m D_m, \qquad
    y_2 = \sum_m |D_m|^2

are accumulated while the displaced Hamiltonians are built (one pair in memory
at a time) and solved per active set with a ridge term
:math:`\lambda = \text{ridge} \cdot \overline{\mathrm{diag}\, G_{SS}}`.  The
residual :math:`y_2 - 2\,\mathrm{Re}\,w^\dagger B + w^\dagger G w` is exact given
the solution and needs no stored configurations.  The forces of the same runs
give the harmonic force constants by the identical fit,
:math:`-\tfrac{1}{2}[F(+u) - F(-u)] = \Phi u`.
"""

import itertools
import os
import warnings

import numpy as np

from ..utils.constants import BOHR_RADIUS_ANGS, RYTOEV

#: Ry/Bohr^2 -> eV/Angstrom^2 (phonopy force-constant units).
RY_BOHR2_TO_EV_ANG2 = RYTOEV / BOHR_RADIUS_ANGS**2


def random_configurations(natoms: int, distance: float, nconfig: int, seed: int = 0) -> np.ndarray:
    """Random displacement patterns with every atom moved by ``distance``.

    Parameters
    ----------
    natoms : int
        Number of atoms in the supercell.
    distance : float
        Displacement amplitude of every atom (Bohr).
    nconfig : int
        Number of configurations ``M``.
    seed : int
        Seed of the ``numpy`` generator (reproducible manifests).

    Returns
    -------
    ndarray, shape (nconfig, natoms, 3)
        Cartesian displacements (Bohr); each atom row has norm ``distance`` and a
        direction uniform on the sphere.
    """
    rng = np.random.default_rng(int(seed))
    n = rng.standard_normal((int(nconfig), int(natoms), 3))
    n /= np.linalg.norm(n, axis=2)[:, :, None]
    return float(distance) * n


def neighbour_sets(positions, cell, cutoff) -> list:
    """Atoms within ``cutoff`` of every atom, the atom itself included.

    Distances are periodic minimum-image distances in the supercell (the 27
    neighbouring images are checked, so skewed cells are handled).  A displaced
    atom moves together with all its periodic images, which is why the active
    set of a Hamiltonian element does not depend on the lattice vector of the
    element: the nearest image is always the one that matters.

    Parameters
    ----------
    positions : array_like, shape (natoms, 3)
        Cartesian positions (Bohr).
    cell : array_like, shape (3, 3)
        Supercell lattice vectors as rows (Bohr).
    cutoff : float or None
        Locality radius (Bohr); ``None`` selects every atom.

    Returns
    -------
    list[ndarray]
        ``sets[kappa]`` are the sorted indices of the atoms in ``N(kappa)``.
    """
    pos = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    nat = len(pos)
    if cutoff is None:
        return [np.arange(nat) for _ in range(nat)]
    frac = pos @ np.linalg.inv(cell)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float) @ cell
    sets = []
    for k in range(nat):
        d = frac - frac[k]
        d -= np.round(d)
        cart = d @ cell
        dist = np.min(np.linalg.norm(cart[:, None, :] + shifts[None, :, :], axis=2), axis=1)
        sets.append(np.flatnonzero(dist <= float(cutoff) + 1.0e-8))
    return sets


class NormalEquations:
    """Streaming normal equations of a linear response model.

    Accumulates ``G = sum_m x_m x_m^T`` (``(n, n)``), ``B = sum_m x_m Y_m``
    (``(n, *shape)``) and ``y2 = sum_m |Y_m|^2`` (``shape``) one configuration at
    a time, so the response arrays never have to be stored together.

    Parameters
    ----------
    nunknown : int
        Number of unknown coefficients ``n`` (``3 * natoms`` for displacements).
    shape : tuple
        Shape of one response array ``Y_m``.
    dtype : dtype
        Element type of the responses.
    """

    def __init__(self, nunknown: int, shape, dtype=complex):
        self.G = np.zeros((int(nunknown), int(nunknown)))
        self.B = np.zeros((int(nunknown),) + tuple(shape), dtype=dtype)
        self.y2 = np.zeros(tuple(shape))
        self.nconfig = 0

    def add(self, x, y) -> None:
        """Add one configuration with coefficients ``x`` and response ``y``."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y)
        if x.size != self.G.shape[0] or y.shape != self.y2.shape:
            raise ValueError('configuration does not match the normal equations.')
        self.G += np.outer(x, x)
        for i in np.flatnonzero(x):
            self.B[i] += x[i] * y
        self.y2 += np.abs(y) ** 2
        self.nconfig += 1


def solve_normal_equations(neq: NormalEquations, groups, ridge: float = 1.0e-4):
    """Ridge least squares per active set; the solution overwrites ``neq.B``.

    Parameters
    ----------
    neq : NormalEquations
        Accumulated normal equations.
    groups : iterable of (active, index)
        ``active`` lists the unknown indices allowed to contribute to the block
        of responses selected by ``index`` (a tuple of integer arrays, one per
        trailing axis of ``neq.B``, combined with :func:`numpy.ix_`).  The blocks
        must not overlap; unknowns outside ``active`` are zero on that block.
    ridge : float
        Regularisation relative to the mean diagonal of ``G`` on the active set.

    Returns
    -------
    W : ndarray
        Fitted coefficients, same shape as ``neq.B`` (and the same array).
    residual : ndarray
        Per-element residual ``sum_m |Y_m - sum_i x_mi W_i|^2`` (shape of ``y2``),
        evaluated from the normal equations; cancellation limits it to about
        ``1e-16 * y2`` (a relative residual floor of ``1e-8``).
    """
    W = neq.B
    nunknown = W.shape[0]
    residual = np.array(neq.y2, copy=True)
    cache = {}
    for active, index in groups:
        active = np.asarray(active, dtype=int)
        key = active.tobytes()
        if key not in cache:
            GS = neq.G[np.ix_(active, active)]
            lam = float(ridge) * np.trace(GS) / len(active)
            inactive = np.setdiff1d(np.arange(nunknown), active)
            cache[key] = (np.linalg.inv(GS + lam * np.eye(len(active))), GS, inactive)
        Ainv, GS, inactive = cache[key]
        sel = np.ix_(active, *index)
        BS = W[sel]
        shp = BS.shape
        BS = BS.reshape(len(active), -1)
        WS = Ainv @ BS
        fit = -2.0 * np.real(np.sum(np.conj(WS) * BS, axis=0))
        fit += np.real(np.sum(np.conj(WS) * (GS @ WS), axis=0))
        residual[np.ix_(*index)] += fit.reshape(shp[1:])
        W[sel] = WS.reshape(shp)
        if inactive.size:
            W[np.ix_(inactive, *index)] = 0.0  # outside the active set the derivative is zero
    return W, np.maximum(residual, 0.0)


def _active_indices(atoms) -> np.ndarray:
    atoms = np.asarray(atoms, dtype=int)
    return (3 * atoms[:, None] + np.arange(3)[None, :]).ravel()


def fit_hamiltonian_derivatives(neq: NormalEquations, orbital_atoms, nsets, ridge: float = 1.0e-4):
    """Solve for ``dV_{kappa alpha}`` with the orbital-pair locality groups.

    Parameters
    ----------
    neq : NormalEquations
        ``B`` of shape ``(3 natoms, nawf, nawf, nrest)`` where ``nrest`` collects
        the lattice-vector and spin axes.
    orbital_atoms : array_like, shape (nawf,)
        Atom index of every supercell orbital.
    nsets : list[ndarray]
        Neighbour sets from :func:`neighbour_sets`.
    ridge : float
        Ridge parameter of :func:`solve_normal_equations`.

    Returns
    -------
    W : ndarray, shape (3 natoms, nawf, nawf, nrest)
        ``W[3 kappa + alpha]`` is the derivative with respect to ``u_{kappa alpha}``.
    residual : ndarray, shape (nawf, nawf, nrest)
        Per-element residual of the fit.
    pair_residual : ndarray, shape (natoms, natoms)
        Relative residual ``sqrt(sum res / sum y2)`` per atom pair ``(A, B)``.
    """
    nat = len(nsets)
    orbital_atoms = np.asarray(orbital_atoms, dtype=int)
    nawf = orbital_atoms.size
    if neq.B.shape[1:3] != (nawf, nawf):
        raise ValueError('orbital_atoms does not match the response shape.')
    orbs = [np.flatnonzero(orbital_atoms == k) for k in range(nat)]
    rest = np.arange(neq.B.shape[3])
    if all(len(s) == nat for s in nsets):
        groups = [(np.arange(3 * nat), (np.arange(nawf), np.arange(nawf), rest))]
    else:
        groups = []
        for a in range(nat):
            for b in range(nat):
                active = _active_indices(np.union1d(nsets[a], nsets[b]))
                groups.append((active, (orbs[a], orbs[b], rest)))
    W, residual = solve_normal_equations(neq, groups, ridge=ridge)

    pair_residual = np.zeros((nat, nat))
    for a in range(nat):
        for b in range(nat):
            sel = np.ix_(orbs[a], orbs[b], rest)
            y2 = float(np.sum(neq.y2[sel]))
            pair_residual[a, b] = np.sqrt(np.sum(residual[sel]) / y2) if y2 > 0 else 0.0
    return W, residual, pair_residual


def fit_force_constants(neq: NormalEquations, nsets, ridge: float = 1.0e-4):
    """Harmonic force constants from the forces of the random configurations.

    The model is ``-[F(+u) - F(-u)] / 2 = Phi u`` with ``neq.B[i, j] =
    sum_m u_mi f_mj``; the row of ``Phi`` of every force component ``j = (kappa,
    alpha)`` is fitted against the displacements of ``N(kappa)``.

    Returns
    -------
    fc : ndarray, shape (natoms, natoms, 3, 3)
        Symmetrised force constants in eV/Angstrom^2 (phonopy layout
        ``fc[i, j, alpha, beta] = d^2 E / du_{i alpha} du_{j beta}``).
    residual : float
        Relative residual of the force fit.
    """
    nat = len(nsets)
    groups = [(_active_indices(nsets[k]), (np.arange(3 * k, 3 * k + 3),)) for k in range(nat)]
    W, residual = solve_normal_equations(neq, groups, ridge=ridge)
    phi = np.array(W.T.real)  # phi[j, i] = -dF_j / du_i
    phi = 0.5 * (phi + phi.T)
    fc = phi.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3) * RY_BOHR2_TO_EV_ANG2
    y2 = float(np.sum(neq.y2))
    rel = float(np.sqrt(np.sum(residual) / y2)) if y2 > 0 else 0.0
    return fc, rel


def pair_configurations(disps) -> list:
    """Group the manifest entries of a random-displacement run into ``+/-`` pairs.

    Returns ``[(i_plus, i_minus, u), ...]`` with ``u`` of shape ``(natoms, 3)``.
    """
    by_config = {}
    for i, d in enumerate(disps):
        by_config.setdefault(int(d['config']), {})[int(d['sign'])] = i
    pairs = []
    for c in sorted(by_config):
        entry = by_config[c]
        if +1 not in entry or -1 not in entry:
            raise ValueError('random configuration %d lacks its +/- partner.' % c)
        u = np.asarray(disps[entry[+1]]['displacement'], dtype=float)
        pairs.append((entry[+1], entry[-1], u))
    return pairs


def read_qe_forces(edir, entry, natoms):
    """Forces (Ry/Bohr) of one displaced run from its ``pw.x`` output, or ``None``."""
    from ..phonon.io import _parse_qe_forces

    name = entry.get('input')
    if not name:
        return None
    out = os.path.join(edir, os.path.splitext(name)[0] + '.out')
    if not os.path.isfile(out):
        return None
    try:
        return _parse_qe_forces(out, natoms)
    except ValueError:
        return None


def fit_random_displacements(
    edir,
    disps,
    hrs_builder,
    sym_inputs,
    basis_positions=None,
    projector=None,
    cutoff=None,
    ridge: float = 1.0e-4,
    forces: bool = True,
) -> dict:
    """Analyse phase of ``displacement_mode='random'``.

    Builds the ``+/-`` Hamiltonians pair by pair, accumulates the normal
    equations and fits the Cartesian derivatives of every supercell atom.

    Parameters
    ----------
    edir : str
        Electron-phonon directory holding the displaced runs.
    disps : list[dict]
        Manifest entries (``config``, ``sign``, ``displacement`` ``(natoms, 3)``,
        ``prefix``, ``input``).
    hrs_builder : callable
        ``hrs_builder(prefix, positions=...)`` returning the supercell ``HRs``
        ``(nawf, nawf, n1, n2, n3, nspin)``.
    sym_inputs : dict
        Reference-cell data from ``build_supercell_HRs`` (``tau``, ``a_vectors``,
        ``alat``, ``naw``).
    basis_positions : ndarray, optional
        Reference positions for the fixed basis (passed to ``hrs_builder``).
    projector : ndarray, optional
        Good-subspace projector applied to every difference.
    cutoff : float, optional
        Locality radius (Bohr); ``None`` fits every element against all atoms.
    ridge : float
        Ridge parameter.
    forces : bool
        Also fit the force constants when the ``pw.x`` outputs are found.

    Returns
    -------
    dict
        ``directional`` (``3 natoms`` entries with unit Cartesian displacements),
        ``fit`` (diagnostics: ``nconfig``, ``cutoff``, ``ridge``,
        ``relative_residual``, ``pair_residual``, ``condition_number``,
        ``neighbours``), ``force_constants`` (eV/Angstrom^2 or ``None``) and
        ``force_residual``.
    """
    from .dvscf_fd import project_dV_good

    positions = np.asarray(sym_inputs['tau'], dtype=float)
    cell = np.asarray(sym_inputs['a_vectors'], dtype=float) * float(sym_inputs['alat'])
    naw = np.asarray(sym_inputs['naw'], dtype=int)
    nat = len(positions)
    if naw.size != nat:
        raise ValueError('naw (%d atoms) does not match tau (%d atoms).' % (naw.size, nat))
    orbital_atoms = np.repeat(np.arange(nat), naw)
    nsets = neighbour_sets(positions, cell, cutoff)
    pairs = pair_configurations(disps)

    neq = None
    neq_f = NormalEquations(3 * nat, (3 * nat,), dtype=float) if forces else None
    nforce = 0
    shape = None
    for ip, im, u in pairs:
        if u.shape != (nat, 3):
            raise ValueError('displacement of configuration %d is not (natoms, 3).' % ip)
        h_plus = hrs_builder(disps[ip]['prefix'], positions=basis_positions)
        h_minus = hrs_builder(disps[im]['prefix'], positions=basis_positions)
        D = 0.5 * (np.asarray(h_plus) - np.asarray(h_minus))
        del h_plus, h_minus
        if projector is not None:
            D = project_dV_good(D, projector)
        if shape is None:
            shape = D.shape
            if shape[0] != orbital_atoms.size:
                raise ValueError(
                    'HRs has %d orbitals but naw sums to %d.' % (shape[0], orbital_atoms.size)
                )
            neq = NormalEquations(3 * nat, (shape[0], shape[1], int(np.prod(shape[2:]))), D.dtype)
        neq.add(u.ravel(), D.reshape(shape[0], shape[1], -1))
        del D
        if neq_f is not None:
            f_plus = read_qe_forces(edir, disps[ip], nat)
            f_minus = read_qe_forces(edir, disps[im], nat)
            if f_plus is not None and f_minus is not None:
                neq_f.add(u.ravel(), (-0.5 * (f_plus - f_minus)).ravel())
                nforce += 1

    nmax = max(
        len(_active_indices(np.union1d(nsets[a], nsets[b]))) for a in range(nat) for b in range(nat)
    )
    if len(pairs) < nmax:
        warnings.warn(
            'random displacements: %d configurations for up to %d unknowns per element; '
            'the fit relies on the ridge term (increase nconfig or lower the cutoff).'
            % (len(pairs), nmax)
        )
    W, residual, pair_residual = fit_hamiltonian_derivatives(neq, orbital_atoms, nsets, ridge=ridge)
    y2 = float(np.sum(neq.y2))
    rel = float(np.sqrt(np.sum(residual) / y2)) if y2 > 0 else 0.0
    cond = float(np.linalg.cond(neq.G))

    directional = []
    for k in range(nat):
        for alpha in range(3):
            e = [0.0, 0.0, 0.0]
            e[alpha] = 1.0
            directional.append(
                {'sc_atom': int(k), 'displacement': e, 'dV': W[3 * k + alpha].reshape(shape)}
            )

    fc, fres = None, None
    if neq_f is not None and nforce > 0:
        if nforce < len(pairs):
            warnings.warn('forces found for %d of %d configuration pairs.' % (nforce, len(pairs)))
        fc, fres = fit_force_constants(neq_f, nsets, ridge=ridge)

    return {
        'directional': directional,
        'fit': {
            'nconfig': len(pairs),
            'cutoff': None if cutoff is None else float(cutoff),
            'ridge': float(ridge),
            'relative_residual': rel,
            'pair_residual': pair_residual,
            'condition_number': cond,
            'neighbours': [int(len(s)) for s in nsets],
            'nforce': nforce,
        },
        'force_constants': fc,
        'force_residual': fres,
    }
