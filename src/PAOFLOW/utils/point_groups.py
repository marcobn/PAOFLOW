"""Point-group operations, third-rank tensor symmetrisation and k-grid reduction.

A third-rank polar tensor :math:`T^{abc}` of a crystal with point group :math:`G` satisfies
:math:`T^{abc} = \\sum_{a'b'c'} R_{aa'}R_{bb'}R_{cc'}T^{a'b'c'}` for every operation :math:`R`
(the same law for proper and improper operations). Two uses:

* **Audit** of a computed tensor: its projection on the invariant subspace (:func:`symmetrize`),
  the part removed by the projection (:func:`symmetry_error`), the components that must vanish
  (:func:`forbidden_components`) and the relations between the others
  (:func:`independent_relations`). The groups of the common crystal classes are closed from a
  few Cartesian generators (:data:`GENERATORS`; the polar axis of ``mm2`` / ``C2v`` along ``x``
  or ``z`` as named, the threefold axis along ``z``, a mirror ``m_x`` perpendicular to ``x``).
* **Reduction** of a uniform k-grid to its irreducible wedge (:func:`reduce_k_grid`) with the
  orbit sizes as weights, and the symmetrisation of the wedge sums: since the symmetrisation is
  linear, :math:`\\sum_{\\mathbf k \\in BZ} T(\\mathbf k) = \\mathrm{Sym}_G[\\sum_{\\mathbf k \\in IBZ}
  w_{\\mathbf k} T(\\mathbf k)]` for a covariant integrand. The operations act on fractional k as
  integer matrices (:func:`fractional_k_from_cartesian`, :func:`cartesian_from_fractional_k`),
  the convention of Quantum ESPRESSO's ``sym_rot`` as stored by PAOFLOW.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

LABELS = 'xyz'


def _rot(axis: str, angle_deg: float) -> NDArray[np.float64]:
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    if axis == 'z':
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if axis == 'x':
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 'y':
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    raise ValueError(axis)


def _mirror(axis: str) -> NDArray[np.float64]:
    m = np.eye(3)
    m[LABELS.index(axis), LABELS.index(axis)] = -1.0
    return m


_C3_111 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # x->y->z
_INVERSION = -np.eye(3)

#: Generators of the common crystal classes (Hermann-Mauguin and Schoenflies names).
GENERATORS: dict[str, list[NDArray[np.float64]]] = {
    '1': [],
    '-1': [_INVERSION],
    'm_x': [_mirror('x')],
    'm_y': [_mirror('y')],
    'm_z': [_mirror('z')],
    '2_z': [_rot('z', 180.0)],
    '2_x': [_rot('x', 180.0)],
    'mm2_z': [_mirror('x'), _mirror('y')],  # polar axis z
    'mm2_x': [_mirror('y'), _mirror('z')],  # polar axis x
    'mm2_y': [_mirror('x'), _mirror('z')],  # polar axis y
    '222': [_rot('x', 180.0), _rot('z', 180.0)],
    '4mm': [_rot('z', 90.0), _mirror('x')],
    '-42m': [_rot('z', 90.0) @ _INVERSION, _rot('x', 180.0)],
    '3': [_rot('z', 120.0)],
    '3m_x': [_rot('z', 120.0), _mirror('x')],  # mirror perpendicular to x (C3v)
    '3m_y': [_rot('z', 120.0), _mirror('y')],
    '-6m2_x': [_rot('z', 120.0), _mirror('z'), _mirror('x')],  # D3h, mirror perpendicular to x
    '-6m2_y': [_rot('z', 120.0), _mirror('z'), _mirror('y')],
    '6mm': [_rot('z', 60.0), _mirror('x')],
    '-43m': [
        _C3_111,
        _rot('z', 180.0),
        _rot('x', 180.0),
        _mirror('x') @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ _mirror('x'),
    ],
    '23': [_C3_111, _rot('z', 180.0), _rot('x', 180.0)],
    '32': [_rot('z', 120.0), _rot('x', 180.0)],
    '622': [_rot('z', 60.0), _rot('x', 180.0)],
    '-6': [_rot('z', 120.0), _mirror('z')],
}
ALIASES = {
    'C1': '1', 'Ci': '-1', 'Cs': 'm_z', 'C2': '2_z', 'C2v': 'mm2_z', 'D2': '222', 'C4v': '4mm',
    'D2d': '-42m', 'C3': '3', 'C3v': '3m_x', 'D3h': '-6m2_x', 'C6v': '6mm', 'Td': '-43m', 'T': '23',
    'D3': '32', 'D6': '622', 'C3h': '-6',
}  # fmt: skip


def close_group(generators: ArrayLike, limit: int = 96) -> NDArray[np.float64]:
    """All products of the generators (the identity included).

    Parameters
    ----------
    generators : array_like ``(ngen, 3, 3)``
    limit : int, optional
        Largest group accepted (raises above it).

    Returns
    -------
    ndarray ``(nops, 3, 3)``
    """
    ops = [np.eye(3)]
    frontier = [np.eye(3)]
    gens = [np.asarray(g, dtype=float) for g in generators]
    while frontier:
        new = []
        for m in frontier:
            for g in gens:
                candidate = g @ m
                if not any(np.allclose(candidate, o, atol=1e-10) for o in ops):
                    ops.append(candidate)
                    new.append(candidate)
        frontier = new
        if len(ops) > limit:
            raise RuntimeError('group closure did not terminate')
    return np.array(ops)


def point_group_operations(name: str) -> NDArray[np.float64]:
    """All Cartesian matrices of a crystal class, closed from its generators.

    Parameters
    ----------
    name : str
        A key of :data:`GENERATORS` (Hermann-Mauguin with the axis convention, e.g. ``'mm2_x'``,
        ``'3m_x'``, ``'-43m'``) or a Schoenflies alias of :data:`ALIASES`.

    Returns
    -------
    ndarray ``(nops, 3, 3)``
    """
    key = ALIASES.get(name, name)
    if key not in GENERATORS:
        raise KeyError('unknown point group %r (known: %s)' % (name, sorted(GENERATORS)))
    return close_group(GENERATORS[key])


def symmetrize(tensor: ArrayLike, operations: ArrayLike) -> NDArray:
    """Projection of a third-rank tensor on the invariant subspace of the operations."""
    t = np.asarray(tensor)
    ops = np.asarray(operations, dtype=float)
    out = np.zeros_like(t, dtype=np.result_type(t, float))
    for R in ops:
        out += np.einsum('ai,bj,ck,ijk->abc', R, R, R, t)
    return out / ops.shape[0]


def forbidden_components(operations: ArrayLike, tol: float = 1e-10) -> list[str]:
    """Components that vanish for the group (labels such as ``'xxx'``)."""
    ops = np.asarray(operations, dtype=float)
    out = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                basis = np.zeros((3, 3, 3))
                basis[a, b, c] = 1.0
                # the component vanishes if no invariant tensor has weight on it:
                # the projection of the unit tensor onto itself is zero
                if abs(symmetrize(basis, ops)[a, b, c]) < tol:
                    out.append(LABELS[a] + LABELS[b] + LABELS[c])
    return out


def independent_relations(operations: ArrayLike, tol: float = 1e-10) -> dict[str, dict[str, float]]:
    """Each allowed component expressed through the symmetrised unit tensors: ``{'yxx': {'yyy': -1.0, ...}}``.

    The projection of the unit tensor ``e_abc`` gives the linear relations that the symmetry
    imposes; the dictionary lists, for every allowed component, the components it is tied to
    and the coefficients (``yxx -> yyy: -1`` means ``T^yxx = -T^yyy`` in 3m with the mirror
    perpendicular to x).
    """
    ops = np.asarray(operations, dtype=float)
    relations: dict[str, dict[str, float]] = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                basis = np.zeros((3, 3, 3))
                basis[a, b, c] = 1.0
                proj = symmetrize(basis, ops)
                own = proj[a, b, c]
                if abs(own) < tol:
                    continue
                label = LABELS[a] + LABELS[b] + LABELS[c]
                links = {}
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if (i, j, k) != (a, b, c) and abs(proj[i, j, k]) > tol:
                                links[LABELS[i] + LABELS[j] + LABELS[k]] = float(
                                    proj[i, j, k] / own
                                )
                relations[label] = links
    return relations


def symmetry_error(tensor: ArrayLike, operations: ArrayLike) -> dict[str, float]:
    """Norm of the part of a tensor removed by the symmetrisation, relative to the tensor norm."""
    t = np.asarray(tensor, dtype=complex)
    sym = symmetrize(t, operations)
    norm = float(np.linalg.norm(t))
    removed = float(np.linalg.norm(t - sym))
    return {'norm': norm, 'removed': removed, 'relative': removed / norm if norm > 0 else 0.0}


def kleinman_error(tensor: ArrayLike) -> dict[str, float]:
    """Deviation from full permutation symmetry (Kleinman): ``|T - sym_perm(T)| / |T|``."""
    t = np.asarray(tensor, dtype=complex)
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    full = sum(np.transpose(t, p) for p in perms) / 6.0
    norm = float(np.linalg.norm(t))
    removed = float(np.linalg.norm(t - full))
    return {'norm': norm, 'removed': removed, 'relative': removed / norm if norm > 0 else 0.0}


def tensor_from_components(values: dict[str, complex], fill=np.nan) -> NDArray:
    """``(3, 3, 3)`` array from ``{'xyz': value, ...}`` (missing components ``fill``)."""
    t = np.full(
        (3, 3, 3),
        fill,
        dtype=complex if any(np.iscomplexobj(v) for v in values.values()) else float,
    )
    for label, value in values.items():
        t[tuple(LABELS.index(ch) for ch in label)] = value
    return t


# --------------------------------------------------------------------------- k-grid reduction
def cartesian_from_fractional_k(w_frac: ArrayLike, b_vectors: ArrayLike) -> NDArray[np.float64]:
    """Cartesian rotation ``R`` of an operation ``W`` acting on fractional k-coordinates.

    Parameters
    ----------
    w_frac : array_like ``(3, 3)`` or ``(nops, 3, 3)``
        ``k'_frac = W k_frac`` (Quantum ESPRESSO's ``sym_rot`` as stored by PAOFLOW).
    b_vectors : array_like ``(3, 3)``
        Reciprocal lattice vectors as rows (any unit).

    Returns
    -------
    ndarray, same leading shape
        ``R = B^T W B^{-T}`` with ``k_cart = B^T k_frac``; orthogonal for a lattice symmetry.
    """
    B = np.asarray(b_vectors, dtype=float)
    W = np.asarray(w_frac, dtype=float)
    return B.T @ W @ np.linalg.inv(B.T)


def fractional_k_from_cartesian(r_cart: ArrayLike, b_vectors: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`cartesian_from_fractional_k` (``W = B^{-T} R B^T``)."""
    B = np.asarray(b_vectors, dtype=float)
    R = np.asarray(r_cart, dtype=float)
    return np.linalg.inv(B.T) @ R @ B.T


def grid_indices(nk: Sequence[int]) -> NDArray[np.int_]:
    """Integer indices ``(i1, i2, i3)`` of a uniform grid in the order of ``meshgrid(indexing='ij')``."""
    axes = [np.arange(int(n)) for n in nk]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def reduce_k_grid(
    nk: Sequence[int], operations_cryst: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Irreducible wedge of a Gamma-centred uniform grid under integer k-space operations.

    Parameters
    ----------
    nk : sequence of 3 int
    operations_cryst : array_like ``(nops, 3, 3)``
        Integer matrices acting on fractional k (``k' = W k``); the identity may be omitted.

    Returns
    -------
    k_irr : ndarray ``(nirr, 3)``
        Fractional coordinates of the representatives (``i / nk``).
    weights : ndarray ``(nirr,)``
        Orbit sizes (they sum to ``prod(nk)``).
    representative : ndarray ``(prod(nk),)``, int
        Index into ``k_irr`` of the representative of every full-grid point.

    Raises
    ------
    ValueError
        If an operation is not integer (it would not map the grid onto itself).
    """
    nk = np.asarray(nk, dtype=int)
    ops = np.asarray(operations_cryst, dtype=float)
    if not np.allclose(ops, np.rint(ops), atol=1e-8):
        raise ValueError('k-space operations must be integer matrices in fractional coordinates')
    ops = np.rint(ops).astype(int)
    idx = grid_indices(nk)  # (N, 3)
    strides = np.array([nk[1] * nk[2], nk[2], 1])
    flat_min = idx @ strides
    for W in ops:
        image = (idx @ W.T) % nk  # k' = W k, folded
        flat_min = np.minimum(flat_min, image @ strides)
    reps, representative, weights = np.unique(flat_min, return_inverse=True, return_counts=True)
    k_irr = idx[reps] / nk
    return k_irr, weights.astype(float), representative.reshape(-1)


def component_closure(
    components: Sequence[Sequence[int]], operations_cart: ArrayLike, tol: float = 1e-10
) -> list[tuple[int, int, int]]:
    """Components that feed the symmetrised values of the requested ones.

    ``T_sym^{abc} = (1/|G|) sum_R R_{aa'} R_{bb'} R_{cc'} T^{a'b'c'}``: the closure is the set
    of ``(a', b', c')`` with a non-zero coefficient for at least one requested ``(a, b, c)``.
    """
    ops = np.asarray(operations_cart, dtype=float)
    out: set[tuple[int, int, int]] = set()
    for a, b, c in components:
        for R in ops:
            weight = np.abs(np.einsum('i,j,k->ijk', R[a], R[b], R[c]))
            for abc in zip(*np.nonzero(weight > tol)):
                out.add(tuple(int(i) for i in abc))
    return sorted(out)


def symmetrize_components(
    values: NDArray,
    components: Sequence[Sequence[int]],
    operations_cart: ArrayLike,
    requested: Sequence[Sequence[int]],
) -> NDArray:
    """Group average of component spectra: ``(ncomp, ...)`` on ``components`` -> ``(nreq, ...)``.

    ``components`` must contain the closure of ``requested`` (:func:`component_closure`);
    components outside the closure would carry zero weight anyway.
    """
    values = np.asarray(values)
    ops = np.asarray(operations_cart, dtype=float)
    position = {tuple(int(i) for i in abc): j for j, abc in enumerate(components)}
    out = np.zeros((len(requested),) + values.shape[1:], dtype=values.dtype)
    for i, (a, b, c) in enumerate(requested):
        for R in ops:
            weight = np.einsum('i,j,k->ijk', R[a], R[b], R[c])
            for abc in zip(*np.nonzero(np.abs(weight) > 1e-12)):
                key = tuple(int(x) for x in abc)
                if key not in position:
                    raise ValueError('component %s missing from the closure' % (key,))
                out[i] += weight[abc] * values[position[key]]
    return out / ops.shape[0]


__all__ = [
    'ALIASES',
    'GENERATORS',
    'LABELS',
    'cartesian_from_fractional_k',
    'close_group',
    'component_closure',
    'forbidden_components',
    'fractional_k_from_cartesian',
    'grid_indices',
    'independent_relations',
    'kleinman_error',
    'point_group_operations',
    'reduce_k_grid',
    'symmetrize',
    'symmetrize_components',
    'symmetry_error',
    'tensor_from_components',
]
