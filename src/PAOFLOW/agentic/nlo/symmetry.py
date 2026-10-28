"""Point-group constraints on third-rank response tensors (shift current, SHG).

A third-rank tensor :math:`T^{abc}` of a crystal with point group :math:`G` satisfies
:math:`T^{abc} = \\sum_{a'b'c'} R_{aa'}R_{bb'}R_{cc'}T^{a'b'c'}` for every operation :math:`R`
(polar tensors: the same law for proper and improper operations).  The group is given by its
Cartesian matrices, generated here from a few generators for the common crystal classes (the
axes follow the usual conventions: the polar axis of ``mm2`` / ``C2v`` along ``x`` or ``z`` as
named, the threefold axis along ``z``, a mirror ``m_x`` perpendicular to ``x``).  Projecting a
computed tensor on the invariant subspace gives its symmetrised form, the symmetry error of a
calculation (the part removed by the projection) and the components that must vanish.
"""

from __future__ import annotations

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
    ops = [np.eye(3)]
    frontier = [np.eye(3)]
    gens = [np.asarray(g, dtype=float) for g in GENERATORS[key]]
    while frontier:
        new = []
        for m in frontier:
            for g in gens:
                candidate = g @ m
                if not any(np.allclose(candidate, o, atol=1e-10) for o in ops):
                    ops.append(candidate)
                    new.append(candidate)
        frontier = new
        if len(ops) > 96:
            raise RuntimeError('group closure did not terminate')
    return np.array(ops)


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


__all__ = [
    'ALIASES',
    'GENERATORS',
    'forbidden_components',
    'independent_relations',
    'kleinman_error',
    'point_group_operations',
    'symmetrize',
    'symmetry_error',
    'tensor_from_components',
]
