"""Disk cache of the doubly real-space vertex ``g(R_e, R_p)`` between runs.

Building ``g(R_e, R_p)`` (:func:`~PAOFLOW.elphon.do_pao_eph_dense_q.prepare_dense_vertex`)
depends only on the coarse coupling, the grids, the masses and the PAO basis, not on the
dense grids, temperatures or properties computed afterwards.  With ``cache_dir`` the array
is written once (``g_ReRp_<key>.npy`` plus a ``<key>.json`` provenance file) and later
runs fill the shared window from disk.  The key hashes the coupling files (names and
sizes), the grids, the masses, the shape of the projections and a caller-supplied
``cache_tag`` that identifies the PAO basis (:func:`basis_tag`); the values of the
projections are not hashed because they are not reproducible across machines at the
rounding level.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def basis_tag(data_controller, pthr: float | None = None) -> str:
    """Signature of the PAO basis of a :class:`~PAOFLOW.DataController` (for ``cache_tag``).

    Parameters
    ----------
    data_controller : DataController
        After ``projections`` (``arrays['basis']``) and ``projectability``.
    pthr : float, optional
        Projectability threshold used, appended to the tag.

    Returns
    -------
    str
        ``"<n orbitals>:<species and shells>:<nbnd>[:pthr]"``.
    """
    arrays, attributes = data_controller.data_dicts()
    orbitals = [
        '%s/%s/%s' % (o.get('atom', ''), o.get('l', ''), o.get('m', '')) for o in arrays['basis']
    ]
    parts = [
        str(len(orbitals)),
        hashlib.sha1('|'.join(orbitals).encode()).hexdigest()[:10],
        str(attributes.get('nbnds', '')),
    ]
    if pthr is not None:
        parts.append('%.6g' % pthr)
    return ':'.join(parts)


def dense_vertex_cache_key(
    coupling_dir: str,
    source: str,
    qgrid_coarse: Sequence[int],
    ng: Sequence[int],
    masses_amu: NDArray[np.float64],
    projection_shape: Sequence[int],
    cache_tag: str | None,
) -> str:
    """Key of a ``g(R_e, R_p)`` cache entry.

    Parameters
    ----------
    coupling_dir : str
        EPW directory or ``ph.x`` outputs.
    source : str
        ``'epw'`` or ``'dfpt'``.
    qgrid_coarse, ng : sequence of int
    masses_amu : array_like
    projection_shape : sequence of int
        ``A.shape`` ``(nbnd, nawf, nk)``.
    cache_tag : str or None
        Identity of the PAO basis (see :func:`basis_tag`).

    Returns
    -------
    str
        A 16-character hexadecimal digest.
    """
    names = []
    for root, _dirs, files in os.walk(coupling_dir):
        for f in sorted(files):
            if (
                '.epb' in f
                or 'dvscf' in f
                or 'patterns' in f
                or ('.dyn' in f and not f.endswith('.xml'))
                or f.endswith('.ukk')
            ):
                path = os.path.join(root, f)
                names.append((os.path.relpath(path, coupling_dir), os.path.getsize(path)))
    text = json.dumps(
        {
            'source': source,
            'files': names,
            'qgrid': [int(n) for n in qgrid_coarse],
            'ng': [int(n) for n in ng],
            'masses': [round(float(m), 6) for m in np.atleast_1d(masses_amu)],
            'shape': [int(n) for n in projection_shape],
            'tag': cache_tag,
        },
        sort_keys=True,
    )
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def cache_paths(cache_dir: str, key: str) -> tuple[str, str]:
    """``(array file, provenance file)`` of a key."""
    return os.path.join(cache_dir, 'g_ReRp_%s.npy' % key), os.path.join(
        cache_dir, 'g_ReRp_%s.json' % key
    )


def load_dense_vertex(cache_dir: str, key: str, g_ReRp: NDArray[np.complex128]) -> bool:
    """Fill ``g_ReRp`` from the cache (row by row, to bound the memory).

    Parameters
    ----------
    cache_dir : str
    key : str
    g_ReRp : ndarray
        The (shared) destination array.

    Returns
    -------
    bool
        ``False`` when no entry exists.

    Raises
    ------
    ValueError
        If the cached shape differs from ``g_ReRp.shape``.
    """
    path, _ = cache_paths(cache_dir, key)
    if not os.path.exists(path):
        return False
    src = np.load(path, mmap_mode='r')
    if src.shape != g_ReRp.shape:
        raise ValueError(
            'vertex cache %s has shape %s, expected %s' % (path, src.shape, g_ReRp.shape)
        )
    for row in range(g_ReRp.shape[0]):
        g_ReRp[row] = src[row]
    del src
    return True


def store_dense_vertex(
    cache_dir: str, key: str, g_ReRp: NDArray[np.complex128], provenance: dict
) -> str:
    """Write ``g_ReRp`` and its provenance atomically; returns the array path."""
    os.makedirs(cache_dir, exist_ok=True)
    path, meta = cache_paths(cache_dir, key)
    tmp = path + '.tmp.npy'
    np.save(tmp, g_ReRp)
    os.replace(tmp, path)
    with open(meta, 'w') as fh:
        json.dump(
            dict(provenance, written=time.strftime('%Y-%m-%d %H:%M:%S'), shape=list(g_ReRp.shape)),
            fh,
            indent=1,
        )
    return path
