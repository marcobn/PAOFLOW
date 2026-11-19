"""Rotation subspace of the DFPT / EPW vertex: the good bands of the PAO Hamiltonian (``vertex_bands``)."""

import numpy as np
import pytest

from PAOFLOW.elphon.workflows.system import ElphonSystem

pytestmark = pytest.mark.unit


class _DataController:
    def __init__(self, bnd):
        self._arrays = {'basis': [{'atom': 'Nb', 'l': 2, 'm': m} for m in range(5)]}
        self._attr = {'bnd': bnd, 'nbnds': 20}

    def data_dicts(self):
        return self._arrays, self._attr


class _Coupling:
    def __init__(self, projections, bnd):
        self.projections = projections
        self.data_controller = _DataController(bnd)


def _system(tmp_path, **kw):
    return ElphonSystem(prefix='x', root=str(tmp_path), masses_amu=[1.0], nelec=2.0, **kw)


def _rotate(A_kq, d, A_k):
    """The band route of ``CoarseVertexSetup.pao_vertex_k``: ``g_ij = sum_mn A_kq[m,i] d_mn A_k[n,j]^*``."""
    return np.einsum('mi,mn,nj->ij', A_kq, d, np.conj(A_k))


def test_good_band_rotation_reproduces_the_band_vertex(rng=np.random.default_rng(7)):
    """Back-rotated with the good bands' PAO eigenvectors, the vertex equals the DFPT band vertex only when
    the poorly projectable bands are left out of the rotation, and then for any number of extra bands."""
    nawf, bnd = 6, 4

    def unitary(n):
        q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        return q

    good_k, good_kq = unitary(nawf)[:bnd], unitary(nawf)[:bnd]  # orthonormal rows: the good bands
    results = {}
    for nbnd in (bnd + 2, bnd + 6):
        extra = 0.3 * (
            rng.standard_normal((nbnd - bnd, nawf)) + 1j * rng.standard_normal((nbnd - bnd, nawf))
        )
        A_k = np.vstack([good_k, extra])
        A_kq = np.vstack([good_kq, extra[::-1]])
        d = rng.standard_normal((nbnd, nbnd)) + 1j * rng.standard_normal((nbnd, nbnd))
        d[:bnd, :bnd] = np.arange(bnd * bnd).reshape(bnd, bnd) * (
            1 + 0.5j
        )  # the same good block every time
        for label, keep in (('all', nbnd), ('good', bnd)):
            Ak, Akq = A_k.copy(), A_kq.copy()
            Ak[keep:], Akq[keep:] = 0.0, 0.0
            g = _rotate(Akq, d, Ak)
            back = np.conj(good_kq) @ g @ good_k.T  # rows: the PAO eigenvectors of the good bands
            results[(label, nbnd)] = back
    exact = np.arange(bnd * bnd).reshape(bnd, bnd) * (1 + 0.5j)
    for nbnd in (bnd + 2, bnd + 6):
        assert np.allclose(results[('good', nbnd)], exact, atol=1e-12)
        assert not np.allclose(results[('all', nbnd)], exact, atol=1e-3)
    assert not np.allclose(results[('all', bnd + 2)], results[('all', bnd + 6)], atol=1e-3)


def test_rotation_projections_zero_the_bands_above_bnd(tmp_path):
    A = np.ones((8, 6, 3), dtype=complex)
    coupling = _Coupling(A, bnd=5)
    good = _system(tmp_path).rotation_projections(coupling)
    assert good is not A and np.all(good[:5] == 1.0) and np.all(good[5:] == 0.0)
    assert np.all(A == 1.0)  # the coupling run keeps its projections
    assert _system(tmp_path, vertex_bands='all').rotation_projections(coupling) is A
    assert _system(tmp_path).rotation_projections(_Coupling(A, bnd=8)) is A  # every band good


def test_vertex_cache_tag_records_the_rotation_subspace(tmp_path):
    dc = _DataController(bnd=10)
    good = _system(tmp_path).vertex_cache_tag(dc)
    every = _system(tmp_path, vertex_bands='all').vertex_cache_tag(dc)
    assert good.endswith('|bands=good10') and good.startswith(every) and good != every
    fd = _system(tmp_path, coupling_source='fd', fd_dir='fd', qgrid=(1, 1, 1)).vertex_cache_tag(dc)
    assert fd == every  # the finite-difference source does not rotate
    with pytest.raises(ValueError, match='vertex_bands'):
        _system(tmp_path, vertex_bands='some')
