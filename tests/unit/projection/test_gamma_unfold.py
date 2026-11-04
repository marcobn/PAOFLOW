"""Gamma-tricks wavefunctions: unfolding of the stored half G-sphere to the full sphere."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import FortranFile

from PAOFLOW.projection.do_atwfc_proj import read_QE_wfc, unfold_gamma_sphere

pytestmark = pytest.mark.unit


def _real_space_real_block(rng, nbnd=4, nmax=2):
    """Coefficients of real functions on a small Miller sphere: c(-G) = c(G)^* with |G| <= nmax."""
    pts = [
        (i, j, k)
        for i in range(-nmax, nmax + 1)
        for j in range(-nmax, nmax + 1)
        for k in range(-nmax, nmax + 1)
        if i * i + j * j + k * k <= nmax * nmax
    ]
    mill = np.array(pts, dtype=np.int64).T  # (3, npw)
    npw = mill.shape[1]
    c = rng.standard_normal((nbnd, npw)) + 1j * rng.standard_normal((nbnd, npw))
    index = {tuple(g): ig for ig, g in enumerate(mill.T)}
    for ig, g in enumerate(mill.T):
        jg = index[(-g[0], -g[1], -g[2])]
        if jg > ig:
            c[:, jg] = np.conj(c[:, ig])
        elif jg == ig:
            c[:, ig] = c[:, ig].real
    return mill, c


def _half_sphere(mill, c):
    """QE's gamma-tricks storage: G = 0 first, then one member of every +G/-G pair."""
    keep = [ig for ig, g in enumerate(mill.T) if tuple(g) > (0, 0, 0) or not np.any(g)]
    keep.sort(key=lambda ig: (np.any(mill[:, ig]), ig))  # G = 0 first
    return mill[:, keep], c[:, keep]


def _sorted(mill, c):
    order = np.lexsort(mill[::-1])
    return mill[:, order], c[:, order]


def test_unfold_recovers_the_full_sphere():
    rng = np.random.default_rng(3)
    mill, c = _real_space_real_block(rng)
    mill_h, c_h = _half_sphere(mill, c)
    assert mill_h.shape[1] == (mill.shape[1] + 1) // 2
    c_u, mill_u = unfold_gamma_sphere(c_h, mill_h)
    assert mill_u.shape == mill.shape
    m1, c1 = _sorted(mill, c)
    m2, c2 = _sorted(mill_u, c_u)
    assert np.array_equal(m1, m2)
    assert np.allclose(c1, c2)
    # the overlap of the half sphere is wrong, that of the unfolded sphere is right
    full = c @ c.conj().T
    assert not np.allclose(c_h @ c_h.conj().T, full)
    assert np.allclose(c_u @ c_u.conj().T, full)


def test_unfold_rejects_bad_blocks():
    rng = np.random.default_rng(4)
    mill, c = _real_space_real_block(rng)
    with pytest.raises(ValueError):
        unfold_gamma_sphere(c, mill)  # both members stored
    mill_h, c_h = _half_sphere(mill, c)
    with pytest.raises(ValueError):
        unfold_gamma_sphere(c_h[:, 1:], mill_h[:, 1:])  # no G = 0


def _write_wfc_dat(path, ik, xk, mill, c, gamma_only, bg):
    nbnd, igwx = c.shape
    with FortranFile(path, 'w') as f:
        head = np.zeros(11, dtype=np.int32)
        head[0] = ik
        head[1:7] = np.frombuffer(np.asarray(xk, dtype=np.float64).tobytes(), dtype=np.int32)
        head[7] = 1
        head[8] = 1 if gamma_only else 0
        head[9:11] = np.frombuffer(np.array([1.0], dtype=np.float64).tobytes(), dtype=np.int32)
        f.write_record(head)
        f.write_record(np.array([igwx, igwx, 1, nbnd], dtype=np.int32))
        f.write_record(np.asarray(bg, dtype=np.float64).T.copy(order='C'))  # Fortran order (3, 3)
        f.write_record(
            np.asarray(mill, dtype=np.int32).T.copy(order='C')
        )  # Fortran order (3, igwx)
        for ib in range(nbnd):
            f.write_record(np.ascontiguousarray(c[ib], dtype=np.complex128))


def _controller(path):
    return SimpleNamespace(data_dicts=lambda: ({}, {'nspin': 1, 'fpath': path}))


def test_reader_unfolds_a_gamma_file(tmp_path):
    rng = np.random.default_rng(5)
    mill, c = _real_space_real_block(rng, nbnd=3)
    mill_h, c_h = _half_sphere(mill, c)
    bg = np.eye(3)
    full_dir, gamma_dir = tmp_path / 'full', tmp_path / 'gamma'
    full_dir.mkdir()
    gamma_dir.mkdir()
    _write_wfc_dat(os.path.join(full_dir, 'wfc1.dat'), 1, np.zeros(3), mill, c, False, bg)
    _write_wfc_dat(os.path.join(gamma_dir, 'wfc1.dat'), 1, np.zeros(3), mill_h, c_h, True, bg)
    gk_f, wf_f = read_QE_wfc(_controller(str(full_dir)), 0, 0, orthonormalize=False)
    gk_g, wf_g = read_QE_wfc(_controller(str(gamma_dir)), 0, 0, orthonormalize=False)
    assert gk_f['gamma_only'] is False and gk_f['gamma_unfolded'] is False
    assert gk_g['gamma_only'] is False and gk_g['gamma_unfolded'] is True
    assert gk_g['igwx'] == gk_f['igwx'] == mill.shape[1]
    m1, c1 = _sorted(gk_f['mill'], wf_f['wfc'])
    m2, c2 = _sorted(gk_g['mill'], wf_g['wfc'])
    assert np.array_equal(m1, m2)
    assert np.allclose(c1, c2)
    # the Loewdin step then gives the same orthonormal bands up to the G order
    _, of = read_QE_wfc(_controller(str(full_dir)), 0, 0)
    _, og = read_QE_wfc(_controller(str(gamma_dir)), 0, 0)
    assert np.allclose(_sorted(gk_f['mill'], of['wfc'])[1], _sorted(gk_g['mill'], og['wfc'])[1])
