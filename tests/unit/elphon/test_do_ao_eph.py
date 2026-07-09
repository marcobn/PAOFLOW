"""Unit tests for the atomic-orbital (Agapito-Bernardi) e-phonon route."""

import numpy as np
import pytest

pytest.importorskip('scipy')
from scipy.io import FortranFile

from PAOFLOW.elphon.do_ao_eph import _k_permutation, vertex_from_qe_elphmat
from PAOFLOW.elphon.qe_elph_io import el_ph_mat_to_cartesian, read_qe_el_ph_mat


def _write_dump(path, nbnd, nksq, nat, nkstot, el, u, xq, xk, et):
    """Write a synthetic ``elphmat.<iq>.dat`` matching the patched-QE format."""
    with FortranFile(str(path), 'w') as f:
        f.write_record(np.array([nbnd, nksq, nat, nkstot], dtype=np.int32))
        f.write_record(el.ravel(order='F'))
        f.write_record(u.ravel(order='F'))
        f.write_record(np.asarray(xq, dtype=np.float64))
        f.write_record(xk.ravel(order='F'))
        f.write_record(et.ravel(order='F'))


def _random_unitary(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, _ = np.linalg.qr(a)
    return q


def test_el_ph_mat_to_cartesian_matches_manual():
    rng = np.random.default_rng(1)
    nbnd, nksq, ncart = 3, 4, 3
    el = rng.standard_normal((nbnd, nbnd, nksq, ncart)) + 1j * rng.standard_normal(
        (nbnd, nbnd, nksq, ncart)
    )
    u = _random_unitary(ncart, seed=2)
    cart = el_ph_mat_to_cartesian(el, u)
    # d_{mn,c} = sum_p conj(u_{c,p}) el_{mn,p}
    manual = np.einsum('cp,mnkp->mnkc', u.conj(), el)
    np.testing.assert_allclose(cart, manual, atol=1e-12)
    assert cart.shape == el.shape


def test_read_qe_el_ph_mat_roundtrip_gamma(tmp_path):
    nbnd, nksq, nat = 3, 5, 1
    ncart, nkstot = 3 * nat, nksq  # lgamma: k and k+q lists coincide
    rng = np.random.default_rng(3)
    el = rng.standard_normal((nbnd, nbnd, nksq, ncart)) + 1j * rng.standard_normal(
        (nbnd, nbnd, nksq, ncart)
    )
    u = _random_unitary(ncart, seed=4)
    xq = np.zeros(3)
    xk = rng.standard_normal((nkstot, 3))
    et = rng.standard_normal((nkstot, nbnd))
    p = tmp_path / 'elphmat.1.dat'
    _write_dump(p, nbnd, nksq, nat, nkstot, el, u, xq, xk.T, et.T)

    d = read_qe_el_ph_mat(p)
    assert (d['nbnd'], d['nksq'], d['nat'], d['nkstot']) == (nbnd, nksq, nat, nkstot)
    np.testing.assert_allclose(d['el_ph_mat'], el, atol=1e-12)
    np.testing.assert_allclose(d['u'], u, atol=1e-12)
    np.testing.assert_allclose(d['xk'], xk, atol=1e-12)  # gamma: ik_k = arange
    np.testing.assert_allclose(d['et'], et, atol=1e-12)


def test_read_qe_el_ph_mat_interleaved_kq(tmp_path):
    """Non-Gamma dump: nkstot = 2*nksq, the k sublist is the even records."""
    nbnd, nksq, nat = 2, 3, 1
    ncart, nkstot = 3 * nat, 2 * nksq
    rng = np.random.default_rng(5)
    el = rng.standard_normal((nbnd, nbnd, nksq, ncart)) + 1j * rng.standard_normal(
        (nbnd, nbnd, nksq, ncart)
    )
    u = _random_unitary(ncart, seed=6)
    xk = rng.standard_normal((nkstot, 3))
    et = rng.standard_normal((nkstot, nbnd))
    p = tmp_path / 'elphmat.2.dat'
    _write_dump(p, nbnd, nksq, nat, nkstot, el, u, np.array([0.1, 0.0, 0.0]), xk.T, et.T)

    d = read_qe_el_ph_mat(p)
    np.testing.assert_allclose(d['xk'], xk[0::2], atol=1e-12)  # even records = k
    np.testing.assert_allclose(d['et'], et[0::2], atol=1e-12)


def test_k_permutation_recovers_shuffle():
    ng = (2, 2, 2)
    ax = [np.arange(n) / n for n in ng]
    grid = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)  # (8,3)
    bg = np.eye(3)
    rng = np.random.default_rng(7)
    perm_true = rng.permutation(grid.shape[0])
    xk_cart = grid[perm_true]  # bg = identity -> cart == cryst
    perm = _k_permutation(grid, xk_cart, bg, ng)
    # perm maps dumped index i -> paoflow index; here paoflow grid == grid,
    # dumped k i corresponds to grid[perm_true[i]] -> paoflow index perm_true[i]
    np.testing.assert_array_equal(perm, perm_true)


def test_k_permutation_rejects_off_grid():
    ng = (2, 2, 2)
    ax = [np.arange(n) / n for n in ng]
    grid = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    # PAOFLOW grid is missing the last point, so its label is absent from the
    # lookup table; a dumped k-point at that label must raise.
    partial = grid[:-1]
    with pytest.raises(ValueError):
        _k_permutation(partial, grid[-1:], np.eye(3), ng)


def test_vertex_from_qe_elphmat_shape(tmp_path):
    ng = (2, 2, 2)
    nk = 8
    nbnd, nawf, nat = 3, 2, 1
    ncart = 3 * nat
    ax = [np.arange(n) / n for n in ng]
    kcry = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    bg = np.eye(3)
    rng = np.random.default_rng(8)
    el = rng.standard_normal((nbnd, nbnd, nk, ncart)) + 1j * rng.standard_normal(
        (nbnd, nbnd, nk, ncart)
    )
    u = np.eye(ncart, dtype=complex)
    et = rng.standard_normal((nk, nbnd))
    p = tmp_path / 'elphmat.1.dat'
    _write_dump(p, nbnd, nk, nat, nk, el, u, np.zeros(3), kcry.T, et.T)
    A = rng.standard_normal((nbnd, nawf, nk)) + 1j * rng.standard_normal((nbnd, nawf, nk))

    gR, q_cryst = vertex_from_qe_elphmat(p, A, kcry, bg, ng)
    assert gR.shape == (nawf, nawf, ncart, ng[0], ng[1], ng[2])
    np.testing.assert_allclose(q_cryst, np.zeros(3), atol=1e-12)
