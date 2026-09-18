"""Parity tests: the packed ``do_find_Weyl`` kernels vs the original per-k loops.

The reference implementations below reproduce the original ``band_loop_H`` /
``gen_eigs`` / ``get_gap`` code verbatim (``np.tensordot`` over the strided spin
axis, full ``numpy.linalg.eigvalsh``), pinning the numerical behaviour that the
BLAS-packed versions must match.  The screening test uses a two-band lattice
Weyl model whose nodes are known analytically, and checks that no box holding a
band crossing is ever discarded.
"""

import numpy as np
import pytest
from numpy import linalg as LAN

from PAOFLOW.topology.do_find_Weyl import (
    band_energies,
    build_Hk,
    build_Hk_batch,
    find_min,
    get_gap,
    get_R_grid_fft,
    get_search_grid,
    pack_HR,
    screen_boxes,
)

NK = 4


def _ref_band_loop_H(HRaux, kq, R):
    nawf, _, _, nspin = HRaux.shape
    kdot = np.tensordot(R, 2.0j * np.pi * kq[:, None], axes=([1], [0]))
    np.exp(kdot, kdot)
    auxh = np.zeros((nawf, nawf, 1, nspin), dtype=complex, order='C')
    for ispin in range(nspin):
        auxh[:, :, 0:1, ispin] = np.tensordot(HRaux[:, :, :, ispin], kdot, axes=([2], [0]))
    return auxh


def _ref_gen_eigs(HRaux, kq, R):
    nawf, _, _, nspin = HRaux.shape
    E_kp = np.zeros((1, nawf, nspin))
    Hks = _ref_band_loop_H(HRaux, kq, R)
    for ispin in range(nspin):
        E_kp[:, :, ispin] = LAN.eigvalsh(Hks[:, :, 0, ispin], UPLO='U')
    return E_kp


def _ref_R_grid(nr1, nr2, nr3):
    R = np.zeros((nr1 * nr2 * nr3, 3))
    for i in range(nr1):
        for j in range(nr2):
            for k in range(nr3):
                n = k + j * nr3 + i * nr2 * nr3
                Rx, Ry, Rz = float(i) / nr1, float(j) / nr2, float(k) / nr3
                if Rx >= 0.5:
                    Rx -= 1.0
                if Ry >= 0.5:
                    Ry -= 1.0
                if Rz >= 0.5:
                    Rz -= 1.0
                R[n] = (Rx * nr1, Ry * nr2, Rz * nr3)
    return R


def _random_HR(rng, nawf, nR, nspin):
    return rng.standard_normal((nawf, nawf, nR, nspin)) + 1j * rng.standard_normal(
        (nawf, nawf, nR, nspin)
    )


def _weyl_HR(R, nawf=2):
    """``sin kx sx + sin ky sy + (m + cos kx + cos ky + cos kz) sz`` with ``m = -2``.

    The two Weyl nodes sit at the fractional k-points ``(0, 0, +-1/4)``.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    idx = {tuple(int(v) for v in R[n]): n for n in range(R.shape[0])}
    HR = np.zeros((nawf, nawf, R.shape[0], 1), dtype=complex)
    HR[:2, :2, idx[(0, 0, 0)], 0] = -2.0 * sz
    for axis, sigma in enumerate((sx, sy)):
        plus, minus = [0, 0, 0], [0, 0, 0]
        plus[axis], minus[axis] = 1, -1
        HR[:2, :2, idx[tuple(plus)], 0] += sigma / (2j)
        HR[:2, :2, idx[tuple(minus)], 0] -= sigma / (2j)
    for axis in range(3):
        plus, minus = [0, 0, 0], [0, 0, 0]
        plus[axis], minus[axis] = 1, -1
        HR[:2, :2, idx[tuple(plus)], 0] += sz / 2
        HR[:2, :2, idx[tuple(minus)], 0] += sz / 2
    return HR


@pytest.mark.parametrize('grid', [(4, 4, 4), (3, 5, 2), (6, 6, 6)])
def test_R_grid_matches_reference_loop(grid):
    np.testing.assert_array_equal(get_R_grid_fft(*grid), _ref_R_grid(*grid))


@pytest.mark.parametrize('nspin', [1, 2])
def test_build_Hk_parity(nspin):
    rng = np.random.default_rng(11)
    nawf = 7
    R = get_R_grid_fft(NK, NK, NK)
    HR = _random_HR(rng, nawf, R.shape[0], nspin)
    kpts = rng.uniform(-0.5, 0.5, size=(5, 3))

    for ispin in range(nspin):
        HRpack = pack_HR(HR, ispin)
        batch = build_Hk_batch(HRpack, nawf, kpts, R)
        for i, kq in enumerate(kpts):
            ref = _ref_band_loop_H(HR, kq, R)[:, :, 0, ispin]
            np.testing.assert_allclose(build_Hk(HRpack, nawf, kq, R), ref, atol=1e-12)
            np.testing.assert_allclose(batch[i], ref, atol=1e-12)


@pytest.mark.parametrize('nspin', [1, 2])
def test_band_energies_and_gap_parity(nspin):
    rng = np.random.default_rng(13)
    nawf, nelec = 9, 4
    R = get_R_grid_fft(NK, NK, NK)
    HR = _random_HR(rng, nawf, R.shape[0], nspin)
    HRpack = pack_HR(HR)
    kpts = rng.uniform(-0.5, 0.5, size=(6, 3))

    ref = np.array([_ref_gen_eigs(HR, kq, R)[0, :, 0] for kq in kpts])
    np.testing.assert_allclose(band_energies(HRpack, nawf, kpts, R), ref, atol=1e-10)

    gaps = np.array([get_gap(HRpack, nawf, kq, R, nelec) for kq in kpts])
    np.testing.assert_allclose(gaps, ref[:, nelec] - ref[:, nelec - 1], atol=1e-10)


def test_band_energies_batches_consistently():
    """Chunking over k must not change the result."""
    rng = np.random.default_rng(17)
    nawf = 5
    R = get_R_grid_fft(NK, NK, NK)
    HRpack = pack_HR(_random_HR(rng, nawf, R.shape[0], 1))
    kpts = rng.uniform(-0.5, 0.5, size=(37, 3))

    full = band_energies(HRpack, nawf, kpts, R)
    one_by_one = np.vstack([band_energies(HRpack, nawf, kq, R) for kq in kpts])
    np.testing.assert_allclose(full, one_by_one, atol=1e-12)


@pytest.mark.parametrize('snk', [4, 8, 16])
def test_screen_boxes_keeps_every_crossing(snk):
    R = get_R_grid_fft(NK, NK, NK)
    HRpack = pack_HR(_weyl_HR(R))

    grid = get_search_grid(snk, snk, snk)
    bbox = np.array([1.0 / snk] * 3)
    alive = screen_boxes(HRpack, 2, R, 1, grid, grid + bbox)

    for node in ([0.0, 0.0, 0.25], [0.0, 0.0, -0.25]):
        inside = np.all((grid <= np.array(node) + 1e-12) & (np.array(node) < grid + bbox), axis=1)
        assert inside.any()
        assert alive[inside].all()

    # no discarded box may hold a near-degeneracy anywhere on a dense submesh
    sub = get_search_grid(7, 7, 7, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], endpoint=True)
    for b in np.flatnonzero(~alive):
        E = band_energies(HRpack, 2, grid[b] + sub * bbox, R)
        assert (E[:, 1] - E[:, 0]).min() > 1e-3


def test_find_min_locates_analytic_weyl_nodes():
    R = get_R_grid_fft(NK, NK, NK)
    HRpack = pack_HR(_weyl_HR(R))

    screened, _ = find_min(HRpack, 2, 1, R, False, False, [8, 8, 8], 2.0)
    full, _ = find_min(HRpack, 2, 1, R, False, False, [8, 8, 8], 0.0)

    screened = np.unique(np.around(screened, 4), axis=0)
    full = np.unique(np.around(full, 4), axis=0)
    np.testing.assert_allclose(screened, full, atol=1e-4)

    expected = np.array([[0.0, 0.0, -0.25], [0.0, 0.0, 0.25]])
    np.testing.assert_allclose(screened, expected, atol=1e-4)
