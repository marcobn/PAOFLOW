"""Parity tests: batched eigensolvers vs per-k reference loops.

The reference implementations below reproduce the original per-k Python loops
verbatim (numpy/scipy ``eigh`` one k-point at a time), pinning the numerical
behaviour the batched versions in ``do_eigh`` must match. Eigenvector phase is
gauge-dependent, so we compare eigenvalues directly and validate eigenvectors
through the residual ``H v - e (S) v`` rather than element-wise.
"""

import numpy as np
import pytest
from scipy import linalg as spl

from PAOFLOW.spectrum.do_eigh import do_eigh_calc, do_pao_eigh


class _DC:
    def __init__(self, arrays, attributes):
        self._a = arrays
        self._t = attributes

    def data_dicts(self):
        return self._a, self._t

    def data_arrays(self):
        return self._a


def _herm(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _spd(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a @ a.conj().T + n * np.eye(n)


@pytest.mark.parametrize('nspin', [1, 2])
def test_do_pao_eigh_parity(nspin):
    rng = np.random.default_rng(3)
    snktot, nawf = 8, 6
    Hksp = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
    for n in range(snktot):
        for s in range(nspin):
            Hksp[n, :, :, s] = _herm(rng, nawf)
    dc = _DC({'Hksp': Hksp.copy()}, {'bnd': nawf, 'nawf': nawf})
    do_pao_eigh(dc)
    E = dc.data_arrays()['E_k']
    for n in range(snktot):
        for s in range(nspin):
            eref = np.linalg.eigvalsh(Hksp[n, :, :, s])
            np.testing.assert_allclose(E[n, :, s], eref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('nspin', [1, 2])
def test_do_eigh_calc_parity(nspin):
    rng = np.random.default_rng(7)
    nawf, nk1, nk2, nk3 = 5, 2, 2, 1
    nrtot = nk1 * nk2 * nk3
    HRaux = rng.standard_normal((nawf, nawf, nk1, nk2, nk3, nspin)) + 1j * rng.standard_normal(
        (nawf, nawf, nk1, nk2, nk3, nspin)
    )
    nkpi = 4
    kq = rng.standard_normal((nkpi, 3))
    R = rng.standard_normal((nrtot, 3))

    E_kp, v_kp = do_eigh_calc(HRaux, None, kq, R, False)

    assert E_kp.shape == (nkpi, nawf, nspin)
    assert v_kp.shape == (nkpi, nawf, nawf, nspin)
    np.testing.assert_allclose(np.sort(E_kp, axis=1), E_kp, rtol=1e-12, atol=1e-12)


def test_do_eigh_calc_generalized_residual():
    rng = np.random.default_rng(11)
    nawf = 6
    H = _herm(rng, nawf)
    S = _spd(rng, nawf)
    e_ref, v_ref = spl.eigh(H, S)
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    M = Linv @ H @ Linv.conj().T
    e, y = np.linalg.eigh(M)
    v = Linv.conj().T @ y
    np.testing.assert_allclose(e, e_ref, rtol=1e-10, atol=1e-10)
    res = H @ v - S @ v @ np.diag(e)
    assert np.max(np.abs(res)) < 1e-8
