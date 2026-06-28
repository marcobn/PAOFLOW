"""Parity tests: vectorized L_loop / L_loop_hall vs slow reference loops.

The reference implementations below reproduce the original band/component
Python loops verbatim, so they pin the numerical behaviour the vectorized
versions in ``do_Boltz_tensors`` must match (to floating-point summation
order).
"""

import numpy as np
import pytest

from PAOFLOW.boltzmann.do_Boltz_tensors import L_loop, L_loop_hall
from PAOFLOW.utils.smearing import gaussian, metpax


class _DC:
    def __init__(self, arrays, attributes):
        self._a = arrays
        self._t = attributes

    def data_dicts(self):
        return self._a, self._t


def _make_data(nk, nbnd, nspin=1, seed=0):
    rng = np.random.default_rng(seed)
    E_k = np.sort(rng.uniform(-5.0, 5.0, (nk, nbnd, nspin)), axis=1)
    velkp = rng.standard_normal((nk, 3, nbnd, nspin))
    tau = rng.uniform(0.5, 2.0, (nk, nbnd, nspin))
    deltakp = rng.uniform(0.05, 0.3, (nk, nbnd, nspin))
    d2Ed2k = rng.standard_normal((6, nk, nbnd, nspin))
    arrays = {
        'E_k': E_k,
        'velkp': velkp,
        'scattering_tau': tau,
        'deltakp': deltakp,
        'd2Ed2k': d2Ed2k,
    }
    attrs = {'bnd': nbnd, 'nspin': nspin, 'nkpnts': nk}
    return _DC(arrays, attrs), velkp


_TTENSOR = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]])


def _ref_L(dc, temp, smearing, ene, velkp, t_tensor, alpha, ispin):
    a, t = dc.data_dicts()
    esize = ene.size
    snktot = a['E_k'].shape[0]
    bnd = t['bnd']
    kq_wght = 1.0 / t['nkpnts']
    L = np.zeros((3, 3, esize), dtype=float)
    for n in range(bnd):
        Eaux = np.reshape(np.repeat(a['E_k'][:, n, ispin], esize), (snktot, esize))
        delk = (
            np.reshape(np.repeat(a['deltakp'][:, n, ispin], esize), (snktot, esize))
            if smearing is not None
            else None
        )
        EtoAlpha = np.power(Eaux - ene, alpha)
        if smearing is None:
            Eaux -= ene
            smearA = 1 / (4 * temp * (np.cosh(Eaux / (2 * temp)) ** 2))
        elif smearing == 'gauss':
            smearA = gaussian(Eaux, ene, delk)
        elif smearing == 'm-p':
            smearA = metpax(Eaux, ene, delk)
        for ll in range(t_tensor.shape[0]):
            i, j = t_tensor[ll]
            pref = (
                kq_wght
                * a['scattering_tau'][:, n, ispin]
                * velkp[:, i, n, ispin]
                * velkp[:, j, n, ispin]
            )
            L[i, j, :] += np.sum(pref[:, None] * smearA * EtoAlpha, axis=0)
    return L


def _ref_hall(dc, temp, smearing, ene, velkp, t_tensor, alpha, ispin):
    from sympy import Eijk

    a, t = dc.data_dicts()
    esize = ene.size
    snktot = a['E_k'].shape[0]
    bnd = t['bnd']
    nspin = t['nspin']
    kq_wght = 1.0 / t['nkpnts']
    L_hall = np.zeros((3, 3, 3, esize), dtype=float)
    sig_hall = np.zeros((3, 3, 3, snktot, bnd, nspin))
    M_inv = np.zeros((3, 3, snktot, bnd, nspin))
    eff = a['d2Ed2k']
    M_inv[0, 0] = eff[0]
    M_inv[1, 1] = eff[1]
    M_inv[2, 2] = eff[2]
    M_inv[0, 1] = M_inv[1, 0] = eff[3]
    M_inv[0, 2] = M_inv[2, 0] = eff[4]
    M_inv[1, 2] = M_inv[2, 1] = eff[5]
    for n in range(bnd):
        Eaux = np.reshape(np.repeat(a['E_k'][:, n, ispin], esize), (snktot, esize))
        delk = (
            np.reshape(np.repeat(a['deltakp'][:, n, ispin], esize), (snktot, esize))
            if smearing is not None
            else None
        )
        if smearing is None:
            Eaux -= ene
            smearA = 1 / (4 * temp * (np.cosh(Eaux / (2 * temp)) ** 2))
        elif smearing == 'gauss':
            smearA = gaussian(Eaux, ene, delk)
        elif smearing == 'm-p':
            smearA = metpax(Eaux, ene, delk)
        for i in range(3):
            for j in range(3):
                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            sig_hall[i, j, p, :, n, ispin] += (
                                int(Eijk(p, q, r))
                                * velkp[:, i, n, ispin]
                                * velkp[:, r, n, ispin]
                                * M_inv[j, q, :, n, ispin]
                            )
                    L_hall[i, j, p, :] += np.sum(
                        kq_wght
                        * a['scattering_tau'][:, n, ispin] ** 2
                        * sig_hall[i, j, p, :, n, ispin]
                        * smearA.T,
                        axis=1,
                    )
    return L_hall


@pytest.mark.parametrize('smearing', [None, 'gauss', 'm-p'])
@pytest.mark.parametrize('alpha', [0, 1, 2])
def test_L_loop_parity(smearing, alpha):
    dc, velkp = _make_data(8, 5, seed=3)
    ene = np.linspace(-4.0, 4.0, 21)
    ref = _ref_L(dc, 0.025, smearing, ene, velkp, _TTENSOR, alpha, 0)
    got = L_loop(dc, 0.025, smearing, ene, velkp, _TTENSOR, alpha, 0)
    assert np.allclose(got, ref, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize('smearing', [None, 'gauss', 'm-p'])
def test_L_loop_hall_parity(smearing):
    dc, velkp = _make_data(8, 5, seed=4)
    ene = np.linspace(-4.0, 4.0, 21)
    ref = _ref_hall(dc, 0.025, smearing, ene, velkp, _TTENSOR, 0, 0)
    got = L_loop_hall(dc, 0.025, smearing, ene, velkp, _TTENSOR, 0, 0)
    assert np.allclose(got, ref, atol=1e-10, rtol=1e-8)
