"""Parity tests: vectorized eps_loop / jdos_loop vs a slow reference loop.

The reference implementations below reproduce the original triple-nested
Python loops verbatim, so they pin the numerical behaviour the vectorized
versions in ``do_epsilon`` must match (to floating-point summation order).
"""

import numpy as np
import pytest

from PAOFLOW.response.do_epsilon import eps_loop, jdos_loop
from PAOFLOW.utils.smearing import gaussian, intgaussian


class _DC:
    def __init__(self, arrays, attributes):
        self._a = arrays
        self._t = attributes

    def data_dicts(self):
        return self._a, self._t


def _make_data(nk, nbnd, nspin=1, insulator=True, adaptive=False, seed=0):
    rng = np.random.default_rng(seed)
    E_k = np.sort(rng.uniform(-5.0, 5.0, (nk, nbnd, nspin)), axis=1)
    pksp = rng.standard_normal((nk, 3, nbnd, nbnd, nspin)) + 1j * rng.standard_normal(
        (nk, 3, nbnd, nbnd, nspin)
    )
    arrays = {'E_k': E_k, 'pksp': pksp, 'kpnts_wght': rng.uniform(0.5, 1.5, nk)}
    if adaptive:
        arrays['deltakp2'] = rng.uniform(0.02, 0.3, (nk, nbnd, nbnd, nspin))
    attrs = {
        'bnd': nbnd,
        'nbnds': nbnd,
        'nspin': nspin,
        'dftSO': False,
        'insulator': insulator,
        'smearing': 'gauss',
        'degauss': 0.5,
        'delta': 0.1,
        'intrasmear': 0.05,
    }
    return _DC(arrays, attrs)


def _ref_eps(dc, ene, ispin, ipol, jpol):
    a, t = dc.data_dicts()
    ne = ene.size
    bndmax = t['bnd']
    Ek = a['E_k'][:, :bndmax, ispin]
    intersmear = t['delta']
    smearing = t['smearing']
    adaptive = 'deltakp2' in a
    if adaptive:
        eta_floor = t.get('adaptive_smearing_floor', ene[1] - ene[0])
        deltakp2 = a['deltakp2'][:, :bndmax, :bndmax, ispin]
    sf = 2 if (t['nspin'] == 1 and not t['dftSO']) else 1
    Ef = 1.0e-9
    epsi = np.zeros(ne)
    epsr = np.zeros(ne)
    dg = t['degauss']
    if smearing == 'gauss' and not t['insulator']:
        fn = sf * intgaussian(Ek, Ef, dg)
    else:
        fn = sf * (Ek <= Ef)
    th0, th1 = 1.0e-3 * sf, 0.5e-4 * sf
    if not t['insulator']:
        intra = t['intrasmear']
        em, er = np.zeros(ne), np.zeros(ne)
        fnF = sf * gaussian(Ek, Ef, dg)
    for ik in range(fn.shape[0]):
        for b2 in range(bndmax):
            for b1 in range(bndmax):
                if b1 != b2:
                    D = Ek[ik, b2] - Ek[ik, b1]
                    f = fn[ik, b2] - fn[ik, b1]
                    if abs(f) > th0 and fn[ik, b1] > th1 and fn[ik, b2] < sf:
                        eta = max(deltakp2[ik, b1, b2], eta_floor) if adaptive else intersmear
                        pk = np.real(
                            a['pksp'][ik, ipol, b1, b2, ispin] * a['pksp'][ik, jpol, b2, b1, ispin]
                        )
                        den = ((D**2 - ene**2) ** 2 + eta**2 * ene**2) * D
                        epsi += pk * eta * ene * fn[ik, b1] / den
                        epsr += pk * (D**2 - ene**2) * fn[ik, b1] / den
                elif not t['insulator']:
                    pk = np.real(
                        a['pksp'][ik, ipol, b1, b1, ispin] * a['pksp'][ik, jpol, b1, b1, ispin]
                    )
                    em += pk * intra * ene * fnF[ik, b1] / (ene**4 + intra**2 * ene**2)
                    er -= pk * fnF[ik, b1] * ene**2 / (ene**4 + intra**2 * ene**2)
    if not t['insulator']:
        epsi += 0.5 * em
        epsr += 0.5 * er
    return epsi, epsr


def _ref_jdos(dc, ene, ispin, kind):
    a, t = dc.data_dicts()
    bndmax = t['nbnds']
    Ek = np.swapaxes(a['my_eigsmat'][:, :, ispin], 0, 1)
    intersmear = t['delta']
    kw = a['kpnts_wght']
    fn = intgaussian(Ek, 1e-9, t['degauss'])
    nkpnts = Ek.shape[0]
    jdos = np.zeros(ene.size)
    count = 0.0
    for ik in range(nkpnts):
        for b2 in range(bndmax):
            for b1 in range(bndmax):
                D = Ek[ik, b2] - Ek[ik, b1]
                if fn[ik, b1] > 1e-4 and fn[ik, b2] < 2.0 and D > 1e-10:
                    f = fn[ik, b1] - fn[ik, b2]
                    if kind == 'gauss':
                        jdos += f * gaussian(D, ene, intersmear) * kw[ik]
                    else:
                        jdos += f * intersmear / (np.pi * ((D - ene) ** 2 + intersmear**2)) * kw[ik]
                    count += f
    sf = 2 if (t['nspin'] == 1 and not t['dftSO']) else 1
    jdos *= nkpnts / count / sf
    return jdos


@pytest.mark.parametrize('insulator', [True, False])
@pytest.mark.parametrize('adaptive', [False, True])
def test_eps_loop_parity(insulator, adaptive):
    dc = _make_data(6, 5, insulator=insulator, adaptive=adaptive, seed=insulator + 2 * adaptive)
    ene = np.linspace(0.2, 8.0, 120)
    epsi, epsr = eps_loop(dc, ene, 0, 0, 0)
    ri, rr = _ref_eps(dc, ene, 0, 0, 0)
    np.testing.assert_allclose(epsi, ri, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(epsr, rr, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('kind', ['gauss', 'lorentz'])
def test_jdos_loop_parity(kind):
    dc = _make_data(6, 5, insulator=False, seed=7)
    dc._a['my_eigsmat'] = np.swapaxes(dc._a['E_k'], 0, 1)
    ene = np.linspace(0.2, 8.0, 100)
    j = jdos_loop(dc, ene, 0, kind)
    ref = _ref_jdos(dc, ene, 0, kind)
    np.testing.assert_allclose(j, ref, rtol=1e-9, atol=1e-12)
