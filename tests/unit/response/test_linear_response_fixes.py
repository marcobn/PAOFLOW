"""Bug fixes of the linear-response routes (Kubo-Bastin campaign P1, 14 November 2026).

One test per fix; the numbers of every route that was right stay frozen in
``test_kubo_legacy_parity.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from PAOFLOW.response.kubo_units import conductivity_conversion

from . import _kubo_cases as kc

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def reference():
    with np.load(kc.REFERENCE) as data:
        return {key: data[key] for key in data.files}


def _read(p, fname: str) -> np.ndarray:
    return np.loadtxt(os.path.join(p.data_controller.data_attributes['opath'], fname))


def _header(p, fname: str) -> str:
    with open(os.path.join(p.data_controller.data_attributes['opath'], fname)) as f:
        return f.readline()


def _window(name: str) -> dict:
    emin, emax = kc.WINDOWS[name]
    return dict(eminH=emin, emaxH=emax, esize=kc.NE)


def test_a_tensor_is_read_without_s_tensor(tmp_path, reference):
    # until the fix a_tensor was read only when s_tensor was given (KeyError otherwise)
    p = kc.prepared('haldane', str(tmp_path))
    p.linear_response(response='cond', a_tensor=[[0, 0]], **_window('haldane'))
    want = reference['haldane/cond_eq1|Cond_xx.dat']
    assert np.allclose(_read(p, 'Cond_xx.dat'), want, rtol=1e-10, atol=1e-10 * np.abs(want).max())


def test_shc_falls_back_to_the_spin_Hall_tensor_in_its_own_order(tmp_path, reference):
    p = kc.prepared('kane_mele', str(tmp_path))
    p.spin_Hall(s_tensor=[[0, 1, 2]], emin=-3.0, emax=3.0, ne=kc.NE)  # (current, field, spin)
    p.linear_response(response='shc', **_window('kane_mele'))  # reads (spin, current, field)
    want = reference['kane_mele/shc_eq3|SHC_EVEN_z_xy.dat']
    assert np.allclose(
        _read(p, 'SHC_EVEN_z_xy.dat'), want, rtol=1e-10, atol=1e-10 * np.abs(want).max()
    )


def test_without_tensors_the_stored_default_is_used(tmp_path):
    # DataController starts with the full 3x3 a_tensor; without a_tensor every component is written
    p = kc.prepared('haldane', str(tmp_path))
    p.linear_response(response='ahc', **_window('haldane'))
    assert all(
        os.path.isfile(os.path.join(str(tmp_path), f'AHC_{a}{b}.dat')) for a in 'xyz' for b in 'xyz'
    )


def test_finite_broadening_spin_Hall_branches_run(tmp_path):
    """Eq. 2 / Eq. 4 / Eq. 5 of 'shc' read the spin Hall tensor (they read a missing key before)."""
    p = kc.prepared('kane_mele', str(tmp_path))
    window = dict(eminH=-0.2, emaxH=0.2, esize=5)  # Kane-Mele gap
    p.linear_response(response='shc', full_chi2=True, gamma=1e-3, s_tensor=[[2, 0, 1]], **window)
    sea = _read(p, 'SHC_EVEN_z_xy.dat')[:, 1]
    p.linear_response(response='shc', delta=1e-3, s_tensor=[[2, 0, 1]], **window)
    berry = _read(p, 'SHC_EVEN_z_xy.dat')[:, 1]
    # both tend to the Berry-curvature limit: in the gap they agree (quantised spin Hall plateau)
    assert np.abs(berry).min() > 0
    assert np.allclose(sea, berry, rtol=0.02)
    p.linear_response(response='shc', t_odd=True, intraband=True, s_tensor=[[2, 0, 1]], **window)
    assert _read(p, 'SHC_ODD_z_xy.dat').shape == (5, 2)


def test_berry_regulariser_is_a_parameter(tmp_path, reference):
    p = kc.prepared('rashba', str(tmp_path))
    p.linear_response(response='ahc', a_tensor=[[0, 1]], delta=0.05, **_window('rashba'))
    want = reference['rashba/ahc_eq3|AHC_xy.dat']
    assert np.allclose(_read(p, 'AHC_xy.dat'), want, rtol=1e-10, atol=1e-10 * np.abs(want).max())
    p.linear_response(response='ahc', a_tensor=[[0, 1]], delta=0.01, **_window('rashba'))
    assert not np.allclose(_read(p, 'AHC_xy.dat')[:, 1], want[:, 1], rtol=1e-3)


def test_sheet_outputs_say_their_unit(tmp_path):
    p = kc.prepared('haldane', str(tmp_path))
    p.linear_response(response='ahc', a_tensor=[[0, 1]], twoD=True, **_window('haldane'))
    assert 'e^2/hbar' in _header(p, 'AHC_xy.dat') and 'S/cm' not in _header(p, 'AHC_xy.dat')
    p.linear_response(response='cond', a_tensor=[[0, 0]], twoD=True, intraband=True, **_window('haldane'))  # fmt: skip
    assert 'e^2/hbar' in _header(p, 'Cond_xx.dat')
    p.linear_response(response='cond', a_tensor=[[0, 0]], **_window('haldane'))
    assert 'S/cm' in _header(p, 'Cond_xx.dat')


def test_anomalous_Hall_sheet_conductivity(tmp_path):
    p = kc.prepared('rashba', str(tmp_path))
    p.anomalous_Hall(a_tensor=[[0, 1]], emin=-3.0, emax=-1.0, ne=kc.NE)
    bulk = _read(p, 'ahcEf_xy.dat')[:, 1]
    p.anomalous_Hall(a_tensor=[[0, 1]], emin=-3.0, emax=-1.0, ne=kc.NE, twoD=True)
    sheet = _read(p, 'ahcEf_xy.dat')[:, 1]
    ratio = conductivity_conversion(p.data_controller, True) / conductivity_conversion(p.data_controller, False)  # fmt: skip
    assert np.allclose(sheet, bulk * ratio, rtol=1e-12)


def test_optical_Hall_conductivity_is_finite_on_models(tmp_path):
    # models have shift = inf: the photon grid was linspace(0, inf) (NaN)
    p = kc.prepared('rashba', str(tmp_path))
    p.anomalous_Hall(a_tensor=[[0, 1]], emin=-3.0, emax=-1.0, ne=kc.NE, do_ac=True)
    arry = p.data_controller.data_arrays
    for name in ('MCDr_xy.dat', 'MCDi_xy.dat'):
        data = _read(p, name)
        assert np.all(np.isfinite(data))
        assert data[-1, 0] == pytest.approx(arry['E_k'].max() - arry['E_k'].min())


def test_energy_window_is_not_overwritten(tmp_path):
    p = kc.prepared('rashba', str(tmp_path))
    attr = p.data_controller.data_attributes
    attr['shift'] = -1.5
    p.anomalous_Hall(a_tensor=[[0, 1]], emin=-3.0, emax=-1.0, ne=kc.NE)
    assert attr['emaxH'] == -1.0
    assert _read(p, 'ahcEf_xy.dat')[-1, 0] == pytest.approx(-1.5)


def test_scalar_orbital_operator(tmp_path):
    p = kc.prepared('sp_square', str(tmp_path))
    p.orbital_operator()
    L = p.data_controller.data_arrays['Lj']
    assert np.abs(L[0] @ L[1] - L[1] @ L[0] - 1j * L[2]).max() < 1e-12
    assert np.allclose(np.linalg.eigvalsh(L[2][1:, 1:]), [-1, 0, 1])


def test_orbital_Hall_without_spin_orbit(tmp_path):
    """Scalar basis: the Berry-curvature sum of the orbital current, both spins counted."""
    p = kc.prepared('sp_square', str(tmp_path))
    p.orbital_Hall(o_tensor=[[0, 1, 2]], emin=-2.0, emax=2.0, ne=kc.NE)
    got = _read(p, 'ohcEf_z_xy.dat')
    arry, attr = p.data_controller.data_dicts()
    from PAOFLOW.utils.perturb_split import perturb_split

    E, V, dH = arry['E_k'][:, :, 0], arry['v_k'][:, :, :, 0], arry['dHksp'][:, :, :, :, 0]
    Lz = arry['Lj'][2]
    jx = 0.5 * (Lz @ dH[:, 0] + dH[:, 0] @ Lz)
    # the basis of the route inside degenerate subspaces (the one perturb_split fixes): the plain
    # eigh basis there depends on the LAPACK build, and so would the per-band occupations
    V = np.stack(
        [perturb_split(jx[k], dH[k, 1], V[k], arry['degen'][0][k], True)[2] if arry['degen'][0][k]
         else V[k] for k in range(V.shape[0])]
    )  # fmt: skip
    a = np.einsum('kin,kij,kjm->knm', V.conj(), jx, V)
    b = np.einsum('kin,kij,kjm->knm', V.conj(), dH[:, 1], V)
    denominator = (E[:, None, :] - E[:, :, None]) ** 2 + 0.05**2
    omega = -2.0 * np.sum(np.imag(a * np.transpose(b, (0, 2, 1))) / denominator, axis=2)
    from PAOFLOW.utils.smearing import intgaussian

    mu = got[:, 0]
    occupation = np.array([intgaussian(E, m, arry['deltakp'][:, :, 0]) for m in mu])
    want = 2.0 * np.einsum('ekn,kn->e', occupation, omega) / attr['nkpnts']
    want *= conductivity_conversion(p.data_controller, False)
    assert np.abs(want).max() > 10.0
    assert np.allclose(got[:, 1], want, rtol=1e-6, atol=1e-6 * np.abs(want).max())
