from types import SimpleNamespace

import numpy as np
import pytest

from PAOFLOW.transport.observables import accumulation


@pytest.mark.unit
def test_accumulate_dos_preserves_weighted_trace_formula():
    dos = np.zeros(3)
    dos_k = np.zeros((3, 2))
    gC = np.diag([1.0 + 2.0j, 3.0 - 0.5j])
    wk = np.array([0.25, 0.75])

    accumulation.accumulate_dos(dos, dos_k, gC, wk, ie_g=1, ik=0)

    expected = -wk[0] * np.imag(np.trace(gC)) / np.pi
    assert dos_k[1, 0] == pytest.approx(expected)
    assert dos[1] == pytest.approx(expected)


@pytest.mark.unit
def test_accumulate_transmission_preserves_weighting(monkeypatch):
    def fake_evaluate_transmittance(**kwargs):
        np.testing.assert_allclose(kwargs['gamma_L'], np.eye(2))
        np.testing.assert_allclose(kwargs['gamma_R'], 2.0 * np.eye(2))
        assert kwargs['formula'] == 'landauer'
        assert kwargs['do_eigenchannels'] is True
        assert kwargs['do_eigplot'] is False
        assert kwargs['eta'] == pytest.approx(1.0e-5)
        return np.array([0.4, 0.2]), None

    monkeypatch.setattr(
        accumulation,
        'evaluate_transmittance',
        fake_evaluate_transmittance,
    )

    data = SimpleNamespace(
        conduct_formula='landauer',
        symmetry=SimpleNamespace(
            do_eigenchannels=True,
            do_eigplot=False,
            ie_eigplot=0,
            ik_eigplot=0,
        ),
        transport_direction=1,
    )
    conduct = np.zeros((3, 4))
    conduct_k = np.zeros((3, 2, 4))
    gC = np.eye(2, dtype=complex)
    sigma_L = -0.5j * np.eye(2)
    sigma_R = -1.0j * np.eye(2)
    wk = np.array([0.25, 0.75])

    accumulation.accumulate_transmission(
        conduct,
        conduct_k,
        gC,
        sigma_L,
        sigma_R,
        wk,
        ie_g=2,
        ik=1,
        data=data,
        delta=1.0e-5,
        rank=0,
        vkpt=np.zeros((3, 2)),
    )

    assert conduct[0, 2] == pytest.approx(wk[1] * 0.6)
    assert conduct_k[0, 1, 2] == pytest.approx(wk[1] * 0.6)
    np.testing.assert_allclose(conduct[1:3, 2], wk[1] * np.array([0.4, 0.2]))
    np.testing.assert_allclose(conduct_k[1:3, 1, 2], wk[1] * np.array([0.4, 0.2]))
