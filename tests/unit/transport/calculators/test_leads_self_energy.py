"""Unit tests for lead self-energy assembly helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.calculators import leads_self_energy as lse_module
from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock


def _make_block(aux_value: float, s_value: float = 1.0):
    block = OperatorBlock('blk')
    block.allocate(dim1=1, dim2=1, nkpnts=1, lhave_aux=True, lhave_ovp=True, lhave_ham=False)
    block.aux[:, :, 0] = aux_value
    block.S[:, :, 0] = s_value
    return block.at_k(0)


@pytest.mark.unit
def test_build_self_energies_identical_leads_uses_shared_transfer(monkeypatch):
    """Identical leads reuse transfer matrices but compute left/right surface G separately."""
    calls = {'transfer': 0, 'green': []}

    def fake_transfer(**kwargs):
        calls['transfer'] += 1
        return np.array([[2.0]]), np.array([[3.0]]), 4

    def fake_green(**kwargs):
        calls['green'].append(kwargs['igreen'])
        return np.array([[10.0 + kwargs['igreen']]])

    monkeypatch.setattr(lse_module, 'compute_surface_transfer_matrices', fake_transfer)
    monkeypatch.setattr(lse_module, 'compute_surface_green_function', fake_green)

    blc_00R = _make_block(1.0)
    blc_01R = _make_block(1.0)
    blc_00L = _make_block(1.0)
    blc_01L = _make_block(1.0)
    blc_CR = _make_block(2.0)
    blc_LC = _make_block(3.0)

    sigma_R, sigma_L, niter_R, niter_L = lse_module.build_self_energies_from_blocks(
        blc_00R=blc_00R,
        blc_01R=blc_01R,
        blc_00L=blc_00L,
        blc_01L=blc_01L,
        blc_CR=blc_CR,
        blc_LC=blc_LC,
        leads_are_identical=True,
    )

    assert calls['transfer'] == 1
    assert calls['green'] == [1, -1]
    np.testing.assert_allclose(sigma_R, [[44.0]])
    np.testing.assert_allclose(sigma_L, [[81.0]])
    assert niter_R == 4
    assert niter_L == 4


@pytest.mark.unit
def test_compute_lead_surface_green_function_direction(monkeypatch):
    """Direction flag selects which surface Green's function is returned."""

    def fake_transfer(*args, **kwargs):
        return np.array([[2.0]]), np.array([[3.0]]), 7

    def fake_green(h_eff, s_eff, t_coupling, transfer_matrix, transfer_matrix_conj, igreen, delta):
        return np.array([[float(igreen)]])

    monkeypatch.setattr(lse_module, 'compute_surface_transfer_matrices', fake_transfer)
    monkeypatch.setattr(lse_module, 'compute_surface_green_function', fake_green)

    h_eff = np.eye(1)
    s_eff = np.eye(1)
    t_coupling = np.eye(1)

    g_right, niter = lse_module.compute_lead_surface_green_function(
        h_eff=h_eff,
        s_eff=s_eff,
        t_coupling=t_coupling,
        direction='right',
    )

    g_left, _ = lse_module.compute_lead_surface_green_function(
        h_eff=h_eff,
        s_eff=s_eff,
        t_coupling=t_coupling,
        direction='left',
    )

    np.testing.assert_allclose(g_right, [[1.0]])
    np.testing.assert_allclose(g_left, [[-1.0]])
    assert niter == 7
