"""Unit tests for non-interacting Green's function builder."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.gzero_maker import compute_non_interacting_gf
from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock


def _make_block(aux: np.ndarray, s_mat: np.ndarray) -> 'OperatorBlock':
    block = OperatorBlock('blc_00C')
    block.allocate(dim1=aux.shape[0], dim2=aux.shape[1], nkpnts=1, lhave_aux=True, lhave_ovp=True)
    block.aux[:, :, 0] = aux
    block.S[:, :, 0] = s_mat
    return block


@pytest.mark.unit
def test_compute_non_interacting_gf_lorentzian_inverse():
    """Lorentzian smearing with calc='inverse' returns the shifted matrix A."""
    aux = np.eye(2, dtype=complex)
    s_mat = np.eye(2, dtype=complex)
    block = _make_block(aux, s_mat).at_k(0)

    gzero = compute_non_interacting_gf(block, smearing_type='lorentzian', delta=0.1, calc='inverse')

    expected = aux + 1j * 0.1 * s_mat
    np.testing.assert_allclose(gzero, expected)


@pytest.mark.unit
def test_compute_non_interacting_gf_lorentzian_direct():
    """Lorentzian smearing with calc='direct' returns the inverse of A."""
    aux = np.diag([2.0, 4.0]).astype(complex)
    s_mat = np.eye(2, dtype=complex)
    block = _make_block(aux, s_mat).at_k(0)

    gzero = compute_non_interacting_gf(block, smearing_type='lorentzian', delta=0.2, calc='direct')

    expected = np.linalg.inv(aux + 1j * 0.2 * s_mat)
    np.testing.assert_allclose(gzero, expected)


@pytest.mark.unit
def test_compute_non_interacting_gf_none_uses_ratio():
    """'none' smearing scales the imaginary part by delta_ratio."""
    aux = np.eye(1, dtype=complex)
    s_mat = np.eye(1, dtype=complex)
    block = _make_block(aux, s_mat).at_k(0)

    gzero = compute_non_interacting_gf(
        block,
        smearing_type='none',
        delta=1.0,
        delta_ratio=0.1,
        calc='inverse',
    )

    np.testing.assert_allclose(gzero, aux + 1j * 0.1 * s_mat)


@pytest.mark.unit
def test_compute_non_interacting_gf_numerical_interpolates():
    """Numerical smearing should interpolate values on the provided grid."""
    aux = np.diag([0.2, 0.8]).astype(complex)
    s_mat = np.eye(2, dtype=complex)
    block = _make_block(aux, s_mat).at_k(0)

    xgrid = np.array([0.0, 1.0])
    g_smear = np.array([1.0 + 1.0j, 2.0 + 0.0j])

    gzero = compute_non_interacting_gf(
        block,
        smearing_type='numerical',
        delta=1.0,
        g_smear=g_smear,
        xgrid=xgrid,
        calc='direct',
    )

    expected = np.diag(
        [
            (1 - 0.2) * g_smear[0] + 0.2 * g_smear[1],
            (1 - 0.8) * g_smear[0] + 0.8 * g_smear[1],
        ]
    )
    np.testing.assert_allclose(gzero, expected)


@pytest.mark.unit
def test_compute_non_interacting_gf_numerical_requires_smear():
    """Numerical smearing requires both g_smear and xgrid."""
    aux = np.eye(1, dtype=complex)
    s_mat = np.eye(1, dtype=complex)
    block = _make_block(aux, s_mat).at_k(0)

    with pytest.raises(ValueError):
        compute_non_interacting_gf(block, smearing_type='numerical', delta=1.0)


@pytest.mark.unit
def test_compute_non_interacting_gf_numerical_requires_hermitian():
    """Numerical smearing rejects non-Hermitian matrices."""
    aux = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    s_mat = np.eye(2, dtype=complex)
    block = _make_block(aux, s_mat).at_k(0)

    with pytest.raises(ValueError):
        compute_non_interacting_gf(
            block,
            smearing_type='numerical',
            delta=1.0,
            g_smear=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
            xgrid=np.array([0.0, 1.0]),
        )
