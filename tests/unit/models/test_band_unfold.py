"""Unit tests of the band unfolding: real-space Fourier conventions and the DFT-run helper.

A single-orbital simple-cubic crystal (one atom per primitive cell, nearest-neighbour hopping
``t``) is folded into a 2x1x1 supercell.  At a generic k the two supercell bands are the
primitive band at k (spectral weight 1) and the band folded from k + G/2 (weight 0), so the
weights are known exactly and the test is sensitive to the sign of R: with the wrong sign the
weights become cos^2(k_x a) and sin^2(k_x a).
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.models.band_unfold import (
    _detect_fourier_sign,
    _extract_hamiltonian,
    paoflow_to_model_dict,
    unfold_bands,
)

T_HOP = -1.0
NGRID = 4


def _fft_grid(a_vectors: np.ndarray, n: int = NGRID) -> np.ndarray:
    """Lattice vectors of an n^3 FFT box folded to [-n/2, n/2), Cartesian (rows of a_vectors)."""
    idx = np.arange(n)
    idx = np.where(idx >= n / 2, idx - n, idx)
    grid = np.array(np.meshgrid(idx, idx, idx, indexing='ij')).reshape(3, -1).T
    return grid @ a_vectors


def _hoppings(tau: np.ndarray, R: np.ndarray, convention: int) -> np.ndarray:
    """HRs[a, b, R] = t when |tau_b + convention*R - tau_a| = 1 (nearest neighbours)."""
    nat = tau.shape[0]
    HRs = np.zeros((nat, nat, R.shape[0], 1), dtype=complex)
    for ir in range(R.shape[0]):
        d = tau[None, :, :] + convention * R[ir] - tau[:, None, :]
        HRs[:, :, ir, 0] = np.where(np.abs(np.linalg.norm(d, axis=2) - 1.0) < 1e-9, T_HOP, 0.0)
    return HRs


def _model_dict(a_vectors, tau, convention, fourier_sign=None):
    R = _fft_grid(a_vectors)
    pao = {
        'HRs': _hoppings(tau, R, convention),
        'R': R,
        'nawf': tau.shape[0],
        'nspin': 1,
        'norbitals': np.ones(tau.shape[0], dtype=int),
    }
    if fourier_sign is not None:
        pao['fourier_sign'] = fourier_sign
    atoms = {str(i): {'name': 'X', 'tau': tau[i].tolist()} for i in range(tau.shape[0])}
    return {
        'label': 'test',
        'alat': 1.0,
        'model': {'a_vectors': a_vectors, 'atoms': atoms},
        '_paoflow': pao,
    }


A_PC = np.eye(3)
TAU_PC = np.zeros((1, 3))
A_SC = np.diag([2.0, 1.0, 1.0])
TAU_SC = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
SYM = {'A': [0.0, 0.0, 0.0], 'B': [1.0 / 3.0, 0.1, 0.2]}  # generic point (fractional, PC basis)


def _band(k):
    return 2.0 * T_HOP * np.cos(2.0 * np.pi * k).sum()


def _unfold(pc, sc, nk=3):
    return unfold_bands(pc, sc, sym_points=SYM, path_str='A-B', nk_per_seg=nk, verbose=False)


def _assert_clean_unfolding(result):
    """At every k, one SC band carries weight 1 at the PC energy and the other weight 0."""
    for ik, k in enumerate(result.kpath_cart):
        e_pc = _band(k)
        order = np.argsort(result.W[ik])
        assert result.W[ik][order[-1]] == pytest.approx(1.0, abs=1e-9)
        assert result.W[ik][order[0]] == pytest.approx(0.0, abs=1e-9)
        assert result.E_sc[ik][order[-1]] == pytest.approx(e_pc, abs=1e-9)
        assert result.E_pc[ik][0] == pytest.approx(e_pc, abs=1e-9)


@pytest.mark.unit
def test_plus_convention_with_explicit_sign():
    pc = _model_dict(A_PC, TAU_PC, +1, fourier_sign=+1)
    sc = _model_dict(A_SC, TAU_SC, +1, fourier_sign=+1)
    _assert_clean_unfolding(_unfold(pc, sc))


@pytest.mark.unit
def test_ifft_convention_with_explicit_sign():
    """The projected-DFT convention (hopping stored at tau_B - R) with fourier_sign = -1."""
    pc = _model_dict(A_PC, TAU_PC, -1, fourier_sign=-1)
    sc = _model_dict(A_SC, TAU_SC, -1, fourier_sign=-1)
    _assert_clean_unfolding(_unfold(pc, sc))


@pytest.mark.unit
def test_mixed_conventions_are_reconciled():
    pc = _model_dict(A_PC, TAU_PC, +1, fourier_sign=+1)
    sc = _model_dict(A_SC, TAU_SC, -1, fourier_sign=-1)
    _assert_clean_unfolding(_unfold(pc, sc))


@pytest.mark.unit
def test_convention_detected_when_sign_absent():
    pc = _model_dict(A_PC, TAU_PC, +1)
    sc = _model_dict(A_SC, TAU_SC, -1)
    _assert_clean_unfolding(_unfold(pc, sc))


@pytest.mark.unit
def test_wrong_sign_gives_mirror_point_weights():
    """Documents the failure mode: with the sign of R wrong the weights are cos^2 / sin^2(k_x a)."""
    pc = _model_dict(A_PC, TAU_PC, +1, fourier_sign=+1)
    sc = _model_dict(A_SC, TAU_SC, -1, fourier_sign=+1)  # ifft-convention data declared as '+'
    result = _unfold(pc, sc, nk=1)
    kx = result.kpath_cart[-1][0]
    expected = sorted([np.cos(2.0 * np.pi * kx) ** 2, np.sin(2.0 * np.pi * kx) ** 2])
    assert sorted(result.W[-1]) == pytest.approx(expected, abs=1e-9)
    assert result.W[-1].sum() == pytest.approx(1.0, abs=1e-9)  # the sum rule cannot see it


@pytest.mark.unit
def test_detect_fourier_sign():
    R = _fft_grid(A_SC)
    norb = np.ones(2, dtype=int)
    assert _detect_fourier_sign(_hoppings(TAU_SC, R, +1), R, TAU_SC, norb) == 1
    assert _detect_fourier_sign(_hoppings(TAU_SC, R, -1), R, TAU_SC, norb) == -1


@pytest.mark.unit
def test_extract_hamiltonian_flips_R_for_ifft_convention():
    sc = _model_dict(A_SC, TAU_SC, -1, fourier_sign=-1)
    _, R, nawf, nspin, norb = _extract_hamiltonian(sc)
    np.testing.assert_allclose(R, -sc['_paoflow']['R'])
    assert (nawf, nspin) == (2, 1)
    np.testing.assert_array_equal(norb, [1, 1])
    with pytest.raises(ValueError):
        _extract_hamiltonian(_model_dict(A_SC, TAU_SC, -1, fourier_sign=2))


class _StubController:
    """Minimal stand-in for DataController.data_dicts() after pao_hamiltonian()."""

    def __init__(self, arrays, attributes):
        self.data_arrays, self.data_attributes = arrays, attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


@pytest.mark.unit
def test_paoflow_to_model_dict_from_a_dft_like_run():
    alat = 10.0
    HRs = _hoppings(TAU_SC, _fft_grid(A_SC), -1).reshape(2, 2, NGRID, NGRID, NGRID, 1)
    arrays = {
        'a_vectors': A_SC,
        'tau': TAU_SC * alat,
        'atoms': ['X', 'X'],
        'HRs': HRs,
        'naw': np.array([1, 1]),
    }
    attributes = {'alat': alat}
    md = paoflow_to_model_dict(_StubController(arrays, attributes))
    assert md['_paoflow']['fourier_sign'] == -1
    assert md['_paoflow']['HRs'].shape == (2, 2, NGRID**3, 1)
    assert md['_paoflow']['R'].shape == (NGRID**3, 3)
    np.testing.assert_allclose(md['model']['atoms']['1']['tau'], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(md['_paoflow']['norbitals'], [1, 1])
    assert 'R' in arrays  # built by get_R_grid_fft on the controller
    # usable directly as the supercell of the unfolding
    _assert_clean_unfolding(_unfold(_model_dict(A_PC, TAU_PC, +1, fourier_sign=+1), md))
