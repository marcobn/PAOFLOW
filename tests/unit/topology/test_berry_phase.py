"""Wilson loops of ``berry_phase``: occupied bands, the cell-centred (Zak) loop, the writers.

Pins the 16 Nov 2026 fixes of ``topology/do_berry_phase.py``:

* the occupied subspace is ``nelec`` divided by the spin factor (it was the lowest ``nelec``
  bands, twice the occupied subspace of a spin-degenerate model);
* ``method='zak'`` is the Wilson loop of the cell-centred states, closed with the state at
  ``k_0 + G``: its phase is 2 pi times the Wannier centre (the old closing link dropped the last
  point and mixed units);
* ``berry_phase(eigvals=True)`` writes its eigenphases (the writer did not exist);
* ``method='berry'`` is unchanged (checked against the previous implementation, verbatim).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import scipy.linalg as la

from PAOFLOW.topology.do_berry_phase import (
    do_berry_bands,
    do_phase,
    occupied_band_count,
    orbital_fractional_positions,
)

pytestmark = pytest.mark.unit

pytest.importorskip('pythtb')


def _skip_without_pythtb2(model) -> None:
    """The *_pythtb labels build PythTB 2 models (pythtb.Lattice; PythTB 2 needs Python >= 3.12)."""
    if isinstance(model, dict) and str(model.get('label', '')).endswith('_pythtb'):
        pythtb = pytest.importorskip('pythtb')
        if not hasattr(pythtb, 'Lattice'):
            pytest.skip('the *_pythtb models need PythTB 2 (Python >= 3.12)')


def _model(spec, tmp_path):
    from PAOFLOW import PAOFLOW

    _skip_without_pythtb2(spec)

    with contextlib.redirect_stdout(io.StringIO()):
        return PAOFLOW.PAOFLOW(model=spec, outputdir=str(tmp_path), verbose=False)


def _loop(p, k0, k1, nk=200, method='berry', eigvals=False, sub=None):
    with contextlib.redirect_stdout(io.StringIO()):
        p.berry_phase(
            kspace_method='path',
            berry_path='A-B',
            high_sym_points={'A': np.asarray(k0, float), 'B': np.asarray(k1, float)},
            nk1=nk,
            method=method,
            eigvals=eigvals,
            sub=sub,
        )
    arry, attr = p.data_controller.data_dicts()
    return arry['berry_phase'] if eigvals else attr['berry_phase']


def _ssh(v, w, tmp_path):
    return _model({'label': 'ssh_pythtb', 'v': v, 'w': w, 'nelec': 2}, tmp_path)


def _rice_mele(dt, delta, tmp_path):
    return _model({'label': 'rice_mele', 't': 1.0, 'dt': dt, 'delta': delta}, tmp_path)


# --------------------------------------------------------------------------- occupied bands
def test_occupied_bands_are_electrons_over_the_spin_factor():
    hrs = np.zeros((4, 4, 1, 1, 1, 1), complex)
    assert occupied_band_count({'HRs': hrs}, {'nelec': 2.0, 'dftSO': False}) == 1
    assert occupied_band_count({'HRs': hrs}, {'nelec': 2.0, 'dftSO': True}) == 2
    with pytest.raises(ValueError):
        occupied_band_count({'HRs': hrs}, {'nelec': 3.0, 'dftSO': False})
    # collinear spin-polarized: the bands of channel 0 below the Fermi level
    hrs2 = np.zeros((3, 3, 1, 1, 1, 2), complex)
    hrs2[:, :, 0, 0, 0, 0] = np.diag([-1.0, -0.5, 1.0])
    hrs2[:, :, 0, 0, 0, 1] = np.diag([-1.0, 0.5, 1.0])
    assert occupied_band_count({'HRs': hrs2}, {'nelec': 3.0}) == 2


def test_ssh_berry_phase_uses_the_occupied_band(tmp_path):
    # two electrons fill the lower band of the spin-degenerate chain: the topological phase is pi
    # (the lowest nelec = 2 bands, both of them, gave 0); no ibrav is needed for an explicit path
    p = _ssh(0.7, 1.3, tmp_path)
    assert 'ibrav' not in p.data_controller.data_attributes
    phase = _loop(p, [0, 0, 0], [1, 0, 0])
    assert abs(abs(phase) - np.pi) < 1e-8


# --------------------------------------------------------------------------- the Zak (cell-centred) loop
@pytest.mark.parametrize('w, centre', [(0.5, 0.25), (1.5, 0.75)])
def test_zak_phase_of_ssh_is_the_wannier_centre(tmp_path, w, centre):
    # orbitals at 0 and a/2: the electron sits on the strong bond, inside (v > w) or between cells
    p = _ssh(1.0, w, tmp_path)
    zak = _loop(p, [0, 0, 0], [1, 0, 0], method='zak')
    assert abs((zak / (2 * np.pi)) % 1.0 - centre) < 1e-8


def test_zak_phase_is_the_lattice_phase_plus_the_orbital_term(tmp_path):
    # Rice-Mele without inversion through the bond: x = phi_lat / 2 pi + sum_a <|c_a|^2> s_a,
    # with the weights averaged on the loop's own points
    p = _rice_mele(0.3, 0.3, tmp_path)
    nk = 400
    phi_lat = _loop(p, [0, 0, 0], [1, 0, 0], nk=nk)
    zak = _loop(p, [0, 0, 0], [1, 0, 0], nk=nk, method='zak')
    arry, attr = p.data_controller.data_dicts()
    vecs = arry['berry_v_k'][
        :-1, :, 0, 0
    ]  # the lower band on the open path (the end repeats k_0 + G)
    weights = (np.abs(vecs) ** 2).mean(axis=0)
    s = orbital_fractional_positions(arry, attr)[:, 0]
    expected = (phi_lat / (2 * np.pi) + weights @ s) % 1.0
    assert abs((zak / (2 * np.pi)) % 1.0 - expected) < 1e-4
    assert abs((zak / (2 * np.pi)) % 1.0 - 0.3310) < 5e-4


def test_zak_centre_pinned_by_inversion(tmp_path):
    # equal hoppings, Delta > 0 on A: inversion through the B site pins the centre on B (a/2),
    # while the lattice-gauge phase alone is not quantized
    p = _rice_mele(0.0, 0.4, tmp_path)
    zak = _loop(p, [0, 0, 0], [1, 0, 0], method='zak')
    phi_lat = _loop(p, [0, 0, 0], [1, 0, 0])
    assert abs((zak / (2 * np.pi)) % 1.0 - 0.5) < 1e-6
    assert abs((phi_lat / (2 * np.pi)) % 1.0 - 0.5) > 0.1


def test_zak_loop_does_not_depend_on_the_end_point(tmp_path):
    # the same loop with and without the end point k_0 + G on the contour
    p = _rice_mele(0.3, 0.3, tmp_path)
    arry, attr = p.data_controller.data_dicts()
    attr.update(berry_kspace_method='manual', berry_eigvals=False, berry_method='zak',
                berry_occupied=True, berry_closed=True)  # fmt: skip
    arry['berry_sub'] = None
    phases = []
    for endpoint in (True, False):
        k = np.linspace(0.0, 1.0, 301, endpoint=endpoint)
        arry['berry_kq'] = np.vstack([k, 0 * k, 0 * k])
        arry['berry_contour'] = np.copy(arry['berry_kq'])
        with contextlib.redirect_stdout(io.StringIO()):
            do_berry_bands(p.data_controller)
        phases.append(do_phase(p.data_controller))
    assert abs(np.angle(np.exp(1j * (phases[0] - phases[1])))) < 1e-3


# --------------------------------------------------------------------------- 'berry' unchanged
def _old_do_phase_berry(v_kp, contour, closed, eigvals, occ_idx):
    """The lattice-gauge branch of do_phase before 16 Nov 2026, verbatim in its arithmetic."""
    nkpts = v_kp.shape[0]
    if np.allclose(contour[:, 0], contour[:, -1]):
        closed = False
    dim = len(occ_idx)
    prd = np.eye(dim, dtype=complex)
    for ik in range(nkpts - 1):
        left = np.zeros((v_kp.shape[1], dim), dtype=complex)
        right = np.zeros((v_kp.shape[1], dim), dtype=complex)
        for idx, occ in enumerate(occ_idx):
            left[:, idx] = v_kp[ik, :, occ, 0]
            right[:, idx] = v_kp[ik + 1, :, occ, 0]
        ovr = np.dot(left.conj().T, right)
        Z, _, W = la.svd(ovr)
        prd = np.dot(prd, np.matmul(Z, W))
    if closed:
        left = np.zeros((v_kp.shape[1], dim), dtype=complex)
        right = np.zeros((v_kp.shape[1], dim), dtype=complex)
        for idx, occ in enumerate(occ_idx):
            left[:, idx] = v_kp[nkpts - 1, :, occ, 0]
            right[:, idx] = v_kp[0, :, occ, 0]
        ovr = np.dot(left.conj().T, right)
        Z, _, W = la.svd(ovr)
        prd = np.dot(prd, np.matmul(Z, W))
    if not eigvals:
        return -1.0 * np.angle(la.det(prd))
    return np.sort(-1.0 * np.angle(la.eig(prd)[0]))


@pytest.mark.parametrize('eigvals', [False, True])
@pytest.mark.parametrize('sub', [[0], [0, 1]])
def test_berry_method_is_unchanged(tmp_path, eigvals, sub):
    from PAOFLOW.models.models import honeycomb_model

    p = _model(honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=np.pi / 2), tmp_path)
    new = _loop(p, [0, 0.3, 0], [1, 0.3, 0], nk=80, eigvals=eigvals, sub=sub)
    arry = p.data_controller.data_arrays
    old = _old_do_phase_berry(arry['berry_v_k'], arry['berry_contour'], True, eigvals, sub)
    np.testing.assert_array_equal(new, old)


# --------------------------------------------------------------------------- writers
def test_eigvals_writers(tmp_path):
    from PAOFLOW.models.models import kane_mele_model

    p = _model(kane_mele_model(soc=0.06, rashba=0.0, delta=0.1), tmp_path)
    opath = p.data_controller.data_attributes['opath']
    phases = _loop(p, [0, 0.2, 0], [1, 0.2, 0], nk=60, eigvals=True)
    assert phases.shape == (2,)  # two occupied spinor bands
    data = np.loadtxt(f'{opath}/berry_phase.dat')
    np.testing.assert_allclose(data[1:], phases, atol=1e-11)

    with contextlib.redirect_stdout(io.StringIO()):
        p.berry_phase(
            kspace_method='track',
            kpath_funct=lambda k: (
                'A-B',
                {'A': np.array([0.0, k, 0.0]), 'B': np.array([1.0, k, 0.0])},
            ),
            nk1=40,
            nk2=5,
            eigvals=True,
            fname='flow',
        )
    flow = p.data_controller.data_arrays['berry_phase']
    data = np.loadtxt(f'{opath}/flow.dat')
    assert data.shape == (5, 3)
    np.testing.assert_allclose(data[:, 0], np.linspace(0.0, 1.0, 5), atol=1e-12)
    np.testing.assert_allclose(data[:, 1:], flow, atol=1e-11)


# --------------------------------------------------------------------------- orbital centres
def test_orbital_fractional_positions_units():
    a = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    s_true = np.array([[1 / 3, 1 / 3, 0.0], [2 / 3, 2 / 3, 0.0]])
    hrs = np.zeros((2, 2, 1, 1, 1, 1), complex)
    # models: tau in alat units
    arry = {'HRs': hrs, 'a_vectors': a, 'tau': s_true @ a}
    s = orbital_fractional_positions(arry, {'alat': 4.65, 'tau_units': 'alat'})
    np.testing.assert_allclose(s, s_true, atol=1e-12)
    # DFT: tau in bohr, one row per atom expanded with naw, and doubled to block spinors
    arry = {'HRs': np.zeros((6, 6, 1, 1, 1, 1), complex), 'a_vectors': a, 'tau': 4.65 * s_true @ a,
            'naw': np.array([1, 2])}  # fmt: skip
    s = orbital_fractional_positions(arry, {'alat': 4.65})
    np.testing.assert_allclose(s, np.vstack([s_true[[0, 1, 1]]] * 2), atol=1e-12)
