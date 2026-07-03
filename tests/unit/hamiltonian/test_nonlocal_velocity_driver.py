"""Phase 3d tests: driver-facing wrapper + module-prereq wiring.

These tests cover the public surface that the PAOFLOW class uses to
plug the non-local velocity correction into ``gradient_and_momenta``,
without standing up the full MPI-backed PAOFLOW pipeline (which is
exercised elsewhere by the cubium Hellmann-Feynman fixture).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    build_nl_real_space_tables,
    build_nonlocal_velocity_kspace,
    compute_nonlocal_velocity_on_grid,
    enumerate_nl_pairs,
    load_beta_projectors,
    load_pao_orbitals,
)
from PAOFLOW.utils.module_prerequisites import module_pre_reqs


class _DC:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


def _si_dc():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    fpath = os.path.join(repo_root, 'examples/qe_examples/example16/Si_ONCV')
    alat = 10.2
    a_cart = 0.5 * alat * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.25 * alat] * 3])
    arry = {
        'species': [('Si', 'Si_ONCV_PBE_sr.UPF')],
        'atoms': ['Si', 'Si'],
        'tau': tau,
        'a_vectors': a_cart / alat,
    }
    attr = {'fpath': fpath, 'alat': alat}
    return _DC(arry, attr), a_cart


@pytest.fixture(scope='module')
def si_tables():
    dc, a_cart = _si_dc()
    b = load_beta_projectors(dc)
    p = load_pao_orbitals(dc)
    pairs = enumerate_nl_pairs(b, p, a_cart, pao_tol=1e-2)
    t = build_nl_real_space_tables(b, p, pairs, q_max=15.0, n_q=300)
    return b, p, t


def test_module_pre_reqs_registers_nl_correction():
    """The schedule checker must know about the new method's dependency."""
    assert 'nonlocal_velocity_correction' in module_pre_reqs
    assert module_pre_reqs['nonlocal_velocity_correction'] == ['gradient_and_momenta']


def test_compute_on_grid_accepts_3xN_layout(si_tables):
    """``arry['kgrid']`` is shape ``(3, nktot)``; the helper must accept it."""
    b, p, t = si_tables
    rng = np.random.default_rng(0)
    k_3xN = rng.normal(size=(3, 4)) * 0.2  # 4 k-points in 2π/alat
    alat = 10.2
    dP = compute_nonlocal_velocity_on_grid(b, p, t, k_3xN, alat)
    assert dP.shape == (4, 3, p.total_nlm, p.total_nlm)


def test_compute_on_grid_matches_low_level_with_unit_conversion(si_tables):
    """The helper is exactly ``build_nonlocal_velocity_kspace`` after
    multiplying the input k-grid by ``2π/alat``."""
    b, p, t = si_tables
    alat = 10.2
    k_2pi_alat = np.array([[0.1, 0.2, -0.05], [0.3, 0.0, 0.4]])  # (nk, 3), 2π/alat units
    dP_helper = compute_nonlocal_velocity_on_grid(b, p, t, k_2pi_alat, alat)
    dP_low = build_nonlocal_velocity_kspace(
        b, p, t, k_2pi_alat * (2.0 * np.pi / alat), units='rydberg'
    )
    np.testing.assert_allclose(dP_helper, dP_low, atol=1e-14)


def test_compute_on_grid_hartree_vs_rydberg(si_tables):
    b, p, t = si_tables
    alat = 10.2
    k = np.array([[0.1, 0.2, -0.05]])
    dP_h = compute_nonlocal_velocity_on_grid(b, p, t, k, alat, units='hartree')
    dP_r = compute_nonlocal_velocity_on_grid(b, p, t, k, alat, units='rydberg')
    np.testing.assert_allclose(dP_r, 2.0 * dP_h, atol=1e-14)


def test_compute_on_grid_rejects_bad_layout(si_tables):
    b, p, t = si_tables
    with pytest.raises(ValueError, match='2-D'):
        compute_nonlocal_velocity_on_grid(b, p, t, np.zeros(3), 10.2)
    with pytest.raises(ValueError, match='length-3'):
        compute_nonlocal_velocity_on_grid(b, p, t, np.zeros((4, 5)), 10.2)


def test_compute_on_grid_hermiticity_on_uniform_grid(si_tables):
    """Sample a tiny 2×2×2 FFT grid in 2π/alat units and verify Hermiticity."""
    b, p, t = si_tables
    alat = 10.2
    nk1 = nk2 = nk3 = 2
    nktot = nk1 * nk2 * nk3
    k = np.zeros((3, nktot))
    for i in range(nk1):
        for j in range(nk2):
            for kk in range(nk3):
                n = kk + j * nk3 + i * nk2 * nk3
                Rx = i / nk1
                Ry = j / nk2
                Rz = kk / nk3
                if Rx >= 0.5:
                    Rx -= 1.0
                if Ry >= 0.5:
                    Ry -= 1.0
                if Rz >= 0.5:
                    Rz -= 1.0
                k[:, n] = (Rx, Ry, Rz)

    dP = compute_nonlocal_velocity_on_grid(b, p, t, k, alat)
    assert dP.shape == (nktot, 3, p.total_nlm, p.total_nlm)
    for ik in range(nktot):
        for alpha in range(3):
            M = dP[ik, alpha]
            np.testing.assert_allclose(M, np.conj(M.T), atol=1e-11)


# ----------------------------------------------------------------------
# Phase 4 step 2: gate plumbing via the public method on a stub PAOFLOW.
# ----------------------------------------------------------------------


class _StubDC:
    """Minimal DataController stand-in for ``nonlocal_velocity_correction``."""

    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


class _StubPF:
    """Minimal PAOFLOW stand-in that satisfies the public method's needs."""

    def __init__(self, dc):
        self.data_controller = dc
        self.reports = []
        self.exceptions = []

    def report_module_time(self, label):
        self.reports.append(label)

    def report_exception(self, label):
        self.exceptions.append(label)


def _si_stub_pf(nspin=1, nk_per_axis=2, seed=11):
    """Build a Si stub with a real catalog + a synthetic random dHksp.

    The dHksp values are arbitrary (we only test that the inject path
    mutates them by the documented amount); the catalog and k-grid use
    the real Si ONCV setup from ``_si_dc``.
    """
    dc_si, _ = _si_dc()
    arry_si, attr_si = dc_si.data_dicts()

    nk1 = nk2 = nk3 = nk_per_axis
    nktot = nk1 * nk2 * nk3
    kgrid = np.zeros((3, nktot))
    for i in range(nk1):
        for j in range(nk2):
            for kk in range(nk3):
                n = kk + j * nk3 + i * nk2 * nk3
                Rx = i / nk1
                Ry = j / nk2
                Rz = kk / nk3
                if Rx >= 0.5:
                    Rx -= 1.0
                if Ry >= 0.5:
                    Ry -= 1.0
                if Rz >= 0.5:
                    Rz -= 1.0
                kgrid[:, n] = (Rx, Ry, Rz)

    # nawf for Si is 8 (2 atoms × (1 s + 3 p)).  We just need a plausible
    # shape; populate with random complex.
    nawf = 8
    rng = np.random.default_rng(seed)
    dHksp = rng.standard_normal((nktot, 3, nawf, nawf, nspin)) + 1j * rng.standard_normal(
        (nktot, 3, nawf, nawf, nspin)
    )

    arry = dict(arry_si)
    arry['kgrid'] = kgrid
    arry['nk1'] = nk1
    arry['nk2'] = nk2
    arry['nk3'] = nk3
    arry['dHksp'] = dHksp

    attr = dict(attr_si)
    attr['nawf'] = nawf
    attr['nspin'] = nspin
    attr['abort_on_exception'] = True

    pf = _StubPF(_StubDC(arry, attr))
    return pf


def _call_nlvc(pf, **kwargs):
    """Invoke ``PAOFLOW.nonlocal_velocity_correction`` on a stub instance."""
    from PAOFLOW.PAOFLOW import PAOFLOW as PF_class

    # Tighter tol on the stub: small lattice, fewer pairs → faster.
    kwargs.setdefault('pao_tol', 1.0e-2)
    kwargs.setdefault('q_max', 15.0)
    kwargs.setdefault('n_q', 300)
    PF_class.nonlocal_velocity_correction(pf, **kwargs)


@pytest.fixture(scope='module')
def si_stub_pf_with_dP():
    """Run the public method once with inject=False and cache the result.

    Doing the catalog/tables build once (expensive) and reusing for
    every sign/inject test below keeps the suite fast.
    """
    pf = _si_stub_pf()
    arry, _ = pf.data_controller.data_dicts()
    dHksp_pristine = arry['dHksp'].copy()
    _call_nlvc(pf, inject=False)
    dP = arry['Delta_pksp'].copy()
    # restore pristine dHksp (the no-inject call should not have touched it)
    np.testing.assert_array_equal(arry['dHksp'], dHksp_pristine)
    return pf, dP, dHksp_pristine


def test_nlvc_stores_Delta_pksp_without_injecting(si_stub_pf_with_dP):
    """``inject=False`` (default) populates ``arry['Delta_pksp']`` only."""
    pf, dP, dHksp_pristine = si_stub_pf_with_dP
    arry, _ = pf.data_controller.data_dicts()
    # dP shape must match the dHksp[:4] block.
    assert dP.shape == arry['dHksp'].shape[:4]
    # dHksp untouched.
    np.testing.assert_array_equal(arry['dHksp'], dHksp_pristine)
    # Reports recorded.
    assert 'NL velocity correction' in pf.reports


def test_nlvc_caches_catalogs_across_calls(si_stub_pf_with_dP):
    """The catalog + tables are stashed on arry and reused on repeat calls."""
    pf, _, _ = si_stub_pf_with_dP
    arry, _ = pf.data_controller.data_dicts()
    for key in ('_NL_beta_catalog', '_NL_pao_catalog', '_NL_pairs', '_NL_tables'):
        assert key in arry, f'missing cache key {key!r}'


def test_nlvc_inject_default_sign_adds_lambda_dP(si_stub_pf_with_dP):
    """For the scalar path, the calibrated default ``sign=+1`` adds λ·dP.

    ``sign=None`` (the new default) resolves per path: ``+1`` for the
    scalar / ad-hoc-SO path (Cu-calibrated) and ``-1`` for the
    fully-relativistic jm-kspace path.  The Si stub is scalar, so the
    resolved default is ``+1``.
    """
    pf, dP, dHksp_pristine = si_stub_pf_with_dP
    arry, _ = pf.data_controller.data_dicts()
    arry['dHksp'] = dHksp_pristine.copy()
    _call_nlvc(pf, inject=True)  # default sign=None -> +1 for scalar path
    expected = dHksp_pristine + RYDBERG_IN_EV * dP[..., None]
    np.testing.assert_allclose(arry['dHksp'], expected, atol=1e-12)


def test_nlvc_inject_negative_sign_subtracts_lambda_dP(si_stub_pf_with_dP):
    """``sign=-1`` subtracts λ·dP (explicit override of the per-path default)."""
    pf, dP, dHksp_pristine = si_stub_pf_with_dP
    arry, _ = pf.data_controller.data_dicts()
    arry['dHksp'] = dHksp_pristine.copy()
    _call_nlvc(pf, inject=True, sign=-1)
    expected = dHksp_pristine - RYDBERG_IN_EV * dP[..., None]
    np.testing.assert_allclose(arry['dHksp'], expected, atol=1e-12)


def test_nlvc_inject_positive_sign_adds_lambda_dP(si_stub_pf_with_dP):
    """``sign=+1`` flips the convention (calibration toggle)."""
    pf, dP, dHksp_pristine = si_stub_pf_with_dP
    arry, _ = pf.data_controller.data_dicts()
    arry['dHksp'] = dHksp_pristine.copy()
    _call_nlvc(pf, inject=True, sign=+1)
    expected = dHksp_pristine + RYDBERG_IN_EV * dP[..., None]
    np.testing.assert_allclose(arry['dHksp'], expected, atol=1e-12)


def test_nlvc_inject_hartree_uses_factor_two(si_stub_pf_with_dP):
    """``units='hartree'`` doubles the conversion factor — and rescales dP."""
    pf, _, dHksp_pristine = si_stub_pf_with_dP
    arry, _ = pf.data_controller.data_dicts()
    arry['dHksp'] = dHksp_pristine.copy()
    _call_nlvc(pf, inject=True, units='hartree')
    dP_h = arry['Delta_pksp']
    # dHksp += +2·λ · dP_h  (Hartree convention, scalar-path default sign=+1)
    expected = dHksp_pristine + 2.0 * RYDBERG_IN_EV * dP_h[..., None]
    np.testing.assert_allclose(arry['dHksp'], expected, atol=1e-12)


# Make RYDBERG_IN_EV visible to the tests above.
from PAOFLOW.hamiltonian.nonlocal_velocity import RYDBERG_IN_EV  # noqa: E402
