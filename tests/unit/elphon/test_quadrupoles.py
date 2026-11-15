"""Dynamical quadrupoles from the long-wavelength expansion of a model density response."""

import numpy as np
import pytest

from PAOFLOW.elphon.long_range import read_quadrupole_fmt
from PAOFLOW.elphon.quadrupoles import (
    DIRECTIONS,
    calibrate_sign,
    cartesian_density_response,
    expansion_coefficients,
    fit_multipoles,
    single_q_phonon_inputs,
    write_quadrupole_fmt,
)

pytestmark = pytest.mark.unit


def _model_crystal(rng):
    """One atom in a cubic cell with a Gaussian response of known dipole and quadrupole moments."""
    alat, n = 20.0, 48
    sigma = 1.6
    tau = np.array([[10.0, 10.0, 10.0]])
    axis = (np.arange(n) + 0.5) * alat / n
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    r = np.stack([x - tau[0, 0], y - tau[0, 1], z - tau[0, 2]], axis=-1)  # (n, n, n, 3)
    g = np.exp(-(r**2).sum(-1) / (2 * sigma**2)) / ((2 * np.pi) ** 1.5 * sigma**3)
    grad = -r / sigma**2 * g[..., None]  # d_a g
    hess = (np.einsum('...a,...g->...ag', r, r) / sigma**4 - np.eye(3) / sigma**2) * g[
        ..., None, None
    ]
    zstar_el = rng.normal(size=(1, 3, 3))
    quad = rng.normal(size=(1, 3, 3, 3))
    quad = 0.5 * (quad + np.swapaxes(quad, 2, 3))  # symmetric in the last two indices
    # dn^(beta)(r) = sum_a z_{a beta} d_a g - 1/2 sum_{ag} Q_{beta, ag} d_a d_g g
    dn = np.einsum('ab,...a->b...', zstar_el[0], grad) - 0.5 * np.einsum(
        'bag,...ag->b...', quad[0], hess
    )
    return alat, tau, zstar_el, quad, dn, np.stack([x, y, z], axis=-1)


ZION = np.array([4.0])


def _samples(alat, tau, dn, xyz, epsilon, magnitudes=(0.01, 0.02), sign=1.0):
    """The electronic response of a finite-q DFPT run: the total (ion + electron) bare charge
    divided by eps_L, minus the point ion that moves rigidly."""
    samples = []
    for d in DIRECTIONS:
        d = np.asarray(d, dtype=float) / np.linalg.norm(d)
        for s in magnitudes:
            q_cart = s * d
            q_bohr = q_cart * 2 * np.pi / alat
            periodic = dn * np.exp(-1j * (xyz @ q_bohr))[None]  # periodic part e^{-iq.r} dn(r)
            bare_el = expansion_coefficients(periodic, q_cart, tau, alat, alat**3)
            ionic = -1j * q_bohr[None, :] * ZION[:, None]
            eps_l = float(d @ epsilon @ d)
            screened_el = (bare_el + ionic) / eps_l - ionic
            samples.append((q_cart, sign * screened_el))
    return samples


def test_moments_are_recovered_from_the_expansion():
    rng = np.random.default_rng(3)
    alat, tau, zstar_el, quad, dn, xyz = _model_crystal(rng)
    epsilon = np.diag([2.0, 3.0, 4.0]) + 0.3 * (np.ones((3, 3)) - np.eye(3))
    zstar = zstar_el + ZION[:, None, None] * np.eye(3)[None]
    fit = fit_multipoles(_samples(alat, tau, dn, xyz, epsilon), alat, epsilon, ZION)
    assert np.allclose(fit['zstar'], zstar, atol=2e-3)
    assert np.allclose(fit['quadrupoles'], quad, atol=5e-3 * np.abs(quad).max())
    assert fit['residual_quadrupole'] < 1e-2
    for q_mag in fit['quadrupoles_by_magnitude'].values():
        assert np.allclose(q_mag, quad, atol=3e-2 * np.abs(quad).max())


def test_sign_calibration_flips_a_charge_density_convention():
    rng = np.random.default_rng(4)
    alat, tau, zstar_el, quad, dn, xyz = _model_crystal(rng)
    epsilon = 5.0 * np.eye(3)
    zstar = zstar_el + ZION[:, None, None] * np.eye(3)[None]
    raw = fit_multipoles(_samples(alat, tau, dn, xyz, epsilon, sign=-1.0), alat)['zstar']
    assert calibrate_sign(raw, zstar, epsilon, ZION) == -1.0
    assert calibrate_sign(-raw, zstar, epsilon, ZION) == 1.0


def test_pattern_contraction_and_fmt_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    nmodes, nr = 6, (4, 4, 4)
    cart = rng.normal(size=(nmodes,) + nr) + 1j * rng.normal(size=(nmodes,) + nr)
    u, _ = np.linalg.qr(rng.normal(size=(nmodes, nmodes)) + 1j * rng.normal(size=(nmodes, nmodes)))
    patterns = np.einsum(
        'vc,c...->v...', u.T, cart
    )  # records in the pattern basis: dv_pat = u^T dv_cart
    assert np.allclose(cartesian_density_response(patterns, u), cart)
    quad = rng.normal(size=(2, 3, 3, 3))
    quad = 0.5 * (quad + np.swapaxes(quad, 2, 3))
    path = write_quadrupole_fmt(str(tmp_path / 'quadrupole.fmt'), quad)
    assert np.allclose(read_quadrupole_fmt(path, 2), quad, atol=1e-6)


def test_single_q_inputs():
    runs = single_q_phonon_inputs('si', magnitudes=(0.02,), directions=DIRECTIONS)
    assert len(runs) == 6
    name, q, text = runs[3]
    assert name == 'q4_0p02' and np.isclose(np.linalg.norm(q), 0.02)
    assert (
        "fildrho = 'drho'" in text
        and 'ldisp = .false.' in text
        and text.strip().endswith('%.10f %.10f %.10f' % tuple(q))
    )
