"""Velocity-gauge (Kubo) second-order response of tight-binding models: the test reference of the
density-matrix engine (plan G2a, wiki ``PAOFLOW_NLO_gyration_plan.md``).

For a tight-binding model the Peierls substitution :math:`H(\\mathbf k + \\boldsymbol\\kappa(t))`,
:math:`\\dot{\\boldsymbol\\kappa} = \\mathbf E` (charge +1, the convention of every PAOFLOW NLO output), is
exact with the orbital-centre position operator, so the velocity gauge is equivalent to the length gauge
of the engine. Expanding to third order in :math:`\\kappa`,

.. math::

    H = H_0 + \\kappa^b h^b + \\tfrac12\\kappa^b\\kappa^c h^{bc}, \\qquad
    j^a = h^a + \\kappa^b h^{ab} + \\tfrac12\\kappa^b\\kappa^c h^{abc},

with :math:`h^{a\\ldots} = \\partial_a\\ldots H^A` (atomic gauge). Fields switched on adiabatically put
:math:`\\omega + i\\eta` in every first-order denominator and in :math:`\\kappa(\\omega) = iE/(\\omega + i\\eta)`,
and :math:`\\omega_1 + \\omega_2 + 2i\\eta` in every second-order one, populations included. No formula is
taken from the literature: the coefficient is evaluated from the expansion of the density matrix.
"""

from __future__ import annotations

import numpy as np

from PAOFLOW.response.density_matrix_response import fermi_dirac_derivatives
from PAOFLOW.response.nonlinear_optics import _k_grid
from PAOFLOW.utils.constants import ELECTRON_CHARGE_SI, EPSILON0_SI, HBAR


def atomic_gauge_vertices(model, k_cryst: np.ndarray, order: int = 3) -> list[np.ndarray]:
    """``[H, dH (nk,3,n,n), d2H (nk,3,3,n,n), d3H (nk,3,3,3,n,n)]`` of an ExactModel in the atomic gauge."""
    kc = model.k_cart(k_cryst)  # (nk, 3) bohr^-1
    n, nk = model.nawf, kc.shape[0]
    out = [np.zeros((nk, n, n), dtype=complex)]
    shapes = [(nk, 3, n, n), (nk, 3, 3, n, n), (nk, 3, 3, 3, n, n)]
    out += [np.zeros(s, dtype=complex) for s in shapes[:order]]
    for i, j, R, h in model.hops:
        d = model.alat * ((np.asarray(R, dtype=float) + model.tau[j] - model.tau[i]) @ model.at)
        phase = h * np.exp(1j * (kc @ d))
        out[0][:, i, j] += phase
        id_ = 1j * d
        if order >= 1:
            out[1][:, :, i, j] += phase[:, None] * id_[None, :]
        if order >= 2:
            out[2][:, :, :, i, j] += phase[:, None, None] * np.einsum('a,b->ab', id_, id_)[None]
        if order >= 3:
            t3 = np.einsum('a,b,c->abc', id_, id_, id_)
            out[3][:, :, :, :, i, j] += phase[:, None, None, None] * t3[None]
    return out


def kubo_second_order(
    model,
    nk,
    frequencies: tuple[float, float],
    components,
    eta: float,
    nelec: float | None = None,
    fermi_energy: float | None = None,
    temperature: float = 0.0,
    spin_factor: float = 2.0,
    kblock: int = 4096,
) -> np.ndarray:
    """Full second-order coefficient of :math:`E^b(\\omega_1)E^c(\\omega_2)` in :math:`J^a`, engine units.

    Parameters
    ----------
    model : ExactModel
    nk : sequence of int
    frequencies : tuple of float
        :math:`(\\hbar\\omega_1, \\hbar\\omega_2)` (eV): ``(w, -w)`` for the zero-frequency photocurrent,
        ``(w, w)`` for SHG.
    components : sequence of ``(a, b, c)``
    eta : float
        Adiabatic switching rate (eV).
    nelec, fermi_energy, temperature, spin_factor
        Band-index occupations (``nelec``) or Fermi-Dirac at ``fermi_energy`` and ``temperature``.
    kblock : int, optional

    Returns
    -------
    ndarray ``(ncomp,)`` complex
        :math:`\\sum_k w_k K^{abc}` in eV bohr per (eV/bohr)^2, the ``acc`` of the engine before its SI
        prefactor (:func:`sigma_si`, :func:`chi_si`).
    """
    w1, w2 = (float(x) for x in frequencies)
    z1, z2, Z = w1 + 1j * eta, w2 + 1j * eta, w1 + w2 + 2j * eta
    k1, k2 = 1j / z1, 1j / z2
    K = _k_grid(tuple(int(x) for x in nk))
    nocc = None if nelec is None else int(round(float(nelec) / float(spin_factor)))
    acc = np.zeros(len(components), dtype=complex)

    def tr(x, y):
        return np.einsum('knm,kmn->k', x, y)

    for b0 in range(0, K.shape[0], int(kblock)):
        H, h1, h2, h3 = atomic_gauge_vertices(model, K[b0 : b0 + int(kblock)])
        E, U = np.linalg.eigh(H)
        Ud = np.conj(np.swapaxes(U, 1, 2))
        if fermi_energy is None:
            f = np.zeros_like(E)
            f[:, :nocc] = 1.0
        else:
            f = fermi_dirac_derivatives(E, float(fermi_energy), float(temperature))[0]

        def band(op):
            return Ud @ op @ U

        eps = E[:, :, None] - E[:, None, :]
        fnm = f[:, :, None] - f[:, None, :]
        for ic, (a, b, c) in enumerate(components):
            ha, hb, hc = band(h1[:, a]), band(h1[:, b]), band(h1[:, c])
            hab, hac, hbc = band(h2[:, a, b]), band(h2[:, a, c]), band(h2[:, b, c])
            habc = band(h3[:, a, b, c])
            rho1_b = -fnm * k1 * hb / (z1 - eps)
            rho1_c = -fnm * k2 * hc / (z2 - eps)
            S = (
                k2 * (hc @ rho1_b - rho1_b @ hc)
                + k1 * (hb @ rho1_c - rho1_c @ hb)
                - k1 * k2 * fnm * hbc
            )
            rho2 = S / (Z - eps)
            J = (
                tr(rho2, ha)
                + k2 * tr(rho1_b, hac)
                + k1 * tr(rho1_c, hab)
                + k1 * k2 * np.einsum('kn,knn->k', f, habc)
            )
            acc[ic] += J.sum()
    return acc


def sigma_si(model, nk, acc: np.ndarray, spin_factor: float = 2.0) -> np.ndarray:
    """Zero-frequency photocurrent :math:`\\sigma^{abc}` (A/V^2) of :func:`kubo_second_order`."""
    norm = float(spin_factor) / (int(np.prod(nk)) * model.volume)
    return ELECTRON_CHARGE_SI / HBAR * norm * acc


def chi_si(model, nk, acc: np.ndarray, w: float, spin_factor: float = 2.0) -> np.ndarray:
    """SHG :math:`\\chi^{abc}` (pm/V) of :func:`kubo_second_order` at ``(w, w)``."""
    norm = float(spin_factor) / (int(np.prod(nk)) * model.volume)
    return 1.0e12 * ELECTRON_CHARGE_SI / EPSILON0_SI * norm * 0.5 / (-2.0j * float(w)) * acc
