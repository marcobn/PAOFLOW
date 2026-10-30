r"""Isotropic Migdal-Eliashberg equations from the Eliashberg function ``a2F(omega)``.

Temperature-dependent superconducting properties in the isotropic,
Fermi-surface-restricted limit, following EPW's ``liso`` / ``limag`` / ``lpade``
/ ``lacon`` / ``tc_linear`` (EPW tutorial 04).  Everything depends only on
``a2F(omega)``, the Coulomb pseudopotential ``mu*`` (applied up to the Matsubara
cutoff ``wscut``, not rescaled) and the temperature:

* imaginary axis: ``Z(i w_n)`` and ``Delta(i w_n)`` on the fermionic Matsubara
  frequencies ``w_n = (2n+1) pi k_B T <= wscut`` (:func:`solve_imag_iso`);
* real axis: ``Z(w)``, ``Delta(w)`` from an N-point Pade approximant
  (:func:`pade_continuation`) and from the iterative analytic continuation of
  Marsiglio, Schossmann and Carbotte (:func:`analytic_continuation_iso`), the
  gap edge ``w = Re Delta(w)`` (:func:`gap_edge`) and the quasiparticle density
  of states (:func:`quasiparticle_dos`);
* the largest eigenvalue of the linearised gap equation, which crosses 1 at
  ``Tc`` (:func:`linearized_max_eigenvalue`).

With ``lambda(n) = int 2 w a2F(w) / (w^2 + (2 pi n k_B T)^2) dw`` and the sums
restricted to positive frequencies, the imaginary-axis equations are

.. math::

    Z_n = 1 + \frac{\pi T}{\omega_n}\sum_{n'}
          [\lambda(n-n') - \lambda(n+n'+1)]\,\frac{\omega_{n'}}{R_{n'}},
    \qquad
    Z_n\Delta_n = \pi T\sum_{n'}
          [\lambda(n-n') + \lambda(n+n'+1) - 2\mu^*]\,\frac{\Delta_{n'}}{R_{n'}},

with ``R = sqrt(w^2 + Delta^2)``.  Energies are in eV and temperatures in
kelvin.  ``a2F`` is resampled on a uniform grid ``nu_j = j * dw`` (as EPW's
``wsph``) and every frequency integral is the rectangle sum on that grid.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve
from scipy.special import expit

from .eph_kq import EV_TO_K, THZ_TO_EV, _gaussian_delta, mcmillan_allen_dynes_tc

KB_EV = 1.0 / EV_TO_K  # Boltzmann constant (eV/K)


# --------------------------------------------------------------------------- #
# Eliashberg function on a uniform grid                                       #
# --------------------------------------------------------------------------- #
def a2f_from_modes(
    lambda_qv: ArrayLike,
    omega_qv_thz: ArrayLike,
    q_weights: ArrayLike | None = None,
    degaussq_ev: float = 1.5e-4,
    nqstep: int = 500,
    omega_max_ev: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eliashberg function from the mode couplings, with an EPW-like phonon smearing.

    Parameters
    ----------
    lambda_qv, omega_qv_thz : array_like, shape ``(nq, nmode)``
        Mode-resolved coupling and frequencies (THz), e.g. the ``lambda_qv`` /
        ``omega_qv_thz`` keys of ``eliashberg_dense_q``.
    q_weights : array_like, shape ``(nq,)``, optional
        q-point weights (normalised here); uniform by default.
    degaussq_ev : float, optional
        Gaussian phonon smearing in eV (default 0.15 meV, EPW's isotropic
        Migdal-Eliashberg value of ``degaussq``).
    nqstep : int, optional
        Number of frequency points (EPW ``nqstep``, default 500).
    omega_max_ev : float, optional
        Highest phonon frequency (eV); defaults to ``max(omega_qv)``.

    Returns
    -------
    omega : ndarray, shape ``(nqstep,)``
        Frequencies ``j * 1.1 * omega_max / nqstep`` (eV), ``j = 1 .. nqstep``.
    a2F : ndarray, shape ``(nqstep,)``
        The Eliashberg function on ``omega``.

    Notes
    -----
    The Eliashberg function is

    .. math::

        \\alpha^2F(\\omega) = \\tfrac12\\sum_{q\\nu} w_q\\,\\lambda_{q\\nu}\\,
            \\omega_{q\\nu}\\,\\delta(\\omega-\\omega_{q\\nu}),

    with a Gaussian ``delta`` of width ``degaussq_ev``.  The grid extends to
    ``1.1 * omega_max`` (EPW ``wsphmax``) and starts at its spacing, the form
    every other function of this module expects.
    """
    lam_qv = np.asarray(lambda_qv, dtype=float)
    omega_qv_ev = np.asarray(omega_qv_thz, dtype=float) * THZ_TO_EV
    nq = lam_qv.shape[0]
    if q_weights is None:
        weights_q = np.full(nq, 1.0 / nq)
    else:
        weights_q = np.asarray(q_weights, dtype=float)
    weights_q = weights_q / weights_q.sum()
    omega_top = float(omega_qv_ev.max()) if omega_max_ev is None else float(omega_max_ev)
    spacing = 1.1 * omega_top / nqstep
    omega = spacing * np.arange(1, nqstep + 1)
    keep = omega_qv_ev > 1.0e-6
    mode_weights = (0.5 * weights_q[:, None] * lam_qv * omega_qv_ev)[keep]
    deltas = _gaussian_delta(omega[:, None] - omega_qv_ev[keep][None, :], degaussq_ev)
    return omega, deltas @ mode_weights


def _uniform_a2f(
    omega: ArrayLike, a2F: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eliashberg function on the uniform grid ``nu_j = j * dw``, ``j = 1 .. n``.

    Parameters
    ----------
    omega, a2F : array_like
        Frequencies (eV) and Eliashberg function on any increasing grid.

    Returns
    -------
    nu, a2F_uniform : ndarray
        The uniform grid and the Eliashberg function on it.

    Notes
    -----
    Grids that already have this form (:func:`a2f_from_modes`, EPW's ``.a2f``)
    are returned unchanged.  Others (e.g. starting at 0) are linearly
    interpolated on ``dw = max(omega) / n``, with ``n`` the number of positive
    frequencies.
    """
    omega = np.asarray(omega, dtype=float)
    a2F = np.asarray(a2F, dtype=float)
    positive = omega > 0.0
    nu = omega[positive]
    npoints = nu.size
    spacing = nu[-1] / npoints
    grid = spacing * np.arange(1, npoints + 1)
    if np.allclose(nu, grid, rtol=0.0, atol=1.0e-6 * spacing):
        return grid, a2F[positive]
    return grid, np.interp(grid, omega, a2F, left=0.0, right=0.0)


def real_axis_grid(omega: ArrayLike, a2F: ArrayLike, wscut: float) -> NDArray[np.float64]:
    """Real-frequency grid of the continuation, as EPW's ``ws``.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    wscut : float
        Upper frequency (eV).

    Returns
    -------
    ndarray
        ``w_i = i * dw <= wscut`` (eV), with ``dw`` the spacing of the uniform
        ``a2F`` grid, so that ``w_i +/- nu_j`` stay on the grid.
    """
    nu, _ = _uniform_a2f(omega, a2F)
    spacing = nu[0]
    return spacing * np.arange(1, int(wscut / spacing) + 1)


def allen_dynes_tc(omega: ArrayLike, a2F: ArrayLike, mu_star: float) -> float:
    """Allen-Dynes critical temperature from the moments of ``a2F``.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    mu_star : float
        Coulomb pseudopotential.

    Returns
    -------
    float
        ``Tc`` in K (:func:`~PAOFLOW.elphon.eph_kq.mcmillan_allen_dynes_tc`), or
        0 when ``lambda <= 0``.
    """
    nu, a2F_uniform = _uniform_a2f(omega, a2F)
    coupling_density = 2.0 * a2F_uniform * nu[0] / nu  # 2 a2F/w dw
    lam = coupling_density.sum()
    if lam <= 0.0:
        return 0.0
    omega_log = np.exp(np.sum(coupling_density * np.log(nu)) / lam)
    omega_2 = np.sqrt(np.sum(coupling_density * nu**2) / lam)
    return mcmillan_allen_dynes_tc(lam, omega_log, omega_2, mu_star)['Tc_allen_dynes_K']


def default_temperatures(
    omega: ArrayLike,
    a2F: ArrayLike,
    mu_star: float,
    nstemp: int = 25,
    tmax_factor: float = 1.5,
) -> NDArray[np.float64]:
    """Temperature grid bracketing ``Tc``, from the Allen-Dynes estimate.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    mu_star : float
        Coulomb pseudopotential.
    nstemp : int, optional
        Number of temperatures (default 25).
    tmax_factor : float, optional
        Highest temperature in units of the Allen-Dynes ``Tc`` (default 1.5).

    Returns
    -------
    ndarray, shape ``(nstemp,)``
        Evenly spaced temperatures (K) from ``Tmax / nstemp`` to ``Tmax``.

    Raises
    ------
    ValueError
        If the Allen-Dynes ``Tc`` is zero.

    Notes
    -----
    The Migdal-Eliashberg ``Tc`` is usually within 20% of Allen-Dynes, so the
    default range brackets it.
    """
    tc_allen_dynes = allen_dynes_tc(omega, a2F, mu_star)
    if tc_allen_dynes <= 0.0:
        raise ValueError('Allen-Dynes Tc is zero; give the temperatures explicitly.')
    tmax = tmax_factor * tc_allen_dynes
    return np.linspace(tmax / nstemp, tmax, nstemp)


def default_wscut(
    omega: ArrayLike, a2F: ArrayLike, factor: float = 5.0, minimum: float = 0.1
) -> float:
    """Matsubara cutoff ``wscut`` from the phonon spectrum.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    factor : float, optional
        Cutoff in units of the highest phonon frequency (default 5).
    minimum : float, optional
        Smallest cutoff returned (eV, default 0.1).

    Returns
    -------
    float
        ``max(minimum, factor * omega_ph)`` rounded to 0.01 eV, and above the
        top of the ``a2F`` grid (required by :func:`analytic_continuation_iso`).

    Notes
    -----
    ``omega_ph`` is the highest frequency with ``a2F > 10^-3 max(a2F)``.  The
    rule reproduces the cutoffs of EPW tutorial 04: 0.1 eV for Pb
    (``omega_ph`` = 9 meV) and 0.5 eV for MgB2 (``omega_ph`` = 100 meV).
    """
    nu, a2F_uniform = _uniform_a2f(omega, a2F)
    significant = np.nonzero(a2F_uniform > 1.0e-3 * a2F_uniform.max())[0]
    omega_ph = nu[significant[-1]] if significant.size else nu[-1]
    wscut = max(minimum, round(factor * float(omega_ph), 2))
    return max(wscut, round(1.2 * float(nu[-1]), 2))


def _gap_guess(omega: ArrayLike, a2F: ArrayLike, mu_star: float) -> float:
    """BCS gap ``1.764 k_B Tc`` from the Allen-Dynes ``Tc`` (EPW ``gap0``), in eV."""
    tc_allen_dynes = allen_dynes_tc(omega, a2F, mu_star)
    return 1.764 * KB_EV * tc_allen_dynes if tc_allen_dynes > 0.0 else 1.0e-3


# --------------------------------------------------------------------------- #
# Imaginary axis                                                              #
# --------------------------------------------------------------------------- #
def matsubara_grid(T: float, wscut: float) -> NDArray[np.float64]:
    """Positive fermionic Matsubara frequencies up to the cutoff.

    Parameters
    ----------
    T : float
        Temperature (K).
    wscut : float
        Cutoff (eV).

    Returns
    -------
    ndarray
        ``w_n = (2n+1) pi k_B T`` (eV) for ``n = 0 .. nsiw-1``, with EPW's count
        ``nsiw = int((wscut / (pi k_B T) - 1) / 2) + 1``.
    """
    pi_kt = np.pi * KB_EV * T
    nsiw = int(0.5 * (wscut / pi_kt - 1.0)) + 1
    return (2 * np.arange(nsiw) + 1) * pi_kt


def matsubara_lambda(omega: ArrayLike, a2F: ArrayLike, T: float, nmax: int) -> NDArray[np.float64]:
    """Coupling on the bosonic Matsubara frequencies.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    T : float
        Temperature (K).
    nmax : int
        Highest index.

    Returns
    -------
    ndarray, shape ``(nmax + 1,)``
        ``lambda(n)`` for ``n = 0 .. nmax``; ``lambda(0)`` is the total coupling.

    Notes
    -----
    .. math::

        \\lambda(n) = \\int_0^\\infty \\frac{2\\omega\\,\\alpha^2F(\\omega)}
            {\\omega^2 + \\nu_n^2}\\,d\\omega,
        \\qquad \\nu_n = 2\\pi n k_B T.
    """
    nu, a2F_uniform = _uniform_a2f(omega, a2F)
    bosonic = 2.0 * np.pi * KB_EV * T * np.arange(nmax + 1)
    return (1.0 / (nu[None, :] ** 2 + bosonic[:, None] ** 2)) @ (2.0 * nu * a2F_uniform * nu[0])


def _kernels(
    omega: ArrayLike, a2F: ArrayLike, T: float, wscut: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Matsubara grid and the kernels ``lambda(n-n') + lambda(n+n'+1)`` and ``... -``.

    Returns
    -------
    wn : ndarray, shape ``(nsiw,)``
        Positive Matsubara frequencies (eV).
    kernel_plus, kernel_minus : ndarray, shape ``(nsiw, nsiw)``
        ``lambda(|n-n'|) + lambda(n+n'+1)`` (gap equation) and
        ``lambda(|n-n'|) - lambda(n+n'+1)`` (renormalisation).
    """
    wn = matsubara_grid(T, wscut)
    nsiw = wn.size
    lam = matsubara_lambda(omega, a2F, T, 2 * nsiw)
    index = np.arange(nsiw)
    difference = np.abs(index[:, None] - index[None, :])
    total = index[:, None] + index[None, :] + 1
    return wn, lam[difference] + lam[total], lam[difference] - lam[total]


def _max_eigenvalue(
    wn: NDArray[np.float64],
    kernel_plus: NDArray[np.float64],
    kernel_minus: NDArray[np.float64],
    T: float,
    mu_star: float,
) -> float:
    """Largest eigenvalue of the symmetrised linearised kernel (normal-state ``Z``)."""
    pi_t = np.pi * KB_EV * T
    Z_normal = 1.0 + pi_t / wn * kernel_minus.sum(axis=1)
    sqrt_d = np.sqrt(pi_t / (Z_normal * wn))
    symmetric = sqrt_d[:, None] * (kernel_plus - 2.0 * mu_star) * sqrt_d[None, :]
    return float(np.linalg.eigvalsh(symmetric)[-1])


def linearized_max_eigenvalue(
    omega: ArrayLike, a2F: ArrayLike, T: float, mu_star: float, wscut: float = 0.1
) -> float:
    """Largest eigenvalue of the linearised isotropic Migdal-Eliashberg kernel.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    T : float
        Temperature (K).
    mu_star : float
        Coulomb pseudopotential (EPW ``muc``).
    wscut : float, optional
        Matsubara cutoff in eV (default 0.1).

    Returns
    -------
    float
        The largest eigenvalue ``rho``: above 1 below ``Tc``, 1 at ``Tc``
        (EPW ``tc_linear``).

    Notes
    -----
    Near ``Tc`` the gap equation linearises to

    .. math::

        \\rho\\,\\Delta_n = \\sum_{n'} \\frac{\\pi T}{Z_n}
            \\left[\\lambda(n-n') + \\lambda(n+n'+1) - 2\\mu^*\\right]
            \\frac{\\Delta_{n'}}{\\omega_{n'}},

    with the normal-state ``Z_n``.  The kernel is similar to the symmetric
    matrix :math:`D^{1/2}[\\ldots]D^{1/2}`, :math:`D_n = \\pi T/(Z_n\\omega_n)`,
    whose eigenvalues are real (``numpy.linalg.eigvalsh``).
    """
    wn, kernel_plus, kernel_minus = _kernels(omega, a2F, T, wscut)
    return _max_eigenvalue(wn, kernel_plus, kernel_minus, T, mu_star)


def solve_imag_iso(
    omega: ArrayLike,
    a2F: ArrayLike,
    T: float,
    mu_star: float,
    wscut: float = 0.1,
    nsiter: int = 500,
    conv_thr: float = 1.0e-4,
    delta0: float | ArrayLike | None = None,
    mix: float = 0.7,
    nhist: int = 8,
) -> dict[str, Any]:
    """Self-consistent isotropic gap ``Delta(i w_n)`` and renormalisation ``Z(i w_n)``.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    T : float
        Temperature (K).
    mu_star : float
        Coulomb pseudopotential (EPW ``muc``).
    wscut : float, optional
        Matsubara cutoff in eV (EPW ``wscut``, default 0.1).
    nsiter : int, optional
        Maximum number of iterations (EPW ``nsiter``).
    conv_thr : float, optional
        Convergence threshold (EPW ``conv_thr_iaxis``).
    delta0 : float or array_like, optional
        Initial gap (eV), scalar or one value per Matsubara frequency.  Defaults
        to ``1.764 k_B Tc`` with the Allen-Dynes ``Tc`` of ``a2F``.
    mix : float, optional
        Anderson mixing parameter (default 0.7, EPW's ``broyden_beta``).
    nhist : int, optional
        Number of previous steps kept by the Anderson mixing.

    Returns
    -------
    dict
        ``wn``, ``Z``, ``delta`` (eV, ``(nsiw,)``), ``niter``, ``converged``
        and ``rho_max`` (largest linearised-kernel eigenvalue).  ``delta`` is
        positive at the lowest frequency.

    Notes
    -----
    The equations are those of the module docstring.  The fixed point
    ``Delta = G(Delta)`` is found with Anderson mixing and stops when
    ``sum|G(Delta) - Delta| / sum|G(Delta)| < conv_thr``, EPW's criterion.  When
    the linearised kernel has no eigenvalue above 1 the only solution is the
    normal state, which is returned directly with ``Z`` from the same cut-off
    sums.
    """
    wn, kernel_plus, kernel_minus = _kernels(omega, a2F, T, wscut)
    pi_t = np.pi * KB_EV * T
    rho_max = _max_eigenvalue(wn, kernel_plus, kernel_minus, T, mu_star)
    if rho_max < 1.0:
        Z_normal = 1.0 + pi_t / wn * kernel_minus.sum(axis=1)
        return {
            'wn': wn,
            'Z': Z_normal,
            'delta': np.zeros_like(wn),
            'niter': 0,
            'converged': True,
            'rho_max': rho_max,
        }

    pairing_kernel = kernel_plus - 2.0 * mu_star

    def gap_map(gap: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One evaluation of the right-hand sides: ``(G(Delta), Z)``."""
        root = np.sqrt(wn**2 + gap**2)
        Z = 1.0 + pi_t / wn * (kernel_minus @ (wn / root))
        return pi_t / Z * (pairing_kernel @ (gap / root)), Z

    if delta0 is None:
        delta0 = _gap_guess(omega, a2F, mu_star)
    gap_trial = np.broadcast_to(np.asarray(delta0, dtype=float), wn.shape).copy()
    gap_mapped, Z, iteration, converged = anderson_fixed_point(
        gap_map, gap_trial, nsiter, conv_thr, mix, nhist
    )
    delta = gap_mapped if gap_mapped[0] >= 0.0 else -gap_mapped
    return {
        'wn': wn,
        'Z': Z,
        'delta': delta,
        'niter': iteration,
        'converged': converged,
        'rho_max': rho_max,
    }


def anderson_fixed_point(
    gap_map: Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], Any]],
    gap_trial: NDArray[np.float64],
    nsiter: int,
    conv_thr: float,
    mix: float,
    nhist: int,
    normal_floor: float | None = None,
) -> tuple[NDArray[np.float64], Any, int, bool]:
    r"""Anderson-mixed fixed point ``Delta = G(Delta)`` of the gap equation.

    Parameters
    ----------
    gap_map : callable
        ``Delta -> (G(Delta), extra)``; ``extra`` (e.g. ``Z``) is returned with
        the last evaluation.
    gap_trial : ndarray
        Initial gap, of any shape.
    nsiter : int
        Maximum number of iterations.
    conv_thr : float
        Threshold on ``sum|G(Delta) - Delta| / sum|G(Delta)|`` (EPW's criterion).
    mix : float
        Mixing parameter.
    nhist : int
        Number of previous steps kept.
    normal_floor : float, optional
        Stop, as converged to the normal state, when ``max|G(Delta)|`` drops
        below this value.

    Returns
    -------
    gap_mapped : ndarray
        The last ``G(Delta)``.
    extra : Any
        The ``extra`` of the last evaluation.
    niter : int
        Iterations used.
    converged : bool
        Whether the threshold (or ``normal_floor``) was reached.

    Notes
    -----
    With the residual :math:`F_i = G(\Delta_i) - \Delta_i` and the differences
    :math:`\delta\Delta_i`, :math:`\delta F_i` of the last ``nhist`` steps, the
    update is

    .. math::

        \Delta_{i+1} = \Delta_i + \beta F_i
            - \sum_j \gamma_j\,(\delta\Delta_j + \beta\,\delta F_j),
        \qquad \gamma = \arg\min\,\|F_i - \textstyle\sum_j \gamma_j \delta F_j\|.
    """
    shape = gap_trial.shape
    trial_steps: list[NDArray[np.float64]] = []
    residual_steps: list[NDArray[np.float64]] = []
    previous_trial = previous_residual = None
    converged = False
    for iteration in range(1, nsiter + 1):
        gap_mapped, extra = gap_map(gap_trial)
        residual = gap_mapped - gap_trial
        norm = np.abs(gap_mapped).sum()
        if norm > 0.0 and np.abs(residual).sum() / norm < conv_thr:
            converged = True
            break
        if normal_floor is not None and np.abs(gap_mapped).max() < normal_floor:
            converged = True
            break
        if previous_trial is not None:
            trial_steps.append((gap_trial - previous_trial).ravel())
            residual_steps.append((residual - previous_residual).ravel())
            del trial_steps[:-nhist], residual_steps[:-nhist]
        previous_trial, previous_residual = gap_trial, residual
        if residual_steps:
            residual_matrix = np.stack(residual_steps, axis=1)
            coefficients = np.linalg.lstsq(residual_matrix, residual.ravel(), rcond=None)[0]
            correction = (np.stack(trial_steps, axis=1) + mix * residual_matrix) @ coefficients
            gap_trial = gap_trial + mix * residual - correction.reshape(shape)
        else:
            gap_trial = gap_trial + mix * residual
    return gap_mapped, extra, iteration, converged


# --------------------------------------------------------------------------- #
# Real axis: Pade approximants                                                #
# --------------------------------------------------------------------------- #
def pade_coefficients(z: ArrayLike, u: ArrayLike) -> NDArray[np.complex128]:
    """Coefficients of the Vidberg-Serene N-point Pade continued fraction.

    Parameters
    ----------
    z : array_like, shape ``(N,)``
        Sampling points (e.g. ``i w_n``).
    u : array_like, shape ``(..., N)``
        Function values at ``z``; leading axes are independent functions.

    Returns
    -------
    ndarray, shape ``(..., N)``, complex
        Coefficients ``a_p`` of :func:`pade_eval`.

    Notes
    -----
    With :math:`g_1(z_i) = u_i`, the recursion is

    .. math::

        g_p(z) = \\frac{g_{p-1}(z_{p-1}) - g_{p-1}(z)}{(z - z_{p-1})\\,g_{p-1}(z)},
        \\qquad a_p = g_p(z_p)

    (H. J. Vidberg and J. W. Serene, J. Low Temp. Phys. **29**, 179 (1977)).
    """
    z = np.asarray(z, dtype=complex)
    g = np.asarray(u, dtype=complex).copy()
    coefficients = np.empty(g.shape, dtype=complex)
    coefficients[..., 0] = g[..., 0]
    for p in range(1, z.size):
        g[..., p:] = (coefficients[..., p - 1, None] - g[..., p:]) / (
            (z[p:] - z[p - 1]) * g[..., p:]
        )
        coefficients[..., p] = g[..., p]
    return coefficients


def pade_eval(
    coefficients: NDArray[np.complex128], z: ArrayLike, w: ArrayLike
) -> NDArray[np.complex128]:
    """Evaluate the continued fraction of :func:`pade_coefficients` at ``w``.

    Parameters
    ----------
    coefficients : ndarray, shape ``(..., N)``
        Output of :func:`pade_coefficients`.
    z : array_like, shape ``(N,)``
        The sampling points used for ``coefficients``.
    w : array_like, shape ``(nw,)``
        Evaluation points.

    Returns
    -------
    ndarray, shape ``(..., nw)``, complex
        The approximant at ``w``.

    Notes
    -----
    :math:`C_N(w) = A_N(w)/B_N(w)` with
    :math:`A_{n+1} = A_n + (w - z_n)\\,a_{n+1}A_{n-1}` (likewise for ``B``),
    :math:`A_0 = 0`, :math:`A_1 = a_1`, :math:`B_0 = B_1 = 1`.  Each step is
    rescaled by :math:`B_{n+1}`, which leaves the ratio unchanged and avoids
    overflow.
    """
    coefficients = np.asarray(coefficients)
    shape = coefficients.shape[:-1] + np.shape(w)
    w = np.asarray(w, dtype=complex)
    A_prev = np.zeros(shape, dtype=complex)
    A_curr = np.broadcast_to(coefficients[..., 0, None], shape).astype(complex)
    B_prev, B_curr = np.ones(shape, dtype=complex), np.ones(shape, dtype=complex)
    for n in range(1, coefficients.shape[-1]):
        A_next = A_curr + (w - z[n - 1]) * coefficients[..., n, None] * A_prev
        B_next = B_curr + (w - z[n - 1]) * coefficients[..., n, None] * B_prev
        A_prev, A_curr = A_curr / B_next, A_next / B_next
        B_prev, B_curr = B_curr / B_next, np.ones(shape, dtype=complex)
    return A_curr / B_curr


def pade_continuation(
    wn: ArrayLike,
    Z: ArrayLike,
    delta: ArrayLike,
    w_real: ArrayLike,
    npade: float = 90,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Real-axis ``Z(w)``, ``Delta(w)`` from Pade approximants of the Matsubara data.

    Parameters
    ----------
    wn, Z, delta : array_like, shape ``(nsiw,)``
        Imaginary-axis solution (:func:`solve_imag_iso`).
    w_real : array_like
        Real frequencies (eV).
    npade : float, optional
        Percentage of the Matsubara frequencies used (EPW ``npade``, default 90).

    Returns
    -------
    Z_real, delta_real : ndarray, complex
        ``Z(w)`` and ``Delta(w)`` on ``w_real``; ``Delta = 0`` for a vanishing gap.
    """
    wn = np.asarray(wn, dtype=float)
    Z = np.asarray(Z)
    delta = np.asarray(delta)
    npoints = max(2, int(npade * len(wn) / 100))
    z = 1j * wn[:npoints]
    Z_real = pade_eval(pade_coefficients(z, Z[:npoints]), z, w_real)
    if not np.any(delta[:npoints]):
        return Z_real, np.zeros_like(Z_real)
    return Z_real, pade_eval(pade_coefficients(z, delta[:npoints]), z, w_real)


# --------------------------------------------------------------------------- #
# Real axis: iterative analytic continuation                                  #
# --------------------------------------------------------------------------- #
def _retarded_sqrt(radicand: ArrayLike, w: ArrayLike) -> NDArray[np.complex128]:
    """``sqrt(radicand)`` on the retarded branch, ``S(-w) = -S(w)*``.

    Notes
    -----
    For ``w > 0`` both ``Re S >= 0`` and ``Im S >= 0`` (EPW conjugates a root
    with ``Im < 0``), which is continuous across the gap edge: negating instead
    would flip the sign of ``S`` above the gap whenever rounding gives the
    radicand a slightly negative imaginary part.
    """
    root = np.sqrt(np.asarray(radicand, dtype=complex))
    root = np.where(root.imag < 0.0, root.conj(), root)
    return np.where(np.asarray(w) < 0.0, -root.conj(), root)


def _lambda_real_minus_imag(
    nu: NDArray[np.float64],
    a2F_weights: NDArray[np.float64],
    w_real: NDArray[np.float64],
    wn: ArrayLike,
) -> NDArray[np.complex128]:
    """``lambda(w_i - i w_m)`` for every real ``w_i`` and Matsubara ``w_m``.

    Parameters
    ----------
    nu : ndarray, shape ``(nnu,)``
        Uniform phonon grid ``j * dw``.
    a2F_weights : ndarray, shape ``(nnu,)``
        ``a2F(nu_j) dw``.
    w_real : ndarray, shape ``(nw,)``
        Real grid ``i * dw`` with the same spacing.
    wn : array_like, shape ``(nm,)``
        Matsubara frequencies.

    Returns
    -------
    ndarray, shape ``(nw, nm)``, complex

    Notes
    -----
    .. math::

        \\lambda(z) = \\sum_j a_j\\left[\\frac{1}{\\nu_j - z} + \\frac{1}{\\nu_j + z}\\right],
        \\qquad z = \\omega_i - i\\omega_m .

    Because ``nu`` and ``w_real`` share the spacing, both terms are discrete
    convolutions over ``i - j`` and ``i + j``.  They are evaluated with FFTs for
    all Matsubara frequencies at once.
    """
    spacing = nu[0]
    nnu, nw = nu.size, w_real.size
    matsubara = np.asarray(wn, dtype=float)[:, None]
    shift_minus = np.arange(-nnu, nw + 1) * spacing  # w_i - nu_j
    shift_plus = np.arange(1, nw + nnu + 1) * spacing  # w_i + nu_j
    term_minus = fftconvolve(
        a2F_weights[None, :], 1.0 / (-shift_minus[None, :] + 1j * matsubara), axes=1
    )
    term_plus = fftconvolve(
        a2F_weights[None, ::-1], 1.0 / (shift_plus[None, :] - 1j * matsubara), axes=1
    )
    window = slice(nnu, nnu + nw)
    return (term_minus[:, window] + term_plus[:, window]).T


def analytic_continuation_iso(
    omega: ArrayLike,
    a2F: ArrayLike,
    T: float,
    mu_star: float,
    wn: NDArray[np.float64],
    Zn: NDArray[np.float64],
    deltan: NDArray[np.float64],
    w_real: ArrayLike,
    Z_init: ArrayLike,
    delta_init: ArrayLike,
    nsiter: int = 500,
    conv_thr: float = 1.0e-4,
    mix: float = 0.5,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int, bool]:
    """Real-axis ``Z(w)``, ``Delta(w)`` by the iterative analytic continuation.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    T : float
        Temperature (K).
    mu_star : float
        Coulomb pseudopotential.
    wn, Zn, deltan : ndarray, shape ``(nsiw,)``
        Converged imaginary-axis solution (:func:`solve_imag_iso`).
    w_real : array_like, shape ``(nw,)``
        Real grid from :func:`real_axis_grid`.
    Z_init, delta_init : array_like, shape ``(nw,)``
        Starting point, normally the Pade result.
    nsiter : int, optional
        Maximum number of iterations.
    conv_thr : float, optional
        Threshold on ``sum|Delta_new - Delta| / sum|Delta_new|`` (EPW
        ``conv_thr_racon``).
    mix : float, optional
        Linear mixing parameter.

    Returns
    -------
    Z, delta : ndarray, shape ``(nw,)``, complex
        ``Z(w)`` and ``Delta(w)``.
    niter : int
        Iterations used.
    converged : bool
        Whether ``conv_thr`` was reached.

    Raises
    ------
    ValueError
        If ``w_real`` is not the grid of :func:`real_axis_grid`, or does not
        extend beyond the highest ``a2F`` frequency.

    Notes
    -----
    Solves the Marsiglio-Schossmann-Carbotte equations (Phys. Rev. B **37**,
    4965 (1988)) for :math:`\\tilde\\omega = \\omega Z` and
    :math:`\\phi = Z\\Delta`:

    .. math::

        \\tilde\\omega(\\omega) = \\omega + i\\pi T\\sum_m \\lambda(\\omega - i\\omega_m)
            \\frac{\\omega_m}{R_m}
          + i\\pi\\int_0^\\infty d\\nu\\,\\alpha^2F(\\nu)
            \\left\\{[N(\\nu)+f(\\nu-\\omega)]\\,\\frac{\\tilde\\omega}{S}(\\omega-\\nu)
                 -[N(\\nu)+f(\\nu+\\omega)]\\,\\frac{\\tilde\\omega}{S}(\\omega+\\nu)\\right\\},

        \\phi(\\omega) = \\pi T\\sum_m [\\lambda(\\omega - i\\omega_m) - \\mu^*]
            \\frac{\\Delta_m}{R_m}
          + i\\pi\\int_0^\\infty d\\nu\\,\\alpha^2F(\\nu)
            \\left\\{[N(\\nu)+f(\\nu-\\omega)]\\,\\frac{\\phi}{S}(\\omega-\\nu)
                 +[N(\\nu)+f(\\nu+\\omega)]\\,\\frac{\\phi}{S}(\\omega+\\nu)\\right\\},

    with the Matsubara sums over all ``m`` (fixed by the imaginary-axis
    solution), :math:`S = \\sqrt{\\tilde\\omega^2 - \\phi^2}` on the retarded
    branch, Bose ``N`` and Fermi ``f``.  Negative frequencies follow from
    :math:`Z(-\\omega) = Z(\\omega)^*`, :math:`\\Delta(-\\omega) = \\Delta(\\omega)^*`;
    above the grid the normal state is assumed.  Both convolutions are
    evaluated with ``numpy.convolve`` on the common grid spacing.
    """
    nu, a2F_uniform = _uniform_a2f(omega, a2F)
    spacing = nu[0]
    w = np.asarray(w_real, dtype=float)
    nnu, nw = nu.size, w.size
    if not np.allclose(w, spacing * np.arange(1, nw + 1), rtol=0.0, atol=1.0e-6 * spacing):
        raise ValueError('w_real must be the grid of real_axis_grid (spacing of a2F).')
    if nw < nnu:
        raise ValueError('wscut must exceed the highest a2F frequency.')
    kT = KB_EV * T
    a2F_weights = a2F_uniform * spacing
    bose_weights = a2F_weights / np.expm1(nu / kT)  # a2F(nu_j) dw N(nu_j)

    # Matsubara terms, fixed during the iteration.
    lam = _lambda_real_minus_imag(nu, a2F_weights, w, wn)
    root_n = np.sqrt(wn**2 + deltan**2)
    matsubara_wt = 1j * np.pi * kT * ((lam - lam.conj()) @ (wn / root_n))
    matsubara_phi = np.pi * kT * ((lam + lam.conj() - 2.0 * mu_star) @ (deltan / root_n))

    shift_minus = np.arange(-nnu, nw + 1)  # w_i - nu_j in units of dw
    shift_plus = np.arange(1, nw + nnu + 1)  # w_i + nu_j in units of dw
    fermi_minus = expit(shift_minus * spacing / kT)  # f(nu_j - w_i)
    fermi_plus = expit(-shift_plus * spacing / kT)  # f(nu_j + w_i)
    window = slice(nnu, nnu + nw)

    def extend(values: NDArray[np.complex128], odd_real: bool) -> NDArray[np.complex128]:
        """``values`` on the shifts ``-nnu .. nw + nnu`` (index ``shift + nnu``)."""
        full = np.zeros(nw + 2 * nnu + 1, dtype=complex)
        full[nnu + 1 : nnu + 1 + nw] = values
        mirror = values[:nnu][::-1].conj()
        full[:nnu] = -mirror if odd_real else mirror
        if odd_real:  # phi/S: purely imaginary at w = 0
            full[nnu] = 1j * values[0].imag
        else:  # w~/S: 0 at w = 0, 1 above the grid
            full[nnu + 1 + nw :] = 1.0
        return full

    def convolutions(
        full: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """The ``w - nu`` and ``w + nu`` integrals of the real-axis kernel."""
        at_minus, at_plus = full[: nnu + nw + 1], full[nnu + 1 :]
        minus = np.convolve(bose_weights, at_minus) + np.convolve(
            a2F_weights, fermi_minus * at_minus
        )
        plus = np.convolve(bose_weights[::-1], at_plus) + np.convolve(
            a2F_weights[::-1], fermi_plus * at_plus
        )
        return minus[window], plus[window]

    Z = np.asarray(Z_init, dtype=complex).copy()
    delta = np.asarray(delta_init, dtype=complex).copy()
    converged = False
    for iteration in range(1, nsiter + 1):
        wt, phi = w * Z, Z * delta
        root = _retarded_sqrt(wt**2 - phi**2, w)
        wt_minus, wt_plus = convolutions(extend(wt / root, False))
        phi_minus, phi_plus = convolutions(extend(phi / root, True))
        Z_new = (w + matsubara_wt + 1j * np.pi * (wt_minus - wt_plus)) / w
        delta_new = (matsubara_phi + 1j * np.pi * (phi_minus + phi_plus)) / Z_new
        error = np.abs(delta_new - delta).sum() / max(np.abs(delta_new).sum(), 1.0e-300)
        Z += mix * (Z_new - Z)
        delta += mix * (delta_new - delta)
        if error < conv_thr:
            converged = True
            break
    return Z, delta, iteration, converged


# --------------------------------------------------------------------------- #
# Real-axis observables                                                       #
# --------------------------------------------------------------------------- #
def gap_edge(w_real: ArrayLike, delta: ArrayLike) -> float:
    """Superconducting gap edge, the first solution of ``w = Re Delta(w)``.

    Parameters
    ----------
    w_real : array_like
        Real frequencies (eV), increasing.
    delta : array_like
        ``Delta(w)`` on ``w_real``.

    Returns
    -------
    float
        The gap edge (eV), linearly interpolated; 0 for a vanishing gap and
        ``nan`` when there is no crossing on the grid.
    """
    w_real = np.asarray(w_real, dtype=float)
    re_delta = np.real(delta)
    if not np.any(np.abs(re_delta) > 1.0e-12):
        return 0.0
    distance = w_real - re_delta
    above = np.nonzero(distance >= 0.0)[0]
    if above.size == 0:
        return float('nan')
    i = above[0]
    if i == 0:
        return float(w_real[0])
    w0, w1, d0, d1 = w_real[i - 1], w_real[i], distance[i - 1], distance[i]
    return float(w0 - d0 * (w1 - w0) / (d1 - d0))


def quasiparticle_dos(
    w_real: ArrayLike, delta: ArrayLike, Z: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Superconducting quasiparticle density of states ``N_S(w)/N_F``.

    Parameters
    ----------
    w_real : array_like
        Real frequencies (eV).
    delta : array_like
        ``Delta(w)`` on ``w_real``.
    Z : array_like, optional
        ``Z(w)``; only its phase matters for the branch of the square root.

    Returns
    -------
    ndarray
        ``N_S/N_F`` on ``w_real``.

    Notes
    -----
    .. math::

        \\frac{N_S(\\omega)}{N_F} = \\mathrm{Re}\\left[
            \\frac{\\omega Z}{\\sqrt{Z^2(\\omega^2 - \\Delta^2(\\omega))}}\\right],

    with the retarded square root; ``w = Delta`` exactly is singular.
    """
    w = np.asarray(w_real, dtype=float)
    Z = np.ones_like(w) if Z is None else np.asarray(Z)
    wt, phi = w * Z, Z * np.asarray(delta)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.real(wt / _retarded_sqrt(wt**2 - phi**2, w))


def tc_from_eigenvalues(temps: ArrayLike, rho: ArrayLike) -> float:
    """Temperature where the linearised-kernel eigenvalue crosses 1.

    Parameters
    ----------
    temps : array_like
        Increasing temperatures (K).
    rho : array_like
        Largest eigenvalues at ``temps``.

    Returns
    -------
    float
        ``Tc`` (K) by linear interpolation of the first downward crossing, or
        ``nan`` without one.
    """
    temps, rho = np.asarray(temps, dtype=float), np.asarray(rho, dtype=float)
    for i in range(1, temps.size):
        if rho[i - 1] >= 1.0 > rho[i]:
            return float(
                temps[i - 1]
                + (1.0 - rho[i - 1]) * (temps[i] - temps[i - 1]) / (rho[i] - rho[i - 1])
            )
    return float('nan')


def tc_from_gap(temps: ArrayLike, gap: ArrayLike) -> float:
    """``Tc`` from a linear extrapolation of ``Delta^2(T)`` to zero.

    Parameters
    ----------
    temps : array_like
        Increasing temperatures (K).
    gap : array_like
        Gap at ``temps`` (any unit).

    Returns
    -------
    float
        ``Tc`` (K) through the two highest-temperature non-zero gaps, or ``nan``
        when there are fewer than two or ``Delta^2`` does not decrease.

    Notes
    -----
    Near a second-order transition :math:`\\Delta^2 \\propto T_c - T`.
    """
    temps, gap = np.asarray(temps, dtype=float), np.asarray(gap, dtype=float)
    nonzero = np.nonzero(gap > 0.0)[0]
    if nonzero.size < 2:
        return float('nan')
    i, j = nonzero[-2], nonzero[-1]
    gap2_low, gap2_high = gap[i] ** 2, gap[j] ** 2
    if gap2_high >= gap2_low:
        return float('nan')
    return float(temps[i] + gap2_low * (temps[j] - temps[i]) / (gap2_low - gap2_high))


# --------------------------------------------------------------------------- #
# Temperature sweeps                                                          #
# --------------------------------------------------------------------------- #
def migdal_eliashberg_iso(
    omega: ArrayLike,
    a2F: ArrayLike,
    temps: ArrayLike,
    mu_star: float = 0.1,
    wscut: float = 0.1,
    nsiter: int = 500,
    conv_thr_iaxis: float = 1.0e-4,
    conv_thr_racon: float = 1.0e-4,
    npade: float = 90,
    lpade: bool = True,
    lacon: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Isotropic Migdal-Eliashberg solution over a set of temperatures.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    temps : array_like
        Temperatures (K); solved in ascending order.
    mu_star : float, optional
        Coulomb pseudopotential (EPW ``muc``, default 0.1).
    wscut : float, optional
        Matsubara cutoff and upper real frequency (eV, default 0.1).
    nsiter : int, optional
        Maximum iterations on each axis (EPW ``nsiter``).
    conv_thr_iaxis, conv_thr_racon : float, optional
        Convergence thresholds on the imaginary and real axes.
    npade : float, optional
        Percentage of Matsubara points in the Pade approximant.
    lpade, lacon : bool, optional
        Compute the Pade and the analytic continuation (EPW ``lpade``, ``lacon``).
    verbose : bool, optional
        Print one summary line per temperature.

    Returns
    -------
    dict
        ``temps`` (K); per-temperature arrays ``gap0_imag`` (``Delta(i w_0)``),
        ``gap_pade``, ``gap_acon`` (gap edges), ``Z0`` (``Z(i w_0)``),
        ``rho_max``, ``niter``, ``converged`` (energies in eV, ``nan`` where not
        computed); ``Tc_gap`` (K, :func:`tc_from_gap` on ``gap0_imag``);
        ``w_real``; ``mu_star``, ``wscut``; and the per-temperature lists
        ``imag`` (``(wn, Z, delta)``), ``pade``, ``acon`` (``(Z, delta)`` on
        ``w_real`` or ``None``) and ``qdos`` (from ``acon``, else ``pade``, or
        ``None``).

    Notes
    -----
    Each temperature starts from the gap of the previous one, interpolated on
    its Matsubara grid.  Temperatures with a non-zero gap are continued to the
    real axis with Pade and, when ``lacon`` is set, refined by the analytic
    continuation, which always starts from the Pade result.
    """
    temps = np.sort(np.asarray(temps, dtype=float))
    w_real = real_axis_grid(omega, a2F, wscut)
    ntemps = temps.size
    out: dict[str, Any] = {
        key: np.full(ntemps, np.nan)
        for key in ('gap0_imag', 'gap_pade', 'gap_acon', 'Z0', 'rho_max')
    }
    out.update(
        niter=np.zeros(ntemps, dtype=int),
        converged=np.zeros(ntemps, dtype=bool),
        imag=[],
        pade=[],
        acon=[],
        qdos=[],
    )
    previous_gap = None
    for it, T in enumerate(temps):
        delta0 = None
        if previous_gap is not None:
            delta0 = np.interp(matsubara_grid(T, wscut), *previous_gap)
        solution = solve_imag_iso(omega, a2F, T, mu_star, wscut, nsiter, conv_thr_iaxis, delta0)
        wn, Zn, deltan = solution['wn'], solution['Z'], solution['delta']
        out['gap0_imag'][it], out['Z0'][it] = deltan[0], Zn[0]
        out['rho_max'][it] = solution['rho_max']
        out['niter'][it], out['converged'][it] = solution['niter'], solution['converged']
        out['imag'].append((wn, Zn, deltan))
        gapped = bool(np.any(deltan))
        if gapped:
            previous_gap = (wn, deltan)
        pade = acon = None
        if gapped and (lpade or lacon):
            pade = pade_continuation(wn, Zn, deltan, w_real, npade)
            if lacon:
                Z_acon, delta_acon, _, _ = analytic_continuation_iso(
                    omega, a2F, T, mu_star, wn, Zn, deltan, w_real, *pade,
                    nsiter=nsiter, conv_thr=conv_thr_racon,
                )  # fmt: skip
                acon = (Z_acon, delta_acon)
                out['gap_acon'][it] = gap_edge(w_real, delta_acon)
            if lpade:
                out['gap_pade'][it] = gap_edge(w_real, pade[1])
            else:
                pade = None
        elif not gapped:
            out['gap_pade'][it] = out['gap_acon'][it] = 0.0
        out['pade'].append(pade)
        out['acon'].append(acon)
        best = acon if acon is not None else pade
        out['qdos'].append(None if best is None else quasiparticle_dos(w_real, best[1], best[0]))
        if verbose:
            print(
                '  T = %6.3f K  N_w = %4d  iter = %3d  Delta_0 = %.4f meV  Z_0 = %.4f'
                '  edge(Pade) = %.4f  edge(acon) = %.4f meV'
                % (
                    T,
                    wn.size,
                    solution['niter'],
                    deltan[0] * 1e3,
                    Zn[0],
                    out['gap_pade'][it] * 1e3,
                    out['gap_acon'][it] * 1e3,
                ),
                flush=True,
            )
    out['temps'] = temps
    out['w_real'] = w_real
    out['Tc_gap'] = tc_from_gap(temps, out['gap0_imag'])
    out['mu_star'], out['wscut'] = float(mu_star), float(wscut)
    return out


def linearized_eigenvalues(
    omega: ArrayLike,
    a2F: ArrayLike,
    temps: ArrayLike,
    mu_star: float = 0.1,
    wscut: float = 0.1,
) -> dict[str, Any]:
    """Largest linearised-kernel eigenvalue at each temperature, and ``Tc``.

    Parameters
    ----------
    omega, a2F : array_like
        Eliashberg function (frequencies in eV).
    temps : array_like
        Temperatures (K).
    mu_star : float, optional
        Coulomb pseudopotential (default 0.1).
    wscut : float, optional
        Matsubara cutoff (eV, default 0.1).

    Returns
    -------
    dict
        ``temps`` (sorted, K), ``max_eigenvalue`` and ``Tc_linear`` (K, where the
        eigenvalue crosses 1, :func:`tc_from_eigenvalues`).
    """
    temps = np.sort(np.asarray(temps, dtype=float))
    rho = np.array([linearized_max_eigenvalue(omega, a2F, T, mu_star, wscut) for T in temps])
    return {'temps': temps, 'max_eigenvalue': rho, 'Tc_linear': tc_from_eigenvalues(temps, rho)}


# --------------------------------------------------------------------------- #
# Input / output                                                              #
# --------------------------------------------------------------------------- #
def a2f_from_npz(
    path: str, degaussq_ev: float = 1.5e-4, nqstep: int = 500
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eliashberg function from an ``eliashberg.npz`` written by the PAOFLOW el-ph drivers.

    Parameters
    ----------
    path : str
        The ``eliashberg.npz`` file.
    degaussq_ev : float, optional
        Phonon smearing (eV) used to rebuild ``a2F`` from the modes.
    nqstep : int, optional
        Number of frequency points.

    Returns
    -------
    omega, a2F : ndarray
        Frequencies (eV) and Eliashberg function.

    Notes
    -----
    ``a2F`` is rebuilt from ``lambda_qv``, the mode frequencies (``omega_qv_thz``
    or ``omega_q``) and ``q_weights`` with :func:`a2f_from_modes`.  Files without
    the mode data fall back to the stored ``omega`` / ``a2F``.
    """
    data = np.load(path)
    if 'omega_qv_thz' in data:
        freqs = data['omega_qv_thz']
    elif 'omega_q' in data:
        freqs = data['omega_q']
    else:
        freqs = None
    if 'lambda_qv' in data and freqs is not None:
        weights = data['q_weights'] if 'q_weights' in data else None
        return a2f_from_modes(data['lambda_qv'], freqs, weights, degaussq_ev, nqstep)
    return np.asarray(data['omega']), np.asarray(data['a2F'])


def a2f_from_epw(path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eliashberg function from an EPW ``<prefix>.a2f`` file (first smearing).

    Parameters
    ----------
    path : str
        The EPW file (:func:`~PAOFLOW.elphon.qe_elph_io.read_epw_a2f`).

    Returns
    -------
    omega, a2F : ndarray
        Frequencies (eV) and Eliashberg function.
    """
    from .qe_elph_io import read_epw_a2f

    omega_mev, a2F, _ = read_epw_a2f(path)
    return omega_mev * 1.0e-3, a2F


def temperature_tag(T: float) -> str:
    """EPW's temperature suffix, e.g. ``000.30`` for 0.3 K."""
    return '%06.2f' % T


def write_me_outputs(
    result: dict[str, Any],
    outdir: str,
    prefix: str,
    linear: dict[str, Any] | None = None,
) -> None:
    """Write the Migdal-Eliashberg results in EPW's file formats.

    Parameters
    ----------
    result : dict
        Output of :func:`migdal_eliashberg_iso`.
    outdir : str
        Output directory (created if needed).
    prefix : str
        File prefix (EPW ``prefix``).
    linear : dict, optional
        Output of :func:`linearized_eigenvalues`.

    Returns
    -------
    None
        Writes, per temperature ``<T>`` (:func:`temperature_tag`) and with
        energies in eV:

        - ``<prefix>.imag_iso_<T>``: ``w_n``, ``Z``, ``Delta``;
        - ``<prefix>.pade_iso_<T>`` / ``<prefix>.acon_iso_<T>``: ``w``, ``Re Z``,
          ``Im Z``, ``Re Delta``, ``Im Delta``;
        - ``<prefix>.qdos_iso_<T>``: ``w``, ``N_S/N_F``;

        plus ``gap_vs_T.dat`` (gaps in meV), ``max_eigenvalue.dat`` (with
        ``linear``) and ``migdal_eliashberg.npz`` with every array.
    """
    os.makedirs(outdir, exist_ok=True)
    w = result['w_real']
    arrays = {
        key: result[key]
        for key in (
            'temps', 'gap0_imag', 'gap_pade', 'gap_acon', 'Z0', 'rho_max', 'niter', 'converged',
            'w_real',
        )
    }  # fmt: skip
    arrays.update(Tc_gap=result['Tc_gap'], mu_star=result['mu_star'], wscut=result['wscut'])
    for i, T in enumerate(result['temps']):
        tag = temperature_tag(T)
        base = os.path.join(outdir, '%s.%%s_iso_%s' % (prefix, tag))
        imag = np.column_stack(result['imag'][i])
        np.savetxt(base % 'imag', imag, header='w_n [eV]  Z(iw_n)  Delta(iw_n) [eV]   T = %g K' % T)
        arrays['imag_' + tag] = imag
        for key in ('pade', 'acon'):
            if result[key][i] is None:
                continue
            Z, delta = result[key][i]
            data = np.column_stack([w, Z.real, Z.imag, delta.real, delta.imag])
            np.savetxt(
                base % key,
                data,
                header='w [eV]  Re Z  Im Z  Re Delta [eV]  Im Delta [eV]   T = %g K' % T,
            )
            arrays['%s_%s' % (key, tag)] = data[:, 1:]
        if result['qdos'][i] is not None:
            np.savetxt(
                base % 'qdos',
                np.column_stack([w, result['qdos'][i]]),
                header='w [eV]  N_S(w)/N_F   T = %g K' % T,
            )
            arrays['qdos_' + tag] = result['qdos'][i]
    np.savetxt(
        os.path.join(outdir, 'gap_vs_T.dat'),
        np.column_stack(
            [
                result['temps'],
                result['gap0_imag'] * 1e3,
                result['gap_pade'] * 1e3,
                result['gap_acon'] * 1e3,
                result['Z0'],
            ]
        ),
        header='T [K]  Delta(iw_0) [meV]  gap edge Pade [meV]  gap edge acon [meV]  Z(iw_0)'
        '   (Tc from Delta^2 extrapolation = %.3f K)' % result['Tc_gap'],
    )
    if linear is not None:
        np.savetxt(
            os.path.join(outdir, 'max_eigenvalue.dat'),
            np.column_stack([linear['temps'], linear['max_eigenvalue']]),
            header='T [K]  max eigenvalue of the linearised ME kernel   (Tc = %.3f K)'
            % linear['Tc_linear'],
        )
        arrays.update(
            lin_temps=linear['temps'],
            max_eigenvalue=linear['max_eigenvalue'],
            Tc_linear=linear['Tc_linear'],
        )
    np.savez(os.path.join(outdir, 'migdal_eliashberg.npz'), **arrays)
