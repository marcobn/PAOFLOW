r"""Anisotropic Migdal-Eliashberg equations on the Fermi surface.

Fermi-surface-restricted (FSR) solution of the anisotropic equations, following
EPW's ``laniso`` / ``limag`` / ``lpade`` (E. R. Margine and F. Giustino, Phys.
Rev. B **87**, 024505 (2013); EPW tutorial 04, MgB2).  The input is the folded
pair coupling ``Lambda_ab(omega)`` of the irreducible Fermi-surface states
``a = (n, k)`` written by
:class:`~PAOFLOW.elphon.fermi_surface_coupling.FermiSurfacePairCoupling`, the
Coulomb pseudopotential ``mu*`` and the temperature.

With the pair coupling on the bosonic Matsubara frequencies
``nu_l = 2 pi l k_B T``,

.. math::

    \lambda_{ab}(l) = \int d\omega\,\Lambda_{ab}(\omega)\,
        \frac{\omega^2}{\omega^2 + \nu_l^2},

and the sums restricted to positive fermionic frequencies
``w_j = (2j+1) pi k_B T <= wscut``, the imaginary-axis equations are

.. math::

    Z_a(j) = 1 + \frac{\pi T}{\omega_j}\sum_{b, j'}
        [\lambda_{ab}(j-j') - \lambda_{ab}(j+j'+1)]\,\frac{\omega_{j'}}{R_b(j')},

    Z_a(j)\Delta_a(j) = \pi T\sum_{b, j'}
        [\lambda_{ab}(j-j') + \lambda_{ab}(j+j'+1) - 2\mu^* C_{ab}]\,
        \frac{\Delta_b(j')}{R_b(j')},

with :math:`R = \sqrt{\omega^2 + \Delta^2}` and :math:`C_{ab}` the Coulomb
weights of :func:`coulomb_weights`: the Fermi-surface weight
:math:`c_b = W_b / N_F` of state ``b``, restricted to the ``(k, k+q)`` pairs of
the phonon kernel.  For a single state they reduce to the
isotropic equations of :mod:`PAOFLOW.elphon.migdal_eliashberg`, with
``Lambda(omega) = 2 a2F(omega) / omega``.

The kernel ``omega^2 / (omega^2 + nu_l^2)`` sampled on the phonon grid and on
the bosonic frequencies has a low numerical rank.  Its singular-value
decomposition makes the kernel separable,
:math:`\lambda_{ab}(l) = \sum_r L_{ab,r} V_r(l)`, so that each iteration costs
``N^2 r n_w`` operations instead of ``N^2 n_w^2``.  Energies are in eV and
temperatures in kelvin.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .migdal_eliashberg import (
    KB_EV,
    _gap_guess,
    anderson_fixed_point,
    gap_edge,
    matsubara_grid,
    pade_coefficients,
    pade_eval,
    quasiparticle_dos,
    tc_from_eigenvalues,
    tc_from_gap,
    temperature_tag,
)


# --------------------------------------------------------------------------- #
# Coupling strengths                                                          #
# --------------------------------------------------------------------------- #
def coupling_strength(coupling: dict[str, Any]) -> dict[str, Any]:
    """State-resolved and isotropic coupling from the Fermi-surface pair coupling.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling
        (:func:`~PAOFLOW.elphon.fermi_surface_coupling.read_fs_coupling`).

    Returns
    -------
    dict
        ``lambda_nk`` ``(N,)`` (EPW ``lambda_nk``), ``lambda`` (isotropic
        average), ``dos_ef`` (``N_F``, states/eV/spin per cell), ``omega`` and
        ``a2F`` (isotropic Eliashberg function on the coupling frequency grid).

    Notes
    -----
    .. math::

        \\lambda_a = \\sum_b\\int d\\omega\\,\\Lambda_{ab}(\\omega),\\qquad
        \\lambda = \\sum_a c_a\\lambda_a,\\qquad
        \\alpha^2F(\\omega) = \\frac{\\omega}{2}\\sum_{ab} c_a\\Lambda_{ab}(\\omega).
    """
    pair = coupling['coupling']
    weights = np.asarray(coupling['weight'], dtype=float)
    dos_ef = float(weights.sum())
    fractions = weights / dos_ef
    omega = np.asarray(coupling['freq_ev'], dtype=float)
    spacing = omega[0]
    lambda_nk = pair.sum(axis=(1, 2))
    lambda_density = np.einsum('a,abj->j', fractions, pair)  # integrated over each step
    return {
        'lambda_nk': lambda_nk,
        'lambda': float(fractions @ lambda_nk),
        'dos_ef': dos_ef,
        'omega': omega,
        'a2F': 0.5 * omega * lambda_density / spacing,
    }


# --------------------------------------------------------------------------- #
# Imaginary axis                                                              #
# --------------------------------------------------------------------------- #
def matsubara_kernel(
    coupling: dict[str, Any], T: float, wscut: float, rank_tol: float = 1.0e-10
) -> dict[str, Any]:
    """Separable anisotropic kernel on the positive Matsubara frequencies.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling.
    T : float
        Temperature (K).
    wscut : float
        Matsubara cutoff (eV).
    rank_tol : float, optional
        Relative singular-value cutoff of the frequency kernel (default 1e-10).

    Returns
    -------
    dict
        ``wn`` ``(nw,)``; ``pair_factors`` ``(N, N * r)``, the factors
        :math:`L_{ab,r}` flattened over ``(b, r)``; ``plus`` and ``minus``
        ``(r, nw, nw)``, :math:`V_r(|j-j'|) \\pm V_r(j+j'+1)`; ``fractions``
        ``(N,)``, the weights ``c_b``; ``coulomb`` ``(N, N)``, the Coulomb
        weights :math:`C_{ab}` (:func:`coulomb_weights`); and ``rank``.

    Notes
    -----
    With :math:`h_{\\beta l} = \\omega_\\beta^2/(\\omega_\\beta^2 + \\nu_l^2)` on
    the coupling grid and :math:`h = U S V^T`, the kept rank ``r`` has
    :math:`S_r > \\mathrm{rank\\_tol}\\,S_0` and
    :math:`L_{ab,r} = \\sum_\\beta \\Lambda_{ab}(\\omega_\\beta) U_{\\beta r} S_r`.
    """
    wn = matsubara_grid(T, wscut)
    nw = wn.size
    freq = np.asarray(coupling['freq_ev'], dtype=float)
    pair = coupling['coupling']
    nstates = pair.shape[0]
    bosonic = 2.0 * np.pi * KB_EV * T * np.arange(2 * nw)
    frequency_kernel = freq[:, None] ** 2 / (freq[:, None] ** 2 + bosonic[None, :] ** 2)
    left, singular, right = np.linalg.svd(frequency_kernel, full_matrices=False)
    rank = int(np.count_nonzero(singular > rank_tol * singular[0]))
    pair_factors = pair.reshape(nstates * nstates, -1) @ (left[:, :rank] * singular[:rank])
    index = np.arange(nw)
    difference = np.abs(index[:, None] - index[None, :])
    total = index[:, None] + index[None, :] + 1
    right = right[:rank]
    weights = np.asarray(coupling['weight'], dtype=float)
    return {
        'wn': wn,
        'pair_factors': pair_factors.reshape(nstates, nstates * rank),
        'plus': right[:, difference] + right[:, total],
        'minus': right[:, difference] - right[:, total],
        'fractions': weights / weights.sum(),
        'coulomb': coulomb_weights(coupling),
        'rank': rank,
    }


def coulomb_weights(coupling: dict[str, Any]) -> NDArray[np.float64]:
    """Weights ``C_ab`` of the Coulomb pseudopotential in the gap equation.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling.

    Returns
    -------
    ndarray, shape ``(N, N)``
        ``C_ab = M_ab / N_F`` from the Coulomb pairs ``M_ab`` of the coupling
        (:mod:`~PAOFLOW.elphon.fermi_surface_coupling`), which restrict
        ``mu*`` to the ``(k, k+q)`` pairs of the phonon kernel; without them,
        ``C_ab = c_b`` for every row.

    Notes
    -----
    The rows sum to about 1.  The two forms agree when the q-grid equals the
    k-grid; with a coarser q-grid only the pair-restricted one is consistent
    with the phonon kernel.
    """
    weights = np.asarray(coupling['weight'], dtype=float)
    if 'coulomb' in coupling:
        return np.asarray(coupling['coulomb'], dtype=float) / weights.sum()
    return np.broadcast_to(weights / weights.sum(), (weights.size, weights.size))


def _apply_kernel(
    kernel: dict[str, Any], frequency_kernel: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``sum_{b, j'} lambda_ab(j, j') values_b(j')`` for the separable kernel.

    Parameters
    ----------
    kernel : dict
        Output of :func:`matsubara_kernel`.
    frequency_kernel : ndarray, shape ``(r, nw, nw)``
        ``kernel['plus']`` or ``kernel['minus']``.
    values : ndarray, shape ``(N, nw)``
        State- and frequency-resolved input.

    Returns
    -------
    ndarray, shape ``(N, nw)``
    """
    nstates, nw = values.shape
    rank = frequency_kernel.shape[0]
    convolved = np.matmul(frequency_kernel, values.T)  # (r, nw, N)
    convolved = convolved.transpose(2, 0, 1).reshape(nstates * rank, nw)
    return kernel['pair_factors'] @ convolved


def solve_imag_aniso(
    coupling: dict[str, Any],
    T: float,
    mu_star: float,
    wscut: float,
    nsiter: int = 500,
    conv_thr: float = 1.0e-4,
    delta0: float | ArrayLike | None = None,
    mix: float = 0.7,
    nhist: int = 8,
    normal_floor: float = 1.0e-7,
    rank_tol: float = 1.0e-10,
) -> dict[str, Any]:
    """Self-consistent anisotropic ``Delta_nk(i w_j)`` and ``Z_nk(i w_j)``.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling
        (:func:`~PAOFLOW.elphon.fermi_surface_coupling.read_fs_coupling`).
    T : float
        Temperature (K).
    mu_star : float
        Coulomb pseudopotential (EPW ``muc``).
    wscut : float
        Matsubara cutoff (eV, EPW ``wscut``).
    nsiter : int, optional
        Maximum number of iterations (EPW ``nsiter``).
    conv_thr : float, optional
        Convergence threshold (EPW ``conv_thr_iaxis``).
    delta0 : float or array_like, optional
        Initial gap (eV): a scalar, one value per state ``(N,)`` or one per state
        and frequency ``(N, nw)``.  Defaults to ``1.764 k_B Tc`` with the
        Allen-Dynes ``Tc`` of the isotropic ``a2F``.
    mix : float, optional
        Anderson mixing parameter.
    nhist : int, optional
        Number of previous steps kept by the mixing.
    normal_floor : float, optional
        Gap (eV) below which the solution is taken as the normal state.
    rank_tol : float, optional
        Singular-value cutoff of :func:`matsubara_kernel`.

    Returns
    -------
    dict
        ``wn`` ``(nw,)``, ``Z`` and ``delta`` ``(N, nw)`` (eV), ``niter``,
        ``converged`` and the kernel ``rank``.  The Fermi-surface average of
        ``delta(i w_0)`` is positive.

    Notes
    -----
    The equations are those of the module docstring, solved with
    :func:`~PAOFLOW.elphon.migdal_eliashberg.anderson_fixed_point`.
    """
    kernel = matsubara_kernel(coupling, T, wscut, rank_tol)
    wn, fractions = kernel['wn'], kernel['fractions']
    nstates = fractions.size
    pi_t = np.pi * KB_EV * T

    def gap_map(gap: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One evaluation of the right-hand sides: ``(G(Delta), Z)``."""
        root = np.sqrt(wn**2 + gap**2)
        Z = 1.0 + pi_t / wn * _apply_kernel(kernel, kernel['minus'], wn / root)
        ratio = gap / root
        coulomb = 2.0 * mu_star * (kernel['coulomb'] @ ratio.sum(axis=1))
        return pi_t / Z * (_apply_kernel(kernel, kernel['plus'], ratio) - coulomb[:, None]), Z

    if delta0 is None:
        strength = coupling_strength(coupling)
        delta0 = _gap_guess(strength['omega'], strength['a2F'], mu_star)
    delta0 = np.asarray(delta0, dtype=float)
    if delta0.ndim == 1:
        delta0 = delta0[:, None]
    gap_trial = np.broadcast_to(delta0, (nstates, wn.size)).copy()
    gap_mapped, Z, niter, converged = anderson_fixed_point(
        gap_map, gap_trial, nsiter, conv_thr, mix, nhist, normal_floor
    )
    if np.abs(gap_mapped).max() < normal_floor:
        gap_mapped, Z = gap_map(np.zeros_like(gap_trial))
    delta = gap_mapped if fractions @ gap_mapped[:, 0] >= 0.0 else -gap_mapped
    return {
        'wn': wn,
        'Z': Z,
        'delta': delta,
        'niter': niter,
        'converged': converged,
        'rank': kernel['rank'],
    }


def linearized_max_eigenvalue_aniso(
    coupling: dict[str, Any],
    T: float,
    mu_star: float,
    wscut: float,
    rank_tol: float = 1.0e-10,
) -> float:
    """Largest eigenvalue of the linearised anisotropic gap equation.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling.
    T : float
        Temperature (K).
    mu_star : float
        Coulomb pseudopotential.
    wscut : float
        Matsubara cutoff (eV).
    rank_tol : float, optional
        Singular-value cutoff of :func:`matsubara_kernel`.

    Returns
    -------
    float
        The eigenvalue ``rho``: above 1 below ``Tc``, 1 at ``Tc``.

    Notes
    -----
    Near ``Tc``

    .. math::

        \\rho\\,\\Delta_a(j) = \\frac{\\pi T}{Z^N_a(j)}\\sum_{b, j'}
            [\\lambda_{ab}(j-j') + \\lambda_{ab}(j+j'+1) - 2\\mu^* C_{ab}]\\,
            \\frac{\\Delta_b(j')}{\\omega_{j'}},

    with the normal-state ``Z^N``.  The leading eigenvalue (largest real part)
    is found with ARPACK on the matrix-free operator.
    """
    from scipy.sparse.linalg import LinearOperator, eigs

    kernel = matsubara_kernel(coupling, T, wscut, rank_tol)
    wn, fractions = kernel['wn'], kernel['fractions']
    nstates, nw = fractions.size, wn.size
    pi_t = np.pi * KB_EV * T
    Z_normal = 1.0 + pi_t / wn * _apply_kernel(kernel, kernel['minus'], np.ones((nstates, nw)))

    def linearized_map(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """The linearised gap equation applied to a flattened ``Delta``."""
        ratio = np.real(vector).reshape(nstates, nw) / wn
        coulomb = 2.0 * mu_star * (kernel['coulomb'] @ ratio.sum(axis=1))
        pairing = _apply_kernel(kernel, kernel['plus'], ratio) - coulomb[:, None]
        return (pi_t / Z_normal * pairing).ravel()

    size = nstates * nw
    if size < 8:
        dense = np.column_stack([linearized_map(column) for column in np.eye(size)])
        return float(np.max(np.linalg.eigvals(dense).real))
    operator = LinearOperator((size, size), matvec=linearized_map, dtype=float)
    eigenvalue = eigs(operator, k=1, which='LR', tol=1.0e-8, return_eigenvectors=False)
    return float(eigenvalue[0].real)


# --------------------------------------------------------------------------- #
# Real axis                                                                   #
# --------------------------------------------------------------------------- #
def pade_continuation_aniso(
    wn: ArrayLike,
    Z: ArrayLike,
    delta: ArrayLike,
    w_real: ArrayLike,
    npade: float = 90,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Real-axis ``Z_nk(w)``, ``Delta_nk(w)`` from Pade approximants, state by state.

    Parameters
    ----------
    wn : array_like, shape ``(nw,)``
        Matsubara frequencies (eV).
    Z, delta : array_like, shape ``(N, nw)``
        Imaginary-axis solution (:func:`solve_imag_aniso`).
    w_real : array_like, shape ``(nreal,)``
        Real frequencies (eV).
    npade : float, optional
        Percentage of the Matsubara frequencies used (EPW ``npade``).

    Returns
    -------
    Z_real, delta_real : ndarray, shape ``(N, nreal)``, complex
    """
    wn = np.asarray(wn, dtype=float)
    npoints = max(2, int(npade * wn.size / 100))
    z = 1j * wn[:npoints]
    Z_real = pade_eval(pade_coefficients(z, np.asarray(Z)[:, :npoints]), z, w_real)
    delta = np.asarray(delta)
    if not np.any(delta[:, :npoints]):
        return Z_real, np.zeros_like(Z_real)
    return Z_real, pade_eval(pade_coefficients(z, delta[:, :npoints]), z, w_real)


def gap_distribution(
    values: ArrayLike, weights: ArrayLike, grid: ArrayLike, smearing: float
) -> NDArray[np.float64]:
    """Gaussian-broadened, weighted distribution of state-resolved values.

    Parameters
    ----------
    values : array_like, shape ``(N,)``
        One value per state (e.g. ``Delta_nk`` or ``lambda_nk``); ``nan`` values
        are ignored.
    weights : array_like, shape ``(N,)``
        Fermi-surface weights.
    grid : array_like
        Points where the distribution is evaluated (units of ``values``).
    smearing : float
        Gaussian width (units of ``values``).

    Returns
    -------
    ndarray
        The distribution on ``grid``, normalised to unit integral.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values)
    values, weights = values[finite], weights[finite]
    if values.size == 0 or weights.sum() <= 0.0:
        return np.zeros(np.shape(grid))
    x = (np.asarray(grid, dtype=float)[:, None] - values[None, :]) / smearing
    gaussians = np.exp(-0.5 * x**2) / (smearing * np.sqrt(2.0 * np.pi))
    return gaussians @ weights / weights.sum()


# --------------------------------------------------------------------------- #
# Temperature sweeps                                                          #
# --------------------------------------------------------------------------- #
def migdal_eliashberg_aniso(
    coupling: dict[str, Any],
    temps: ArrayLike,
    mu_star: float = 0.1,
    wscut: float = 0.1,
    nsiter: int = 500,
    conv_thr_iaxis: float = 1.0e-4,
    npade: float = 90,
    lpade: bool = True,
    real_axis_max: float | None = None,
    n_real: int = 2000,
    rank_tol: float = 1.0e-10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Anisotropic Migdal-Eliashberg solution over a set of temperatures.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling
        (:func:`~PAOFLOW.elphon.fermi_surface_coupling.read_fs_coupling`).
    temps : array_like
        Temperatures (K); solved in ascending order.
    mu_star : float, optional
        Coulomb pseudopotential (EPW ``muc``, default 0.1).
    wscut : float, optional
        Matsubara cutoff (eV, default 0.1).
    nsiter : int, optional
        Maximum iterations (EPW ``nsiter``).
    conv_thr_iaxis : float, optional
        Convergence threshold on the imaginary axis.
    npade : float, optional
        Percentage of Matsubara points in the Pade approximants.
    lpade : bool, optional
        Continue each state to the real axis (EPW ``lpade``).
    real_axis_max : float, optional
        Upper real frequency (eV); defaults to ``0.2 * wscut``.
    n_real : int, optional
        Number of real frequencies (default 2000).
    rank_tol : float, optional
        Singular-value cutoff of :func:`matsubara_kernel`.
    verbose : bool, optional
        Print one summary line per temperature.

    Returns
    -------
    dict
        ``temps`` (K); ``gap0`` and ``Z0`` ``(ntemps, N)``, ``Delta_nk(i w_0)``
        (eV) and ``Z_nk(i w_0)``; ``gap_edge`` ``(ntemps, N)``, the Pade gap edge
        of every state (eV, ``nan`` where not computed); ``gap_mean``,
        ``gap_min``, ``gap_max`` (Fermi-surface average and range of ``gap0``);
        ``niter``, ``converged``; ``qdos`` ``(ntemps, n_real)`` (``N_S/N_F``,
        ``nan`` where not computed) and ``pade_mean`` (list of the
        Fermi-surface averaged ``(Z(w), Delta(w))`` or ``None``); ``imag``
        (list of ``(wn, Z, delta)``); ``w_real``; ``Tc_gap`` (K,
        :func:`~PAOFLOW.elphon.migdal_eliashberg.tc_from_gap` on the largest
        gap); ``mu_star``, ``wscut``; and the coupling summary of
        :func:`coupling_strength`.

    Warns
    -----
    UserWarning
        If the coupling was computed with a q-grid coarser than the k-grid, whose
        sublattices of ``k+q`` are independent superconductors.

    Notes
    -----
    Each temperature starts from the gap of the previous one, interpolated on
    its Matsubara grid.
    """
    nk_dense, nq_dense = coupling.get('nk_dense'), coupling.get('nq_dense')
    if nk_dense is not None and nq_dense is not None and int(nk_dense) != int(nq_dense):
        warnings.warn(
            'Fermi-surface coupling with nk_dense = %d and nq_dense = %d: k+q reaches only a'
            ' sublattice of the k-grid and the equations split into independent sublattice'
            ' problems.  Recompute it with nq_dense = nk_dense.' % (nk_dense, nq_dense),
            stacklevel=2,
        )
    temps = np.sort(np.asarray(temps, dtype=float))
    strength = coupling_strength(coupling)
    fractions = np.asarray(coupling['weight'], dtype=float) / strength['dos_ef']
    nstates, ntemps = fractions.size, temps.size
    if real_axis_max is None:
        real_axis_max = 0.2 * wscut
    w_real = real_axis_max / n_real * np.arange(1, n_real + 1)
    out: dict[str, Any] = {
        'gap0': np.zeros((ntemps, nstates)),
        'Z0': np.zeros((ntemps, nstates)),
        'gap_edge': np.full((ntemps, nstates), np.nan),
        'qdos': np.full((ntemps, n_real), np.nan),
        'niter': np.zeros(ntemps, dtype=int),
        'converged': np.zeros(ntemps, dtype=bool),
        'imag': [],
        'pade_mean': [],
    }
    previous_gap = None
    for it, T in enumerate(temps):
        delta0 = None
        if previous_gap is not None:
            previous_wn, previous_delta = previous_gap
            wn_next = matsubara_grid(T, wscut)
            delta0 = np.array([np.interp(wn_next, previous_wn, row) for row in previous_delta])
        solution = solve_imag_aniso(
            coupling, T, mu_star, wscut, nsiter, conv_thr_iaxis, delta0, rank_tol=rank_tol
        )
        wn, Zn, deltan = solution['wn'], solution['Z'], solution['delta']
        out['gap0'][it], out['Z0'][it] = deltan[:, 0], Zn[:, 0]
        out['niter'][it], out['converged'][it] = solution['niter'], solution['converged']
        out['imag'].append((wn, Zn, deltan))
        gapped = bool(np.any(deltan))
        if gapped:
            previous_gap = (wn, deltan)
        pade_mean = None
        if gapped and lpade:
            Z_real, delta_real = pade_continuation_aniso(wn, Zn, deltan, w_real, npade)
            out['gap_edge'][it] = [gap_edge(w_real, row) for row in delta_real]
            with np.errstate(divide='ignore', invalid='ignore'):
                out['qdos'][it] = fractions @ quasiparticle_dos(w_real, delta_real, Z_real)
            pade_mean = (fractions @ Z_real, fractions @ delta_real)
        elif not gapped:
            out['gap_edge'][it] = 0.0
            out['qdos'][it] = 1.0
        out['pade_mean'].append(pade_mean)
        if verbose:
            gap0 = out['gap0'][it]
            print(
                '  T = %6.3f K  N_w = %4d  iter = %3d  Delta_0 = %.4f .. %.4f meV'
                '  (FS average %.4f meV)'
                % (T, wn.size, solution['niter'], gap0.min() * 1e3, gap0.max() * 1e3,
                   fractions @ gap0 * 1e3),
                flush=True,
            )  # fmt: skip
    out['gap_mean'] = out['gap0'] @ fractions
    out['gap_min'] = out['gap0'].min(axis=1)
    out['gap_max'] = out['gap0'].max(axis=1)
    out['temps'] = temps
    out['w_real'] = w_real
    out['Tc_gap'] = tc_from_gap(temps, out['gap_max'])
    out['mu_star'], out['wscut'] = float(mu_star), float(wscut)
    out.update(strength)
    return out


def linearized_eigenvalues_aniso(
    coupling: dict[str, Any],
    temps: ArrayLike,
    mu_star: float = 0.1,
    wscut: float = 0.1,
) -> dict[str, Any]:
    """Largest linearised anisotropic eigenvalue at each temperature, and ``Tc``.

    Parameters
    ----------
    coupling : dict
        Fermi-surface coupling.
    temps : array_like
        Temperatures (K).
    mu_star : float, optional
        Coulomb pseudopotential (default 0.1).
    wscut : float, optional
        Matsubara cutoff (eV, default 0.1).

    Returns
    -------
    dict
        ``temps`` (sorted, K), ``max_eigenvalue`` and ``Tc_linear`` (K).
    """
    temps = np.sort(np.asarray(temps, dtype=float))
    rho = np.array([linearized_max_eigenvalue_aniso(coupling, T, mu_star, wscut) for T in temps])
    return {'temps': temps, 'max_eigenvalue': rho, 'Tc_linear': tc_from_eigenvalues(temps, rho)}


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def write_me_aniso_outputs(
    result: dict[str, Any],
    coupling: dict[str, Any],
    outdir: str,
    prefix: str,
    linear: dict[str, Any] | None = None,
    smearing_mev: float = 0.1,
    lambda_smearing: float = 0.01,
    n_grid: int = 500,
) -> None:
    """Write the anisotropic Migdal-Eliashberg results in EPW's file formats.

    Parameters
    ----------
    result : dict
        Output of :func:`migdal_eliashberg_aniso`.
    coupling : dict
        The Fermi-surface coupling used.
    outdir : str
        Output directory (created if needed).
    prefix : str
        File prefix (EPW ``prefix``).
    linear : dict, optional
        Output of :func:`linearized_eigenvalues_aniso`.
    smearing_mev : float, optional
        Gaussian width of the gap distributions (meV, default 0.1).
    lambda_smearing : float, optional
        Gaussian width of the ``lambda_nk`` distribution (default 0.01).
    n_grid : int, optional
        Number of points of the distributions.

    Returns
    -------
    None
        Writes ``<prefix>.lambda_FS`` (``k`` in Cartesian 2 pi / alat, band,
        ``e - E_F`` in eV, ``lambda_nk``) and ``<prefix>.lambda_k_pairs``
        (distribution of ``lambda_nk``); for every temperature ``<T>``
        (:func:`~PAOFLOW.elphon.migdal_eliashberg.temperature_tag`)
        ``<prefix>.imag_aniso_<T>`` (``w_j``, ``e - E_F``, ``Z``, ``Delta``, eV),
        ``<prefix>.imag_aniso_gap0_<T>`` and ``<prefix>.pade_aniso_gap0_<T>``
        (distributions of ``Delta_nk(i w_0)`` and of the Pade gap edges, meV),
        ``<prefix>.imag_aniso_gap_FS_<T>`` (``Delta_nk(i w_0)`` of every state),
        ``<prefix>.pade_aniso_<T>`` (Fermi-surface averaged ``Z(w)``,
        ``Delta(w)``) and ``<prefix>.qdos_<T>``; ``gap_vs_T_aniso.dat``,
        ``max_eigenvalue_aniso.dat`` (with ``linear``) and
        ``migdal_eliashberg_aniso.npz``.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, prefix + '.%s')
    weights = np.asarray(coupling['weight'], dtype=float)
    energy = np.asarray(coupling['energy_ev'], dtype=float)
    band = np.asarray(coupling['band'], dtype=int)
    k_cart = np.asarray(coupling['k_cryst'], dtype=float) @ np.asarray(coupling['bg'])
    lambda_nk = result['lambda_nk']
    state_columns = np.column_stack([k_cart, band + 1, energy])
    np.savetxt(
        path % 'lambda_FS',
        np.column_stack([state_columns, lambda_nk]),
        fmt=['%12.8f'] * 3 + ['%5d', '%12.8f', '%12.8f'],
        header='kx ky kz [2pi/alat]  band  e-E_F [eV]  lambda_nk   (irreducible states)',
    )
    lambda_grid = np.linspace(0.0, 1.2 * max(float(lambda_nk.max()), 1.0e-6), n_grid)
    lambda_density = gap_distribution(lambda_nk, weights, lambda_grid, lambda_smearing)
    np.savetxt(
        path % 'lambda_k_pairs',
        np.column_stack([lambda_grid, lambda_density]),
        header='lambda_nk  rho(lambda_nk)   (lambda = %.4f)' % result['lambda'],
    )

    gap_top = 1.2 * max(float(np.nanmax(result['gap0'])), float(np.nanmax(result['gap_edge'])))
    gap_grid = np.linspace(0.0, max(gap_top * 1e3, 1.0e-3), n_grid)  # meV
    ntemps = result['temps'].size
    arrays: dict[str, Any] = {
        'gap_grid_mev': gap_grid,
        'gap0_distribution': np.zeros((ntemps, n_grid)),
        'gap_edge_distribution': np.zeros((ntemps, n_grid)),
        'lambda_grid': lambda_grid,
        'lambda_distribution': lambda_density,
    }
    for i, T in enumerate(result['temps']):
        tag = temperature_tag(T)
        wn, Z, delta = result['imag'][i]
        nw = wn.size
        np.savetxt(
            path % ('imag_aniso_' + tag),
            np.column_stack(
                [np.tile(wn, energy.size), np.repeat(energy, nw), Z.ravel(), delta.ravel()]
            ),
            header='w_j [eV]  e-E_F [eV]  Z(iw_j)  Delta(iw_j) [eV]   T = %g K' % T,
        )
        arrays['gap0_distribution'][i] = gap_distribution(
            result['gap0'][i] * 1e3, weights, gap_grid, smearing_mev
        )
        np.savetxt(
            path % ('imag_aniso_gap0_' + tag),
            np.column_stack([gap_grid, arrays['gap0_distribution'][i]]),
            header='Delta_nk(iw_0) [meV]  rho(Delta)   T = %g K' % T,
        )
        np.savetxt(
            path % ('imag_aniso_gap_FS_' + tag),
            np.column_stack([state_columns, result['gap0'][i] * 1e3]),
            fmt=['%12.8f'] * 3 + ['%5d', '%12.8f', '%12.8f'],
            header='kx ky kz [2pi/alat]  band  e-E_F [eV]  Delta_nk(iw_0) [meV]   T = %g K' % T,
        )
        if result['pade_mean'][i] is not None:
            arrays['gap_edge_distribution'][i] = gap_distribution(
                result['gap_edge'][i] * 1e3, weights, gap_grid, smearing_mev
            )
            np.savetxt(
                path % ('pade_aniso_gap0_' + tag),
                np.column_stack([gap_grid, arrays['gap_edge_distribution'][i]]),
                header='gap edge [meV]  rho   (Pade)   T = %g K' % T,
            )
            Z_mean, delta_mean = result['pade_mean'][i]
            np.savetxt(
                path % ('pade_aniso_' + tag),
                np.column_stack(
                    [result['w_real'], Z_mean.real, Z_mean.imag, delta_mean.real, delta_mean.imag]
                ),
                header='w [eV]  Re Z  Im Z  Re Delta [eV]  Im Delta [eV]'
                '   (Fermi-surface average)   T = %g K' % T,
            )
            np.savetxt(
                path % ('qdos_' + tag),
                np.column_stack([result['w_real'], result['qdos'][i]]),
                header='w [eV]  N_S(w)/N_F   T = %g K' % T,
            )
    np.savetxt(
        os.path.join(outdir, 'gap_vs_T_aniso.dat'),
        np.column_stack(
            [
                result['temps'],
                result['gap_mean'] * 1e3,
                result['gap_min'] * 1e3,
                result['gap_max'] * 1e3,
            ]
        ),
        header='T [K]  <Delta_nk(iw_0)>_FS [meV]  min [meV]  max [meV]'
        '   (Tc from Delta_max^2 extrapolation = %.3f K)' % result['Tc_gap'],
    )
    if linear is not None:
        np.savetxt(
            os.path.join(outdir, 'max_eigenvalue_aniso.dat'),
            np.column_stack([linear['temps'], linear['max_eigenvalue']]),
            header='T [K]  max eigenvalue of the linearised anisotropic kernel   (Tc = %.3f K)'
            % linear['Tc_linear'],
        )
        arrays.update(
            lin_temps=linear['temps'],
            max_eigenvalue=linear['max_eigenvalue'],
            Tc_linear=linear['Tc_linear'],
        )
    for key in (
        'temps', 'gap0', 'Z0', 'gap_edge', 'gap_mean', 'gap_min', 'gap_max', 'qdos', 'niter',
        'converged', 'w_real', 'Tc_gap', 'mu_star', 'wscut', 'lambda_nk', 'lambda', 'dos_ef',
        'omega', 'a2F',
    ):  # fmt: skip
        arrays[key] = result[key]
    arrays.update(weight=weights, energy_ev=energy, band=band, k_cart=k_cart)
    np.savez(os.path.join(outdir, 'migdal_eliashberg_aniso.npz'), **arrays)
