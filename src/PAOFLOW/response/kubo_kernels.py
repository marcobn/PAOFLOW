r"""Kubo-Bastin kernels of the linear response at constant broadening.

Every linear response of PAOFLOW is a limit of the Kubo-Bastin formula. For the response of an
observable :math:`A` to a field coupled through :math:`B` (the velocity for an electric field), with
the band energies :math:`\varepsilon_n` and the matrix elements :math:`A_{nm}`, :math:`B_{mn}` at one
k-point, the Bastin form with a constant broadening :math:`\eta` is

.. math::

    \chi_{AB}(\mu) = -2\int d\varepsilon\, f(\varepsilon)\,
        \mathrm{Im}\,\mathrm{Tr}\big[A\,\partial_\varepsilon G^R(\varepsilon)\,B\,
        \delta(\varepsilon - H)\big]
    = \underbrace{\pi\sum_{nm}\mathrm{Re}P_{nm}\int(-f')L_nL_m}_{\text{surface}}
    + \underbrace{2\sum_{nm}\mathrm{Im}P_{nm}\int f\,L_n\,
        \mathrm{Re}\frac{1}{(\varepsilon-\varepsilon_m+i\eta)^2}}_{\text{sea}},

with :math:`P_{nm} = A_{nm}B_{mn}`, :math:`G^R = (\varepsilon - H + i\eta)^{-1}` and the normalised
Lorentzian :math:`L_n(\varepsilon) = (\eta/\pi)/((\varepsilon-\varepsilon_n)^2+\eta^2)`. The split is
exact (Bonbien-Manchon "true" Fermi-surface and Fermi-sea terms): the symmetric part of
:math:`P` gives only the surface term, the antisymmetric part only the sea term. Both are evaluated
here in closed form at :math:`T = 0`; a finite temperature follows from the exact identity
:math:`\chi(\mu, T) = \int d\mu'\,(-\partial_{\mu'}f)\,\chi(\mu', 0)` (:func:`thermal_average`),
since the formula is linear in :math:`f`.

Limits (tutorial03 equations): the surface term is Eq. 1 (exact at :math:`T = 0`), its diagonal
part the intraband Eq. 4 before the :math:`\eta\to 0` limit, and the sea term tends to the
Berry-curvature form :math:`2\sum_n f_n\sum_m \mathrm{Im}P_{nm}/\varepsilon_{nm}^2`, minus the
:math:`\sum_n f_n\Omega_n` of Eq. 3 and of ``do_Hall``.

References
----------
- A. Crépieux and P. Bruno, Phys. Rev. B 64, 014416 (2001).
- V. Bonbien and A. Manchon, Phys. Rev. B 102, 085113 (2020).
- O. Ly, J. Phys.: Condens. Matter 38, 435802 (2026).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Pairs closer than this fraction of the broadening use the series of the sea kernel.
NEAR_DEGENERATE_FRACTION = 0.25
_SERIES_TERMS = 30


def lorentzian(energies: NDArray[np.float64], mu: NDArray[np.float64], eta: float) -> NDArray:
    """Normalised Lorentzian spectral weight of every band at every chemical potential.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)``
        Band energies (eV).
    mu : ndarray, shape ``(nmu,)``
        Chemical potentials (eV).
    eta : float
        Broadening (eV).

    Returns
    -------
    ndarray, shape ``(nmu, nb)``
        :math:`L_n(\\mu) = (\\eta/\\pi)/((\\mu-\\varepsilon_n)^2+\\eta^2)` (1/eV).
    """
    return (eta / np.pi) / ((mu[:, None] - energies[None, :]) ** 2 + eta**2)


def surface_kernel(
    energies: NDArray[np.float64],
    pair: NDArray[np.complex128],
    mu: NDArray[np.float64],
    eta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fermi-surface term at :math:`T = 0` and its intraband part, summed over k-points.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)`` or ``(nk, nb)``
        Band energies (eV) of one k-point or a block of k-points.
    pair : ndarray, shape ``(nb, nb)`` or ``(nk, nb, nb)``, complex
        :math:`P_{nm} = A_{nm}B_{mn}`.
    mu : ndarray, shape ``(nmu,)``
        Chemical potentials (eV).
    eta : float
        Broadening (eV).

    Returns
    -------
    total : ndarray, shape ``(nmu,)``
        :math:`\\pi\\sum_k\\sum_{nm}\\mathrm{Re}P_{nm}L_n(\\mu)L_m(\\mu)`.
    intraband : ndarray, shape ``(nmu,)``
        The :math:`n = m` part, :math:`\\pi\\sum_k\\sum_n P_{nn}L_n(\\mu)^2`.

    Notes
    -----
    Exact for a constant broadening at :math:`T = 0`; tutorial03 Eq. 1 is the same expression.
    """
    energies = np.atleast_2d(energies)
    real_pair = np.real(pair).reshape((-1,) + pair.shape[-2:])
    weights = (eta / np.pi) / ((mu[None, :, None] - energies[:, None, :]) ** 2 + eta**2)
    total = np.pi * np.einsum('ken,knm,kem->e', weights, real_pair, weights, optimize=True)
    diagonal = np.diagonal(real_pair, axis1=1, axis2=2)
    intraband = np.pi * np.einsum('ken,kn->e', weights**2, diagonal)
    return total, intraband


def sea_kernel(
    energies: NDArray[np.float64],
    pair: NDArray[np.complex128],
    mu: NDArray[np.float64],
    eta: float,
) -> NDArray[np.float64]:
    """Fermi-sea term at :math:`T = 0` in closed form, summed over k-points.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)`` or ``(nk, nb)``
        Band energies (eV) of one k-point or a block of k-points.
    pair : ndarray, shape ``(nb, nb)`` or ``(nk, nb, nb)``, complex
        :math:`P_{nm} = A_{nm}B_{mn}` (only the imaginary, antisymmetric part enters).
    mu : ndarray, shape ``(nmu,)``
        Chemical potentials (eV).
    eta : float
        Broadening (eV), > 0.

    Returns
    -------
    ndarray, shape ``(nmu,)``
        :math:`2\\sum_{nm}\\mathrm{Im}P_{nm}\\,S_{nm}(\\mu)` with
        :math:`S_{nm}(\\mu) = \\int_{-\\infty}^{\\mu}L_n(\\varepsilon)\\,
        \\mathrm{Re}(\\varepsilon-\\varepsilon_m+i\\eta)^{-2}\\,d\\varepsilon`.

    Notes
    -----
    With :math:`s_n = \\varepsilon_n + i\\eta`, :math:`q_m = \\varepsilon_m - i\\eta` and partial
    fractions,

    .. math::

        S_{nm} = \\frac{1}{2\\pi}\\mathrm{Im}\\big[I(s_n, q_m) + I(s_n, \\bar q_m)\\big],\\qquad
        I(a, b) = \\int_{-\\infty}^{\\mu}\\frac{d\\varepsilon}{(\\varepsilon-a)(\\varepsilon-b)^2}
        = \\frac{\\ell(a)-\\ell(b)}{(a-b)^2} + \\frac{1}{(a-b)(\\mu-b)},

    :math:`\\ell(p) = \\log(\\mu-p) + i\\pi\\,\\mathrm{sgn}\\,\\mathrm{Im}\\,p`. The pair coefficients
    depend on :math:`\\varepsilon_n-\\varepsilon_m` only and the :math:`\\mu` dependence on single
    bands, so the cost is :math:`O(n_b^2 + n_b n_\\mu)` per k-point. For :math:`|\\varepsilon_n -
    \\varepsilon_m| <` ``NEAR_DEGENERATE_FRACTION`` :math:`\\eta` the second integral uses the series
    :math:`I(b+d, b) = -\\sum_{j\\ge0} d^j/((j+2)(\\mu-b)^{j+2})` (:math:`|\\mu - b|\\ge\\eta`).
    As :math:`\\eta\\to0`, :math:`S_{nm}\\to\\theta(\\mu-\\varepsilon_n)/\\varepsilon_{nm}^2`.
    """
    energies = np.atleast_2d(energies)
    imag_pair = np.imag(pair).reshape((-1,) + pair.shape[-2:])
    nb = energies.shape[1]
    difference = energies[:, :, None] - energies[:, None, :]  # e_n - e_m
    off_diagonal = ~np.eye(nb, dtype=bool)[None]
    # (s_n, q_m): a - b = e_nm + 2 i eta, never small
    d_opposite = difference + 2j * eta
    alpha_opposite = imag_pair / d_opposite**2
    gamma_opposite = imag_pair / d_opposite
    # (s_n, conj q_m): a - b = e_nm (real), series when small
    near = (np.abs(difference) < NEAR_DEGENERATE_FRACTION * eta) & off_diagonal
    far = ~near & off_diagonal
    safe = np.where(far, difference, 1.0)
    alpha_same = np.where(far, imag_pair / safe**2, 0.0)
    gamma_same = np.where(far, imag_pair / safe, 0.0)

    shift = mu[None, :, None] - energies[:, None, :]  # (nk, nmu, nb)
    log_upper = np.log(shift - 1j * eta) + 1j * np.pi  # l(s_n) = l(conj q_n)
    log_lower = np.log(shift + 1j * eta) - 1j * np.pi  # l(q_n)
    inv_upper = 1.0 / (shift - 1j * eta)
    inv_lower = 1.0 / (shift + 1j * eta)

    total = (
        np.einsum('ken,kn->e', log_upper, alpha_opposite.sum(axis=2) + alpha_same.sum(axis=2))
        - np.einsum('ken,kn->e', log_lower, alpha_opposite.sum(axis=1))
        - np.einsum('ken,kn->e', log_upper, alpha_same.sum(axis=1))
        + np.einsum('ken,kn->e', inv_lower, gamma_opposite.sum(axis=1))
        + np.einsum('ken,kn->e', inv_upper, gamma_same.sum(axis=1))
    )
    ks, rows, cols = np.nonzero(near & (imag_pair != 0.0))
    if ks.size:
        z = inv_upper[ks, :, cols]  # (npairs, nmu): 1/(mu - conj q_m)
        d = difference[ks, rows, cols][:, None]
        term = z**2
        series = np.zeros_like(z)
        for j in range(_SERIES_TERMS):
            series += term / (j + 2)
            term = term * d * z
        total -= np.sum(imag_pair[ks, rows, cols][:, None] * series, axis=0)
    return np.imag(total) / np.pi


def berry_curvature(
    energies: NDArray[np.float64],
    response: NDArray[np.complex128],
    field: NDArray[np.complex128],
    regulariser: float,
    cutoff: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Band-resolved Berry-curvature-like quantity of PAOFLOW's Eq. 3 and ``do_Hall``, one k-point.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)``
        Band energies (eV).
    response, field : ndarray, shape ``(nb, nb)``, complex
        :math:`A_{nm}` and :math:`B_{nm}` in the band basis.
    regulariser : float
        :math:`\\delta` of the denominators (eV).
    cutoff : float
        Pairs with :math:`\\varepsilon_{nm}^2 + \\delta^2 <` ``cutoff`` (eV^2) are left out.

    Returns
    -------
    ndarray, shape ``(nb,)``
        :math:`\\Omega_n = -2\\sum_m\\mathrm{Im}[A_{nm}B_{mn}]/(\\varepsilon_{nm}^2+\\delta^2)`.

    Notes
    -----
    :math:`\\sum_n f_n\\Omega_n` is minus the :math:`\\eta\\to0` limit of :func:`sea_kernel`.
    """
    denominator = (energies - energies[:, None]) ** 2 + regulariser**2
    denominator[denominator < cutoff] = np.inf
    return -2.0 * np.sum(np.imag(response * field.T) / denominator, axis=1)


def fermi_dirac(energies: NDArray[np.float64], mu: NDArray[np.float64], kT: float) -> NDArray:
    """Fermi-Dirac occupations, overflow-safe.

    Parameters
    ----------
    energies : ndarray, shape ``(..., nb)``
        Band energies (eV).
    mu : ndarray, shape ``(nmu,)``
        Chemical potentials (eV).
    kT : float
        Temperature (eV), > 0.

    Returns
    -------
    ndarray, shape ``(nmu, ..., nb)``
        :math:`f = 1/(e^{(\\varepsilon-\\mu)/kT}+1)`.
    """
    x = (energies[None, ...] - mu.reshape((-1,) + (1,) * energies.ndim)) / kT
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def thermal_grid(
    mu: NDArray[np.float64], eta: float, kT: float, cutoff: float = 36.0
) -> NDArray[np.float64]:
    """Fine chemical-potential grid for :func:`thermal_average`.

    Parameters
    ----------
    mu : ndarray, shape ``(nmu,)``
        Target chemical potentials (eV).
    eta : float
        Broadening (eV): the narrowest feature of the :math:`T = 0` response.
    kT : float
        Temperature (eV).
    cutoff : float
        Half-width of the thermal window in units of :math:`kT` (:math:`e^{-36}\\approx 2\\cdot10^{-16}`).

    Returns
    -------
    ndarray
        Uniform grid covering ``[min(mu) - cutoff kT, max(mu) + cutoff kT]`` with a spacing of at most
        ``min(eta, kT) / 4``.
    """
    lower, upper = float(np.min(mu)) - cutoff * kT, float(np.max(mu)) + cutoff * kT
    spacing = min(eta, kT) / 4.0
    count = int(np.ceil((upper - lower) / spacing)) + 1
    return np.linspace(lower, upper, count)


def thermal_average(
    fine_mu: NDArray[np.float64],
    values: NDArray[np.float64],
    mu: NDArray[np.float64],
    kT: float,
) -> NDArray[np.float64]:
    """Finite-temperature response from the zero-temperature one.

    Parameters
    ----------
    fine_mu : ndarray, shape ``(nfine,)``
        Uniform grid of :func:`thermal_grid`.
    values : ndarray, shape ``(nfine, ...)``
        :math:`\\chi(\\mu', T = 0)` on ``fine_mu``.
    mu : ndarray, shape ``(nmu,)``
        Chemical potentials of the result (eV).
    kT : float
        Temperature (eV).

    Returns
    -------
    ndarray, shape ``(nmu, ...)``
        :math:`\\chi(\\mu, T) = \\int d\\mu'\\,(-\\partial_{\\mu'} f(\\mu'-\\mu))\\,\\chi(\\mu', 0)`,
        trapezoidal rule.

    Notes
    -----
    Exact identity for any term linear in the occupation: :math:`f_T(\\varepsilon-\\mu) =
    \\int d\\mu'\\,\\theta(\\mu'-\\varepsilon)\\,(-\\partial_{\\mu'}f_T(\\mu'-\\mu))`.
    """
    x = (fine_mu[None, :] - mu[:, None]) / (2.0 * kT)
    weights = (fine_mu[1] - fine_mu[0]) * 0.25 / (kT * np.cosh(np.clip(x, -350.0, 350.0)) ** 2)
    weights[:, [0, -1]] *= 0.5  # trapezoid end points
    return np.tensordot(weights, values, axes=(1, 0))


def optical_kernel(
    energies: NDArray[np.float64],
    pair: NDArray[np.complex128],
    occupations: NDArray[np.float64],
    fermi_window: NDArray[np.float64],
    omegas: NDArray[np.float64],
    eta: float,
    degenerate_threshold: float = 1.0e-4,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r"""Frequency-dependent (Kubo-Greenwood) response, interband and intraband, summed over k-points.

    Parameters
    ----------
    energies : ndarray, shape ``(nb,)`` or ``(nk, nb)``
        Band energies (eV).
    pair : ndarray, shape ``(nb, nb)`` or ``(nk, nb, nb)``, complex
        :math:`P_{nm} = A_{nm}(\hbar v_b)_{mn}`.
    occupations : ndarray, shape like ``energies``
        :math:`f_n` at the chemical potential.
    fermi_window : ndarray, shape like ``energies``
        :math:`-\partial f/\partial\varepsilon` at :math:`\varepsilon_n` (1/eV).
    omegas : ndarray, shape ``(nw,)``
        Photon energies :math:`\hbar\omega` (eV).
    eta : float
        Broadening (eV): the half-width :math:`\eta` of the interband transitions,
        :math:`\gamma = 2\eta = \hbar/\tau` the damping of the oscillators and of the Drude term.
    degenerate_threshold : float
        Pairs with :math:`|\varepsilon_m-\varepsilon_n|` below it (eV) belong to the intraband term.

    Returns
    -------
    interband, intraband : ndarray, shape ``(nw,)``, complex

    Notes
    -----
    With :math:`\varepsilon_{mn} = \varepsilon_m - \varepsilon_n`, the interband position
    :math:`r^b_{mn} = -i(\hbar v_b)_{mn}/\varepsilon_{mn}` (length gauge) and
    :math:`D_{mn}(\omega) = \varepsilon_{mn}^2 - (\hbar\omega)^2 - i\gamma\hbar\omega`,

    .. math::

        \chi^{\rm inter}(\omega) = -i\sum_{\varepsilon_{mn}\ne0}
            \frac{(f_n-f_m)\,P_{nm}\,(\varepsilon_{mn}+\hbar\omega)}{\varepsilon_{mn}\,D_{mn}(\omega)},
        \qquad
        \chi^{\rm intra}(\omega) = \frac{i}{\hbar\omega+i\gamma}
            \sum_{\varepsilon_{mn}=0}\Big(-\frac{\partial f}{\partial\varepsilon}\Big)_n P_{nm}.

    For Hermitian :math:`A` and :math:`B` (:math:`P_{mn} = P_{nm}^*`), the pairs
    :math:`(n, m)` and :math:`(m, n)` together give

    .. math::

        -\frac{2i\,(f_n-f_m)\,\mathrm{Re}P_{nm}\,\hbar\omega}{\varepsilon_{mn}D_{mn}}
        + \frac{2\,(f_n-f_m)\,\mathrm{Im}P_{nm}}{D_{mn}},

    the damped (Lorentz) oscillator of each transition. This is the lineshape of epsilon.x and of
    :func:`~PAOFLOW.response.do_epsilon.do_dielectric_tensor`, near resonance a Lorentzian of
    half-width :math:`\gamma/2 = \eta`. The textbook substitution
    :math:`\hbar\omega\to\hbar\omega+i\eta` in :math:`1/(\varepsilon_{mn}-\hbar\omega)` gives the same
    resonances, but an insulator then conducts at :math:`\omega = 0`
    (:math:`\mathrm{Re}\,\sigma_{aa}(0)\propto\eta`, a sub-gap :math:`\varepsilon_2\propto\eta/\omega`).
    The oscillator form does not:

    - :math:`\mathrm{Re}\,\sigma_{aa}\propto\omega^2` as :math:`\omega\to0`;
    - at :math:`\omega = 0` the Hall part is :math:`\sum(f_n-f_m)\mathrm{Im}P_{nm}/\varepsilon_{mn}^2`,
      the clean-limit Berry-curvature sea of :func:`berry_curvature` (the ``intrinsic`` term), at any
      :math:`\eta`;
    - as :math:`\varepsilon_{mn}\to0` (at :math:`\omega\ne0`) a transition becomes a Drude term with
      the same damping :math:`\gamma`;
    - the high-frequency tail, hence the f-sum rule, is that of the undamped response.

    The degenerate blocks enter the intraband term as a whole (:math:`\mathrm{Tr}_D\,P`, gauge
    invariant); at :math:`\omega = 0` it is the Boltzmann limit of the Fermi-surface term,
    :math:`\sum_n(-f'_n)P_{nn}/2\eta`. The Hall part is Eq. 7 of Yao et al.,
    PRL 92, 037204 (2004), with :math:`(\hbar\omega+i\eta)^2` replaced by
    :math:`(\hbar\omega)^2 + i\gamma\hbar\omega`.
    """
    energies = np.atleast_2d(energies)
    pair = pair.reshape((-1,) + pair.shape[-2:])
    occupations = np.atleast_2d(occupations)
    fermi_window = np.atleast_2d(fermi_window)
    difference = energies[:, None, :] - energies[:, :, None]  # [k, n, m] = e_m - e_n
    degenerate = np.abs(difference) < degenerate_threshold
    drude_weight = np.sum(np.where(degenerate, fermi_window[:, :, None] * pair, 0.0))
    transitions = ~degenerate
    weights = ((occupations[:, :, None] - occupations[:, None, :]) * pair)[transitions]
    gaps = difference[transitions]
    weights = weights / gaps
    keep = weights != 0.0
    weights, gaps = weights[keep], gaps[keep]
    damping = 2.0 * eta
    interband = np.empty(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        oscillator = gaps * gaps - omega * omega - 1j * damping * omega
        interband[i] = -1j * np.sum(weights * (gaps + omega) / oscillator)
    intraband = 1j * drude_weight / (omegas + 1j * damping)
    return interband, intraband
