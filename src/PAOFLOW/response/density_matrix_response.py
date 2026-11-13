r"""Second-order optical response from the density-matrix equations of motion: metals and insulators.

The length-gauge equations of motion are solved order by order in the band basis without the
rearrangements that the insulator formulas of :mod:`~PAOFLOW.response.nonlinear_optics` rely on, so
the occupations may depend on k (a Fermi surface). With the electron charge :math:`-e` the perturbation
is :math:`e\mathbf E\cdot\mathbf r`, the position splits into its interband part :math:`r_i` and the
intraband part, which acts as the covariant k-derivative, and

.. math::

    (\hbar\Omega + i\Gamma_{nm} - \varepsilon_{nm})\,\rho^{(N)}_{nm}
    = eE^c\Big([r_i^c, \rho^{(N-1)}]_{nm} + i\big(\rho^{(N-1)}_{nm}\big)_{;c}\Big),
    \qquad \rho^{(0)}_{nm} = f_n\delta_{nm},

with :math:`\Gamma_{nm} = \eta` between bands (dephasing) and :math:`\Gamma_{nm} = \gamma = \hbar/\tau` inside a
band or a degenerate cluster (intraband relaxation). At first order this gives the interband density
:math:`-eE^b f_{nm}r^b_{nm}/(\hbar\omega - \varepsilon_{nm} + i\eta)` and the intraband (Fermi-surface)
density :math:`i eE^b\,\partial_b f_n/(\hbar\omega + i\gamma)`, :math:`\partial_b f_n = f'(\varepsilon_n)v^b_{nn}`.
The second order is collected by origin:

* ``fermi_sea``: the interband first-order density and the parts of its derivative without
  :math:`\partial f` (the response of an insulator); split into the interband elements of
  :math:`\rho^{(2)}` (``fermi_sea_inter``, the shift-current part at zero frequency) and the
  intraband ones (``fermi_sea_intra``, populations: the injection part, :math:`\propto\tau` at zero
  frequency);
* ``fermi_surface_mixed``: :math:`\partial f` through the derivative of the interband density and the
  interband commutator of the intraband density (in the zero-frequency limit the Berry-curvature
  dipole, :math:`\propto\tau`);
* ``fermi_surface_drude``: the derivative of the intraband density, :math:`\partial\partial f`
  (the nonlinear Drude response, :math:`\propto\tau^2` at zero frequency).

The current is :math:`J^a = -e\,{\rm Tr}[v^a\rho^{(2)}]`; the second-harmonic susceptibility follows from
:math:`P(2\omega) = J/(-2i\omega)`, :math:`P^a = \epsilon_0\chi^{abc}E^bE^c` (symmetrised in ``b, c``),
and the zero-frequency photocurrent from :math:`J^a = \sigma^{abc}E^b(\omega)E^c(-\omega)` with
:math:`\mathbf E(t) = \mathbf E e^{-i\omega t} + {\rm c.c.}`. The Fermi-surface terms need a smooth
occupation: Fermi–Dirac at an electronic temperature (they vanish with step occupations).
:math:`\tau` is a constant or a function :math:`\tau(\varepsilon)` (e.g. electron–phonon rates).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.constants import ELECTRON_CHARGE_SI, EPSILON0_SI, HBAR
from .nonlinear_optics import (
    KELVIN_TO_EV,
    band_geometry,
    component_label,
    dipole_gradient_by_differences,
    parse_components,
)

#: response channels of :func:`second_order_block`, in output order
CHANNELS = ('fermi_sea_inter', 'fermi_sea_intra', 'fermi_surface_mixed', 'fermi_surface_drude')
#: hbar in eV fs
HBAR_EV_FS = HBAR * 1.0e15


def fermi_dirac_derivatives(
    energies: NDArray[np.float64], fermi_energy: float, temperature_k: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Occupations and their first and second energy derivatives (per eV and eV^-2).

    Parameters
    ----------
    energies : ndarray
        Band energies (eV).
    fermi_energy : float
        Chemical potential (eV).
    temperature_k : float
        Electronic temperature (K); 0 gives a step and vanishing derivatives.

    Returns
    -------
    tuple of ndarray
        ``f``, ``df/de``, ``d2f/de2``, shaped as ``energies``.
    """
    e = np.asarray(energies, dtype=float)
    if temperature_k <= 0.0:
        return (e < float(fermi_energy)).astype(float), np.zeros_like(e), np.zeros_like(e)
    kt = KELVIN_TO_EV * float(temperature_k)
    x = np.clip((e - float(fermi_energy)) / kt, -200.0, 200.0)
    f = 0.5 * (1.0 - np.tanh(0.5 * x))
    fp = -f * (1.0 - f) / kt
    fpp = -fp * (1.0 - 2.0 * f) / kt
    return f, fp, fpp


def intraband_rates(
    energies: NDArray[np.float64],
    relaxation_time: float | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    default_rate: float,
    step: float = 1.0e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Intraband relaxation rates :math:`\gamma = \hbar/\tau` (eV) and their energy derivative.

    Parameters
    ----------
    energies : ndarray
        Band energies (eV).
    relaxation_time : float, callable or None
        :math:`\tau` in fs, a constant or a function of the band energy (eV -> fs, vectorised);
        ``None`` uses ``default_rate``.
    default_rate : float
        :math:`\gamma` (eV) without a relaxation time (the interband :math:`\eta`, as the insulator
        formulas use one damping everywhere).
    step : float, optional
        Energy step of the central difference for :math:`d\gamma/d\varepsilon` with a callable (eV).

    Returns
    -------
    tuple of ndarray
        :math:`\gamma` (eV) and :math:`d\gamma/d\varepsilon` (dimensionless), shaped as ``energies``.
    """
    e = np.asarray(energies, dtype=float)
    if relaxation_time is None:
        return np.full_like(e, float(default_rate)), np.zeros_like(e)
    if callable(relaxation_time):
        tau = np.asarray(relaxation_time(e), dtype=float)
        tau_p = np.asarray(relaxation_time(e + step), dtype=float)
        tau_m = np.asarray(relaxation_time(e - step), dtype=float)
        return HBAR_EV_FS / tau, (HBAR_EV_FS / tau_p - HBAR_EV_FS / tau_m) / (2.0 * step)
    return np.full_like(e, HBAR_EV_FS / float(relaxation_time)), np.zeros_like(e)


def band_curvature(
    eigenvectors: NDArray[np.complex128],
    omega: NDArray[np.float64],
    pair_mask: NDArray[np.bool_],
    velocity_orb: NDArray[np.complex128],
    curvature_orb: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""Intra-cluster inverse-mass tensor :math:`M^{bc} = (v^b)_{;c} = (w^{bc} + [G^c, v^b])_{\rm intra}`.

    Parameters
    ----------
    eigenvectors : ndarray ``(nb, n, n)``
    omega : ndarray ``(nb, n, n)``
        :math:`\omega_{nm} = \varepsilon_n - \varepsilon_m` (eV).
    pair_mask : ndarray ``(nb, n, n)``, bool
        Pairs in different degenerate clusters.
    velocity_orb, curvature_orb : ndarray
        Tight-binding velocity ``(nb, 3, n, n)`` and curvature ``(nb, 3, 3, n, n)`` operators.

    Returns
    -------
    ndarray ``(nb, 3, 3, n, n)``
        ``[b, c]`` -> :math:`M^{bc}` inside the clusters (zero between them), eV bohr^2. For a
        non-degenerate band the diagonal is :math:`\partial_b\partial_c\varepsilon_n`.

    Notes
    -----
    The covariant derivative with the intra-cluster connection of :math:`v^b = U^\dagger\partial_bH\,U`
    is the full derivative :math:`w^{bc} = U^\dagger\partial_b\partial_cH\,U` plus the commutator with the
    interband connection :math:`\xi_{\rm inter} = -iG`, :math:`G^c_{pq} = v^c_{pq}/\omega_{pq}`.
    """
    U = eigenvectors
    v = np.einsum('kim,kaij,kjn->kamn', U.conj(), velocity_orb, U, optimize=True)
    w = np.einsum('kim,kabij,kjn->kabmn', U.conj(), curvature_orb, U, optimize=True)
    inv = np.where(pair_mask, 1.0 / np.where(pair_mask, omega, 1.0), 0.0)
    G = v * inv[:, None]
    nb, n = omega.shape[0], omega.shape[1]
    M = np.zeros((nb, 3, 3, n, n), dtype=complex)
    intra = ~pair_mask
    for b in range(3):
        for c in range(3):
            M[:, b, c] = (w[:, b, c] + G[:, c] @ v[:, b] - v[:, b] @ G[:, c]) * intra
    return M


def dc_population_kernel(
    x: NDArray[np.float64],
    widths: NDArray[np.float64],
    gamma: NDArray[np.float64],
    eta: float,
    model: str = 'hybrid',
) -> NDArray[np.float64]:
    r"""Weight of a pair in the zero-frequency second-order populations (eV^-2).

    Summed over the two field orders, the populations driven by the interband first-order density are
    :math:`\rho^{(2)}_{nn'} = -\sum_m f_{nm}\,[r^c_{nm}r^b_{mn'}K(\hbar\omega-\varepsilon_{mn}) +
    r^b_{nm}r^c_{mn'}K(\hbar\omega+\varepsilon_{mn})]` (the principal parts cancel exactly). The kernel is

    * ``'relaxation'``: :math:`K = (2\eta/\gamma)/(x^2+\eta^2)`, populations relaxing with :math:`\gamma` toward the
      bare occupations. Its off-resonant tail :math:`2\eta/\gamma x^2` is the dressing of the Floquet state times
      :math:`2\eta/\gamma`; only :math:`\gamma = 2\eta` (adiabatic switching) cancels the principal-value term of the
      coherences, as the absence of a DC current below the gap requires;
    * ``'hybrid'``: :math:`K = (2\pi/\gamma)\,\delta(x) + \mathrm{Re}[(x+i\eta)^{-2}]`: real (absorbed) populations
      with the physical :math:`\tau = \hbar/\gamma` and a short-tailed (Gaussian) :math:`\delta` of width ``widths``,
      virtual ones with the adiabatic weight, :math:`\mathrm{Re}[(x+i\eta)^{-2}] = (x^2-\eta^2)/(x^2+\eta^2)^2`, which
      tends to :math:`1/x^2` and carries no :math:`\delta` weight.

    Parameters
    ----------
    x : ndarray
        Detuning :math:`\hbar\omega \mp \varepsilon_{mn}` (eV).
    widths : ndarray
        Gaussian widths of the real-population :math:`\delta` (eV), broadcast with ``x``.
    gamma : ndarray
        Population relaxation :math:`\gamma` of the band ``n`` (eV), broadcast with ``x``.
    eta : float
        Interband dephasing (eV).
    model : {'hybrid', 'relaxation'}, optional

    Returns
    -------
    ndarray
        :math:`K(x)`, shaped as ``x``.
    """
    eta = float(eta)
    if model == 'relaxation':
        return (2.0 * eta / gamma) / (x * x + eta * eta)
    if model != 'hybrid':
        raise ValueError("model must be 'hybrid' or 'relaxation'")
    gauss = np.exp(-((x / widths) ** 2)) / (widths * np.sqrt(np.pi))
    return (2.0 * np.pi / gamma) * gauss + (x * x - eta * eta) / (x * x + eta * eta) ** 2


def second_order_block(
    geometry,
    curvature: NDArray[np.complex128],
    occupation: tuple[NDArray[np.float64], NDArray[np.float64]],
    rates: tuple[NDArray[np.float64], NDArray[np.float64]],
    energies_scissored: NDArray[np.float64],
    frequencies: tuple[float, float],
    eta: float,
    components: Sequence[tuple[int, int, int]],
    k_weights: NDArray[np.float64] | None = None,
    populations: str = 'hybrid',
    resonance_widths: NDArray[np.float64] | None = None,
    coherence_damping: float = 1.0,
) -> dict[str, NDArray[np.complex128]]:
    r""":math:`\sum_k {\rm Tr}[v^a\rho^{(2)}]` per channel for a k-block, both field orders summed.

    Parameters
    ----------
    geometry : BandGeometry
        Of :func:`~PAOFLOW.response.nonlinear_optics.band_geometry` (``position``,
        ``position_derivative``, ``velocity_intra``, ``pair_mask``).
    curvature : ndarray ``(nb, 3, 3, n, n)``
        Of :func:`band_curvature`: ``[a, c]`` -> :math:`(v^a)_{;c}` inside the clusters.
    occupation : tuple
        ``(f, df/de)``, each ``(nb, n)`` (dimensionless, eV^-1).
    rates : tuple
        ``(gamma, dgamma/de)``, each ``(nb, n)`` (eV, dimensionless).
    energies_scissored : ndarray ``(nb, n)``
        Energies in the resonance denominators (the scissor applied to the conduction bands).
    frequencies : tuple of float
        :math:`(\hbar\omega_1, \hbar\omega_2)` of the two fields (eV): ``(w, w)`` for SHG,
        ``(w, -w)`` for the zero-frequency photocurrent.
    eta : float
        Interband dephasing (eV).
    components : sequence of ``(a, b, c)``
    k_weights : ndarray ``(nb,)``, optional
    populations : {'hybrid', 'relaxation'}, optional
        Zero-frequency populations driven by the interband density (the ``fermi_sea_intra`` channel):
        :func:`dc_population_kernel`. ``'hybrid'`` (default): real populations with :math:`\tau`, virtual
        ones with the adiabatic weight (no DC current below the gap, whatever :math:`\tau`);
        ``'relaxation'``: all populations relax with :math:`\gamma` (the first implementation). Other
        frequencies (SHG) always use :math:`1/(\hbar\Omega + i\gamma)`.
    resonance_widths : ndarray ``(nb, n, n)``, optional
        Gaussian widths of the real-population :math:`\delta` per pair (eV; adaptive widths); default ``eta``.
    coherence_damping : float, optional
        Damping of the second-order interband elements in units of ``eta`` (2: adiabatic switching, the
        :math:`\omega\to\omega+i\eta` of every frequency, as the velocity-gauge Kubo formula and the
        convergent SHG form; 1: one dephasing rate for every coherence).

    Returns
    -------
    dict
        Channel of :data:`CHANNELS` -> ``(ncomp,)`` complex: :math:`X^{abc} = \sum_k w_k
        {\rm Tr}[v^a\rho^{(2)}(b\,{\rm at}\,\omega_1, c\,{\rm at}\,\omega_2)] + (b, \omega_1
        \leftrightarrow c, \omega_2)` in eV bohr per (eV/bohr)^2 (field factors :math:`eE = 1` eV/bohr).

    Notes
    -----
    With :math:`\rho^{(2)} = g\circ S`, :math:`g_{nm} = 1/(\hbar\Omega - \varepsilon_{nm} + i\Gamma_{nm})`
    and :math:`S = [r_i^c, \rho^{(1)}] + i\rho^{(1)}_{;c}`, the derivative is moved onto the current
    vertex by parts (:math:`\sum_k{\rm Tr}[A\,O_{;c}] = -\sum_k{\rm Tr}[A_{;c}O]`):

    .. math::

        \sum_k{\rm Tr}[v^a\rho^{(2)}] = \sum_k\sum_{nm}\Big(A_{nm}[r_i^c, \rho^{(1)}]_{nm}
        - i\,(A_{;c})_{nm}\rho^{(1)}_{nm}\Big),\qquad A_{nm} = v^a_{mn}g_{nm},

    .. math::

        (A_{;c})_{nm} = (v^a_{;c})_{mn}\,g_{nm} + v^a_{mn}\big(\Delta^c_{nm} - i\,\partial_c\Gamma_{nm}\big)g_{nm}^2,
        \qquad v^a_{;c} = i[v^c_{\rm intra}, r^a] + i\,\varepsilon'\circ r^a_{;c} + M^{ac},

    so the resonant first-order denominators stay simple poles (the double pole sits on
    :math:`g`, off resonance at zero frequency) and the Fermi-surface terms need only :math:`f'`.
    The channels follow the first-order density they come from: the interband one feeds
    ``fermi_sea_inter`` (interband elements of :math:`\rho^{(2)}`) and ``fermi_sea_intra``
    (populations); the intraband one feeds ``fermi_surface_mixed`` (through the interband commutator)
    and ``fermi_surface_drude`` (through the intraband vertex).
    """
    E = energies_scissored
    nb = E.shape[0]
    mask = geometry.pair_mask
    intra = ~mask
    f, fp = occupation[0], occupation[1]
    gamma, dgamma = rates
    om = E[:, :, None] - E[:, None, :]  # omega'_nm
    fnm = f[:, :, None] - f[:, None, :]
    fp_pair = 0.5 * (fp[:, :, None] + fp[:, None, :]) * intra
    g_pair = 0.5 * (gamma[:, :, None] + gamma[:, None, :])
    dg_pair = 0.5 * (dgamma[:, :, None] + dgamma[:, None, :]) * intra
    damping = np.where(mask, float(eta), g_pair)
    r = geometry.position  # (nb, 3, n, n), between clusters
    rd = geometry.position_derivative  # [c, a] -> r^a_{;c}
    vi = geometry.velocity_intra  # (nb, 3, n, n), inside clusters
    # scissor-consistent velocity: v_nm = i omega'_nm r_nm between clusters, the band velocities inside
    v = 1j * om[:, None] * r + vi
    fermi_surface = bool(np.any(fp_pair))
    Df = [fp_pair * vi[:, b] for b in range(3)]
    weights = np.ones(nb) if k_weights is None else np.asarray(k_weights, dtype=float)

    def comm(x, y):
        return x @ y - y @ x

    damping2 = np.where(mask, float(coherence_damping) * float(eta), g_pair)

    def vertex(a: int, c: int, W: float) -> tuple:
        """``A`` and ``A_{;c}`` indexed ``[k, n, m]`` (to multiply the ``nm`` elements)."""
        g = 1.0 / (W - om + 1j * damping2)
        va_d = (
            1j * comm(vi[:, c], r[:, a]) + 1j * om * rd[:, c, a] + curvature[:, a, c]
        )  # (v^a)_{;c}
        dgv = dg_pair * vi[:, c]
        shift = (
            comm(v[:, a], vi[:, c]) - 0.5j * (dgv @ v[:, a] + v[:, a] @ dgv) * intra
        )  # v^a_mn (D - i dG)
        A = np.swapaxes(v[:, a], 1, 2) * g
        A_d = np.swapaxes(va_d, 1, 2) * g + np.swapaxes(shift, 1, 2) * g * g
        return A, A_d

    def first_order(b: int, w: float) -> tuple:
        den = w - om + 1j * damping
        rho_inter = np.where(mask, -fnm * r[:, b] / den, 0.0)
        rho_intra = 1j * Df[b] / den if fermi_surface else None
        return rho_inter, rho_intra

    def wsum(x: NDArray[np.complex128]) -> complex:
        return complex(np.einsum('k,knm->', weights, x))

    out = {ch: np.zeros(len(components), dtype=complex) for ch in CHANNELS}
    w1, w2 = (float(x) for x in frequencies)
    W = w1 + w2
    if populations not in ('hybrid', 'relaxation'):
        raise ValueError("populations must be 'hybrid' or 'relaxation'")
    pair_populations = populations == 'hybrid' and abs(W) < 1.0e-12
    if pair_populations:
        widths = (
            np.full(om.shape, float(eta))
            if resonance_widths is None
            else np.asarray(resonance_widths, dtype=float)
        )
        g_row = gamma[:, :, None]
        # x = w - eps_mn = w + omega'_nm (first term), w + eps_mn (second), between clusters only
        Kx = fnm * dc_population_kernel(w1 + om, widths, g_row, eta) * mask
        Ky = fnm * dc_population_kernel(w1 - om, widths, g_row, eta) * mask
    firsts, vertices = {}, {}
    for ic, (a, b, c) in enumerate(components):
        if pair_populations:
            rho2 = -((r[:, c] * Kx) @ r[:, b] + (r[:, b] * Ky) @ r[:, c]) * intra
            out['fermi_sea_intra'][ic] += wsum(np.swapaxes(v[:, a], 1, 2) * rho2)
        for (bb, wb), (cc, _) in (((b, w1), (c, w2)), ((c, w2), (b, w1))):
            if (bb, wb) not in firsts:
                firsts[(bb, wb)] = first_order(bb, wb)
            if (a, cc) not in vertices:
                vertices[(a, cc)] = vertex(a, cc, W)
            rho_inter, rho_intra = firsts[(bb, wb)]
            A, A_d = vertices[(a, cc)]
            s_inter = A * comm(r[:, cc], rho_inter)
            out['fermi_sea_inter'][ic] += wsum(s_inter * mask - 1j * A_d * rho_inter)
            if not pair_populations:
                out['fermi_sea_intra'][ic] += wsum(s_inter * intra)
            if rho_intra is not None:
                out['fermi_surface_mixed'][ic] += wsum(A * comm(r[:, cc], rho_intra))
                out['fermi_surface_drude'][ic] += wsum(-1j * A_d * rho_intra)
    return out


def channel_band_edges(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    orbital_positions_cryst: NDArray[np.float64] | None,
    nelec: float,
    spin_factor: float = 2.0,
    ispin: int = 0,
    hopping_cutoff: float | None = None,
    refine: int = 2,
    block: int = 256,
) -> dict:
    """Occupied-band count and band edges of a spin channel, on a grid finer than that of ``HRs``.

    Parameters
    ----------
    HRs, a_vectors, alat, orbital_positions_cryst, nelec
        As :func:`density_matrix_spectra`.
    spin_factor : float, optional
    ispin : int, optional
    hopping_cutoff : float, optional
    refine : int, optional
        The edges are taken on the coarse grid of ``HRs`` times ``refine`` per axis (Fourier
        interpolated): a semimetal's pockets can fall between the coarse k-points (TaAs on 10^3).
    block : int, optional
        k-points per diagonalisation block (memory ``block * nawf^2`` complex numbers).

    Returns
    -------
    dict
        ``'nocc'`` (bands below the band-index cut), ``'vbm'``, ``'cbm'`` (top of band ``nocc`` and
        bottom of band ``nocc + 1``, eV), ``'direct_gap'`` (eV), ``'integer_filling'`` and ``'metallic'``
        (the two bands overlap or the filling is not an integer).
    """
    from ..hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
    from .nonlinear_optics import _k_grid

    terms = bloch_terms_from_HRs(
        HRs, a_vectors, alat, orbital_positions_cryst, ispin=ispin, hopping_cutoff=hopping_cutoff
    )
    n = terms.nawf
    nocc_float = float(nelec) / float(spin_factor)
    integer_filling = abs(nocc_float - round(nocc_float)) <= 1.0e-6
    nocc = int(round(nocc_float)) if integer_filling else int(np.floor(nocc_float))
    nocc = min(max(nocc, 1), n - 1)
    grid = tuple(int(refine) * int(x) for x in np.asarray(HRs).shape[2:5])
    K = _k_grid(grid)
    # eigenvalues only, in blocks: H and its nine derivatives on the whole refined grid took
    # 22 GB per rank for TaAs (116 orbitals, 20^3) and exhausted the Mac (12 Nov 2026)
    vbm, cbm, direct = -np.inf, np.inf, np.inf
    for b0 in range(0, K.shape[0], int(block)):
        e = np.linalg.eigvalsh(hamiltonian_block(terms, K[b0 : b0 + int(block)]))
        vbm = max(vbm, float(e[:, nocc - 1].max()))
        cbm = min(cbm, float(e[:, nocc].min()))
        direct = min(direct, float(np.min(e[:, nocc] - e[:, nocc - 1])))
    return {
        'nocc': nocc,
        'vbm': vbm,
        'cbm': cbm,
        'direct_gap': direct,
        'integer_filling': integer_filling,
        'metallic': bool((cbm <= vbm) or not integer_filling),
    }


def density_matrix_spectra(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    cell_volume: float,
    orbital_positions_cryst: NDArray[np.float64] | None,
    nelec: float,
    nk: Sequence[int],
    energy_grid: NDArray[np.float64],
    *,
    response: str = 'shg',
    components='all',
    eta: float = 0.05,
    relaxation_time: float | Callable | None = None,
    fermi_energy: float | None = None,
    temperature: float = 0.0,
    scissor: float = 0.0,
    degeneracy_tol: float = 1.0e-2,
    spin_factor: float = 2.0,
    ispin: int = 0,
    kblock: int = 64,
    hopping_cutoff: float | None = None,
    position_dipoles: Callable[[NDArray[np.float64]], NDArray[np.complex128]] | None = None,
    dipole_step: float = 1.0e-4,
    symmetry_ops_cryst: NDArray[np.float64] | None = None,
    comm=None,
    progress: bool = False,
    populations: str = 'hybrid',
    adaptive_smearing: float | None = None,
    coherence_damping: float = 1.0,
) -> dict:
    r"""Second-harmonic susceptibility or zero-frequency photocurrent of ``HRs`` from the equations of motion.

    Parameters
    ----------
    HRs, a_vectors, alat, cell_volume, orbital_positions_cryst, nelec, nk, energy_grid
        As :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra`.
    response : {'shg', 'dc'}, optional
        :math:`\chi^{(2)}(-2\omega;\omega,\omega)` in pm/V, or :math:`\sigma^{(2)}(0;\omega,-\omega)` in
        A/V^2 (:math:`J^a = \sigma^{abc}E^b(\omega)E^c(-\omega)`, complex: the real part for linear and the
        imaginary part for circular polarisation).
    components : 'all', 'diag' or sequence
    eta : float, optional
        Interband dephasing (eV).
    relaxation_time : float, callable or None, optional
        Intraband :math:`\tau` (fs), constant or a function of the band energy (eV -> fs); ``None``
        uses :math:`\gamma = \eta`.
    fermi_energy : float, optional
        Required for a metallic channel (0.0: PAOFLOW Hamiltonians are referred to the Fermi level).
    temperature : float, optional
        Electronic temperature (K); the Fermi-surface channels need it (they vanish at 0 K).
    scissor, degeneracy_tol, spin_factor, ispin, kblock, hopping_cutoff, position_dipoles,
    dipole_step, symmetry_ops_cryst, comm, progress
        As :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra`.

    populations : {'hybrid', 'relaxation'}, optional
        Zero-frequency populations driven by the interband density (:func:`dc_population_kernel`):
        ``'hybrid'`` (default) gives the real (absorbed) populations the relaxation time and the virtual
        ones (the off-resonant dressing of the Floquet state) the adiabatic weight, so that a clean
        insulator below its gap carries no DC current whatever :math:`\tau`; ``'relaxation'`` relaxes all of
        them with :math:`\gamma` (the first implementation; a spurious off-resonant current
        :math:`\propto \eta/\gamma - 1/2`).
    adaptive_smearing : float, optional
        Yates prefactor ``a`` of per-pair Gaussian widths :math:`a|\mathbf v_n - \mathbf v_m|\Delta k`,
        floored at ``eta``, for the real populations of ``'hybrid'`` (as the injection current of
        :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra`).
    coherence_damping : float, optional
        Second-order interband damping in units of ``eta`` (:func:`second_order_block`).

    Returns
    -------
    dict
        ``'energy_ev'``, ``'components'``, ``'response'`` and its key (``'shg'`` or ``'dc'``,
        ``(ncomp, ne)`` complex), ``'channels'`` (:data:`CHANNELS` -> ``(ncomp, ne)`` complex in the same
        units, summing to the total), ``'metallic'``, ``'eta_ev'``, ``'relaxation_time'``, ``'nk'``.

    Raises
    ------
    ValueError
        For a metallic channel without ``fermi_energy``, a scissor on a metal, or an unknown response.
    """
    from ..hamiltonian.bloch_derivatives import (
        bloch_terms_from_HRs,
        hamiltonian_derivatives_block,
        velocity_and_curvature_operators,
    )
    from ..utils.communication import load_balancing
    from ..utils.point_groups import (
        cartesian_from_fractional_k,
        component_closure,
        reduce_k_grid,
        symmetrize_components,
    )
    from .nonlinear_optics import _k_grid

    if response not in ('shg', 'dc'):
        raise ValueError("response must be 'shg' or 'dc', got %r" % (response,))
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    comps = parse_components(components)
    energy_grid = np.asarray(energy_grid, dtype=float)
    terms = bloch_terms_from_HRs(
        HRs, a_vectors, alat, orbital_positions_cryst, ispin=ispin, hopping_cutoff=hopping_cutoff
    )
    n = terms.nawf
    nk = tuple(int(x) for x in nk)
    nkd = int(np.prod(nk))
    # occupations: band index for insulators, energies at the Fermi level for metals
    edges = channel_band_edges(
        HRs, a_vectors, alat, orbital_positions_cryst, nelec, spin_factor, ispin, hopping_cutoff
    )
    nocc, vbm, cbm, metallic = edges['nocc'], edges['vbm'], edges['cbm'], edges['metallic']
    if metallic and fermi_energy is None:
        raise ValueError(
            'metallic channel (band %d reaches %.3f eV, band %d starts at %.3f eV): pass fermi_energy '
            '(0.0 for PAOFLOW Hamiltonians) and a temperature' % (nocc, vbm, nocc + 1, cbm)
        )
    if metallic and float(scissor) != 0.0:
        raise ValueError('a scissor needs a gap: the channel is metallic (set scissor=0)')
    mu = fermi_energy
    if mu is None and float(temperature) > 0.0:
        mu = 0.5 * (vbm + cbm)  # an insulator at finite temperature: the mid-gap
    conduction = (np.arange(n) >= nocc).astype(float)
    K = _k_grid(nk)
    spacing = (8.0 * np.pi**3 / (float(cell_volume) * nkd)) ** (1.0 / 3.0)  # 1/bohr
    requested, k_weights_all, ops_cart = list(comps), None, None
    if symmetry_ops_cryst is not None:
        ops_frac = np.rint(np.asarray(symmetry_ops_cryst, dtype=float))
        bg = np.linalg.inv(np.asarray(a_vectors, dtype=float)).T
        ops_cart = cartesian_from_fractional_k(ops_frac, bg)
        comps = component_closure(requested, ops_cart)
        K, k_weights_all, _ = reduce_k_grid(nk, ops_frac)
    kstart, kstop = load_balancing(size, rank, K.shape[0])
    ne, ncomp = energy_grid.size, len(comps)
    acc = {ch: np.zeros((ncomp, ne), dtype=complex) for ch in CHANNELS}
    nblocks = max(1, -(-(kstop - kstart) // int(kblock)))
    for ib, b0 in enumerate(range(kstart, kstop, int(kblock))):
        if progress and rank == 0 and (ib % max(1, nblocks // 10) == 0 or ib == nblocks - 1):
            print('  density-matrix response: k-block %d of %d' % (ib + 1, nblocks), flush=True)
        kb = K[b0 : min(b0 + int(kblock), kstop)]
        kw = None if k_weights_all is None else k_weights_all[b0 : min(b0 + int(kblock), kstop)]
        H, dH, d2H = hamiltonian_derivatives_block(terms, kb)
        v_orb, w_orb = velocity_and_curvature_operators(terms, H, dH, d2H)
        E, U = np.linalg.eigh(H)
        if position_dipoles is not None:
            d_orb = np.asarray(position_dipoles(kb), dtype=complex)
            dg_orb = dipole_gradient_by_differences(
                position_dipoles, kb, a_vectors, alat, dipole_step
            )
            dg_orb += 1j * terms.separation_bohr[None, :, None] * d_orb[:, None]
            geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol, d_orb, dg_orb)
        else:
            geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol)
        curvature = band_curvature(U, geometry.omega, geometry.pair_mask, v_orb, w_orb)
        if mu is not None:
            occupation = fermi_dirac_derivatives(E, float(mu), float(temperature))[:2]
        else:
            f = np.zeros_like(E)
            f[:, :nocc] = 1.0
            occupation = (f, np.zeros_like(E))
        rates = intraband_rates(E, relaxation_time, float(eta))
        widths = None
        if adaptive_smearing:
            dvel = np.sqrt(np.sum(geometry.delta_velocity**2, axis=1))  # (nb, n, n) |v_n - v_m|
            widths = np.maximum(float(adaptive_smearing) * dvel * spacing, float(eta))
        E_s = E + float(scissor) * conduction[None, :]
        for ie, w in enumerate(energy_grid):
            freqs = (w, w) if response == 'shg' else (w, -w)
            block = second_order_block(
                geometry, curvature, occupation, rates, E_s, freqs, float(eta), comps, kw,
                populations=populations, resonance_widths=widths,
                coherence_damping=coherence_damping,
            )  # fmt: skip
            for ch in CHANNELS:
                acc[ch][:, ie] += block[ch]
    if comm is not None and size > 1:
        from mpi4py import MPI

        from ..utils.communication import allreduce_inplace

        for array in acc.values():
            allreduce_inplace(array, MPI.SUM, communicator=comm)
    if ops_cart is not None:
        for ch in CHANNELS:
            acc[ch] = symmetrize_components(acc[ch], comps, ops_cart, requested)
        comps = requested
    # the charge enters as e^3 with e > 0, the convention of nonlinear_optics (electrons: times -1)
    norm = float(spin_factor) / (nkd * float(cell_volume))
    if response == 'shg':
        # chi = P/(eps0 E^2), P = J/(-2i w), J = e Tr[v rho]; the two field orders are averaged
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = 1.0e12 * ELECTRON_CHARGE_SI / EPSILON0_SI * norm * 0.5 / (-2.0j * energy_grid)
        channels = {ch: factor[None, :] * acc[ch] for ch in CHANNELS}
    else:
        # J = e Tr[v rho]/hbar: A/V^2 per (V/bohr)^2 with the field factors eE = 1 eV/bohr
        factor = ELECTRON_CHARGE_SI / HBAR * norm
        channels = {ch: factor * acc[ch] for ch in CHANNELS}
    total = sum(channels.values())
    return {
        'energy_ev': energy_grid,
        'components': [component_label(cmp) for cmp in comps],
        'response': response,
        response: total,
        'channels': channels,
        'metallic': bool(metallic),
        'nocc': int(nocc),
        'gap_ev': cbm - vbm,
        'direct_gap_ev': edges['direct_gap'],
        'scissor_ev': float(scissor),
        'spin_factor': float(spin_factor),
        'symmetry': (
            {'operations': int(ops_cart.shape[0]), 'k_irreducible': int(K.shape[0]), 'k_full': nkd}
            if ops_cart is not None
            else None
        ),
        'fermi_energy_ev': mu,
        'temperature_k': float(temperature),
        'eta_ev': float(eta),
        'relaxation_time': relaxation_time,
        'populations': populations,
        'adaptive_smearing': adaptive_smearing,
        'nk': nk,
    }


__all__ = [
    'CHANNELS',
    'band_curvature',
    'channel_band_edges',
    'dc_population_kernel',
    'density_matrix_spectra',
    'fermi_dirac_derivatives',
    'intraband_rates',
    'second_order_block',
]
