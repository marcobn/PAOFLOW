"""Second-order optical responses of the PAO Hamiltonian: shift and injection currents, SHG.

The bulk photovoltaic (shift and injection) currents and the second-harmonic susceptibility are
evaluated from the band energies, the interband position matrix
:math:`r^a_{nm} = -i\\,v^a_{nm}/\\omega_{nm}` (:math:`n\\neq m`), its generalized derivative
:math:`r^b_{nm;a}` and the velocity differences :math:`\\Delta^a_{nm} = v^a_{nn}-v^a_{mm}`, all
built at arbitrary k from ``HRs`` with :mod:`PAOFLOW.hamiltonian.bloch_derivatives`.  The PAO basis
is finite, so the sum rule for :math:`r^b_{nm;a}` over all ``nawf`` bands is exact within the
tight-binding approximation (orbital centres as the position operator):

.. math::

    r^b_{nm;a} = \\frac{i}{\\omega_{nm}}\\Big[\\frac{v^b_{nm}\\Delta^a_{nm}}{\\omega_{nm}} - w^{ab}_{nm}
    + \\sum_{p\\neq n,m}\\Big(\\frac{v^b_{np}v^a_{pm}}{\\omega_{pm}} - \\frac{v^a_{np}v^b_{pm}}{\\omega_{np}}\\Big)\\Big],

with :math:`w^{ab}_{nm} = \\langle n|\\partial_a\\partial_b H|m\\rangle`.  Conventions (``hbar = 1``
inside the kernels; energies in eV, lengths in bohr; SI prefactors applied once at the end):

* **shift current** (normal, T-even), :math:`J^a = \\sigma^{abc}(\\omega)E^b(\\omega)E^c(-\\omega)`
  for a field :math:`E(t) = E(\\omega)e^{-i\\omega t} + E(-\\omega)e^{i\\omega t}`:

  .. math::

      \\sigma^{abc}(\\omega) = \\frac{\\pi e^3 g_s}{\\hbar^2}\\int\\!\\frac{d^3k}{(2\\pi)^3}\\sum_{nm}
      f_{nm}\\,\\mathrm{Im}\\big[r^b_{mn}r^c_{nm;a} + r^c_{mn}r^b_{nm;a}\\big]\\,\\delta(\\omega_{nm}-\\omega),

  which for :math:`b = c` is Eq. (2) of Wang *et al.* (2017) and twice the
  :math:`\\sigma^{abb}` of Sipe and Shkrebtii (2000), who write :math:`J = 2\\sigma EE^*`;
* **magnetic shift current** (T-odd), :math:`J^a = \\sigma_M^{abc}\\,\\mathrm{Im}[E^b(\\omega)E^c(-\\omega)]`,
  :math:`\\sigma_M^{abc} = -\\frac{\\pi e^3 g_s}{\\hbar^2}\\int\\sum f_{nm}\\mathrm{Re}[r^b_{mn}r^c_{nm;a}
  - r^c_{mn}r^b_{nm;a}]\\delta(\\omega_{nm}-\\omega)`;
* **injection currents**, :math:`dJ^a/dt = \\eta_M^{abc}\\,\\mathrm{Re}[E^bE^{c*}] + \\eta_C^{abc}\\,
  \\mathrm{Im}[E^bE^{c*}]` with the circular (T-even, two-band Berry curvature) and magnetic (T-odd,
  quantum metric) coefficients

  .. math::

      \\eta_C^{abc} = \\frac{2\\pi e^3 g_s}{\\hbar^2}\\int\\sum_{nm} f_{mn}\\Delta^a_{mn}\\,
      \\mathrm{Im}[r^b_{mn}r^c_{nm}]\\,\\delta(\\omega_{nm}-\\omega),\\qquad
      \\eta_M^{abc} = -\\frac{2\\pi e^3 g_s}{\\hbar^2}\\int\\sum_{nm} f_{mn}\\Delta^a_{mn}\\,
      \\mathrm{Re}[r^b_{mn}r^c_{nm}]\\,\\delta(\\omega_{nm}-\\omega)

  (Chen *et al.* 2022, Eqs. 9, 12, 13);
* **SHG**, :math:`P^a(2\\omega) = \\epsilon_0\\chi^{abc}E^b(\\omega)E^c(\\omega)`, the low-frequency
  convergent form of Chen *et al.* (2022), Eqs. (C1)-(C3) (Rashkeev 1998 / Wang 2017 with the
  T-odd term), with :math:`\\omega\\to\\omega+i\\eta` in every resonance denominator; the resonant
  factors are accumulated as transition-energy histograms and convolved once.

A scissor shifts the conduction energies in every resonance and explicit :math:`1/\\omega_{mn}`
factor while :math:`r`, :math:`r_{;a}` and :math:`\\Delta` keep the DFT eigenvectors.  Pairs
closer than ``degeneracy_tol`` are excluded from :math:`r` and from the band sums.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..hamiltonian.bloch_derivatives import (
    bloch_terms_from_HRs,
    hamiltonian_derivatives_block,
    velocity_and_curvature_operators,
)
from ..utils.constants import ELECTRON_CHARGE_SI, EPSILON0_SI, HBAR_SI

KELVIN_TO_EV = 8.617333262e-5
#: shift current per unit of the dimensionless k-sum: pi e^2 / hbar (A/V^2)
C_SHIFT = np.pi * ELECTRON_CHARGE_SI**2 / HBAR_SI
#: injection current per unit: 2 pi e^3 / hbar^2 (A V^-2 s^-1)
C_INJECT = 2.0 * np.pi * ELECTRON_CHARGE_SI**3 / HBAR_SI**2
#: SHG susceptibility per unit: e / (2 eps0), in pm/V
C_SHG = 1.0e12 * ELECTRON_CHARGE_SI / (2.0 * EPSILON0_SI)
LABELS = 'xyz'
RESPONSES = ('shift', 'injection', 'shg')


# --------------------------------------------------------------------------- components
def parse_components(components) -> list[tuple[int, int, int]]:
    """Tensor components ``(a, b, c)`` from ``'all'``, ``'diag'``, strings (``'xyz'``) or triples."""
    if isinstance(components, str):
        if components == 'all':
            return [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        if components == 'diag':
            return [(a, a, a) for a in range(3)]
        components = [components]
    out = []
    for item in components:
        if isinstance(item, str):
            if len(item) != 3 or any(ch not in LABELS for ch in item):
                raise ValueError('component %r must be three of x, y, z' % item)
            out.append(tuple(LABELS.index(ch) for ch in item))
        else:
            trip = tuple(int(i) for i in item)
            if len(trip) != 3 or any(i < 0 or i > 2 for i in trip):
                raise ValueError('component %r must be three indices in 0..2' % (item,))
            out.append(trip)
    return out


def component_label(abc: Sequence[int]) -> str:
    """``(0, 1, 2) -> 'xyz'``."""
    return ''.join(LABELS[i] for i in abc)


# --------------------------------------------------------------------------- band geometry
@dataclass
class BandGeometry:
    """Band-basis quantities of a block of k-points (``hbar = 1``, eV and bohr).

    Attributes
    ----------
    energies : ndarray ``(nb, n)``
    eigenvectors : ndarray ``(nb, n, n)``
    omega : ndarray ``(nb, n, n)``
        :math:`\\omega_{nm} = E_n - E_m`.
    pair_mask : ndarray ``(nb, n, n)``, bool
        Pairs with :math:`|\\omega_{nm}|` above the degeneracy tolerance (diagonal excluded).
    inv_omega : ndarray ``(nb, n, n)``
        :math:`1/\\omega_{nm}` on the mask, 0 elsewhere.
    velocity : ndarray ``(nb, 3, n, n)``
        :math:`v^a_{nm}` (eV bohr).
    delta_velocity : ndarray ``(nb, 3, n, n)``
        :math:`\\Delta^a_{nm} = v^a_{nn} - v^a_{mm}` (real).
    position : ndarray ``(nb, 3, n, n)``
        :math:`r^a_{nm}` (bohr), zero on the diagonal and inside degenerate pairs.
    position_derivative : ndarray ``(nb, 3, 3, n, n)``
        ``[a, b]`` -> :math:`r^b_{nm;a}` (bohr^2).
    """

    energies: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    omega: NDArray[np.float64]
    pair_mask: NDArray[np.bool_]
    inv_omega: NDArray[np.float64]
    velocity: NDArray[np.complex128]
    delta_velocity: NDArray[np.float64]
    position: NDArray[np.complex128]
    position_derivative: NDArray[np.complex128]


def band_geometry(
    energies: NDArray[np.float64],
    eigenvectors: NDArray[np.complex128],
    velocity_orb: NDArray[np.complex128],
    curvature_orb: NDArray[np.complex128],
    degeneracy_tol: float = 1.0e-4,
) -> BandGeometry:
    """Rotate the orbital operators to the band basis and build :math:`r` and :math:`r_{;a}`.

    Parameters
    ----------
    energies : ndarray ``(nb, n)``
    eigenvectors : ndarray ``(nb, n, n)``
        Columns.
    velocity_orb : ndarray ``(nb, 3, n, n)``
        :math:`v_a` of :func:`~PAOFLOW.hamiltonian.bloch_derivatives.velocity_and_curvature_operators`.
    curvature_orb : ndarray ``(nb, 3, 3, n, n)``
        :math:`w_{ab}`.
    degeneracy_tol : float, optional
        Pairs with :math:`|\\omega_{nm}|` below it are treated as degenerate (eV).

    Returns
    -------
    BandGeometry
    """
    U = eigenvectors
    v = np.einsum('kim,kaij,kjn->kamn', U.conj(), velocity_orb, U, optimize=True)
    w = np.einsum('kim,kabij,kjn->kabmn', U.conj(), curvature_orb, U, optimize=True)
    omega = energies[:, :, None] - energies[:, None, :]
    mask = np.abs(omega) > float(degeneracy_tol)
    inv = np.where(mask, 1.0 / np.where(mask, omega, 1.0), 0.0)
    r = -1j * v * inv[:, None]
    diag = np.real(np.diagonal(v, axis1=2, axis2=3))  # (nb, 3, n)
    delta = diag[:, :, :, None] - diag[:, :, None, :]
    G = v * inv[:, None]  # G^a_pm = v^a_pm / omega_pm (0 on degenerate pairs)
    nb, n = energies.shape
    rd = np.empty((nb, 3, 3, n, n), dtype=complex)
    for a in range(3):
        for b in range(3):
            commutator = v[:, b] @ G[:, a] - G[:, a] @ v[:, b]
            rd[:, a, b] = 1j * inv * (v[:, b] * delta[:, a] * inv - w[:, a, b] + commutator)
    rd *= mask[:, None, None]
    return BandGeometry(
        energies=energies,
        eigenvectors=U,
        omega=omega,
        pair_mask=mask,
        inv_omega=inv,
        velocity=v,
        delta_velocity=delta,
        position=r,
        position_derivative=rd,
    )


def occupations(
    energies: NDArray[np.float64],
    nocc: int,
    temperature_k: float = 0.0,
    fermi_energy: float | None = None,
) -> NDArray[np.float64]:
    """Band occupations: a step by band index at 0 K (or by energy if ``fermi_energy`` is given), else Fermi-Dirac."""
    e = np.asarray(energies, dtype=float)
    if temperature_k <= 0.0:
        if fermi_energy is None:
            f = np.zeros_like(e)
            f[:, : int(nocc)] = 1.0
            return f
        return (e < float(fermi_energy)).astype(float)
    if fermi_energy is None:
        raise ValueError('a finite temperature needs the Fermi energy')
    x = (e - float(fermi_energy)) / (KELVIN_TO_EV * float(temperature_k))
    return 0.5 * (1.0 - np.tanh(0.5 * np.clip(x, -200.0, 200.0)))


def delta_kernel(x: NDArray[np.float64], width, smearing: str = 'gauss') -> NDArray[np.float64]:
    """Normalised :math:`\\delta(x)` of width ``width``: PAOFLOW's Gaussian or a Lorentzian."""
    width = np.asarray(width, dtype=float)
    if smearing == 'gauss':
        return np.exp(-((x / width) ** 2)) / (width * np.sqrt(np.pi))
    if smearing == 'lorentz':
        return (width / np.pi) / (x * x + width * width)
    raise ValueError("smearing must be 'gauss' or 'lorentz'")


# --------------------------------------------------------------------------- kernels
def accumulate_shift_and_injection(
    geometry: BandGeometry,
    occupation: NDArray[np.float64],
    omega_scissored: NDArray[np.float64],
    energy_grid: NDArray[np.float64],
    widths: NDArray[np.float64],
    components: Sequence[tuple[int, int, int]],
    smearing: str,
    out: dict,
) -> None:
    """Add the :math:`\\delta`-function sums of a k-block to ``out`` (unnormalised).

    Parameters
    ----------
    geometry : BandGeometry
    occupation : ndarray ``(nb, n)``
    omega_scissored : ndarray ``(nb, n, n)``
        :math:`\\omega'_{nm}` with the scissor.
    energy_grid : ndarray ``(ne,)``
    widths : ndarray ``(nb, n, n)``
        Width of :math:`\\delta(\\omega'_{nm}-\\omega)` per pair (eV).
    components : sequence of ``(a, b, c)``
    smearing : {'gauss', 'lorentz'}
    out : dict
        Accumulators ``'shift'``, ``'shiftm'``, ``'injectc'``, ``'injectm'`` of shape
        ``(ncomp, ne)``.
    """
    F = occupation[:, None, :] - occupation[:, :, None]  # f_m - f_n  (n: first band index)
    span = 5.0 * float(np.max(widths))
    select = (
        geometry.pair_mask
        & (omega_scissored > 0.0)
        & (F > 1.0e-10)
        & (omega_scissored > energy_grid[0] - span)
        & (omega_scissored < energy_grid[-1] + span)
    )
    ik_all, n_all, m_all = np.nonzero(select)
    if ik_all.size == 0:
        return
    r, rd, dv = geometry.position, geometry.position_derivative, geometry.delta_velocity
    chunk = max(1, int(2.0e6 // max(energy_grid.size, 1)))  # pairs per kernel block (~16 MB)
    for start in range(0, ik_all.size, chunk):
        sl = slice(start, start + chunk)
        ik, n, m = ik_all[sl], n_all[sl], m_all[sl]
        kernel = delta_kernel(
            energy_grid[None, :] - omega_scissored[ik, n, m][:, None],
            widths[ik, n, m][:, None],
            smearing,
        )  # (P, ne)
        f = F[ik, n, m]
        for ic, (a, b, c) in enumerate(components):
            rb_mn, rc_mn = r[ik, b, m, n], r[ik, c, m, n]
            rc_nm = r[ik, c, n, m]
            rdc, rdb = rd[ik, a, c, n, m], rd[ik, a, b, n, m]  # r^c_{nm;a}, r^b_{nm;a}
            shift = np.imag(rb_mn * rdc + rc_mn * rdb)
            shift_m = np.real(rb_mn * rdc - rc_mn * rdb)
            inject = dv[ik, a, m, n] * rb_mn * rc_nm  # Delta^a_mn r^b_mn r^c_nm
            out['shift'][ic] += (-f * shift) @ kernel
            out['shiftm'][ic] += (f * shift_m) @ kernel
            out['injectc'][ic] += (f * inject.imag) @ kernel
            out['injectm'][ic] += (-f * inject.real) @ kernel


def _bincount_complex(
    energy: NDArray[np.float64],
    weight: NDArray[np.complex128],
    edge0: float,
    width: float,
    nbins: int,
) -> NDArray[np.complex128]:
    """Histogram of complex weights with linear (two-point) binning of the energies."""
    position = (np.broadcast_to(energy, weight.shape).ravel() - edge0) / width - 0.5
    position = np.clip(position, 0.0, nbins - 1.0)
    low = np.floor(position).astype(int)
    frac = position - low
    high = np.minimum(low + 1, nbins - 1)
    w = weight.ravel()
    out = np.bincount(low, w.real * (1.0 - frac), minlength=nbins) + 1j * np.bincount(
        low, w.imag * (1.0 - frac), minlength=nbins
    )
    out += np.bincount(high, w.real * frac, minlength=nbins) + 1j * np.bincount(
        high, w.imag * frac, minlength=nbins
    )
    return out


def accumulate_shg_histograms(
    geometry: BandGeometry,
    occupation: NDArray[np.float64],
    energies_scissored: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    components: Sequence[tuple[int, int, int]],
    out: dict,
    degeneracy_tol: float = 1.0e-4,
    chunk: int | None = None,
) -> None:
    """Add the SHG weights of a k-block to the transition-energy histograms ``out``.

    Every term of Eqs. (C2)-(C3) of Chen *et al.* (2022) has the form
    ``weight / (E - c w)`` with ``E`` a (scissored) transition energy and ``c = 1`` or ``2``; the
    weights are binned on ``bin_edges`` into ``out['hist1']`` and ``out['hist2']``
    (``(ncomp, nbins)``, complex, unnormalised) and convolved later by
    :func:`shg_from_histograms`.  Array axes are ``[k, n, m]`` (``n`` first); ``r[:, a]`` is
    :math:`r^a_{nm}`, ``rd[:, a, b]`` is :math:`r^b_{nm;a}`, ``dv[:, a]`` is
    :math:`\\Delta^a_{nm}`.

    Parameters
    ----------
    geometry : BandGeometry
    occupation : ndarray ``(nb, n)``
    energies_scissored : ndarray ``(nb, n)``
        :math:`E'_n` (the scissor applied to the conduction bands).
    bin_edges : ndarray ``(nbins + 1,)``
    components : sequence of ``(a, b, c)``
    out : dict
        ``'hist1'``, ``'hist2'``.
    degeneracy_tol : float
    chunk : int, optional
        k-points per inner chunk of the three-band term (memory).
    """
    nbins = bin_edges.size - 1
    width = bin_edges[1] - bin_edges[0]
    nb, n = geometry.energies.shape
    chunk = chunk or max(1, int(2.0e6 // max(n**3, 1)))
    index = np.arange(n)
    distinct = (
        (index[:, None, None] != index[None, :, None])
        & (index[:, None, None] != index[None, None, :])
        & (index[None, :, None] != index[None, None, :])
    )  # (n, m, l)
    for start in range(0, nb, chunk):
        sl = slice(start, start + chunk)
        E_s = energies_scissored[sl]
        f = occupation[sl]
        r = geometry.position[sl]
        rd = geometry.position_derivative[sl]
        dv = geometry.delta_velocity[sl]
        mask = geometry.pair_mask[sl]
        om_nm = E_s[:, :, None] - E_s[:, None, :]  # omega'_nm
        inv_nm = np.where(mask, 1.0 / np.where(mask, om_nm, 1.0), 0.0)
        inv_mn = -inv_nm  # 1 / omega'_mn
        fnm = f[:, :, None] - f[:, None, :]  # f_n - f_m
        om_mn = om_nm.transpose(0, 2, 1)  # [n, m] -> omega'_mn
        edge0 = float(bin_edges[0])
        # three-band denominator omega'_ln - omega'_ml = 2E'_l - E'_n - E'_m, axes (k, n, m, l)
        den = 2.0 * E_s[:, None, None, :] - E_s[:, :, None, None] - E_s[:, None, :, None]
        ok = distinct[None] & (np.abs(den) > degeneracy_tol)
        D = np.where(ok, 1.0 / np.where(ok, den, 1.0), 0.0)
        for ic, (a, b, c) in enumerate(components):
            # ---- interband term: r^a_nm (r^b_ml r^c_ln + r^c_ml r^b_ln) / (w_ln - w_ml)
            B = np.einsum('kml,kln->kmln', r[:, b], r[:, c]) + np.einsum(
                'kml,kln->kmln', r[:, c], r[:, b]
            )
            T = np.einsum('knm,kmln,knml->knml', r[:, a], B, D)
            # 2 f_nm / (w_mn - 2w)
            out['hist2'][ic] += _bincount_complex(
                om_mn[:, :, :, None], 2.0 * fnm[:, :, :, None] * T, edge0, width, nbins
            )
            # f_ln / (w_ln - w): [n, l] arrays
            out['hist1'][ic] += _bincount_complex(
                om_mn[:, :, None, :], fnm.transpose(0, 2, 1)[:, :, None, :] * T, edge0, width, nbins
            )
            # f_ml / (w_ml - w): [m, l] arrays
            out['hist1'][ic] += _bincount_complex(
                om_nm[:, None, :, :], fnm[:, None, :, :] * T, edge0, width, nbins
            )
            # ---- mixed terms (overall factor i), resonant at omega'_mn
            ra_nm = r[:, a]
            rb_mn, rc_mn = r[:, b].transpose(0, 2, 1), r[:, c].transpose(0, 2, 1)
            t1 = (
                2.0
                * ra_nm
                * (rd[:, c, b].transpose(0, 2, 1) + rd[:, b, c].transpose(0, 2, 1))
                * inv_mn
            )
            t2 = (rd[:, c, a] * rb_mn + rd[:, b, a] * rc_mn) * inv_mn
            x3 = (
                ra_nm
                * (rb_mn * dv[:, c].transpose(0, 2, 1) + rc_mn * dv[:, b].transpose(0, 2, 1))
                * inv_mn**2
            )
            t4 = -0.5 * (rb_mn * rd[:, a, c] + rc_mn * rd[:, a, b]) * inv_mn
            t5 = -0.5 * dv[:, a].transpose(0, 2, 1) * np.real(r[:, b] * rc_mn) * inv_mn**2
            out['hist1'][ic] += _bincount_complex(
                om_mn, 1j * fnm * (t2 + x3 + t4 + t5), edge0, width, nbins
            )
            out['hist2'][ic] += _bincount_complex(
                om_mn, 1j * fnm * (t1 - 4.0 * x3), edge0, width, nbins
            )


def shg_from_histograms(
    hist1: NDArray[np.complex128],
    hist2: NDArray[np.complex128],
    bin_centres: NDArray[np.float64],
    energy_grid: NDArray[np.float64],
    eta: float,
) -> NDArray[np.complex128]:
    """:math:`\\sum_b h_1(E_b)/(E_b-\\omega-i\\eta) + h_2(E_b)/(E_b-2\\omega-i\\eta)` on the grid."""
    w = np.asarray(energy_grid, dtype=float)
    d1 = 1.0 / (bin_centres[None, :] - w[:, None] - 1j * eta)  # (ne, nbins)
    d2 = 1.0 / (bin_centres[None, :] - 2.0 * w[:, None] - 1j * eta)
    return hist1 @ d1.T + hist2 @ d2.T


# --------------------------------------------------------------------------- driver
def _k_grid(nk: Sequence[int]) -> NDArray[np.float64]:
    axes = [np.arange(int(n)) / int(n) for n in nk]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def nonlinear_optical_spectra(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    cell_volume: float,
    orbital_positions_cryst: NDArray[np.float64] | None,
    nelec: float,
    nk: Sequence[int],
    energy_grid: NDArray[np.float64],
    *,
    components='all',
    eta: float = 0.05,
    smearing: str = 'gauss',
    adaptive_smearing: float | None = None,
    scissor: float = 0.0,
    temperature: float = 0.0,
    fermi_energy: float | None = None,
    degeneracy_tol: float = 1.0e-2,
    spin_factor: float = 2.0,
    ispin: int = 0,
    kblock: int = 128,
    want: Sequence[str] = RESPONSES,
    shg_bin: float | None = None,
    hopping_cutoff: float | None = None,
    comm=None,
    progress: bool = False,
) -> dict:
    """Shift current, injection currents and SHG of ``HRs`` on a uniform k-grid.

    Parameters
    ----------
    HRs : ndarray ``(nawf, nawf, n1, n2, n3, nspin)``
        Real-space PAO Hamiltonian (eV).
    a_vectors : ndarray ``(3, 3)``
        Direct lattice vectors (rows, alat units).
    alat : float
        Lattice parameter (bohr).
    cell_volume : float
        Cell volume (bohr^3).
    orbital_positions_cryst : ndarray ``(nawf, 3)`` or None
        Orbital centres (crystal coordinates); ``None`` puts all orbitals at the origin.
    nelec : float
        Electrons per cell.
    nk : sequence of int
        Dense k-grid ``(n1, n2, n3)``.
    energy_grid : ndarray ``(ne,)``
        Photon energies (eV).
    components : 'all', 'diag', or sequence of ``'abc'`` strings / index triples
        Tensor components ``(a, b, c)``: current (or polarisation) direction ``a``, fields ``b, c``.
    eta : float
        Width of the :math:`\\delta` functions (shift, injection) and imaginary part of the SHG
        resonances (eV).
    smearing : {'gauss', 'lorentz'}
        Shape of the :math:`\\delta` functions.
    adaptive_smearing : float, optional
        Yates prefactor ``a``: per-pair widths :math:`a|\\mathbf v_n-\\mathbf v_m|\\Delta k`
        floored at ``eta``.
    scissor : float
        Rigid shift of the conduction bands (eV) in the resonance energies.
    temperature : float
        Electronic temperature (K); 0 gives step occupations.
    fermi_energy : float, optional
        Chemical potential (eV); default: band-index occupations at 0 K, the mid-gap at finite T.
    degeneracy_tol : float
        Pairs closer than this (eV) are treated as degenerate and dropped from :math:`r`
        and from the band sums.  DFT-derived PAO Hamiltonians have many nearly degenerate
        pairs whose :math:`1/\\omega` factors amplify their meV-level numerical asymmetries
        (the leakage into symmetry-forbidden components of GaAs drops from 30 % to below 1 %
        between 0.1 and 10 meV), hence the 10 meV default; analytic models can use 1e-4.
    spin_factor : float
        :math:`g_s` (2 for spin-degenerate bands, 1 for spinors or per collinear channel).
    ispin : int
        Spin channel of ``HRs``.
    kblock : int
        k-points per block.
    want : sequence of {'shift', 'injection', 'shg'}
        Responses computed.
    shg_bin : float, optional
        Histogram bin of the SHG transition energies (eV; default ``eta / 5``).
    hopping_cutoff : float, optional
        Drop real-space cells with ``max |H(n)|`` below it (eV).
    comm : MPI communicator, optional
        k-points are split over its ranks; ``None`` runs serially.
    progress : bool, optional
        Print the k-block progress of rank 0.

    Returns
    -------
    dict
        ``'energy_ev'``, ``'components'`` (labels), ``'shift'`` and ``'shiftm'`` ``(ncomp, ne)`` in
        A/V^2, ``'injectc'`` and ``'injectm'`` in A V^-2 s^-1, ``'shg'`` ``(ncomp, ne)`` complex in
        pm/V (keys present for the responses in ``want``), ``'nocc'``, ``'gap_ev'``,
        ``'direct_gap_ev'``, ``'nk'``, ``'scissor_ev'``, ``'eta_ev'``, ``'spin_factor'``.
    """
    rank, size = 0, 1
    if comm is not None:
        rank, size = comm.Get_rank(), comm.Get_size()
    want = tuple(want)
    unknown = [w for w in want if w not in RESPONSES]
    if unknown:
        raise ValueError('unknown responses %s (choose from %s)' % (unknown, RESPONSES))
    comps = parse_components(components)
    energy_grid = np.asarray(energy_grid, dtype=float)
    terms = bloch_terms_from_HRs(
        HRs, a_vectors, alat, orbital_positions_cryst, ispin=ispin, hopping_cutoff=hopping_cutoff
    )
    n = terms.nawf
    nk = tuple(int(x) for x in nk)
    K = _k_grid(nk)
    nkd = K.shape[0]
    # occupations and the scissor need the band count below the gap
    nocc_float = float(nelec) / float(spin_factor)
    if abs(nocc_float - round(nocc_float)) > 1.0e-6:
        raise ValueError(
            'nelec / spin_factor = %g is not an integer: pass fermi_energy and a temperature'
            % nocc_float
        )
    nocc = int(round(nocc_float))
    if nocc < 1 or nocc >= n:
        raise ValueError('%d occupied bands of %d: check nelec and the spin factor' % (nocc, n))
    # band edges on the coarse grid (the histogram range and the default Fermi level)
    coarse = _k_grid(np.asarray(HRs).shape[2:5])
    H0, _, _ = hamiltonian_derivatives_block(terms, coarse)
    e0 = np.linalg.eigvalsh(H0)
    vbm, cbm = float(e0[:, nocc - 1].max()), float(e0[:, nocc].min())
    bandwidth = float(e0.max() - e0.min()) + abs(float(scissor)) + 2.0 * float(eta) + 1.0
    mu = fermi_energy
    if temperature > 0.0 and mu is None:
        if cbm <= vbm:
            raise ValueError('no gap above band %d: pass fermi_energy at finite temperature' % nocc)
        mu = 0.5 * (vbm + cbm)
    conduction = (np.arange(n) >= nocc).astype(float)
    scissor_pair = float(scissor) * (conduction[:, None] - conduction[None, :])  # (n, n)
    spacing = (8.0 * np.pi**3 / (float(cell_volume) * nkd)) ** (1.0 / 3.0)  # 1/bohr
    # accumulators
    ne, ncomp = energy_grid.size, len(comps)
    out = {key: np.zeros((ncomp, ne)) for key in ('shift', 'shiftm', 'injectc', 'injectm')}
    bin_width = float(shg_bin) if shg_bin else float(eta) / 5.0
    nbins = int(np.ceil(2.0 * bandwidth / bin_width)) + 1
    bin_edges = -bandwidth + bin_width * np.arange(nbins + 1)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    hist = {
        'hist1': np.zeros((ncomp, nbins), dtype=complex),
        'hist2': np.zeros((ncomp, nbins), dtype=complex),
    }
    do_delta = 'shift' in want or 'injection' in want
    do_shg = 'shg' in want
    # k loop
    from ..utils.communication import load_balancing

    kstart, kstop = load_balancing(size, rank, nkd)
    nblocks = max(1, -(-(kstop - kstart) // int(kblock)))
    for ib, b0 in enumerate(range(kstart, kstop, int(kblock))):
        if progress and rank == 0 and (ib % max(1, nblocks // 10) == 0 or ib == nblocks - 1):
            print(
                '  nonlinear optics: k-block %d of %d (%d k-points per rank)'
                % (ib + 1, nblocks, kstop - kstart),
                flush=True,
            )
        kb = K[b0 : min(b0 + int(kblock), kstop)]
        H, dH, d2H = hamiltonian_derivatives_block(terms, kb)
        v_orb, w_orb = velocity_and_curvature_operators(terms, H, dH, d2H)
        E, U = np.linalg.eigh(H)
        geometry = band_geometry(E, U, v_orb, w_orb, degeneracy_tol)
        f = occupations(E, nocc, temperature, mu)
        omega_s = geometry.omega + scissor_pair[None]
        if do_delta:
            if adaptive_smearing:
                dvel = np.sqrt(np.sum(geometry.delta_velocity**2, axis=1))  # (nb, n, n) |v_n - v_m|
                widths = np.maximum(float(adaptive_smearing) * dvel * spacing, float(eta))
            else:
                widths = np.full(omega_s.shape, float(eta))
            accumulate_shift_and_injection(
                geometry, f, omega_s, energy_grid, widths, comps, smearing, out
            )
        if do_shg:
            accumulate_shg_histograms(
                geometry,
                f,
                E + float(scissor) * conduction[None, :],
                bin_edges,
                comps,
                hist,
                degeneracy_tol,
            )
    if comm is not None and size > 1:
        from mpi4py import MPI

        for array in list(out.values()) + list(hist.values()):
            comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
    norm = float(spin_factor) / (nkd * float(cell_volume))
    results = {
        'energy_ev': energy_grid,
        'components': [component_label(c) for c in comps],
        'nocc': nocc,
        'gap_ev': cbm - vbm,
        'direct_gap_ev': float(np.min(e0[:, nocc] - e0[:, nocc - 1])),
        'nk': nk,
        'scissor_ev': float(scissor),
        'eta_ev': float(eta),
        'spin_factor': float(spin_factor),
        'fermi_energy_ev': mu,
        'temperature_k': float(temperature),
    }
    if 'shift' in want:
        results['shift'] = C_SHIFT * norm * out['shift']
        results['shiftm'] = C_SHIFT * norm * out['shiftm']
    if 'injection' in want:
        results['injectc'] = C_INJECT * norm * out['injectc']
        results['injectm'] = C_INJECT * norm * out['injectm']
    if do_shg:
        chi = shg_from_histograms(
            hist['hist1'], hist['hist2'], bin_centres, energy_grid, float(eta)
        )
        results['shg'] = C_SHG * norm * chi
        results['shg_bin_ev'] = bin_width
    return results


# --------------------------------------------------------------------------- output
#: file prefix -> (result key, transform, 3D unit, 2D unit)
OUTPUT_FILES = (
    ('shift', 'shift', lambda x: 1.0e6 * x, 'uA/V^2', 'nm uA/V^2'),
    ('shiftm', 'shiftm', lambda x: 1.0e6 * x, 'uA/V^2', 'nm uA/V^2'),
    ('injectc', 'injectc', lambda x: x, 'A V^-2 s^-1', 'nm A V^-2 s^-1'),
    ('injectm', 'injectm', lambda x: x, 'A V^-2 s^-1', 'nm A V^-2 s^-1'),
    ('shgr', 'shg', lambda x: x.real, 'pm/V', 'pm^2/V'),
    ('shgi', 'shg', lambda x: x.imag, 'pm/V', 'pm^2/V'),
    ('shga', 'shg', lambda x: np.abs(x), 'pm/V', 'pm^2/V'),
)


def write_nonlinear_outputs(
    data_controller, results: dict, slab_height_bohr: float | None = None, suffix: str = ''
) -> list[str]:
    """Write ``<prefix>_<abc><suffix>.dat`` (energy, value) through the data controller.

    Parameters
    ----------
    data_controller : DataController
    results : dict
        Of :func:`nonlinear_optical_spectra`.
    slab_height_bohr : float, optional
        For a 2D system the responses are multiplied by this length (shift and injection in
        nm units, SHG in pm): ``nm uA/V^2``, ``nm A V^-2 s^-1``, ``pm^2/V``.
    suffix : str, optional
        Appended to the file stem (e.g. ``'_0'`` for a spin channel).

    Returns
    -------
    list of str
        File names written.
    """
    from ..utils.constants import BOHR_RADIUS_ANGS

    energy = results['energy_ev']
    scale_nm = scale_pm = 1.0
    if slab_height_bohr is not None:
        scale_nm = float(slab_height_bohr) * BOHR_RADIUS_ANGS * 0.1
        scale_pm = float(slab_height_bohr) * BOHR_RADIUS_ANGS * 100.0
    written = []
    for prefix, key, transform, _unit3d, _unit2d in OUTPUT_FILES:
        if key not in results:
            continue
        scale = scale_pm if key == 'shg' else scale_nm
        for ic, label in enumerate(results['components']):
            name = '%s_%s%s.dat' % (prefix, label, suffix)
            data_controller.write_file_row_col(name, energy, scale * transform(results[key][ic]))
            written.append(name)
    return written


__all__ = [
    'C_INJECT',
    'C_SHG',
    'C_SHIFT',
    'OUTPUT_FILES',
    'RESPONSES',
    'BandGeometry',
    'accumulate_shg_histograms',
    'accumulate_shift_and_injection',
    'band_geometry',
    'component_label',
    'delta_kernel',
    'nonlinear_optical_spectra',
    'occupations',
    'parse_components',
    'shg_from_histograms',
    'write_nonlinear_outputs',
]
