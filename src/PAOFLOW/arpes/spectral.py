"""Spectral functions with band-resolved self-energies, Fermi factor and resolution broadening."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

KB_EV = 8.617333262e-5
#: Smallest imaginary part of the Green's function denominator (eV), as chinook.
ETA_MIN = 5.0e-5
SIGMA_PER_FWHM = 1.0 / np.sqrt(8.0 * np.log(2.0))


def fermi_function(w: NDArray[np.float64], temperature_k: float | None) -> NDArray[np.float64]:
    """Fermi-Dirac occupation on the energy grid ``w`` (eV, relative to E_F); ones without T."""
    w = np.asarray(w, dtype=float)
    if temperature_k is None or temperature_k <= 0.0:
        return np.ones_like(w)
    x = np.clip(w / (KB_EV * temperature_k), -500.0, 500.0)
    return 1.0 / (np.exp(x) + 1.0)


def spectral_function(
    energies: NDArray[np.float64],
    self_energy: NDArray[np.complex128] | complex,
    w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``A[k, n, w] = -Im[1 / (w - e_nk - Sigma_nk(w) + i eta)] / pi``.

    Parameters
    ----------
    energies : ndarray ``(nk, nb)``
        Band energies (eV).
    self_energy : complex or ndarray broadcastable to ``(nk, nb, nw)``
        ``Sigma`` (eV): a constant, a function of ``w`` ``(nw,)``, or state-resolved.
    w : ndarray ``(nw,)``

    Returns
    -------
    ndarray ``(nk, nb, nw)``
    """
    e = np.asarray(energies, dtype=float)[..., None]
    sigma = np.asarray(self_energy, dtype=complex)
    denominator = np.asarray(w, dtype=float)[None, None, :] - e - sigma + 1j * ETA_MIN
    return -np.imag(1.0 / denominator) / np.pi


def assemble_intensity(
    weights: NDArray[np.float64],
    energies: NDArray[np.float64],
    w: NDArray[np.float64],
    self_energy,
    temperature_k: float | None = None,
    kpoints: NDArray[np.float64] | None = None,
    kblock: int = 256,
    bands: NDArray[np.int_] | None = None,
) -> NDArray[np.float64]:
    """``I[k, w] = sum_n |M_nk|^2 A_nk(w) f(w)``, in blocks of k-points.

    Parameters
    ----------
    weights : ndarray ``(nk, nb)``
        Matrix-element weights.
    energies : ndarray ``(nk, nb)``
        Band energies (eV, relative to E_F).
    w : ndarray ``(nw,)``
    self_energy : complex, ndarray or callable
        A constant, an array broadcastable to ``(nk, nb, nw)``, or ``f(w, energies, kpoints, bands)``
        returning one of these for the block it is given (``kpoints`` of the block when
        supplied).
    temperature_k : float, optional
    kpoints : ndarray ``(nk, 3)``, optional
        Passed to a callable self-energy.
    kblock : int, optional
    bands : ndarray of int, optional
        Band indices of the second axis, passed to a callable self-energy as its fourth argument.

    Returns
    -------
    ndarray ``(nk, nw)``
    """
    nk = weights.shape[0]
    fermi = fermi_function(w, temperature_k)
    out = np.zeros((nk, np.asarray(w).size))
    for s0 in range(0, nk, kblock):
        s1 = min(s0 + kblock, nk)
        if callable(self_energy):
            kp = None if kpoints is None else kpoints[s0:s1]
            sigma = self_energy(w, energies[s0:s1], kp, bands)
        elif np.ndim(self_energy) == 3:
            sigma = self_energy[s0:s1]
        else:
            sigma = self_energy
        A = spectral_function(energies[s0:s1], sigma, w)
        out[s0:s1] = np.einsum('kn,knw->kw', weights[s0:s1], A) * fermi[None, :]
    return out


class ConstantSelfEnergy:
    """``Sigma = -i Gamma``: a constant linewidth (eV, half width)."""

    def __init__(self, gamma_ev: float) -> None:
        self.gamma_ev = float(gamma_ev)

    def __call__(self, w, energies=None, kpoints=None, bands=None) -> complex:
        return -1j * abs(self.gamma_ev)


class GridSelfEnergy:
    """``Sigma(w)`` interpolated from a tabulated complex function (k- and band-independent)."""

    def __init__(self, w_grid: NDArray[np.float64], sigma: NDArray[np.complex128]) -> None:
        self.w_grid = np.asarray(w_grid, dtype=float)
        self.sigma = np.asarray(sigma, dtype=complex)

    def __call__(self, w, energies=None, kpoints=None, bands=None) -> NDArray[np.complex128]:
        w = np.asarray(w, dtype=float)
        return np.interp(w, self.w_grid, self.sigma.real) + 1j * np.interp(
            w, self.w_grid, self.sigma.imag
        )


def kramers_kronig_self_energy(
    w: NDArray[np.float64], im_sigma: NDArray[np.float64], pad: int = 50
) -> NDArray[np.complex128]:
    """``Sigma(w)`` from its imaginary part by the Kramers-Kronig relation (Hilbert transform).

    Parameters
    ----------
    w : ndarray ``(nw,)``
        Uniform energy grid (eV).
    im_sigma : ndarray ``(nw,)``
        ``Im Sigma(w) <= 0``; it is extended by its edge values over ``pad`` times the window
        on both sides (chinook ``gen_SE_KK`` extends over 100 windows) before the transform.
    pad : int, optional

    Returns
    -------
    ndarray ``(nw,)``, complex
        ``Re Sigma + i Im Sigma`` on ``w``.

    Notes
    -----
    :math:`\\mathrm{Re}\\,\\Sigma(\\omega) = -\\mathcal H[\\mathrm{Im}\\,\\Sigma](\\omega)` with
    :math:`\\mathcal H` the Hilbert transform (``scipy.signal.hilbert`` returns
    ``f + i H[f]``), the retarded self-energy being analytic in the upper half plane.
    """
    w = np.asarray(w, dtype=float)
    im = -np.abs(np.asarray(im_sigma, dtype=float))
    nw = w.size
    extended = np.concatenate([np.full(pad * nw, im[0]), im, np.full(pad * nw, im[-1])])
    analytic = hilbert(extended)
    re = -np.imag(analytic)[pad * nw : pad * nw + nw]
    return re + 1j * im


def gaussian_broadening(
    intensity: NDArray[np.float64], fwhm_per_axis, spacing_per_axis
) -> NDArray[np.float64]:
    """Gaussian resolution broadening along every axis (FWHM and grid spacing per axis).

    Axes with a zero FWHM or an undefined spacing are not broadened.
    """
    sigmas = []
    for fwhm, spacing in zip(fwhm_per_axis, spacing_per_axis):
        if fwhm is None or fwhm <= 0.0 or spacing is None or spacing <= 0.0:
            sigmas.append(0.0)
        else:
            sigmas.append(SIGMA_PER_FWHM * float(fwhm) / float(spacing))
    return gaussian_filter(np.asarray(intensity, dtype=float), sigmas)
