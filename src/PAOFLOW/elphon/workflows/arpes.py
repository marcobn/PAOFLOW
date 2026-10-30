"""ARPES workflow: photoemission setup, k-paths and the electron-phonon self-energies.

Helpers shared by the ``arpes/`` drivers: the Bloch Hamiltonian and photoemission basis of
a system, segments of a high-symmetry direction (across the Fermi crossing for metals),
the isotropic Engelsberg-Schrieffer self-energy from alpha^2F, the anisotropic one from the
Fermi-surface pair coupling and the band-resolved Fan-Migdal self-energy from the dense-q
vertex.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ...arpes import BlochHamiltonian, PhotoemissionBasis
from ...arpes.radial import BOHR_ANGSTROM
from ..anisotropic_eliashberg import coupling_strength
from ..fermi_surface_coupling import read_fs_coupling
from ..self_energy import eliashberg_self_energy, fan_migdal_self_energy, mass_enhancement
from .system import Electrons, ElphonSystem


def photoemission(electrons: Electrons) -> tuple[BlochHamiltonian, PhotoemissionBasis]:
    """Bloch Hamiltonian and photoemission basis of the coupling save.

    Parameters
    ----------
    electrons : Electrons
        Of :meth:`ElphonSystem.electrons`.

    Returns
    -------
    tuple
        ``(bloch, basis)`` for :func:`PAOFLOW.arpes.simulate`.
    """
    return (
        BlochHamiltonian.from_data_controller(electrons.data_controller),
        PhotoemissionBasis.from_data_controller(electrons.data_controller),
    )


def path_points(
    nscf: dict, direction: Sequence[float], npts: int, fractions: Sequence[float] = (0.0, 1.0)
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A segment of ``Gamma -> direction`` in crystal coordinates and in Cartesian 1/A.

    Parameters
    ----------
    nscf : dict
        Lattice data (``bg``, ``alat``).
    direction : sequence of float
        End point of the line in crystal coordinates (e.g. ``(0.5, 0, 0.5)`` for X of fcc).
    npts : int
        Points of the segment.
    fractions : (start, end), optional
        The segment, as fractions of the line.

    Returns
    -------
    tuple
        ``(k_cryst, k_cart)`` with shapes ``(npts, 3)``.
    """
    direction = np.asarray(direction, dtype=float)
    k_cryst = np.linspace(fractions[0], fractions[1], npts)[:, None] * direction[None, :]
    k_cart = k_cryst @ nscf['bg'] * 2.0 * np.pi / (nscf['alat'] * BOHR_ANGSTROM)
    return k_cryst, k_cart


def fermi_crossing(
    bloch: BlochHamiltonian,
    direction: Sequence[float],
    *,
    segment_ev: float,
    nscan: int = 4001,
) -> tuple[int, float, tuple[float, float]]:
    """The first band crossing E_F along ``Gamma -> direction`` and a segment around it.

    Parameters
    ----------
    bloch : BlochHamiltonian
    direction : sequence of float
        End point of the line (crystal coordinates).
    segment_ev : float
        The segment spans about ``+- segment_ev`` around the crossing (from the local slope).
    nscan : int, optional
        Points of the scan.

    Returns
    -------
    tuple
        ``(band, x_f, (start, end))``: the band index, the crossing and the segment as
        fractions of the line.

    Raises
    ------
    ValueError
        If no band crosses the Fermi level along the line.
    """
    fractions = np.linspace(0.0, 1.0, nscan)
    scan = fractions[:, None] * np.asarray(direction, dtype=float)[None, :]
    energies, _ = bloch.eigenstates(scan)
    crossings = []
    for n in range(energies.shape[1]):
        e = energies[:, n]
        for i in np.nonzero(np.diff(np.sign(e)))[0]:
            crossings.append(
                (fractions[i], n, abs(e[i + 1] - e[i]) / (fractions[1] - fractions[0]))
            )
    if not crossings:
        raise ValueError('no band crosses E_F along Gamma -> %s' % (tuple(direction),))
    x_f, band, slope = sorted(crossings)[0]  # the first crossing from Gamma
    half = segment_ev / slope
    return band, x_f, (max(x_f - half, 0.0), min(x_f + half, 1.0))


def isotropic_self_energy(
    eliashberg_npz: str, w: NDArray[np.float64], temperature_k: float
) -> tuple[NDArray[np.complex128], float]:
    """Engelsberg-Schrieffer self-energy ``Sigma(w, T)`` from the alpha^2F of an Eliashberg run.

    Parameters
    ----------
    eliashberg_npz : str
        ``eliashberg.npz`` of the superconductivity workflow.
    w : ndarray
        Energies (eV) relative to E_F.
    temperature_k : float

    Returns
    -------
    tuple
        ``(sigma, lambda)`` with the lambda of the run.
    """
    data = np.load(eliashberg_npz)
    return eliashberg_self_energy(data['omega'], data['a2F'], w, temperature_k), float(
        data['lambda']
    )


def state_self_energies(
    coupling: dict, w: NDArray[np.float64], temperature_k: float
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """``alpha^2F_a(w)`` and ``Sigma_a(w, T)`` of every irreducible Fermi-surface state.

    Parameters
    ----------
    coupling : dict
        :func:`~PAOFLOW.elphon.fermi_surface_coupling.read_fs_coupling` of ``fs_coupling.npz``.
    w : ndarray
        Energies (eV).
    temperature_k : float

    Returns
    -------
    tuple
        ``(a2f_state, sigma)`` with shapes ``(na, nw_phonon)`` and ``(na, nw)``.

    Notes
    -----
    The pair coupling is integrated over each frequency step, so
    ``alpha^2F_a = (omega / 2) sum_b Lambda_ab / d omega``.
    """
    pair = coupling['coupling']  # (na, nb, nw)
    omega = np.asarray(coupling['freq_ev'], dtype=float)
    spacing = omega[0]
    a2f_state = 0.5 * omega[None, :] * pair.sum(axis=1) / spacing
    return a2f_state, eliashberg_self_energy(omega, a2f_state, w, temperature_k)


def nearest_state(
    coupling: dict, rots: NDArray[np.float64], k_cryst: NDArray[np.float64], band: int
) -> int | None:
    """Index of the irreducible state of ``band`` whose star is closest to ``k_cryst``.

    Parameters
    ----------
    coupling : dict
        The Fermi-surface coupling.
    rots : ndarray, shape ``(nsym, 3, 3)``
        Rotations in crystal coordinates.
    k_cryst : ndarray, shape ``(3,)``
    band : int

    Returns
    -------
    int or None
        ``None`` when the band has no Fermi-surface state.
    """
    states = np.nonzero(coupling['band'] == band)[0]
    if states.size == 0:
        return None
    reps = coupling['k_cryst'][states]
    best, best_d = None, np.inf
    for rot in rots:
        kr = rot @ k_cryst
        d = np.abs(reps - kr[None, :])
        d = np.minimum(d, 1.0 - d).max(axis=1)  # periodic distance (crystal coordinates)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best, best_d = states[i], d[i]
    return best


class AnisotropicSelfEnergy:
    """Self-energy of the nearest irreducible Fermi-surface state (the ARPES protocol).

    Parameters
    ----------
    coupling : dict
        The Fermi-surface coupling.
    rots : ndarray
        Rotations in crystal coordinates (``nscf['s_cryst']``).
    bloch : BlochHamiltonian
        For the crystal coordinates of the k-points.
    sigma_states : ndarray, shape ``(na, nw)``
        :func:`state_self_energies`.
    w_grid : ndarray
        Its energy grid.
    """

    def __init__(self, coupling, rots, bloch, sigma_states, w_grid) -> None:
        self.coupling, self.rots, self.bloch = coupling, np.asarray(rots, dtype=float), bloch
        self.sigma_states, self.w_grid = sigma_states, w_grid

    def __call__(self, w, energies=None, kpoints=None, bands=None) -> NDArray[np.complex128]:
        """``Sigma(w)`` of shape ``(nk, nbands, nw)``."""
        k_cryst = self.bloch.crystal_coordinates(kpoints)
        out = np.zeros((k_cryst.shape[0], len(bands), w.size), dtype=complex)
        for ik, k in enumerate(k_cryst):
            for ib, band in enumerate(bands):
                a = nearest_state(self.coupling, self.rots, k, int(band))
                if a is not None:
                    s = self.sigma_states[a]
                    out[ik, ib] = np.interp(w, self.w_grid, s.real) + 1j * np.interp(
                        w, self.w_grid, s.imag
                    )
        return out


def anisotropic_self_energies(
    fs_coupling_npz: str,
    nscf: dict,
    bloch: BlochHamiltonian,
    w: NDArray[np.float64],
    temperature_k: float,
) -> dict:
    """Everything the anisotropic ARPES protocol needs from ``fs_coupling.npz``.

    Parameters
    ----------
    fs_coupling_npz : str
        Of the superconductivity workflow (``fs_coupling=True``).
    nscf : dict
        For the rotations.
    bloch : BlochHamiltonian
    w : ndarray
        Energies (eV).
    temperature_k : float

    Returns
    -------
    dict
        ``coupling``, ``strength`` (lambda, lambda_nk, N_F), ``sigma_states``,
        ``sigma_iso`` (from the isotropic alpha^2F of the same coupling) and
        ``self_energy`` (an :class:`AnisotropicSelfEnergy`).
    """
    coupling = read_fs_coupling(fs_coupling_npz)
    strength = coupling_strength(coupling)
    _, sigma_states = state_self_energies(coupling, w, temperature_k)
    rots = np.asarray(nscf['s_cryst'], dtype=float)
    return {
        'coupling': coupling,
        'strength': strength,
        'sigma_states': sigma_states,
        'sigma_iso': eliashberg_self_energy(strength['omega'], strength['a2F'], w, temperature_k),
        'self_energy': AnisotropicSelfEnergy(coupling, rots, bloch, sigma_states, w),
    }


def fan_migdal(
    system: ElphonSystem,
    electrons: Electrons,
    *,
    k_cryst: NDArray[np.float64],
    w_ev: NDArray[np.float64],
    temps_k: Sequence[float],
    nq: int,
    eta_ev: float = 0.01,
    adaptive_smearing: float | None = None,
    bands: Sequence[int] | None = None,
    window_margin_ev: float = 0.5,
    doping_cm3: float = 0.0,
    scissor_ev=None,
    target_gap_ev=None,
    screening_eps: float | None = None,
    screening_eps_dft: float | None = None,
    comm=None,
) -> dict:
    """Band-resolved Fan-Migdal self-energy ``Sigma_nk(w, T)`` on a k-path from the dense-q vertex.

    Parameters
    ----------
    system, electrons
        The system and its electrons.
    k_cryst : ndarray, shape ``(nk, 3)``
        The path (crystal coordinates).
    w_ev : ndarray
        Energies relative to the chemical potential.
    temps_k : sequence of float
    nq : int
        Dense q-grid per axis.
    eta_ev : float
        Lorentzian width (the floor with ``adaptive_smearing``).
    adaptive_smearing : float, optional
        Yates prefactor of the per-state widths (metals need it at affordable q-grids).
    bands, window_margin_ev, doping_cm3, scissor_ev, target_gap_ev, screening_eps,
    screening_eps_dft
        Passed to :func:`~PAOFLOW.elphon.self_energy.fan_migdal_self_energy`.
    comm : MPI communicator, optional

    Returns
    -------
    dict
        The backend result (``sigma_ev`` of shape ``(ntemp, nk, nband, nw)``,
        ``energies_ev``, ``bands``, ``w_ev``, ``temps_k``, ...).
    """
    return fan_migdal_self_energy(
        *system.positional_arguments(electrons),
        nelec=system.nelec,
        k_cryst=np.asarray(k_cryst, dtype=float),
        w_ev=np.asarray(w_ev, dtype=float),
        temps_k=list(temps_k),
        nq_dense=nq,
        eta_ev=eta_ev,
        adaptive_smearing=adaptive_smearing,
        bands=bands,
        window_margin_ev=window_margin_ev,
        doping_cm3=doping_cm3,
        scissor_ev=scissor_ev,
        target_gap_ev=target_gap_ev,
        screening_eps=screening_eps,
        screening_eps_dft=screening_eps_dft,
        spin_degeneracy=system.degeneracy(electrons),
        comm=comm,
        **system.coupling_arguments(electrons, symmetry=False),
    )


def linewidth_table(result: dict, labels: Sequence[str] = ('start', 'end')) -> list[tuple]:
    """``-Im Sigma`` and ``Re Sigma`` at the band energy of every window state at the path ends.

    Parameters
    ----------
    result : dict
        Of :func:`fan_migdal`.
    labels : (str, str)
        Names of the first and last k-point.

    Returns
    -------
    list of tuple
        ``(band, label, energy_ev, widths_mev (per T), shifts_mev (per T))``.
    """
    w, e, sigma = result['w_ev'], result['energies_ev'], result['sigma_ev']
    rows = []
    for ib, band in enumerate(result['bands']):
        for ik, label in ((0, labels[0]), (e.shape[0] - 1, labels[1])):
            iw = int(np.argmin(np.abs(w - e[ik, ib])))
            rows.append(
                (
                    int(band),
                    label,
                    e[ik, ib],
                    -1e3 * sigma[:, ik, ib, iw].imag,
                    1e3 * sigma[:, ik, ib, iw].real,
                )
            )
    return rows


def on_shell_widths(result: dict, temperature_index: int) -> NDArray[np.float64]:
    """``-Im Sigma_nk(e_nk)`` (eV) of every window state, shape ``(nk, nband)``."""
    w, e, sigma = result['w_ev'], result['energies_ev'], result['sigma_ev']
    return np.array(
        [
            [
                -sigma[temperature_index, ik, ib, int(np.argmin(np.abs(w - e[ik, ib])))].imag
                for ib in range(e.shape[1])
            ]
            for ik in range(e.shape[0])
        ]
    )


__all__ = [
    'AnisotropicSelfEnergy',
    'anisotropic_self_energies',
    'fan_migdal',
    'fermi_crossing',
    'isotropic_self_energy',
    'linewidth_table',
    'mass_enhancement',
    'nearest_state',
    'on_shell_widths',
    'path_points',
    'photoemission',
    'state_self_energies',
]
