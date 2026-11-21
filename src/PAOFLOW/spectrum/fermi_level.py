r"""Charge-neutral Fermi level of the PAO bands, by electron counting on a dense k-grid.

QE and VASP runs refer every PAOFLOW energy to the Fermi level of the DFT run, which carries the
smearing of that run (e.g. Marzari-Vanderbilt 0.27 eV) and its coarse k-grid.  In a metal with a
large density of states at :math:`E_F` the difference from the charge-neutral level at the
temperature of a response calculation is a few meV; in a semimetal or a small-pocket metal it is
tens of meV (TaAs +74 meV at 300 K, CoSi -19 meV at 10 K, Nov 2026).  The functions here find the
chemical potential :math:`\mu` at which the bands interpolated from ``HRs`` hold the electrons of
the crystal at a given temperature.

The count is split by band index into valence states :math:`v` (holes :math:`1 - f`) and
conduction states :math:`c` (electrons :math:`f`), and the neutrality condition is solved in
logarithmic form, so that it stays exact in a gap at any temperature (insulators give the
intrinsic level) as well as in a metal.  Eigenvalues only; the k-points are distributed over the
MPI ranks in blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

#: Boltzmann constant (eV/K).
BOLTZMANN_EV_PER_K = 8.617333262e-5
#: Temperature (K) at which a requested 0 K is evaluated (the limit of the count).
ZERO_TEMPERATURE_FLOOR_K = 1.0
#: Half-width (eV) of the energy window kept around the DFT level, plus 50 kT.
DEFAULT_WINDOW_EV = 2.0
#: Largest difference (electrons per cell) allowed between the count at the DFT level and nelec.
NELEC_MISMATCH_LIMIT = 0.5
_BISECTION_STEPS = 64
_VALENCE_TOLERANCE_EV = 1.0e-6


@dataclass
class BandCount:
    """Local (this rank's) part of the band energies needed by the count.

    Attributes
    ----------
    valence_window, conduction_window : ndarray, 1D
        Energies (eV) of the valence- and conduction-indexed states within the window.
    holes_outside : float
        Valence-indexed states above the window (fully empty, one hole each).
    electrons_outside : float
        Conduction-indexed states below the window (fully occupied).
    valence_states : float
        Valence-indexed states on this rank's k-points.
    valence_top, conduction_bottom : float
        Highest valence-indexed and lowest conduction-indexed energy (eV) on this rank.
    """

    valence_window: NDArray[np.float64]
    conduction_window: NDArray[np.float64]
    holes_outside: float
    electrons_outside: float
    valence_states: float
    valence_top: float
    conduction_bottom: float


def full_k_grid(nk: Sequence[int]) -> NDArray[np.float64]:
    """Crystal coordinates of the uniform grid ``nk`` (Gamma-centred, ``[0, 1)`` per axis).

    Parameters
    ----------
    nk : sequence of int
        Points per reciprocal axis ``(n1, n2, n3)``.

    Returns
    -------
    ndarray ``(n1 * n2 * n3, 3)``
    """
    axes = [np.arange(int(n)) / int(n) for n in nk]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def valence_band_counts(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    orbital_positions_cryst: NDArray[np.float64] | None = None,
    hopping_cutoff: float | None = None,
    comm=None,
    kblock: int = 256,
) -> list[int]:
    r"""Bands at or below the DFT Fermi level of each spin channel, on the grid of ``HRs``.

    The grid of ``HRs`` is where the interpolated bands are the DFT (projected) ones.  The count
    only fixes the band-index split of :func:`charge_neutral_levels`: any split gives the same
    :math:`\mu`, and the one at the DFT level keeps the hole and electron sums exact in a gap.

    Parameters
    ----------
    HRs : ndarray ``(nawf, nawf, n1, n2, n3, nspin)``
        Real-space Hamiltonian (eV, relative to the DFT Fermi level).
    a_vectors : ndarray ``(3, 3)``
        Direct lattice vectors (rows, alat units).
    alat : float
        Lattice parameter (bohr).
    orbital_positions_cryst : ndarray ``(nawf, 3)``, optional
        Orbital centres (crystal coordinates).
    hopping_cutoff : float, optional
        As :func:`PAOFLOW.hamiltonian.bloch_derivatives.bloch_terms_from_HRs`.
    comm : MPI communicator, optional
        k-point blocks are dealt round-robin over its ranks (collective).
    kblock : int, optional
        k-points per diagonalisation block.

    Returns
    -------
    list of int
        One count per spin channel, the same on every rank: the mean over the grid of the bands
        at or below ``1e-6`` eV, rounded.
    """
    from ..hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block

    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    HRs = np.asarray(HRs)
    grid = full_k_grid(HRs.shape[2:5])
    counts = []
    for ispin in range(HRs.shape[-1]):
        terms = bloch_terms_from_HRs(
            HRs, a_vectors, alat, orbital_positions_cryst, ispin=ispin,
            hopping_cutoff=hopping_cutoff,
        )  # fmt: skip
        below = 0.0
        for start in range(rank * int(kblock), grid.shape[0], size * int(kblock)):
            energies = np.linalg.eigvalsh(hamiltonian_block(terms, grid[start : start + kblock]))
            below += float(np.count_nonzero(energies <= _VALENCE_TOLERANCE_EV))
        counts.append(int(np.rint(_allreduce(below, comm) / grid.shape[0])))
    return counts


def collect_band_count(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    orbital_positions_cryst: NDArray[np.float64] | None,
    nk: Sequence[int],
    valence_bands: Sequence[int],
    window_ev: float,
    comm=None,
    kblock: int = 256,
    hopping_cutoff: float | None = None,
) -> BandCount:
    """Eigenvalues of this rank's k-points of the grid ``nk``, reduced to what the count needs.

    Parameters
    ----------
    HRs, a_vectors, alat, orbital_positions_cryst, hopping_cutoff
        As :func:`valence_band_counts`.
    nk : sequence of int
        Dense grid ``(n1, n2, n3)``.
    valence_bands : sequence of int
        Valence-indexed bands of each spin channel (:func:`valence_band_counts`).
    window_ev : float
        States with ``|E| <= window_ev`` are kept; the others are counted as full or empty.
    comm : MPI communicator, optional
        k-point blocks are dealt round-robin over its ranks; ``None`` keeps every block.
    kblock : int, optional
        k-points per diagonalisation block (memory ``kblock * nawf^2`` complex numbers).

    Returns
    -------
    BandCount
        This rank's part; :func:`charge_neutral_levels` reduces it over ``comm``.
    """
    from ..hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block

    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    HRs = np.asarray(HRs)
    grid = full_k_grid(nk)
    valence_window, conduction_window = [], []
    holes_outside = electrons_outside = valence_states = 0.0
    valence_top, conduction_bottom = -np.inf, np.inf
    for ispin in range(HRs.shape[-1]):
        terms = bloch_terms_from_HRs(
            HRs, a_vectors, alat, orbital_positions_cryst, ispin=ispin,
            hopping_cutoff=hopping_cutoff,
        )  # fmt: skip
        n_valence = min(max(int(valence_bands[ispin]), 0), terms.nawf)
        for start in range(rank * int(kblock), grid.shape[0], size * int(kblock)):
            energies = np.linalg.eigvalsh(hamiltonian_block(terms, grid[start : start + kblock]))
            valence, conduction = energies[:, :n_valence], energies[:, n_valence:]
            valence_states += valence.size
            valence_window.append(valence[np.abs(valence) <= window_ev])
            conduction_window.append(conduction[np.abs(conduction) <= window_ev])
            holes_outside += float(np.count_nonzero(valence > window_ev))
            electrons_outside += float(np.count_nonzero(conduction < -window_ev))
            if valence.size:
                valence_top = max(valence_top, float(valence[:, -1].max()))
            if conduction.size:
                conduction_bottom = min(conduction_bottom, float(conduction[:, 0].min()))
    empty = np.zeros(0)
    return BandCount(
        valence_window=np.concatenate(valence_window) if valence_window else empty,
        conduction_window=np.concatenate(conduction_window) if conduction_window else empty,
        holes_outside=holes_outside,
        electrons_outside=electrons_outside,
        valence_states=valence_states,
        valence_top=valence_top,
        conduction_bottom=conduction_bottom,
    )


def _allreduce(value: float, comm, operation: str = 'sum') -> float:
    if comm is None:
        return float(value)
    from mpi4py import MPI

    ops = {'sum': MPI.SUM, 'max': MPI.MAX, 'min': MPI.MIN}
    return float(comm.allreduce(float(value), op=ops[operation]))


def _log_sum_exp(log_terms: NDArray[np.float64], comm) -> float:
    """``log(sum(exp(log_terms)))`` over all ranks, without overflow or underflow."""
    local_max = float(np.max(log_terms)) if log_terms.size else -np.inf
    global_max = _allreduce(local_max, comm, 'max')
    if not np.isfinite(global_max):
        return -np.inf
    local_sum = float(np.sum(np.exp(log_terms - global_max))) if log_terms.size else 0.0
    return global_max + np.log(_allreduce(local_sum, comm))


def _log_occupation_sums(count: BandCount, mu: float, kT: float, comm) -> tuple[float, float]:
    """``log`` of the conduction electrons and of the valence holes inside the window at ``mu``."""
    log_electrons = _log_sum_exp(-np.logaddexp(0.0, (count.conduction_window - mu) / kT), comm)
    log_holes = _log_sum_exp(-np.logaddexp(0.0, (mu - count.valence_window) / kT), comm)
    return log_electrons, log_holes


def _log(value: float) -> float:
    return float(np.log(value)) if value > 0.0 else -np.inf


def occupied_states(count: BandCount, mu: float, temperature_k: float, comm=None) -> float:
    """Occupied states of ``count`` (whole grid, all channels) at the chemical potential ``mu``.

    Parameters
    ----------
    count : BandCount
        This rank's part of the band energies (:func:`collect_band_count`).
    mu : float
        Chemical potential (eV, relative to the DFT Fermi level).
    temperature_k : float
        Electronic temperature (K); values below ``ZERO_TEMPERATURE_FLOOR_K`` use the floor.
    comm : MPI communicator, optional
        Every rank must call this function (collective reductions).

    Returns
    -------
    float
        :math:`N_v + \\sum_c f_c - \\sum_v (1 - f_v)`, the same on every rank.
    """
    kT = BOLTZMANN_EV_PER_K * max(float(temperature_k), ZERO_TEMPERATURE_FLOOR_K)
    log_electrons, log_holes = _log_occupation_sums(count, float(mu), kT, comm)
    return (
        _allreduce(count.valence_states, comm) + np.exp(log_electrons) - np.exp(log_holes)
        + _allreduce(count.electrons_outside, comm) - _allreduce(count.holes_outside, comm)
    )  # fmt: skip


def solve_charge_neutral_level(
    count: BandCount,
    target_states: float,
    temperature_k: float,
    window_ev: float,
    comm=None,
) -> float:
    r"""Chemical potential at which the states of ``count`` hold ``target_states`` electrons.

    Parameters
    ----------
    count : BandCount
        This rank's part of the band energies (:func:`collect_band_count`).
    target_states : float
        Occupied states summed over the whole grid and the spin channels (electrons divided by
        the spin factor).
    temperature_k : float
        Electronic temperature (K); values below ``ZERO_TEMPERATURE_FLOOR_K`` use the floor.
    window_ev : float
        The window of :func:`collect_band_count`.
    comm : MPI communicator, optional
        Every rank must call this function (collective reductions).

    Returns
    -------
    float
        Chemical potential :math:`\mu` (eV, relative to the DFT Fermi level), the same on every
        rank.

    Raises
    ------
    ValueError
        If :math:`\mu` reaches the edge of the window (the target is not within reach of the
        bands near the DFT level).

    Notes
    -----
    With :math:`N_v` valence-indexed states, the count
    :math:`N(\mu) = N_v + \sum_c f_c - \sum_v (1 - f_v)` equals the target
    :math:`N_v + \delta` when

    .. math::

        \sum_c f_c + \max(-\delta, 0) = \sum_v (1 - f_v) + \max(\delta, 0),

    solved by bisection on the difference of the logarithms of the two sides (increasing in
    :math:`\mu`).  Each side is a sum of positive terms evaluated with ``logaddexp``, so the
    condition stays exact when both are below the double-precision resolution of the total
    count (a gap at low temperature).
    """
    kT = BOLTZMANN_EV_PER_K * max(float(temperature_k), ZERO_TEMPERATURE_FLOOR_K)
    valence_total = _allreduce(count.valence_states, comm)
    excess = float(target_states) - valence_total
    electrons_floor = _allreduce(count.electrons_outside, comm) + max(-excess, 0.0)
    holes_floor = _allreduce(count.holes_outside, comm) + max(excess, 0.0)
    log_electrons_floor, log_holes_floor = _log(electrons_floor), _log(holes_floor)

    def balance(mu: float) -> float:
        log_electrons, log_holes = _log_occupation_sums(count, mu, kT, comm)
        return np.logaddexp(log_electrons, log_electrons_floor) - np.logaddexp(
            log_holes, log_holes_floor
        )

    edge = float(window_ev) - 30.0 * kT
    lower, upper = -edge, edge
    for _ in range(_BISECTION_STEPS):  # fixed count: the same collective calls on every rank
        middle = 0.5 * (lower + upper)
        if balance(middle) < 0.0:
            lower = middle
        else:
            upper = middle
    mu = 0.5 * (lower + upper)
    if abs(abs(mu) - edge) < 1.0e-6:
        raise ValueError(
            'the charge-neutral Fermi level reaches the edge of the %.2f eV window around the DFT '
            'Fermi level: the electron count (nelec, doping) does not match the PAO bands' % edge
        )
    return float(mu)


def charge_neutral_levels(
    HRs: NDArray[np.complex128],
    a_vectors: NDArray[np.float64],
    alat: float,
    orbital_positions_cryst: NDArray[np.float64] | None,
    nk: Sequence[int],
    temperatures_k: Sequence[float],
    nelec: float,
    spin_factor: float,
    doping: float = 0.0,
    comm=None,
    kblock: int = 256,
    hopping_cutoff: float | None = None,
    window_ev: float = DEFAULT_WINDOW_EV,
    check_nelec: bool = True,
) -> list[dict]:
    r"""Charge-neutral (or doped) Fermi level of the bands of ``HRs`` at each temperature.

    Parameters
    ----------
    HRs : ndarray ``(nawf, nawf, n1, n2, n3, nspin)``
        Real-space Hamiltonian (eV, relative to the DFT Fermi level).
    a_vectors : ndarray ``(3, 3)``
        Direct lattice vectors (rows, alat units).
    alat : float
        Lattice parameter (bohr).
    orbital_positions_cryst : ndarray ``(nawf, 3)`` or None
        Orbital centres (crystal coordinates) for the pair-resolved Wigner-Seitz cells.
    nk : sequence of int
        Dense grid of the count.
    temperatures_k : sequence of float
        Electronic temperatures (K).
    nelec : float
        Electrons per cell held by the PAO bands (all spin channels).
    spin_factor : float
        Electrons per state: 2 without spin polarisation or spin-orbit coupling, else 1.
    doping : float, optional
        Extra electrons per cell (negative for holes).
    comm : MPI communicator, optional
        Every rank must call this function.
    kblock : int, optional
        k-points per diagonalisation block.
    hopping_cutoff : float, optional
        As :func:`PAOFLOW.hamiltonian.bloch_derivatives.bloch_terms_from_HRs`.
    window_ev : float, optional
        Half-width of the window around the DFT level kept in memory (plus 50 kT).
    check_nelec : bool, optional
        Check that the bands hold about ``nelec`` electrons at the DFT Fermi level (for the DFT
        valence count; off for an explicit electron count, e.g. a doped model).

    Returns
    -------
    list of dict
        One record per temperature: ``temperature_k``, ``fermi_level_ev`` (relative to the DFT
        Fermi level), ``electrons`` (the target per cell), ``electrons_at_dft_level`` (the count at
        :math:`\mu = 0`, per cell), ``doping``, ``nk``, ``gapped`` (the electrons fill the bands up
        to a gap on the grid), ``valence_top_ev`` and ``conduction_bottom_ev`` (of the bands at
        the DFT Fermi level; ``None`` without valence or conduction bands).

    Raises
    ------
    ValueError
        With ``check_nelec``, if the bands hold, at the DFT Fermi level, more than
        ``NELEC_MISMATCH_LIMIT`` electrons more or fewer than ``nelec`` (states of the valence count
        outside the PAO basis, such as semicore shells: pass the electrons the basis holds); if
        :math:`\mu` leaves the window.

    Notes
    -----
    :math:`\mu(T)` solves

    .. math::

        \frac{s}{N_k} \sum_{n\mathbf k} f\!\left(\frac{\varepsilon_{n\mathbf k} - \mu}{k_B T}\right)
        = N_\mathrm{el} + \delta N,

    with :math:`s` the spin factor, :math:`f` the Fermi-Dirac function and :math:`\delta N` the
    doping, on the full grid ``nk`` (no symmetry reduction), by
    :func:`solve_charge_neutral_level`.
    """
    nk = tuple(int(n) for n in nk)
    nk_total = int(np.prod(nk))
    temperatures = [float(t) for t in np.atleast_1d(temperatures_k)]
    kT_max = BOLTZMANN_EV_PER_K * max(max(temperatures), ZERO_TEMPERATURE_FLOOR_K)
    window = float(window_ev) + 50.0 * kT_max
    valence_bands = valence_band_counts(
        HRs, a_vectors, alat, orbital_positions_cryst, hopping_cutoff=hopping_cutoff, comm=comm,
        kblock=kblock,
    )  # fmt: skip
    count = collect_band_count(
        HRs, a_vectors, alat, orbital_positions_cryst, nk, valence_bands, window, comm=comm,
        kblock=kblock, hopping_cutoff=hopping_cutoff,
    )  # fmt: skip
    valence_top = _allreduce(count.valence_top, comm, 'max')
    conduction_bottom = _allreduce(count.conduction_bottom, comm, 'min')
    neutral_states = float(nelec) / float(spin_factor) * nk_total
    target_states = (float(nelec) + float(doping)) / float(spin_factor) * nk_total
    # a gap at the cut of the electron count: the target fills the valence-indexed bands exactly
    gapped = bool(
        abs(target_states / nk_total - sum(valence_bands)) < 1.0e-6
        and conduction_bottom > valence_top
    )
    records = []
    for temperature in temperatures:
        states_at_zero = occupied_states(count, 0.0, temperature, comm=comm)
        mismatch = (states_at_zero - neutral_states) * float(spin_factor) / nk_total
        if check_nelec and abs(mismatch) > NELEC_MISMATCH_LIMIT:
            raise ValueError(
                'the PAO bands hold %.3f electrons per cell at the DFT Fermi level, not '
                'nelec = %g: states of the DFT valence are outside the PAO basis (semicore '
                'shells?). Pass the electrons the basis holds (nelec).'
                % (float(nelec) + mismatch, float(nelec))
            )
        mu = solve_charge_neutral_level(count, target_states, temperature, window, comm=comm)
        records.append(
            {
                'temperature_k': temperature,
                'fermi_level_ev': mu,
                'electrons': float(nelec) + float(doping),
                'electrons_at_dft_level': states_at_zero * float(spin_factor) / nk_total,
                'doping': float(doping),
                'nk': list(nk),
                'gapped': gapped,
                'valence_top_ev': valence_top if np.isfinite(valence_top) else None,
                'conduction_bottom_ev': (
                    conduction_bottom if np.isfinite(conduction_bottom) else None
                ),
            }
        )
    return records


def neutral_level_differs(fermi_level_ev: float, temperature_k: float) -> bool:
    """Whether a charge-neutral level is far enough from the DFT one to warn about it.

    Parameters
    ----------
    fermi_level_ev : float
        Charge-neutral level relative to the DFT Fermi level (eV).
    temperature_k : float
        Electronic temperature (K).

    Returns
    -------
    bool
        ``|mu| > max(5 meV, kT / 2)``.
    """
    return abs(float(fermi_level_ev)) > max(
        0.005, 0.5 * BOLTZMANN_EV_PER_K * max(float(temperature_k), 0.0)
    )
