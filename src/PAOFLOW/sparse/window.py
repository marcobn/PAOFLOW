"""The energy window of one sparse solve, derived from the property ranges.

There is no window option. Before each pass the engine widens every property's
:meth:`~PAOFLOW.sparse.properties.MeshProperty.energy_range` by its margin and solves either
from the bottom up to ``min(usable top, max(emax + margin))`` (usable top = PAO shift minus
``shift_margin``, ``SparseEngine._projectable_top``), or, under ``interior_window={}``, inside
``[min(emin - margin), max(emax + margin)]``.

The margins are physical, not tunable:

- thermal (transport, doping): ``10 kB tmax``, the reach of ``-df/dE`` past every ``mu``;
- smearing (DoS, PDoS, conductivity, intra-band Rashba-Edelstein): ``4 sigma_max``, since a
  state four adaptive widths ``sigma = afac |v| dk`` away still contributes. ``sigma_max`` is
  estimated at the :func:`probe_points` (``SparseEngine._probe``); after the pass a warning
  names the shortfall if solved states end under ``SIGMA_CHECK`` measured widths past the range.

Arithmetic only (no MPI, no Hamiltonian), so it tests on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

K_BOLTZMANN_EV = 8.617333e-5
# occupation-derivative reach, in units of kT
KT_FACTOR = 10.0
# Gaussian reach, in units of the largest adaptive width, used to place the
# window from the widths estimated at the probe points
SIGMA_FACTOR = 4.0
# reach below which the widths measured on the whole mesh trigger a warning
# after the pass: a Gaussian tail 3 widths out is ~1% of its peak
SIGMA_CHECK = 3.0
# k-points at which the window is probed: Gamma and the 7 zone corners
# (band extrema, for the state count), then generic points of the R3
# low-discrepancy sequence (for the band velocities, which vanish on the
# symmetry points; on example01 eight of them find the steepest band of a
# 24^3 mesh to 5%, against 34% for strided mesh points)
NPROBE = 16
# R3 sequence: the powers of 1/phi_3, phi_3 the real root of x^4 = x + 1
R3_ALPHA = (0.8191725133961645, 0.6710436067037893, 0.5497004779019703)
# padding of the probed band count: max(PAD_MIN, PAD_FRACTION * count)
PAD_MIN = 8
PAD_FRACTION = 0.02

SMEARING = 'smearing'
THERMAL = 'thermal'
NONE = 'none'


def probe_points() -> list[tuple[float, float, float]]:
    """The ``NPROBE`` probe k-points, in crystal coordinates."""
    corners = [(x, y, z) for x in (0.0, 0.5) for y in (0.0, 0.5) for z in (0.0, 0.5)]
    generic = [
        tuple((0.5 + i * a) % 1.0 for a in R3_ALPHA) for i in range(1, NPROBE - len(corners) + 1)
    ]
    return corners + generic


def kt_margin(tmax: float) -> float:
    """Thermal margin (eV): ``KT_FACTOR * kB * tmax``."""
    return KT_FACTOR * K_BOLTZMANN_EV * float(tmax)


def smearing_margin(dmax: float) -> float:
    """Smearing margin (eV) for a largest adaptive width ``dmax`` (eV)."""
    return SIGMA_FACTOR * float(dmax)


def padded_count(count: int) -> int:
    """A probed band count with the padding added."""
    count = int(count)
    return count + max(PAD_MIN, math.ceil(PAD_FRACTION * count))


@dataclass(frozen=True)
class Range:
    """The energy range one property of a pass needs, margin resolved.

    Attributes
    ----------
    method : str
        PAOFLOW method name, for the log.
    emin, emax : float
        Requested range (eV); ``emin`` may be ``-inf`` (e.g. the dielectric tensor).
    kind : {'smearing', 'thermal', 'none'}
        Which margin applies.
    margin : float
        Margin in eV; ``0.0`` for ``'none'`` or a ``'smearing'`` range with no width estimate.
    tmax : float or None
        Highest temperature (K) of a ``'thermal'`` range.
    """

    method: str
    emin: float
    emax: float
    kind: str
    margin: float
    tmax: float | None = None

    @property
    def lo(self) -> float:
        return self.emin - self.margin

    @property
    def hi(self) -> float:
        return self.emax + self.margin


@dataclass(frozen=True)
class PassWindow:
    """The resolved window of one pass.

    Attributes
    ----------
    mode : {'bottom', 'interior'}
        From the bottom of the spectrum, or inside ``[elo, ehi]``.
    elo : float or None
        Bottom of an interior window (eV).
    ehi : float or None
        Window top (eV); ``None`` with neither a PAO shift nor a property range.
    nev : int or None
        Bands per k of a from-the-bottom solve; ``None`` under interior (varies with k).
    at_top : bool
        ``nev`` is the projectable band count (all real states, as dense); coverage checks no-op.
    ranges : tuple of Range
        The property ranges the window was built from.
    nev_source : str
        ``'projectable'``, ``'probe'``, ``'given'`` or ``'interior'``.
    probed : int or None
        Most states below ``ehi`` found at a probe k-point.
    """

    mode: str
    elo: float | None
    ehi: float | None
    nev: int | None
    at_top: bool
    ranges: tuple = ()
    nev_source: str = 'projectable'
    probed: int | None = None

    @property
    def interior(self) -> tuple[float, float] | None:
        return (self.elo, self.ehi) if self.mode == 'interior' else None

    def margin(self, kind: str, tmax: float | None = None) -> float:
        """The margin of ``kind`` used for this window (eV)."""
        for r in self.ranges:
            if r.kind == kind and (kind != THERMAL or r.tmax == tmax):
                return r.margin
        if kind == THERMAL and tmax is not None:
            return kt_margin(tmax)
        return 0.0


def derive_bottom(ranges: list[Range], ptop: float | None) -> tuple[float | None, bool]:
    """Top of a from-the-bottom window: ``(min(ptop, max(r.hi)), whether that is ptop)``.

    Without ranges it is the whole well-projected range, ``(ptop, True)``.
    """
    if not ranges:
        return ptop, True
    top = max(r.hi for r in ranges)
    if ptop is None:
        return top, False
    if top >= ptop:
        return ptop, True
    return top, False


def derive_interior(ranges: list[Range], ptop: float | None) -> tuple[float, float]:
    """Bounds of an automatic interior window.

    Raises
    ------
    ValueError
        Without a property range, or if the window lies wholly above ``ptop``.
    """
    if not ranges:
        raise ValueError(
            'sparse interior_window={}: the bounds come from the property ranges, and no '
            'property of this pass has one (bands() and the path properties have none). Call '
            "bands() after a property, or give interior_window={'elo': ..., 'ehi': ...}."
        )
    elo = min(r.lo for r in ranges)
    ehi = max(r.hi for r in ranges)
    if ptop is not None:
        if elo >= ptop:
            raise ValueError(
                f'sparse interior window: [{elo:.3f}, {ehi:.3f}] eV lies above the '
                f'well-projected bands, which end at {ptop:.3f} eV. Lower the property ranges, '
                'or raise the PAO shift with more bands or a lower pthr in projectability().'
            )
        ehi = min(ehi, ptop)
    return elo, ehi
