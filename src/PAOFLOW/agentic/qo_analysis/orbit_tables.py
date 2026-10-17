"""Tables of calculated extremal orbits: robustness cut, sheets, listings, run comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

COPY_BINS = [0, 2, 4, 10, 20, 50, 80, 100, 150, 200, 300, 500, 1000, 5000, 10**6]


def robust_orbits(
    orbits: pd.DataFrame, min_copies: int = 50, min_frequency: float = 5.0
) -> pd.DataFrame:
    """Keep orbits found in at least ``min_copies`` supercell copies and above ``min_frequency`` (T)."""
    keep = (orbits.num_orbit_copies >= min_copies) & (orbits.freq_T > min_frequency)
    return orbits.loc[keep].copy()


def copy_count_histogram(orbits: pd.DataFrame) -> pd.DataFrame:
    """Histogram of ``num_orbit_copies``, used to justify the robustness cut.

    Notes
    -----
    Genuine orbits are found in many supercell copies (100-500 at high interpolation)
    and wall artefacts in 1-10. At ``num_interpolation`` around 200-320 the
    distribution may not be bimodal: symmetry-paired, smoothly tracking genuine
    orbits can have 50-90 copies. Justify the cut from that evidence and repeat
    key results at a stricter cut.
    """
    counts, edges = np.histogram(orbits.num_orbit_copies, bins=COPY_BINS)
    return pd.DataFrame(
        dict(copies=[f'{int(a)}-{int(b)}' for a, b in zip(edges[:-1], edges[1:])], entries=counts)
    )


def orbits_at_angle(orbits: pd.DataFrame, angle: float, sweep_angle: str = 'polar') -> pd.DataFrame:
    """Orbits at the sweep grid point nearest to ``angle`` (empty if farther than ~half a step)."""
    column = f'{sweep_angle}_deg'
    grid = np.sort(orbits[column].unique())
    if not len(grid):
        return orbits.iloc[0:0]
    nearest = grid[np.argmin(np.abs(grid - angle))]
    half_step = 0.6 * np.median(np.diff(grid)) if len(grid) > 1 else 1e-3
    if abs(nearest - angle) > max(1e-3, half_step):
        return orbits.iloc[0:0]
    return orbits.loc[np.isclose(orbits[column], nearest, atol=1e-6)].sort_values('freq_T')


def merge_into_sheets(orbits: pd.DataFrame, merge_fraction: float = 0.015) -> pd.DataFrame:
    """Collapse duplicate orbits within each degenerate group into physical sheets.

    Parameters
    ----------
    orbits : pandas.DataFrame
        Robust orbits at one field direction (with a ``group`` column).
    merge_fraction : float, optional
        Relative frequency window (plus 0.5 T) around the cluster anchor.

    Returns
    -------
    pandas.DataFrame
        One row per sheet: ``sheet``, ``group``, ``freq_T``, ``spread_pct`` (numerical
        error bar), ``mstar_me``, ``carrier``, ``extremum``, ``copies``, ``amp_factor``,
        ``n_entries``.

    Notes
    -----
    Every candidate is compared with the cluster *anchor* (its lowest member), not with
    the last member, so a dense ladder of distinct orbits cannot chain into one wide
    sheet. Use about 0.015 when symmetry-equivalent orbits are distinct but close.
    """
    rows = []
    for group, members in orbits.groupby('group'):
        members = members.sort_values('freq_T')
        clusters, current = [], [members.iloc[0]]
        for _, orbit in list(members.iterrows())[1:]:
            if abs(orbit.freq_T - current[0].freq_T) <= merge_fraction * current[0].freq_T + 0.5:
                current.append(orbit)
            else:
                clusters.append(current)
                current = [orbit]
        clusters.append(current)
        for cluster in clusters:
            frequencies = np.array([orbit.freq_T for orbit in cluster])
            carriers = [orbit.carrier for orbit in cluster]
            extrema = [orbit.extremum for orbit in cluster]
            rows.append(
                dict(
                    group=group,
                    freq_T=float(frequencies.mean()),
                    spread_pct=float(100 * np.ptp(frequencies) / frequencies.mean()),
                    mstar_me=float(np.mean([orbit.mstar_abs for orbit in cluster])),
                    carrier=max(set(carriers), key=carriers.count),
                    extremum=max(set(extrema), key=extrema.count),
                    copies=int(np.mean([orbit.num_orbit_copies for orbit in cluster])),
                    amp_factor=float(np.mean([orbit.amp_factor for orbit in cluster])),
                    n_entries=len(cluster),
                )
            )
    sheets = pd.DataFrame(rows).sort_values('freq_T').reset_index(drop=True)
    sheets.insert(0, 'sheet', [f'S{i + 1}' for i in range(len(sheets))])
    return sheets


def per_angle_listing(
    orbits: pd.DataFrame,
    peaks: pd.DataFrame,
    angles: list[float],
    sweep_angle: str = 'polar',
    weak_threshold: float = 0.15,
    min_frequency: float = 12.0,
) -> str:
    """Side-by-side text of measured and calculated frequencies at every measured angle.

    Calculated entries are written ``<F><carrier><group>``; near-duplicates (< 1.5 %)
    within a group are collapsed. This plain listing is the most useful tool for
    reasoning about branch assignments. ``peaks`` needs ``calc_angle``, ``freq_T``
    and ``rel_mag``.
    """
    lines = []
    for angle in angles:
        calculated, last = [], None
        for orbit in orbits_at_angle(orbits, angle, sweep_angle).itertuples():
            if orbit.freq_T < min_frequency:
                continue
            if last and orbit.group == last[0] and abs(orbit.freq_T - last[1]) / last[1] < 0.015:
                continue
            calculated.append(f'{orbit.freq_T:.0f}{orbit.carrier}{orbit.group}')
            last = (orbit.group, orbit.freq_T)
        here = peaks.loc[(peaks.calc_angle - angle).abs() < 1e-6]
        strong = ' '.join(f'{f:.0f}' for f in here.loc[here.rel_mag >= weak_threshold].freq_T)
        weak = ' '.join(f'{f:.0f}' for f in here.loc[here.rel_mag < weak_threshold].freq_T)
        lines.append(f'{angle:6.1f} EXP : {strong}' + (f'   (weak: {weak})' if weak else ''))
        lines.append('       CALC: ' + ' '.join(calculated))
    return '\n'.join(lines)


def angular_coverage(
    orbits: pd.DataFrame, min_frequency: float, max_frequency: float, sweep_angle: str = 'polar'
) -> list[int]:
    """Rounded sweep angles at which any orbit lies between the two frequencies (T)."""
    window = orbits.loc[orbits.freq_T.between(min_frequency, max_frequency)]
    return sorted(set(np.round(window[f'{sweep_angle}_deg']).astype(int)))


def orbit_shifts(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    angles: list[float],
    sweep_angle: str = 'polar',
    min_frequency: float = 100.0,
    pairing_tolerance: float = 0.15,
) -> pd.DataFrame:
    """Orbit-by-orbit relative shift of ``other`` with respect to ``reference``.

    Each reference orbit above ``min_frequency`` is paired with the nearest orbit of the
    same carrier type at the same angle, if within ``pairing_tolerance``.
    """
    rows = []
    for angle in angles:
        ref_here = orbits_at_angle(reference, angle, sweep_angle)
        other_here = orbits_at_angle(other, angle, sweep_angle)
        for orbit in ref_here.loc[ref_here.freq_T > min_frequency].itertuples():
            same_carrier = other_here.loc[other_here.carrier == orbit.carrier]
            if same_carrier.empty:
                continue
            nearest = (same_carrier.freq_T - orbit.freq_T).abs().idxmin()
            shift = (same_carrier.loc[nearest, 'freq_T'] - orbit.freq_T) / orbit.freq_T
            if abs(shift) <= pairing_tolerance:
                rows.append(
                    dict(
                        angle=angle,
                        F_reference=orbit.freq_T,
                        F_other=same_carrier.loc[nearest, 'freq_T'],
                        shift_pct=100 * shift,
                        group_reference=orbit.group,
                        group_other=same_carrier.loc[nearest, 'group'],
                        carrier=orbit.carrier,
                    )
                )
    return pd.DataFrame(rows)
