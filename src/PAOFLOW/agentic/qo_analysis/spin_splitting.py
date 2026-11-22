r"""Spin splitting of the calculated orbits against the measured quantum oscillations.

Reads the tables of :meth:`PAOFLOW.PAOFLOW.orbit_spin_splitting` (one
``orbit_spin_splitting_<band>.dat`` per Kramers pair, ``band`` the 1-based ``BAND:`` index of the
BXSF), attaches the pyskeaf properties of each orbit (sheet, group, carrier type, curvature,
copies) and compares the spin reduction factor of the first harmonic,

.. math::

    R_s = \cos\Bigl(\frac{\pi}{2}\,\frac{g^*m^*}{m_e}\Bigr),

with the measured peak strengths. Two forms of :math:`R_s` are carried: the adiabatic one (the
local Zeeman splitting accumulated along the orbit; what the areas of field-split Fermi surfaces
give) and the magnitude from the path-ordered propagator of the pair, which adds the non-Abelian
Berry transport (``absRs_<level>_prop``). The levels are ``spin`` (g_s S), ``spin_atomic``
(+ atomic L) and ``total`` (+ itinerant Bloch moment).

:math:`g^*m^*` here uses the band mass; electron-phonon renormalisation is taken to cancel in the
product and no exchange enhancement of :math:`g` is included.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ('spin', 'spin_atomic', 'total')
_BAND_LINE_RE = re.compile(r'^\s*BAND\s*:\s*(\d+)', re.IGNORECASE)
_SHEET_RE = re.compile(r'Fermi_surf_band_(\d+)\.bxsf(?:\.gz)?$')
_TABLE_RE = re.compile(r'orbit_spin_splitting_(\d+)\.dat$')


def read_spin_splitting(source: str | Path) -> pd.DataFrame:
    """Read ``orbit_spin_splitting_<band>.dat`` tables (a folder or one file).

    Parameters
    ----------
    source : str or Path
        Folder holding the tables, or one table.

    Returns
    -------
    DataFrame
        The columns of the tables plus ``pao_band`` (1-based, from the file name) and, per level,
        ``gm_<level>`` = g* m* / m_e.
    """
    path = Path(source)
    files = sorted(path.glob('orbit_spin_splitting_*.dat')) if path.is_dir() else [path]
    tables = []
    for file in files:
        match = _TABLE_RE.search(file.name)
        if not match:
            continue
        with open(file) as handle:
            columns = handle.readline().lstrip('#').split()
        table = pd.read_csv(file, sep=r'\s+', comment='#', header=None, names=columns)
        table['pao_band'] = int(match.group(1))
        tables.append(table)
    if not tables:
        raise FileNotFoundError(f'no orbit_spin_splitting_<band>.dat in {source}')
    table = pd.concat(tables, ignore_index=True)
    for level in LEVELS:
        table[f'gm_{level}'] = table[f'g_{level}'] * table.m_star
    table['freq_T'] = 1e3 * table.frequency_kT
    return table


def sheet_of_pao_band(bxsf_dir: str | Path) -> dict[int, int]:
    """``{pao_band: sheet}`` from the ``BAND:`` line of every ``Fermi_surf_band_<sheet>.bxsf``.

    ``pao_band`` is 1-based (the BXSF convention), ``sheet`` the pyskeaf band number.
    """
    import gzip

    mapping = {}
    for file in Path(bxsf_dir).glob('Fermi_surf_band_*.bxsf*'):
        match = _SHEET_RE.search(file.name)
        if not match:
            continue
        opener = gzip.open if file.name.endswith('.gz') else open
        with opener(file, 'rt') as handle:
            for line in handle:
                band = _BAND_LINE_RE.match(line)
                if band:
                    mapping[int(band.group(1))] = int(match.group(1))
                    break
    return mapping


def attach_orbits(
    spin: pd.DataFrame,
    orbits: pd.DataFrame,
    sheet_map: dict[int, int],
    sweep_angle: str = 'polar',
    angle_tolerance: float = 0.01,
    frequency_tolerance_T: float = 0.2,
) -> pd.DataFrame:
    """Attach sheet, group, carrier type, curvature and copies of the matching pyskeaf orbit.

    Parameters
    ----------
    spin : DataFrame
        Of :func:`read_spin_splitting`.
    orbits : DataFrame
        pyskeaf orbits at the same angles (``band``, ``group``, ``<sweep>_deg``, ``freq_T``,
        ``orbit_type``, ``curv_kTA2``, ``num_orbit_copies``).
    sheet_map : dict
        Of :func:`sheet_of_pao_band`.
    sweep_angle : {'polar', 'azimuthal'}
        The rotated angle.
    angle_tolerance : float
        Same-angle window (deg).
    frequency_tolerance_T : float
        Largest frequency difference (T); the spin-splitting tables print the frequency to 0.1 T.

    Returns
    -------
    DataFrame
        ``spin`` with ``sheet``, ``group``, ``orbit_type``, ``curv_kTA2``, ``num_orbit_copies``
        (``nan`` where no orbit matches).
    """
    table = spin.copy()
    table['sheet'] = table.pao_band.map(sheet_map)
    for column in ('group', 'orbit_type', 'curv_kTA2', 'num_orbit_copies'):
        table[column] = np.nan if column != 'group' else None
    angle_column = f'{sweep_angle}_deg'
    for index, row in table.iterrows():
        candidates = orbits.loc[
            (orbits.band == row.sheet)
            & ((orbits[angle_column] - row[sweep_angle]).abs() < angle_tolerance)
        ]
        if candidates.empty:
            continue
        nearest = (candidates.freq_T - row.freq_T).abs().idxmin()
        if abs(candidates.freq_T[nearest] - row.freq_T) <= frequency_tolerance_T:
            for column in ('group', 'orbit_type', 'curv_kTA2', 'num_orbit_copies'):
                table.at[index, column] = candidates.at[nearest, column]
    return table


def match_peaks(
    spin: pd.DataFrame,
    peaks: pd.DataFrame,
    sweep_angle: str = 'polar',
    tolerance: float = 0.05,
    angle_tolerance: float = 0.5,
) -> pd.DataFrame:
    """Nearest calculated orbit (within ``tolerance`` in frequency) of every measured peak.

    Only peaks at angles where spin factors exist are considered (``calc_angle`` within
    ``angle_tolerance`` of a calculated angle).

    Returns
    -------
    DataFrame
        One row per matched peak: ``angle``, ``freq_T`` and ``rel_mag`` of the peak, and the
        orbit's ``freq_calc_T``, ``group``, ``m_star``, ``absRs_<level>_prop``, ``Rs_<level>``.
    """
    rows = []
    angles = np.unique(spin[sweep_angle].round(4))
    for _, peak in peaks.iterrows():
        near = angles[np.abs(angles - peak.calc_angle) < angle_tolerance]
        if not near.size:
            continue
        candidates = spin.loc[(spin[sweep_angle].round(4) == near[0])]
        candidates = candidates.loc[
            (candidates.freq_T - peak.freq_T).abs() <= tolerance * peak.freq_T
        ]
        if candidates.empty:
            continue
        best = candidates.loc[(candidates.freq_T - peak.freq_T).abs().idxmin()]
        record = dict(angle=float(near[0]), freq_T=peak.freq_T, rel_mag=peak.rel_mag,
                      freq_calc_T=best.freq_T, group=best.get('group'), m_star=best.m_star)  # fmt: skip
        for level in LEVELS:
            record[f'Rs_{level}'] = best[f'Rs_{level}']
            record[f'absRs_{level}_prop'] = best[f'absRs_{level}_prop']
        rows.append(record)
    return pd.DataFrame(rows)


def rank_correlation_test(
    measured: np.ndarray, predicted: np.ndarray, trials: int = 5000, seed: int = 7
) -> dict:
    """Spearman correlation of two samples and its one-sided permutation p-value.

    Returns
    -------
    dict
        ``rho``, ``p_value`` (fraction of shuffles with a correlation at least as large),
        ``n``.
    """
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    keep = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[keep], predicted[keep]
    if measured.size < 3:
        return dict(rho=float('nan'), p_value=float('nan'), n=int(measured.size))

    def ranks(values: np.ndarray) -> np.ndarray:
        return pd.Series(values).rank().to_numpy()

    measured_rank, predicted_rank = ranks(measured), ranks(predicted)
    rho = float(np.corrcoef(measured_rank, predicted_rank)[0, 1])
    rng = np.random.default_rng(seed)
    shuffled = np.array(
        [np.corrcoef(measured_rank, rng.permutation(predicted_rank))[0, 1] for _ in range(trials)]
    )
    return dict(rho=rho, p_value=float(np.mean(shuffled >= rho)), n=int(measured.size))


def assigned_group(
    assignments: pd.DataFrame | None, label: str, angle: float, max_angle_gap: float = 30.0
) -> str | None:
    """Group assigned to an LK branch: the assignment row naming ``label`` nearest to ``angle``.

    ``assignments`` has ``label`` (one name or a comma-separated list), ``status``, ``group`` and
    ``arrow_angle_deg``; only ``assigned`` and ``tentative`` rows within ``max_angle_gap`` (deg) of
    ``angle`` count (an assignment made near one direction says nothing about another).
    """
    if assignments is None or not len(assignments):
        return None
    best, distance = None, np.inf
    for _, row in assignments.iterrows():
        if str(row.get('status', '')).strip() not in ('assigned', 'tentative') or pd.isna(
            row.get('group')
        ):
            continue
        names = [name.strip() for name in str(row.label).split(',')]
        if label not in names:
            continue
        gap = abs(float(row.get('arrow_angle_deg', angle)) - angle)
        if gap <= max_angle_gap and gap < distance:
            best, distance = str(row.group).replace(' ', ''), gap
    return best


def lk_branch_table(
    spin: pd.DataFrame,
    lk_rows: pd.DataFrame,
    angle: float,
    sweep_angle: str = 'polar',
    tolerance: float = 0.12,
    angle_tolerance: float = 0.5,
    assignments: pd.DataFrame | None = None,
    blind_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Calculated spin factors of the LK branches measured along one direction.

    Parameters
    ----------
    spin : DataFrame
        Of :func:`attach_orbits` (or :func:`read_spin_splitting`).
    lk_rows : DataFrame
        LK rows of the direction (``label``, ``freq_T``, ``mstar_me``).
    angle : float
        The direction's calculated sweep angle (deg).
    sweep_angle : {'polar', 'azimuthal'}
        The rotated angle.
    tolerance : float, optional
        Relative frequency window of an assigned branch, searched in its group (assigned
        branches can sit 10 % off the measured frequency).
    angle_tolerance : float, optional
        Same-angle window (deg).
    assignments : DataFrame, optional
        Branch assignments (:func:`assigned_group`); an assigned branch is looked for only in its
        group.
    blind_tolerance : float, optional
        Relative window of an unassigned branch, searched in every group.

    Returns
    -------
    DataFrame
        Per LK row: the assigned group (if any), the nearest orbit within ``tolerance``
        (``freq_calc_T``, ``group``, ``m_band``), the enhancement ``m_exp / m_band - 1``, g* and
        the spin factors.
    """
    at_angle = spin.loc[(spin[sweep_angle] - angle).abs() < angle_tolerance]
    rows = []
    for _, branch in lk_rows.iterrows():
        group = assigned_group(assignments, str(branch.label), angle)
        record = dict(
            label=branch.label, freq_T=branch.freq_T, m_exp=branch.mstar_me, assigned=group
        )
        candidates = at_angle if group is None else at_angle.loc[at_angle.group == group]
        window = blind_tolerance if group is None else tolerance
        near = candidates.loc[(candidates.freq_T - branch.freq_T).abs() <= window * branch.freq_T]
        if near.empty:
            rows.append(record)
            continue
        best = near.loc[(near.freq_T - branch.freq_T).abs().idxmin()]
        record.update(freq_calc_T=best.freq_T, shift_pct=100.0 * (best.freq_T / branch.freq_T - 1.0),
                      group=best.get('group'), m_band=best.m_star,
                      enhancement=branch.mstar_me / best.m_star - 1.0,
                      g_spin_atomic=best.g_spin_atomic, g_total=best.g_total,
                      Rs_total=best.Rs_total, absRs_total_prop=best.absRs_total_prop)  # fmt: skip
        rows.append(record)
    return pd.DataFrame(rows)


def plot_spin_factors(
    spin: pd.DataFrame,
    path: str | Path,
    sweep_angle: str = 'polar',
    colors: dict | None = None,
    labels: dict | None = None,
    title: str = '',
    axis_label: str | None = None,
) -> None:
    """g* and |R_s| of every orbit against the rotation angle, one colour per degenerate group."""
    from .qo_plots import _pyplot, group_colors

    plt = _pyplot()
    groups = [g for g in spin.group.dropna().unique()] or ['all']
    palette = group_colors(groups, colors)
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    for group in groups:
        subset = spin.loc[spin.group == group] if group != 'all' else spin
        size = 6 + 0.05 * subset.num_orbit_copies.fillna(50).clip(0, 400)
        label = (labels or {}).get(group, group)
        axes[0].scatter(
            subset[sweep_angle], subset.g_total, s=size, color=palette[group], label=label
        )
        axes[1].scatter(subset[sweep_angle], subset.Rs_total.abs(), s=size, color=palette[group])
        axes[2].scatter(subset[sweep_angle], subset.absRs_total_prop, s=size, color=palette[group])
    axes[0].set_ylabel('g* (spin + L + Bloch)')
    axes[1].set_ylabel('|R_s| adiabatic')
    axes[2].set_ylabel('|R_s| with Berry transport')
    for ax in axes[1:]:
        ax.set_ylim(-0.02, 1.02)
    axes[2].set_xlabel(axis_label or f'{sweep_angle} angle (deg)')
    axes[0].legend(fontsize=8, loc='upper right')
    if title:
        axes[0].set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


__all__ = [
    'LEVELS',
    'assigned_group',
    'attach_orbits',
    'lk_branch_table',
    'match_peaks',
    'plot_spin_factors',
    'rank_correlation_test',
    'read_spin_splitting',
    'sheet_of_pao_band',
]
