"""Theory/experiment matching with explicit chance baselines.

On a dense multi-sheet Fermi surface a +/-5 % window around any frequency contains a
calculated orbit 30-40 % of the time, so every match rate is quoted against two
baselines: log-uniform random frequencies at the same angles, and the measured
peaks moved to random measured angles (angle-shuffled; it only rewards agreement
that is specific to each angle). Assignments are then made from branch shape over
the rotation, not from rates.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ONSAGER_SLOPE = 8.637  # T per meV per electron mass


def nearest_deviation(
    candidate_angles: NDArray[np.float64],
    candidate_frequencies: NDArray[np.float64],
    angle: float,
    frequency: float,
    angle_window: float,
) -> tuple[float, int]:
    """Relative deviation of the nearest candidate (in log F) within ``angle_window`` degrees.

    Returns
    -------
    tuple[float, int]
        ``(deviation, index)``; ``(inf, -1)`` when no candidate lies inside the window.
    """
    inside = np.abs(candidate_angles - angle) <= angle_window
    if not inside.any():
        return np.inf, -1
    indices = np.flatnonzero(inside)
    best = indices[np.argmin(np.abs(np.log(candidate_frequencies[indices]) - np.log(frequency)))]
    return (candidate_frequencies[best] - frequency) / frequency, int(best)


def match_rate(
    candidate_angles: NDArray[np.float64],
    candidate_frequencies: NDArray[np.float64],
    angles: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    angle_window: float,
    tolerance: float,
) -> tuple[float, NDArray[np.bool_]]:
    """Fraction of (angle, frequency) points with a candidate within ``tolerance``."""
    hits = np.array(
        [
            abs(nearest_deviation(candidate_angles, candidate_frequencies, a, f, angle_window)[0])
            <= tolerance
            for a, f in zip(angles, frequencies)
        ],
        dtype=bool,
    )
    return (float(hits.mean()) if len(hits) else float('nan')), hits


def chance_baselines(
    candidate_angles: NDArray[np.float64],
    candidate_frequencies: NDArray[np.float64],
    angles: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    angle_window: float,
    tolerance: float,
    measured_angles: NDArray[np.float64],
    trials: int = 200,
    seed: int = 7,
    frequency_range: tuple[float, float] = (15.0, 8000.0),
) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of the match rate for random frequencies and shuffled angles."""
    rng = np.random.default_rng(seed)
    log_low, log_high = np.log(frequency_range[0]), np.log(frequency_range[1])
    random_frequencies = [
        match_rate(candidate_angles, candidate_frequencies, angles,
                   np.exp(rng.uniform(log_low, log_high, len(frequencies))), angle_window, tolerance)[0]
        for _ in range(trials)
    ]  # fmt: skip
    shuffled_angles = [
        match_rate(candidate_angles, candidate_frequencies, rng.choice(measured_angles, len(frequencies)),
                   frequencies, angle_window, tolerance)[0]
        for _ in range(trials)
    ]  # fmt: skip
    return dict(
        loguniform=(float(np.mean(random_frequencies)), float(np.std(random_frequencies))),
        shuffled=(float(np.mean(shuffled_angles)), float(np.std(shuffled_angles))),
    )


def sweep_match(
    orbits: pd.DataFrame,
    peaks: pd.DataFrame,
    sweep_angle: str = 'polar',
    angle_window: float = 2.6,
    tolerance: float = 0.05,
    weak_threshold: float = 0.15,
    trials: int = 200,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Pointwise test of the significant sweep peaks against robust calculated orbits.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Per-peak table (nearest orbit, deviation, matched flag) and a summary with
        ``rate``, ``n``, ``loguniform`` and ``shuffled`` baselines.
    """
    significant = peaks.loc[peaks.rel_mag >= weak_threshold]
    orbit_angles = orbits[f'{sweep_angle}_deg'].to_numpy()
    orbit_frequencies = orbits.freq_T.to_numpy()
    rows = []
    for peak in significant.itertuples():
        deviation, index = nearest_deviation(orbit_angles, orbit_frequencies, peak.calc_angle,
                                             peak.freq_T, angle_window)  # fmt: skip
        orbit = orbits.iloc[index] if index >= 0 else None
        rows.append(
            dict(angle_deg=peak.angle_deg, calc_angle=peak.calc_angle, F_exp=peak.freq_T,
                 rel_mag=peak.rel_mag, F_calc=None if orbit is None else round(orbit.freq_T, 1),
                 group=None if orbit is None else orbit.group,
                 carrier=None if orbit is None else orbit.carrier,
                 mstar_band=None if orbit is None else round(orbit.mstar_abs, 3),
                 copies=None if orbit is None else int(orbit.num_orbit_copies),
                 dev_pct=round(100 * deviation, 1) if np.isfinite(deviation) else None,
                 matched=bool(np.isfinite(deviation) and abs(deviation) <= tolerance))
        )  # fmt: skip
    table = pd.DataFrame(rows)
    baselines = chance_baselines(orbit_angles, orbit_frequencies, significant.calc_angle.to_numpy(),
                                 significant.freq_T.to_numpy(), angle_window, tolerance,
                                 np.unique(peaks.calc_angle), trials, seed)  # fmt: skip
    return table, dict(rate=float(table.matched.mean()), n=len(table), **baselines)


def direction_candidates(
    sheets: pd.DataFrame, lk_rows: pd.DataFrame, tolerance: float = 0.05
) -> pd.DataFrame:
    """All sheets within ``tolerance`` of each measured LK frequency (ambiguities kept)."""
    rows = []
    for peak in lk_rows.itertuples():
        candidates = sheets.loc[(sheets.freq_T - peak.freq_T).abs() <= tolerance * peak.freq_T]
        if candidates.empty:
            nearest = sheets.iloc[(sheets.freq_T - peak.freq_T).abs().argsort()[:1]]
            rows.append(
                dict(label=peak.label, F_exp=peak.freq_T, sheet='UNMATCHED',
                     group=nearest.group.iloc[0] if len(nearest) else None,
                     F_calc=nearest.freq_T.iloc[0] if len(nearest) else None,
                     dev_pct=100 * (nearest.freq_T.iloc[0] - peak.freq_T) / peak.freq_T
                     if len(nearest) else None)
            )  # fmt: skip
            continue
        for sheet in candidates.itertuples():
            rows.append(
                dict(label=peak.label, F_exp=peak.freq_T, sheet=sheet.sheet, group=sheet.group,
                     F_calc=sheet.freq_T, dev_pct=100 * (sheet.freq_T - peak.freq_T) / peak.freq_T,
                     mstar_band=sheet.mstar_me, carrier=sheet.carrier, extremum=sheet.extremum,
                     copies=sheet.copies)
            )  # fmt: skip
    return pd.DataFrame(rows)


def anisotropy_candidates(
    sheets_reference: pd.DataFrame,
    sheets_compare: pd.DataFrame,
    lk_reference: pd.DataFrame,
    lk_compare: pd.DataFrame,
    window: float = 0.35,
    mass_weight: float = 0.2,
) -> pd.DataFrame:
    """Rank same-group sheet pairs by how well they reproduce the measured anisotropy.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``label`` and ``score``; an empty table with all columns when no
        label is measured in both directions with a same-group candidate.

    Notes
    -----
    .. math::

        s = \\left|\\frac{F_2^{c}}{F_1^{c}} - \\frac{F_2^{e}}{F_1^{e}}\\right|
            + \\frac{1}{2}\\sum_i \\frac{|F_i^{c} - F_i^{e}|}{F_i^{e}}
            + \\frac{w}{2}\\sum_i \\left|\\ln\\frac{m_i^{e}}{m_i^{c}}\\right|

    Every pair of orbits of the same group is tried: a warped sheet can have several
    extrema and the closest in frequency is often not the one tracking the anisotropy.
    """
    columns = ['label', 'group', 'F_ref_exp', 'F_ref_calc', 'F_cmp_exp', 'F_cmp_calc', 'ratio_exp',
               'ratio_calc', 'm_ref_exp', 'm_ref_band', 'm_cmp_exp', 'm_cmp_band', 'score']  # fmt: skip
    reference = {row.label: row for row in lk_reference.itertuples()}
    compare = {row.label: row for row in lk_compare.itertuples()}
    rows = []
    for label in sorted(set(reference) & set(compare)):
        peak_ref, peak_cmp = reference[label], compare[label]
        near_ref = sheets_reference.loc[
            (sheets_reference.freq_T - peak_ref.freq_T).abs() <= window * peak_ref.freq_T
        ]
        near_cmp = sheets_compare.loc[
            (sheets_compare.freq_T - peak_cmp.freq_T).abs() <= window * peak_cmp.freq_T
        ]
        for group in sorted(set(near_ref.group) & set(near_cmp.group)):
            pairs = product(near_ref.loc[near_ref.group == group].iterrows(),
                            near_cmp.loc[near_cmp.group == group].iterrows())  # fmt: skip
            for (_, sheet_ref), (_, sheet_cmp) in pairs:
                ratio_exp = peak_cmp.freq_T / peak_ref.freq_T
                ratio_calc = sheet_cmp.freq_T / sheet_ref.freq_T
                m_ref, m_cmp = (
                    getattr(peak_ref, 'mstar_me', np.nan),
                    getattr(peak_cmp, 'mstar_me', np.nan),
                )
                mass_penalty = 0.0
                if mass_weight and np.isfinite(m_ref) and np.isfinite(m_cmp):
                    mass_penalty = mass_weight * 0.5 * (abs(np.log(m_ref / sheet_ref.mstar_me))
                                                        + abs(np.log(m_cmp / sheet_cmp.mstar_me)))  # fmt: skip
                score = (abs(ratio_calc - ratio_exp)
                         + 0.5 * (abs(sheet_ref.freq_T - peak_ref.freq_T) / peak_ref.freq_T
                                  + abs(sheet_cmp.freq_T - peak_cmp.freq_T) / peak_cmp.freq_T)
                         + mass_penalty)  # fmt: skip
                rows.append(
                    dict(label=label, group=group, F_ref_exp=peak_ref.freq_T, F_ref_calc=sheet_ref.freq_T,
                         F_cmp_exp=peak_cmp.freq_T, F_cmp_calc=sheet_cmp.freq_T, ratio_exp=ratio_exp,
                         ratio_calc=ratio_calc, m_ref_exp=m_ref, m_ref_band=sheet_ref.mstar_me,
                         m_cmp_exp=m_cmp, m_cmp_band=sheet_cmp.mstar_me, score=score)
                )  # fmt: skip
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values(['label', 'score']).reset_index(drop=True)


def anisotropy_verdict(candidate) -> str:
    """Verbal reading of one anisotropy candidate (see the agent playbook, Stage 7)."""
    same_sign = (np.sign(candidate.ratio_calc - 1) == np.sign(candidate.ratio_exp - 1)
                 or abs(candidate.ratio_exp - 1) < 0.01)  # fmt: skip
    if abs(candidate.ratio_calc - candidate.ratio_exp) < 0.03:
        verdict = 'anisotropy reproduced'
    elif same_sign:
        verdict = 'right sign, wrong magnitude'
    else:
        verdict = 'WRONG SIGN - likely coincidental'
    dev_ref = 100 * (candidate.F_ref_calc - candidate.F_ref_exp) / candidate.F_ref_exp
    dev_cmp = 100 * (candidate.F_cmp_calc - candidate.F_cmp_exp) / candidate.F_cmp_exp
    if abs(dev_ref - dev_cmp) < 3 and min(abs(dev_ref), abs(dev_cmp)) > 5:
        verdict += '; uniform offset (right shape, wrong size)'
    return verdict


def mass_enhancement(experimental_mass, band_mass):
    """:math:`1 + \\lambda = m^*_{LK} / m^*_{band}`; > 1 and similar for well-assigned branches."""
    return np.asarray(experimental_mass, float) / np.asarray(band_mass, float)


def onsager_slope(band_mass, carrier) -> NDArray[np.float64]:
    """Rigid-band dF/dE in T/meV.

    Notes
    -----
    From :math:`F = \\hbar A / (2\\pi e)` and :math:`dA/dE = 2\\pi m^*/\\hbar^2`,

    .. math::

        \\frac{dF}{dE} = \\tau\\,\\frac{m^*}{m_e}\\times 8.637\\ \\mathrm{T/meV},

    with :math:`\\tau = +1` for electron and :math:`-1` for hole orbits.
    """
    sign = np.where(np.asarray(carrier) == 'e', 1.0, -1.0)
    return sign * np.asarray(band_mass, float) * ONSAGER_SLOPE


def fermi_shift_pairs(
    lower: pd.DataFrame,
    middle: pd.DataFrame,
    upper: pd.DataFrame,
    energy_step_meV: float,
    sweep_angle: str = 'polar',
    pairing_tolerance: float = 0.08,
    min_frequency: float = 100.0,
) -> pd.DataFrame:
    """Pair orbits of runs at E_F - dE, E_F and E_F + dE and compare dF/dE with Onsager.

    Returns
    -------
    pandas.DataFrame
        ``angle``, ``F0``, ``mstar``, ``carrier``, ``dFdE_meas``, ``dFdE_onsager``, ``ratio``.
        A median ratio near 1 confirms the energy units and a rigid response.
    """
    column = f'{sweep_angle}_deg'
    rows = []
    for angle in np.unique(middle[column]):
        centre = middle.loc[np.isclose(middle[column], angle) & (middle.freq_T > min_frequency)]
        low = lower.loc[np.isclose(lower[column], angle)]
        high = upper.loc[np.isclose(upper[column], angle)]
        for orbit in centre.itertuples():
            partners = []
            for shifted in (low, high):
                same_type = shifted.loc[shifted.orbit_type == orbit.orbit_type]
                if same_type.empty:
                    break
                partners.append(
                    same_type.freq_T.iloc[np.argmin(np.abs(same_type.freq_T - orbit.freq_T))]
                )
            if (
                len(partners) < 2
                or max(abs(f - orbit.freq_T) for f in partners) > pairing_tolerance * orbit.freq_T
            ):
                continue
            rows.append(
                dict(angle=angle, F0=orbit.freq_T, mstar=orbit.mstar_abs, carrier=orbit.carrier,
                     dFdE_meas=(partners[1] - partners[0]) / (2 * energy_step_meV),
                     dFdE_onsager=float(onsager_slope(orbit.mstar_abs, orbit.carrier)))
            )  # fmt: skip
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs['ratio'] = pairs.dFdE_meas / pairs.dFdE_onsager
    return pairs


def combination_test(
    peaks: pd.DataFrame,
    weak_threshold: float = 0.15,
    tolerance: float = 0.015,
    trials: int = 500,
    seed: int = 11,
    subset: str = 'weak',
) -> tuple[pd.DataFrame, dict]:
    """Are peaks harmonics or sums/differences of significant peaks at the same angle?

    Parameters
    ----------
    peaks : pandas.DataFrame
        ``angle_deg``, ``freq_T``, ``rel_mag``.
    weak_threshold : float, optional
        Peaks below this relative magnitude are weak.
    tolerance : float, optional
        Relative tolerance for ``2 F_i``, ``F_i + F_j`` and ``|F_i - F_j|``.
    trials, seed : int, optional
        Chance-baseline settings.
    subset : {'weak', 'significant'}
        Which peaks to test (a significant peak never explains itself).

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Per-peak table (``combination``, ``parent_i``, ``parent_j``, ``op``) and a summary
        with ``rate``, ``harmonics`` and both chance baselines.

    Notes
    -----
    A clear excess over chance indicates frequency mixing (harmonics, magnetic
    interaction, torque nonlinearity) and/or magnetic breakdown; attributing it to
    breakdown requires parents on touching sheets (:func:`combination_parents`).
    """
    significant = (peaks.rel_mag >= weak_threshold).to_numpy()
    tested = ~significant if subset == 'weak' else significant
    angles = peaks.angle_deg.to_numpy()
    frequencies = peaks.freq_T.to_numpy()
    measured_angles = np.unique(angles)

    def best(angle: float, frequency: float, exclude: int | None = None):
        parents = [i for i in np.flatnonzero(significant & (angles == angle)) if i != exclude]
        found = None
        for position, i in enumerate(parents):
            options = [(2 * frequencies[i], (i, i), '2x')]
            options += [
                (frequencies[i] + frequencies[j], (i, j), '+') for j in parents[position + 1 :]
            ]
            options += [
                (abs(frequencies[i] - frequencies[j]), (i, j), '-') for j in parents[position + 1 :]
            ]
            for value, pair, operation in options:
                deviation = abs(value - frequency) / frequency
                if (
                    value > 15
                    and deviation <= tolerance
                    and (found is None or deviation < found[0])
                ):
                    found = (deviation, pair, operation)
        return found

    rows = []
    for i in np.flatnonzero(tested):
        found = best(angles[i], frequencies[i], exclude=None if subset == 'weak' else i)
        if found is None:
            expression, pi, pj, op = '', None, None, None
        else:
            (pi, pj), op = found[1], found[2]
            expression = (f'2 x {frequencies[pi]:.0f}' if op == '2x'
                          else f'{frequencies[pi]:.0f} {op} {frequencies[pj]:.0f}')  # fmt: skip
        rows.append(dict(index=int(i), angle_deg=angles[i], freq_T=frequencies[i],
                         rel_mag=peaks.rel_mag.iloc[i], combination=expression,
                         parent_i=pi, parent_j=pj, op=op))  # fmt: skip
    table = pd.DataFrame(rows)
    hit = table.combination != ''
    rng = np.random.default_rng(seed)
    tested_frequencies = table.freq_T.to_numpy()
    shuffled = [np.mean([best(a, f) is not None
                         for a, f in zip(rng.choice(measured_angles, len(tested_frequencies)), tested_frequencies)])
                for _ in range(trials)]  # fmt: skip
    random_frequencies = [np.mean([best(a, f) is not None for a, f in
                                   zip(table.angle_deg, np.exp(rng.uniform(np.log(15), np.log(8000),
                                                                           len(tested_frequencies))))])
                          for _ in range(trials)]  # fmt: skip
    summary = dict(rate=float(hit.mean()), n=len(table), harmonics=int((table.op == '2x').sum()),
                   shuffled=(float(np.mean(shuffled)), float(np.std(shuffled))),
                   loguniform=(float(np.mean(random_frequencies)), float(np.std(random_frequencies))))  # fmt: skip
    return table, summary


def combination_parents(
    combinations: pd.DataFrame,
    peaks: pd.DataFrame,
    group_of_peak: Callable[[float, float], str | None],
    touching_groups: set[frozenset],
) -> pd.DataFrame:
    """Relation of the two parent sheets of every sum/difference combination.

    ``relation`` is ``'contact'`` (sheets touch: breakdown-compatible), ``'same sheet'`` or
    ``'no contact'`` (mixing). Only combinations whose two parents are assigned are kept.
    """
    rows = []
    for combo in combinations.loc[combinations.op.isin(['+', '-'])].itertuples():
        group_i = group_of_peak(combo.angle_deg, peaks.freq_T.iloc[int(combo.parent_i)])
        group_j = group_of_peak(combo.angle_deg, peaks.freq_T.iloc[int(combo.parent_j)])
        if group_i is None or group_j is None:
            continue
        if group_i == group_j:
            relation = 'same sheet'
        elif frozenset((group_i, group_j)) in touching_groups:
            relation = 'contact'
        else:
            relation = 'no contact'
        rows.append(dict(angle_deg=combo.angle_deg, freq_T=combo.freq_T, combination=combo.combination,
                         groups=f'{group_i}/{group_j}', relation=relation))  # fmt: skip
    return pd.DataFrame(rows)
