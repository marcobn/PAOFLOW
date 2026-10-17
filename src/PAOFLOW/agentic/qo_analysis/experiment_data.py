"""Experimental quantum-oscillation data in any form, converted to two canonical tables.

Canonical tables (CSV; ``#`` lines hold the provenance):

1. LK table (single directions)::

       direction,exp_angle_deg,day,label,freq_T,freq_err_T,mstar_me,mstar_err_me,observed,source

   one row per (direction, label); ``observed = no`` for peaks looked for and not seen.
2. Sweep peaks (rotations)::

       angle_deg,freq_T,rel_mag

   ``angle_deg`` is the experimental rotation angle; ``rel_mag`` the FFT magnitude
   normalised to the largest in the data set.

Sources: tables are renamed into the schema; free text goes through
:func:`parse_text_peaks` and is then checked by hand; images of FFT-vs-angle maps
go through :func:`digitize_fft_map` and :func:`validate_digitization`; slide decks
are unpacked with :func:`extract_slide_deck`. Always label digitised or
transcribed numbers as such.
"""

from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GREEK_NAMES = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon', 'ζ': 'zeta',
    'η': 'eta', 'θ': 'theta', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi',
    'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi',
    'ω': 'omega',
}  # fmt: skip
_NUMBER = r'([0-9]+(?:\.[0-9]+)?)'


def read_canonical_table(path: str | Path) -> pd.DataFrame:
    """Read a canonical CSV (``#`` comment lines allowed)."""
    return pd.read_csv(path, comment='#')


def read_sweep_peaks(
    path: str | Path, angle_scale: float = 1.0, angle_offset: float = 0.0
) -> pd.DataFrame:
    """Sweep peaks with a ``calc_angle = angle_scale * angle_deg + angle_offset`` column."""
    peaks = read_canonical_table(path)
    if 'rel_mag' not in peaks:
        peaks['rel_mag'] = 1.0
    peaks['calc_angle'] = angle_scale * peaks.angle_deg + angle_offset
    return peaks


def read_lk_table(path: str | Path) -> pd.DataFrame:
    """LK table with a default ``observed = 'yes'`` column."""
    table = read_canonical_table(path)
    if 'observed' not in table:
        table['observed'] = 'yes'
    return table


def lk_rows_for_direction(lk_table: pd.DataFrame, direction: dict) -> pd.DataFrame:
    """Observed LK rows of one direction (and measurement ``day``, if the direction gives one)."""
    rows = lk_table.loc[
        (lk_table.direction.astype(str) == str(direction['label']))
        & (lk_table.observed.astype(str).str.lower() == 'yes')
    ]
    if direction.get('day') is not None and 'day' in rows:
        rows = rows.loc[pd.to_numeric(rows.day, errors='coerce') == float(direction['day'])]
    return rows


def parse_text_peaks(text: str) -> pd.DataFrame:
    """First-pass extraction of label, frequency and mass statements from free text.

    Recognises forms such as ``gamma 16.5 T, m* = 0.285``, ``β: F = 292 T (0.60 me)``,
    ``delta2 at 2.194 kT with mass 6.60`` and Markdown table rows ``| alpha | 488 | 0.575 |``.
    Greek letters are normalised to names; kT is converted to T.

    Returns
    -------
    pandas.DataFrame
        ``label``, ``freq_T``, ``mstar_me`` (NaN if absent), ``source_line``. Every
        row must be checked against the source before entering the LK table.
    """
    rows = []
    for raw in text.splitlines():
        line = raw
        for letter, name in GREEK_NAMES.items():
            line = line.replace(letter, name)
        match = re.search(r'\b([a-z]+[0-9]?)\b[^0-9]{0,20}' + _NUMBER + r'\s*(k?T)\b', line, re.I)
        if not match:
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if (
                len(cells) >= 2
                and re.fullmatch(r'[a-zA-Z]+[0-9]?', cells[0] or '')
                and re.fullmatch(_NUMBER, cells[1] or '')
            ):
                mass = (
                    float(cells[2])
                    if len(cells) > 2 and re.fullmatch(_NUMBER, cells[2])
                    else np.nan
                )
                rows.append(
                    dict(label=cells[0].lower(), freq_T=float(cells[1]), mstar_me=mass,
                         source_line=raw.strip())
                )  # fmt: skip
            continue
        frequency = float(match[2]) * (1000.0 if match[3].lower() == 'kt' else 1.0)
        mass_match = re.search(r'(?:m\*|mstar|mass)[^0-9]{0,8}' + _NUMBER, line, re.I) or re.search(
            _NUMBER + r'\s*m_?e\b', line, re.I
        )
        rows.append(
            dict(label=match[1].lower(), freq_T=frequency,
                 mstar_me=float(mass_match[1]) if mass_match else np.nan, source_line=raw.strip())
        )  # fmt: skip
    return pd.DataFrame(rows)


def digitize_fft_map(
    image: str | Path,
    frequency_axis: str,
    frequency_pixels: tuple[float, float],
    frequency_values: tuple[float, float],
    angle_pixels: tuple[float, float],
    angle_values: tuple[float, float],
    data_pixel_range: tuple[int, int],
    colorbar: dict,
    angles: list[float],
    log_frequency: bool = True,
    half_width: int = 3,
    prominence: float | None = None,
    min_frequency: float = 0.0,
    pixel_nudges: dict | None = None,
) -> pd.DataFrame:
    """Digitise an FFT-magnitude map (frequency along one image axis, angle along the other).

    Parameters
    ----------
    image : str or Path
        PNG/JPG of the map, in its original resolution.
    frequency_axis : {'x', 'y'}
        Image axis that carries the frequency.
    frequency_pixels, frequency_values : tuple of float
        Pixel positions of two frequency ticks and their values (T).
    angle_pixels, angle_values : tuple of float
        Pixel positions of two angle ticks and their values (deg).
    data_pixel_range : tuple of int
        Pixel range along the frequency axis that contains data.
    colorbar : dict
        Vertical bar ``{'x': column, 'y0': pixel of 0, 'y1': pixel of vmax, 'vmax': value}``
        or horizontal bar ``{'y': row, 'x0': ..., 'x1': ..., 'vmax': ...}``.
    angles : list of float
        Measured angles to sample (contour maps interpolate between them).
    log_frequency : bool, optional
        Whether the frequency axis is logarithmic.
    half_width : int, optional
        Rows/columns averaged around each angle.
    prominence : float or None, optional
        Peak prominence in magnitude units; default 4 % of ``vmax``.
    min_frequency : float, optional
        Discard peaks below this frequency (T).
    pixel_nudges : dict or None, optional
        ``{angle: pixel offset}`` for rows that lie on an axis or frame line.

    Returns
    -------
    pandas.DataFrame
        ``angle_deg``, ``freq_T``, ``magnitude``, ``rel_mag``, ``prominence``.

    Notes
    -----
    Pixel colours are mapped to magnitude through the colour bar (nearest colour).
    Dark grey contour lines (R=G=B < 110) are masked and interpolated over; the light
    grey used for the top of the scale is kept.
    """
    from PIL import Image
    from scipy.signal import find_peaks

    pixels = np.asarray(Image.open(image).convert('RGB')).astype(float)
    f_pix, f_val = np.asarray(frequency_pixels, float), np.asarray(frequency_values, float)
    a_pix, a_val = np.asarray(angle_pixels, float), np.asarray(angle_values, float)
    vmax = float(colorbar['vmax'])
    if 'x' in colorbar:
        positions = np.arange(
            min(colorbar['y0'], colorbar['y1']), max(colorbar['y0'], colorbar['y1']) + 1
        )
        bar_colours = pixels[positions, int(colorbar['x'])]
        bar_values = vmax * (colorbar['y0'] - positions) / (colorbar['y0'] - colorbar['y1'])
    else:
        positions = np.arange(
            min(colorbar['x0'], colorbar['x1']), max(colorbar['x0'], colorbar['x1']) + 1
        )
        bar_colours = pixels[int(colorbar['y']), positions]
        bar_values = vmax * (positions - colorbar['x0']) / (colorbar['x1'] - colorbar['x0'])

    def pixel_to_frequency(pixel):
        fraction = (np.asarray(pixel, float) - f_pix[0]) / (f_pix[1] - f_pix[0])
        if log_frequency:
            return 10 ** (np.log10(f_val[0]) + fraction * (np.log10(f_val[1]) - np.log10(f_val[0])))
        return f_val[0] + fraction * (f_val[1] - f_val[0])

    nudges = {str(key): float(value) for key, value in (pixel_nudges or {}).items()}
    low, high = (int(value) for value in data_pixel_range)
    prominence = 0.04 * vmax if prominence is None else float(prominence)
    across = pixels.shape[0] if frequency_axis == 'x' else pixels.shape[1]
    rows = []
    for angle in angles:
        centre = a_pix[0] + (angle - a_val[0]) * (a_pix[1] - a_pix[0]) / (a_val[1] - a_val[0])
        centre += nudges.get(str(angle), nudges.get(str(float(angle)), 0.0))
        centre = int(min(max(round(centre), half_width), across - half_width - 1))
        if frequency_axis == 'x':
            strip = pixels[centre - half_width : centre + half_width + 1, low:high]
        else:
            strip = np.transpose(
                pixels[low:high, centre - half_width : centre + half_width + 1], (1, 0, 2)
            )
        grey = np.ptp(strip, axis=2) < 12
        contour_line = grey & (strip.mean(axis=2) < 110)
        distance = ((strip[:, :, None, :] - bar_colours[None, None, :, :]) ** 2).sum(-1)
        magnitude = bar_values[distance.argmin(-1)]
        magnitude[contour_line] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # columns that are all contour line
            profile = np.nanmedian(magnitude, axis=0)
        valid = np.isfinite(profile)
        profile = np.interp(np.arange(profile.size), np.flatnonzero(valid), profile[valid])
        peak_index, properties = find_peaks(profile, prominence=prominence, distance=8)
        frequencies = pixel_to_frequency(np.arange(low, high))
        for index, peak_prominence in zip(peak_index, properties['prominences']):
            rows.append(
                dict(angle_deg=angle, freq_T=round(float(frequencies[index]), 1),
                     magnitude=float(profile[index]), rel_mag=round(float(profile[index]) / vmax, 4),
                     prominence=float(peak_prominence))
            )  # fmt: skip
    peaks = pd.DataFrame(rows)
    return peaks.loc[peaks.freq_T >= min_frequency].reset_index(drop=True)


def validate_digitization(
    sweep_peaks: pd.DataFrame, lk_rows: pd.DataFrame, tolerance: float = 0.03
) -> pd.DataFrame:
    """Nearest digitised peak for every observed LK frequency at the same angle.

    A median deviation of about 1-2 % or less means the axis calibration is right;
    systematic deviations point to wrong tick pixels.
    """
    rows = []
    observed = lk_rows.loc[lk_rows.observed.astype(str).str.lower() == 'yes']
    for entry in observed.itertuples():
        here = sweep_peaks.loc[(sweep_peaks.angle_deg - entry.exp_angle_deg).abs() < 1e-6]
        if here.empty or not np.isfinite(entry.freq_T):
            continue
        nearest = (np.log(here.freq_T) - np.log(entry.freq_T)).abs().idxmin()
        deviation = (here.loc[nearest, 'freq_T'] - entry.freq_T) / entry.freq_T
        rows.append(
            dict(angle=entry.exp_angle_deg, label=entry.label, F_table=entry.freq_T,
                 F_digitised=here.loc[nearest, 'freq_T'], dev_pct=round(100 * deviation, 2),
                 rel_mag=here.loc[nearest, 'rel_mag'], ok=abs(deviation) <= tolerance)
        )  # fmt: skip
    return pd.DataFrame(rows)


def regular_peak_runs(
    sweep_peaks: pd.DataFrame,
    min_run: int = 4,
    spacing_tolerance: float = 0.15,
    max_spacing_fraction: float = 0.1,
) -> pd.DataFrame:
    """Runs of evenly spaced peaks within one angle (possible contour-image striping).

    Only runs whose spacing is small compared with the frequency
    (``< max_spacing_fraction * F``) are reported.
    """
    rows = []
    for angle, here in sweep_peaks.groupby('angle_deg'):
        frequencies = np.sort(here.freq_T.to_numpy())
        steps = np.diff(frequencies)
        start = 0
        while start < len(steps):
            end = start
            while (
                end + 1 < len(steps)
                and abs(steps[end + 1] - steps[start]) <= spacing_tolerance * steps[start]
            ):
                end += 1
            spacing = float(np.mean(steps[start : end + 1]))
            if end - start + 2 >= min_run and spacing < max_spacing_fraction * frequencies[start]:
                rows.append(
                    dict(angle=angle, n=end - start + 2, f_from=frequencies[start],
                         f_to=frequencies[end + 1], spacing=spacing)
                )  # fmt: skip
            start = end + 1
    return pd.DataFrame(rows)


def extract_slide_deck(deck: str | Path, destination: str | Path) -> Path:
    """Dump slide text and images of a ``.pptx`` deck.

    Returns
    -------
    Path
        ``destination/slides.md`` (one section per slide: text and image names); the
        images are written to ``destination/media/``.
    """
    destination = Path(destination)
    (destination / 'media').mkdir(parents=True, exist_ok=True)
    lines = [f'# Slides of {Path(deck).name}', '']
    with zipfile.ZipFile(deck) as archive:
        names = archive.namelist()
        slides = sorted(
            (name for name in names if re.fullmatch(r'ppt/slides/slide\d+\.xml', name)),
            key=lambda name: int(re.findall(r'\d+', name)[-1]),
        )
        for name in names:
            if name.startswith('ppt/media/'):
                (destination / 'media' / Path(name).name).write_bytes(archive.read(name))
        for slide in slides:
            number = int(re.findall(r'\d+', slide)[-1])
            text = ' | '.join(
                re.findall(r'<a:t>([^<]*)</a:t>', archive.read(slide).decode('utf8', 'ignore'))
            )
            relations = f'ppt/slides/_rels/slide{number}.xml.rels'
            media = (
                re.findall(r'media/([^"]+)', archive.read(relations).decode('utf8', 'ignore'))
                if relations in names
                else []
            )
            lines += [f'## Slide {number}', '', text or '(no text)', '',
                      'images: ' + (', '.join(media) if media else 'none'), '']  # fmt: skip
    (destination / 'slides.md').write_text('\n'.join(lines))
    return destination / 'slides.md'
