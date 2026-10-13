import zipfile

import numpy as np
import pandas as pd
import pytest

from PAOFLOW.qo_analysis.experiment_data import (
    digitize_fft_map,
    extract_slide_deck,
    lk_rows_for_direction,
    parse_text_peaks,
    regular_peak_runs,
    validate_digitization,
)

pytestmark = pytest.mark.unit


def test_text_parser_handles_common_forms():
    text = (
        'gamma 22.7 T, m* = 0.581(43)\n'
        'β: F = 312.6 T (0.706 me)\n'
        '| kappa | 1370 | 1.35 |\n'
        'delta2 at 2.194 kT with mass 6.60\n'
        'no numbers here'
    )
    parsed = parse_text_peaks(text).set_index('label')
    assert parsed.loc['gamma', 'freq_T'] == 22.7 and parsed.loc['gamma', 'mstar_me'] == 0.581
    assert parsed.loc['beta', 'mstar_me'] == 0.706
    assert parsed.loc['kappa', 'freq_T'] == 1370.0
    assert parsed.loc['delta2', 'freq_T'] == 2194.0


def synthetic_map(path, peaks_by_angle):
    """F on x (log, 10 T at x=100, 10 kT at x=700), angle on y (0 deg at y=280, 90 at y=100),
    colour bar in column 800 (value 0 at y=300, vmax at y=20)."""
    import matplotlib

    cmap = matplotlib.colormaps['jet']
    image = np.zeros((320, 820, 3), np.uint8)
    rows = np.arange(20, 301)
    image[rows, 795:806] = (np.array([cmap((300 - y) / 280)[:3] for y in rows]) * 255).astype(
        np.uint8
    )[:, None, :]
    background = (np.array(cmap(0.0)[:3]) * 255).astype(np.uint8)
    image[90:300, 100:701] = background
    for angle, frequencies in peaks_by_angle.items():
        row = int(round(280 - 2 * angle))
        x = np.arange(100, 701)
        value = np.zeros(x.size)
        for f in frequencies:
            centre = 100 + 600 * (np.log10(f) - 1) / 3
            value = np.maximum(value, 0.8 * np.exp(-0.5 * ((x - centre) / 3.0) ** 2))
        colours = (np.array([cmap(v)[:3] for v in value]) * 255).astype(np.uint8)
        image[row - 4 : row + 5, 100:701] = colours[None, :, :]
    from PIL import Image

    Image.fromarray(image).save(path)


def test_digitiser_recovers_known_peaks(tmp_path):
    truth = {0: [102.0, 978.0], 45: [255.0, 1240.0], 90: [312.0, 2184.0]}
    synthetic_map(tmp_path / 'map.png', truth)
    peaks = digitize_fft_map(tmp_path / 'map.png', frequency_axis='x', frequency_pixels=(100, 700),
                             frequency_values=(10.0, 10000.0), angle_pixels=(280, 100), angle_values=(0.0, 90.0),
                             data_pixel_range=(100, 700), colorbar=dict(x=800, y0=300, y1=20, vmax=1.0),
                             angles=[0, 45, 90], prominence=0.2)  # fmt: skip
    for angle, frequencies in truth.items():
        found = peaks.loc[peaks.angle_deg == angle, 'freq_T'].to_numpy()
        for f in frequencies:
            assert np.min(np.abs(found - f) / f) < 0.015
    lk = pd.DataFrame(
        dict(direction='[x]', exp_angle_deg=45, label=['beta'], freq_T=[256.0], observed='yes')
    )
    assert validate_digitization(peaks, lk).ok.all()


def test_lk_rows_select_direction_and_day():
    lk = pd.DataFrame(
        dict(
            direction=['[001]', '[001]', '[110]'],
            day=[3, 5, np.nan],
            label=['a', 'a', 'b'],
            freq_T=[1.0, 2.0, 3.0],
            observed='yes',
        )
    )
    assert lk_rows_for_direction(lk, dict(label='[001]', day=3)).freq_T.tolist() == [1.0]
    assert len(lk_rows_for_direction(lk, dict(label='[001]'))) == 2


def test_regular_runs_are_flagged():
    peaks = pd.DataFrame(
        dict(angle_deg=5, freq_T=[204.0, 215.0, 226.0, 237.0, 249.0, 900.0], rel_mag=0.06)
    )
    runs = regular_peak_runs(peaks)
    assert len(runs) == 1 and runs.n.iloc[0] == 5


def test_slide_deck_extraction(tmp_path):
    deck = tmp_path / 'deck.pptx'
    with zipfile.ZipFile(deck, 'w') as archive:
        archive.writestr('ppt/slides/slide1.xml', '<p><a:t>gamma</a:t><a:t>22.7 T</a:t></p>')
        archive.writestr(
            'ppt/slides/_rels/slide1.xml.rels', '<Relationship Target="../media/image1.png"/>'
        )
        archive.writestr('ppt/media/image1.png', b'png')
    slides = extract_slide_deck(deck, tmp_path / 'out').read_text()
    assert 'gamma | 22.7 T' in slides and 'image1.png' in slides
    assert (tmp_path / 'out' / 'media' / 'image1.png').exists()
