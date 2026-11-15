"""A calibrated band-map image becomes an intensity map in the model layout."""

import numpy as np
import pytest

from PAOFLOW.agentic.arpes import curves
from PAOFLOW.agentic.arpes.images import band_map_from_image
from PAOFLOW.agentic.thermo_optics.digitize import Axis


def test_band_map_from_image_recovers_a_drawn_band():
    rows, cols = 300, 400
    # frame interior: columns 50..350 map k = -0.3..0.3; rows 20..280 map E = 0.02..-0.1 (top to bottom)
    k_axis = Axis([50.0, 350.0], [-0.3, 0.3])
    e_axis = Axis([20.0, 280.0], [0.02, -0.1])
    image = np.ones((rows, cols, 3))
    col = np.arange(50, 350)
    k = k_axis.to_value(col.astype(float))
    e_band = 1.5 * (k - 0.1)  # a linear band crossing E_F at k = 0.1
    row = e_axis.to_pixel(e_band)
    for c, r in zip(col, row):
        if 20 <= r < 280:
            lo, hi = int(max(20, r - 3)), int(min(280, r + 4))
            image[lo:hi, c, :] = 0.2  # a dark 7-pixel band
    out = band_map_from_image(image, k_axis, e_axis, rows=(20, 280), cols=(50, 350))
    assert out['intensity'].shape == (300, 260)
    assert out['k'][0] < out['k'][-1] and out['energy_ev'][0] < out['energy_ev'][-1]
    d = curves.mdc_dispersion(out['k'], out['energy_ev'], out['intensity'], energy_window=(-0.08, -0.01),
                              min_contrast=0.5)  # fmt: skip
    assert d['energy_ev'].size > 10
    assert np.allclose(d['k'], 0.1 + d['energy_ev'] / 1.5, atol=4e-3)
    full = curves.mdc_dispersion(out['k'], out['energy_ev'], out['intensity'], min_contrast=0.5)
    assert curves.fermi_momentum(full['energy_ev'], full['k']) == pytest.approx(0.1, abs=5e-3)
    red = band_map_from_image(image, k_axis, e_axis, rows=(20, 280), cols=(50, 350), channel='r',
                              dark_is_high=False)  # fmt: skip
    assert red['intensity'].max() == pytest.approx(1.0) and red['intensity'].min() == pytest.approx(
        0.2
    )
    custom = band_map_from_image(image[..., 0], k_axis, e_axis, channel=lambda a: a)
    assert custom['intensity'].shape == (cols, rows)
