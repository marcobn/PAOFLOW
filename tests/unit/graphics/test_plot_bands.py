"""Tests for the band-structure plot of GPAO."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import GPAO


def test_plot_bands_returns_open_axes(tmp_path):
    """With show=False the figure stays open and more data can be drawn on it."""
    bands = tmp_path / 'bands_0.dat'
    k = np.arange(11)
    np.savetxt(bands, np.column_stack([k, np.sin(k / 3), np.cos(k / 3) + 2]))
    ax = GPAO.GPAO().plot_bands(str(bands), sym_points=([0, 10], ['G', 'X']), show=False)
    assert plt.fignum_exists(ax.figure.number)
    assert len(ax.get_lines()) == 2
    ax.scatter([5], [1.0])
    plt.close(ax.figure)
