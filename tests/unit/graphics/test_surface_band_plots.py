"""GPAO figure of the surface spectral function, drawn from synthetic compute_surface_bands files."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PAOFLOW.GPAO import GPAO  # noqa: E402
from PAOFLOW.graphics.surface_plots import read_surface_bands, surface_label  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    yield
    plt.close('all')


def _write_run(path, postfix='_surf'):
    """5 energies x 4 k-points, labels at the two ends and in the middle (as the writer does)."""
    energy = np.linspace(-0.5, 0.5, 5)
    spectral = np.outer(np.exp(-(energy**2) / 0.05), [1.0, 2.0, 30.0, 0.5])
    np.savetxt(path / f'surfband{postfix}.dat', spectral)
    np.savetxt(path / f'surfband_egrid{postfix}.dat', energy)
    with open(path / f'surfband_kpath{postfix}.dat', 'w') as handle:
        handle.write('# index   kdist            label\n')
        for index, (kdist, label) in enumerate(zip([0.0, 0.5, 1.0, 1.4], ['K', '', 'gG', 'M'])):
            handle.write(f'{index:6d} {kdist:15.9f}  {label}\n')
    return spectral, energy


def test_reader_returns_the_map_and_its_axes(tmp_path):
    spectral, energy = _write_run(tmp_path)
    spectrum = read_surface_bands(str(tmp_path), '_surf')
    np.testing.assert_allclose(spectrum.spectral, spectral)
    np.testing.assert_allclose(spectrum.energy, energy)
    np.testing.assert_allclose(spectrum.kdist, [0.0, 0.5, 1.0, 1.4])
    assert spectrum.ticks == [0, 2, 3] and spectrum.labels == ['K', 'gG', 'M']


def test_reader_rejects_a_map_that_does_not_match_its_axes(tmp_path):
    _write_run(tmp_path)
    np.savetxt(tmp_path / 'surfband_egrid_surf.dat', np.linspace(-0.5, 0.5, 6))
    with pytest.raises(ValueError, match='shape'):
        read_surface_bands(str(tmp_path), '_surf')


def test_surface_labels_get_an_overbar_and_gamma():
    assert surface_label('gG') == r'$\overline{\Gamma}$'
    assert surface_label('K') == r'$\overline{\mathrm{K}}$'
    assert surface_label('K', {'K': 'K1'}) == 'K1'


@pytest.mark.parametrize('scale', ['log', 'linear'])
def test_plot_saves_a_png(tmp_path, scale):
    _write_run(tmp_path)
    png = tmp_path / 'surfband_surf.png'
    ax = GPAO().plot_surface_bands(str(tmp_path), '_surf', filename=str(png), scale=scale)
    assert png.exists() and png.stat().st_size > 0
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        r'$\overline{\mathrm{K}}$',
        r'$\overline{\Gamma}$',
        r'$\overline{\mathrm{M}}$',
    ]


def test_plot_into_an_existing_axis_neither_saves_nor_closes(tmp_path):
    _write_run(tmp_path)
    fig, (left, right) = plt.subplots(1, 2)
    ax = GPAO().plot_surface_bands(str(tmp_path), '_surf', ax=right, y_lim=(-0.2, 0.2))
    assert ax is right and len(right.collections) == 1
    assert right.get_ylim() == (-0.2, 0.2)
    assert plt.fignum_exists(fig.number)
    assert list(tmp_path.glob('*.png')) == []


def test_unknown_scale_raises(tmp_path):
    _write_run(tmp_path)
    with pytest.raises(ValueError, match='scale'):
        GPAO().plot_surface_bands(str(tmp_path), '_surf', scale='sqrt')
