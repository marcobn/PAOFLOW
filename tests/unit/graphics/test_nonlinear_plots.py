"""GPAO figures of the non-linear and coupled optical responses, drawn from synthetic output files."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PAOFLOW.GPAO import GPAO  # noqa: E402
from PAOFLOW.graphics.nonlinear_plots import component_files  # noqa: E402
from PAOFLOW.response.coupled_crystal import KERNEL_TAGS  # noqa: E402
from PAOFLOW.response.nonlinear_optics import SHG_TERMS  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    yield
    plt.close('all')


def _write(path, name, x, y):
    np.savetxt(path / name, np.column_stack([x, y]))


def test_nonlinear_spectra_and_units(tmp_path):
    e = np.linspace(0.0, 3.0, 31)
    for comp in ('xyz', 'yzx'):
        _write(tmp_path, 'shga_%s.dat' % comp, e, np.abs(np.sin(e)))
        _write(tmp_path, 'shift_%s.dat' % comp, e, np.cos(e))
    _write(tmp_path, 'shga_xyz_0.dat', e, e)  # a spin channel: not mixed with the totals
    assert list(component_files(str(tmp_path), 'shga')) == ['xyz', 'yzx']
    assert list(component_files(str(tmp_path), 'shga', spin=0)) == ['xyz']
    g = GPAO()
    g.plot_nonlinear(str(tmp_path), 'shg', part='abs', filename=str(tmp_path / 'shg.png'))
    ax = plt.gcf().axes[0]
    assert len(ax.get_lines()) == 3 and 'pm/V' in ax.get_ylabel()  # two components + zero line
    g.plot_nonlinear(str(tmp_path), 'shift', components=['xyz'], two_d=True)
    assert 'nm' in plt.gcf().axes[0].get_ylabel()
    assert (tmp_path / 'shg.png').exists()
    with pytest.raises(FileNotFoundError):
        g.plot_nonlinear(str(tmp_path), 'injectc')


def test_shg_terms_figure(tmp_path):
    e = np.linspace(0.0, 2.0, 21)
    _write(tmp_path, 'shgr_xyz.dat', e, np.ones_like(e))
    _write(tmp_path, 'shgi_xyz.dat', e, np.zeros_like(e))
    for name in SHG_TERMS:
        _write(tmp_path, 'shgtr_%s_xyz.dat' % name, e, np.ones_like(e) / len(SHG_TERMS))
        _write(tmp_path, 'shgti_%s_xyz.dat' % name, e, np.zeros_like(e))
    g = GPAO()
    g.plot_shg_terms(str(tmp_path), 'xyz')
    re_axis = plt.gcf().axes[0]
    assert len(re_axis.get_lines()) == 1 + 4 + 1  # zero line, four groups, total
    g.plot_shg_terms(str(tmp_path), 'xyz', grouped=False)
    assert len(plt.gcf().axes[0].get_lines()) == 1 + len(SHG_TERMS) + 1
    with pytest.raises(FileNotFoundError, match='shg_terms=True'):
        g.plot_shg_terms(str(tmp_path), 'yyy')


def test_coupled_levels_and_molecular_beta(tmp_path):
    e = np.linspace(0.0, 5.0, 26)
    for level, tag in KERNEL_TAGS.items():
        _write(tmp_path, 'cepsi_%s_xx.dat' % tag, e, np.exp(-((e - 3.0) ** 2)))
    _write(tmp_path, 'cshga_ipa_xyz.dat', [0.1, 1.0], [180.0, 250.0])
    (tmp_path / 'coupled_response.dat').write_text('# alpha_lrc 0.200000\n# alpha_boot 0.118768\n')
    g = GPAO()
    g.plot_coupled_response(str(tmp_path), 'epsilon', 'imag')
    labels = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
    assert (
        len(labels) == 5 and any('0.119' in t for t in labels) and any('0.200' in t for t in labels)
    )
    g.plot_coupled_response(str(tmp_path), 'shg', 'abs')
    assert plt.gcf().axes[0].get_lines()[1].get_marker() == 'o'  # two energies: markers
    with pytest.raises(FileNotFoundError):
        g.plot_coupled_response(str(tmp_path), 'epsilon', 'real')
    table = np.column_stack([e] + [np.full_like(e, v) for v in range(1, 9)])
    np.savetxt(tmp_path / 'beta_coupled.dat', table)
    g.plot_molecular_beta(str(tmp_path))
    assert len(plt.gcf().axes) == 2
