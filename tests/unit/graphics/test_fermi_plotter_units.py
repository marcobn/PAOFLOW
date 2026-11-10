"""fermi-plotter on a pyskeaf input file draws what it draws on FermiSurf_<s>.bxsf.

``Fermi_surf_band_<n>.bxsf`` (write2bxsf4skeaf) holds one band in Rydberg with the reciprocal
vectors without 2π, as SKEAF wants them; ``FermiSurf_<s>.bxsf`` (write2bxsf) holds every band in
eV with the 2π.  The reader brings the first to the units of the second.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.gen import fermi_plotter as fp

pytestmark = pytest.mark.unit

RY = 13.605693122994


def _bxsf(path, bands, vectors):
    shape = next(iter(bands.values())).shape
    lines = ['', 'BEGIN_INFO', '  Fermi Energy:     0.000000000', 'END_INFO', '']
    lines += ['BEGIN_BLOCK_BANDGRID_3D', 'band_energies', 'BEGIN_BANDGRID_3D_BANDS']
    lines += ['%12d' % len(bands), '%12d%12d%12d' % shape, '    0.000000  0.000000  0.000000']
    lines += ['  '.join('%10.6f' % v for v in row) for row in vectors]
    for label, grid in bands.items():
        lines.append('  BAND: %5d' % label)
        flat = grid.ravel()
        lines += [' '.join('%15.9f' % v for v in flat[i : i + 6]) for i in range(0, flat.size, 6)]
    lines += ['END_BANDGRID_3D', 'END_BLOCK_BANDGRID_3d']
    path.write_text('\n'.join(lines) + '\n')


@pytest.fixture
def files(tmp_path):
    k = np.linspace(0, 1, 9)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    band = -1.0 - 0.6 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz))
    vectors = np.array(
        [
            [-0.608142, -0.608142, 0.608142],
            [0.608142, 0.608142, 0.608142],
            [-0.608142, 0.608142, -0.608142],
        ]
    )
    _bxsf(tmp_path / 'FermiSurf_0.bxsf', {2: band}, vectors)
    _bxsf(tmp_path / 'Fermi_surf_band_1.bxsf', {2: band / RY}, vectors / (2 * np.pi))
    return tmp_path


def test_reader_brings_pyskeaf_files_to_ev(files):
    whole = fp.read_fermi_bxsf(files / 'FermiSurf_0.bxsf')
    sheet = fp.read_fermi_bxsf(files / 'Fermi_surf_band_1.bxsf')
    assert not whole.skeaf and sheet.skeaf
    assert np.allclose(sheet.recip, whole.recip, atol=1e-5)
    assert np.allclose(sheet.bands[2], whole.bands[2], atol=1e-6)
    raw = fp.read_fermi_bxsf(files / 'Fermi_surf_band_1.bxsf', skeaf=False)
    assert np.allclose(raw.recip * 2 * np.pi, whole.recip, atol=1e-5)
    assert fp.is_skeaf_input('out/Fermi_surf_band_12.bxsf') and not fp.is_skeaf_input(
        'FermiSurf_0.bxsf'
    )


def test_command_line_converts_and_says_so(files, monkeypatch, capsys):
    drawn = {}

    def plot(data, labels, **kwargs):  # Mayavi is not needed to check what would be drawn
        drawn.update(data=data, labels=labels, shift=kwargs['fermi_shift'])

    monkeypatch.setattr(fp, 'plot_fermi_surface', plot)
    assert fp.main([str(files / 'Fermi_surf_band_1.bxsf'), '--fermi-shift', '-0.5']) == 0
    assert drawn['data'].skeaf and drawn['shift'] == -0.5 and drawn['labels'] == [2]
    assert 'converted to eV' in capsys.readouterr().out
    assert np.isclose(
        drawn['data'].bands[2].min(),
        fp.read_fermi_bxsf(files / 'FermiSurf_0.bxsf').bands[2].min(),
        atol=1e-6,
    )
    fp.main([str(files / 'Fermi_surf_band_1.bxsf'), '--units', 'fermisurf'])
    assert not drawn['data'].skeaf
