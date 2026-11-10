"""Results layer: discovery, loaders, figures, exports, report and the output manifest."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.results import ResultsFolder
from PAOFLOW.results.columns import header_parameters, read_blocks, read_columns
from PAOFLOW.results.discovery import discover, match_file
from PAOFLOW.results.export import matplotlib_figure, to_csv
from PAOFLOW.results.figures import figure_spec
from PAOFLOW.results.loaders import load, read_kpath
from PAOFLOW.results.model import MAX_ORDINAL, ordinal_roles, spread

EXAMPLES = Path(__file__).resolve().parents[3] / 'examples'
SI = EXAMPLES / 'qe_examples' / 'example01' / 'Reference'
PT = EXAMPLES / 'qe_examples' / 'example08' / 'Reference'


def _by_kind(folder):
    refs = discover(str(folder))
    return {ref.kind: [r for r in refs if r.kind == ref.kind] for ref in refs}, refs


def test_discovery_of_the_silicon_reference():
    kinds, refs = _by_kind(SI)
    assert set(kinds) == {'bands', 'dos', 'pdos', 'boltzmann'}
    assert kinds['dos'][0].labels == ['DOS', 'Σ PDOS']
    assert len(kinds['pdos'][0].files) == 18 and len(kinds['boltzmann']) == 4
    assert all(ref.id for ref in refs) and len({ref.id for ref in refs}) == len(refs)


@pytest.mark.parametrize(
    'name, folder, kind',
    [
        ('bands_0.dat', '.', 'bands'),
        ('Si.bands_1.dat', '.', 'bands'),
        ('bands_UV_0.dat', '.', 'bands'),
        ('Si1_3px_pdos_0.dat', '.', 'pdos'),
        ('pdos_sum_0.dat', '.', 'dos'),
        ('sigmagauss_0.dat', '.', 'boltzmann'),
        ('hall_trace_gauss_0.dat', '.', 'hall_coefficient'),
        ('epsi_xx.dat', '.', 'optics'),
        ('alpha_zz_1.dat', '.', 'optics'),
        ('alpha_300.0K.dat', 'optics/output', 'phonon_assisted_absorption'),
        ('emish_xx.dat', '.', 'emissivity_spectral'),
        ('emis_th30.0_xx.dat', '.', 'emissivity_directional'),
        ('shgr_xyz.dat', '.', 'nlo'),
        ('cepsi_alda_xx.dat', '.', 'coupled_dielectric'),
        ('phonon_band.dat', '.', 'phonon_bands'),
        ('phonon_dos.dat', '.', 'phonon_dos'),
        ('mobility.dat', 'transport/output', 'mobility'),
        ('alpha2F_tr.dat', '.', 'alpha2f'),
        ('conductance_bulk.dat', '.', 'conductance'),
        ('FermiSurf_0.bxsf', '.', 'fermi_surface'),
        ('Fermi_surf_band_3.bxsf', '.', 'fermi_sheet'),
        ('Spin_Berry_z_xy.bxsf', '.', 'berry_bxsf'),
        ('state.json', 'sscha_T300', 'sscha'),
        ('paoflow_outputs.json', '.', 'manifest'),
        ('arpes_iso_10K.npz', 'arpes/output', 'arpes_map'),
    ],
)
def test_file_kinds(name, folder, kind):
    found = match_file(name, folder)
    assert found is not None and found[0].name == kind


@pytest.mark.parametrize(
    'name', ['main.py', 'scf.in', 'kpath_points.txt', 'phonon_band.labels', 'x.png']
)
def test_skipped_files(name):
    assert match_file(name, '.') is None


def test_bands_loader_and_figure():
    kinds, _ = _by_kind(SI)
    dataset = load(kinds['bands'][0], str(SI))
    series = dataset.series[0]
    assert series.y.ndim == 2 and series.y.shape[0] == len(series.x)
    assert dataset.ticks[0][1] == 'Γ' and dataset.options['window']['value'][0] >= -10
    spec = figure_spec(dataset)
    assert len(spec['data']) == 1 and spec['data'][0]['line']['color'] == 'series-1'
    assert spec['layout']['xaxis']['ticktext'][0] == 'Γ' and 'yaxis2' not in spec['layout']
    assert any(shape['y0'] == 0 for shape in spec['layout']['shapes'] if shape.get('yref') == 'y')


def test_transport_options_and_ordinal_cap():
    kinds, _ = _by_kind(SI)
    ref = next(r for r in kinds['boltzmann'] if r.group['q'] == 'sigma')
    dataset = load(ref, str(SI), {'component': 'xx'})
    assert dataset.options['component']['value'] == 'xx'
    assert dataset.options['temperatures']['value'] == [300.0]
    assert ordinal_roles(5) == ['ord-1', 'ord-2', 'ord-3', 'ord-4', 'ord-5'] and ordinal_roles(
        1
    ) == ['ord-3']
    assert (
        spread(list(range(12))) == [0, 3, 6, 8, 11] and len(spread(list(range(12)))) == MAX_ORDINAL
    )


def test_pdos_folds_to_eight_series_plus_total():
    kinds, _ = _by_kind(SI)
    dataset = load(kinds['pdos'][0], str(SI))
    names = [s.name for s in dataset.series]
    assert len(names) == 9 and names[-1] == 'total' and names[-2].startswith('other')
    assert names[0].startswith('orbital ')


def test_dense_and_sparse_kpath(tmp_path):
    (tmp_path / 'kpath_points.txt').write_text('gG 10\nX 0\nK 5\nW 1\n\n0.1 0.2 0.3\n')
    ticks, by_distance = read_kpath(str(tmp_path))
    assert not by_distance and ticks == [(0, 'Γ'), (10, 'X|K'), (15, 'W')]
    (tmp_path / 'kpath_points.txt').write_text('0.0  G\n0.5  X\n')
    ticks, by_distance = read_kpath(str(tmp_path))
    assert by_distance and ticks == [(0.0, 'Γ'), (0.5, 'X')]


def test_columns_reader_variants(tmp_path):
    path = tmp_path / 't.dat'
    path.write_text('# omega(meV)  alpha^2F  (lambda=0.5767, w_log=724.11K)\n1.0D0 2.0\n2.0 3.0\n')
    data, header = read_columns(str(path))
    assert data.tolist() == [[1.0, 2.0], [2.0, 3.0]]
    assert header_parameters(header) == {'lambda': 0.5767, 'w_log': 724.11}
    path.write_text('Theta(deg),Phi(deg),Freq(kT)\n 0.0, 0.0, 3.1E-01\n')
    assert read_columns(str(path))[0].tolist() == [[0.0, 0.0, 0.31]]
    path.write_text('0 -1 0 0 1\n1 -1 0 0 1\n\n0 1 0 0 -1\n')
    blocks, _ = read_blocks(str(path))
    assert [b.shape for b in blocks] == [(2, 5), (1, 5)]


def test_multi_column_kinds_keep_one_unit(tmp_path):
    out = tmp_path / 'transport' / 'output'
    out.mkdir(parents=True)
    rows = np.array([[200.0, 0.7, 1e2, 1e2, 800.0, 140.0, 2100.0, 260.0, 3e-12, 7e-12, -2600.0]])
    np.savetxt(out / 'mobility.dat', rows, header='T (K), mu_F (eV), ...')
    folder = ResultsFolder(str(tmp_path))
    dataset = folder.load(folder.find('mobility'))
    assert (
        dataset.y_label.startswith('Mobility')
        and [s.name for s in dataset.series][0] == 'electrons SERTA'
    )
    other = folder.load(folder.find('mobility'), {'quantity': 'Seebeck coefficient'})
    assert other.y_label.startswith('Seebeck') and len(other.series) == 1


def test_temperature_series_from_files(tmp_path):
    out = tmp_path / 'optics' / 'output'
    out.mkdir(parents=True)
    for t in (300, 500, 700, 900, 1100, 1300, 1500):
        np.savetxt(
            out / ('alpha_%.1fK.dat' % t),
            np.array([[0.1, t, 2 * t], [0.2, t, 2 * t]]),
            header='eta (eV) = 0.01 0.05',
        )
    folder = ResultsFolder(str(tmp_path))
    dataset = folder.load(folder.find('phonon_assisted_absorption'))
    assert len(dataset.series) == MAX_ORDINAL and dataset.series[0].name == '300 K'
    assert dataset.options['column']['choices'] == ['η = 0.01', 'η = 0.05']
    assert [s.role for s in dataset.series] == ordinal_roles(MAX_ORDINAL)


def test_exports(tmp_path):
    kinds, _ = _by_kind(SI)
    dataset = load(kinds['dos'][0], str(SI))
    text = to_csv(dataset)
    assert text.splitlines()[1].startswith('"Energy (eV)"') and len(text.splitlines()) > 100
    pytest.importorskip('matplotlib')
    assert matplotlib_figure(dataset, 'pdf')[:4] == b'%PDF'
    assert matplotlib_figure(load(kinds['bands'][0], str(SI)), 'png')[:4] == b'\x89PNG'


def test_fermi_surface_mesh():
    pytest.importorskip('skimage')
    kinds, _ = _by_kind(PT)
    dataset = load(kinds['fermi_surface'][0], str(PT), {'bands': [59]})
    assert [s['band'] for s in dataset.mesh['sheets']] == [59]
    assert dataset.options['bands']['choices'] == [57, 58, 59, 60] and dataset.mesh['edges']
    spec = figure_spec(dataset)
    assert spec['data'][0]['type'] == 'mesh3d' and spec['data'][-1]['type'] == 'scatter3d'
    slice_ = load(kinds['berry_bxsf'][0], str(PT), {'axis': 'k1'})
    assert slice_.surface['diverging'] and figure_spec(slice_)['data'][0]['type'] == 'heatmap'


def test_fermi_surface_band_selection_is_kept():
    """The ticked bands are the drawn ones, an empty selection included (it drew every band, so
    the unticked boxes came back)."""
    pytest.importorskip('skimage')
    kinds, _ = _by_kind(PT)
    ref = kinds['fermi_surface'][0]
    every = load(ref, str(PT), {})
    assert every.options['bands']['value'] == [57, 58, 59, 60]
    some = load(ref, str(PT), {'bands': [58, 60]})
    assert some.options['bands']['value'] == [58, 60] and len(some.mesh['sheets']) == 2
    none = load(ref, str(PT), {'bands': []})
    assert none.mesh['sheets'] == [] and none.options['bands']['value'] == []
    assert none.options['bands']['choices'] == [57, 58, 59, 60]
    assert 'No band selected.' in none.notes
    gap = load(ref, str(PT), {'shift': 40.0})
    assert gap.options['bands']['choices'] == [] and any('No band crosses' in n for n in gap.notes)


def test_report_is_self_contained(tmp_path):
    pytest.importorskip('plotly')
    from PAOFLOW.results.report import write_report

    path = write_report(str(SI), str(tmp_path / 'r.html'))
    text = Path(path).read_text()
    assert 'Band structure' in text and 'const FIGURES' in text
    assert '<script src=' not in text and '<link' not in text


def test_output_manifest(tmp_path):
    from PAOFLOW.writers.output_manifest import MANIFEST, write_output_manifest

    out = tmp_path / 'output'
    out.mkdir()
    shutil.copy(SI / 'bands_0.dat', out / 'bands_0.dat')
    attributes = {'opath': str(out), 'Efermi': np.float64(5.26), 'nspin': 1, 'savedir': 'si.save'}
    write_output_manifest(attributes, start_time=0.0)
    write_output_manifest(attributes, start_time=0.0)
    manifest = json.loads((out / MANIFEST).read_text())
    run = manifest['runs'][-1]
    assert len(manifest['runs']) == 2 and run['fermi_energy_dft_ev'] == 5.26
    assert run['files'] == [
        {
            'file': 'bands_0.dat',
            'bytes': os.path.getsize(out / 'bands_0.dat'),
            'kind': 'bands',
            'category': 'Electronic structure',
            'labels': {'spin': '0'},
        }
    ]
    folder = ResultsFolder(str(out))
    assert folder.load(folder.find('manifest')).table[0]['name'] == 'paoflow'


def test_stacked_pdos():
    from PAOFLOW.results.palette import resolve_spec

    kinds, _ = _by_kind(SI)
    dataset = load(kinds['pdos'][0], str(SI), {'display': 'stacked'})
    spec = resolve_spec(figure_spec(dataset), 'light')
    stacked = [t for t in spec['data'] if t.get('stackgroup') == 'pdos']
    assert len(stacked) == len(spec['data']) - 1 and stacked[0]['fillcolor'].startswith('rgba(')
    assert spec['data'][-1]['name'] == 'total' and 'stackgroup' not in spec['data'][-1]
    pytest.importorskip('matplotlib')
    assert matplotlib_figure(dataset, 'png')[:4] == b'\x89PNG'


def test_overlay_figure_panels_per_unit():
    """Same unit: one panel; another unit: a panel under it on the shared x axis (no second y axis);
    colour per property (its slot), line style per series."""
    from PAOFLOW.results.figures import axis_name, axis_unit, overlay_figure

    assert axis_unit('Absorption coefficient (1/m)') == '1/m' and axis_unit('Re ε') == ''
    assert axis_name('Absorption coefficient (1/m)') == 'Absorption coefficient'
    kinds, _ = _by_kind(SI)
    dos = load(kinds['dos'][0], str(SI))
    pdos = load(kinds['pdos'][0], str(SI))
    figure, names, shown = overlay_figure([dos, pdos])
    assert [k for k in figure['layout'] if k.startswith('yaxis')] == ['yaxis']
    assert {t['yaxis'] for t in figure['data']} == {'y'}
    # DOS and PDOS share no series name: each property still appears, with its first series.
    assert [t['line']['color'] for t in figure['data']] == ['series-1', 'series-2']
    assert shown == [dos.series[0].name, pdos.series[0].name]
    assert figure['data'][1]['name'].endswith(pdos.series[0].name)

    boltzmann = {load(r, str(SI)).y_label: load(r, str(SI)) for r in kinds['boltzmann']}
    seebeck = next(d for label, d in boltzmann.items() if label.startswith('Seebeck'))
    sigma = next(d for label, d in boltzmann.items() if label.startswith('Electrical'))
    figure, names, shown = overlay_figure([seebeck, sigma], slots=[1, 4])
    layout = figure['layout']
    assert layout['yaxis']['domain'][0] > layout['yaxis2']['domain'][1]  # stacked, first on top
    assert layout['xaxis']['anchor'] == 'y2' and 'yaxis3' not in layout
    assert {t['yaxis'] for t in figure['data']} == {'y', 'y2'}
    assert figure['data'][-1]['line']['color'] == 'series-4'  # the slot, not the position
    if len(names) > 1:
        figure, _, shown = overlay_figure([seebeck, sigma], series=names[:2])
        assert shown == names[:2] and {t['line']['dash'] for t in figure['data']} == {
            'solid',
            'dash',
        }


def test_axis_limits_of_the_exports(monkeypatch):
    """The Axes card's limits reach the matplotlib figure and the Python snippet."""
    pytest.importorskip('matplotlib')
    import matplotlib.axes

    from PAOFLOW.results.export import clean_limits, matplotlib_figure, python_snippet

    assert clean_limits({'x': [0, 6], 'y': [None, 2.5]}) == {'x': [0.0, 6.0], 'y': [None, 2.5]}
    assert clean_limits({'x': [None, None], 'y': None}) == {}
    with pytest.raises(ValueError, match='below'):
        clean_limits({'x': [3, 1]})
    with pytest.raises(ValueError):
        clean_limits({'y': ['a', 1]})
    calls = []
    monkeypatch.setattr(
        matplotlib.axes.Axes, 'set_xlim', lambda self, *a, **k: calls.append(('x', a))
    )
    monkeypatch.setattr(
        matplotlib.axes.Axes, 'set_ylim', lambda self, *a, **k: calls.append(('y', a))
    )
    kinds, _ = _by_kind(SI)
    dataset = load(kinds['dos'][0], str(SI))
    assert (
        matplotlib_figure(dataset, 'png', limits={'x': [-5, 5], 'y': [0, None]})[:4] == b'\x89PNG'
    )
    assert ('x', (-5.0, 5.0)) in calls and ('y', (0.0, None)) in calls
    text = python_snippet(dataset, str(SI), {}, {'x': [-5, 5]})
    assert "limits={'x': [-5.0, 5.0]}" in text


def _bxsf(path, bands, vectors, fermi=0.0):
    """A BXSF file as the PAOFLOW writers make it (general grid, BAND blocks)."""
    shape = next(iter(bands.values())).shape
    lines = ['', 'BEGIN_INFO', '  Fermi Energy: %15.9f' % fermi, 'END_INFO', '']
    lines += ['BEGIN_BLOCK_BANDGRID_3D', 'band_energies', 'BEGIN_BANDGRID_3D_BANDS']
    lines += ['%12d' % len(bands), '%12d%12d%12d' % shape, '    0.000000  0.000000  0.000000']
    lines += ['  '.join('%10.6f' % v for v in row) for row in vectors]
    for label, grid in bands.items():
        lines.append('  BAND: %5d' % label)
        flat = grid.ravel()
        lines += [' '.join('%15.9f' % v for v in flat[i : i + 6]) for i in range(0, flat.size, 6)]
    lines += ['END_BANDGRID_3D', 'END_BLOCK_BANDGRID_3d']
    path.write_text('\n'.join(lines) + '\n')


def test_pyskeaf_sheets_match_the_fermi_surface(tmp_path):
    """Fermi_surf_band_<n>.bxsf (SKEAF input: Rydberg, vectors without 2π, file n = position)
    and FermiSurf_0.bxsf (eV, 2π) of the same bands draw the same sheets at the same level."""
    pytest.importorskip('skimage')
    n = 13
    k = np.linspace(0, 1, n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    cosines = np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz)
    bands = {7: -1.0 - 0.6 * cosines, 8: 1.5 + 0.8 * cosines}  # eV from E_F
    vectors = np.array([[-0.6, -0.6, 0.6], [0.6, 0.6, 0.6], [-0.6, 0.6, -0.6]])
    _bxsf(tmp_path / 'FermiSurf_0.bxsf', bands, vectors)
    for number, label in enumerate(bands, start=1):
        _bxsf(
            tmp_path / ('Fermi_surf_band_%d.bxsf' % number),
            {label: bands[label] / 13.605693122994},
            vectors / (2 * np.pi),
        )
    kinds, _ = _by_kind(tmp_path)
    for shift in (-1.2, 1.0):
        for zone in ('cell', 'ws'):
            options = {'shift': shift, 'zone': zone}
            whole = load(kinds['fermi_surface'][0], str(tmp_path), options)
            sheets = load(kinds['fermi_sheet'][0], str(tmp_path), options)
            assert whole.options['bands']['choices'] == sheets.options['bands']['choices']
            pairs = list(zip(whole.mesh['sheets'], sheets.mesh['sheets']))
            assert pairs and [a['band'] for a, _ in pairs] == [b['band'] for _, b in pairs]
            for a, b in pairs:
                if zone == 'cell':  # unfolded: the same points (the 6-decimal vectors aside)
                    assert np.allclose(a['x'], b['x'], atol=1e-5)
                    assert np.allclose(a['speed'], b['speed'], rtol=1e-4)
                    continue
                # Folded: the 6-decimal vectors move a few triangles across the zone faces.
                assert abs(len(a['x']) - len(b['x'])) <= 0.01 * len(a['x'])
                for key in ('x', 'y', 'z'):
                    assert np.isclose(np.abs(a[key]).max(), np.abs(b[key]).max(), atol=1e-4)
                assert np.isclose(np.max(a['speed']), np.max(b['speed']), rtol=1e-3)
                assert np.isclose(np.min(a['speed']), np.min(b['speed']), rtol=1e-3)
    assert any('Rydberg' in note for note in sheets.notes)
    assert sheets.options['bands']['choices'] == [8]  # the BAND line of file 2, not 2
