"""Results layer: discovery, loaders, figures, exports, report and the output manifest."""

from __future__ import annotations

import json
import os
import platform
import re
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

DATA = Path(__file__).resolve().parents[2] / 'data'
SI = DATA / 'silicon' / 'Reference'
PT = DATA / 'platinum' / 'Reference'


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
        ('current.dat', '.', 'current'),
        ('nernst_gauss_0.dat', '.', 'nernst'),
        ('nernst_0.dat', '.', 'nernst'),
        ('current_self_consistent_alsi.dat', '.', 'finite_bias_current'),
        ('current_linear.dat', '.', 'finite_bias_current'),
        ('potential_self_consistent_alsi.dat', '.', 'finite_bias_potential'),
        ('potential_planar_average_self_consistent_3d.dat', '.', 'finite_bias_planar_average'),
        ('charge_linear_wire.dat', '.', 'finite_bias_charge'),
        ('transmission_self_consistent_alsi.dat', '.', 'finite_bias_transmission'),
        ('FermiSurf_0.bxsf', '.', 'fermi_surface'),
        ('Fermi_surf_band_3.bxsf', '.', 'fermi_sheet'),
        ('Spin_Berry_z_xy.bxsf', '.', 'berry_bxsf'),
        ('kubo_charge_current_xy.dat', '.', 'kubo_response'),
        ('kubo_spin_current_z_xy.dat', '.', 'kubo_response'),
        ('kubo_optical_charge_current_xy.dat', '.', 'kubo_optical'),
        ('kubo_optical_spin_current_z_xy.dat', '.', 'kubo_optical'),
        ('AHC_xy.dat', '.', 'linear_response'),
        ('state.json', 'sscha_T300', 'sscha'),
        ('paoflow_outputs.json', '.', 'manifest'),
        ('arpes_iso_10K.npz', 'arpes/output', 'arpes_map'),
    ],
)
def test_file_kinds(name, folder, kind):
    found = match_file(name, folder)
    assert found is not None and found[0].name == kind


@pytest.mark.parametrize(
    'name', ['scf.out', 'NOTES.md', 'kpath_points.txt', 'phonon_band.labels', 'x.png']
)
def test_skipped_files(name):
    assert match_file(name, '.') is None


@pytest.mark.parametrize(
    ('name', 'kind'),
    [('main.py', 'script'), ('scf.in', 'dft_input'), ('INCAR', 'dft_input'), ('case.json', 'case')],
)
def test_inputs_pass_the_skip_list(name, kind):
    assert match_file(name, '.')[0].name == kind


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
    (entry,) = run['files']
    assert (
        entry.items()
        >= {
            'file': 'bands_0.dat',
            'bytes': os.path.getsize(out / 'bands_0.dat'),
            'kind': 'bands',
            'category': 'Electronic structure',
            'labels': {'spin': '0'},
        }.items()
    )
    assert len(entry['sha256']) == 64 and manifest['manifest'] == 2
    # an absolute save path names the user: only its last name is recorded
    write_output_manifest({**attributes, 'savedir': '/home/someone/si.save'}, start_time=0.0)
    assert json.loads((out / MANIFEST).read_text())['runs'][-1]['savedir'] == 'si.save'
    folder = ResultsFolder(str(out))
    assert folder.load(folder.find('manifest')).table[0]['name'] == 'paoflow'


def test_every_term_is_defined_and_documented():
    """Each catalogue entry has a definition, and the docs page shows the current table."""
    from PAOFLOW.results.catalogue import KINDS
    from PAOFLOW.results.vocabulary import TERMS, vocabulary_markdown

    assert {kind.name for kind in KINDS} == set(TERMS)
    assert all(definition.endswith('.') for definition, _ in TERMS.values())
    page = Path(__file__).parents[3] / 'docs' / 'tutorials' / 'howtos' / 'run-records.md'
    assert vocabulary_markdown() in page.read_text(encoding='utf-8')


def _validate(path):
    """Check a run record against the JSON Schema that ships with the package."""
    jsonschema = pytest.importorskip('jsonschema')
    from PAOFLOW.writers import output_manifest

    schema = Path(output_manifest.__file__).with_name('run_record.schema.json')
    jsonschema.validate(json.loads(Path(path).read_text()), json.loads(schema.read_text()))


def _si_bands(work, outputdir):
    from PAOFLOW.PAOFLOW import PAOFLOW

    paoflow = PAOFLOW(workpath=str(work), savedir='silicon.save', outputdir=outputdir)
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=2, nk=20)
    paoflow.pao_eigh()
    paoflow.dos(emin=-12.0, emax=2.0, ne=200)
    paoflow.finish_execution()


def test_the_recorder_does_not_change_the_outputs(tmp_path, monkeypatch):
    """The same run with and without the recorder writes byte-identical files."""
    from PAOFLOW.writers import output_manifest

    shutil.copytree(SI.parent / 'silicon.save', tmp_path / 'silicon.save')
    _si_bands(tmp_path, 'recorded')
    monkeypatch.setattr(output_manifest, '_record_call', lambda *args: None)
    _si_bands(tmp_path, 'plain')
    recorded = {p.name: p.read_bytes() for p in (tmp_path / 'recorded').iterdir()}
    plain = {p.name: p.read_bytes() for p in (tmp_path / 'plain').iterdir()}
    assert recorded.pop(output_manifest.MANIFEST) and len(plain) > 5
    assert recorded == plain


def test_plain_function_steps_share_one_entry(tmp_path):
    """The electron-phonon writers and pyskeaf record through one explicit call each."""
    from PAOFLOW.writers.output_manifest import MANIFEST, record_function_call

    shutil.copy(SI / 'dosdk_0.dat', tmp_path / 'dosdk_0.dat')
    os.utime(tmp_path / 'dosdk_0.dat')
    record_function_call(str(tmp_path), 'elphon.write_mobility', {'label': 'n-type'})
    phonons = tmp_path / 'phonon'
    phonons.mkdir()
    (phonons / 'si.dyn1').write_text('dynamical matrix')
    (phonons / 'phonopy.yaml').write_text('phonopy:\n  version: "2.38.1"\n')
    (phonons / 'ph.out').write_text('a log, not hashed')
    record_function_call(
        str(tmp_path), 'elphon.write_eliashberg', {'mu_star': 0.1}, inputs=[str(phonons), None]
    )
    record_function_call(str(tmp_path / 'missing'), 'never raises')
    _validate(tmp_path / MANIFEST)
    (run,) = json.loads((tmp_path / MANIFEST).read_text())['runs']
    assert [call['method'] for call in run['calls']] == [
        'elphon.write_mobility',
        'elphon.write_eliashberg',
    ]
    assert run['status'] == 'finished' and run['files'][0]['kind'] == 'dos'
    assert run['input_versions'] == {'phonopy': '2.38.1'}
    assert sorted(os.path.basename(stamp['file']) for stamp in run['input_files']) == [
        'phonopy.yaml',
        'si.dyn1',
    ]
    assert len(run['files'][0]['sha256']) == 64


def test_run_record_follows_the_run(tmp_path, monkeypatch):
    """The record is rewritten after each call of the script: unfinished until
    ``finish_execution``, one entry per run, with the inputs, the calls and no absolute path."""
    from PAOFLOW.PAOFLOW import PAOFLOW
    from PAOFLOW.writers.output_manifest import MANIFEST

    work = tmp_path / 'case'
    shutil.copytree(SI.parent / 'silicon.save', work / 'silicon.save')
    monkeypatch.chdir(work)
    paoflow = PAOFLOW(savedir='silicon.save', outputdir=str(work / 'output'), verbose=False)
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=2, nk=20)
    path = work / 'output' / MANIFEST
    _validate(path)
    (run,) = json.loads(path.read_text())['runs']
    assert run['status'] == 'unfinished' and 'finished' not in run
    assert [call['method'] for call in run['calls']] == [
        'PAOFLOW',
        'PAOFLOW.read_atomic_proj_QE',
        'PAOFLOW.projectability',
        'PAOFLOW.pao_hamiltonian',
        'PAOFLOW.bands',
    ]
    assert run['calls'][-1]['arguments']['nk'] == 20 and run['replayable']
    assert run['calls'][0]['arguments']['outputdir'] == 'output'
    assert 'sha256' not in run['files'][0]
    inputs = run['inputs']
    assert inputs['engine'] == 'Quantum ESPRESSO' and inputs['functional'] == 'PBE'
    assert re.fullmatch(r'\d+\.\d+(\.\d+)?', inputs['version'])
    assert np.shape(inputs['cell_bohr']) == (3, 3) and len(inputs['positions_bohr']) == 2
    assert {'silicon.save/data-file-schema.xml', 'silicon.save/atomic_proj.xml'} <= {
        entry['file'] for entry in inputs['files']
    }
    paoflow.finish_execution()
    _validate(path)
    text = path.read_text()
    (done,) = json.loads(text)['runs']
    assert done['run_id'] == run['run_id'] and done['status'] == 'finished'
    assert done['started_utc'].endswith('Z') and done['finished_utc'].endswith('Z')
    assert all(len(entry['sha256']) == 64 for entry in done['files'])
    assert any(axis['unit'] for entry in done['files'] for axis in entry.get('axes', []))
    assert done['infrastructure']['mpi_ranks'] == 1 and done['environment']['numpy']
    assert str(tmp_path) not in text and platform.node() not in text


def test_run_record_arguments_and_failures(tmp_path):
    """Arrays, callables and objects are described, not stored; a call that raises is recorded as
    failed and the exception still reaches the script."""
    from PAOFLOW.writers.output_manifest import MANIFEST, encode_argument, recorded

    assert encode_argument([1, 'a', {'b': 2.0}]) == ([1, 'a', {'b': 2.0}], True)
    encoded, rebuildable = encode_argument(np.zeros((2, 3)))
    assert encoded['array'] == [2, 3] and len(encoded['sha256']) == 64 and not rebuildable
    assert encode_argument(len) == ({'callable': 'builtins.len'}, False)
    assert encode_argument(Path('x'))[0]['object'].endswith('Path')
    assert encode_argument('/nowhere/secret/si.save') == ({'path_outside_folder': 'si.save'}, False)

    class Controller:
        rank = 0
        data_attributes = {'opath': str(tmp_path)}
        data_arrays = {}

    @recorded
    class Driver:
        def __init__(self):
            self.data_controller = Controller()

        def step(self, weights, scale=2.0):
            """Docstring kept."""
            return self.inner(scale)

        def inner(self, scale):
            return scale

        def broken(self):
            raise ValueError('stop')

    driver = Driver()
    assert driver.step(np.ones(3)) == 2.0 and Driver.step.__doc__ == 'Docstring kept.'
    with pytest.raises(ValueError):
        driver.broken()
    (run,) = json.loads((tmp_path / MANIFEST).read_text())['runs']
    assert [(call['method'], call['status']) for call in run['calls']] == [
        ('Driver', 'ok'),
        ('Driver.step', 'ok'),
        ('Driver.broken', 'failed'),
    ]
    assert run['status'] == 'unfinished' and not run['replayable']


def test_output_manifest_records_the_source_commit(monkeypatch, tmp_path):
    """From a git checkout the manifest carries the commit and whether tracked files were modified;
    an installed package (no checkout around it) records the version only."""
    import subprocess

    from PAOFLOW.writers import output_manifest

    package = Path(output_manifest.__file__).resolve().parents[1]
    revision = output_manifest.source_revision()
    try:
        top = subprocess.run(['git', '-C', str(package), 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        in_checkout = (Path(top) / 'src' / 'PAOFLOW').resolve() == package
    except (OSError, subprocess.CalledProcessError):
        in_checkout = False
    if in_checkout:
        assert len(revision['commit']) == 40 and isinstance(revision['modified'], bool)
        out = tmp_path / 'output'
        out.mkdir()
        output_manifest.write_output_manifest({'opath': str(out)}, start_time=0.0)
        run = json.loads((out / output_manifest.MANIFEST).read_text())['runs'][-1]
        assert run['paoflow_commit'] == revision['commit']
        assert run['paoflow_tree_modified'] == revision['modified']
    else:
        assert revision is None

    def no_git(*args, **kwargs):
        raise FileNotFoundError('git')

    output_manifest.source_revision.cache_clear()
    monkeypatch.setattr(output_manifest.subprocess, 'run', no_git)
    try:
        assert output_manifest.source_revision() is None
        record = output_manifest.run_record({}, 0.0, str(tmp_path))
        assert 'paoflow_commit' not in record and record['paoflow']
    finally:
        output_manifest.source_revision.cache_clear()


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


def test_large_fermi_mesh_is_coarsened_not_holed(tmp_path, monkeypatch):
    """Above the browser budget the sheets are computed on a coarser marching-cubes step, so they
    stay whole (faces sampled at random punched holes into a smoothed surface)."""
    pytest.importorskip('skimage')
    from skimage import measure

    from PAOFLOW.results import volumes

    n = 17
    k = np.linspace(0, 1, n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    band = -0.5 - 1.0 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz))
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _bxsf(tmp_path / 'FermiSurf_0.bxsf', {5: band}, vectors)
    path = [str(tmp_path / 'FermiSurf_0.bxsf')]
    full = volumes.fermi_mesh(path, interp=2, zone='cell')
    assert full['step'] == 1
    faces = len(full['sheets'][0]['i'])
    monkeypatch.setattr(volumes, 'MAX_FACES', faces // 2)
    coarse = volumes.fermi_mesh(path, interp=2, zone='cell')
    assert coarse['step'] == 2 and len(coarse['sheets'][0]['i']) <= faces // 2
    # every face of the coarser marching cubes is there: none was dropped
    from PAOFLOW.gen.fermi_plotter import fft_upsample

    _verts, expected = measure.marching_cubes(fft_upsample(band, 2), level=0.0, step_size=2)[:2]
    assert len(coarse['sheets'][0]['i']) == len(expected)
    # folded into the zone, the triangles share their vertices (smooth shading, smaller data)
    folded = volumes.fermi_mesh(path, interp=1, zone='ws')['sheets'][0]
    xyz = np.stack([folded['x'], folded['y'], folded['z']], axis=1)
    assert len(np.unique(np.round(xyz, 5), axis=0)) == len(xyz) < 3 * len(folded['i'])
    faces = np.stack([folded['i'], folded['j'], folded['k']], axis=1)
    assert faces.max() < len(xyz) and np.all(faces[:, 0] != faces[:, 1])


def test_photocurrent_channels_of_a_metal_have_readable_names(tmp_path):
    """pc{r,i}_<channel>_<abc>.dat (the metal engine of nonlinear_optics) by channel, named as the
    legends of plot.py; the part of a complex quantity is spelt out in the title."""
    from PAOFLOW.graphics.nonlinear_plots import CHANNEL_LABELS
    from PAOFLOW.results import ResultsFolder

    energies = np.linspace(0.1, 2.0, 5)
    for part in 'ri':
        np.savetxt(tmp_path / ('pc%s_xxz.dat' % part), np.c_[energies, energies])
        for channel in CHANNEL_LABELS:
            np.savetxt(tmp_path / ('pc%s_%s_xxz.dat' % (part, channel)), np.c_[energies, energies])
    folder = ResultsFolder(str(tmp_path))
    channels = [r for r in folder.datasets if r.kind == 'photocurrent_channels']
    assert sorted(r.title for r in channels) == [
        'Photocurrent by channel (imaginary part) σ_xxz',
        'Photocurrent by channel (real part) σ_xxz',
    ]
    names = {s.name for s in folder.load(channels[0].id).series}
    assert names == {label for label, _ in CHANNEL_LABELS.values()}
    totals = [r for r in folder.datasets if r.kind == 'nlo']
    assert len(totals) == 2 and any('circular' in r.title for r in totals)


def test_finite_bias_and_nernst_loaders(tmp_path):
    """The finite-bias files of Transport.compute_current_self_consistent and the Nernst table of
    PAOFLOW.transport(do_hall=True), in the formats of their writers."""
    biases, energies = (0.0, 0.1, 0.2), np.linspace(-0.3, 0.3, 7)
    with open(tmp_path / 'transmission_self_consistent_x.dat', 'w') as handle:
        handle.write('# V (volt)   E (eV)   T(E; V)\n')
        for bias in biases:
            for energy in energies:
                handle.write('%12.6f %14.6f %16.8e\n' % (bias, energy, 1.0 - bias))
            handle.write('\n')
    z = np.linspace(0.0, 10.0, 5)
    np.savetxt(tmp_path / 'potential_self_consistent_x.dat', np.column_stack([z, 0 * z, z, -z]),
               header='z (bohr), then eV at V = +0.0000 +0.1000 +0.2000')  # fmt: skip
    rows = [[300.0, e] + [k + e for k in range(27)] for e in energies]
    np.savetxt(tmp_path / 'nernst_gauss_0.dat', rows)
    kinds, _refs = _by_kind(tmp_path)
    transmission = load(kinds['finite_bias_transmission'][0], str(tmp_path))
    assert [series.name for series in transmission.series] == ['0 V', '0.1 V', '0.2 V']
    assert np.allclose(transmission.series[1].y, 0.9)
    assert transmission.options['temperatures']['label'] == 'Biases (V)'
    potential = load(kinds['finite_bias_potential'][0], str(tmp_path))
    assert [series.name for series in potential.series] == ['+0.0000', '+0.1000', '+0.2000']
    nernst = load(kinds['nernst'][0], str(tmp_path))
    assert [series.name for series in nernst.series] == ['300 K']
    assert np.allclose(nernst.series[0].y, energies)  # N_xyz, the first component


def test_inputs_of_a_calculation_are_listed(tmp_path):
    """Scripts, DFT inputs, README and case.json show under Inputs, whole and read-only; the
    other files with these endings stay out, and an output folder points to the folder above."""
    from PAOFLOW.results.discovery import discover
    from PAOFLOW.results.loaders import load
    from PAOFLOW.writers.output_manifest import MANIFEST

    (tmp_path / 'output').mkdir()
    script = ''.join('x%d = %d\n' % (n, n) for n in range(600))  # longer than the text preview
    (tmp_path / 'main.py').write_text(script)
    (tmp_path / 'scf.in').write_text('&control\n/\n')
    (tmp_path / 'README.md').write_text('# Si\n')
    (tmp_path / 'case.json').write_text('{"title": "Si"}\n')
    (tmp_path / 'NOTES.md').write_text('not an input')
    (tmp_path / 'scf.out').write_text('a log')
    run = {'run_id': 'a', 'paoflow': '4', 'status': 'finished', 'script': {'file': 'main.py'}}
    (tmp_path / 'output' / MANIFEST).write_text(json.dumps({'manifest': 2, 'runs': [run]}))
    inputs = {ref.files[0]: ref for ref in discover(str(tmp_path)) if ref.category == 'Inputs'}
    assert set(inputs) == {'main.py', 'scf.in', 'README.md', 'case.json'}
    assert load(inputs['main.py'], str(tmp_path), {}).text == script.rstrip('\n')
    record = [ref for ref in discover(str(tmp_path / 'output')) if ref.kind == 'manifest'][0]
    assert (
        'open the folder the script ran in' in load(record, str(tmp_path / 'output'), {}).notes[0]
    )
    record = [ref for ref in discover(str(tmp_path)) if ref.kind == 'manifest'][0]
    assert not load(record, str(tmp_path), {}).notes
