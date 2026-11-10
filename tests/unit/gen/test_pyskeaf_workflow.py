"""The pyskeaf workflow: questions, main.pyskeaf.py, its run step, and the dHvA plot axis."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.gen.outputs import run_plan, workflow_outputs
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form

pytestmark = pytest.mark.unit

PT = Path(__file__).resolve().parents[3] / 'examples' / 'qe_examples' / 'example08' / 'Reference'


def _resolve(workdir, answers):
    form = resolve_form(WORKFLOWS['pyskeaf'], answers, Context(workdir=str(workdir)))
    items = {i['key']: i for i in form.items if 'key' in i}
    notes = [i['text'] for i in form.items if i.get('type') == 'note' and i['visible']]
    return form, items, notes


@pytest.fixture
def run_folder(tmp_path):
    (tmp_path / 'output').mkdir()
    for band in (1, 2):
        shutil.copy(PT / ('Fermi_surf_band_%d.bxsf' % band), tmp_path / 'output')
    return tmp_path


def test_rotation_by_default_and_the_driver(run_folder):
    form, items, notes = _resolve(run_folder, {})
    assert form.complete and items['outputdir']['value'] == 'output'
    assert any('1, 2' in n for n in notes)
    cfg = form.config
    assert cfg['b_field'] == 'rotation' and cfg['azimuthal'] == (0.0, 0.0)
    assert cfg['polar'] == (0.0, 90.0) and cfg['num_angles'] == 37 and cfg['bands'] == 'all'
    driver = workflow_outputs('pyskeaf', cfg)
    assert [f.path for f in driver] == ['main.pyskeaf.py', 'plot.pyskeaf.py']
    compile(driver[1].content(), 'plot.pyskeaf.py', 'exec')
    text = driver[0].content()
    compile(text, 'main.pyskeaf.py', 'exec')
    assert 'THETA = (0.0, 0.0)' in text and 'PHI = (0.0, 90.0)' in text
    assert 'paoflow.pyskeaf(' in text and 'run_main(main)' in text
    step, figures = run_plan('pyskeaf', cfg)
    assert step.mpi and step.argv == ('python', 'main.pyskeaf.py')
    assert figures.argv == ('python', 'plot.pyskeaf.py') and not figures.interactive  # saves PDFs


def test_field_modes_and_checks(run_folder):
    form, items, _ = _resolve(
        run_folder, {'b_field': 'non_principal', 'theta': '30', 'phi': '45', 'bands': '2'}
    )
    assert form.complete and not items['theta_start']['visible']
    assert (form.config['azimuthal'], form.config['polar'], form.config['num_angles']) == (
        30.0,
        45.0,
        1,
    )
    assert form.config['bands'] == [2]
    form, _, _ = _resolve(run_folder, {'b_field': 'b3', 'fermi_energy': '-0.05, 0, 0.05'})
    assert form.config['azimuthal'] == 0.0 and form.config['fermi_energy'] == [-0.05, 0.0, 0.05]
    form, _, _ = _resolve(run_folder, {'num_angles': '1'})
    assert 'num_angles' in form.errors
    form, _, _ = _resolve(run_folder, {'bands': 'x'})
    assert 'bands' in form.errors


def test_without_bxsf_files_the_form_says_so(tmp_path):
    form, _, notes = _resolve(tmp_path, {})
    assert form.complete  # the driver can be written before the Property run
    assert any('No Fermi_surf_band_*.bxsf' in n for n in notes)


def test_dhva_axis_follows_the_varying_angle(tmp_path):
    from PAOFLOW.results import ResultsFolder

    header = ' Azimuthal(deg) ,    Polar(deg)  ,      Freq(kT)    ,     mstar(me) \n'
    polar = np.array([[0.0, p, 1.0 + p / 90, 0.5] for p in (0.0, 45.0, 90.0)])
    np.savetxt(
        tmp_path / 'qo_EF_0_freqvsangle_1.out', polar, delimiter=',', header=header, comments=''
    )
    folder = ResultsFolder(str(tmp_path))
    ref = next(r for r in folder.datasets if r.kind == 'dhva')
    dataset = folder.load(ref.id)
    assert dataset.options['angle']['value'] == 'polar φ' and 'φ' in dataset.x_label
    assert list(dataset.series[0].x) == [0.0, 45.0, 90.0]
    dataset = folder.load(ref.id, {'angle': 'azimuthal θ'})
    assert list(dataset.series[0].x) == [0.0, 0.0, 0.0] and 'θ' in dataset.x_label


def test_the_plot_script_draws_the_results(run_folder):
    """plot.pyskeaf.py saves the dHvA figure of the Results layer as a PDF."""
    import subprocess
    import sys

    pytest.importorskip('matplotlib')
    header = ' Azimuthal(deg) ,    Polar(deg)  ,      Freq(kT)    ,     mstar(me) \n'
    rows = np.array([[0.0, p, 1.0 + p / 90, 0.5] for p in (0.0, 45.0, 90.0)])
    np.savetxt(
        run_folder / 'output' / 'qo_EF_0_freqvsangle_1.out',
        rows,
        delimiter=',',
        header=header,
        comments='',
    )
    form, _, _ = _resolve(run_folder, {})
    for generated in workflow_outputs('pyskeaf', form.config):
        generated.write(str(run_folder))
    done = subprocess.run(
        [sys.executable, 'plot.pyskeaf.py'], cwd=run_folder, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    pdf = run_folder / 'output' / 'figures' / 'qo_EF_0_freqvsangle_1.pdf'
    assert pdf.read_bytes()[:4] == b'%PDF'
