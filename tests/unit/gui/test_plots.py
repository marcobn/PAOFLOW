"""Plots tab endpoints: the plot scripts of a folder, a run without a display, its figures."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.unit

PLOT = '''\
"""Figures of the demo results."""
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true')
parser.add_argument("--emin", type=float)
args = parser.parse_args()
if not args.all:
    raise SystemExit('nothing selected')
plt.plot([0, 1], [1, 0])
plt.title('Band structure')
plt.show()
'''


def _wait(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get('/api/jobs/log?id=%s' % job_id)[1]
        if body['status'] != 'running':
            return body
        time.sleep(0.2)
    raise AssertionError('job still running')


def test_scripts_of_a_folder(client, gui_root):
    work = gui_root / 'Si'
    (work / 'plot.py').write_text(PLOT)
    (work / 'plot.acbn0.py').write_text('print(1)\n')
    (work / 'superconductivity').mkdir()
    (work / 'superconductivity' / 'plot_me.py').write_text('print(1)\n')
    (work / 'output').mkdir()
    (work / 'output' / 'plot_old.py').write_text('print(1)\n')  # not a script folder
    (work / 'plotting_notes.txt').write_text('')
    status, body, _ = client.get('/api/plots?path=Si')
    assert status == 200 and body['folder'] == 'Si' and body['writable']
    scripts = {s['script']: s for s in body['scripts']}
    assert list(scripts) == ['plot.acbn0.py', 'plot.py', 'superconductivity/plot_me.py']
    plot = scripts['plot.py']
    assert plot['title'] == 'Figures of the demo results.'
    assert plot['options'] == ['--all', '--emin'] and plot['arguments'] == ['--all']
    assert plot['capture'] is None and plot['job'] is None
    assert scripts['plot.acbn0.py']['arguments'] == []


def test_a_script_runs_without_a_display_and_its_figures_are_served(client, gui_root):
    work = gui_root / 'Si'
    (work / 'plot.py').write_text(PLOT)
    step = {
        'label': 'Figures',
        'argv': ['python', 'plot.py', '--all'],
        'cwd': 'Si',
        'capture': True,
    }
    status, job, _ = client.post('/api/jobs/launch', {'step': step, 'ranks': 1})
    assert status == 200 and 'PAOFLOW.graphics.figure_capture' in job['argv']
    log = _wait(client, job['id'])
    assert log['status'] == 'finished', log['text']
    body = client.get('/api/plots?path=Si')[1]
    plot = next(s for s in body['scripts'] if s['script'] == 'plot.py')
    assert plot['job']['id'] == job['id'] and plot['job']['status'] == 'finished'
    capture = plot['capture']
    assert capture['status'] == 0 and capture['folder'] == 'Si/figures/plot'
    assert capture['arguments'] == ['--all']
    [figure] = capture['figures']
    assert figure['title'] == 'Band structure'
    assert figure['files'] == {
        'png': 'Si/figures/plot/01_Band_structure.png',
        'pdf': 'Si/figures/plot/01_Band_structure.pdf',
    }
    status, png, response = client.get('/api/plots/file?path=%s' % figure['files']['png'])
    assert status == 200 and png.startswith(b'\x89PNG')
    assert response.getheader('Content-Type') == 'image/png'
    assert response.getheader('Content-Disposition').startswith('inline')
    status, pdf, response = client.get('/api/plots/file?path=%s' % figure['files']['pdf'])
    assert status == 200 and pdf.startswith(b'%PDF')
    assert response.getheader('Content-Disposition').startswith('attachment')
    # the scripts and folders the server serves nothing else from
    assert client.get('/api/plots/file?path=Si/plot.py')[0] == 400
    assert client.get('/api/plots/file?path=../x.png')[0] == 403


def test_a_failed_run_is_reported(client, gui_root):
    work = gui_root / 'Si'
    (work / 'plot.py').write_text(PLOT)
    step = {'label': 'Figures', 'argv': ['python', 'plot.py'], 'cwd': 'Si', 'capture': True}
    job = client.post('/api/jobs/launch', {'step': step, 'ranks': 1})[1]
    log = _wait(client, job['id'])
    assert log['status'] == 'failed' and 'nothing selected' in log['text']
    plot = client.get('/api/plots?path=Si')[1]['scripts'][0]
    assert plot['capture']['status'] == 1 and plot['capture']['figures'] == []
    assert plot['job']['status'] == 'failed'


def test_read_only_session_lists_but_does_not_run(make_client, gui_root):
    (gui_root / 'Si' / 'plot.py').write_text(PLOT)
    reader = make_client(read_only=True)
    body = reader.get('/api/plots?path=Si')[1]
    assert not body['writable'] and [s['script'] for s in body['scripts']] == ['plot.py']
    step = {'argv': ['python', 'plot.py', '--all'], 'cwd': 'Si', 'capture': True}
    assert reader.post('/api/jobs/launch', {'step': step})[0] == 403
