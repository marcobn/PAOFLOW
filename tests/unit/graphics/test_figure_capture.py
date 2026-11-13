"""python -m PAOFLOW.graphics.figure_capture: a plot script without a display, figures as files.

The runner patches matplotlib and ``input`` for the script it runs, so the tests run it in a
subprocess, as the Plots tab of paoflow-studio does.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from PAOFLOW.graphics.figure_capture import capture_folder, figure_title

pytestmark = pytest.mark.unit

SCRIPT = """\
import argparse

import matplotlib

matplotlib.use('TkAgg')  # ignored: there is no display
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true')
args = parser.parse_args()
print('all', args.all)
print('answer [%s]' % input('EMIN [-5]: '))
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
ax.set_title('Band structure')
fig.savefig('bands.png')
fig.savefig('bands.pdf')  # the same figure again: one figure
plt.show()  # already captured: not again
ax.plot([1, 0])
fig.savefig('bands_2.png')  # changed: a new figure
second = plt.figure()
second.suptitle('DOS')
second.add_subplot().plot([1, 2])
# left open: captured when the script ends
"""


def _run(folder, *arguments):
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'PAOFLOW.graphics.figure_capture',
            '--out',
            'figures/demo',
            *arguments,
        ],
        cwd=folder,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def test_figures_of_a_script_are_kept_once_each(tmp_path):
    (tmp_path / 'plot_demo.py').write_text(SCRIPT)
    result = _run(tmp_path, 'plot_demo.py', '--all')
    assert result.returncode == 0, result.stderr
    assert 'all True' in result.stdout and 'answer []' in result.stdout  # default answer
    assert (tmp_path / 'bands.png').is_file()  # the script's own files are still written
    manifest = json.loads((tmp_path / 'figures' / 'demo' / 'figures.json').read_text())
    assert manifest['status'] == 0 and manifest['arguments'] == ['--all']
    figures = manifest['figures']
    assert [f['title'] for f in figures] == ['Band structure', 'Band structure', 'DOS']
    assert [f['saved_as'] for f in figures] == ['bands.png', 'bands_2.png', None]
    for figure in figures:
        for fmt in ('png', 'pdf'):
            assert (tmp_path / 'figures' / 'demo' / figure['files'][fmt]).stat().st_size > 0
    assert figures[0]['files']['png'] == '01_Band_structure.png'


def test_a_new_run_replaces_the_files_and_an_error_keeps_the_figures(tmp_path):
    (tmp_path / 'plot_demo.py').write_text(SCRIPT)
    assert _run(tmp_path, 'plot_demo.py').returncode == 0
    folder = tmp_path / 'figures' / 'demo'
    (folder / 'notes.txt').write_text('mine')
    (tmp_path / 'plot_fail.py').write_text(
        'import matplotlib.pyplot as plt\n'
        "plt.plot([0, 1]); plt.title('Before the error')\n"
        'plt.show()\n'
        "raise SystemExit('no data: run main.py first')\n"
    )
    result = _run(tmp_path, 'plot_fail.py')
    assert result.returncode == 1 and 'no data' in result.stderr
    manifest = json.loads((folder / 'figures.json').read_text())
    assert manifest['status'] == 1
    assert [f['title'] for f in manifest['figures']] == ['Before the error']
    names = sorted(p.name for p in folder.iterdir())
    # the files of the previous run are gone; a file the runner did not write stays
    assert names == [
        '01_Before_the_error.pdf',
        '01_Before_the_error.png',
        'figures.json',
        'notes.txt',
    ]


def test_capture_folder_and_figure_title():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert capture_folder('plot.py') == 'figures/plot'
    assert capture_folder('plot.acbn0.py') == 'figures/plot.acbn0'
    assert capture_folder('superconductivity/plot_me.py') == 'figures/superconductivity_plot_me'
    fig, axes = plt.subplots(1, 2)
    assert figure_title(fig) == ''
    axes[1].set_title('right panel')
    assert figure_title(fig) == 'right panel'
    fig.suptitle('Whole figure')
    assert figure_title(fig) == 'Whole figure'
    plt.close(fig)


def test_untitled_figures_take_the_section_the_script_prints(tmp_path):
    (tmp_path / 'plot_sections.py').write_text(
        'import matplotlib.pyplot as plt\n'
        "print('\\n== Photocurrent by channel ==')\n"
        'for _ in range(2):\n'
        '    plt.plot([0, 1])\n'
        '    plt.show()\n'
        "print('== Band structure ==')\n"
        "plt.plot([1, 0]); plt.title('Bands of Si')\n"  # its own title wins
        'plt.show()\n'
    )
    assert _run(tmp_path, 'plot_sections.py').returncode == 0
    manifest = json.loads((tmp_path / 'figures' / 'demo' / 'figures.json').read_text())
    assert [f['title'] for f in manifest['figures']] == [
        'Photocurrent by channel',
        'Photocurrent by channel (2)',
        'Bands of Si',
    ]
