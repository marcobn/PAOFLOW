"""PAOFLOW Studio in a real browser (headless Chrome through Selenium).

Marked ``integration`` and ``slow``; skipped without Selenium or a Chrome that starts.  The tests
drive the three tabs on a copy of the silicon example: the Setup form writes the files the form
resolver previews, the band-path preview draws, the Results tab plots (stacked PDOS, map cuts),
the Runs tab runs a driver, and no page logs an error.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

webdriver = pytest.importorskip('selenium.webdriver')
By = pytest.importorskip('selenium.webdriver.common.by').By

EXAMPLE = Path(__file__).resolve().parents[3] / 'examples' / 'qe_examples' / 'example01'


@pytest.fixture(scope='module')
def browser():
    options = webdriver.ChromeOptions()
    for argument in (
        '--headless=new',
        '--window-size=1500,1000',
        '--use-angle=swiftshader',
        '--enable-unsafe-swiftshader',
    ):
        options.add_argument(argument)
    options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as exc:  # no Chrome or no driver
        pytest.skip('Chrome is not available: %s' % exc)
    yield driver
    driver.quit()


@pytest.fixture()
def studio(tmp_path):
    from PAOFLOW.gui.server import GuiApp, start_in_thread

    work = tmp_path / 'Si'
    work.mkdir()
    shutil.copytree(EXAMPLE / 'silicon.save', work / 'silicon.save')
    shutil.copy(EXAMPLE / 'scf.in', work / 'scf.in')
    shutil.copy(EXAMPLE / 'Si.pbe-tm-new-gipaw-v2.1.UPF', work / 'Si.pbe-tm-new-gipaw-v2.1.UPF')
    shutil.copytree(EXAMPLE / 'Reference', work / 'output')
    app = GuiApp(tmp_path, start='Si', max_ranks=2)
    server, _ = start_in_thread(app)
    yield server, work
    server.shutdown()
    server.server_close()


def _wait(condition, timeout=20.0, step=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(step)
    raise AssertionError('condition not met in %.0f s' % timeout)


def _errors(driver):
    return [e['message'] for e in driver.get_log('browser') if e['level'] == 'SEVERE']


def _open(driver, server, tab):
    driver.get(server.url + '#' + tab)
    _wait(
        lambda: driver.execute_script(
            'return window.paoflowApp && window.paoflowApp.setup && window.paoflowApp.setup.workflows.length > 0'
        )
    )


def test_setup_writes_the_resolved_files(browser, studio):
    from PAOFLOW.gen.outputs import workflow_outputs
    from PAOFLOW.gen.questions import WORKFLOWS
    from PAOFLOW.gen.questions.model import Context
    from PAOFLOW.gen.questions.resolve import resolve_form

    server, work = studio
    _open(browser, server, 'setup')
    browser.execute_script('localStorage.clear()')
    browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="run"]').click()
    cards = _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '.choice-card'))
    cards[0].click()
    cards[1].click()
    _wait(lambda: 'files' in browser.find_element(By.CSS_SELECTOR, '.preview-foot .muted').text)
    _wait(
        lambda: browser.find_elements(By.CSS_SELECTOR, '.bz-preview .main-svg, .bz-preview canvas')
    )
    browser.find_element(By.XPATH, "//button[text()='Write files']").click()
    _wait(lambda: (work / 'main.py').exists() and (work / 'plot.py').exists())
    form = resolve_form(
        WORKFLOWS['run'], {'properties': ['bands', 'dos']}, Context(workdir=str(work))
    )
    expected = {f.path: f.content() for f in workflow_outputs('run', form.config)}
    assert (work / 'main.py').read_text() == expected['main.py']
    assert (work / 'plot.py').read_text() == expected['plot.py']
    assert _errors(browser) == []


def test_results_stacked_pdos_and_map_cuts(browser, studio):
    server, work = studio
    arpes = work / 'arpes' / 'output'
    arpes.mkdir(parents=True)
    k = np.linspace(0.0, 1.0, 21)
    w = np.linspace(-1.0, 0.2, 41)
    intensity = np.exp(-((w[None, :] + 0.5 * k[:, None]) ** 2) / 0.01)
    np.savez(
        arpes / 'arpes_iso_10K.npz',
        kpoints=np.stack([k, 0 * k, 0 * k], axis=1),
        w=w,
        intensity=intensity,
    )
    _open(browser, server, 'results')
    browser.execute_script("window.paoflowApp.results.open('Si')")
    _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '.tree-item'))
    item = next(
        i
        for i in browser.find_elements(By.CSS_SELECTOR, '.tree-item')
        if i.text.startswith('Projected density')
    )
    item.click()
    _wait(
        lambda: browser.execute_script(
            "const p = document.querySelector('.plot'); return p && p.data && p.data.length > 2"
        )
    )
    browser.execute_script("window.paoflowApp.results.setOption('display', 'stacked')")
    _wait(
        lambda: browser.execute_script(
            "return document.querySelector('.plot').data.some((t) => t.stackgroup === 'pdos')"
        )
    )
    item = next(
        i
        for i in browser.find_elements(By.CSS_SELECTOR, '.tree-item')
        if i.text.startswith('ARPES')
    )
    item.click()
    _wait(
        lambda: browser.execute_script(
            "const p = document.querySelector('.plot'); return p && p.data && p.data[0].type === 'heatmap'"
        )
    )
    browser.execute_script(
        'const v = window.paoflowApp.results;'
        "v.showCuts({points: [{data: {type: 'heatmap'}, pointIndex: [10, 5]}]}, v.result);"
    )
    _wait(lambda: len(browser.find_elements(By.CSS_SELECTOR, '.cut-plot .main-svg')) >= 2)
    assert _errors(browser) == []


def test_runs_tab_runs_a_driver(browser, studio):
    server, work = studio
    (work / 'hello.py').write_text("print('hello from the Runs tab')\n")
    _open(browser, server, 'runs')
    browser.execute_script(
        "window.paoflowApp.showRuns('Si', [{label: 'hello', argv: ['python', 'hello.py'], cwd: '.', command: 'python hello.py', mpi: false}], 'test')"
    )
    button = _wait(
        lambda: [
            b for b in browser.find_elements(By.XPATH, "//button[text()='Run']") if b.is_displayed()
        ]
    )
    button[0].click()
    _wait(
        lambda: 'hello from the Runs tab' in browser.find_element(By.CSS_SELECTOR, 'pre.log').text,
        timeout=60,
    )
    _wait(
        lambda: 'finished' in browser.find_element(By.CSS_SELECTOR, '#view-runs .result-head').text,
        timeout=60,
    )
    assert _errors(browser) == []


def test_picker_reaches_read_only_folders(browser, studio):
    from PAOFLOW.gui.server import paoflow_data_folders

    folders = [f for f in paoflow_data_folders() if f.name == 'PSEUDOS']
    if not folders:
        pytest.skip('no PSEUDOS folder next to this PAOFLOW')
    server, work = studio
    _open(browser, server, 'setup')
    browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="qe_input"]').click()
    _wait(lambda: browser.find_elements(By.XPATH, "//button[text()='Browse…']"))
    browser.find_elements(By.XPATH, "//button[text()='Browse…']")[0].click()
    places = _wait(
        lambda: [
            b
            for b in browser.find_elements(By.CSS_SELECTOR, '.places button')
            if 'PSEUDOS' in b.text
        ]
    )
    places[0].click()
    _wait(lambda: 'read-only' in browser.find_element(By.CSS_SELECTOR, '.crumbs').text)
    browser.find_element(By.XPATH, "//button[text()='Use this folder']").click()
    field = _wait(
        lambda: [
            i
            for i in browser.find_elements(By.CSS_SELECTOR, '#view-setup input[type="text"]')
            if i.get_attribute('value') == str(folders[0])
        ]
    )
    assert field
    assert _errors(browser) == []
