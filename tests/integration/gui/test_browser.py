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
    # The guide opens by itself on the first visit of an origin; close it so it covers nothing.
    driver.execute_script('window.paoflowApp.help.close()')


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


def test_guide_opens_on_first_visit(browser, studio):
    server, _work = studio
    browser.get(server.url + '#setup')
    panel = _wait(
        lambda: [
            p for p in browser.find_elements(By.CSS_SELECTOR, '.help-panel') if p.is_displayed()
        ]
    )
    assert 'Getting started' in panel[0].find_element(By.CSS_SELECTOR, '.help-page h2').text
    assert '[object' not in panel[0].text and 'Property run' in panel[0].text
    links = len(browser.find_elements(By.CSS_SELECTOR, '.help-link'))
    search = panel[0].find_element(By.CSS_SELECTOR, 'input[type="search"]')
    search.send_keys('pw.x')
    titles = _wait(lambda: [b.text for b in browser.find_elements(By.CSS_SELECTOR, '.help-link')])
    assert 0 < len(titles) < links and 'Runs' in titles
    next(
        b for b in browser.find_elements(By.CSS_SELECTOR, '.help-link') if b.text == 'Runs'
    ).click()
    page = browser.find_element(By.CSS_SELECTOR, '.help-page')
    assert page.find_element(By.TAG_NAME, 'h2').text == 'Runs'
    assert '[object' not in page.text and len(page.find_elements(By.TAG_NAME, 'li')) >= 5
    assert page.find_elements(By.TAG_NAME, 'code')
    browser.find_element(By.TAG_NAME, 'body').send_keys('')  # Escape
    _wait(lambda: not browser.find_element(By.CSS_SELECTOR, '.help-panel').is_displayed())
    browser.find_element(By.ID, 'help').click()
    _wait(lambda: browser.find_element(By.CSS_SELECTOR, '.help-panel').is_displayed())
    assert _errors(browser) == []


def test_qe_input_then_pw_run_then_setup(browser, studio, tmp_path, monkeypatch):
    import os
    import types

    from PAOFLOW.gen.qe_input import cli

    server, work = studio
    bin_dir = tmp_path / 'qe_bin'
    bin_dir.mkdir()
    program = bin_dir / 'pw.x'
    program.write_text('#!/bin/sh\necho "Program PWSCF (stand-in)"\necho "JOB DONE."\n')
    program.chmod(0o755)
    monkeypatch.setenv('PATH', '%s:%s' % (bin_dir, os.environ['PATH']))
    record = types.SimpleNamespace(compound='Si2', natoms=2, species=['Si'], metallic=False, magnetic=False,
                                   dimensionality='3D', kpoints=None, source='aflow', spacegroup=None)  # fmt: skip
    monkeypatch.setattr(cli, 'generate_qe_input', lambda **kw: (record, '&control\n/\n'))
    _open(browser, server, 'setup')
    browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="qe_input"]').click()
    identifier = _wait(
        lambda: browser.find_elements(By.CSS_SELECTOR, 'input[placeholder^="aflow:"]')
    )
    identifier[0].send_keys('aflow:abc')
    browser.find_element(By.XPATH, "//button[text()='Fetch & preview']").click()
    write = _wait(
        lambda: [
            b for b in browser.find_elements(By.XPATH, "//button[text()='Write']") if b.is_enabled()
        ]
    )
    write[0].click()
    _wait(lambda: browser.find_elements(By.XPATH, "//button[text()='Open in Runs']"))[0].click()
    ranks = _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '#view-runs input[type="number"]'))
    ranks[0].clear()
    ranks[0].send_keys('1')
    [b for b in browser.find_elements(By.XPATH, "//button[text()='Run']") if b.is_displayed()][
        0
    ].click()
    _wait(
        lambda: 'finished' in browser.find_element(By.CSS_SELECTOR, '#view-runs .result-head').text,
        timeout=60,
    )
    assert 'JOB DONE.' in (work / 'Si2.scf.out').read_text()
    next(
        b
        for b in browser.find_elements(By.CSS_SELECTOR, '#view-runs button')
        if b.text.startswith('Open Setup')
    ).click()
    _wait(  # read in the page: the workflow list re-renders while the Setup tab opens
        lambda: browser.execute_script(
            "const item = document.querySelector('.workflow-item.selected');"
            'return item && item.dataset.name;'
        )
        == 'run'
    )
    assert browser.find_element(By.CSS_SELECTOR, '.tab[aria-selected="true"]').text == 'Setup'
    assert _errors(browser) == []


def _switch_of(driver, name):
    """The checkbox and the visible track of the switch in the Setup field called ``name``."""
    xpath = (
        "//*[@id='view-setup']//div[contains(@class, 'field')]"
        "[.//label[contains(@class, 'name') and text()=\"%s\"]]//*[contains(@class, 'switch')]"
        % name
    )
    switch = _wait(lambda: [s for s in driver.find_elements(By.XPATH, xpath) if s.is_displayed()])[
        0
    ]
    return switch.find_element(By.CSS_SELECTOR, 'input'), switch.find_element(
        By.CSS_SELECTOR, 'span'
    )


def test_switches_toggle_and_reach_the_server(browser, studio, monkeypatch):
    """Clicking the track of a switch flips it (a switch outside a <label> did not), and the
    QE-input switches reach the generator."""
    import types

    from PAOFLOW.gen.qe_input import cli

    server, _work = studio
    calls = []
    record = types.SimpleNamespace(compound='MoS2', natoms=3, species=['Mo', 'S'], metallic=False, magnetic=False,
                                   dimensionality='2D', kpoints=None, source='c2db', spacegroup=None)  # fmt: skip

    def generate(**kwargs):
        calls.append(kwargs)
        return record, '&system\n/\n'

    monkeypatch.setattr(cli, 'generate_qe_input', generate)
    _open(browser, server, 'setup')
    fields = {
        'run': ['Always write the plot script'],
        'basis': ['Keep existing tables'],
        'qe_input': ['Spin-orbit cards', "assume_isolated = '2D'"],
    }
    for workflow, names in fields.items():
        browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="%s"]' % workflow).click()
        for name in names:
            box, track = _switch_of(browser, name)
            before = box.is_selected()
            track.click()
            assert box.is_selected() is not before, name
    # Spin-orbit starts off and assume_isolated on: both are flipped now.
    browser.find_element(By.CSS_SELECTOR, 'input[placeholder^="aflow:"]').send_keys('2MoS2-1')
    browser.find_element(By.XPATH, "//button[text()='Fetch & preview']").click()
    _wait(lambda: calls)
    assert calls[-1]['assume_isolated_2d'] is False
    assert calls[-1]['soc'] is True
    assert _errors(browser) == []


def test_existing_pw_input_runs_from_setup_and_runs(browser, studio, tmp_path, monkeypatch):
    """An input already on disk: listed in the QE-input tool with its run, and loadable in Runs."""
    import os

    server, work = studio
    bin_dir = tmp_path / 'qe_bin'
    bin_dir.mkdir()
    program = bin_dir / 'pw.x'
    program.write_text('#!/bin/sh\necho "Program PWSCF (stand-in)"\necho "JOB DONE."\n')
    program.chmod(0o755)
    monkeypatch.setenv('PATH', '%s:%s' % (bin_dir, os.environ['PATH']))
    (work / 'gaas.scf.in').write_text("&control\n  calculation = 'scf'\n/\n")
    _open(browser, server, 'setup')
    browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="qe_input"]').click()
    run = _wait(
        lambda: [
            row.find_element(By.TAG_NAME, 'button')
            for row in browser.find_elements(By.CSS_SELECTOR, '.input-row')
            if 'gaas.scf.in' in row.text
        ]
    )
    run[0].click()
    _wait(lambda: browser.find_elements(By.XPATH, "//button[text()='Open in Runs']"))[0].click()
    ranks = _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '#view-runs input[type="number"]'))
    ranks[0].clear()
    ranks[0].send_keys('1')
    [b for b in browser.find_elements(By.XPATH, "//button[text()='Run']") if b.is_displayed()][
        0
    ].click()
    _wait(
        lambda: 'finished' in browser.find_element(By.CSS_SELECTOR, '#view-runs .result-head').text,
        timeout=60,
    )
    assert 'JOB DONE.' in (work / 'gaas.scf.out').read_text()
    # Runs tab: the same plan from the file picker, with no plan in memory.
    browser.execute_script("window.paoflowApp.runs.setPlan(null, [], '')")
    browser.find_element(By.XPATH, "//button[text()='Run a pw.x input…']").click()
    entry = _wait(
        lambda: [
            e
            for e in browser.find_elements(By.CSS_SELECTOR, 'dialog .browser-entry')
            if 'gaas.scf.in' in e.text
        ]
    )
    entry[0].click()
    browser.find_element(By.XPATH, "//button[text()='Use the selected file']").click()
    plan_card = browser.find_element(By.CSS_SELECTOR, '#view-runs .side .card')
    _wait(lambda: 'pw.x -in gaas.scf.in > gaas.scf.out' in plan_card.text)
    assert 'Self-consistent run (pw.x)' in plan_card.text
    assert _errors(browser) == []


def test_qe_build_is_chosen_in_the_page(browser, studio, tmp_path, monkeypatch):
    """Two QE builds: the one chosen in the QE-input tool runs pw.x, and the choice persists."""
    Select = pytest.importorskip('selenium.webdriver.support.select').Select
    server, work = studio
    bins = {}
    for version in ('7.4.1', '7.5'):
        folder = tmp_path / 'Programs' / ('qe-%s' % version)
        (folder / 'bin').mkdir(parents=True)
        (folder / 'include').mkdir()
        (folder / 'include' / 'qe_version.h').write_text("version_number = '%s'\n" % version)
        program = folder / 'bin' / 'pw.x'
        program.write_text('#!/bin/sh\necho "Program PWSCF v.%s"\necho "JOB DONE."\n' % version)
        program.chmod(0o755)
        bins[version] = str(folder / 'bin')
    monkeypatch.setenv('PATH', '%s:/usr/bin:/bin' % bins['7.4.1'])
    (work / 'gaas.scf.in').write_text("&control\n  calculation = 'scf'\n/\n")
    _open(browser, server, 'setup')
    browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="qe_input"]').click()
    picker = _wait(
        lambda: [
            s
            for s in browser.find_elements(By.CSS_SELECTOR, '#view-setup .qe-picker select')
            if s.find_elements(By.TAG_NAME, 'option')
        ]
    )[0]
    assert Select(picker).first_selected_option.get_attribute('value') == bins['7.4.1']  # PATH
    Select(picker).select_by_value(bins['7.5'])
    runs_picker = browser.find_element(By.CSS_SELECTOR, '#view-runs .qe-picker select')
    assert runs_picker.get_attribute('value') == bins['7.5']  # the Runs tab follows
    next(
        row.find_element(By.TAG_NAME, 'button')
        for row in browser.find_elements(By.CSS_SELECTOR, '.input-row')
        if 'gaas.scf.in' in row.text
    ).click()
    _wait(lambda: browser.find_elements(By.XPATH, "//button[text()='Open in Runs']"))[0].click()
    ranks = _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '#view-runs input[type="number"]'))
    ranks[0].clear()
    ranks[0].send_keys('1')
    [b for b in browser.find_elements(By.XPATH, "//button[text()='Run']") if b.is_displayed()][
        0
    ].click()
    _wait(
        lambda: 'finished' in browser.find_element(By.CSS_SELECTOR, '#view-runs .result-head').text,
        timeout=60,
    )
    assert 'Program PWSCF v.7.5' in (work / 'gaas.scf.out').read_text()
    browser.refresh()  # the choice is remembered
    _wait(
        lambda: browser.execute_script(
            'return Boolean(window.paoflowApp && window.paoflowApp.runs)'
        )
    )
    _wait(
        lambda: browser.find_element(By.CSS_SELECTOR, '#view-runs .qe-picker select').get_attribute(
            'value'
        )
        == bins['7.5']
    )
    assert _errors(browser) == []


def test_zone_preview_drawn_in_a_hidden_tab_fills_its_box(browser, studio):
    """The page opens on Runs and Setup draws the zone while hidden; shown, it fills its box."""
    server, _work = studio
    _open(browser, server, 'setup')
    root = browser.execute_script('return window.paoflowApp.session.root')
    browser.execute_script(
        'localStorage.setItem(arguments[0], JSON.stringify("run"));'
        'localStorage.setItem(arguments[1], JSON.stringify({answers: {properties: ["bands"]}}));',
        'paoflow-studio:workflow:%s:Si' % root,
        'paoflow-studio:answers:%s:Si:run' % root,
    )
    browser.get(server.url + '#runs')
    _wait(
        lambda: browser.execute_script(
            'return Boolean(window.paoflowApp && window.paoflowApp.setup)'
        )
    )
    browser.execute_script('window.paoflowApp.help.close()')
    _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '.bz-preview .main-svg'))
    browser.find_element(By.CSS_SELECTOR, '.tab[data-tab="setup"]').click()
    sizes = """const box = document.querySelector('.bz-preview');
      const svg = box.querySelector('.main-svg');
      return [box.clientWidth, svg.clientWidth, box.clientHeight, svg.clientHeight];"""
    _wait(
        lambda: (lambda s: s[0] > 0 and abs(s[0] - s[1]) <= 4 and abs(s[2] - s[3]) <= 4)(
            browser.execute_script(sizes)
        )
    )
    assert _errors(browser) == []


def _field_value(driver, key):
    return driver.execute_script(
        'const n = document.querySelector(`.field[data-key="${arguments[0]}"] input[type=text]`);'
        'return n ? n.value : null;',
        key,
    )


def test_defaults_follow_the_folder(browser, studio):
    """The save appears while the form is open (pw.x finished): the defaults follow, no retyping."""
    server, work = studio
    fresh = work.parent / 'Fresh'
    fresh.mkdir()
    shutil.copy(work / 'scf.in', fresh / 'scf.in')
    _open(browser, server, 'setup')
    browser.execute_script("window.paoflowApp.setup.openFolder('Fresh')")
    _wait(lambda: browser.execute_script("return window.paoflowApp.setup.workdir === 'Fresh'"))
    browser.find_element(By.CSS_SELECTOR, '.workflow-item[data-name="run"]').click()
    _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '.choice-card'))[0].click()
    _wait(lambda: _field_value(browser, 'savedir') == 'pwscf.save')
    shutil.copytree(work / 'silicon.save', fresh / 'silicon.save')
    shutil.copy(work / 'Si.pbe-tm-new-gipaw-v2.1.UPF', fresh / 'silicon.save')
    _wait(lambda: _field_value(browser, 'savedir') == 'silicon.save', timeout=15)
    assert _field_value(browser, 'upfs') == 'silicon.save/Si.pbe-tm-new-gipaw-v2.1.UPF'
    assert _field_value(browser, 'ibrav') == '2'
    _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '.bz-preview .main-svg'))
    assert len(browser.find_elements(By.CSS_SELECTOR, '.field[data-key="high_sym"] tr')) > 2
    assert browser.execute_script('return window.paoflowApp.setup.form.answers') == {
        'properties': ['bands']
    }
    assert _errors(browser) == []


def test_stale_save_answer_offers_the_found_save(browser, studio):
    """A remembered Save directory that holds no save: a warning and a button for the found one."""
    server, _work = studio
    _open(browser, server, 'setup')
    root = browser.execute_script('return window.paoflowApp.session.root')
    browser.execute_script(
        'localStorage.setItem(arguments[0], JSON.stringify("run"));'
        'localStorage.setItem(arguments[1], JSON.stringify({answers: '
        '{properties: ["bands"], savedir: "gaas.save"}}));',
        'paoflow-studio:workflow:%s:Si' % root,
        'paoflow-studio:answers:%s:Si:run' % root,
    )
    browser.get(server.url + '#setup')
    _wait(
        lambda: browser.execute_script(
            'return Boolean(window.paoflowApp && window.paoflowApp.setup)'
        )
    )
    browser.execute_script('window.paoflowApp.help.close()')
    warning = _wait(
        lambda: [
            w
            for w in browser.find_elements(
                By.CSS_SELECTOR, '.field[data-key="savedir"] .field-warning'
            )
            if 'gaas.save does not exist' in w.text
        ]
    )[0]
    assert browser.find_elements(
        By.CSS_SELECTOR, '.field[data-key="savedir"] button'
    )  # Browse… is there
    warning.find_element(By.XPATH, ".//button[text()='Use silicon.save']").click()
    _wait(lambda: _field_value(browser, 'savedir') == 'silicon.save')
    _wait(lambda: browser.find_elements(By.CSS_SELECTOR, '.bz-preview .main-svg'))
    _wait(
        lambda: not browser.find_element(
            By.CSS_SELECTOR, '.field[data-key="savedir"] .field-warning'
        ).is_displayed()
    )
    assert _errors(browser) == []
