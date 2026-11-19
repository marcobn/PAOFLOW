"""The front-end modules parse (node --check) and insert no HTML strings."""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / '.github' / 'lint' / 'check_js.py'
MODULES = sorted((ROOT / 'src' / 'PAOFLOW' / 'gui' / 'static' / 'js').glob('*.js'))


def _checker():
    spec = importlib.util.spec_from_file_location('check_js', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('path', MODULES, ids=lambda p: p.name)
def test_module_is_clean(path):
    checker = _checker()
    assert checker.problems(path, shutil.which('node')) == []


def test_forbidden_calls_are_reported(tmp_path):
    bad = tmp_path / 'bad.js'
    bad.write_text("export const x = 1;\ndocument.body.innerHTML = '<b>x</b>';\neval('2');\n")
    found = _checker().problems(bad, None)
    assert len(found) == 2 and 'innerHTML' in found[0] and 'eval' in found[1]


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_syntax_errors_are_reported(tmp_path):
    bad = tmp_path / 'broken.js'
    bad.write_text('export function f( {\n')
    assert _checker().problems(bad, shutil.which('node'))
