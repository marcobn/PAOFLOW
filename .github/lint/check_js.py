#!/usr/bin/env python3
"""Checks of the PAOFLOW Studio front end (``src/PAOFLOW/gui/static/js``): syntax and unsafe calls.

* Syntax: every module is parsed by ``node --check`` (as an ES module).  Without ``node`` on the
  PATH the syntax check is skipped with a message (the pattern checks still run).
* Safety: the GUI inserts text only (``textContent`` and text nodes), so ``innerHTML``,
  ``outerHTML``, ``insertAdjacentHTML``, ``document.write``, ``eval`` and ``new Function`` are refused.

Usage: ``python .github/lint/check_js.py [files...]`` (default: every module).  Used by the
pre-commit hook ``js-check`` and by ``tests/unit/gui/test_static_js.py``.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATIC = ROOT / 'src' / 'PAOFLOW' / 'gui' / 'static' / 'js'
FORBIDDEN = {
    r'\.innerHTML\b': 'innerHTML (insert text nodes instead)',
    r'\.outerHTML\b': 'outerHTML',
    r'\binsertAdjacentHTML\b': 'insertAdjacentHTML',
    r'\bdocument\.write\b': 'document.write',
    r'\beval\s*\(': 'eval',
    r'\bnew\s+Function\b': 'new Function',
}


def problems(path: pathlib.Path, node: str | None) -> list[str]:
    """Problems of one module (empty when clean).

    Parameters
    ----------
    path : pathlib.Path
        The JavaScript file.
    node : str or None
        The ``node`` executable (``None`` skips the syntax check).

    Returns
    -------
    list of str
    """
    found = []
    text = path.read_text(encoding='utf-8')
    for number, line in enumerate(text.splitlines(), 1):
        for pattern, name in FORBIDDEN.items():
            if re.search(pattern, line) and not line.strip().startswith('//'):
                found.append('%s:%d: %s' % (path, number, name))
    if node is not None:
        with tempfile.TemporaryDirectory() as folder:
            module = pathlib.Path(folder) / (path.stem + '.mjs')
            module.write_text(text, encoding='utf-8')
            result = subprocess.run([node, '--check', str(module)], capture_output=True, text=True)
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip().replace(str(module), str(path))
                found.append(message)
    return found


def main(argv: list[str]) -> int:
    files = [pathlib.Path(a) for a in argv] or sorted(STATIC.glob('*.js'))
    node = shutil.which('node')
    if node is None:
        print('check_js: node not found, syntax check skipped')
    report = [p for path in files for p in problems(path, node)]
    for line in report:
        print(line)
    return 1 if report else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
