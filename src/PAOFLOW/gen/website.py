"""Links from PAOFLOW (Studio, ``paoflow-gen``) to the pages of its website on Read the Docs.

A page is named by its path relative to ``docs/`` (``learn/tutorials/tutorial01.ipynb``, with an
optional ``#anchor``), as in the ``docs`` of a ``case.json``. Links go to the docs version of the
installed PAOFLOW: ``latest`` (the ``develop`` branch) for a source checkout, the release tag
(``v4.0.0``) otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import __version__
from .cases import _repository_folder

#: Root of the website; the docs version and the page follow.
DOCS_BASE = 'https://paoflow.readthedocs.io/en/'


def docs_version() -> str:
    """Read the Docs version of the installed PAOFLOW.

    Returns
    -------
    str
        ``'latest'`` for a source checkout or an editable install (it has the ``docs`` folder),
        ``'v<version>'`` (the release tag) for an installed wheel.
    """
    return 'latest' if _repository_folder('docs') else 'v' + __version__


def docs_url(page: str) -> str:
    """URL of a page of the website, in the docs version of the installed PAOFLOW.

    Parameters
    ----------
    page : str
        Source of the page relative to ``docs/``, with an optional anchor
        (``'learn/theory/berry-topology.md#anomalous-hall-conductivity'``).

    Returns
    -------
    str
        ``https://paoflow.readthedocs.io/en/<version>/learn/theory/berry-topology.html#...``.
    """
    path, _, anchor = page.partition('#')
    url = DOCS_BASE + docs_version() + '/' + str(Path(path).with_suffix('.html'))
    return url + '#' + anchor if anchor else url


def page_title(page: str) -> str:
    """Title of a page: its first ``# `` heading, read from the ``docs`` folder of a checkout.

    Parameters
    ----------
    page : str
        Source of the page relative to ``docs/`` (``.md`` or ``.ipynb``).

    Returns
    -------
    str
        The heading, or the file name without suffix when the page cannot be read.
    """
    docs = _repository_folder('docs')
    source = docs / page if docs else None
    if source is None or not source.is_file():
        return Path(page).stem
    text = source.read_text(encoding='utf-8')
    if source.suffix == '.ipynb':
        cells = json.loads(text)['cells']
        text = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'markdown')
    for line in text.splitlines():
        if line.startswith('# '):
            return line[2:].strip()
    return Path(page).stem
