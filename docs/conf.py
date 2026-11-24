import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def write_examples_page() -> None:
    """Write ``learn/examples.md``, the catalogue of ``examples/cases``, from their ``case.json``.

    The case files are read as JSON, not through ``PAOFLOW.gen.cases``: importing PAOFLOW here would
    pull in mpi4py. The page is rewritten only when it changes, so sphinx-autobuild does not loop.
    """
    docs = Path(__file__).resolve().parent
    cases = docs.parent / 'examples' / 'cases'
    # the GitHub links follow the branch or tag Read the Docs builds
    ref = os.environ.get('READTHEDOCS_GIT_IDENTIFIER', 'develop')
    tree = 'https://github.com/marcobn/PAOFLOW/tree/%s/examples/cases/' % ref
    lines = [
        '# Examples',
        '',
        'Every case of `examples/cases`: one calculation on one system, with its inputs, its run '
        'steps and a README that says how to run it. `paoflow-gen --example <case>` or PAOFLOW '
        'Studio (*Examples…*) copies a case into a folder of your own; `--with-data` (Studio: '
        '*Use the precomputed data of the case*) adds its published DFT output, so no Quantum '
        'ESPRESSO run is needed. The tutorials and how-tos built on a case are listed under it.',
        '',
    ]
    for folder in sorted(cases.iterdir()):
        if not (folder / 'case.json').is_file():
            continue
        case = json.loads((folder / 'case.json').read_text(encoding='utf-8'))
        facts = ['workflow `%s`' % case['workflow']]
        if case.get('tags'):
            facts.append('tags: ' + ', '.join(case['tags']))
        if case.get('data'):
            facts.append('precomputed data')
        facts.append('[folder on GitHub](%s%s)' % (tree, folder.name))
        lines += ['## `%s`' % folder.name, '', '**%s**' % case['title'], '']
        lines += [case['summary'], '', ' · '.join(facts), '']
        if case.get('docs'):
            pages = ['{doc}`/%s`' % Path(page).with_suffix('') for page in case['docs']]
            lines += ['Explained in: ' + ', '.join(pages), '']
    text = '\n'.join(lines)
    page = docs / 'learn' / 'examples.md'
    if not page.is_file() or page.read_text(encoding='utf-8') != text:
        page.write_text(text, encoding='utf-8')


write_examples_page()

project = 'PAOFLOW'
author = 'PAOFLOW developers'
release = '4.0.0'

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx_copybutton',
    'autoapi.extension',
]

# Fast live-preview mode (see serve_docs.sh): viewcode regenerates every
# _modules/*.html page on every build, which dominates incremental rebuilds.
DOCS_FAST = os.environ.get('PAOFLOW_DOCS_FAST', '') not in ('', '0')
if DOCS_FAST:
    extensions.remove('sphinx.ext.viewcode')

nb_execution_mode = 'off'

# GitHub-style anchors (#section-title) for the headings of the Markdown pages
myst_heading_anchors = 4
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
    'colon_fence',
    'deflist',
]

autosummary_generate = True
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_mock_imports = [
    'mpi4py',
    'vtk',
    'numba',
    'z2pack',
    'tbmodels',
    'petsc4py',
    'slepc4py',
    'skimage',
    'shapely',
    'joblib',
    'mendeleev',
]

autosummary_generate = True
autosummary_imported_members = False

autodoc_typehints = 'none'
autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'exclude-members': '__weakref__',
}

autoapi_keep_files = True

autoapi_type = 'python'

autoapi_dirs = [
    '../src/PAOFLOW',
]
autoapi_ignore = [
    '*defs*',
    '*PAOFLOW/qo_analysis*',  # back-compat shim of PAOFLOW.agentic.qo_analysis
]
autoapi_root = 'api/autoapi'
autoapi_add_toctree_entry = False

autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

autoapi_member_order = 'bysource'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

nitpicky = False

suppress_warnings = [
    'ref.python',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

html_theme = 'pydata_sphinx_theme'

html_logo = '_static/images/logos/logo_documentation.png'

html_favicon = '_static/images/logos/favicon.ico'

html_show_sourcelink = False

html_theme_options = {
    'github_url': 'https://github.com/marcobn/PAOFLOW',
    'show_nav_level': 0,
    # the sidebar (_templates/sidebar-tree.html) expands only the branch of the current page: drawn
    # whole, it put the ~450 API pages on every page and the build time grew as pages x pages
    # (166 s -> 49 s locally, Nov 2026; Read the Docs had reached its 30-minute limit)
    'collapse_navigation': True,
    'navigation_depth': 2,
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'navbar_start': ['navbar-logo'],
    'navbar_center': [],
    'navbar_end': ['search-field', 'theme-switcher', 'navbar-icon-links'],
    'navbar_persistent': [],
}

html_sidebars = {
    'index': [],
    '**': ['sidebar-tree'],
}

html_static_path = ['_static']
html_css_files = [
    'css/sidebar-tree.css',
    'css/landing.css',
    'css/matrix-theme.css',
    'css/team.css',
]
html_js_files = [
    'js/sidebar-tree.js',
    'js/landing.js',
    'js/theme-toggle.js',
    'js/citations.js',
]
html_title = 'PAOFLOW'

templates_path = ['_templates']

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'api/generated/PAOFLOW.defs.rst',
    'api/generated/PAOFLOW.defs.*.rst',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

master_doc = 'index'
