import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

project = 'PAOFLOW'
author = 'PAOFLOW developers'
release = '2.9.3'

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

nb_execution_mode = 'off'

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

html_logo = '_static/images/logo_documentation.svg'

html_show_sourcelink = False

html_theme_options = {
    'github_url': 'https://github.com/marcobn/PAOFLOW',
    'show_nav_level': 0,
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
html_css_files = ['sidebar-tree.css', 'landing.css', 'matrix-theme.css']
html_js_files = ['sidebar-tree.js', 'landing.js', 'theme-toggle.js']
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
