import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

project = 'PAOFLOW'
author = 'PAOFLOW developers'
release = '2.9.3'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx_copybutton',
]

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

html_theme_options = {
    'github_url': 'https://github.com/marcobn/PAOFLOW',
    'show_toc_level': 2,
    'navigation_with_keys': True,
}

html_static_path = ['_static']

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
    '.md': 'markdown',
}

master_doc = 'index'
