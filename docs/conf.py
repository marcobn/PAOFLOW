import sys
from pathlib import Path

from sphinx.ext.apidoc import main

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

project = 'PAOFLOW'
author = 'PAOFLOW developers'
release = '2.9.3'


main(
    [
        '-f',
        '-e',
        '-M',
        '-o',
        'docs/api/generated',
        '../src/PAOFLOW',
    ]
)

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
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Mock heavy or optional dependencies unavailable in the docs build environment.
# mpi4py is a core dep but requires a system MPI library that may be absent on RTD.
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

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
