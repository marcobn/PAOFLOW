from setuptools import setup
import os

defs = os.path.join('src','defs')

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='PAOFLOW',
      version='2.1.1',
      description='Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)',
      author='Marco Buongiorno Nardelli',
      author_email='mbn@unt.edu',
      platforms='Linux',
      url='https://github.com/marcobn/PAOFLOW',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['PAOFLOW', 'PAOFLOW.defs'],
      package_dir={'PAOFLOW':'src'},
      install_requires=['numpy','scipy'],
      extras_require={'weyl_search':['z2pack', 'tbmodels']},
      python_requires='>=3.6'
)
