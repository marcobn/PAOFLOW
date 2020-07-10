from setuptools import setup
import os

defs = os.path.join('src','defs')

setup(name='PAOFLOW',
      version='2.0.1',
      description='Electronic Structure Post-processing Tools',
      author='Marco Buongiorno Nardelli',
      author_email='mbn@unt.edu',
      platforms='Linux',
      url='http://aflowlib.org/src/paoflow/',
      packages=['PAOFLOW', 'PAOFLOW.defs'],
      package_dir={'PAOFLOW':'src'},
      install_requires=['numpy','scipy'],
      extras_require={'weyl_search':['z2pack', 'tbmodels']},
      python_requires='>=3.6'
)
