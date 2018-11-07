from distutils.core import setup
import os

defs = os.path.join('src','defs')

setup(name='PAOFOO',
      version='1.0',
      summary='Electronic Structure Post-processing Tools',
      author='Marco Buongiorno Nardelli',
      author_email='mbn@unt.edu',
      platforms='Linux',
      url='ERMES',
      packages=['PAOFLOW', 'PAOFLOW.defs'],
      package_dir={'PAOFLOW':'src'})
