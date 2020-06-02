#import setuptools
from distutils.core import setup
import os

defs = os.path.join('src','defs')

setup(name='PAOFLOW',
      version='1.0.2',
      description='Electronic Structure Post-processing Tools',
      author='Marco Buongiorno Nardelli',
      author_email='mbn@unt.edu',
      platforms='Linux',
      url='http://aflowlib.org/src/paoflow/',
      packages=['PAOFLOW', 'PAOFLOW.defs'],
      package_dir={'PAOFLOW':'src'},
#      install_requires=["numpy","scipy","mpi4py","z2pack","tbmodels"]
)
