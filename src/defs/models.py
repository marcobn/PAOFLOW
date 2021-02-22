#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#


def graphene( data_controller, params ):
  from .constants import ANGSTROM_AU
  from scipy.fftpack import fftshift
  from mpi4py import MPI
  import numpy as np

  arry,attr = data_controller.data_dicts()

  attr['nk1'] = 3
  attr['nk2'] = 3
  attr['nk3'] = 1

  attr['nawf'] = 2
  attr['nspin'] = 1
  attr['natoms'] = 2

  attr['alat'] = 2.46

  arry['HRs'] = np.zeros((attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin']),dtype=complex)

  # H00
  arry['HRs'][0,1,0,0,0,0] = params['t']
  arry['HRs'][1,0,0,0,0,0] = params['t']

  # H10
  arry['HRs'][1,0,1,0,0,0] = params['t']

  #H20
  arry['HRs'][:,:,2,0,0,0] = np.conj(arry['HRs'][:,:,1,0,0,0]).T

  #H01
  arry['HRs'][1,0,0,1,0,0] = params['t']

  #H02
  arry['HRs'][:,:,0,1,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

  # Lattice Vectors
  arry['a_vectors'] = np.zeros((3,3),dtype=float)
  arry['a_vectors'] = np.array([[1., 0, 0], [0.5, 3 ** .5 / 2, 0], [0, 0, 10]])
  arry['a_vectors'] = arry['a_vectors']/ANGSTROM_AU

  # Atomic coordinates
  arry['tau'] = np.zeros((2,3),dtype=float) 

  arry['tau'][0,0] = 0.50000 ;  arry['tau'][0,1] = 0.28867
  arry['tau'][1,0] = 1.00000 ;  arry['tau'][1,1] = 0.57735


  # Reciprocal Lattice
  arry['b_vectors'] = np.zeros((3,3),dtype=float)
  volume = np.dot(np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]),arry['a_vectors'][2,:])
  arry['b_vectors'][0,:] = (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/volume
  arry['b_vectors'][1,:] = (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/volume
  arry['b_vectors'][2,:] = (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/volume 

  arry['species']=["C","C"]


def cubium( data_controller, params ):
  from .constants import ANGSTROM_AU
  from scipy.fftpack import fftshift
  from mpi4py import MPI
  import numpy as np

  arry,attr = data_controller.data_dicts()

  attr['nk1'] = 3
  attr['nk2'] = 3
  attr['nk3'] = 3

  attr['nawf'] = 1
  attr['nspin'] = 1
  attr['natoms'] = 1
  attr['bnd']=1
  attr['shift']=0
  attr['omega']=1
  attr['dftSO']=False
  attr['nkpnts']=attr['nk1']*attr['nk2']*attr['nk3']
  attr['nbnds']=1

  attr['alat'] = 1.0

  arry['HRs'] = np.zeros((attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin']),dtype=complex)

  # H000
  arry['HRs'][0,0,0,0,0,0] = 0.0

  # H100
  arry['HRs'][0,0,1,0,0,0] = params['t']

  #H200
  arry['HRs'][:,:,2,0,0,0] = np.conj(arry['HRs'][:,:,1,0,0,0]).T

  #H010
  arry['HRs'][0,0,0,1,0,0] = params['t']

  #H020
  arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

  #H001
  arry['HRs'][0,0,0,0,1,0] = params['t']

  #H002
  arry['HRs'][:,:,0,0,2,0] = np.conj(arry['HRs'][:,:,0,0,1,0]).T

  # Lattice Vectors
  arry['a_vectors'] = np.zeros((3,3),dtype=float)
  arry['a_vectors'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  arry['a_vectors'] = arry['a_vectors']/ANGSTROM_AU

  # Atomic coordinates
  arry['tau'] = np.zeros((1,3),dtype=float) 

  # Reciprocal Lattice
  arry['b_vectors'] = np.zeros((3,3),dtype=float)
  volume = np.dot(np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]),arry['a_vectors'][2,:])
  arry['b_vectors'][0,:] = (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/volume
  arry['b_vectors'][1,:] = (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/volume
  arry['b_vectors'][2,:] = (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/volume 

  arry['atoms']=["Cu"]

def Kane_Mele( data_controller, params ):
  from .constants import ANGSTROM_AU
  from scipy.fftpack import fftshift
  from mpi4py import MPI
  import numpy as np

  arry,attr = data_controller.data_dicts()

  attr['nk1'] = 3
  attr['nk2'] = 3
  attr['nk3'] = 1

  attr['nawf'] = 4
  attr['nspin'] = 1
  attr['natoms'] = 2

  attr['alat'] = params['alat']

  arry['HRs'] = np.zeros((attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin']),dtype=complex)

  # H00
  arry['HRs'][0,2,0,0,0,0]=params['t']
  arry['HRs'][1,3,0,0,0,0]=params['t']
  arry['HRs'][2,0,0,0,0,0]=params['t']
  arry['HRs'][3,1,0,0,0,0]=params['t']

  # H10
  arry['HRs'][2,0,1,0,0,0]=params['t']
  arry['HRs'][3,1,1,0,0,0]=params['t']

  arry['HRs'][0,0,1,0,0,0]=-complex(0.0,params['soc_par'])
  arry['HRs'][1,1,1,0,0,0]=complex(0.0,params['soc_par'])
  arry['HRs'][2,2,1,0,0,0]=complex(0.0,params['soc_par'])
  arry['HRs'][3,3,1,0,0,0]=-complex(0.0,params['soc_par'])

  #H20
  arry['HRs'][:,:,2,0,0,0]=np.conj(arry['HRs'][:,:,1,0,0,0]).T

  #H01
  arry['HRs'][2,0,0,1,0,0]=params['t']
  arry['HRs'][3,1,0,1,0,0]=params['t']

  arry['HRs'][0,0,0,1,0,0]=complex(0.0,params['soc_par'])
  arry['HRs'][1,1,0,1,0,0]=-complex(0.0,params['soc_par'])
  arry['HRs'][2,2,0,1,0,0]=-complex(0.0,params['soc_par'])
  arry['HRs'][3,3,0,1,0,0]=complex(0.0,params['soc_par'])


  #H02
  arry['HRs'][:,:,0,2,0,0]=np.conj(arry['HRs'][:,:,0,1,0,0]).T

  #H21
  arry['HRs'][0,0,2,1,0,0]=-complex(0.0,params['soc_par'])
  arry['HRs'][1,1,2,1,0,0]=complex(0.0,params['soc_par'])
  arry['HRs'][2,2,2,1,0,0]=complex(0.0,params['soc_par'])
  arry['HRs'][3,3,2,1,0,0]=-complex(0.0,params['soc_par'])

  #H12
  arry['HRs'][:,:,1,2,0,0]=np.conj(arry['HRs'][:,:,2,1,0,0]).T

  # Lattice Vectors
  arry['a_vectors'] = np.zeros((3,3),dtype=float)
  arry['a_vectors'] = np.array([[1., 0, 0], [0.5, 3 ** .5 / 2, 0], [0, 0, 10]])
  arry['a_vectors'] = arry['a_vectors']/ANGSTROM_AU

  # Atomic coordinates
  arry['tau'] = np.zeros((2,3),dtype=float) 

  arry['tau'][0,0] = 0.50000 ;  arry['tau'][0,1] = 0.28867
  arry['tau'][1,0] = 1.00000 ;  arry['tau'][1,1] = 0.57735


  # Reciprocal Lattice
  arry['b_vectors'] = np.zeros((3,3),dtype=float)
  volume = np.dot(np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]),arry['a_vectors'][2,:])
  arry['b_vectors'][0,:] = (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/volume
  arry['b_vectors'][1,:] = (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/volume
  arry['b_vectors'][2,:] = (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/volume 

  arry['species']=["KM","KM"]


def build_TB_model ( data_controller, parameters ):

  if parameters['label'].upper() == 'GRAPHENE':
    graphene(data_controller, parameters)
  elif parameters['label'].upper() == 'CUBIUM':
    cubium(data_controller, parameters)
  elif parameters['label'].upper() == 'KANE_MELE':
    Kane_Mele(data_controller, parameters)
  else:
    print('ERROR: Label "%s" not found in builtin models.'%label)

