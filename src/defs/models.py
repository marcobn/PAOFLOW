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

def Slater_Koster( data_controller, params ):
  # generalized Slater-Koster TB model in the two-center approximation (1st nearest neighbors only)
  from .constants import ANGSTROM_AU
  from scipy.fftpack import fftshift
  from mpi4py import MPI
  import numpy as np

  arry,attr = data_controller.data_dicts()
  # Lattice Vectors
  arry['a_vectors'] = np.array(params['model']['a_vectors'])
  attr['alat'] = 1.0
  
  # Atomic coordinates
  natoms = len(params['model']['atoms'])
  tau = np.zeros((natoms,3),dtype=float)
  for ia in range(natoms):
    tau[ia] = np.array(params['model']['atoms'][str(ia)]['tau'])  
  atoms = []
  shells = []
  for ia in range(natoms):
    atoms.append(params['model']['atoms'][str(ia)]['name'])
    shells.append(params['model']['atoms'][str(ia)])
  arry['tau'] = tau
  arry['atoms'] = atoms
  attr['natoms'] = natoms
  arry['shells'] = shells
  attr['nspin'] = 1 # only unpolarized case for now
  attr['dftSO'] = False # no spin-orbit
  
  # Reciprocal Lattice
  arry['b_vectors'] = np.zeros((3,3),dtype=float)
  volume = np.dot(np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]),arry['a_vectors'][2,:])
  attr['omega'] = volume
  arry['b_vectors'][0,:] = (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/volume
  arry['b_vectors'][1,:] = (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/volume
  arry['b_vectors'][2,:] = (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/volume 

  # dimensions of the supercell for two-center approximation
  nk1 = nk2 = nk3 = 3
  nkpnts = nk1*nk2*nk3
  attr['nk1'] = nk1
  attr['nk2'] = nk2
  attr['nk3'] = nk3
  attr['nkpnts'] = nkpnts
  
  # on site and hopping parameters
  norbitals = np.zeros(natoms,dtype=int)
  for ia in range(natoms):
    norbitals[ia] = len(params['model']['atoms'][str(ia)]['orbitals']) 
  
  nawf = 0
  for ia in range(natoms):
    nawf += norbitals[ia]
  attr['nawf'] = nawf
  attr['bnd'] = nawf
  attr['nbnds'] = nawf
  attr['shift'] = 0

  # generate all the orbitals positions in the supercell
  sctau = np.zeros((natoms,nk1,nk2,nk3,3),dtype=float)
  for i in range(-1,2):
    for j in range(-1,2):
      for k in range(-1,2):
        for ia in range(natoms):
          sctau[ia,i,k,j,:] = tau[ia] + i*arry['a_vectors'][0] + j*arry['a_vectors'][1] + k*arry['a_vectors'][2]
  sctau = np.reshape(sctau,(natoms*27,3),order='C')
  # make the list of neighbors and find cutoff for two-center approximation
  distance = lambda x,y : np.sqrt(np.sum((x-y)**2))
  cosines = lambda x,y : (y-x)/np.sqrt(np.sum((x-y)**2))
  dist = []
  for ia in range(natoms):
      for n in range(natoms*27):
          dist.append(distance(tau[ia],sctau[n]))
  cutoff = np.sort(np.unique(dist))[1]+(np.sort(np.unique(dist))[2]-np.sort(np.unique(dist))[1])/2
  sctau = np.reshape(sctau,(natoms,nk1,nk2,nk3,3),order='C')
  
  # debug
  arry['sctau'] = sctau
  attr['cutoff'] = cutoff
  arry['norbitals'] = norbitals
  
  HRs = np.zeros((nawf,nawf,nk1,nk2,nk3,1),dtype=complex)

  # on-site matrix elements
  for ia in range(natoms):
    for no in range(norbitals[ia]):
      HRs [ia*norbitals[ia]+no,ia*norbitals[ia]+no,0,0,0,0] = \
      params['model']['atoms'][str(ia)][params['model']['atoms'][str(ia)]['orbitals'][no]]

  # hopping matrix elements
  for i in range(-1,2):
    for j in range(-1,2):
      for k in range(-1,2):
        for ia in range(natoms):
          for ib in range(natoms):
            if distance(tau[ia],sctau[ib,i,j,k,:]) > 0 and distance(tau[ia],sctau[ib,i,j,k,:]) < cutoff:
              lx = cosines(tau[ia],sctau[ib,i,j,k,:])[0]
              ly = cosines(tau[ia],sctau[ib,i,j,k,:])[1]
              lz = cosines(tau[ia],sctau[ib,i,j,k,:])[2]
              
              for noa in range(norbitals[ia]):
                for nob in range(norbitals[ib]):
#                  print(ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k)
                  if noa == 0 and nob == 0:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = params['model']['hoppings']['sss']
                  elif noa == 0 and nob == 1:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = lx*params['model']['hoppings']['sps']
                  elif noa == 0 and nob == 2:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = ly*params['model']['hoppings']['sps']
                  elif noa == 0 and nob == 3:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = lz*params['model']['hoppings']['sps']
                  elif noa == 1 and nob == 1:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lx**2*params['model']['hoppings']['pps']+(1.0-lx**2)*params['model']['hoppings']['ppp']
                  elif noa == 2 and nob == 2:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    ly**2*params['model']['hoppings']['pps'] + (1.0-ly**2)*params['model']['hoppings']['ppp']
                  elif noa == 3 and nob == 3:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lz**2*params['model']['hoppings']['pps'] + (1.0-lz**2)*params['model']['hoppings']['ppp']
                  elif noa == 1 and nob == 2:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lx*ly*(params['model']['hoppings']['pps'] - params['model']['hoppings']['ppp'])
                  elif noa == 2 and nob == 3:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    ly*lz*(params['model']['hoppings']['pps'] - params['model']['hoppings']['ppp'])
                  elif noa == 1 and nob == 3:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lx*lz*(params['model']['hoppings']['pps'] - params['model']['hoppings']['ppp'])
                  elif (noa == 1 and nob == 0): 
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = -lx*params['model']['hoppings']['sps']
                  elif (noa == 2 and nob == 0):
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = -ly*params['model']['hoppings']['sps']
                  elif (noa == 3 and nob == 0):
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = -lz*params['model']['hoppings']['sps']
                  elif noa == 1 and nob == 1:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lx**2*params['model']['hoppings']['pps']+(1.0-lx**2)*params['model']['hoppings']['ppp']
                  elif noa == 2 and nob == 2:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    ly**2*params['model']['hoppings']['pps'] + (1.0-ly**2)*params['model']['hoppings']['ppp']
                  elif noa == 3 and nob == 3:
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lz**2*params['model']['hoppings']['pps'] + (1.0-lz**2)*params['model']['hoppings']['ppp']
                  elif (noa == 2 and nob == 1):
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lx*ly*(params['model']['hoppings']['pps'] - params['model']['hoppings']['ppp'])
                  elif (noa == 3 and nob == 2):
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    ly*lz*(params['model']['hoppings']['pps'] - params['model']['hoppings']['ppp'])
                  elif (noa == 3 and nob == 1):
                    HRs[ia*norbitals[ia]+noa,ib*norbitals[ib]+nob,i,j,k,0] = \
                    lx*lz*(params['model']['hoppings']['pps'] - params['model']['hoppings']['ppp'])
                  
    arry['HRs'] = HRs

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

  attr['alat'] = 2.46 * ANGSTROM_AU

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
  arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

  # Lattice Vectors
  arry['a_vectors'] = np.zeros((3,3),dtype=float)
  arry['a_vectors'] = np.array([[1., 0, 0], [0.5, 3 ** .5 / 2, 0], [0, 0, 10]])
  arry['a_vectors'] = arry['a_vectors']

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

  arry['atoms']=['C','C']

def graphene2( data_controller, params ):
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

  arry['naw'] = np.array([1,1])

  attr['alat'] = 2.46 * ANGSTROM_AU

  arry['HRs'] = np.zeros((attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin']),dtype=complex)

  # H00
  arry['HRs'][0,0,0,0,0,0] = params['delta']/2
  arry['HRs'][1,1,0,0,0,0] = -params['delta']/2

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
  arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

  # Lattice Vectors
  arry['a_vectors'] = np.zeros((3,3),dtype=float)
  arry['a_vectors'] = np.array([[1., 0, 0], [0.5, 3 ** .5 / 2, 0], [0, 0, 10]])
  arry['a_vectors'] = arry['a_vectors']

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

  arry['atoms']=['C','C']


def cubium( data_controller, params ):
  from .constants import ANGSTROM_AU
  from scipy.fftpack import fftshift
  from mpi4py import MPI
  import numpy as np

  arry,attr = data_controller.data_dicts()

  attr['nk1'] = 3
  attr['nk2'] = 3
  attr['nk3'] = 3
  attr['Efermi'] = 6*params['t']
  attr['nawf'] = 1
  attr['nspin'] = 1
  attr['natoms'] = 1
  attr['bnd']=1
  attr['shift']=0
  attr['dftSO']=False
  attr['nkpnts']=attr['nk1']*attr['nk2']*attr['nk3']
  attr['nbnds']=1
  attr['nelec']=2

  attr['alat'] = 1.0*ANGSTROM_AU
  attr['omega'] = attr['alat']**3

  arry['HRs'] = np.zeros((attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin']),dtype=complex)

  # H000
  arry['HRs'][0,0,0,0,0,0] = 0.0 - attr['Efermi']

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

  # Atomic coordinates
  arry['tau'] = np.zeros((1,3),dtype=float) 

  # Reciprocal Lattice
  arry['b_vectors'] = np.zeros((3,3),dtype=float)
  volume = np.dot(np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]),arry['a_vectors'][2,:])
  arry['b_vectors'][0,:] = (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/volume
  arry['b_vectors'][1,:] = (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/volume
  arry['b_vectors'][2,:] = (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/volume 

  arry['atoms']=["Cu"]
  
def cubium2( data_controller, params ):
  from .constants import ANGSTROM_AU
  from scipy.fftpack import fftshift
  from mpi4py import MPI
  import numpy as np

  arry,attr = data_controller.data_dicts()

  attr['nk1'] = 3
  attr['nk2'] = 3
  attr['nk3'] = 3

  attr['nawf'] = 2
  attr['nspin'] = 1
  attr['natoms'] = 1
  attr['bnd']=2
  attr['shift']=0
  attr['dftSO']=False
  attr['nkpnts']=attr['nk1']*attr['nk2']*attr['nk3']
  attr['nbnds']=2
  attr['nelec']=2
  attr['alat'] = 1.0*ANGSTROM_AU
  attr['omega'] = attr['alat']**3

  arry['HRs'] = np.zeros((attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin']),dtype=complex)

  # H000
  arry['HRs'][0,0,0,0,0,0] = -params['Eg']/2 -6.0*params['t']
  arry['HRs'][1,1,0,0,0,0] = params['Eg']/2 +6.0*params['t']

  # H100
  arry['HRs'][0,0,1,0,0,0] = params['t']
  arry['HRs'][1,1,1,0,0,0] = -params['t']

  #H200
  arry['HRs'][:,:,2,0,0,0] = np.conj(arry['HRs'][:,:,1,0,0,0]).T

  #H010
  arry['HRs'][0,0,0,1,0,0] = params['t']
  arry['HRs'][1,1,0,1,0,0] = -params['t']

  #H020
  arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

  #H001
  arry['HRs'][0,0,0,0,1,0] = params['t']
  arry['HRs'][1,1,0,0,1,0] = -params['t']

  #H002
  arry['HRs'][:,:,0,0,2,0] = np.conj(arry['HRs'][:,:,0,0,1,0]).T

  # Lattice Vectors
  arry['a_vectors'] = np.zeros((3,3),dtype=float)
  arry['a_vectors'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

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
  elif parameters['label'].upper() == 'GRAPHENE2':
    graphene2(data_controller, parameters)
  elif parameters['label'].upper() == 'CUBIUM':
    cubium(data_controller, parameters)
  elif parameters['label'].upper() == 'CUBIUM2':
    cubium2(data_controller, parameters)
  elif parameters['label'].upper() == 'KANE_MELE':
    Kane_Mele(data_controller, parameters)
  elif parameters['label'].upper() == 'SLATER_KOSTER':
    Slater_Koster(data_controller, parameters)
  elif parameters['label'].upper() == 'MAGNETIC_BILAYER':
    from .magnetic_bilayer import magnetic_bilayer
    magnetic_bilayer(data_controller, parameters)
  else:
    print('ERROR: Label "%s" not found in builtin models.'%label)

