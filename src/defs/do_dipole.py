#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from .do_atwfc_proj import *

# Function to calculate dipole matrix element from coefficients of wavefunction, 
# following the routine of epsilon.x
def calc_dipole(arry,attr, ik, ispin, b_vector):
  from scipy.io import FortranFile
  import os
  if attr['nspin'] == 1 or attr['nspin'] == 4:
    wfcfile = 'wfc{0}.dat'.format(ik+1)
  elif attr['nspin'] == 2 and ispin == 0:
    wfcfile = 'wfcdw{0}.dat'.format(ik+1)
  elif attr['nspin'] == 2 and ispin == 1:
    wfcfile = 'wfcup{0}.dat'.format(ik+1)
  else:
    print('no wfc file found')

  with FortranFile(os.path.join(attr['fpath'], wfcfile), 'r') as f:
    record = f.read_ints(np.int32)
    assert len(record) == 11, 'something wrong reading fortran binary file'

    ik_ = record[0]
    assert ik+1 == ik_, 'wrong k-point in wfc file???'

    # xk = np.frombuffer(record[1:7], np.float64)
    # ispin = record[7]
    # gamma_only = (record[8] != 0)
    scalef = np.frombuffer(record[9:], np.float64)[0]

    ngw, igwx, npol, nbnds = f.read_ints(np.int32)
    f.read_reals(np.float64).reshape(3,3,order='F')
    mill = f.read_ints(np.int32).reshape(3,igwx,order='F')
    mill = b_vector.T@mill + np.full((igwx,3),arry['kpnts'][ik]).T

    wfc = []
    for i in range(nbnds):
      wfc.append(f.read_reals(np.complex128))

  
  dipole_aux = np.zeros((3,nbnds,nbnds),dtype=np.complex128)
  for iband2 in range(nbnds):
    for iband1 in range(nbnds):
      if attr['dftSO']:
        dipole_aux[:,iband1,iband2] = (wfc[iband2][:igwx]*mill)@np.conjugate(wfc[iband1][:igwx]) 
        + (wfc[iband2][igwx:]*mill)@np.conjugate(wfc[iband1][igwx:])
      else:
        dipole_aux[:,iband1,iband2] = (wfc[iband2]*mill)@np.conjugate(wfc[iband1]) 
  return dipole_aux

# Function to calculate dipole matrix element from the eigenvector of the PAO Hamiltonian
# expanded in the real space of the atomic basis functions
def calc_dipole_internal(data_controller, ik, ispin):
  from .constants import RYTOEV

  arry, attr = data_controller.data_dicts()
  basis = arry['basis']
  gkspace = calc_gkspace(data_controller,ik,gamma_only=False)
  xk, igwx, mill, bg, _ = [gkspace[s] for s in ('xk', 'igwx', 'mill', 'bg', 'gamma_only')]
  atwfcgk = calc_atwfc_k(basis,gkspace)
  oatwfcgk = ortho_atwfc_k(atwfcgk) # these are the atomic orbitals on the G vector grid

  # build the full wavefunction with the coefficients v_k
  bnd = attr['bnd']
  wfc = []
  # for nb in range(attr['bnd']):
  for nb in range(bnd):
    wfc.append(np.tensordot(arry['v_k'][ik,:,nb,ispin],oatwfcgk,axes=(0,0)))

  # build k+G
  mill = arry['b_vectors'].T@mill + np.full((igwx,3),arry['kgrid'][:,ik]).T
  # mill = bg.T@mill + np.full((igwx,3),xk).T 

  nbnds = attr['nawf']
  dipole_aux = np.zeros((3,nbnds,nbnds),dtype=np.complex128)
  for iband2 in range(bnd):
    for iband1 in range(bnd):
      if attr['dftSO']:
        # check indexing with nbnds and bnd!!!!!
        dipole_aux[:,iband1,iband2] = (wfc[iband2][:igwx]*mill)@np.conjugate(wfc[iband1][:igwx]) 
        + (wfc[iband2][igwx:]*mill)@np.conjugate(wfc[iband1][igwx:])
      else:
        dipole_aux[:,iband1,iband2] = (wfc[iband2]*mill)@np.conjugate(wfc[iband1]) 
  return dipole_aux
