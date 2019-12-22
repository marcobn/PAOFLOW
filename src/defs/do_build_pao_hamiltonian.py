#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import numpy.random as rd
from scipy import linalg as spl
from numpy import linalg as npl

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

### Reformat
def build_Hks ( data_controller ):

  arrays,attributes = data_controller.data_dicts()

  bnd = attributes['bnd']
  nawf = attributes['nawf']
  eta = attributes['shift']
  nspin = attributes['nspin']
  nkpnts = attributes['nkpnts']
  shift_type = attributes['shift_type']

  U = arrays['U'] 
  my_eigsmat = arrays['my_eigsmat']
  
  minimal = False
  Hksaux = np.zeros((nawf,nawf,nkpnts,nspin), dtype=complex)
  if minimal:
    Hks = np.zeros((bnd,bnd,nkpnts,nspin), dtype=complex)
  else:
    Hks = np.zeros((nawf,nawf,nkpnts,nspin), dtype=complex)

  for ik in range(nkpnts):
    for ispin in range(nspin):
      my_eigs = my_eigsmat[:,ik,ispin]

      #Building the Hamiltonian matrix
      E = np.diag(my_eigs)
      UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
      norms = 1./np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
      UU[:,:nawf] = UU[:,:nawf]*norms[:nawf]

      # Choose only the eigenvalues that are below the energy shift
      bnd_ik = 0
      for n in range(bnd):
        if my_eigs[n] <= eta:
          bnd_ik += 1
      if bnd_ik == 0:
        if rank == 0:
          print('No Eigenvalues in the selected energy range')
        comm.Abort()
      ac = UU[:,:bnd_ik]  # filtering: bnd is defined by the projectabilities
      ee1 = E[:bnd_ik,:bnd_ik]
      if shift_type == 0:
        #option 1 (PRB 2013)
        Hksaux[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(np.conj(ac).T))

      elif shift_type == 1:
        #option 2 (PRB 2016)
        aux_p=spl.inv(np.dot(np.conj(ac).T,ac))
        Hksaux[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(aux_p).dot(np.conj(ac).T))

      elif shift_type == 2:
        # no shift
        Hksaux[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T)

      else:
        if rank == 0:
          print('\'shift_type\' Not Recognized')
        comm.Abort()

      # Enforce Hermiticity (just in case...)
      Hksaux[:,:,ik,ispin] = 0.5*(Hksaux[:,:,ik,ispin] + np.conj(Hksaux[:,:,ik,ispin].T))

      if minimal:
        Sbd = np.zeros((nawf,nawf),dtype=complex)
        Sbdi = np.zeros((nawf,nawf),dtype=complex)
        S = sv = np.zeros((nawf,nawf),dtype=complex)
        e = se = np.zeros(nawf,dtype=float)

        e,S = npl.eigh(Hksaux[:,:,ik,ispin])
        S11 = S[:bnd,:bnd] + 1.0*rd.random(bnd)/10000.
        S21 = S[:bnd,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
        S12 = S21.T
        S22 = S[bnd:,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
        S22 = S22 + S21.T.dot(np.dot(spl.inv(S11),S12.T))
        Sbd[:bnd,:bnd] = 0.5*(S11+np.conj(S11.T))
        Sbd[bnd:,bnd:] = 0.5*(S22+np.conj(S22.T))
        Sbdi = spl.inv(np.dot(Sbd,np.conj(Sbd.T)))
        se,sv = npl.eigh(Sbdi)
        se = np.sqrt(se+0.0j)*np.identity(nawf,dtype=complex)
        Sbdi = sv.dot(se).dot(np.conj(sv).T)
        T = S.dot(np.conj(Sbd.T)).dot(Sbdi)
        Hbd = np.conj(T.T).dot(np.dot(Hksaux[:,:,ik,ispin],T))
        Hks[:,:,ik,ispin] = 0.5*(Hbd[:bnd,:bnd]+np.conj(Hbd[:bnd,:bnd].T))
      else:
        Hks = Hksaux
  return Hks


def do_build_pao_hamiltonian ( data_controller ):
  from PAOFLOW.defs.pao_sym import open_grid_wrapper


  #------------------------------
  # Building the PAO Hamiltonian
  #------------------------------
  arry,attr = data_controller.data_dicts()

  ashape = (attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin'])
  
  arry['Hks'] = build_Hks(data_controller)

  if attr['expand_wedge']:
    print(attr['expand_wedge'])
    open_grid_wrapper(data_controller)

  if rank != 0:
    return

  # NOTE: Take care of non-orthogonality, if needed
  # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to 
  # apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
  # non_ortho flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
  # non_ortho flag == 1 - makes H orthogonal (rotated basis) 
  #  Hks = do_non_ortho(Hks,Sks)
  #  Hks = do_ortho(Hks,Sks)
  if attr['non_ortho']:
    from .do_non_ortho import do_non_ortho

    # This is needed for consistency of the ordering of the matrix elements
    # Important in ACBN0 file writing
    arry['Sks'] = np.transpose(arry['Sks'], (1,0,2))

    arry['Hks'] = do_non_ortho(arry['Hks'],arry['Sks'])
    try:
      arry['Sks'] = np.reshape(arry['Sks'], ashape[:-1])
    except: pass

    data_controller.write_Hk_acbn0()

  arry['Hks'] = np.reshape(arry['Hks'], ashape)


def do_Hks_to_HRs ( data_controller ):
  from scipy import fftpack as FFT

  arry,attr = data_controller.data_dicts()

  #----------------------------------------------------------
  # Define the Hamiltonian and overlap matrix in real space:
  #   HRs and SRs (noinv and nosym = True in pw.x)
  #----------------------------------------------------------
  if rank == 0:
    # Original k grid to R grid
    arry['HRs'] = np.zeros_like(arry['Hks'])
    arry['HRs'] = FFT.ifftn(arry['Hks'], axes=[2,3,4])

    if attr['non_ortho']:
      arry['SRs'] = np.zeros_like(arry['Sks'])
      arry['SRs'] = FFT.ifftn(arry['Sks'], axes=[2,3,4])
      del arry['Sks']
