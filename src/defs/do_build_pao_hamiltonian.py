#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
#F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli, Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
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

### Reformat
def build_Hks ( data_controller ):
  from scipy import linalg as spl

  arrays,attributes = data_controller.data_dicts()

  bnd = attributes['bnd']
  nawf = attributes['nawf']
  eta = attributes['shift']
  nspin = attributes['nspin']
  nkpnts = attributes['nkpnts']
  shift_type = attributes['shift_type']

  U = arrays['U'] 
  my_eigsmat = arrays['my_eigsmat']

  Hksaux = np.zeros((nawf,nawf,nkpnts,nspin), dtype=complex)
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
      Hks = Hksaux
  return Hks


def do_build_pao_hamiltonian ( data_controller ):
  #------------------------------
  # Building the PAO Hamiltonian
  #------------------------------
  arry,attr = data_controller.data_dicts()

  arry['Hks'] = build_Hks(data_controller)

  ashape = (attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin'])

  if attr['expand_wedge']:
    from .pao_sym import open_grid_wrapper
    open_grid_wrapper(data_controller)

  attr['nkpnts'] = nkpnts = np.prod(ashape[2:5])

  # NOTE: Take care of non-orthogonality, if needed
  # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to 
  # apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
  # acbn0 flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
  # acbn0 flag == 1 - makes H orthogonal (rotated basis) 

  if rank == 0 and attr['expand_wedge']:
    from .do_Efermi import E_Fermi
    arry['Hks'] = np.reshape(arry['Hks'], ashape)

    # Shift the Fermi energy to zero
    tshape = (ashape[0],ashape[1],nkpnts,ashape[5])
   # Ef = E_Fermi(arry['Hks'].reshape(tshape), data_controller)
    Ef = 0
    dinds = np.diag_indices(attr['nawf'])
    arry['Hks'][dinds[0],dinds[1]] -= Ef

  if attr['acbn0']:
    import sys
    if rank == 0:
      from .do_non_ortho import do_non_ortho

      # This is needed for consistency of the ordering of the matrix elements
      # Important in ACBN0 file writing
      arry['Sks'] = np.transpose(arry['Sks'], (1,0,2))

      tshape = (ashape[0],ashape[1],nkpnts,ashape[5])
      arry['Hks'] = do_non_ortho(arry['Hks'].reshape(tshape), arry['Sks'])

      data_controller.write_Hk_acbn0()
    sys.exit(0)
    comm.Barrier()
    

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
