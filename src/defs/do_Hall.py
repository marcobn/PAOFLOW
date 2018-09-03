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

def do_spin_Hall ( data_controller, do_ac ):
  import numpy as np
  from mpi4py import MPI
  from do_Hall_conductivity import do_spin_Hall_conductivity
  from do_spin_Berry_curvature import do_spin_Berry_curvature
  from constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  s_tensor = arrays['s_tensor']

  #-----------------------
  # Spin Hall calculation
  #-----------------------
  if attributes['dftSO'] == False:
    if rank == 0:
      print('Relativistic calculation with SO required')
      comm.Abort()
    comm.Barrier()

  for n in range(s_tensor.shape[0]):
    ipol = s_tensor[n][0]
    jpol = s_tensor[n][1]
    spol = s_tensor[n][2]
    #----------------------------------------------
    # Compute the spin current operator j^l_n,m(k)
    #----------------------------------------------
    jksp = do_spin_current(data_controller, spol)

    #---------------------------------
    # Compute spin Berry curvature... 
    #---------------------------------

    ene,shc,Om_k = do_spin_Berry_curvature(data_controller, jksp, ipol, jpol)

    if rank == 0:
      cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attributes['omega'])
      shc *= cgs_conv

    cart_indices = (str(LL[spol]),str(LL[ipol]),str(LL[jpol]))

    fBerry = 'Spin_Berry_%s_%s%s.dat'%cart_indices
    nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']
    Om_kps = (np.empty((nk1,nk2,nk3,2), dtype=float) if rank==0 else None)
    if rank == 0:
      Om_kps[:,:,:,0] = Om_kps[:,:,:,1] = Om_k[:,:,:]
    data_controller.write_bxsf(fBerry, Om_kps, 2)

    Om_k = Om_kps = None

    fshc = 'shcEf_%s_%s%s.dat'%cart_indices
    data_controller.write_file_row_col(fshc, ene, shc)

    ene = shc = None

    if do_ac:
      ene,sigxy = do_spin_Hall_conductivity(data_controller, jksp, ipol, jpol)
      if rank == 0:
        sigxy *= cgs_conv

      fsigI = 'SCDi_%s_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigI, ene, np.imag(sigxy))

      fsigR = 'SCDr_%s_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigR, ene, np.real(sigxy))


def do_anomalous_Hall ( data_controller, do_ac ):
  import numpy as np
  from mpi4py import MPI
  from do_Hall_conductivity import do_Berry_conductivity
  from do_spin_Berry_curvature import do_spin_Berry_curvature
  from constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  a_tensor = arrays['a_tensor']

  #----------------------------
  # Anomalous Hall calculation
  #----------------------------
  if attributes['dftSO'] == False:
    if rank == 0:
      print('Relativistic calculation with SO required')
      comm.Abort()
    comm.Barrier()

  for n in range(a_tensor.shape[0]):
    ipol = a_tensor[n][0]
    jpol = a_tensor[n][1]

    ene,ahc,Om_k = do_spin_Berry_curvature(data_controller, arrays['pksp'], ipol, jpol)

    if rank == 0:
      cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attributes['omega'])

    cart_indices = (str(LL[ipol]),str(LL[jpol]))

    fBerry = 'Berry_%s%s.dat'%cart_indices
    nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']
    Om_kps = (np.empty((nk1,nk2,nk3,2), dtype=float) if rank==0 else None)
    if rank == 0:
      Om_kps[:,:,:,0] = Om_kps[:,:,:,1] = Om_k[:,:,:]
    data_controller.write_bxsf(fBerry, Om_kps, 2)

    Om_k = Om_kps = None

    if rank == 0:
      ahc *= cgs_conv
    fahc = 'ahcEf_%s%s.dat'%cart_indices
    data_controller.write_file_row_col(fahc, ene, ahc)

    ene = ahc = None

    if do_ac:
      ene,sigxy = do_Berry_conductivity(data_controller, arrays['pksp'], ipol, jpol)
      if rank == 0:
        sigxy *= cgs_conv

      fsigI = 'MCDi_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigI, ene, np.imag(sigxy))

      fsigR = 'MCDr_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigR, ene, np.real(sigxy))


def do_spin_current ( data_controller, spol ):
  import numpy as np

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  Sj = arrays['Sj']
  bnd = attributes['bnd']
  snktot,_,nawf,nawf,nspin = arrays['dHksp'].shape

  jdHksp = np.empty_like(arrays['dHksp'])

  for l in range(3):
    for ispin in range(nspin):
      for ik in range(snktot):
        jdHksp[ik,l,:,:,ispin] = 0.5*(np.dot(Sj[l],arrays['dHksp'][ik,l,:,:,ispin])+np.dot(arrays['dHksp'][ik,l,:,:,ispin],Sj[l]))

  jksp = np.zeros((snktot,3,bnd,bnd,nspin), dtype=complex)

  for l in range(3):
    for ispin in range(nspin):
      for ik in range(snktot):
        jksp[ik,l,:,:,ispin] = np.conj(arrays['v_k'][ik,:,:,ispin].T).dot(jdHksp[ik,l,:,:,ispin]).dot(arrays['v_k'][ik,:,:,ispin])[:bnd,:bnd]

  jdHksp = None

  return jksp
