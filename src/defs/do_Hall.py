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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def do_spin_Hall ( data_controller, do_ac ):
  from .perturb_split import perturb_split
  from .do_spin_Berry_curvature import do_spin_Berry_curvature
  from .constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL

  arry,attr = data_controller.data_dicts()

  s_tensor = arry['s_tensor']

  #-----------------------
  # Spin Hall calculation
  #-----------------------
  if attr['dftSO'] == False:
    if rank == 0:
      print('Relativistic calculation with SO required')
      comm.Abort()
    comm.Barrier()

  if rank == 0 and attr['verbose']:
    print('Writing bxsf files for Spin Berry Curvature')

  for n in range(s_tensor.shape[0]):
    ipol = s_tensor[n][0]
    jpol = s_tensor[n][1]
    spol = s_tensor[n][2]
    #----------------------------------------------
    # Compute the spin current operator j^l_n,m(k)
    #----------------------------------------------
    jdHksp = do_spin_current(data_controller, spol, ipol)

    jksp_is = np.empty_like(jdHksp)
    pksp_j = np.empty_like(jdHksp)

    for ik in range(jdHksp.shape[0]):
      for ispin in range(jdHksp.shape[3]):
        jksp_is[ik,:,:,ispin],pksp_j[ik,:,:,ispin] = perturb_split(jdHksp[ik,:,:,ispin], arry['dHksp'][ik,jpol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])
    jdHksp = None

    #---------------------------------
    # Compute spin Berry curvature... 
    #---------------------------------
    ene,shc,Om_k = do_spin_Berry_curvature(data_controller, jksp_is, pksp_j)

    if rank == 0:
      cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega'])
      shc *= cgs_conv

    cart_indices = (str(LL[spol]),str(LL[ipol]),str(LL[jpol]))

    fBerry = 'Spin_Berry_%s_%s%s.bxsf'%cart_indices
    nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']
    Om_kps = (np.empty((nk1,nk2,nk3,2), dtype=float) if rank==0 else None)
    if rank == 0:
      Om_kps[:,:,:,0] = Om_kps[:,:,:,1] = Om_k[:,:,:]
    data_controller.write_bxsf(fBerry, Om_kps, 2)

    Om_k = Om_kps = None

    fshc = 'shcEf_%s_%s%s.dat'%cart_indices
    data_controller.write_file_row_col(fshc, ene, shc)

    ene = shc = None

    if do_ac:

      jdHksp = do_spin_current(data_controller, spol, jpol)

      jksp_js = np.empty_like(jdHksp)
      pksp_i = np.empty_like(jdHksp)

      for ik in range(jdHksp.shape[0]):
        for ispin in range(jdHksp.shape[3]):
          jksp_js[ik,:,:,ispin],pksp_i[ik,:,:,ispin] = perturb_split(jdHksp[ik,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])
      jdHksp = None

      ene,sigxy = do_spin_Hall_conductivity(data_controller, jksp_js, pksp_i, ipol, jpol)
      if rank == 0:
        sigxy *= cgs_conv

      fsigI = 'SCDi_%s_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigI, ene, np.imag(sigxy))

      fsigR = 'SCDr_%s_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigR, ene, np.real(sigxy))


def do_anomalous_Hall ( data_controller, do_ac ):
  from .perturb_split import perturb_split
  from .do_spin_Berry_curvature import do_spin_Berry_curvature
  from .constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL

  arry,attr = data_controller.data_dicts()

  a_tensor = arry['a_tensor']

  #----------------------------
  # Anomalous Hall calculation
  #----------------------------
  if attr['dftSO'] == False:
    if rank == 0:
      print('Relativistic calculation with SO required')
      comm.Abort()
    comm.Barrier()

  if rank == 0 and attr['verbose']:
    print('Writing bxsf files for Berry Curvature')

  for n in range(a_tensor.shape[0]):
    ipol = a_tensor[n][0]
    jpol = a_tensor[n][1]

    dks = arry['dHksp'].shape

    pksp_i = np.zeros((dks[0],dks[2],dks[3],dks[4]),order="C",dtype=complex)
    pksp_j = np.zeros_like(pksp_i)

    for ik in range(dks[0]):
      for ispin in range(dks[4]):
        pksp_i[ik,:,:,ispin],pksp_j[ik,:,:,ispin] = perturb_split(arry['dHksp'][ik,ipol,:,:,ispin], arry['dHksp'][ik,jpol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])

    ene,ahc,Om_k = do_spin_Berry_curvature(data_controller, pksp_i, pksp_j)

    if rank == 0:
      cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega'])

    cart_indices = (str(LL[ipol]),str(LL[jpol]))

    fBerry = 'Berry_%s%s.dat'%cart_indices
    nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']
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
      ene,sigxy = do_Berry_conductivity(data_controller, pksp_i, pksp_j, ipol, jpol)
      if rank == 0:
        sigxy *= cgs_conv

#### Scaling??
        for n in range(ene.size):
          sigxy[n] *= ene[n]/105.4571

      fsigI = 'MCDi_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigI, ene, np.imag(sigxy))

      fsigR = 'MCDr_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigR, ene, np.real(sigxy))


def do_spin_current ( data_controller, spol, ipol ):

  arry,attr = data_controller.data_dicts()

  Sj = arry['Sj'][spol]
  bnd = attr['bnd']
  snktot,_,nawf,nawf,nspin = arry['dHksp'].shape

  jdHksp = np.empty((snktot,nawf,nawf,nspin), dtype=complex)

  for ispin in range(nspin):
    for ik in range(snktot):
      jdHksp[ik,:,:,ispin] = 0.5*(np.dot(Sj,arry['dHksp'][ik,ipol,:,:,ispin])+np.dot(arry['dHksp'][ik,ipol,:,:,ispin],Sj))

  return jdHksp


def do_spin_Hall_conductivity ( data_controller, jksp, pksp, ipol, jpol ):
  from .communication import gather_full
  from .smearing import intgaussian, intmetpax

  arry,attr = data_controller.data_dicts()

  snktot = jksp.shape[0]
  nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']

  # Compute the optical conductivity tensor sigma_xy(ene)

  ispin = 0

  emin = 0.0
  emax = attr['shift']
#### Hardcode 'de'
  esize = 500
  ene = np.linspace(emin, emax, esize)

  sigxy_aux = smear_sigma_loop(data_controller, ene, jksp, pksp, ispin, ipol, jpol)

  sigxy = (np.zeros((esize),dtype=complex) if rank==0 else None)

  comm.Reduce(sigxy_aux, sigxy, op=MPI.SUM)
  sigxy_aux = None

  if rank == 0:
    sigxy /= float(attr['nkpnts'])
    return(ene, sigxy)
  else:
    return(None, None)


def do_Berry_conductivity ( data_controller, pksp_i, pksp_j, ipol, jpol ):

  arry,attr = data_controller.data_dicts()

  snktot = pksp_j.shape[0]
  bnd = attr['bnd']

  # Compute the optical conductivity tensor sigma_xy(ene)

  ispin = 0

  emin = 0.0
  emax = attr['shift']
##### Hardcoded 'de'
  esize = 500
  ene = np.linspace(emin, emax, esize)

  sigxy_aux = np.zeros((esize),dtype=complex)

  sigxy_aux = smear_sigma_loop(data_controller, ene, pksp_i, pksp_j, ispin, ipol, jpol)

  sigxy = (np.zeros((esize),dtype=complex) if rank==0 else None)

  comm.Reduce(sigxy_aux, sigxy, op=MPI.SUM)
  sigxy_aux = None

  if rank == 0:
    sigxy /= float(attr['nkpnts'])
    return(ene, sigxy)
  else:
    return(None, None)


def smear_sigma_loop ( data_controller, ene, pksp_i, pksp_j, ispin, ipol, jpol ):
  from .smearing import intgaussian,intmetpax

  arry,attr = data_controller.data_dicts()

  esize = ene.size
  sigxy = np.zeros((esize), dtype=complex)

  snktot,nawf,_,nspin = pksp_j.shape
  f_nm = np.zeros((snktot,nawf,nawf), dtype=float)
  E_diff_nm = np.zeros((snktot,nawf,nawf), dtype=float)

  Ef = 0.0
  eps = 1.0e-16
  delta = 0.05

  if attr['smearing'] == None:
    fn = 1.0/(np.exp(arry['E_k'][:,:,ispin]/attr['temp'])+1)
  elif attr['smearing'] == 'gauss':
    fn = intgaussian(arry['E_k'][:,:,ispin], Ef, arry['deltakp'][:,:,ispin])
  elif smearing == 'm-p':
    fn = intmetpax(arry['E_k'][:,:,ispin], Ef, arry['deltakp'][:,:,ispin]) 

  # Collapsing the sum over k points
  for n in range(nawf):
    for m in range(nawf):
      if m != n:
        E_diff_nm[:,n,m] = (arry['E_k'][:,n,ispin]-arry['E_k'][:,m,ispin])**2
        f_nm[:,n,m] = (fn[:,n] - fn[:,m])*np.imag(pksp_i[:,n,m,ispin]*pksp_j[:,m,n,ispin])

  fn = None

  for e in range(esize):
    if attr['smearing'] != None:
      sigxy[e] = np.sum(f_nm[:,:,:]/(E_diff_nm[:,:,:]-(ene[e]+1.j*arry['deltakp2'][:,:nawf,:nawf,ispin])**2+eps))
    else:
      sigxy[e] = np.sum(f_nm[:,:,:]/(E_diff_nm[:,:,:]-(ene[e]+1.j*arry['delta'])**2+eps))

  F_nm = None
  E_diff_nm = None

  return np.nan_to_num(sigxy)
