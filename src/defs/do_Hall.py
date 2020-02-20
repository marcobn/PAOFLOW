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
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def do_spin_Hall ( data_controller, twoD, do_ac ):
  from .perturb_split import perturb_split
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
    ene,shc,Om_k = do_Berry_curvature(data_controller, jksp_is, pksp_j)

    if rank == 0:
      if twoD:
        av0,av1 = arry['a_vectors'][0,:],arry['a_vectors'][1,:]
        cgs_conv = 1./(np.linalg.norm(np.cross(av0,av1))*attr['alat']**2)
      else:
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

      jdHksp = do_spin_current(data_controller, spol, ipol)

      jksp_js = np.empty_like(jdHksp)
      pksp_i = np.empty_like(jdHksp)

      for ik in range(jdHksp.shape[0]):
        for ispin in range(jdHksp.shape[3]):
          jksp_js[ik,:,:,ispin],pksp_i[ik,:,:,ispin] = perturb_split(jdHksp[ik,:,:,ispin], arry['dHksp'][ik,jpol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])
      jdHksp = None

      ene,sigxy = do_ac_conductivity(data_controller, jksp_js, pksp_i, ipol, jpol)
      if rank == 0:
        sigxy *= cgs_conv

      sigxyi = np.imag(ene*sigxy/105.4571) if rank==0 else None
      sigxyr = np.real(sigxy) if rank==0 else None
      sigxy = None

      fsigI = 'SCDi_%s_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigI, ene, sigxyi)

      fsigR = 'SCDr_%s_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigR, ene, sigxyr)


def do_anomalous_Hall ( data_controller, do_ac ):
  from .perturb_split import perturb_split
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

    ene,ahc,Om_k = do_Berry_curvature(data_controller, pksp_i, pksp_j)

    if rank == 0:
      cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega'])

    cart_indices = (str(LL[ipol]),str(LL[jpol]))

    fBerry = 'Berry_%s%s.bxsf'%cart_indices
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
      ene,sigxy = do_ac_conductivity(data_controller, pksp_i, pksp_j, ipol, jpol)
      if rank == 0:
        sigxy *= cgs_conv

      sigxyi = np.imag(ene*sigxy/105.4571) if rank==0 else None
      sigxyr = np.real(sigxy) if rank==0 else None
      sigxy = None

      fsigI = 'MCDi_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigI, ene, sigxyi)

      fsigR = 'MCDr_%s%s.dat'%cart_indices
      data_controller.write_file_row_col(fsigR, ene, sigxyr)

def do_Berry_curvature ( data_controller, jksp, pksp ):
  #----------------------
  # Compute spin Berry curvature
  #----------------------
  from .communication import gather_full
  from .smearing import intgaussian, intmetpax

  arrays,attributes = data_controller.data_dicts()

  snktot,nawf,_,nspin = pksp.shape
  fermi_up,fermi_dw = attributes['fermi_up'],attributes['fermi_dw']
  nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']

  # Compute only Omega_z(k)
  Om_znkaux = np.zeros((snktot,nawf), dtype=float)

  deltap = 0.05
  for ik in range(snktot):
    E_nm = (arrays['E_k'][ik,:,0] - arrays['E_k'][ik,:,0][:,None])**2 #+ deltap**2
    E_nm[np.where(E_nm<1.e-4)] = np.inf
    Om_znkaux[ik] = -2.0*np.sum(np.imag(jksp[ik,:,:,0]*pksp[ik,:,:,0].T)/E_nm, axis=1)
  E_nm = None

  attributes['emaxH'] = np.amin(np.array([attributes['shift'],attributes['emaxH']]))
  ### Hardcoded 'de'
  esize = 500
  ene = np.linspace(attributes['eminH'], attributes['emaxH'], esize)

  Om_zkaux = np.zeros((snktot,esize), dtype=float)

  for i in range(esize):
    if attributes['smearing'] == 'gauss':
      Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intgaussian(arrays['E_k'][:,:,0],ene[i],arrays['deltakp'][:,:,0]), axis=1)
    elif attributes['smearing'] == 'm-p':
      Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intmetpax(arrays['E_k'][:,:,0],ene[i],arrays['deltakp'][:,:,0]), axis=1)
    else:
      Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*(0.5 * (-np.sign(arrays['E_k'][:,:,0]-ene[i]) + 1)), axis=1)

  Om_zk = gather_full(Om_zkaux, attributes['npool'])
  Om_zkaux = None

  shc = None
  if rank == 0:
    shc = np.sum(Om_zk, axis=0)/float(attributes['nkpnts'])

  n0 = 0
  n = esize-1
  if rank == 0:
    for i in range(esize-1):
      if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
        n0 = i
      if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
        n = i
    Om_zk = np.reshape(Om_zk, (nk1,nk2,nk3,esize), order='C')
    Om_zk = Om_zk[:,:,:,n]-Om_zk[:,:,:,n0]

  return(ene, shc, Om_zk)

def do_ac_conductivity ( data_controller, jksp, pksp, ipol, jpol ):
  from .communication import gather_full
  from .smearing import intgaussian, intmetpax

  arry,attr = data_controller.data_dicts()

  # Compute the optical conductivity tensor sigma_xy(ene)

  ispin = 0

  emin = 0.0
  emax = attr['shift']
  ### Hardcode 'de'
  esize = 501
  ene = np.linspace(emin, emax, esize)

  sigxy_aux = smear_sigma_loop(data_controller, ene, jksp, pksp, ispin, ipol, jpol)

  sigxy = (np.zeros((esize),dtype=complex) if rank==0 else None)
  sigxyR = (np.zeros((esize),dtype=float) if rank==0 else None)
  sigxyI = (np.zeros((esize),dtype=float) if rank==0 else None)
  
  sigxy_auxR = np.ascontiguousarray(np.real(sigxy_aux))
  sigxy_auxI = np.ascontiguousarray(np.imag(sigxy_aux))
  
  comm.Reduce(sigxy_auxR, sigxyR, op=MPI.SUM)
  comm.Reduce(sigxy_auxI, sigxyI, op=MPI.SUM)

  sigxy_aux = sigxy_auxR = sigxy_auxI = None

  if rank == 0:
    sigxy = (sigxyR+1j*sigxyI)/float(attr['nkpnts'])
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

  st=time.time()
  if attr['smearing'] == None:
    fn = 1.0/(np.exp(arry['E_k'][:,:,ispin]/attr['temp'])+1)
  elif attr['smearing'] == 'gauss':
    fn = intgaussian(arry['E_k'][:,:,ispin], Ef, arry['deltakp'][:,:,ispin])
  elif smearing == 'm-p':
    fn = intmetpax(arry['E_k'][:,:,ispin], Ef, arry['deltakp'][:,:,ispin]) 

  arry['E_k']  = np.asfortranarray(arry["E_k"])

  pksp_i = np.asfortranarray(pksp_i)
  pksp_j = np.asfortranarray(pksp_j)
  # Collapsing the sum over k points
  st=time.time()
  for n in range(nawf):
    for m in range(nawf):
      if m != n:
        E_diff_nm[:,n,m] = (arry['E_k'][:,n,ispin]-arry['E_k'][:,m,ispin])**2
        f_nm[:,n,m] = (fn[:,n] - fn[:,m])*np.imag(pksp_j[:,n,m,ispin]*pksp_i[:,m,n,ispin])

  fn = None

  E_diff_nm  = np.ascontiguousarray(E_diff_nm)
  f_nm  = np.ascontiguousarray(f_nm)

  for e in range(esize):
    if attr['smearing'] != None:
      sigxy[e] = np.sum(f_nm[:,:,:]/(E_diff_nm[:,:,:]-(ene[e]+1.j*arry['deltakp2'][:,:,:,ispin])**2+eps))
    else:
      sigxy[e] = np.sum(f_nm[:,:,:]/(E_diff_nm[:,:,:]-(ene[e]+1.j*arry['delta'])**2+eps))

  F_nm = None
  E_diff_nm = None

  return np.nan_to_num(sigxy)

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

