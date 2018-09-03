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

import os
#import matplotlib.pyplot as plt


# Compute Z2 invariant and topological properties on a selected path in the BZ
def do_topology ( data_controller ):
  import numpy as np
  from mpi4py import MPI
  from pfaffian import pfaffian 
  from scipy.fftpack import fftshift
  from constants import LL, ANGSTROM_AU
  from get_R_grid_fft import get_R_grid_fft
  from communication import scatter_full,gather_full
  from kpnts_interpolation_mesh import kpnts_interpolation_mesh

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  npool = attributes['npool']

  if 'kq' not in arrays:
    kpnts_interpolation_mesh(data_controller)

  if 'Rfft' not in arrays:
    get_R_grid_fft(data_controller)

  bnd = attributes['bnd']
  nkpi = arrays['kq'].shape[1]
  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  ipol = attributes['ipol']
  jpol = attributes['jpol']
  spol = attributes['spol']

  Berry = attributes['Berry']
  eff_mass = attributes['eff_mass']
  spin_Hall = attributes['spin_Hall']

  alat = attributes['alat'] / ANGSTROM_AU
  b_vectors = arrays['b_vectors']

  # Compute Z2 according to Fu, Kane and Mele (2007)
  # Define TRIM points in 2(0-3)/3D(0-7)
## 
  if nspin == 1 and spin_Hall:
    pass
##
  if False:
    from do_eigh_calc import do_eigh_calc
  # NOT IMPLEMENTED IN PAOFLOW_CLASS
    print('Topology with nspin==1 and spin_Hall under development in PAOFLOW_CLASS')
    quit()

    nktrim = 16
    ktrim = np.zeros((nktrim,3),dtype=float)
    ktrim[0] = np.zeros(3,dtype=float)                                  #0 0 0 0
    ktrim[1] = b_vectors[0,:]/2.0                                       #1 1 0 0
    ktrim[2] = b_vectors[1,:]/2.0                                       #2 0 1 0
    ktrim[3] = b_vectors[0,:]/2.0+b_vectors[1,:]/2.0                    #3 1 1 0
    ktrim[4] = b_vectors[2,:]/2.0                                       #4 0 0 1
    ktrim[5] = b_vectors[1,:]/2.0+b_vectors[2,:]/2.0                    #5 0 1 1
    ktrim[6] = b_vectors[2,:]/2.0+b_vectors[0,:]/2.0                    #6 1 0 1
    ktrim[7] = b_vectors[0,:]/2.0+b_vectors[1,:]/2.0+b_vectors[2,:]/2.0 #7 1 1 1
    ktrim[8:16] = -ktrim[:8]
    # Compute eigenfunctions at the TRIM points
    E_ktrim,v_ktrim = do_eigh_calc(HRs,SRs,ktrim,R_wght,R,idx,non_ortho)
    # Define time reversal operator
    theta = -1.0j*clebsch_gordan(nawf,sh,nl,1)
    wl = np.zeros((nktrim/2,nawf,nawf),dtype=complex)
    for ik in range(nktrim/2):
      wl[ik,:,:] = np.conj(v_ktrim[ik,:,:,0].T).dot(theta).dot(np.conj(v_ktrim[ik+nktrim/2,:,:,0]))
      wl[ik,:,:] = wl[ik,:,:]-wl[ik,:,:].T  # enforce skew symmetry
    delta_ik = np.zeros(nktrim/2,dtype=complex)
    for ik in range(nktrim/2):
      delta_ik[ik] = pfaffian(wl[ik,:nelec,:nelec])/np.sqrt(LAN.det(wl[ik,:nelec,:nelec]))

    f=open(os.path.join(attributes['inputpath'],'Z2'+'.dat'),'w')
    p2D = np.real(np.prod(delta_ik[:4]))
    if p2D+1.0 < 1.e-5:
      v0 = 1
    elif p2D-1.0 < 1.e-5:
      v0 = 0
    f.write('2D case: v0 = %1d \n' %(v0))
    p3D = np.real(np.prod(delta_ik))
    if p3D+1.0 < 1.e-5:
      v0 = 1
    elif p3D-1.0 < 1.e-5:
      v0 = 0
    p3D = delta_ik[1]*delta_ik[3]*delta_ik[6]*delta_ik[7]
    if p3D+1.0 < 1.e-5:
      v1 = 1
    elif p3D-1.0 < 1.e-5:
      v1 = 0
    p3D = delta_ik[2]*delta_ik[3]*delta_ik[5]*delta_ik[7]
    if p3D+1.0 < 1.e-5:
      v2 = 1
    elif p3D-1.0 < 1.e-5:
      v2 = 0
    p3D = delta_ik[4]*delta_ik[6]*delta_ik[5]*delta_ik[7]
    if p3D+1.0 < 1.e-5:
      v3 = 1
    elif p3D-1.0 < 1.e-5:
      v3 = 0
    f.write('3D case: v0;v1,v2,v3 = %1d;%1d,%1d,%1d \n' %(v0,v1,v2,v3))
    f.close()


  # Compute momenta and kinetic energy

  kq_aux = scatter_full(arrays['kq'].T, npool)
  kq_aux = kq_aux.T

  # Compute R*H(R)
  arrays['HRs'] = fftshift(arrays['HRs'], axes=(2,3,4))
  arrays['Rfft'] = np.reshape(arrays['Rfft'], (nk1*nk2*nk3,3), order='C')
  arrays['HRs'] = np.reshape(arrays['HRs'], (nawf,nawf,nk1*nk2*nk3,nspin), order='C')
  arrays['HRs'] = np.moveaxis(arrays['HRs'], 2, 0)

  HRs_aux = scatter_full(arrays['HRs'], npool)
  Rfft_aux = scatter_full(arrays['Rfft'], npool)

  arrays['HRs'] = np.reshape(np.moveaxis(arrays['HRs'],0,2), (nawf,nawf,nk1,nk2,nk3,nspin), order='C')

  if spin_Hall:
    Sj = arrays['Sj']
    jks = np.zeros((kq_aux.shape[1],3,bnd,bnd,nspin), dtype=complex)

  pks = np.zeros((kq_aux.shape[1],3,bnd,bnd,nspin), dtype=complex)
  for l in range(3):
    dHRs  = np.zeros((HRs_aux.shape[0],nawf,nawf,nspin),dtype=complex)
    for ispin in range(nspin):
      for n in range(nawf):
        for m in range(nawf):
          dHRs[:,n,m,ispin] = 1.0j*alat*ANGSTROM_AU*Rfft_aux[:,l]*HRs_aux[:,n,m,ispin]

    dHRs = gather_full(dHRs, npool)  
    if rank != 0:
      dHRs = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin), dtype=complex)
    comm.Bcast(dHRs)
    dHRs = np.moveaxis(dHRs,0,2)

    # Compute dH(k)/dk on the path
    dHks_aux = band_loop_H(dHRs, arrays['Rfft'], kq_aux, nawf, nspin)

    dHRs = None

    # Compute momenta
    for ik in range(dHks_aux.shape[0]):
      for ispin in range(nspin):
        pks[ik,l,:,:,ispin] = np.conj(arrays['v_k'][ik,:,:,ispin].T).dot(dHks_aux[ik,:,:,ispin]).dot(arrays['v_k'][ik,:,:,ispin])[:bnd,:bnd]

    if spin_Hall:
      for ik in range(pks.shape[0]):
        for ispin in range(nspin):
          jks[ik,l,:,:,ispin] = (np.conj(arrays['v_k'][ik,:,:,ispin].T).dot \
            (0.5*(np.dot(Sj[l],dHks_aux[ik,:,:,ispin])+np.dot(dHks_aux[ik,:,:,ispin],Sj[l]))).dot(arrays['v_k'][ik,:,:,ispin]))[:bnd,:bnd]

  if eff_mass == True: 
    tks = np.zeros((kq_aux.shape[1],3,3,bnd,bnd,nspin), dtype=complex)

    for l in range(3):
      for lp in range(3):
        d2HRs = np.zeros((HRs_aux.shape[0],nawf,nawf,nspin), dtype=complex)
        for ispin in range(nspin):
          for n in range(nawf):
            for m in range(nawf):
              d2HRs[:,n,m,ispin] = -1.0*alat**2*ANGSTROM_AU**2*Rfft_aux[:,l]*Rfft_aux[:,lp]*HRs_aux[:,n,m,ispin]

        d2HRs = gather_full(d2HRs, npool)  
        if rank != 0:
          d2HRs = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin), dtype=complex)      
        comm.Bcast(d2HRs)
        d2HRs = np.moveaxis(d2HRs, 0, 2)

        # Compute d2H(k)/dk*dkp on the path
        d2Hks_aux = band_loop_H(d2HRs, arrays['Rfft'], kq_aux, nawf, nspin)

        d2HRs = None

        # Compute kinetic energy
        for ik in range(d2Hks_aux.shape[0]):
          for ispin in range(nspin):
            tks[ik,l,lp,:,:,ispin] = (np.conj(arrays['v_k'][ik,:,:,ispin].T).dot(d2Hks_aux[ik,:,:,ispin]).dot(arrays['v_k'][ik,:,:,ispin]))[:bnd,:bnd]

        d2Hks_aux = None

    # Compute effective mass
    mkm1 = np.zeros((tks.shape[0],bnd,3,3,nspin), dtype=complex)
    for ik in range(tks.shape[0]):
      for ispin in range(nspin):
        for n in range(bnd):
          for m in range(bnd):
            if m != n:
              mkm1[ik,n,ipol,jpol,ispin] += (pks[ik,ipol,n,m,ispin]*pks[ik,jpol,m,n,ispin]+pks[ik,jpol,n,m,ispin]*pks[ik,ipol,m,n,ispin]) / \
                            (arrays['E_k'][ik,n,ispin]-arrays['E_k'][ik,m,ispin]+1.e-16)
            else:
              mkm1[ik,n,ipol,jpol,ispin] += tks[ik,ipol,jpol,n,n,ispin]


    tks = None

    mkm1 = gather_full(mkm1, npool)

    #mkm1 *= ELECTRONVOLT_SI**2/H_OVER_TPI**2*ELECTRONMASS_SI
    if rank == 0:
      for ispin in range(nspin):
        f = open(os.path.join(attributes['opath'],'effmass'+'_'+str(LL[ipol])+str(LL[jpol])+'_'+str(ispin)+'.dat'),'w')
        for ik in range(nkpi):
          s="%d\t"%ik
          for  j in np.real(mkm1[ik,:bnd,ipol,jpol,ispin]):s += "% 3.5f\t"%j
          s+="\n"
          f.write(s)
        f.close()

    mkm1 = None

  HRs_aux = None
  HRs = None
  
  # Compute Berry curvature
  if Berry or spin_Hall:
    deltab = 0.05
    mu = -0.2 # chemical potential in eV)
    Om_zk = np.zeros((pks.shape[0],1), dtype=float)
    Om_znk = np.zeros((pks.shape[0],bnd), dtype=float)
    Omj_zk = (np.zeros((pks.shape[0],1), dtype=float) if spin_Hall else None)
    Omj_znk = (np.zeros((pks.shape[0],bnd), dtype=float) if spin_Hall else None)
    for ik in range(pks.shape[0]):
      for n in range(bnd):
        for m in range(bnd):
          if m != n:
            if Berry:
              Om_znk[ik,n] += -1.0*np.imag(pks[ik,jpol,n,m,0]*pks[ik,ipol,m,n,0]-pks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0])/((arrays['E_k'][ik,m,0] - arrays['E_k'][ik,n,0])**2 + deltab**2)
            if spin_Hall:
              Omj_znk[ik,n] += -2.0*np.imag(jks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0])/((arrays['E_k'][ik,m,0] - arrays['E_k'][ik,n,0])**2 + deltab**2)
      Om_zk[ik] = np.sum(Om_znk[ik,:]*(0.5 * (-np.sign(arrays['E_k'][ik,:bnd,0]) + 1)))  # T=0.0K
      if spin_Hall:
        Omj_zk[ik] = np.sum(Omj_znk[ik,:]*(0.5 * (-np.sign(arrays['E_k'][ik,:bnd,0]-mu) + 1)))  # T=0.0K

  indices = (LL[spol], LL[ipol], LL[jpol])
  lrng = (list(range(nkpi)) if rank==0 else None)

  if Berry:
    Om_zk = gather_full(Om_zk, npool)
    fOm_zk = 'Omega_%s_%s%s.dat'%indices
    data_controller.write_file_row_col(fOm_zk, lrng, (Om_zk[:,0] if rank==0 else None))

  if spin_Hall:
    Omj_zk = gather_full(Omj_zk, npool)
    fOmj_zk = 'Omegaj_%s_%s%s.dat'%indices
    data_controller.write_file_row_col(fOmj_zk, lrng, (Omj_zk[:,0] if rank==0 else None))

  return
#### Velocity write
#### Same as Band write
#### Write DataController band write
  pks = gather_full(pks, npool)
  if rank == 0:
    if attributes['do_spin_orbit']:
      bnd *= 2
##      attributes['bnd'] *= 2
    velk = np.zeros((nkpi,3,bnd,nspin), dtype=float)
    for n in range(bnd):
      velk[:,:,n,:] = np.real(pks[:,:,n,n,:])
    for ispin in range(nspin):
      for l in range(3):
        f=open(os.path.join(attributes['inputpath'],'velocity_'+str(l)+'_'+str(ispin)+'.dat'),'w')
        for ik in range(nkpi):
          s="%d\t"%ik
          for  j in velk[ik,l,:bnd,ispin]:s += "%3.5f\t"%j
          s+="\n"
          f.write(s)
        f.close()


def band_loop_H ( HRaux, R, kq, nawf, nspin ):
  import numpy as np

  kdot = np.zeros((kq.shape[1],R.shape[0]),dtype=complex,order="C")
  kdot = np.tensordot(R,2.0j*np.pi*kq,axes=([1],[0]))
  np.exp(kdot,kdot)

  Haux = np.zeros((nawf,nawf,kq.shape[1],nspin),dtype=complex,order="C")

  for ispin in range(nspin):
    Haux[:,:,:,ispin] = np.tensordot(HRaux[:,:,:,ispin], kdot, axes=([2],[0]))

  kdot  = None
  Haux = np.transpose(Haux,(2,0,1,3))
  return Haux