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

from .smearing import intgaussian,gaussian
from .smearing import intmetpax,metpax
from .do_atwfc_proj import *
from .do_dipole import *

def do_dielectric_tensor ( data_controller, ene, from_wfc):
  from .constants import LL, RYTOEV

  arrays,attributes = data_controller.data_dicts()
  d_tensor = arrays['d_tensor']
  esize = ene.size

  nspin = attributes['nspin']

  if from_wfc == None:
    pass
  elif from_wfc == 'external':
    nbnds = attributes['nbnds']
    nkpnts = attributes['nkpnts']
    nspin = attributes['nspin']
    arrays['pksp'] = np.empty((nkpnts,3,nbnds,nbnds,nspin),dtype=np.complex128)
    for ispin in range(nspin):
      for ik in range(nkpnts):
        arrays['pksp'][ik,:,:,:,ispin] = calc_dipole(arrays,attributes, ik, ispin, arrays['b_vectors'])
  elif from_wfc == 'internal':
    nbnds = attributes['nawf']
    nkpnts = attributes['nkpnts']
    nspin = attributes['nspin']
    arrays['pksp'] = np.empty((nkpnts,3,nbnds,nbnds,nspin),dtype=np.complex128)
    for ispin in range(nspin):
      for ik in range(nkpnts):
        arrays['pksp'][ik,:,:,:,ispin] = calc_dipole_internal(data_controller, ik, ispin)
        # for n in range(attributes['nawf']):
        #   for m in range(attributes['nawf']):
        #     for l in range(3):
        #       arrays['pksp'][ik,l,n,m,ispin] = 1j*arrays['pksp'][ik,l,n,m,ispin]*(arrays['E_k'][ik,n]-arrays['E_k'][ik,m])
  else:
    raise Exception('ERROR: no dipole mode specified')

  if nspin == 1:
    for n in range(d_tensor.shape[0]):
      ipol = d_tensor[n][0]
      jpol = d_tensor[n][1]

      epsi,epsr,eels,ieps = do_epsilon(data_controller, ene, 0, ipol, jpol, from_wfc)
      # Write files
      indices = (LL[ipol], LL[jpol])
      for ep,es in [(epsi,'epsi'),(epsr,'epsr'),(eels,'eels'),(ieps,'ieps')]:
        fn = '%s_%s%s.dat'%((es,)+indices)
        data_controller.write_file_row_col(fn, ene, ep)

      if rank == 0 and ipol == jpol:
        renorm = np.sqrt((2./np.pi)*np.trapezoid(epsi*ene,x=ene))
        component = LL[ipol]+LL[jpol]
        print('Component', component, ', plasmon frequency = ',renorm,'eV')

  else:
    for n in range(d_tensor.shape[0]):
      ipol = d_tensor[n][0]
      jpol = d_tensor[n][1]

      epsi_0,epsr_0,eels_0,ieps_0 = do_epsilon(data_controller, ene, 0, ipol, jpol, from_wfc)
      epsi_1,epsr_1,eels_1,ieps_1 = do_epsilon(data_controller, ene, 1, ipol, jpol, from_wfc)
      # Write files
      indices = (LL[ipol], LL[jpol],0)
      for ep,es in [(epsi_0,'epsi'),(epsr_0,'epsr'),
                         (eels_0,'eels'),(ieps_0,'ieps')]:
        fn = '%s_%s%s_%d.dat'%((es,)+indices)
        data_controller.write_file_row_col(fn, ene, ep)
      indices = (LL[ipol], LL[jpol],1)
      for ep,es in [(epsi_1,'epsi'),(epsr_1,'epsr'),(eels_1,'eels'),(ieps_1,'ieps')]:
        fn = '%s_%s%s_%d.dat'%((es,)+indices)
        data_controller.write_file_row_col(fn, ene, ep)

      if rank == 0 and ipol == jpol:
        epsi = epsi_0 + epsi_1
        renorm = np.sqrt((2./np.pi)*np.trapezoid(epsi*ene,x=ene))
        component = LL[ipol]+LL[jpol]
        print('Component', component, ', plasmon frequency = ',renorm,'eV')
        

def do_jdos( data_controller, ene, jdos_smeartype):
  _,attributes = data_controller.data_dicts()
  esize = ene.size

  nspin = attributes['nspin']
  if nspin == 1:
    jdos_aux = jdos_loop(data_controller, ene, 0, jdos_smeartype)
    jdos = np.zeros(esize, dtype=float)
    comm.Allreduce(jdos_aux, jdos, op=MPI.SUM)
    jdos_aux = None

    fn = 'jdos.dat'
    data_controller.write_file_row_col(fn, ene, jdos)

    if rank == 0:
      print('Integration over JDOS = ', (np.trapezoid(jdos,x=ene)))
  else:
    jdos_aux0 = jdos_loop(data_controller, ene, 0, jdos_smeartype)
    jdos_aux1 = jdos_loop(data_controller, ene, 1, jdos_smeartype)
    jdos0 = np.zeros(esize, dtype=float)
    jdos1 = np.zeros(esize, dtype=float)
    comm.Allreduce(jdos_aux0, jdos0, op=MPI.SUM)
    comm.Allreduce(jdos_aux1, jdos1, op=MPI.SUM)
    jdos_aux0 = None
    jdos_aux1 = None

    fn0 = 'jdos_0.dat'
    data_controller.write_file_row_col(fn0, ene, jdos0)
    fn1 = 'jdos_1.dat'
    data_controller.write_file_row_col(fn1, ene, jdos1)

    if rank == 0:
      print('Integration over JDOS = ', (np.trapezoid(jdos0+jdos1,x=ene)))


def do_epsilon ( data_controller, ene, ispin, ipol, jpol, from_wfc):
  from .constants import BOHR_RADIUS_ANGS, RYTOEV,ELECTRONVOLT_SI

  # Compute the dielectric tensor

  _,attributes = data_controller.data_dicts()

  esize = ene.size
  if ene[0] == 0.:
    ene[0] = .00001
  
  if from_wfc == None:
    # 8.8541878188e-12 = \epsilon_0
    factor = ELECTRONVOLT_SI*(1e10)/(8.8541878188e-12)*BOHR_RADIUS_ANGS**2\
      /attributes['nkpnts']/(attributes['omega']*BOHR_RADIUS_ANGS**3)
  elif from_wfc == 'external':
    factor = 2*(np.pi/attributes['alat'])**2*RYTOEV**3\
      *64.0*np.pi/(attributes['omega']*attributes['nkpnts'])
  elif from_wfc == 'internal':
    # factor = ELECTRONVOLT_SI*(1e10)/(8.8541878188e-12)*BOHR_RADIUS_ANGS**2\
    #   /attributes['nkpnts']/(attributes['omega']*BOHR_RADIUS_ANGS**3)
    factor = 2*(np.pi/attributes['alat'])**2*RYTOEV**3\
      *64.0*np.pi/(attributes['omega']*attributes['nkpnts'])
  else:
    raise Exception('ERROR: no dipole mode specified')
     
  #=======================
  # EPS
  #=======================

  epsi_aux,epsr_aux = eps_loop(data_controller, ene, ispin, ipol, jpol, from_wfc)
  epsi = np.zeros(esize, dtype=float)
  comm.Allreduce(epsi_aux, epsi, op=MPI.SUM)
  epsi_aux = None

  epsr = np.zeros(esize, dtype=float)
  comm.Allreduce(epsr_aux, epsr, op=MPI.SUM)
  epsr_aux = None


  ### TNeeds revision. Each processor is allocating zeros here, when only rank 0 needs it. 
  ### Can be condensed
  
  ieps = np.zeros(esize, dtype=float)
    
  epsi *= factor
  epsr =  1.*(ipol==jpol) + epsr*factor 
  eels = epsi/(epsi**2+epsr**2)
  for i in range(esize):
    for j in range(1,esize):
        ieps[i] += ene[j]*epsi[j]/(ene[i]**2+ene[j]**2)
  ieps = 1.0 + (2./np.pi)*ieps*(ene[3]-ene[2])

  return(epsi, epsr, eels, ieps)


def eps_loop ( data_controller, ene, ispin, ipol, jpol, from_wfc):

  orig_over_err = np.geterr()['over']
  np.seterr(over='raise')

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  if from_wfc == None or from_wfc == 'internal':
    bndmax = attributes['bnd']
    Ek = arrays['E_k'][:,:bndmax,ispin]
  elif from_wfc == 'external':
    bndmax = attributes['bnd']
    Ek = np.swapaxes(arrays['my_eigsmat'][:,:,ispin],0,1)
  else:
    raise Exception('ERROR: no dipole mode specified')
  # 

  intersmear = attributes['delta']  
  smearing = attributes['smearing']
  degauss = attributes['degauss']
  
  # spin_factor = 2 if attributes['nspin']==1 and not attributes['dftSO'] else 1
  spin_factor = 2 if attributes['nspin']==1 else 1
  Ef = 1.e-9

  epsi = np.zeros(esize, dtype=float)
  epsr = np.zeros(esize, dtype=float)
    
  if smearing == None or attributes['insulator']:
      if rank == 0: print("No smearing, fixed occupation")
  else:
    if 'deltakp' in arrays:  # check whether adaptive smearing is used
      degauss = arrays['deltakp'][:,:bndmax,ispin] 
      if rank == 0: print("Using adaptive smearing")
    else:
      degauss = attributes['degauss']
      if rank == 0: print("Using fixed smearing = %.3f eV" %degauss)

  if smearing == None or attributes['insulator']:
    fn = spin_factor*(Ek <= Ef)  # fixed occupation for insulator, no smearing
  elif smearing == 'gauss':
    
    fn = spin_factor*intgaussian(Ek, Ef, degauss) 
  else: # smearing == 'm-p':

    fn = spin_factor*intmetpax(Ek, Ef, degauss)

  th0 = 1.e-3*spin_factor
  th1 = 0.5e-4*spin_factor
  if attributes['dftSO'] and from_wfc == None: 
    fac = 1
  else:
    fac = 2
  for ik in range(fn.shape[0]):
    for iband2 in range(bndmax):
       for iband1 in range(bndmax):
          if iband1 != iband2:
             E_diff_nm = Ek[ik,iband2] - Ek[ik,iband1]
             f_nm =  fn[ik,iband2]-fn[ik,iband1]
             if np.abs(f_nm) > th0 and fn[ik,iband1] > th1 and fn[ik,iband2] < spin_factor:
                pksp2 = np.real(arrays['pksp'][ik,ipol,iband1,iband2,ispin]\
                        *arrays['pksp'][ik,jpol,iband2,iband1,ispin])
                # pksp2 in unit of (AU*eV)^2
                epsi[:] += fac*pksp2*intersmear*ene[:]*fn[ik,iband1]\
                /(((E_diff_nm**2-ene[:]**2)**2+intersmear**2*ene[:]**2)*(E_diff_nm))
                epsr[:] += fac*pksp2*(E_diff_nm**2-ene[:]**2)*fn[ik,iband1]\
                /(((E_diff_nm**2-ene[:]**2)**2+intersmear**2*ene[:]**2)*(E_diff_nm))

  
  if not attributes['insulator']:
    epsi_metal = np.zeros_like(epsi)
    epsr_metal = np.zeros_like(epsr)

    if smearing == 'gauss':
      fnF = spin_factor*gaussian(Ek, Ef, degauss)
    elif smearing == 'm-p':
      fnF = spin_factor*metpax(Ek, Ef, degauss)
    else:
      print("Smearing is None for a metal, switching to gaussian smearing")
      fnF = spin_factor*gaussian(Ek, Ef, degauss)

    intrasmear = attributes['intrasmear']

    for ik in range(fn.shape[0]):
      for iband1 in range(bndmax):
          pksp2 = np.real(arrays['pksp'][ik,ipol,iband1,iband1,ispin]\
                          *arrays['pksp'][ik,jpol,iband1,iband1,ispin])
          epsi_metal[:] += pksp2*intrasmear*ene[:]*fnF[ik,iband1]/(ene[:]**4+intrasmear**2*ene[:]**2)
          epsr_metal[:] -= pksp2*fnF[ik,iband1]*ene[:]**2/(ene[:]**4+intrasmear**2*ene[:]**2)

    if from_wfc != None:
      from .constants import RYTOEV
      epsi_metal /= RYTOEV/4
      epsr_metal /= RYTOEV/4

    epsi += epsi_metal
    epsr += epsr_metal

  np.seterr(over=orig_over_err)
  return(epsi, epsr)


def jdos_loop(data_controller, ene, ispin, jdos_smeartype):

  arrays,attributes = data_controller.data_dicts()
  intersmear = attributes['delta']  
  smearing = attributes['smearing']
  esize = ene.size
  bndmax = attributes['bnd']
  Ek = arrays['E_k'][:,:bndmax,ispin]
  # bndmax = attributes['nbnds']
  # Ek = np.swapaxes(arrays['my_eigsmat'][:,:,ispin],0,1)
  nkpnts = Ek.shape[0]
  jdos = np.zeros(esize, dtype=float)
  Ef = 1.e-9
  degauss = attributes['degauss']

  # spin_factor = 2 if attributes['nspin']==1 and not attributes['dftSO'] else 1
  spin_factor = 2 if attributes['nspin']==1 else 1
  if smearing == None or attributes['insulator']:
    fn = spin_factor*(Ek <= Ef)  # fixed occupation for insulator, no smearing
  elif smearing == 'gauss':
    fn = spin_factor*intgaussian(Ek, Ef, degauss) 
  else: # smearing == 'm-p':
    fn = spin_factor*intmetpax(Ek, Ef, degauss)

  # count = 0.0
  if jdos_smeartype == 'gauss':

    for ik in range(nkpnts):
      for iband2 in range(bndmax):
        for iband1 in range(bndmax):
            E_diff_nm = Ek[ik,iband2] - Ek[ik,iband1]
            if fn[ik,iband1] > 1.e-4 and fn[ik,iband2] < 2.0 and E_diff_nm > 1e-10:
              f_nm = fn[ik,iband1]-fn[ik,iband2]
              jdos += f_nm * gaussian(E_diff_nm, ene, intersmear) 
              # count += f_nm

  elif jdos_smeartype == 'lorentz':
  
    for ik in range(nkpnts):
      for iband2 in range(bndmax):
        for iband1 in range(bndmax):
            E_diff_nm = Ek[ik,iband2] - Ek[ik,iband1]
            if fn[ik,iband1] > 1.e-4 and fn[ik,iband2] < 2.0 and E_diff_nm > 1e-10:
              f_nm = fn[ik,iband1]-fn[ik,iband2]
              jdos +=  f_nm * intersmear / (np.pi*((E_diff_nm-ene)**2+intersmear**2))
              # count += f_nm

  else:
    raise ValueError("jdos_smeartype must be 'gauss' or 'lorentz' ")
  jdos /= nkpnts

  return jdos

   

"""
def epsr_kramerskronig ( data_controller, ene, epsi ):
  from .smearing import intmetpax
  from scipy.integrate import simpson
  from .communication import load_balancing

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  de = ene[1] - ene[0]

  epsr = np.zeros(esize, dtype=float)

  ini_ie,end_ie = load_balancing(comm.Get_size(), rank, esize)

  # Range checks for Simpson Integrals
  if end_ie == ini_ie:
    return
  if ini_ie < 3:
    ini_ie = 3
  if end_ie == esize:
    end_ie = esize-1

  f_ene = intmetpax(ene, attributes['shift'], 1.)
  for ie in range(ini_ie, end_ie):
    I1 = simpson(ene[1:(ie-1)]*de*epsi[1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2))
    I2 = simpson(ene[(ie+1):esize]*de*epsi[(ie+1):esize]*f_ene[(ie+1):esize]/(ene[(ie+1):esize]**2-ene[ie]**2))
    epsr[ie] = 2.*(I1+I2)/np.pi

  return epsr
"""