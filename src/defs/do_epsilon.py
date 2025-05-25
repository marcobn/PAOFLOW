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

def do_dielectric_tensor ( data_controller, ene, from_wfc):
  from .constants import LL

  arrays,attributes = data_controller.data_dicts()
  d_tensor = arrays['d_tensor']
  esize = ene.size

  nspin = attributes['nspin']

  if from_wfc:
    # nbnds = attributes['nbnds']
    nbnds = attributes['nawf']
    nkpnts = attributes['nkpnts']
    nspin = attributes['nspin']
    arrays['pksp'] = np.empty((nkpnts,3,nbnds,nbnds,nspin),dtype=np.complex128)
    for ispin in range(nspin):
      for ik in range(nkpnts):
        # arrays['pksp'][ik,:,:,:,ispin] = calc_dipole(arrays,attributes, ik, ispin, arrays['b_vectors'])
        arrays['pksp'][ik,:,:,:,ispin] = calc_dipole_internal(data_controller, ik, ispin)

  
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
        renorm = np.sqrt((2./np.pi)*np.trapz(epsi*ene,x=ene))
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
        renorm = np.sqrt((2./np.pi)*np.trapz(epsi*ene,x=ene))
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
      print('Integration over JDOS = ', (np.trapz(jdos,x=ene)))
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
      print('Integration over JDOS = ', (np.trapz(jdos0+jdos1,x=ene)))


def do_epsilon ( data_controller, ene, ispin, ipol, jpol, from_wfc):
  from .constants import BOHR_RADIUS_ANGS, RYTOEV,ELECTRONVOLT_SI

  # Compute the dielectric tensor

  _,attributes = data_controller.data_dicts()

  esize = ene.size
  if ene[0] == 0.:
    ene[0] = .00001
  
  if from_wfc:
    factor = 2*(np.pi/attributes['alat'])**2*RYTOEV**3\
      *64.0*np.pi/(attributes['omega']*attributes['nkpnts'])
    # factor = ELECTRONVOLT_SI*(1e10)/(8.8541878188e-12)*BOHR_RADIUS_ANGS**2\
    #   /attributes['nkpnts']/(attributes['omega']*BOHR_RADIUS_ANGS**3)
  else:
    factor = ELECTRONVOLT_SI*(1e10)/(8.8541878188e-12)*BOHR_RADIUS_ANGS**2\
      /attributes['nkpnts']/(attributes['omega']*BOHR_RADIUS_ANGS**3)
     
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
  if from_wfc:
    # bndmax = attributes['bnd']
    # Ek = np.swapaxes(arrays['my_eigsmat'][:,:,ispin],0,1)
    bndmax = attributes['bnd']
    Ek = arrays['E_k'][:,:bndmax,ispin]
  else:
    bndmax = attributes['bnd']
    Ek = arrays['E_k'][:,:bndmax,ispin]
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
  if attributes['dftSO'] and not from_wfc: 
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

    if from_wfc:
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
