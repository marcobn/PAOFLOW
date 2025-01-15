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

def do_dielectric_tensor ( data_controller, ene, from_H=True):
  from .constants import LL

  arrays,attributes = data_controller.data_dicts()
  smearing = attributes['smearing']
  d_tensor = arrays['d_tensor']

  for ispin in range(attributes['nspin']):
    for n in range(d_tensor.shape[0]):
      ipol = d_tensor[n][0]
      jpol = d_tensor[n][1]

      epsi,epsr,eels,jdos,ieps = do_epsilon(data_controller, ene, ispin, ipol, jpol,from_H)

      # Write files
      indices = (LL[ipol], LL[jpol], ispin)
      for ep,es in [(epsi,'epsi'),(epsr,'epsr'),(eels,'eels'),(jdos,'jdos'),(ieps,'ieps')]:
        fn = '%s_%s%s_%d.dat'%((es,)+indices)
        data_controller.write_file_row_col(fn, ene, ep)

      if rank == 0:
        renorm = np.sqrt((2./np.pi)*(ene[3]-ene[2])*np.sum(epsi*ene))
        direction = ['x','y','z']
        component = direction[int(ipol)]+direction[int(jpol)]
        print('Component',component,', plasmon frequency = ',renorm,'eV')
        print('Integration over JDOS = ', (ene[3]-ene[2])*np.sum(jdos))


def do_epsilon ( data_controller, ene, ispin, ipol, jpol, from_H ):
  from .constants import BOHR_RADIUS_ANGS, RYTOEV,ELECTRONVOLT_SI

  # Compute the dielectric tensor

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  if ene[0] == 0.:
    ene[0] = .00001

  #=======================
  # dipole matrix element 
  #=======================
  spin_factor = 2 if attributes['nspin']==1 and not attributes['dftSO'] else 1
  if (not from_H) or ('pksp' not in arrays):
    print("Calculate dipole matrix element from wavefunction")
    bndmax = attributes['nbnds']
    nspin = attributes['nspin']
    nktot = attributes['nkpnts']
    arrays['pksp'] = np.empty((nktot,3,bndmax,bndmax,nspin),dtype=np.complex128)
    for ispin in range(nspin):
      for ik in range(nktot):
        arrays['pksp'][ik,:,:,:,ispin] = calc_dipole(attributes,ik,ispin)
        
    factor = spin_factor*(2*np.pi/attributes['alat'])**2*RYTOEV**3\
      *64.0*np.pi/(attributes['omega']*attributes['nkpnts'])
  else:
    factor = spin_factor*ELECTRONVOLT_SI*(1e10)/(8.8541878188e-12)*BOHR_RADIUS_ANGS**2\
      /attributes['nkpnts']/(attributes['omega']*BOHR_RADIUS_ANGS**3)
     
  #=======================
  # EPS
  #=======================
  epsi_aux,epsr_aux,jdos_aux,count_aux = eps_loop(data_controller, ene, ispin, ipol, jpol, from_H)

  ### TNeeds revision. Each processor is allocating zeros here, when only rank 0 needs it. 
  ### Can be condensed
  epsi = np.zeros(esize, dtype=float)
  comm.Allreduce(epsi_aux, epsi, op=MPI.SUM)
  epsi_aux = None

  epsr = np.zeros(esize, dtype=float)
  comm.Allreduce(epsr_aux, epsr, op=MPI.SUM)
  epsr_aux = None

  jdos = np.zeros(esize, dtype=float)
  comm.Allreduce(jdos_aux, jdos, op=MPI.SUM)

  count = np.zeros(1,dtype=float)
  comm.Allreduce(count_aux, count, op=MPI.SUM)
  count_aux = None

  ieps = np.zeros(esize, dtype=float)
    
  epsi *= factor
  epsr =  1. + epsr*factor 
  eels = epsi/(epsi**2+epsr**2)
  for i in range(esize):
    for j in range(1,esize):
        ieps[i] += ene[j]*epsi[j]/(ene[i]**2+ene[j]**2)
  ieps = 1.0 + (2./np.pi)*ieps*(ene[3]-ene[2])
  jdos /= (4.*count[0])

  return(epsi, epsr, eels, jdos, ieps)


def eps_loop ( data_controller, ene, ispin, ipol, jpol, from_H):
  from .constants import RYTOEV

### What is this?
  orig_over_err = np.geterr()['over']
  np.seterr(over='raise')

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  if from_H:
    bndmax = attributes['bnd']
    Ek = arrays['E_k'][:,:bndmax,ispin]
  else:
    bndmax = attributes['nbnds']
    Ek = np.swapaxes(arrays['my_eigsmat'],0,1)

  intersmear = attributes['delta']  
  smearing = attributes['smearing']
  
  spin_factor = 2 if attributes['nspin']==1 and not attributes['dftSO'] else 1
  Ef = 1.e-9

  jdos = np.zeros(esize, dtype=float)
  epsi = np.zeros(esize, dtype=float)
  epsr = np.zeros(esize, dtype=float)
    
  if smearing == None or attributes['insulator']:
    print("No smearing, fixed occupation")
  else:
    if 'deltakp' in arrays:  # check whether adaptive smearing is used
      degauss = arrays['deltakp'][:,:bndmax,ispin] 
      print("Using adaptive smearing")
    else:
      degauss = attributes['degauss']
      print("Using fixed smearing = %.3f eV" %degauss)

  if smearing == None or attributes['insulator']:
    fn = 2.*(Ek <= Ef)  # fixed occupation for insulator, no smearing
  elif smearing == 'gauss':
    from .smearing import intgaussian,gaussian
    fn = 2.*intgaussian(Ek, Ef, degauss) 
  else: # smearing == 'm-p':
    from .smearing import intmetpax,metpax
    fn = 2.*intmetpax(Ek, Ef, degauss)

  
  count = np.zeros(1,dtype=float)
  for ik in range(fn.shape[0]):
    for iband2 in range(bndmax):
       for iband1 in range(bndmax):
          if iband1 != iband2:
             E_diff_nm = Ek[ik,iband2] - Ek[ik,iband1]
             f_nm =  fn[ik,iband2]-fn[ik,iband1]
             if np.abs(f_nm) > 2.e-3 and fn[ik,iband1] > 1.e-4 and fn[ik,iband2] < 2.0:
                pksp2 = np.real(arrays['pksp'][ik,ipol,iband1,iband2,ispin]*arrays['pksp'][ik,jpol,iband2,iband1,ispin])
                # pksp2 in unit of (AU*eV)^2
                epsi[:] +=  pksp2*intersmear*ene[:]*fn[ik,iband1]/(((E_diff_nm**2-ene[:]**2)**2+intersmear**2*ene[:]**2)*(E_diff_nm))
                epsr[:] +=  pksp2*(E_diff_nm**2-ene[:]**2)*fn[ik,iband1]/(((E_diff_nm**2-ene[:]**2)**2+intersmear**2*ene[:]**2)*(E_diff_nm))
                jdos[:] +=  intersmear*(fn[ik,iband1]-fn[ik,iband2])/(np.pi*((E_diff_nm-ene[:])**2+intersmear**2))
                count[0] += (fn[ik,iband1]-fn[ik,iband2])

  if not attributes['insulator']:
    epsi_metal = np.zeros_like(epsi)
    epsr_metal = np.zeros_like(epsr)

    if smearing == 'gauss':
      fnF = gaussian(Ek, Ef, degauss)
    elif smearing == 'm-p':
      fnF = metpax(Ek, Ef, degauss)
    else:
      print("Smearing is None for a metal, switching to gaussian smearing")
      fnF = gaussian(Ek, Ef, degauss)

    intrasmear = attributes['intrasmear']

    for ik in range(fn.shape[0]):
      for iband1 in range(bndmax):
        pksp2 = np.real(arrays['pksp'][ik,ipol,iband1,iband1,ispin]*arrays['pksp'][ik,jpol,iband1,iband1,ispin])
        epsi_metal[:] += pksp2*intrasmear*ene[:]*fnF[ik,iband1]/(ene[:]**4+intrasmear**2*ene[:]**2)
        epsr_metal[:] -= pksp2*fnF[ik,iband1]*ene[:]**2/(ene[:]**4+intrasmear**2*ene[:]**2)

    if not from_H:
      epsi_metal *= spin_factor/RYTOEV*3
      epsr_metal *= spin_factor/RYTOEV*3

    epsi += epsi_metal
    epsr += epsr_metal

  np.seterr(over=orig_over_err)
  return(epsi, epsr, jdos, count)

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
def calc_dipole(attr, ik, ispin):
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

        xk = np.frombuffer(record[1:7], np.float64)
        ispin = record[7]
        gamma_only = (record[8] != 0)
        scalef = np.frombuffer(record[9:], np.float64)[0]

        ngw, igwx, npol, nbnds = f.read_ints(np.int32)
        bg = f.read_reals(np.float64).reshape(3,3,order='F')
        mill = f.read_ints(np.int32).reshape(3,igwx,order='F')
    
        wfc = []
        for i in range(nbnds):
            wfc.append(f.read_reals(np.complex128))

    dipole_aux = np.zeros((3,nbnds,nbnds),dtype=np.complex128)
    for iband2 in range(nbnds):
        for iband1 in range(nbnds):
            dipole_aux[:,iband1,iband2] = (wfc[iband2]*mill)@np.conjugate(wfc[iband1])
    return dipole_aux

