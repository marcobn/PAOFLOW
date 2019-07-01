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

def do_dielectric_tensor ( data_controller, ene ):
  from .constants import LL

  arrays,attributes = data_controller.data_dicts()

  smearing = attributes['smearing']
  if smearing != None and smearing != 'gauss' and smearing != 'm-p':
    if rank == 0:
      print('%s Smearing Not Implemented.'%smearing)
    quit()

  d_tensor = arrays['d_tensor']

  for ispin in range(attributes['nspin']):
    for n in range(d_tensor.shape[0]):
      ipol = d_tensor[n][0]
      jpol = d_tensor[n][1]

      epsi,epsr,eels,jdos,ieps = do_epsilon(data_controller, ene, ispin, ipol, jpol)

      # Write files
      indices = (LL[ipol], LL[jpol], ispin)
      for ep,es in [(epsi,'epsi'),(epsr,'epsr'),(eels,'eels'),(jdos,'jdos'),(ieps,'ieps')]:
        fn = '%s_%s%s_%d.dat'%((es,)+indices)
        data_controller.write_file_row_col(fn, ene, ep)

      if rank == 0:
        renorm = np.sqrt((2./np.pi)*(ene[3]-ene[2])*np.sum(epsi*ene))
        print(ipol,jpol,' plasmon frequency = ',renorm,' eV')
        print(' integration over JDOS = ', (ene[3]-ene[2])*np.sum(jdos))


def do_epsilon ( data_controller, ene, ispin, ipol, jpol ):
  from .constants import EPS0, EVTORY, RYTOEV

  # Compute the dielectric tensor

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  if ene[0] == 0.:
    ene[0] = .00001

  #=======================
  # EPS
  #=======================
  epsi_aux,epsr_aux,jdos_aux,count_aux = eps_loop(data_controller, ene, ispin, ipol, jpol)

  ### TNeeds revision. Each processor is allocating zeros here, when only rank 0 needs it. 
  ### Can be condensed
  epsi = np.zeros(esize, dtype=float)
  comm.Allreduce(epsi_aux, epsi, op=MPI.SUM)
  epsi_aux = None

  epsr = np.zeros(esize, dtype=float)
  comm.Allreduce(epsr_aux, epsr, op=MPI.SUM)
  epsr_aux = None

  epsr_aux = epsr_kramerskronig(data_controller, ene, epsi)
  epsr0 = np.zeros(esize, dtype=float)
  comm.Allreduce(epsr_aux, epsr0, op=MPI.SUM)
  epsr_aux = None

  jdos = np.zeros(esize, dtype=float)
  comm.Allreduce(jdos_aux, jdos, op=MPI.SUM)
  jods_aux = None

  count = np.zeros(1,dtype=float)
  comm.Allreduce(count_aux, count, op=MPI.SUM)
  count_aux = None

  ieps = np.zeros(esize, dtype=float)

  kq_wght = 1./attributes['nkpnts']
  epsi *= 64.0*np.pi*kq_wght/(attributes['omega'])
  # includes correction for apparent rigid shift of epsr - solved by getting the right e -> 0 limit from KK.
  if not attributes['metal']:
    epsr =  1. + epsr*64.0*np.pi/(attributes['omega']*attributes['nkpnts']) - (epsr[4]-epsr0[4])*64.0*np.pi/(attributes['omega']*attributes['nkpnts'])
  else:
    epsr =  1. + epsr*64.0*np.pi/(attributes['omega']*attributes['nkpnts']) 
  eels = epsi/(epsi**2+epsr**2)
  for i in range(esize):
    for j in range(1,esize):
        ieps[i] += ene[j]*epsi[j]/(ene[i]**2+ene[j]**2)
  ieps = 1.0 + (2./np.pi)*ieps*(ene[3]-ene[2])
  jdos /= (4.*count[0])

  return(epsi, epsr, eels, jdos, ieps)


def eps_loop ( data_controller, ene, ispin, ipol, jpol):
  from .constants import EPS0, EVTORY, RYTOEV, BOHR_RADIUS_ANGS
  from .smearing import intgaussian,gaussian,intmetpax,metpax

### What is this?
  orig_over_err = np.geterr()['over']
  np.seterr(over='raise')

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  bnd = attributes['bnd']
  temp = attributes['temp']
  delta = attributes['delta']
  snktot = arrays['pksp'].shape[0]
  smearing = attributes['smearing']

  Ef = 0.
  eps=1.e-8
  kq_wght = 1./attributes['nkpnts']

  jdos = np.zeros(esize, dtype=float)
  epsi = np.zeros(esize, dtype=float)
  epsr = np.zeros(esize, dtype=float)

  fn = None
  if smearing == None:
    fn = 2.*1./(1.+np.exp(arrays['E_k'][:,:bnd,ispin]/temp))
  elif smearing == 'gauss':
    fn = 2.*intgaussian(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])
  elif smearing == 'm-p':
    fn = 2.*intmetpax(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])

  # apparently there are numerical instabilities if energy levels are not completely occupied or completely empty - needs to be tested for metals
  if not attributes['metal']:
    fn = np.round(fn,0)
  count = np.zeros(1,dtype=float)

  bndmax = bnd

  for ik in range(fn.shape[0]):
    for iband2 in range(bndmax):
       for iband1 in range(bndmax):
          if iband1 != iband2:
             E_diff_nm = arrays['E_k'][ik,iband2,ispin] - arrays['E_k'][ik,iband1,ispin]
             f_nm =  fn[ik,iband2]-fn[ik,iband1]
             if np.abs(f_nm) > 2.e-3 and fn[ik,iband1] > 1.e-4 and fn[ik,iband2] < 2.0:
                pksp2 = np.real(arrays['pksp'][ik,ipol,iband1,iband2,ispin]*arrays['pksp'][ik,jpol,iband2,iband1,ispin])
                pksp2 *= attributes['alat']*BOHR_RADIUS_ANGS/(EPS0*RYTOEV)
                epsi[:] +=  pksp2*delta*ene[:]*fn[ik,iband1]/(((E_diff_nm**2-ene[:]**2)**2+delta**2*ene[:]**2)*(E_diff_nm))
                epsr[:] +=  pksp2*(E_diff_nm**2-ene[:]**2)*fn[ik,iband1]/(((E_diff_nm**2-ene[:]**2)**2+delta**2*ene[:]**2)*(E_diff_nm))
                jdos[:] +=  delta*(fn[ik,iband1]-fn[ik,iband2])/(np.pi*((E_diff_nm-ene[:])**2+delta**2))
                count[0] += (fn[ik,iband1]-fn[ik,iband2])

  if attributes['metal']:
    if rank == 0: print('NOT TESTED - needs different delta for intraband transitions and degauss from QE + check on units!!!')
    degauss=0.05
    fnF = None
    if smearing is None:
      fnF = np.empty((snktot,bnd), dtype=float)
      for n in range(bnd):
        for i in range(snktot):
          try:
            fnF[i,n] = .5/(1.+np.cosh(arrays['E_k'][i,n,ispin]/temp))
          except:
            fnF[i,n] = 1e8
      fnF /= temp
    elif smearing == 'gauss':
 ## Why .03* here?
      fnF = gaussian(arrays['E_k'][:,:bnd,ispin], Ef, .03*arrays['deltakp'][:,:bnd,ispin])
    elif smearing == 'm-p':
      fnF = metpax(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])

    for ik in range(fn.shape[0]):
      for iband1 in range(bndmax):
        pksp2 = np.real(arrays['pksp'][ik,ipol,iband1,iband1,ispin]*arrays['pksp'][ik,jpol,iband1,iband1,ispin])
        pksp2 *= attributes['alat']*BOHR_RADIUS_ANGS/(EPS0*RYTOEV**3)
        epsi[:] +=  pksp2*delta*ene[:]*fnF[ik,iband1]/((ene[:]**4+delta**2*ene[:]**2)*degauss)
        epsr[:] -=  pksp2*fnF[ik,iband1]*ene[:]**2/((ene[:]**4+delta**2*ene[:]**2)*degauss)

  np.seterr(over=orig_over_err)
  return(epsi, epsr, jdos, count)


def epsr_kramerskronig ( data_controller, ene, epsi ):
  from .smearing import intmetpax
  from scipy.integrate import simps
  from .load_balancing import load_balancing

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
    I1 = simps(ene[1:(ie-1)]*de*epsi[1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2))
    I2 = simps(ene[(ie+1):esize]*de*epsi[(ie+1):esize]*f_ene[(ie+1):esize]/(ene[(ie+1):esize]**2-ene[ie]**2))
    epsr[ie] = 2.*(I1+I2)/np.pi

  return epsr
