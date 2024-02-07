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
import scipy.optimize 
import scipy.integrate

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _fd_criterion_gen(threshold):

    def _fd_criterion(x):
        return 1. / (np.exp(x) + 1.) - threshold

    return _fd_criterion

def FD(ene,mu,temp):
  
  _FD_THRESHOLD = 1e-8
  _FD_XMAX = scipy.optimize.newton(_fd_criterion_gen(_FD_THRESHOLD), 0.)

  temp_ev = temp*8.617332478e-5
  if temp == 0.:
    dela = ene-mu
    nruter = np.where(dela < 0., 1., 0.)
    nruter[np.isclose(dela, 0.)] = .5
  else:
    x = (ene-mu) / temp_ev
    nruter = np.where(x < 0., 1., 0.)
    indices = np.logical_and(x > -_FD_XMAX, x < _FD_XMAX)
    nruter[indices] = 1. / (np.exp(x[indices]) + 1.)
  return nruter

def calc_N(data_controller,ene,dos, mu, temp, dosweight=2.):

  arry,attr = data_controller.data_dicts()
  core_electrons = attr['core_electrons']

  if temp == 0.: 
    occ = np.where(ene < mu, 1., 0.)
    occ[ene==mu] = .5
  else:
    occ = FD(ene, mu, temp)
  dos_occ = dos*occ
  return -dosweight * scipy.integrate.simps(dos_occ,ene)-core_electrons
  
def solve_for_mu(data_controller,ene,dos,N0,temp,refine=False,try_center=False,dosweight=2.):

  _FD_THRESHOLD_GAP = 1e-3
  _FD_XMAX_GAP = scipy.optimize.newton(_fd_criterion_gen(_FD_THRESHOLD_GAP), 0.)

  dela = np.empty_like(ene)

  if rank ==0:
    for i,e in enumerate(ene):
      dela[i] = (calc_N(data_controller,ene, dos, e, temp, dosweight)) + N0
  
    dela = np.abs(dela)
    pos = dela.argmin()
    mu = ene[pos]
    center = False
########################################
#checking if dos is zero takes care not to include band gaps in the integral calculation
#######################################

    if dos[pos] == 0.:
      lpos = -1
      hpos = -1
      for i in range(pos, -1, -1):
        if dos[i] != 0.:
          lpos = i
          break
      for i in range(pos, dos.size):
        if dos[i] != 0.:
           hpos = i
           break
      if -1 in (lpos, hpos):
        raise ValueError("mu0 lies outside the range of band energies")
      hene = ene[hpos]
      lene = ene[lpos]
      if (try_center and min(hene - mu, mu - lene) >= _FD_XMAX_GAP * temp / 2.):
          pos = int(round(.5 * (lpos + hpos)))
          mu = ene[pos]
          center = True
    if refine:
      if center:
        mu = .5 * (lene + hene)
      else:
        residual = calc_N(data_controller,ene, dos, mu, temp, dosweight) + N0
        if np.isclose(residual, 0):
           lpos = pos
           hpos = pos
        elif residual > 0:
           lpos = pos
           hpos = min(pos + 1, ene.size - 1)
        else:
           lpos = max(0, pos - 1)
           hpos = pos
        if hpos != lpos:
           lmu = ene[lpos]
           hmu = ene[hpos]

           def calc_abs_residual(muarg):
              return abs(calc_N(data_controller,ene, dos, muarg, temp, dosweight) + N0)
           result = scipy.optimize.minimize_scalar(calc_abs_residual,bounds=(lmu, hmu),method="bounded")
           mu = result.x
    return mu

def do_doping( data_controller, temps, ene, fname ):

  arry,attr = data_controller.data_dicts()
  temp_conv,omega_conv = 11604.52500617,1.481847093e-25

  if attr['smearing'] is None:
    dos = arry['dos']
  else:
    dos = arry['dosdk']

  doping = attr['doping_conc']
  nelec,omega = attr['nelec'],attr['omega']*omega_conv

  margin = 9. * temps.max()
  mumin = ene.min() + margin
  mumax = ene.max() - margin
  nT = len(temps)
  mu = np.empty(nT)
  dos_int = np.empty(nT)
  msize = mu.size
  Nc = np.empty(nT)
  N = nelec - doping * omega

  for iT,temp in enumerate(temps):

    itemp = temp/temp_conv

    if rank == 0:
      mu[iT] = solve_for_mu(data_controller,ene,dos,N,temp,refine=True,try_center=True)

    mu[iT] = comm.bcast(mu[iT], root=0)

  fdope = '%s%s%s.dat'%(fname, 'n' if doping<0 else 'p', np.abs(doping))
  data_controller.write_file_row_col(fdope, temps, mu)

    
