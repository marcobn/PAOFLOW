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

def do_spin_Berry_curvature ( data_controller, jksp, pksp ):
  import numpy as np
  from mpi4py import MPI
  from .communication import gather_full
  from .smearing import intgaussian, intmetpax

  rank = MPI.COMM_WORLD.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  #----------------------
  # Compute spin Berry curvature
  #----------------------

  snktot,nawf,_,nspin = pksp.shape
  fermi_up,fermi_dw = attributes['fermi_up'],attributes['fermi_dw']
  nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']

  # Compute only Omega_z(k)
  Om_znkaux = np.zeros((snktot,nawf), dtype=float)

  deltap = 0.05
  for ik in range(snktot):
    E_nm = (arrays['E_k'][ik,:,0] - arrays['E_k'][ik,:,0][:,None])**2
    E_nm[np.where(E_nm<1.e-4)] = np.inf
    Om_znkaux[ik] = -2.0*np.sum(np.imag(jksp[ik,:,:,0]*pksp[ik,:,:,0].T)/E_nm, axis=1)
  E_nm = None

  attributes['emaxH'] = np.amin(np.array([attributes['shift'],attributes['emaxH']]))
  de = (attributes['emaxH']-attributes['eminH'])/500
  ene = np.arange(attributes['eminH'], attributes['emaxH'], de)
  esize = ene.size

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
  Om_k = None
  if rank == 0:
    Om_k = np.zeros((nk1,nk2,nk3,esize), dtype=float)
    for i in range(esize-1):
      if ene[i] <= fermi_dw and ene[i+1] >= fermi_dw:
        n0 = i
      if ene[i] <= fermi_up and ene[i+1] >= fermi_up:
        n = i
    Om_k = np.reshape(Om_zk, (nk1,nk2,nk3,esize), order='C')
    Om_k = Om_k[:,:,:,n]-Om_k[:,:,:,n0]

  return(ene, shc, Om_k)
