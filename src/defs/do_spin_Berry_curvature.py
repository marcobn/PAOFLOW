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

def do_spin_Berry_curvature ( data_controller, jksp, ipol, jpol ):
  import numpy as np
  from mpi4py import MPI
  from communication import gather_full
  from smearing import intgaussian, intmetpax

  rank = MPI.COMM_WORLD.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  #----------------------
  # Compute spin Berry curvature
  #----------------------

  snktot,_,bnd,_,nspin = jksp.shape
  fermi_dw,fermi_up = attributes['fermi_dw'],attributes['fermi_up']
  nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']

  # Compute only Omega_z(k)
  Om_znkaux = np.zeros((snktot,bnd), dtype=float)

  deltap = 0.05
  for n in range(bnd):
    for m in range(bnd):
      if m != n:
        Om_znkaux[:,n] += -2.0*np.imag(jksp[:,ipol,n,m,0]*arrays['pksp'][:,jpol,m,n,0]) / \
        ((arrays['E_k'][:,m,0] - arrays['E_k'][:,n,0])**2 + deltap**2)

  de = (attributes['emaxSH']-attributes['eminSH'])/500
  ene = np.arange(attributes['eminSH'], attributes['emaxSH'], de)
  esize = ene.size

  Om_zkaux = np.zeros((snktot,esize), dtype=float)

  for i in range(esize):
    if attributes['smearing'] == 'gauss':
      Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intgaussian(arrays['E_k'][:,:bnd,0],ene[i],arrays['deltakp'][:,:bnd,0]), axis=1)
    elif attributes['smearing'] == 'm-p':
      Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*intmetpax(arrays['E_k'][:,:bnd,0],ene[i],arrays['deltakp'][:,:bnd,0]), axis=1)
    else:
      Om_zkaux[:,i] = np.sum(Om_znkaux[:,:]*(0.5 * (-np.sign(arrays['E_k'][:,:bnd,0]-ene[i]) + 1)), axis=1)

  Om_zk = gather_full(Om_zkaux, attributes['npool'])
  Om_zk_aux = None

  shc = None
  if rank == 0:
    shc = np.sum(Om_zk, axis=0)/float(attributes['nkpnts'])

  n0 = 0
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
