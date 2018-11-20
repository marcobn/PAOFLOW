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


def do_Boltz_tensors_no_smearing ( data_controller, temp, ene, velkp, ispin ):
  import numpy as np
  from mpi4py import MPI
  # Compute the L_alpha tensors for Boltzmann transport

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size

  smearing = None

#  t_tensor = arrays['t_tensor']
  t_tensor = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]], dtype=int)

  L0aux = L_loop(data_controller, temp, smearing, ene, velkp, t_tensor, ispin, 0)
  L0 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None

  L1aux = L_loop(data_controller, temp, smearing, ene, velkp, t_tensor, ispin, 1)
  L1 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L1aux, L1, op=MPI.SUM)
  L1aux = None

  L2aux = L_loop(data_controller, temp, smearing, ene, velkp, t_tensor, ispin, 2)
  L2 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L2aux, L2, op=MPI.SUM)
  L2aux = None

  if rank == 0:
    L0[1,0] = L0[0,1]
    L0[2,0] = L0[2,0]
    L0[2,1] = L0[1,2]

    L1[1,0] = L1[0,1]
    L1[2,0] = L1[2,0]
    L1[2,1] = L1[1,2]

    L2[1,0] = L2[0,1]
    L2[2,0] = L2[2,0]
    L2[2,1] = L2[1,2]

    return(L0, L1, L2)

  else:
    return(None, None, None)


def do_Boltz_tensors_smearing ( data_controller, temp, ene, velkp, ispin ):
  import numpy as np
  from mpi4py import MPI
  # Compute the L_0 tensor for Boltzmann Transport with Smearing

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  t_tensor = arrays['t_tensor']

  esize = ene.size

  L0aux = L_loop(data_controller, temp, attributes['smearing'], ene, velkp, t_tensor, ispin, 0)
  L0 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None

  return L0


def L_loop ( data_controller, temp, smearing, ene, velkp, t_tensor, ispin, alpha ):
  import numpy as np
  from .smearing import gaussian,metpax
  # We assume tau=1 in the constant relaxation time approximation

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size

  snktot = arrays['E_k'].shape[0]

  bnd = attributes['bnd']
  kq_wght = 1./attributes['nkpnts']

  if smearing is not None and smearing != 'gauss' and smearing != 'm-p':
    print('%s Smearing Not Implemented.'%smearing)
    comm.Abort()

  L = np.zeros((3,3,esize), dtype=float)

  for n in range(bnd):
    Eaux = np.reshape(np.repeat(arrays['E_k'][:,n,ispin],esize), (snktot,esize))
    delk = (np.reshape(np.repeat(arrays['deltakp'][:,n,ispin],esize), (snktot,esize)) if smearing!=None else None)
    EtoAlpha = np.power(Eaux[:,:]-ene, alpha)
    if smearing is None:
      Eaux -= ene
      smearA = .5/(temp*(1.+.5*(np.exp(Eaux/temp)+np.exp(-Eaux/temp))))
    else:
      if smearing == 'gauss':
        smearA = gaussian(Eaux, ene, delk)
      elif smearing == 'm-p':
        smearA = metpax(Eaux, ene, delk)
    for l in range(t_tensor.shape[0]):
      i = t_tensor[l][0]
      j = t_tensor[l][1]
      L[i,j,:] += np.sum(kq_wght*velkp[:,i,n,ispin]*velkp[:,j,n,ispin]*(smearA*EtoAlpha).T, axis=1)

  return L
