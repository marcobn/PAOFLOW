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

def do_Boltz_tensors ( data_controller, smearing, temp, ene, velkp, ispin ):
  # Compute the L_alpha tensors for Boltzmann transport

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size

#### Forced t_tensor to have all components
  t_tensor = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]], dtype=int)

  # Quick call function for L_loop (None is smearing type)
  fLloop = lambda spol : L_loop(data_controller, temp, smearing, ene, velkp, t_tensor, spol, ispin)

  # Quick call function for Zeros on rank Zero
  zoz = lambda r: (np.zeros((3,3,esize), dtype=float) if r==0 else None)

  L0 = zoz(rank)
  L0aux = fLloop(0)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None

  if rank == 0:
    # Assign lower triangular to upper triangular
    sym = lambda L : (L[0,1], L[0,2], L[1,2])
    L0[1,0],L0[2,0],L0[2,1] = sym(L0)

  L1 = L2 = None
  if smearing is None:

    L1 = zoz(rank)
    L1aux = fLloop(1)
    comm.Reduce(L1aux, L1, op=MPI.SUM)
    L1aux = None

    L2 = zoz(rank)
    L2aux = fLloop(2)
    comm.Reduce(L2aux, L2, op=MPI.SUM)
    L2aux = None

    if rank == 0:
      L1[1,0],L1[2,0],L1[2,1] = sym(L1)
      L2[1,0],L2[2,0],L2[2,1] = sym(L2)

  return (L0, L1, L2) if rank==0 else (None, None, None)


def get_tau ( data_controller, channels ):
  import numpy as np

  arry,attr = data_controller.data_dicts()
  snktot = arry['E_k'].shape[0]

  taus = []
  for c in channels:
    if c == 'acoustic':
      a_tau = np.ones((snktot), dtype=float)
      taus.append(a_tau)

    if c == 'optical':
      o_tau = np.ones((snktot), dtype=float)
      taus.append(o_tau)

  tau = np.zeros((snktot), dtype=float)
  for t in taus:
    tau += 1./t
  return len(channels)/tau


def L_loop ( data_controller, temp, smearing, ene, velkp, t_tensor, alpha, ispin ):
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
      tau = get_tau(data_controller, ['acoustic', 'optical'])
      L[i,j,:] += np.sum(kq_wght*tau*velkp[:,i,n,ispin]*velkp[:,j,n,ispin]*(smearA*EtoAlpha).T, axis=1)

  return L


def do_Hall_tensors( E_k_range,velkp_range,d2Ed2k_range,kq_wght,temp,ene ):
  # calculated components of the Hall tensor R_ijk


  if rank==0:
      R = np.zeros((3,3,3,ene.size),dtype=float,order="C")
  else: R = None

  Raux = H_loop(ene,E_k_range,velkp_range,d2Ed2k_range,kq_wght,temp)
  comm.Reduce(Raux,R,op=MPI.SUM)

  return(R)



def H_loop(ene,E_k,velkp,d2Ed2k,kq_wght,temp):
  # We assume tau=1 in the constant relaxation time approximation

  R = np.zeros((3,3,3,ene.size),dtype=float,order="C")

  # mapping of the unique 2nd rank tensor components
  d2Ed2k_ind = np.array([[0,3,5],
                       [3,1,4],
                       [5,4,2]],dtype=int,order="C")

  # precompute the 6 unique v_i*v_j
  v2=np.zeros((6,E_k.shape[0]),dtype=float,order="C")
  ij_ind = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]],dtype=int)

  for l in range(ij_ind.shape[0]):
      i     = ij_ind[l][0]
      j     = ij_ind[l][1]
      v2[l] = velkp[i]*velkp[j]

  # precompute sig_xyz
  sig_xyz=np.zeros((3,3,3,E_k.shape[0]),order="C")    
  for a in range(3):
      for b in range(3):
          if a==b and b==0: continue
          sig_xyz[a,b,0] = d2Ed2k[d2Ed2k_ind[b,1]]*v2[d2Ed2k_ind[a,2]] - \
                           d2Ed2k[d2Ed2k_ind[b,2]]*v2[d2Ed2k_ind[a,1]]
  for a in range(3):
      for b in range(3):                          
          if a==b and b==1: continue
          sig_xyz[a,b,1] = d2Ed2k[d2Ed2k_ind[b,2]]*v2[d2Ed2k_ind[a,0]] - \
                           d2Ed2k[d2Ed2k_ind[b,0]]*v2[d2Ed2k_ind[a,2]]
  for a in range(3):
      for b in range(3):                          
          if a==b and b==2: continue 
          sig_xyz[a,b,2] = d2Ed2k[d2Ed2k_ind[b,0]]*v2[d2Ed2k_ind[a,1]] - \
                           d2Ed2k[d2Ed2k_ind[b,1]]*v2[d2Ed2k_ind[a,0]]


  for n in range(ene.shape[0]):        
      # dfermi(e-e_f)/de
      Eaux = 1.0/(1.0+np.cosh((E_k-ene[n])/temp))
      for a in range(3):
          for b in range(3):
              for g in range(3):                    
                  if a==b and b==g: continue                                           
                  R[a,b,g,n] = np.sum(sig_xyz[a,b,g]*Eaux)

  return(R*kq_wght[0]*0.5/temp)


