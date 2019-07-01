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

def do_Boltz_tensors_no_smearing ( data_controller, temp, ene, velkp, ispin ):
  # Compute the L_alpha tensors for Boltzmann transport

  arrays,attributes = data_controller.data_dicts()
  esize = ene.size
  arrays['tau_t'] = get_tau(temp,data_controller,['impurity','accoustic','optical','intervalley','polar optical'])

#### Forced t_tensor to have all components
  t_tensor = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]], dtype=int)

  # Quick call function for L_loop (None is smearing type)
  fLloop = lambda spol : L_loop(data_controller, temp, None, ene, velkp, t_tensor, spol, ispin)

  # Quick call function for Zeros on rank Zero
  zol = lambda r,l: (np.zeros_like(l) if r==0 else None)

  L0aux, tau_aux, norm_aux = fLloop(0)
  L0 = zol(rank, L0aux) 
  tau = zol(rank, tau_aux) 
  norm = zol(rank, norm_aux) 
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  comm.Reduce(tau_aux, tau, op=MPI.SUM)
  comm.Reduce(norm_aux, norm, op=MPI.SUM)
  L0aux = norm_aux = tau_aux = None
  if rank == 0:
    arrays['tau_avg'] = []
    arrays['tau_avg'].append(tau/norm)
    arrays['tau_avg'] = np.array(arrays['tau_avg'])

  L1aux, tau_aux, norm_aux = fLloop(1)
  L1 = zol(rank,L1aux)
  #tau = zol(rank, tau_aux) 
  #norm = zol(rank, norm_aux) 
  comm.Reduce(L1aux, L1, op=MPI.SUM)
  #comm.Reduce(tau_aux, tau, op=MPI.SUM)
  #comm.Reduce(norm_aux, norm, op=MPI.SUM)
  #L1aux = norm_aux = tau_aux = None
  L1aux = None
  #if rank == 0:
  #  arrays['tau_avg'].append(tau/norm)


  L2aux, tau_aux, norm_aux = fLloop(2)
  L2 = zol(rank,L2aux)
  #tau = zol(rank, tau_aux) 
  #norm = zol(rank, norm_aux) 
  comm.Reduce(L2aux, L2, op=MPI.SUM)
  #comm.Reduce(tau_aux, tau, op=MPI.SUM)
  #comm.Reduce(norm_aux, norm, op=MPI.SUM)
  #L2aux =  norm_aux = tau_aux = None
  L2aux = None
  #if rank == 0:
    #arrays['tau_avg'].append(tau/norm)
    #arrays['tau_avg'] = np.array(arrays['tau_avg'])
  tau = norm = None

  if rank == 0:
    # Assign lower triangular to upper triangular
    sym = lambda L : (L[0,1], L[0,2], L[1,2])
    L0[1,0],L0[2,0],L0[2,1] = sym(L0)
    L1[1,0],L1[2,0],L1[2,1] = sym(L1)
    L2[1,0],L2[2,0],L2[2,1] = sym(L2)

  return (L0, L1, L2) if rank==0 else (None, None, None)


# Compute the L_0 tensor for Boltzmann Transport with Smearing
def do_Boltz_tensors_smearing ( data_controller, temp, ene, velkp, ispin ):

  arrays,attributes = data_controller.data_dicts()
  esize = ene.size
  arrays['tau_t'] = get_tau(temp,data_controller,['impurity','accoustic','optical','intervalley','polar optical'])

  t_tensor = arrays['t_tensor']
  L0aux, t, n = L_loop(data_controller, temp, attributes['smearing'], ene, velkp, t_tensor, 0, ispin)
  L0 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None
  
  return L0


def get_tau (temp,data_controller, channels ):

  import numpy as np
  import scipy.constants as cp
  h = cp.hbar
  kb = cp.Boltzmann
  hw0 = 0.02447*1.60217662e-19 #joules
  temp *= 1.60217662e-19
  e = 1.60217662e-19
  arry,attr = data_controller.data_dicts()
  snktot = arry['E_k'].shape[0]
  nspin = arry['E_k'].shape[2]
  bnd = attr['bnd']
  taus = []
  nd = 38e26
  nI = nd
  vl = 2.219e3
  vt = 4.438e3
  v = (2*vt+vl)/3
  kiv =  0.00042619750098771096
  kimp1 =  17800.88543819998318
  kimp2 =  300.125
  kac =  1218589.1650226777/3
  kop1 =  0.001564292955050779
  kop2 =  00.2557185005926266
  kpo =  25.15576475
  E = abs(1.60217662e-19*(arry['E_k'][:,:bnd])) 
  for c in channels: 

      if c == 'impurity':
          cf1 = (((e**2)*nd)/(8.854187817e-12*temp))*((h**2)/(2*9.10938e-31))  #every constant starting with cf are conversion factors
          cf2 = 16*np.pi*np.sqrt(2*9.10938e-31)*(8.854187817e-12**2)/((e**4)*nI)
          i_tau = (cf2*kimp1*(E**1.5))/((np.log(1+(4*E/(kimp2*cf1)))-((4*E/(kimp2*cf1))/(1+(4*E/(kimp2*cf1))))))
          taus.append(i_tau)

      if c == 'accoustic':
          cf3 = (2*np.pi*h**4*1e3)/((2*9.10938e-31)**1.5*1.60217662e-19**2)
          a_tau = (cf3*kac*np.power(E/(temp),(-0.5)))/((np.power(temp,1.5)))
          taus.append(a_tau)

      if c == 'optical':
          Nop = (temp/hw0)-0.5
          x = E/temp
          xo = hw0/temp
          X = x-xo
          X[X<0] = 0
          cf4 = (2/np.pi)*(h**2*1e-10**2*1e3)/((2*9.10938e-31)**1.5)
          cf5 = (np.sqrt(2)*np.pi*(1.60217662e-19/temp)*(h**2)*1e3)/((9.10938e-31**1.5)*1.60217662e-19*Nop)
          o_tau_no_inels = cf4*kop1*((E/temp)**-0.5)/(temp**1.5)
          o_tau = np.sqrt(temp)*cf5*kop2/(np.sqrt(x+xo)+np.sqrt(X))
          taus.append(o_tau)

      #if c == 'intervalley':
      #    Nop = (temp/hw0)-0.5
      #    x = E/temp
      #    xo = hw0/temp
      #    X = x-xo
      #    X[X<0] = 0
      #    cf6 = (np.sqrt(2)*np.pi*(1.60217662e-19/temp)*(h**2)*1e3)/((9.10938e-31**1.5)*1.60217662e-19*Nop)
      #    iv_tau =  np.sqrt(temp)*cf6*kiv/(np.sqrt(x+xo)+np.sqrt(X))
      #    taus.append(iv_tau)


      if c == 'polar optical':
          cf7 = (8.854187817e-12*h**2)/((np.power(2*9.10938e-31,0.5))*e**2)
          po_tau = (cf7*kpo*np.power(E/(temp),(0.5)))/(np.power(temp,0.5))
          taus.append(po_tau)

      tau = np.zeros((snktot,bnd,nspin), dtype=float)
      for t in taus:
          tau += 1./t
      tau = 1/tau
  #tau_new = np.reshape(tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
 # o_tau_new = np.reshape(o_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #o_tau_new_2 = np.reshape(o_tau_no_inels,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #a_tau_new = np.reshape(a_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #i_tau_new = np.reshape(i_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #iv_tau_new = np.reshape(iv_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #po_tau_new = np.reshape(po_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #E_re = np.reshape(E/1.60217662e-19,(snktot,bnd)) #i do this because i am not able to saave 3d arrays to a file
  #np.savetxt('tau.dat',tau_new)
  #np.savetxt('o_tau.dat',o_tau_new)
  #np.savetxt('o_tau_no_inels.dat',o_tau_new_2)
  #np.savetxt('a_tau.dat',a_tau_new)
  #np.savetxt('i_tau.dat',i_tau_new)
  #np.savetxt('iv_tau.dat',iv_tau_new)
  #np.savetxt('po_tau.dat',po_tau_new)
  #np.savetxt('E.dat',E_re)
  return tau

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
  tau_avg = np.zeros((3,3,esize), dtype=float)
  Nm = np.zeros((3,3,esize), dtype=float)
  for n in range(bnd):
    Eaux = np.reshape(np.repeat(arrays['E_k'][:,n,ispin],esize), (snktot,esize))
    tau_re = np.reshape(np.repeat(arrays['tau_t'][:,n,0],bnd), (snktot,bnd))
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
      L[i,j,:] += np.sum(kq_wght*velkp[:,i,n,ispin]*tau_re[:,n]*velkp[:,j,n,ispin]*(smearA*EtoAlpha).T, axis=1)
      tau_avg += np.sum(kq_wght*tau_re[:,n]*(smearA*EtoAlpha).T, axis=1)
      Nm += np.sum(kq_wght*(smearA*EtoAlpha).T, axis=1)
  return(L, tau_avg, Nm)
