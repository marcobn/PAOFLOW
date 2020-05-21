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
  arrays['tau_t'] = get_tau(temp,data_controller,['impurity','accoustic','polar optical'])

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
  comm.Reduce(L1aux, L1, op=MPI.SUM)
  L1aux = None


  L2aux, tau_aux, norm_aux = fLloop(2)
  L2 = zol(rank,L2aux)
  comm.Reduce(L2aux, L2, op=MPI.SUM)
  L2aux = None
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
  arrays['tau_t'] = get_tau(temp,data_controller,['impurity','accoustic','polar optical'])

  t_tensor = arrays['t_tensor']
  L0aux, t, n = L_loop(data_controller, temp, attributes['smearing'], ene, velkp, t_tensor, 0, ispin)
  L0 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None
  
  return L0

def fermi(E,temp,Ef):

    return 1./(np.exp((E-Ef)/temp)+1.)

def planck(hwlo,temp):
    
    return 1/(np.exp(hwlo/temp)-1)

def get_tau (temp,data_controller, channels ):

  import numpy as np
  import scipy.constants as cp
  arry,attr = data_controller.data_dicts()
  hbar = cp.hbar
  kb = cp.Boltzmann
  temp *= 1.60217662e-19
  nd = attr['doping_conc']*1e6 #doping in /m^3
  snktot = arry['E_k'].shape[0]
  nspin = arry['E_k'].shape[2]
  bnd = attr['bnd']
  rate = []
  e = 1.60217662e-19
  ev2j= 1.60217662e-19
  epso = 8.854187817e-12
  tpi = np.pi*2
  fpi = np.pi*4
  me = 9.10938e-31
  E = abs(ev2j*(arry['E_k'][:,:bnd]))
  Ef = 0.04*ev2j
  D = 6.5*ev2j
  rho = 3.9375e3   #kg/m^3 
  a = 8.6883e-10 #metres    
  nd = attr['doping_conc']*1e6 #doping in /m^3
  nI = nd #no.of impuritites/m^3
  eps_inf = 14.2*epso
  eps_0 = 26.7*epso
  eps = eps_inf+eps_0
  eps_inv = 1/eps_inf - 1/eps
  v = 2.7e3
  Zi = 1.
  ms = 0.3*me*np.ones((snktot,bnd,nspin), dtype=float) #effective mass tensor in kg 
  hwlo = ev2j * np.array([0.0205,0.0248,0.031])
  for c in channels:

      if c == 'impurity':
          qo = np.sqrt(e**2*nI/(eps*temp))
          x = (hbar*qo)**2/(8*ms*E)
          P_imp = (np.pi*nI*Zi**2*(e**4)/(E**1.5*np.sqrt(2*ms)*(fpi*eps)**2))
          P_imp *= (np.log(1+1./x)-1./(1+x))
          rate.append(P_imp)

      if c == 'accoustic':
          P_ac = ((2*ms)**1.5*(D**2)*np.sqrt(E)*temp)/(tpi*hbar**4*rho*v**2)
          rate.append(P_ac)

      if c == 'polar optical':
	  P_pol=0.
          for i in range(len(hwlo)):
            ff = fermi(E+hwlo[i],temp,Ef)
            fff= fermi(E-hwlo[i],temp,Ef)
            f = fermi(E,temp,Ef)
            n = planck(hwlo[i],temp)
            nn = planck(hwlo[i]+1,temp)
            Wo = e**2/(fpi*hbar)*np.sqrt(2*ms*hwlo[i]/hbar**2)*eps_inv
            Z = 2/(Wo*np.sqrt(hwlo[i]))
            A = (n+1)*ff/f*((2*E+hwlo[i])*np.arcsinh(np.sqrt(E/hwlo[i]))-np.sqrt(E*(E+hwlo[i])))
            B = np.heaviside(E-hwlo[i],1)*n*fff/f*((2*E-hwlo[i])*np.arccosh(np.sqrt(E/hwlo[i]))-np.sqrt(E*(E-hwlo[i])))  
            where_are_NaNs = np.isnan(B)
            B[where_are_NaNs] = 0
            C = (n+1)*ff/f*np.arcsinh(np.sqrt(E/hwlo[i]))
            t2 = np.heaviside(E-hwlo[i],1)*n*fff/f*np.arccosh(np.sqrt(E/hwlo[i]))
            where_are_NaNs = np.isnan(t2)
            t2[where_are_NaNs] = 0 
            C = (2*E)*(C+t2)
            P = (C-A-B)/(Z*(E**1.5))
            P_pol += P            
          rate.append(P_pol)
      
      if c == None:
	tau = np.ones((snktot,bnd,nspin), dtype=float)

      tau = np.zeros((snktot,bnd,nspin), dtype=float)
      for r in rate:
          tau += r
      tau = 1/tau
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
