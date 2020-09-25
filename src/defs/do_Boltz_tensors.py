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

def do_Boltz_tensors_no_smearing (data_controller, temp, ene, velkp, ispin,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac):
  # Compute the L_alpha tensors for Boltzmann transport

  arrays,attributes = data_controller.data_dicts()
  esize = ene.size
  arrays['tau_t'] = get_tau(temp,data_controller,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac)

#### Forced t_tensor to have all components
  t_tensor = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]], dtype=int)

  # Quick call function for L_loop (None is smearing type)
  fLloop = lambda spol : L_loop(data_controller, temp, None, ene, velkp, t_tensor, spol, ispin)

  # Quick call function for Zeros on rank Zero
  zol = lambda r,l: (np.zeros_like(l) if r==0 else None)

  L0aux = fLloop(0)
  L0 = zol(rank, L0aux) 
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux =  None

  L1aux = fLloop(1)
  L1 = zol(rank,L1aux)
  comm.Reduce(L1aux, L1, op=MPI.SUM)
  L1aux = None

  L2aux = fLloop(2)
  L2 = zol(rank,L2aux)
  comm.Reduce(L2aux, L2, op=MPI.SUM)
  L2aux = None

  if rank == 0:
    # Assign lower triangular to upper triangular
    sym = lambda L : (L[0,1], L[0,2], L[1,2])
    L0[1,0],L0[2,0],L0[2,1] = sym(L0)
    L1[1,0],L1[2,0],L1[2,1] = sym(L1)
    L2[1,0],L2[2,0],L2[2,1] = sym(L2)

  return (L0, L1, L2) if rank==0 else (None, None, None)


# Compute the L_0 tensor for Boltzmann Transport with Smearing
def do_Boltz_tensors_smearing ( data_controller, temp, ene, velkp, ispin,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac):

  arrays,attributes = data_controller.data_dicts()
  esize = ene.size
  arrays['tau_t'] = get_tau(temp,data_controller,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac)

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

def get_tau (temp,data_controller,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac):

  import numpy as np
  import scipy.constants as cp
  arry,attr = data_controller.data_dicts()
  channels = attr['scattering_channels']
  nd = attr['doping_conc']*1e6 #doping in /m^3
  snktot = arry['E_k'].shape[0]
  bnd = attr['bnd']
  nspin = arry['E_k'].shape[2]
  
  if channels == None:
    tau = np.ones((snktot,bnd,nspin), dtype=float) #constant relaxation time approximation with tau = 1

  else:
    hbar = cp.hbar
    kb = cp.Boltzmann
    rate = []
    e = 1.60217662e-19
    ev2j= 1.60217662e-19
    epso = 8.854187817e-12
    tpi = np.pi*2
    fpi = np.pi*4
    me = 9.10938e-31
    temp *= ev2j 
    E = abs(arry['E_k'][:,:bnd])*ev2j
    Ef = abs(attr['tau_dict']['Ef'])*ev2j #fermi energy
    D_ac = attr['tau_dict']['D_ac']*ev2j #acoustic deformation potential in J
    rho = attr['tau_dict']['rho']   #mass density kg/m^3 
    a = attr['tau_dict']['a'] # lattice constant metres    
    piezo = attr['tau_dict']['piezo']  #piezoelectric constant
    eps_inf = attr['tau_dict']['eps_inf']*epso #high freq dielectirc const
    eps_0 = attr['tau_dict']['eps_0']*epso #low freq dielectric const
    v = attr['tau_dict']['v'] #velocity in m/s
    Zi = attr['tau_dict']['Zi'] #number of charge units of impurity
    ms = ms*me*np.ones((snktot,bnd,nspin), dtype=float) #effective mass tensor in kg 
    hwlo = np.array(attr['tau_dict']['hwlo'])*ev2j #phonon freq
    Zf = attr['tau_dict']['Zf'] #number of equivalent valleys if considering interevalley scattering
    D_op = attr['tau_dict']['D_op']*ev2j #optical deformation potential in J/m
    nI = attr['tau_dict']['nI']*1e6
    for c in channels: 

      if c == 'impurity':                                       
          eps = eps_inf+eps_0
          qo = np.sqrt(e**2*nI/(eps*temp))
          x = (hbar*qo)**2/(8*ms*E)
          P_imp = (np.pi*nI*Zi**2*(e**4)/(E**1.5*np.sqrt(2*ms)*(fpi*eps)**2))   #formula from fiorentini paper on Mg3Sb2
          P_imp *= (np.log(1+1./x)-1./(1+x))
          rate.append(P_imp/a_imp)

      if c == 'acoustic':
          P_ac = ((2*ms)**1.5*(D_ac**2)*np.sqrt(E)*temp)/(tpi*hbar**4*rho*v**2) #formula from fiorentini paper on Mg3Sb2
          rate.append(P_ac/a_ac)

      if c == 'polar optical':
	  P_pol=0.
          eps = eps_inf+eps_0
          eps_inv = 1/eps_inf - 1/eps
          for i in range(len(hwlo)):
            ff = fermi(E+hwlo[i],temp,Ef)
            fff= fermi(E-hwlo[i],temp,Ef)
            f = fermi(E,temp,Ef)
            n = planck(hwlo[i],temp)
            nn = planck(hwlo[i]+1,temp)
            Wo = e**2/(fpi*hbar)*np.sqrt(2*ms*hwlo[i]/hbar**2)*eps_inv
            Z = 2/(Wo*np.sqrt(hwlo[i]))
            A = (n+1)*ff/f*((2*E+hwlo[i])*np.arcsinh(np.sqrt(E/hwlo[i]))-np.sqrt(E*(E+hwlo[i])))
            where_are_NaNs = np.isnan(A)
            A[where_are_NaNs] = 0
            B = np.heaviside(E-hwlo[i],1)*n*fff/f*((2*E-hwlo[i])*np.arccosh(np.sqrt(E/hwlo[i]))-np.sqrt(E*(E-hwlo[i])))  
            where_are_NaNs = np.isnan(B)
            B[where_are_NaNs] = 0
            C = (n+1)*ff/f*np.arcsinh(np.sqrt(E/hwlo[i]))
            where_are_NaNs = np.isnan(C)
            C[where_are_NaNs] = 0
            t2 = np.heaviside(E-hwlo[i],1)*n*fff/f*np.arccosh(np.sqrt(E/hwlo[i]))
            where_are_NaNs = np.isnan(t2)
            t2[where_are_NaNs] = 0 
            C = (2*E)*(C+t2)
            P = (C-A-B)/(Z*(E**1.5))
            P_pol += P            #formula from fiorentini paper on Mg3Sb2
          print('P_pol',P_pol)
          rate.append(P_pol/a_pop)

      if c == 'optical':
          Nop=1/(np.exp(hwlo/temp)-1)
          x = E/temp
          xo = hwlo/temp
          X = x-xo
          X[X<0] = 0
          P_op = ((ms**1.5)*(D_op**2)*(Nop*np.sqrt(x+xo)+(Nop+1)*np.sqrt(X)))/(np.sqrt(2*temp)*np.pi*xo*(hbar**2)*rho) #formula from jacoboni theory of electron transport in semiconductors
          rate.append(P_op/a_op)

      if c == 'polar acoustic':
          eps = eps_inf+eps_0
          qo = np.sqrt(((e**2)*abs(nd))/(eps*temp))
          eps_o = ((hbar**2)*(qo**2))/(2*ms)
          P_pac = ((piezo**2*e**2*ms**0.5*temp)/(np.sqrt(2*E)*2*np.pi*eps**2*hbar**2*rho*v**2))*(1-(eps_o/(2*E))*np.log(1+4*E/eps_o)+1/(1+4*E/eps_o))
          where_are_NaNs = np.isnan(P_pac)
          P_pac[where_are_NaNs] = 0
          rate.append(P_pac/a_pac)     

      if c == 'intervalley':
          Nop=1/(np.exp(hwlo/temp)-1)
          x = E/temp
          xo = hwlo/temp
          X = x-xo
          X[X<0] = 0
          P_iv =  ((me**1.5)*Zf*(D_op**2)*(Nop*np.sqrt(x+xo)+(Nop+1)*np.sqrt(X)))/(np.sqrt(2*temp)*np.pi*xo*(hbar**2)*rho) #formula from jacoboni theory of electron transport in semiconductors
          rate.append(P_iv/a_iv)

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
  return(L)
