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

#### Forced t_tensor to have all components
  t_tensor = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]], dtype=int)

  # Quick call function for L_loop (None is smearing type)
  fLloop = lambda spol : L_loop(data_controller, temp, None, ene, velkp, t_tensor, spol, ispin)

  # Quick call function for Zeros on rank Zero
  zoz = lambda r: (np.zeros((3,3,esize), dtype=float) if r==0 else None)

  L0 = zoz(rank)
  L0aux = fLloop(0)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None

  L1 = zoz(rank)
  L1aux = fLloop(1)
  comm.Reduce(L1aux, L1, op=MPI.SUM)
  L1aux = None

  L2 = zoz(rank)
  L2aux = fLloop(2)
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
def do_Boltz_tensors_smearing ( data_controller, temp, ene, velkp, ispin ):

  arrays,attributes = data_controller.data_dicts()
  esize = ene.size

  t_tensor = arrays['t_tensor']

  L0aux = L_loop(data_controller, temp, attributes['smearing'], ene, velkp, t_tensor, 0, ispin)
  L0 = (np.zeros((3,3,esize), dtype=float) if rank==0 else None)
  comm.Reduce(L0aux, L0, op=MPI.SUM)
  L0aux = None
  
  return L0


def get_tau (temp,data_controller, channels ):

  import numpy as np
  import scipy.constants as cp
  hj = cp.hbar
  h = 6.58211951440e-16
  hw0j = 0.063*1.60217662e-19 #joules
  hw0 = 0.063 
  rho = 2.330e3   #kg/m^3
  #Eo = 41.45*1.60217662e-19  #joules
  a = 5.43e-10 #metres
  Eaj = 9*1.60217662e-19 #joules
  Ea = 9
  vl = 9.04e3    #m/s
  vt = 5.34e3    #m/s
  tempj = temp*1.60217662e-19
  nI = 1e23 #no.of impuritites/m^3
  e = 1.60217662e-19
  n = 1e19 #electron density /m^3
  et =11.9*8.854187817e-12 # dielectric constant*permitivtty of free space
  arry,attr = data_controller.data_dicts()
  snktot = arry['E_k'].shape[0]
  nspin = arry['E_k'].shape[2]
  bnd = attr['bnd']
  taus = []
  ms = 0.295
  DtKj = 8e10*1.60217662e-19 #J/m
  DtK = 8e10 
  di_inf = 32.6 #PbTe
  di_0 = 400 #PbTe
  F_scr = 1
  #ms = ((mt**2)*ml)**(1/3)
  #v = ((vt**2)*vl)**(1./3)
  v = (2*vt+vl)/3
  mo = 9.10938e-31
  vo = 343
  vs = v/vo
  me = ms*9.10938e-31*np.ones((snktot,bnd,nspin), dtype=float) #effective mass tensor in kg 
  E_j = (1.60217662e-19*(arry['E_k'][:,:bnd]))
  E_j[E_j>1e-25] -= (np.min(E_j[E_j>1e-25])+1e-35)
  E_j[E_j<=1e-25] = np.max(E_j[E_j<=1e-25])+1e-35 - E_j[E_j<=1e-25]
  E = ((arry['E_k'][:,:bnd]))
  E[E>=0.1] -= (np.min(E[E>=0.1]))
  E[E<0.1] = (np.max(E[E<0.1]) - E[E<0.1])
  E =1.60217662e-19* (E + 1e-5)
  for c in channels: 
  
   #   if c == 'impurity':
    #      qo = np.sqrt(((e**2)*n)/(et*tempj))
     #     epso = ((hj**2)*(qo**2))/(2*me)
    #      i_tau = (16*np.pi*np.sqrt(2*me)*(et**2)*(E_j**1.5))/((np.log(1+(4*E_j/epso))-((4*E_j/epso)/(1+(4*E_j/epso))))*(e**4)*nI)
    #      taus.append(i_tau)

      if c == 'accoustic':
          a_tau = (2*np.pi*hj**4*rho*(v**2)*np.power(E_j/(tempj),(-0.5)))/((np.power(2*me*tempj,1.5)*Eaj**2))
          #a_tau *= np.sqrt(e)
          taus.append(a_tau)

      if c == 'optical':
          #o_tau = (2*(hw0/Eo)**2*h**2*a**2*rho*np.power(E/(temp),(-0.5)))/((np.power(2*me*temp,1.5))*np.pi)
	  Nop = (tempj/hw0j)-0.5
          x = E_j/tempj
          xo = hw0j/tempj
          X = x-xo
          X[X<0] = 0
          o_tau = (np.sqrt(2*tempj)*np.pi*xo*(hj**2)*rho)/((me**1.5)*(DtKj**2)*(Nop*np.sqrt(x+xo)+(Nop+1)*np.sqrt(X)))
          taus.append(o_tau)
    

      #if c == 'polar optical':
  	  #di = ((1/di_inf)-(1/di_0))**-1
          #po_tau = (di*h**2*np.power(E/(temp),(0.5)))/((np.power(2*me*temp,0.5))*F_scr*e**2)
          #po_tau = (2*np.pi*di)
          #taus.append(po_tau)

      tau = np.zeros((snktot,bnd,nspin), dtype=float)
      for t in taus:
          tau += 1./t
      tau = 1/tau
  #tau_new = np.reshape(tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #o_tau_new = np.reshape(o_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #a_tau_new = np.reshape(a_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #i_tau_new = np.reshape(i_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #po_tau_new = np.reshape(po_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  #E_re = np.reshape(E,(snktot,bnd)) #i do this because i am not able to saave 3d arrays to a file
  #Ej_re = np.reshape(E_j,(snktot,bnd)) #i do this because i am not able to saave 3d arrays to a file
  #np.savetxt('tau.dat',tau_new)
  #np.savetxt('o_tau.dat',o_tau_new)
  #np.savetxt('a_tau.dat',a_tau_new)
  #np.savetxt('i_tau.dat',i_tau_new)
  #np.savetxt('po_tau.dat',po_tau_new)
  #np.savetxt('Ej.dat',Ej_re)
  #np.savetxt('E.dat',E_re)
  return(tau) 

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
  tau_t = get_tau(temp,data_controller,['accoustic','optical'])
  for n in range(bnd):
    Eaux = np.reshape(np.repeat(arrays['E_k'][:,n,ispin],esize), (snktot,esize))
    tau_re = np.reshape(np.repeat(tau_t[:,n,0],esize), (snktot,esize))
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

  return L
