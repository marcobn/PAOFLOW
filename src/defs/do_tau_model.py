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
import scipy.constants as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_tau(data_controller,temp,channels,tau_dict):

  arry,attr = data_controller.data_dicts()
  h = cp.hbar
  kb = cp.Boltzmann
  snktot = arry['E_k'].shape[0]
  nspin = arry['E_k'].shape[2]
  bnd = attr['bnd']
  taus = []
  e = 1.60217662e-19

  hw0 = tau_attr['hw0']*1.60217662e-19
  rho = tau_attr['mass_density']*1e3
  a = tau_attr['lattice_param']*1e-10
  Ea = tau_attr['ac_deformation']*1.60217662e-19
  Eo = tau_attr['op_deformation']*1.60217662e-19
  nI = tau_attr['impurity_density']
  n = tau_attr['electron_density']
  nd = tau_attr['doping']
  ml = tau_attr['longitudanal_mass']
  mt = tau_attr['transverse_mass']
  di_inf = tau_attr['di_inf']*8.854187817e-12
  di_0 = tau_attr['di_0']*8.854187817e-12
  Zf = tau_attr['valley']
  v = tau_attr['velocity']
  ms = (ml*(mt**2))**(1./3)
  me = ms*9.10938e-31*np.ones((snktot,bnd,nspin), dtype=float) #effective mass tensor in kg 
  E = abs(1.60217662e-19*(arry['E_k'][:,:bnd]))
  temp *= 1.60217662e-19
  et =di_inf*8.854187817e-12 # dielectric constant*permitivtty of free space

  for c in channels:

      if c == 'impurity':
          qo = np.sqrt(((e**2)*n)/(et*temp))
          epso = ((h**2)*(qo**2))/(2*me)
          i_tau = (16*np.pi*np.sqrt(2*me)*(et**2)*(E**1.5))/((np.log(1+(4*E/epso))-((4*E/epso)/(1+(4*E/epso))))*(e**4)*nI)
          taus.append(i_tau)

      if c == 'acoustic':
          #a_tau = (2*np.pi*(h**4)*rho*v**2*((E/temp)**-0.5))/((np.power(2*me*temp,1.5)*Ea**2))
          a_tau = (2*np.pi*(h**4)*rv2*((E/temp)**-0.5))/((np.power(2*me*temp,1.5)*Ea**2))
          taus.append(a_tau)

      if c == 'optical':
         # Nop = (temp/hw0)-0.5
         # x = E/temp
         # xo = hw0/temp
         # X = x-xo
         # X[X<0] = 0
         # o_tau = (np.sqrt(2*temp)*np.pi*xo*(h**2)*rho)/((me**1.5)*(DtK**2)*(Nop*np.sqrt(x+xo)+(Nop+1)*np.sqrt(X)))#elastic +inelastic
          #o_tau_no_inels = (2/np.pi)*((hw0/Eo)**2)*(h**2*a**2*rho)*((E/temp)**-0.5)/((2*me*temp)**1.5)
          o_tau = (2/np.pi)*((hw0/Eo)**2)*(h**2*a**2*rho)*((E/temp)**-0.5)/((2*me*temp)**1.5)
          taus.append(o_tau)


      if c == 'polar optical screened' or c == 'polar optical unscreened':
          ro = ((di_inf*temp)/(4*np.pi*e**2*nd))**0.5
          deltap = (2*me*E*(2*ro)**2)/h**2
          di = ((1/di_inf)-(1/di_0))**-1

          if c == 'polar optical screened':

              F_scr = 1 - ((2/deltap)*np.log(deltap+1))+1/(deltap+1)
              po_tau = (di*h**2*np.power(E/(temp),(0.5)))/((np.power(2*me*temp,0.5))*F_scr*e**2)
              taus.append(po_tau)

          else:

              po_tau_no_scr = (di*h**2*np.power(E/(temp),(0.5)))/((np.power(2*me*temp,0.5))*e**2)
              taus.append(po_tau_no_scr)

      tau = np.zeros((snktot,bnd,nspin), dtype=float)
      for t in taus:
          tau += 1./t
      tau = 1/tau

      if c == 'constant':
        tau = np.ones((snktot,bnd,nspin), dtype=float)

  E_re = np.reshape(E/1.60217662e-19,(snktot,bnd)) #i do this because i am not able to saave 3d arrays to a file
  tau_new = np.reshape(tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  o_tau_new = np.reshape(o_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  a_tau_new = np.reshape(a_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  po_scr_tau_new = np.reshape(po_tau,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  po_no_scr_tau_new = np.reshape(po_tau_no_scr,(snktot,bnd))   #i do this because i am not able to saave 3d arrays to a file
  np.savetxt('E.dat',E_re)
  np.savetxt('tau.dat',tau_new)
  np.savetxt('o_tau.dat',o_tau_new)
  np.savetxt('a_tau.dat',a_tau_new)
  np.savetxt('po_scr_tau.dat',po_scr_tau_new)
  np.savetxt('po_no_scr_tau.dat',po_no_scr_tau_new)
  return tau


