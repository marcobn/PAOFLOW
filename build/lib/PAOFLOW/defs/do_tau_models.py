#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
from scipy.constants import hbar
from scipy.constants import Boltzmann as kb

me = 9.10938e-31
e = 1.60217662e-19
ev2j = 1.60217662e-19
epso = 8.854187817e-12

def acoustic_model ( temp, eigs, params ):
  # Formula from fiorentini paper on Mg3Sb2
  temp *= ev2j
  E = eigs * ev2j # Eigenvalues in J
  v = params['v'] # Velocity in m/s
  rho = params['rho'] # Mass density kg/m^3
  ms = params['ms']*me #effective mass tensor in kg 
  D_ac = params['D_ac']*ev2j # Acoustic deformation potential in J

  return (2*np.pi*rho*(hbar**2*v)**2)/((2*ms)**1.5*(D_ac**2)*np.sqrt(E)*temp)


def optical_model ( temp, eigs, params ):
  # Formula from jacoboni theory of electron transport in semiconductors
  temp *= ev2j
  E = eigs * ev2j
  hwlo = np.array(params['hwlo'])*ev2j # Phonon freq
  rho = params['rho'] # Mass density kg/m^3
  D_op = params['D_op']*ev2j # Acoustic deformation potential in J
  ms = params['ms']*me #effective mass tensor in kg 

  x = E/temp
  x0 = hwlo/temp
  X = x - x0
  X[X<0] = 0

  Nop = 1 / (np.exp(x0)-1)

  return (np.sqrt(2*temp)*np.pi*x0*rho*hbar**2)/((ms**1.5)*(D_op**2)*(Nop*np.sqrt(x+x0)+(Nop+1)*np.sqrt(X)))

def polar_acoustic_model ( temp, eigs, params ):

  temp *= ev2j
  E = eigs * ev2j
  piezo = params['piezo']  # Piezoelectric constant
  nd = np.abs(params['doping_conc'])*1e6 # Doping concentration in /m^3
  eps_0 = params['eps_0']*epso # Low freq dielectric const
  eps_inf = params['eps_inf']*epso # High freq dielectirc const
  ms = params['ms']*me #effective mass tensor in kg 
  rho = params['rho']
  v = params['v']

  eps = eps_inf + eps_0
  qo = np.sqrt(abs(nd)*e**2/(eps*temp))
  eps_o = ((hbar*qo)**2)/(2*ms)
  P_pac = (((piezo*e)**2*ms**0.5*temp)/(np.sqrt(2*E)*2*np.pi*eps**2*hbar**2*rho*v**2))*(1-(eps_o/(2*E))*np.log(1+4*E/eps_o)+1/(1+4*E/eps_o))
  P_pac[np.isnan(P_pac)] = 0
  return 1 / P_pac

def polar_optical_model ( temp, eigs, params ):
  # Formula from fiorentini paper on Mg3Sb2
  temp *= ev2j
  E = eigs * ev2j
  Ef = params['Ef']*ev2j #fermi energy
  hwlo = np.array(params['hwlo'])*ev2j # Phonon freq
  eps_0 = params['eps_0']*epso #low freq dielectric const
  eps_inf = params['eps_inf']*epso #high freq dielectirc const
  ms = params['ms']*me #effective mass tensor in kg


  fermi = lambda E,Ef,T : 1/(np.exp((E-Ef)/T)+1.)
  planck = lambda hwlo,T : 1/(np.exp(hwlo/T)-1)

  P_pol = 0.
  eps = eps_inf + eps_0
  eps_inv = 1/eps_inf - 1/eps

  for hw in hwlo:
    f = fermi(E, Ef, temp)
    fp = fermi(E+hw, Ef, temp)
    fm = fermi(E-hw, Ef, temp)
    n = planck(hw, temp)
    Wo = e**2/(4*np.pi*hbar**2) * np.sqrt(2*ms*hw)*eps_inv
    Z = 2/(Wo*np.sqrt(hw))

    def remove_NaN ( arr ):
      arr[np.isnan(arr)] = 0
      return arr

    A = remove_NaN((n+1)*fp/f * ((2*E+hw)*np.arcsinh(np.sqrt(E/hw))-np.sqrt(E*(E+hw))))
    B = remove_NaN(np.heaviside(E-hw,1)*n*fm/f * ((2*E-hw)*np.arccosh(np.sqrt(E/hw))-np.sqrt(E*(E-hw))))
    t1 = remove_NaN((n+1)*fp/f * np.arcsinh(np.sqrt(E/hw)))
    t2 = remove_NaN(np.heaviside(E-hw,1)*n*fm/f * np.arccosh(np.sqrt(E/hw)))
    C = 2*E*(t1+t2)
    P = (C-A-B)/(Z*E**1.5)
    P_pol += P

  return 1 / P_pol


def impurity_model ( temp, eigs, params ):
  #formula from fiorentini paper on Mg3Sb2
  temp *= ev2j
  E = eigs * ev2j
  nI = np.abs(params['nI'])*1e6 # impurity conc in /m^3
  Zi = params['Zi']
  ms = params['ms']*me #effective mass tensor in kg 
  eps_0 = params['eps_0']*epso #low freq dielectric const
  eps_inf = params['eps_inf']*epso #high freq dielectirc const

  eps = eps_inf+eps_0
  qo = np.sqrt(e**2*nI/(eps*temp))
  x = (hbar*qo)**2/(8*ms*E)
  P_imp = np.pi*nI*Zi**2*e**4/(E**1.5*np.sqrt(2*ms)*(4*np.pi*eps)**2)
  return 1 / (P_imp * (np.log(1+1./x)-1./(1+x)))

def builtin_tau_model ( label, params, weight ):
  from .TauModel import TauModel

  model = TauModel(params=params, weight=weight)

  if label == 'acoustic':
    model.function = acoustic_model
  elif label == 'optical':
    model.function = optical_model
  elif label == 'polar_optical':
    model.function = polar_optical_model
  elif label == 'polar_acoustic':
    model.function = polar_acoustic_model
  elif label == 'impurity':
    model.function = impurity_model
  else:
    print('Model not implemented.')
    return None

  return model
