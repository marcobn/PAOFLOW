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

class TauModel:
  '''
  Object containing a functional form for Tau, depending on termperature and an energy eigenvalue (both in units of eV).
  All TauModel's included in the scattering_channels arugment of PAOFLOW's transport routine
   are homically summed to obtain the effective relaxation time for each (temp,eig) combination.
  An additional argument, params (a dictionary), is required. Further parameters can be passed into    the routine by including them in this dictionary.
  '''

  def __init__ ( self, function=None, params=None, weight=1. ):
    '''
    Arguments:
      function (func): A function requiring 3 arguments, (temp,eig,params)
      params (dict): A dictionary with any additional constants of variables the function may require
      weight (float): A weight, w_i, incorporated in the harmonic sum of tau. 1/Tau = Sum(w_i/Tau_i)
    '''
    self.function = function
    self.params = params
    self.weight = weight

  def evaluate ( self, temp, eigs ):
    return self.function(temp, eigs, self.params)
