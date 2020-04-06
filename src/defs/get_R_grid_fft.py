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


def get_R_grid_fft ( data_controller, nr1, nr2, nr3):
  import numpy as np
  from scipy import fftpack as FFT

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  nrtot = nr1*nr2*nr3

  a_vectors = arrays['a_vectors']

  arrays['R'] = np.zeros((nrtot,3), dtype=float)
  arrays['idx'] = np.zeros((nr1,nr2,nr3), dtype=int)
  arrays['Rfft'] = np.zeros((nr1,nr2,nr3,3), dtype=float)
  arrays['R_wght'] = np.ones((nrtot), dtype=float)

  for i in range(nr1):
    for j in range(nr2):
      for k in range(nr3):
        n = k + j*nr3 + i*nr2*nr3
        Rx = float(i)/float(nr1)
        Ry = float(j)/float(nr2)
        Rz = float(k)/float(nr3)
        if Rx >= 0.5: Rx=Rx-1.0
        if Ry >= 0.5: Ry=Ry-1.0
        if Rz >= 0.5: Rz=Rz-1.0
        Rx -= int(Rx)
        Ry -= int(Ry)
        Rz -= int(Rz)

        arrays['R'][n,:] = Rx*nr1*a_vectors[0,:] + Ry*nr2*a_vectors[1,:] + Rz*nr3*a_vectors[2,:]
        arrays['Rfft'][i,j,k,:] = arrays['R'][n,:]
        arrays['idx'][i,j,k] = n
