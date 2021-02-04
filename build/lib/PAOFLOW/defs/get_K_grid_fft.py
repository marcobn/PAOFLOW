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


def get_K_grid_fft ( data_controller ):
  import numpy as np

  arrays,attributes = data_controller.data_dicts()

  nk1 = attributes['nk1']
  nk2 = attributes['nk2']
  nk3 = attributes['nk3']
  b_vectors = arrays['b_vectors']

  nktot = nk1*nk2*nk3
  arrays['kq_wght'] = np.ones((nktot), dtype=float)/nktot
  return

### Not used
  Kint = np.zeros((3,nktot), dtype=float)
  idk = np.zeros((nk1,nk2,nk3), dtype=int)

  for i in range(nk1):
    for j in range(nk2):
      for k in range(nk3):
        n = k + j*nk3 + i*nk2*nk3
        Rx = float(i)/float(nk1)
        Ry = float(j)/float(nk2)
        Rz = float(k)/float(nk3)
        if Rx >= 0.5: Rx=Rx-1.0
        if Ry >= 0.5: Ry=Ry-1.0
        if Rz >= 0.5: Rz=Rz-1.0
        Rx -= int(Rx)
        Ry -= int(Ry)
        Rz -= int(Rz)
        idk[i,j,k]=n
        Kint[:,n] = Rx*b_vectors[0,:]+Ry*b_vectors[1,:]+Rz*b_vectors[2,:]



def get_K_grid_fft_crystal ( nk1,nk2,nk3 ):
  import numpy as np

  nktot = nk1*nk2*nk3

  Kint = np.zeros((nktot,3), dtype=float)

  for i in range(nk1):
    for j in range(nk2):
      for k in range(nk3):
        n = k + j*nk3 + i*nk2*nk3
        Rx = float(i)/float(nk1)
        Ry = float(j)/float(nk2)
        Rz = float(k)/float(nk3)
        if Rx >= 0.5: Rx=Rx-1.0
        if Ry >= 0.5: Ry=Ry-1.0
        if Rz >= 0.5: Rz=Rz-1.0
        Rx -= int(Rx)
        Ry -= int(Ry)
        Rz -= int(Rz)

        Kint[n] = Rx,Ry,Rz

  return Kint

