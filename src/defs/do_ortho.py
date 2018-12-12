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

def do_ortho ( Hks, Sks ):
  from scipy import linalg as spl
  from numpy import linalg as npl

  # If orthogonality is required, we have to apply a basis change to Hks as
  # Hks -> Sks^(-1/2)*Hks*Sks^(-1/2)

  nawf,_,nkpnts,nspin = Hks.shape
  S2k  = np.zeros((nawf,nawf,nkpnts), dtype=complex)
  for ik in range(nkpnts):
      S2k[:,:,ik] = npl.inv(spl.sqrtm(Sks[:,:,ik]))

  Hks_o = np.zeros((nawf,nawf,nkpnts,nspin), dtype=complex)
  for ispin in range(nspin):
      for ik in range(nkpnts):
          Hks_o[:,:,ik,ispin] = np.dot(S2k[:,:,ik], Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])

  return Hks_o

def do_orthogonalize ( data_controller ):
  from scipy import fftpack as FFT

  arrays,attributes = data_controller.data_dicts()

  nktot = attributes['nkpnts']
  nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

  if attributes['use_cuda']:
    from .cuda_fft import cuda_fftn
    arrays['Hks'] = cuda_fftn(np.moveaxis(arrays['HRs'],[0,1],[3,4]), axes=[0,1,2])
    arrays['Sks'] = cuda_fftn(np.moveaxis(arrays['SRs'],[0,1],[3,4]), axes=[0,1,2])
    arrays['Hks'] = np.reshape(np.moveaxis(arrays['Hks'],[3,4],[0,1]), (nawf,nawf,nktot,nspin), order='C')
    arrays['Sks'] = np.reshape(np.moveaxis(arrays['Sks'],[3,4],[0,1]), (nawf,nawf,nktot), order='C')
  else:
    arrays['Hks'] = FFT.fftn(arrays['HRs'],axes=[2,3,4])
    arrays['Sks'] = FFT.fftn(arrays['SRs'], axes=[2,3,4])
    arrays['Hks'] = np.reshape(arrays['Hks'], (nawf,nawf,nktot,nspin), order='C')
    arrays['Sks'] = np.reshape(arrays['Sks'], (nawf,nawf,nktot), order='C')

  arrays['Hks'] = do_ortho(arrays['Hks'], arrays['Sks'])
  arrays['Hks'] = np.reshape(arrays['Hks'], (nawf,nawf,nk1,nk2,nk3,nspin), order='C')
  arrays['Sks'] = np.reshape(arrays['Sks'], (nawf,nawf,nk1,nk2,nk3), order='C')
  if attributes['use_cuda']:
    from .cuda_fft import cuda_ifftn
    arrays['HRs'] = np.moveaxis(cuda_ifftn(np.moveaxis(arrays['Hks'],[0,1],[3,4]), axes=[0,1,2]),[3,4],[0,1])
  else:
    arrays['HRs'] = FFT.ifftn(arrays['Hks'], axes=[2,3,4])

  data_controller.broadcast_single_array('HRs')

  attributes['non_ortho'] = False
  del arrays['SRs']
