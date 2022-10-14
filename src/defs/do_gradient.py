#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
#F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli, Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

def do_gradient ( data_controller ):
  import numpy as np
  from scipy import fftpack as FFT
  from .get_R_grid_fft import get_R_grid_fft

  arry,attr = data_controller.data_dicts()

  #----------------------
  # Compute the gradient of the k-space Hamiltonian
  #----------------------

  nktot = attr['nkpnts']
  snawf,nk1,nk2,nk3,nspin = arry['Hksp'].shape

  # fft grid in R shifted to have (0,0,0) in the center
  get_R_grid_fft(data_controller, nk1, nk2, nk3)

  arry['dHksp'] = np.empty((snawf,nk1,nk2,nk3,3,nspin), dtype=complex, order='C')
  for ispin in range(nspin):
    for n in range(snawf):
      ########################################
      ### real space grid replaces k space ###
      ########################################
      if attr['use_cuda']:
        arry['Hksp'][n,:,:,:,ispin] = cuda_ifftn(arry['Hksp'][n,:,:,:,ispin])*1.0j*attr['alat']
      else:
        arry['Hksp'][n,:,:,:,ispin] = FFT.ifftn(arry['Hksp'][n,:,:,:,ispin])*1.0j*attr['alat']

      # Compute R*H(R)
      for l in range(3):
        arry['dHksp'][n,:,:,:,l,ispin] = FFT.fftn(arry['Rfft'][:,:,:,l]*arry['Hksp'][n,:,:,:,ispin])

