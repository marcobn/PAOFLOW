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
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

def do_gradient ( data_controller ):
  import numpy as np
  from scipy import fftpack as FFT
  from .get_R_grid_fft import get_R_grid_fft

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  #----------------------
  # Compute the gradient of the k-space Hamiltonian
  #----------------------

  nktot = attributes['nkpnts']
  snawf,nk1,nk2,nk3,nspin = arrays['Hksp'].shape

  # fft grid in R shifted to have (0,0,0) in the center
  get_R_grid_fft(data_controller)
  #reshape R grid and each proc's piece of Hr

  arrays['Rfft'] = np.reshape(arrays['Rfft'], (nk1,nk2,nk3,3), order='C')

  arrays['dHksp'] = np.empty((snawf,nk1,nk2,nk3,3,nspin), dtype=complex, order='C')
  for ispin in range(nspin):
    for n in range(snawf):
      ########################################
      ### real space grid replaces k space ###
      ########################################
      if attributes['use_cuda']:
        arrays['Hksp'][n,:,:,:,ispin] = cuda_ifftn(arrays['Hksp'][n,:,:,:,ispin])*1.0j*attributes['alat']
      else:
        arrays['Hksp'][n,:,:,:,ispin] = FFT.ifftn(arrays['Hksp'][n,:,:,:,ispin])*1.0j*attributes['alat']
      # Compute R*H(R)
      for l in range(3):
        arrays['dHksp'][n,:,:,:,l,ispin] = FFT.fftn(arrays['Rfft'][:,:,:,l]*arrays['Hksp'][n,:,:,:,ispin])
