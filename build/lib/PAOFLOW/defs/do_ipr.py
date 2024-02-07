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

def inverse_participation_ratio (data_controller):

  arry,attr = data_controller.data_dicts()

  nbands = attr['bnd']

  if 'Hksp' in arry:
    kpts = arry['kpnts']
  else:
    kpts = arry['kq'].T

  nkpts = kpts.shape[0]

  nspin = attr['nspin']

  ipr = np.zeros((nspin,nkpts,nbands,3),dtype=object)

  for ispin in range(nspin):
    for ikpt in range(nkpts):
      for iband in range(nbands):

        vk = arry['v_k'][ikpt,:,iband,ispin]
        ek = arry['E_k'][ikpt,iband,ispin]

        vk_abs = np.abs(vk)

        ipr[ispin,ikpt,iband,0] = kpts[ikpt]
        ipr[ispin,ikpt,iband,1] = ek
        ipr[ispin,ikpt,iband,2] = np.sum(vk_abs**4) / ( np.sum(vk_abs**2)**2 )

  return ipr

