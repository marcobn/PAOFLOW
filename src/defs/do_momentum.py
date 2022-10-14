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

def do_momentum ( data_controller ):
  import numpy as np
  from .perturb_split import perturb_split

  arry,attr = data_controller.data_dicts()

  nktot,_,nawf,nawf,nspin = arry['dHksp'].shape

  arry['pksp'] = np.zeros_like(arry['dHksp'])

  for ispin in range(nspin):
    for ik in range(nktot):
      for l in range(3):
         arry['pksp'][ik,l,:,:,ispin],_ = perturb_split(arry['dHksp'][ik,l,:,:,ispin], 
                                                        arry['dHksp'][ik,l,:,:,ispin], 
                                                        arry['v_k'][ik,:,:,ispin],
                                                        arry['degen'][ispin][ik])
