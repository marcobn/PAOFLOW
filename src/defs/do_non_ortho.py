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

def do_non_ortho ( Hks, Sks ):
  import numpy as np
  from scipy import linalg as spl

  # Take care of non-orthogonality, if needed
  # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
  # Hks -> Sks^(1/2)*Hks*Sks^(1/2)

  nawf,_,nkpnts,nspin = Hks.shape
  S2k  = np.zeros((nawf,nawf,nkpnts), dtype=complex)
  for ik in range(nkpnts):
    S2k[:,:,ik] = spl.sqrtm(Sks[:nawf,:nawf,ik])

  Hks_no = np.zeros((nawf,nawf,nkpnts,nspin), dtype=complex)
  for ispin in range(nspin):
    for ik in range(nkpnts):
      Hks_no[:,:,ik,ispin] = np.dot(S2k[:,:,ik], Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])

  return Hks_no
