#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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

def perturb_split(rot_op1,rot_op2,v_k,degen,return_v_k=False):
    import numpy as np
    from scipy import linalg as LAN

    op1 = np.dot(np.conj(v_k.T),np.dot(rot_op1,v_k))
    if len(degen)==0:
        op2 = np.dot(np.conj(v_k.T),np.dot(rot_op2,v_k))
        if return_v_k:
          return op1,op2,np.array([[]])
        else:
          return op1,op2

    v_k_temp = np.copy(v_k)
    
    for i in range(len(degen)):
        # degenerate subspace indices upper and lower lim
        ll = degen[i][0]
        ul = degen[i][-1]+1

        # diagonalize in degenerate subspace
        vals,weight = LAN.eigh(op1[ll:ul,ll:ul])

        # linear combination of eigenvectors of H that diagonalize
        v_k_temp[:,ll:ul]=np.dot(v_k_temp[:,ll:ul],weight)


    # return new operator in non degenerate basis
    op1 = np.dot(np.conj(v_k_temp.T),np.dot(rot_op1,v_k_temp))
    op2 = np.dot(np.conj(v_k_temp.T),np.dot(rot_op2,v_k_temp))

    if return_v_k:
      return(op1, op2, v_k_temp)
    else:
      return(op1, op2)
