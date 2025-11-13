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

import numpy as np
import scipy.linalg as la


def orbital_array ( data_controller):

    arry,attr = data_controller.data_dicts()

    naw = np.zeros(len(arry['atoms']),dtype=int)

    if (attr['dftSO'] == True):
        for i in range (len(arry['atoms'])):
            n_atom=0
            for j in (arry['shells'][arry['atoms'][i]]):

                if(j==0):
                    n_atom+=2
                elif(j==1):
                    n_atom+=3
                elif(j==2):
                    n_atom+=5
                elif(j==3):
                    n_atom+=7

            naw[i] = n_atom
    return naw

def do_projection_operator ( data_controller ):

    arry,attr = data_controller.data_dicts()

    P = np.zeros((attr['nawf'],attr['nawf']),dtype=float)

    for i in range(arry['shc_proj'].shape[0]):

        idx = np.sum(arry['naw'][0:arry['shc_proj'][i]])
        fdx = idx + arry['naw'][arry['shc_proj'][i]]
 
        P[idx:fdx,idx:fdx]= np.eye(arry['naw'][i])

    return (P)

