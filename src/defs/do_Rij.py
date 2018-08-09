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
import cmath
from math import cosh
import sys, time

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import scatter_array
from communication import gather_full
from smearing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_Hall_tensors(E_k,velkp,M_ij,kq_wght,temp,ispin,deltak,smearing,t_tensor,emin,emax,ne):
    # Compute the L_alpha tensors for Boltzmann transport
    if emin!=emax:
        de = (emax-emin)/ne
        ene = np.arange(emin,emax,de)
    else:
        ene=np.array([emax])
    
    if rank==0:
        R = np.zeros((3,3,3,ene.size),dtype=float,order="C")
    else: R = None

    Raux = H_loop(ene,E_k,velkp,M_ij,kq_wght,temp,ispin)
    comm.Reduce(Raux,R,op=MPI.SUM)

    return(ene,R)




def H_loop(ene,E_k,velkp,M_ij,kq_wght,temp,ispin):
    # We assume tau=1 in the constant relaxation time approximation

    R = np.zeros((3,3,3,ene.size),dtype=float,order="C")

    # mapping of the unique 2nd rank tensor components
    M_ij_ind = np.array([[0,3,5],
                         [3,1,4],
                         [5,4,2]],dtype=int,order="C")

    # precompute the 6 unique v_i*v_j
    v2=np.zeros((6,E_k.shape[0]),dtype=float,order="C")
    ij_ind = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]],dtype=int)

    for l in range(ij_ind.shape[0]):
        i     = ij_ind[l][0]
        j     = ij_ind[l][1]
        v2[l] = velkp[i]*velkp[j]
    
    # precompute sig_xyz
    sig_xyz=np.zeros((3,3,3,E_k.shape[0]),order="C")    
    for a in range(3):
        for b in range(3):
            if a==b and b==0: continue
            sig_xyz[a,b,0] = M_ij[M_ij_ind[b,1]]*v2[M_ij_ind[a,2]] - \
                             M_ij[M_ij_ind[b,2]]*v2[M_ij_ind[a,1]]
    for a in range(3):
        for b in range(3):                          
            if a==b and b==1: continue
            sig_xyz[a,b,1] = M_ij[M_ij_ind[b,2]]*v2[M_ij_ind[a,0]] - \
                             M_ij[M_ij_ind[b,0]]*v2[M_ij_ind[a,2]]
    for a in range(3):
        for b in range(3):                          
            if a==b and b==2: continue 
            sig_xyz[a,b,2] = M_ij[M_ij_ind[b,0]]*v2[M_ij_ind[a,1]] - \
                             M_ij[M_ij_ind[b,1]]*v2[M_ij_ind[a,0]]
                        

    for n in range(ene.shape[0]):        
        # dfermi(e-e_f)/de
        Eaux = 1.0/(1.0+np.cosh((E_k-ene[n])/temp))
        for a in range(3):
            for b in range(3):
                for g in range(3):                    
                    if a==b and b==g: continue                                           
                    R[a,b,g,n] = np.sum(sig_xyz[a,b,g]*Eaux)

    return(R*kq_wght[0]*0.5/temp)


