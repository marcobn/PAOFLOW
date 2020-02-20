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

import numpy as np
import sys, time
from numpy import linalg as LAN

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from .load_balancing import *
from .get_R_grid_fft import *
from .communication import *
from .constants import *
from .perturb_split import *
# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    from cuda_fft import *
except: pass
from scipy import fftpack as FFT


def do_d2Hd2k_ij(Hksp,Rfft,alat,npool,v_kp,bnd,degen):
    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------
    Rfft=np.transpose(Rfft,(3,0,1,2))

    num_n,nk1,nk2,nk3,nspin = Hksp.shape

    _,nk1,nk2,nk3,nspin = Hksp.shape
    
    M_ij   = np.zeros((6,v_kp.shape[0],bnd,v_kp.shape[3]),dtype=float,order="C")
    ij_ind = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]],dtype=int)

    nktot = nk1*nk2*nk3



    comm.Barrier()
    ########################################
    ### real space grid replaces k space ###
    ########################################
    # c1=c2=0
    # for ik in range(len(degen[0])):
    #     if len(degen[0][ik]) != 0:
    #         c1+=1
    #     else:
    #         c2+=1
    # print(c1+c2,c1,c2)
    

    #############################################################################################
    #############################################################################################
    #############################################################################################
    num_n = Hksp.shape[0]

    dvec_list=[]

    for ij in range(M_ij.shape[0]):
        dir_tmp=[]
        d2Hksp = None
        d2Hksp = np.zeros((num_n,nk1,nk2,nk3,nspin),dtype=complex,order='C')    
        
        ipol = ij_ind[ij][0]
        jpol = ij_ind[ij][1]

        RIJ = Rfft[ipol]*Rfft[jpol]

        for ispin in range(d2Hksp.shape[4]):
            for n in range(d2Hksp.shape[0]):                
                # because of the way this is coded...Hksp is actually HR*1.0j*alat
                d2Hksp[n,:,:,:,ispin] = FFT.fftn(RIJ*Hksp[n,:,:,:,ispin]*1.0j*alat)

        #############################################################################################
        #############################################################################################
        #############################################################################################

        #gather the arrays into flattened dHk
        d2Hksp = np.reshape(d2Hksp,(num_n,nk1*nk2*nk3,nspin),order='C')        
        d2Hksp = gather_scatter(d2Hksp,1,npool)
        nawf   = int(np.sqrt(d2Hksp.shape[0]))

        d2Hksp = np.reshape(d2Hksp,(nawf,nawf,d2Hksp.shape[1],nspin),order='C')

        tksp = np.zeros_like(d2Hksp)

        #find non-degenerate set of psi(k) for d2H/d2k_ij
        for ispin in range(tksp.shape[3]):
            isp_tmp=[]
            for ik in range(tksp.shape[2]):

                # we save dvec so that it can be used when calculating the second term in d2E/d2k
                tksp[:,:,ik,ispin],_,dvec = perturb_split(d2Hksp[:,:,ik,ispin],
                                                          d2Hksp[:,:,ik,ispin],
                                                          v_kp[ik,:,:,ispin],
                                                          degen[ispin][ik],return_v_k=True)

                isp_tmp.append(dvec)
            dir_tmp.append(isp_tmp)
        dvec_list.append(dir_tmp)

        
        # get the value for d2H/d2k
        for ispin in range(d2Hksp.shape[3]):
            for n in range(bnd):                
                M_ij[ij,:,n,ispin] = tksp[n,n,:,ispin].real

        comm.Barrier()

    Hksp_aux= d2Hksp = None    

    return M_ij,dvec_list



