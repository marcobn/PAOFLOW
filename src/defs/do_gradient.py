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


from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from get_R_grid_fft import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    from cuda_fft import *
except: pass
from scipy import fftpack as FFT
scipyfft = True

def do_gradient(Hksp,a_vectors,alat,nthread,npool,using_cuda):
    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------

    _,nk1,nk2,nk3,nspin = Hksp.shape
    nktot = nk1*nk2*nk3

    # fft grid in R shifted to have (0,0,0) in the center
    _,Rfft,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

    Rfft = np.reshape(Rfft,(3,nk1,nk2,nk3),order='C')

    comm.Barrier()

    #############################################################################################
    #############################################################################################
    #############################################################################################

    num_n = Hksp.shape[0]
    dHksp = np.zeros((num_n,nk1,nk2,nk3,3,nspin),dtype=complex,order='C')
    for ispin in range(dHksp.shape[5]):
        for n in range(dHksp.shape[0]):
            ########################################
            ### real space grid replaces k space ###
            ########################################
            if using_cuda:
                Hksp[n,:,:,:,ispin] = cuda_ifftn(Hksp[n,:,:,:,ispin])*1.0j*alat

            elif scipyfft:
                Hksp[n,:,:,:,ispin] = FFT.ifftn(Hksp[n,:,:,:,ispin])*1.0j*alat

            # Compute R*H(R)            
            for l in range(dHksp.shape[4]):
                dHksp[n,:,:,:,l,ispin] = FFT.fftn(Rfft[l]*Hksp[n,:,:,:,ispin])

                
    #############################################################################################
    #############################################################################################
    #############################################################################################

    return(dHksp)



