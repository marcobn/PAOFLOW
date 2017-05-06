#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
import numpy as np
import sys, time

from scipy import fftpack as FFT

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from Gatherv_Scatterv_wrappers import *
from load_balancing import *
from get_R_grid_fft import *


# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.set_printoptions(precision=3, threshold=200, edgeitems=200, linewidth=250, suppress=False)
def do_gradient(Hksp,a_vectors,alat):
    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------
    scipyfft=True
    index=None
    if rank == 0:
        nk1,nk2,nk3,nawf,nawf,nspin = Hksp.shape
        nktot = nk1*nk2*nk3
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin,"nk1":nk1,"nk2":nk2,"nk3":nk3}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']
    nk1=index["nk1"]
    nk2=index["nk2"]
    nk3=index["nk3"]

    #############################################################################################
    #############################################################################################
    #############################################################################################
    if rank==0:
        #roll the axes so nawf,nawf are first
        #only needed because we're splitting 
        #between procs on the first indice
        Hksp =  np.rollaxis(Hksp,3,0)
        Hksp =  np.rollaxis(Hksp,4,1)
        #have flattened H(k) as first indice
        Hksp =  np.ascontiguousarray(np.reshape(Hksp,(nawf*nawf,nk1,nk2,nk3,nspin)))
    else:
        Hksp=None

    Hkaux_tri  = Scatterv_wrap(Hksp)

    #real space 
    HRaux_tri  = np.zeros_like(Hkaux_tri)
    #set scipy fft for now
    scipyfft=True
    #receiving slice of array

    if scipyfft:
        HRaux_tri[:,:,:,:,:] = FFT.ifftn(Hkaux_tri[:,:,:,:,:],axes=(1,2,3))

    HRaux_tri = FFT.fftshift(HRaux_tri,axes=(1,2,3))           

    #############################################################################################
    #############################################################################################
    #############################################################################################

    # fft grid in R shifted to have (0,0,0) in the center
    _,Rfft,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)
    #reshape R grid and each proc's piece of Hr
    Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')

    H_parts=HRaux_tri.shape[0]
    #reshape Hr for multiplying by the three parts of Rfft grid
    HRaux_tri=np.reshape(HRaux_tri,(H_parts,nk1*nk2*nk3,nspin),order='C')
    dHRaux_tri = np.zeros((H_parts,3,nk1*nk2*nk3,nspin),dtype=np.complex128,order='C')
    # Compute R*H(R)
    for ispin in xrange(nspin):
        for l in xrange(3):
            dHRaux_tri[:,l,:,ispin] = 1.0j*alat*Rfft[:,l]*HRaux_tri[...,ispin]

    Rfft=None
    HRaux_tri=None
    #reshape for fft
    dHRaux_tri = np.reshape(dHRaux_tri,(H_parts,3,nk1,nk2,nk3,nspin),order='C')
    # Compute dH(k)/dk
    dHkaux_tri  = np.zeros_like(dHRaux_tri)
    if scipyfft:
        dHkaux_tri[:,:,:,:,:,:] = FFT.fftn(dHRaux_tri[:,:,:,:,:,:],axes=(2,3,4))
    dHraux_tri = None
    #############################################################################################
    #############################################################################################
    #############################################################################################


    #gather the arrays into flattened dHk
    dHksp_tri = Gatherv_wrap(dHkaux_tri)

    if rank!=0:
        dHksp_tri=None

    dHkaux_tri = None

    if rank==0:
        #reshape dHksp 
        dHksp = np.reshape(dHksp_tri,(nawf,nawf,3,nk1,nk2,nk3,nspin),order='C')
    dHksp_tri=None

    if rank==0:
        #roll back the axes
        #second nawf
        dHksp =  np.rollaxis(dHksp,1,6)
        #first nawf
        dHksp =  np.rollaxis(dHksp,0,5) 
        #3 components of the gradient
        dHksp =  np.rollaxis(dHksp,0,4) 
        #num entried for the dHksp is 3 times Hksp
        return  np.ascontiguousarray(np.reshape(dHksp,(nk1*nk2*nk3,3,nawf,nawf,nspin),order='C'),dtype=np.complex128)



