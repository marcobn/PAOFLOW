#
# PAOFLOW
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
import multiprocessing

try:
    import pyfftw
except:
    from scipy import fftpack as FFT

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import *
from get_R_grid_fft import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_gradient(Hksp,a_vectors,alat,nthread,npool,scipyfft):
    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------

    index = None

    if rank == 0:
        nk1,nk2,nk3,nawf,nawf,nspin = Hksp.shape
        nktot = nk1*nk2*nk3
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        # fft grid in R shifted to have (0,0,0) in the center
        _,Rfft,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

        if scipyfft:
            HRaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
            HRaux[:,:,:,:,:,:] = FFT.ifftn(Hksp[:,:,:,:,:,:],axes=[0,1,2])
            HRaux = FFT.fftshift(HRaux,axes=(0,1,2))
            Hksp = None
        else:
            HRaux  = np.zeros_like(Hksp)
            for ispin in xrange(nspin):
                for n in xrange(nawf):
                    for m in xrange(nawf):
                        fft = pyfftw.FFTW(Hksp[:,:,:,n,m,ispin],HRaux[:,:,:,n,m,ispin],axes=(0,1,2), direction='FFTW_BACKWARD',\
                              flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                        HRaux[:,:,:,n,m,ispin] = fft()
            HRaux = FFT.fftshift(HRaux,axes=(0,1,2))
            Hksp = None

        dHksp  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
        Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')
        HRaux = np.reshape(HRaux,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
        dHRaux  = np.zeros((nk1*nk2*nk3,3,nawf,nawf,nspin),dtype=complex)
        d2HRaux  = np.zeros((nk1*nk2*nk3,3,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp  = None
        d2Hksp  = None
        Rfft = None
        HRaux = None
        dHRaux  = None
        d2HRaux  = None

    for pool in xrange(npool):
        ini_ip, end_ip = load_balancing(npool,pool,nktot)
        nkpool = end_ip - ini_ip

        if rank == 0:
            HRaux_split = HRaux[ini_ip:end_ip]
            dHRaux_split = dHRaux[ini_ip:end_ip]
            d2HRaux_split = d2HRaux[ini_ip:end_ip]
            Rfft_split = Rfft[ini_ip:end_ip]
        else:
            HRaux_split = None
            dHRaux_split = None
            d2HRaux_split = None
            Rfft_split = None

        dHRaux1 = scatter_array(dHRaux_split)
        d2HRaux1 = scatter_array(d2HRaux_split)
        HRaux1 = scatter_array(HRaux_split)
        Rfftaux = scatter_array(Rfft_split)

        # Compute R*H(R)
        for l in xrange(3):
            for ispin in xrange(nspin):
                for n in xrange(nawf):
                    for m in xrange(nawf):
                        dHRaux1[:,l,n,m,ispin] = 1.0j*alat*Rfftaux[:,l]*HRaux1[:,n,m,ispin]
                        for lp in xrange(3):
                            d2HRaux1[:,l,lp,n,m,ispin] = -1.0*alat**2*Rfftaux[:,l]*Rfftaux[:,lp]*HRaux1[:,n,m,ispin]

        gather_array(dHRaux_split, dHRaux1)
        gather_array(d2HRaux_split, d2HRaux1)

        if rank == 0:
            dHRaux[ini_ip:end_ip,:,:,:,:] = dHRaux_split[:,:,:,:,:,]
            d2HRaux[ini_ip:end_ip,:,:,:,:,:] = d2HRaux_split[:,:,:,:,:,:,]

    if rank == 0:
        dHRaux = np.reshape(dHRaux,(nk1,nk2,nk3,3,nawf,nawf,nspin),order='C')
        d2HRaux = np.reshape(d2HRaux,(nk1,nk2,nk3,3,3,nawf,nawf,nspin),order='C')

    if rank == 0:
        # Compute dH(k)/dk and d2H(k)/dkdk'
        if scipyfft:
            dHksp  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
            d2Hksp  = np.zeros((nk1,nk2,nk3,3,3,nawf,nawf,nspin),dtype=complex)
            for l in xrange(3):
                dHksp[:,:,:,l,:,:,:] = FFT.fftn(dHRaux[:,:,:,l,:,:,:],axes=[0,1,2])
                for lp in xrange(3):
                    d2Hksp[:,:,:,l,lp,:,:,:] = FFT.fftn(d2HRaux[:,:,:,l,lp,:,:,:],axes=[0,1,2])
            dHraux = None
        else:
            for l in xrange(3):
                for ispin in xrange(nspin):
                    for n in xrange(nawf):
                        for m in xrange(nawf):
                            fft = pyfftw.FFTW(dHRaux[:,:,:,l,n,m,ispin],dHksp[:,:,:,l,n,m,ispin],axes=(0,1,2), \
                            direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                            dHksp[:,:,:,l,n,m,ispin] = fft()
                            for lp in xrange(3):
                                fft = pyfftw.FFTW(d2HRaux[:,:,:,l,lp,n,m,ispin],d2Hksp[:,:,:,l,lp,n,m,ispin],axes=(0,1,2), \
                                direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                                d2Hksp[:,:,:,l,lp,n,m,ispin] = fft()

    return(dHksp,d2Hksp)
