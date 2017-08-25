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

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import *

from get_R_grid_fft import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    using_cuda = False
    scipyfft = False
    try:
        import inputfile
        using_cuda = inputfile.use_cuda
    except:
        pass

    if using_cuda:
        from cuda_fft import *
    else:
        try:
            import pyfftw
        except:
            from scipy import fftpack as FFT
            scipyfft = True

def do_gradient(Hksp,a_vectors,alat,nthread,npool):
  try:
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

        if using_cuda:
            HRaux = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
            HRaux[:,:,:,:,:,:] = cuda_ifftn(Hksp[:,:,:,:,:,:])

        elif scipyfft:
            HRaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
            HRaux[:,:,:,:,:,:] = FFT.ifftn(Hksp[:,:,:,:,:,:],axes=[0,1,2])

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
    else:
        dHksp  = None
        Rfft = None
        HRaux = None
        dHRaux  = None

    for pool in xrange(npool):
        ini_ip, end_ip = load_balancing(npool,pool,nktot)
        nkpool = end_ip - ini_ip

        if rank == 0:
            HRaux_split = HRaux[ini_ip:end_ip]
            dHRaux_split = dHRaux[ini_ip:end_ip]
            Rfft_split = Rfft[ini_ip:end_ip]
        else:
            HRaux_split = None
            dHRaux_split = None
            Rfft_split = None

        dHRaux1 = scatter_array(dHRaux_split)
        HRaux1 = scatter_array(HRaux_split)
        Rfftaux = scatter_array(Rfft_split)

        # Compute R*H(R)
        for l in xrange(3):
            for ispin in xrange(nspin):
                for n in xrange(nawf):
                    for m in xrange(nawf):
                        dHRaux1[:,l,n,m,ispin] = 1.0j*alat*Rfftaux[:,l]*HRaux1[:,n,m,ispin]

        gather_array(dHRaux_split, dHRaux1)

        if rank == 0:
            dHRaux[ini_ip:end_ip,:,:,:,:] = dHRaux_split[:,:,:,:,:]

    if rank == 0:
        dHRaux = np.reshape(dHRaux,(nk1,nk2,nk3,3,nawf,nawf,nspin),order='C')

        # Compute dH(k)/dk
        if using_cuda:
            dHksp  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
            dHksp[:,:,:,:,:,:,:] = cuda_fftn(dHRaux[:,:,:,:,:,:,:])

        elif scipyfft:
            dHksp  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
            for l in xrange(3):
                dHksp[:,:,:,l,:,:,:] = FFT.fftn(dHRaux[:,:,:,l,:,:,:],axes=[0,1,2])
                
        else:
            for l in xrange(3):
                for ispin in xrange(nspin):
                    for n in xrange(nawf):
                        for m in xrange(nawf):
                            fft = pyfftw.FFTW(dHRaux[:,:,:,l,n,m,ispin],dHksp[:,:,:,l,n,m,ispin],axes=(0,1,2), \
                            direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                            dHksp[:,:,:,l,n,m,ispin] = fft()

        dHRaux = None
    return(dHksp)
  except Exception as e:
    raise e
