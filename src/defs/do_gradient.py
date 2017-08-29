#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
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
from collections import deque
from communication import *

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
        Hksp =  np.reshape(Hksp,(nawf*nawf,nk1,nk2,nk3,nspin),order="C")
        #get list of array chunks. 
        Hksp_list = deque(np.array_split(Hksp,npool,axis=0))

        start_n = 0
        end_n   = 0
        dHksp = np.zeros((nk1*nk2*nk3,3,nawf*nawf,nspin),order="C",dtype=complex)
        

    else:
        Hksp=None
    #does npool separate splits of the TB hamiltonian matrix
    for pool in xrange(npool):        
        if rank==0:
            print pool
        if rank==0:            
            #scatter first entry 
            nentry = Hksp_list[0].shape[0]
            Hkaux_tri  = scatter_array(np.ascontiguousarray(Hksp_list.popleft()))
        else:
            Hkaux_tri  = scatter_array(None)

        #real space 
        HRaux_tri  = np.zeros_like(Hkaux_tri)
        #set scipy fft for now
        #receiving slice of array

        if scipyfft:
            #scipyFFT doesn't like arrays that have over 2^31 elements...
            for n in xrange(HRaux_tri.shape[0]):
                for ispin in xrange(HRaux_tri.shape[4]):
                    HRaux_tri[n,:,:,:,ispin] = FFT.ifftn(Hkaux_tri[n,:,:,:,ispin],axes=(0,1,2))
                    HRaux_tri[n,:,:,:,ispin] = FFT.fftshift(HRaux_tri[n,:,:,:,ispin],axes=(0,1,2))

        else:
            for n in xrange(HRaux_tri.shape[0]):
                for ispin in xrange(HRaux_tri.shape[4]):
                    fft = pyfftw.FFTW(Hkaux_tri[n,:,:,:,ispin],HRaux_tri[n,:,:,:,ispin],axes=(0,1,2),
                                      direction='FFTW_BACKWARD',flags=('FFTW_MEASURE', ),
                                      threads=nthread, planning_timelimit=None )

                    HRaux_tri[n,:,:,:,ispin] = fft()
                    HRaux_tri[n,:,:,:,ispin] = FFT.fftshift(HRaux_tri[n,:,:,:,ispin],axes=(0,1,2))

        Hkaux_tri=None

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
        dHRaux_tri = np.zeros((H_parts,3,nk1*nk2*nk3,nspin),dtype=complex,order='C')
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
            for n in xrange(dHkaux_tri.shape[0]):
                for l in xrange(dHkaux_tri.shape[1]):
                    for ispin in xrange(dHkaux_tri.shape[5]):
                        dHkaux_tri[n,l,:,:,:,ispin] = FFT.fftn(dHRaux_tri[n,l,:,:,:,ispin],axes=(0,1,2))

        else:
            for n in xrange(dHkaux_tri.shape[0]):
                for l in xrange(dHkaux_tri.shape[1]):
                    for ispin in xrange(dHkaux_tri.shape[5]):
                        fft = pyfftw.FFTW(dHkaux_tri[n,l,:,:,:,ispin]
                                          ,dHRaux_tri[n,l,:,:,:,ispin],axes=(0,1,2),
                                          direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ),
                                          threads=nthread, planning_timelimit=None )

                        dHkaux_tri[n,l,:,:,:,ispin] = fft()

        dHraux_tri = None
        #############################################################################################
        #############################################################################################
        #############################################################################################

        #gather the arrays into flattened dHk
        if rank==0:
            temp = np.zeros((nentry,3,nk1,nk2,nk3,nspin),order="C",dtype=complex)
            gather_array(temp,dHkaux_tri)
            temp  = np.rollaxis(temp,0,5)    
            temp  = np.rollaxis(temp,0,4) 
            temp  = temp.reshape(nk1*nk2*nk3,3,nentry,nspin)

            end_n   += nentry
            dHksp[:,:,start_n:end_n,:] = np.copy(temp)
            start_n += nentry
            temp = None
        else:
            gather_array(None,dHkaux_tri)
        
    dHkaux_tri = None
    dHksp_tri=None

    if rank!=0:
        dHksp_tri=None


        dHRaux = None
    return(dHksp)
  except Exception as e:
    raise e

