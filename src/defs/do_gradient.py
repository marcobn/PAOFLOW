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
    if using_cuda:
        return do_gradient_cuda(Hksp,a_vectors,alat,nthread,npool)
    else:
        return do_gradient_mpi(Hksp,a_vectors,alat,nthread,npool)


def do_gradient_cuda(Hksp,a_vectors,alat,nthread,npool):
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



        dHRaux = None
    return(dHksp)


def do_gradient_mpi(Hksp,a_vectors,alat,nthread,npool):

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

    # fft grid in R shifted to have (0,0,0) in the center
    _,Rfft,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)
    #reshape R grid and each proc's piece of Hr
    Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')

    #############################################################################################
    #############################################################################################
    #############################################################################################
    if rank==0:
        #roll the axes so nawf,nawf are first
        #only needed because we're splitting 
        #between procs on the first indice
        Hksp =  np.reshape(Hksp,(nk1,nk2,nk3,nawf*nawf,nspin),order="C")
        Hksp =  np.rollaxis(Hksp,3,0)
        #have flattened H(k) as first indice
        Hksp = np.ascontiguousarray(Hksp)

        dHksp = np.zeros((nk1*nk2*nk3,3,nawf*nawf,nspin),order="C",dtype=complex)
    else:
        Hksp=None
        dHksp=None

    #does npool separate splits of the TB hamiltonian matrix
    for pool in xrange(npool):        
        if rank==0:            
            #scatter first entry
            start_n,end_n=load_balancing(npool,pool,Hksp.shape[0])

            H_aux  = scatter_array(Hksp[start_n:end_n])
        else:
            H_aux  = scatter_array(None)

        ########################################
        ### real space grid replaces k space ###
        ########################################
        if scipyfft:
            for n in xrange(H_aux.shape[0]):
                for ispin in xrange(H_aux.shape[4]):
                    FFT.ifftn(H_aux[n,:,:,:,ispin],axes=(0,1,2),overwrite_x=True)
                    H_aux[n,:,:,:,ispin] = FFT.fftshift(H_aux[n,:,:,:,ispin],axes=(0,1,2))

        else:
            Hkaux_tri = np.copy(H_aux)
            for n in xrange(H_aux.shape[0]):
                for ispin in xrange(H_aux.shape[4]):
                    fft = pyfftw.FFTW(Hkaux_tri[n,:,:,:,ispin],H_aux[n,:,:,:,ispin],axes=(0,1,2),
                                      direction='FFTW_BACKWARD',flags=('FFTW_MEASURE', ),
                                      threads=nthread, planning_timelimit=None )

                    H_aux[n,:,:,:,ispin] = fft()
                    H_aux[n,:,:,:,ispin] = FFT.fftshift(H_aux[n,:,:,:,ispin],axes=(0,1,2))

        #############################################################################################
        #############################################################################################
        #############################################################################################

        num_n = H_aux.shape[0]

        #reshape Hr for multiplying by the three parts of Rfft grid
        H_aux  = np.reshape(H_aux,(num_n,nk1*nk2*nk3,nspin),order='C')
        dH_aux = np.zeros((num_n,3,nk1*nk2*nk3,nspin),dtype=complex,order='C')

        # Compute R*H(R)
        for ispin in xrange(nspin):
            for l in xrange(3):
                dH_aux[:,l,:,ispin] = 1.0j*alat*Rfft[:,l]*H_aux[...,ispin]


        H_aux=None


        #reshape for fft
        dH_aux = np.reshape(dH_aux,(num_n,3,nk1,nk2,nk3,nspin),order='C')
        # Compute dH(k)/dk

        if scipyfft:
            for n in xrange(dH_aux.shape[0]):
                for l in xrange(dH_aux.shape[1]):
                    for ispin in xrange(dH_aux.shape[5]):
                        FFT.fftn(dH_aux[n,l,:,:,:,ispin],axes=(0,1,2),overwrite_x=True)
                                 
        else:
            for n in xrange(dH_aux.shape[0]):
                for l in xrange(dH_aux.shape[1]):
                    for ispin in xrange(dH_aux.shape[5]):
                        fft = pyfftw.FFTW(dH_aux[n,l,:,:,:,ispin]
                                          ,dH_aux[n,l,:,:,:,ispin],axes=(0,1,2),
                                          direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ),
                                          threads=nthread, planning_timelimit=None )

                        dH_aux[n,l,:,:,:,ispin] = fft()


        #############################################################################################
        #############################################################################################
        #############################################################################################

        #gather the arrays into flattened dHk
        if rank==0:
            temp = np.zeros((end_n-start_n,3,nk1,nk2,nk3,nspin),order="C",dtype=complex)
            gather_array(temp,dH_aux)
            temp  = np.rollaxis(temp,0,5)    
            temp  = np.rollaxis(temp,0,4) 
            temp  = temp.reshape(nk1*nk2*nk3,3,end_n-start_n,nspin)

            dHksp[:,:,start_n:end_n,:] = np.copy(temp)
            temp = None
        else:
            gather_array(None,dH_aux)
        
        dH_aux=None

    return(dHksp)
