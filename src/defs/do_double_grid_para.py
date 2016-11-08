#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys, time

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

sys.path.append('./')

from zero_pad import zero_pad

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_double_grid(nfft1,nfft2,nfft3,HRaux,SRaux,read_S):
    # Fourier interpolation on extended grid (zero padding)
    nawf = HRaux.shape[0]
    nk1 = HRaux.shape[2]
    nk2 = HRaux.shape[3]
    nk3 = HRaux.shape[4]
    nspin = HRaux.shape[5]
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    nktotp= nk1p*nk2p*nk3p
    if rank == 0: print('Number of k vectors for zero padding Fourier interpolation ',nktotp)

    # Extended R to k (with zero padding)
    HRauxp  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
    SRauxp  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p),dtype=complex)
    Hksp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
    Sksp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p),dtype=complex)
    aux = np.zeros((nk1,nk2,nk3),dtype=complex)
    idn = np.zeros((nawf,nawf),dtype=int)

    for ispin in range(nspin):
        nw = 0
        for i in range(nawf):
            for j in range(nawf):
                aux = HRaux[i,j,:,:,:,ispin]
                HRauxp[nw,:,:,:,ispin] = zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3)
                if read_S and ispin == 0:
                    aux = HRaux[i,j,:,:,:,ispin]
                    SRauxp[nw,:,:,:] = zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3)
                idn[i,j] = nw
                nw += 1

    Hks  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
    Sks  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p),dtype=complex)
    Hksaux  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p,nspin,1),dtype=complex)
    Sksaux  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p,1),dtype=complex)
    Hksaux1  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p,nspin,1),dtype=complex)
    Sksaux1  = np.zeros((nawf*nawf,nk1p,nk2p,nk3p,1),dtype=complex)

    # Load balancing
    ini_i = np.zeros((size),dtype=int)
    end_i = np.zeros((size),dtype=int)
    splitsize = 1.0/size*nawf*nawf
    for i in range(size):
        ini_i[i] = int(round(i*splitsize))
        end_i[i] = int(round((i+1)*splitsize))
    ini_in = ini_i[rank]
    end_in = end_i[rank]

    Hksaux[:,:,:,:,:,0] = do_fft3d_H(ini_in,end_in,HRauxp,nspin)

    if rank == 0:
        Hks[:,:,:,:,:]=Hksaux[:,:,:,:,:,0]
        for i in range(1,size):
            comm.Recv(Hksaux1,ANY_SOURCE)
            Hks[:,:,:,:,:] += Hksaux1[:,:,:,:,:,0]
    else:
        comm.Send(Hksaux,0)
    Hks = comm.bcast(Hks)

    if read_S:
        Sksaux[:,:,:,:,0] = do_fft3d_S(ini_in,end_in,SRauxp)

        if rank == 0:
            Sks[:,:,:,:]=Sksaux[:,:,:,:,0]
            for i in range(1,size):
                comm.Recv(Sksaux1,ANY_SOURCE)
                Sks[:,:,:,:] += Sksaux1[:,:,:,:,0]
        else:
            comm.Send(Sksaux,0)
        Sks = comm.bcast(Sks)

    for i in range(nawf):
        for j in range(nawf):
            Hksp[i,j,:,:,:,:] = Hks[idn[i,j],:,:,:,:]
            Sksp[i,j,:,:,:] = Sks[idn[i,j],:,:,:]

    nk1 = nk1p
    nk2 = nk2p
    nk3 = nk3p
    aux = None
    return(Hksp,Sksp,nk1,nk2,nk3)

def do_fft3d_H(ini_in,end_in,HRauxp,nspin):

    nwx,nk1p,nk2p,nk3p,nspin = HRauxp.shape

    Hksaux  = np.zeros((nwx,nk1p,nk2p,nk3p,nspin),dtype=complex)

#   for n in range(ini_in,end_in):
#       for ispin in range(nspin):
    Hksaux[ini_in:end_in,:,:,:,:] = FFT.fftn(HRauxp[ini_in:end_in,:,:,:,:],axes=[1,2,3])

    return(Hksaux)

def do_fft3d_S(ini_in,end_in,SRauxp,nspin,read_S):

    nwx,nk1p,nk2p,nk3p = SRauxp.shape

    Sksaux  = np.zeros((nwx,nk1p,nk2p,nk3p),dtype=complex)

    for n in range(ini_in,end_in):
        Sksaux[n,:,:,:] = FFT.fftn(SRauxp[n,:,:,:])

    return(Sksaux)
