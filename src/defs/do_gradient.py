#
# PAOpy
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

from scipy import fftpack as FFT
import numpy as np
import sys, time
import pyfftw
import multiprocessing

from get_R_grid_fft import *

def do_gradient(Hksp,a_vectors,alat,nthread):
    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------

    nk1,nk2,nk3,nawf,nawf,nspin = Hksp.shape
    scipy = False
    if scipy :
        # fft grid in R shifted to have (0,0,0) in the center
        _,Rfft,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

        HRaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
        HRaux[:,:,:,:,:,:] = FFT.ifftn(Hksp[:,:,:,:,:,:],axes=[0,1,2])
        HRaux = FFT.fftshift(HRaux,axes=(0,1,2))
        # Compute R*H(R)
        dHRaux  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
        Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')
        HRaux = np.reshape(HRaux,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
        dHRaux  = np.zeros((nk1*nk2*nk3,3,nawf,nawf,nspin),dtype=complex)
        for l in xrange(3):
            for ispin in xrange(nspin):
                for n in xrange(nawf):
                    for m in xrange(nawf):
                        dHRaux[:,l,n,m,ispin] = 1.0j*alat*Rfft[:,l]*HRaux[:,n,m,ispin]
        dHRaux = np.reshape(dHRaux,(nk1,nk2,nk3,3,nawf,nawf,nspin),order='C')
        # Compute dH(k)/dk
        dHksp  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
        dHksp[:,:,:,:,:,:,:] = FFT.fftn(dHRaux[:,:,:,:,:,:,:],axes=[0,1,2])
        dHraux = None
    else:
        # fft grid in R shifted to have (0,0,0) in the center
        _,Rfft,_,_,_ = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

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
        for l in xrange(3):
            #aux1 = np.zeros(nk1*nk2*nk3,dtype=float)
            #aux1 = Rfft[:,l]
            # Compute R*H(R)
            dHRaux  = np.zeros((nk1*nk2*nk3,3,nawf,nawf,nspin),dtype=complex)
            for ispin in xrange(nspin):
                for n in xrange(nawf):
                    for m in xrange(nawf):
                        #aux2 = np.zeros(nk1*nk2*nk3,dtype=complex)
                        #aux2 = HRaux[:,n,m,ispin]
                        #dHRaux[:,l,n,m,ispin] = ne.evaluate('1.0j*alat*aux1*aux2')
                        dHRaux[:,l,n,m,ispin] = 1.0j*alat*Rfft[:,l]*HRaux[:,n,m,ispin]
            dHRaux = np.reshape(dHRaux,(nk1,nk2,nk3,3,nawf,nawf,nspin),order='C')

            # Compute dH(k)/dk
            for ispin in xrange(nspin):
                for n in xrange(nawf):
                    for m in xrange(nawf):
                        fft = pyfftw.FFTW(dHRaux[:,:,:,l,n,m,ispin],dHksp[:,:,:,l,n,m,ispin],axes=(0,1,2), \
                        direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                        dHksp[:,:,:,l,n,m,ispin] = fft()
    return(dHksp)
