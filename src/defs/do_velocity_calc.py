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
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from kpnts_interpolation_mesh import *
from do_non_ortho import *
from do_momentum import *
from load_balancing import *
from constants import *
from clebsch_gordan import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_velocity_calc(HRs,E_k,v_kp,Rfft,ibrav,alat,a_vectors,b_vectors,dkres,bnd,ipol,jpol,spin_Hall,spol):
    # Compute bands on a selected path in the BZ
    # Define k-point mesh for bands interpolation
    kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
    nkpi=kq.shape[1]
    for n in xrange(nkpi):
        kq[:,n]=kq[:,n].dot(b_vectors)

    nawf,nawf,nk1,nk2,nk3,nspin = HRs.shape
    # Compute R*H(R)
    HRs = FFT.fftshift(HRs,axes=(2,3,4))
    dHRs  = np.zeros((3,nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
    Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')
    HRs = np.reshape(HRs,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
    dHRs  = np.zeros((3,nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
    for l in xrange(3):
        for ispin in xrange(nspin):
            for n in xrange(nawf):
                for m in xrange(nawf):
                    dHRs[l,n,m,:,ispin] = 1.0j*alat*ANGSTROM_AU*Rfft[:,l]*HRs[n,m,:,ispin]

    # Compute dH(k)/dk on the path

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nkpi)

    dHks  = np.zeros((3,nawf,nawf,nkpi,nspin),dtype=complex) # final data arrays
    Hks_aux  = np.zeros((3,nawf,nawf,nkpi,nspin),dtype=complex) # read data arrays from tasks

    Hks_aux[:,:,:,:,:] = band_loop_dH(ini_ik,end_ik,nspin,nawf,nkpi,dHRs,kq,Rfft)

    comm.Reduce(Hks_aux,dHks,op=MPI.SUM)

    # Compute momenta
    pks = np.zeros((nkpi,3,nawf,nawf,nspin),dtype=complex)
    for ik in xrange(nkpi):
        for ispin in xrange(nspin):
            for l in xrange(3):
                pks[ik,l,:,:,ispin] = np.conj(v_kp[ik,:,:,ispin].T).dot \
                            (dHks[l,:,:,ik,ispin]).dot(v_kp[ik,:,:,ispin])
    spin_orbit = False
    if spin_Hall:
        # Compute spin current matrix elements
        # Pauli matrices (x,y,z)
        sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
        if spin_orbit:
            # Spin operator matrix 
            Sj = np.zeros((nawf,nawf),dtype=complex)
            for i in xrange(nawf/2):
                Sj[i,i] = sP[spol][0,0]
                Sj[i,i+1] = sP[spol][0,1]
            for i in xrange(nawf/2,nawf):
                Sj[i,i-1] = sP[spol][1,0]
                Sj[i,i] = sP[spol][1,1]
                # NOTE: The above works if spin_orbit == True
        else:
            # Testing on S_z
            Sj = clebsch_gordan()

        jdHks = np.zeros((3,nawf,nawf,nkpi,nspin),dtype=complex)
        for ik in xrange(nkpi):
            for ispin in xrange(nspin):
                for l in xrange(3):
                        jdHks[l,:,:,ik,ispin] = \
                            0.5*(np.dot(Sj,dHks[l,:,:,ik,ispin])+np.dot(dHks[l,:,:,ik,ispin],Sj))

        jks = np.zeros((nkpi,3,nawf,nawf,nspin),dtype=complex)
        for ik in xrange(nkpi):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    jks[ik,l,:,:,ispin] = np.conj(v_kp[ik,:,:,ispin].T).dot \
                                (jdHks[l,:,:,ik,ispin]).dot(v_kp[ik,:,:,ispin])

        Omj_znk = np.zeros((nkpi,nawf),dtype=float)
        Omj_zk = np.zeros((nkpi),dtype=float)

    # Compute Berry curvature
    ########NOTE The indeces of the polarizations (x,y,z) should be changed according to the direction of the magnetization
    deltab = 0.05
    Om_znk = np.zeros((nkpi,nawf),dtype=float)
    Om_zk = np.zeros((nkpi),dtype=float)
    for ik in xrange(nkpi):
        for n in xrange(nawf):
            for m in xrange(nawf):
                if m!= n:
                    Om_znk[ik,n] += -1.0*np.imag(pks[ik,jpol,n,m,0]*pks[ik,ipol,m,n,0]-pks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]) / \
                    ((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
                    if spin_Hall:
                        #Omj_znk[ik,n] += -1.0*np.imag(jks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]-jks[ik,jpol,n,m,0]*pks[ik,ipol,m,n,0]) / \
                        #((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
                        Omj_znk[ik,n] += -2.0*np.imag(jks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]) / \
                        ((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
        Om_zk[ik] = np.sum(Om_znk[ik,:]*(0.5 * (-np.sign(E_k[ik,:,0]) + 1)))  # T=0.0K
        if spin_Hall: Omj_zk[ik] = np.sum(Omj_znk[ik,:]*(0.5 * (-np.sign(E_k[ik,:,0]) + 1)))  # T=0.0K

    if rank == 0:
        f=open('Omega_z'+'.dat','w')
        for ik in xrange(nkpi):
            f.write('%3d  %.5f \n' %(ik,-Om_zk[ik]))
        f.close()
        if spin_Hall:
            f=open('Omegaj_z'+'.dat','w')
            for ik in xrange(nkpi):
                f.write('%3d  %.5f \n' %(ik,Omj_zk[ik]))
            f.close()

    if rank == 0:
        velk = np.zeros((nkpi,3,nawf,nspin),dtype=float)
        for n in xrange(nawf):
            velk[:,:,n,:] = np.real(pks[:,:,n,n,:])
        for ispin in xrange(nspin):
            for l in xrange(3):
                f=open('velocity_'+str(l)+'_'+str(ispin)+'.dat','w')
                for ik in xrange(nkpi):
                    s="%d\t"%ik
                    for  j in velk[ik,l,:bnd,ispin]:s += "%3.5f\t"%j
                    s+="\n"
                    f.write(s)
                f.close()

    return()

def band_loop_dH(ini_ik,end_ik,nspin,nawf,nkpi,dHRaux,kq,R):

    auxh = np.zeros((3,nawf,nawf,nkpi,nspin),dtype=complex)

    for ik in xrange(ini_ik,end_ik):
        for ispin in xrange(nspin):
            for l in xrange(3):
                auxh[l,:,:,ik,ispin] = np.sum(dHRaux[l,:,:,:,ispin]*np.exp(2.0*np.pi*kq[:,ik].dot(R[:,:].T)*1j),axis=2)

    return(auxh)

