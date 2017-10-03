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
from do_eigh_calc import *

import pfaffian as pf

import matplotlib.pyplot as plt

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_topology_calc(HRs,SRs,non_ortho,kq,E_k,v_kp,R,Rfft,R_wght,idx,alat,b_vectors,nelec,bnd,Berry,ipol,jpol,spin_Hall,spol,spin_orbit,sh,nl,inputpath,npool):
    # Compute Z2 invariant and topological properties on a selected path in the BZ

    nkpi=kq.shape[1]
    nawf,nawf,nk1,nk2,nk3,nspin = HRs.shape

    # Compute Z2 according to Fu, Kane and Mele (2007)
    # Define TRIM points in 2(0-3)/3D(0-7)
    if nspin == 1 and spin_Hall:
        nktrim = 16
        ktrim = np.zeros((nktrim,3),dtype=float)
        ktrim[0] = np.zeros(3,dtype=float)                                  #0 0 0 0
        ktrim[1] = b_vectors[0,:]/2.0                                       #1 1 0 0
        ktrim[2] = b_vectors[1,:]/2.0                                       #2 0 1 0
        ktrim[3] = b_vectors[0,:]/2.0+b_vectors[1,:]/2.0                    #3 1 1 0
        ktrim[4] = b_vectors[2,:]/2.0                                       #4 0 0 1
        ktrim[5] = b_vectors[1,:]/2.0+b_vectors[2,:]/2.0                    #5 0 1 1
        ktrim[6] = b_vectors[2,:]/2.0+b_vectors[0,:]/2.0                    #6 1 0 1
        ktrim[7] = b_vectors[0,:]/2.0+b_vectors[1,:]/2.0+b_vectors[2,:]/2.0 #7 1 1 1
        ktrim[8:16] = -ktrim[:8]
        # Compute eigenfunctions at the TRIM points
        E_ktrim,v_ktrim = do_eigh_calc(HRs,SRs,ktrim,R_wght,R,idx,non_ortho)
        # Define time reversal operator
        theta = -1.0j*clebsch_gordan(nawf,sh,nl,1)
        wl = np.zeros((nktrim/2,nawf,nawf),dtype=complex)
        for ik in xrange(nktrim/2):
            wl[ik,:,:] = np.conj(v_ktrim[ik,:,:,0].T).dot(theta).dot(np.conj(v_ktrim[ik+nktrim/2,:,:,0]))
            wl[ik,:,:] = wl[ik,:,:]-wl[ik,:,:].T  # enforce skew symmetry
        delta_ik = np.zeros(nktrim/2,dtype=complex)
        for ik in xrange(nktrim/2):
            delta_ik[ik] = pf.pfaffian(wl[ik,:nelec,:nelec])/np.sqrt(LAN.det(wl[ik,:nelec,:nelec]))

        f=open(inputpath+'Z2'+'.dat','w')
        p2D = np.real(np.prod(delta_ik[:4]))
        if p2D+1.0 < 1.e-5:
            v0 = 1
        elif p2D-1.0 < 1.e-5:
            v0 = 0
        f.write('2D case: v0 = %1d \n' %(v0))
        p3D = np.real(np.prod(delta_ik))
        if p3D+1.0 < 1.e-5:
            v0 = 1
        elif p3D-1.0 < 1.e-5:
            v0 = 0
        p3D = delta_ik[1]*delta_ik[3]*delta_ik[6]*delta_ik[7]
        if p3D+1.0 < 1.e-5:
            v1 = 1
        elif p3D-1.0 < 1.e-5:
            v1 = 0
        p3D = delta_ik[2]*delta_ik[3]*delta_ik[5]*delta_ik[7]
        if p3D+1.0 < 1.e-5:
            v2 = 1
        elif p3D-1.0 < 1.e-5:
            v2 = 0
        p3D = delta_ik[4]*delta_ik[6]*delta_ik[5]*delta_ik[7]
        if p3D+1.0 < 1.e-5:
            v3 = 1
        elif p3D-1.0 < 1.e-5:
            v3 = 0
        f.write('3D case: v0;v1,v2,v3 = %1d;%1d,%1d,%1d \n' %(v0,v1,v2,v3))
        f.close()

    # Compute momenta

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

    dHks = gather_full(Hks_aux,npool)



    #if rank == 0: 
    #    plt.matshow(abs(dHks[0,:,:,1445,0]))
    #    plt.colorbar()
    #    plt.show()

    # Compute momenta
    pks = np.zeros((nkpi,3,nawf,nawf,nspin),dtype=complex)
    for ik in xrange(nkpi):
        for ispin in xrange(nspin):
            for l in xrange(3):
                pks[ik,l,:,:,ispin] = np.conj(v_kp[ik,:,:,ispin].T).dot \
                            (dHks[l,:,:,ik,ispin]).dot(v_kp[ik,:,:,ispin])

    #if rank == 0: 
    #    plt.matshow(abs(pks[1445,0,:,:,0]))
    #    plt.colorbar()
    #    plt.show()

    if spin_Hall:
        # Compute spin current matrix elements
        # Pauli matrices (x,y,z)
        sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
        if spin_orbit:
            # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
            Sj = np.zeros((nawf,nawf),dtype=complex)
            for i in xrange(nawf/2):
                Sj[i,i] = sP[spol][0,0]
                Sj[i,i+1] = sP[spol][0,1]
            for i in xrange(nawf/2,nawf):
                Sj[i,i-1] = sP[spol][1,0]
                Sj[i,i] = sP[spol][1,1]
        else:
            # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
            Sj = clebsch_gordan(nawf,sh,nl,spol)

        #jdHks = np.zeros((3,nawf,nawf,nkpi,nspin),dtype=complex)
        jks = np.zeros((nkpi,3,nawf,nawf,nspin),dtype=complex)
        for ik in xrange(nkpi):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    jks[ik,l,:,:,ispin] = np.conj(v_kp[ik,:,:,ispin].T).dot \
                                (0.5*(np.dot(Sj,dHks[l,:,:,ik,ispin])+np.dot(dHks[l,:,:,ik,ispin],Sj))).dot(v_kp[ik,:,:,ispin])

        Omj_znk = np.zeros((nkpi,nawf),dtype=float)
        Omj_zk = np.zeros((nkpi),dtype=float)

    # Compute Berry curvature
    if Berry or spin_Hall:
        deltab = 0.05
        mu = -0.2 # chemical potential in eV)
        Om_znk = np.zeros((nkpi,nawf),dtype=float)
        Om_zk = np.zeros((nkpi),dtype=float)
        for ik in xrange(nkpi):
            for n in xrange(nawf):
                for m in xrange(nawf):
                    if m!= n:
                        if Berry:
                            Om_znk[ik,n] += -1.0*np.imag(pks[ik,jpol,n,m,0]*pks[ik,ipol,m,n,0]-pks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]) / \
                            ((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
                        if spin_Hall:
                            Omj_znk[ik,n] += -2.0*np.imag(jks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]) / \
                            ((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
            Om_zk[ik] = np.sum(Om_znk[ik,:]*(0.5 * (-np.sign(E_k[ik,:,0]) + 1)))  # T=0.0K
            if spin_Hall: Omj_zk[ik] = np.sum(Omj_znk[ik,:]*(0.5 * (-np.sign(E_k[ik,:,0]-mu) + 1)))  # T=0.0K

    if rank == 0:
        if Berry:
            f=open(inputpath+'Omega_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
            for ik in xrange(nkpi):
                f.write('%3d  %.5f \n' %(ik,-Om_zk[ik]))
            f.close()
        if spin_Hall:
            f=open(inputpath+'Omegaj_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
            for ik in xrange(nkpi):
                f.write('%3d  %.5f \n' %(ik,Omj_zk[ik]))
            f.close()

    if rank == 0:
        if spin_orbit: bnd *= 2
        velk = np.zeros((nkpi,3,nawf,nspin),dtype=float)
        for n in xrange(nawf):
            velk[:,:,n,:] = np.real(pks[:,:,n,n,:])
        for ispin in xrange(nspin):
            for l in xrange(3):
                f=open(inputpath+'velocity_'+str(l)+'_'+str(ispin)+'.dat','w')
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

