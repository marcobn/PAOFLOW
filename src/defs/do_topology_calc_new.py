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
import sys,os

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

def do_topology_calc(HRs,SRs,non_ortho,kq,E_k,v_kp,R,Rfft,R_wght,idx,alat,b_vectors,nelec,bnd,Berry,ipol,jpol,spin_Hall,spol,spin_orbit,sh,nl,eff_mass,inputpath,npool):
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

        f=open(os.path.join(inputpath,'Z2'+'.dat'),'w')
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

    # Compute momenta and kinetic energy

    kq_aux = scatter_full(kq.T,npool)
    kq_aux = kq_aux.T

    # Compute R*H(R)
    HRs = FFT.fftshift(HRs,axes=(2,3,4))
    Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')
    HRs = np.reshape(HRs,(nawf*nawf,nk1*nk2*nk3,nspin),order='C')

    HRs = np.swapaxes(HRs,0,1)
    HRs = np.reshape(HRs,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
    HRs_aux = scatter_full(HRs,npool)
    Rfft_aux = scatter_full(Rfft,npool)

################################################################################################################################
################################################################################################################################
################################################################################################################################

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
        jks = np.zeros((kq_aux.shape[1],3,bnd,bnd,nspin),dtype=complex)

################################################################################################################################
################################################################################################################################
################################################################################################################################


    pks = np.zeros((kq_aux.shape[1],3,bnd,bnd,nspin),dtype=complex)
    for l in xrange(3):
        dHRs  = np.zeros((HRs_aux.shape[0],nawf,nawf,nspin),dtype=complex)
        for ispin in xrange(nspin):
            for n in xrange(nawf):
                for m in xrange(nawf):
                    dHRs[:,n,m,ispin] = 1.0j*alat*ANGSTROM_AU*Rfft_aux[:,l]*HRs_aux[:,n,m,ispin]



        # Compute dH(k)/dk on the path

        # Load balancing
        dHRs = gather_full(dHRs,npool)    

        if rank!=0:
            dHRs = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin),dtype=complex)            
        comm.Bcast(dHRs)

        dHRs = np.reshape(dHRs,(nk1*nk2*nk3,nawf*nawf,nspin),order='C')            
        dHRs = np.swapaxes(dHRs,0,1)
        dHRs = np.reshape(dHRs,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')            

        dHks_aux = np.zeros((kq_aux.shape[1],nawf,nawf,nspin),dtype=complex) # read data arrays from tasks

        dHks_aux[:,:,:,:] = band_loop_H(nspin,nawf,dHRs,kq_aux,Rfft)

        dHRs = None

        # Compute momenta
        for ik in xrange(dHks_aux.shape[0]):
            for ispin in xrange(nspin):
                pks[ik,l,:,:,ispin] = np.conj(v_kp[ik,:,:,ispin].T).dot \
                    (dHks_aux[ik,:,:,ispin]).dot(v_kp[ik,:,:,ispin])[:bnd,:bnd]


        if spin_Hall:
            for ik in xrange(pks.shape[0]):
                for ispin in xrange(nspin):
                    jks[ik,l,:,:,ispin] = (np.conj(v_kp[ik,:,:,ispin].T).dot \
                        (0.5*(np.dot(Sj,dHks_aux[ik,:,:,ispin])+np.dot(dHks_aux[ik,:,:,ispin],Sj))).dot(v_kp[ik,:,:,ispin]))[:bnd,:bnd]
        if spin_Hall:
            Omj_znk = np.zeros((pks.shape[0],bnd),dtype=float)
            Omj_zk = np.zeros((pks.shape[0],1),dtype=float)


################################################################################################################################
################################################################################################################################
################################################################################################################################

    if eff_mass == True: 
        tks = np.zeros((kq_aux.shape[1],3,3,bnd,bnd,nspin),dtype=complex)

        for l in xrange(3):
            for lp in xrange(3):
                d2HRs = np.zeros((HRs_aux.shape[0],nawf,nawf,nspin),dtype=complex)
                for ispin in xrange(nspin):
                    for n in xrange(nawf):
                        for m in xrange(nawf):
                            d2HRs[:,n,m,ispin] = -1.0*alat**2*ANGSTROM_AU**2*Rfft_aux[:,l]*Rfft_aux[:,lp]*HRs_aux[:,n,m,ispin]



                d2HRs = gather_full(d2HRs,npool)    
                if rank!=0:
                    d2HRs = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin),dtype=complex)            
                comm.Bcast(d2HRs)
                d2HRs = np.reshape(d2HRs,(nk1*nk2*nk3,nawf*nawf,nspin),order='C')            
                d2HRs = np.swapaxes(d2HRs,0,1)
                d2HRs = np.reshape(d2HRs,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')            


                # Compute d2H(k)/dk*dkp on the path
                d2Hks_aux = np.zeros((kq_aux.shape[1],nawf,nawf,nspin),dtype=complex) # read data arrays from tasks

                d2Hks_aux[:,:,:,:] = band_loop_H(nspin,nawf,d2HRs[:,:,:,:],kq_aux,Rfft)

                d2HRs = None

                # Compute kinetic energy

                for ik in xrange(d2Hks_aux.shape[0]):
                    for ispin in xrange(nspin):
                        tks[ik,l,lp,:,:,ispin] = (np.conj(v_kp[ik,:,:,ispin].T).dot \
                            (d2Hks_aux[ik,:,:,ispin]).dot(v_kp[ik,:,:,ispin]))[:bnd,:bnd]


                d2Hks_aux=None



        # Compute effective mass
        mkm1 = np.zeros((tks.shape[0],bnd,3,3,nspin),dtype=complex)
        for ik in xrange(tks.shape[0]):
            for ispin in xrange(nspin):
                for n in xrange(bnd):
                    for m in xrange(bnd):
                        if m != n:
                            mkm1[ik,n,ipol,jpol,ispin] += (pks[ik,ipol,n,m,ispin]*pks[ik,jpol,m,n,ispin]+pks[ik,jpol,n,m,ispin]*pks[ik,ipol,m,n,ispin]) / \
                                                        (E_k[ik,n,ispin]-E_k[ik,m,ispin]+1.e-16)
                        else:
                            mkm1[ik,n,ipol,jpol,ispin] += tks[ik,ipol,jpol,n,n,ispin]


        tks=None

        mkm1 = gather_full(mkm1,npool)

        #mkm1 *= ELECTRONVOLT_SI**2/H_OVER_TPI**2*ELECTRONMASS_SI
        if rank == 0:
            for ispin in xrange(nspin):
                f=open(os.path.join(inputpath,'effmass'+'_'+str(LL[ipol])+str(LL[jpol])+'_'+str(ispin)+'.dat'),'w')
                for ik in xrange(nkpi):
                    s="%d\t"%ik
                    for  j in np.real(mkm1[ik,:bnd,ipol,jpol,ispin]):s += "% 3.5f\t"%j
                    s+="\n"
                    f.write(s)
                f.close()

        mkm1=None

################################################################################################################################
################################################################################################################################
################################################################################################################################    
    HRs_aux = None
    HRs = None
    




        #if rank == 0:
        #    plt.matshow(abs(tks[0,ipol,jpol,:,:,0]))
        #    plt.colorbar()
        #    plt.show()



    # Compute Berry curvature
    if Berry or spin_Hall:
        deltab = 0.05
        mu = -0.2 # chemical potential in eV)
        Om_znk = np.zeros((pks.shape[0],bnd),dtype=float)
        Om_zk = np.zeros((pks.shape[0],1),dtype=float)
        for ik in xrange(pks.shape[0]):
            for n in xrange(bnd):
                for m in xrange(bnd):
                    if m!= n:
                        if Berry:
                            Om_znk[ik,n] += -1.0*np.imag(pks[ik,jpol,n,m,0]*pks[ik,ipol,m,n,0]-pks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]) / \
                            ((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
                        if spin_Hall:
                            Omj_znk[ik,n] += -2.0*np.imag(jks[ik,ipol,n,m,0]*pks[ik,jpol,m,n,0]) / \
                            ((E_k[ik,m,0] - E_k[ik,n,0])**2 + deltab**2)
            Om_zk[ik] = np.sum(Om_znk[ik,:]*(0.5 * (-np.sign(E_k[ik,:bnd,0]) + 1)))  # T=0.0K
            if spin_Hall: Omj_zk[ik] = np.sum(Omj_znk[ik,:]*(0.5 * (-np.sign(E_k[ik,:bnd,0]-mu) + 1)))  # T=0.0K

    if Berry:
        Om_zk = gather_full(Om_zk,npool)
        if rank == 0:
            f=open(os.path.join(inputpath,'Omega_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
            for ik in xrange(nkpi):
                f.write('%3d  %.5f \n' %(ik,-Om_zk[ik,0]))
            f.close()
    if spin_Hall:
        Omj_zk = gather_full(Omj_zk,npool)
        if rank == 0:
            f=open(os.path.join(inputpath,'Omegaj_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
            for ik in xrange(nkpi):
                f.write('%3d  %.5f \n' %(ik,Omj_zk[ik,0]))
            f.close()

    pks = gather_full(pks,npool)
    if rank == 0:
        if spin_orbit: bnd *= 2
        velk = np.zeros((nkpi,3,bnd,nspin),dtype=float)
        for n in xrange(bnd):
            velk[:,:,n,:] = np.real(pks[:,:,n,n,:])
        for ispin in xrange(nspin):
            for l in xrange(3):
                f=open(os.path.join(inputpath,'velocity_'+str(l)+'_'+str(ispin)+'.dat'),'w')
                for ik in xrange(nkpi):
                    s="%d\t"%ik
                    for  j in velk[ik,l,:bnd,ispin]:s += "%3.5f\t"%j
                    s+="\n"
                    f.write(s)
                f.close()

    return()
def band_loop_H(nspin,nawf,HRaux,kq,R):

    kdot = np.zeros((kq.shape[1],R.shape[0]),dtype=complex,order="C")
    kdot = np.tensordot(R,2.0j*np.pi*kq,axes=([1],[0]))
    np.exp(kdot,kdot)

    auxh = np.zeros((nawf,nawf,kq.shape[1],nspin),dtype=complex,order="C")

    for ispin in xrange(nspin):
        auxh[:,:,:,ispin]=np.tensordot(HRaux[:,:,:,ispin],kdot,axes=([2],[0]))

    kdot  = None
    auxh = np.transpose(auxh,(2,0,1,3))
    return auxh


# def band_loop_dH(nspin,nawf,dHRaux,kq,R):

#     auxh = np.zeros((kq.shape[1],3,nawf,nawf,nspin),dtype=complex)

#     for ik in xrange(kq.shape[1]):
#         for ispin in xrange(nspin):
#             for l in xrange(3):
#                 auxh[ik,l,:,:,ispin] = np.sum(dHRaux[:,:,l,:,ispin]*np.exp(2.0*np.pi*kq[:,ik].dot(R[:,:].T)*1j),axis=2)


#     return(auxh)




# def band_loop_dH_single(nspin,nawf,dHRaux,kq,R):

#     auxh = np.zeros((kq.shape[1],nawf,nawf,nspin),dtype=complex)

#     for ik in xrange(kq.shape[1]):
#         for ispin in xrange(nspin):
#             auxh[ik,:,:,ispin] = np.sum(dHRaux[:,:,:,ispin]*np.exp(2.0*np.pi*kq[:,ik].dot(R[:,:].T)*1j),axis=2)


#     return(auxh)

