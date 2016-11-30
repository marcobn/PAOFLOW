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
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

#########################################################################################
#                                                                                       #
#                                    AFLOWpi(TB) functions                              #
#                                                                                       #
#########################################################################################

from __future__ import print_function
from scipy import fftpack as FFT
from scipy import linalg as LA
from numpy import linalg as LAN
import cmath
import xml.etree.ElementTree as ET
import numpy as np
import sys, time, re
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import pyfftw
import pyfftw.interfaces.scipy_fftpack as sciFFTW
import multiprocessing

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#=====================================Physical constants

K_BOLTZMAN_SI    = 1.38066e-23        # J K^-1
K_BOLTZMAN_AU    = 3.1667e-6          # Hartree K^-1
K_BOLTZMAN_M1_AU = 315795.26e0        # Hartree^-1 K
FACTEM           = 315795.26e0        # Hartree^-1 K
H_OVER_TPI       = 1.054571e-34       # J sec

BOHR_RADIUS_SI   = 0.529177e-10       # m
BOHR_RADIUS_CM   = 0.529177e-8        # cm
BOHR_RADIUS_ANGS = 0.529177e0         # angstrom
ELECTRONMASS_SI  = 9.10953e-31        # Kg
ELECTRONMASS_UMA = 5.4858e-4          # uma

ELECTRONVOLT_SI  = 1.6021892e-19      # J
UMA_SI           = 1.66057e-27        # Kg
DEBYE_SI         = 3.33564e-30        # Coulomb meter
DEBYE_AU         = 0.393427228        # e * Bohr
ANGSTROM_AU      = 1.889727e0         # au
AU_TO_OHMCMM1    = 46000.0e0          # (ohm cm)^-1
AU_KB            = 294210.0e0         # Kbar
KB_AU            = 1.0e0/294210.0e0   # au
AU               = 27.211652e0        # eV
RYD              = 13.605826e0        # eV
SCMASS           = 1822.89e0          # uma to au
UMA_AU           = 1822.89e0          # au
AU_TERAHERTZ     = 2.418e-5           # THz
TERAHERTZ        = 2.418e-5           # from au to THz
AU_SEC           = 2.4189e-17         # sec

EPS0             = 1.0/(4.0 * 3.14159265358979323846 ) # vacuum dielectric constant in Ry
RYTOEV           = 13.6058e0     # conversion from Ry to eV
EVTORY           = 1.0/13.6058e0   # conversion from eV to Ry
AMCONV           = 1.66042e-24/9.1095e-28*0.5e0  # mass conversion: a.m.u to a.u. (Ry)
UAKBAR           = 147105.e0  # pressure conversion from Ry/(a.u)^3 to K
DEGTORAD         = (3.14159265358979323846)/(180)  # Degrees to radians
CM1TOEV          = 1.23981e-4  # cm^-1 to eV
EVTOCM1          = 1.0/1.23981e-4  # eV to cm^-1

E2               = 2.0 # e^2

# Logos
AFLOW = 'AFLOW'
p = u"\u03C0"
pp = p.encode('utf8')
TB = '(TB)'
AFLOWPITB = str(AFLOW)+str(pp)+str(TB)

C = u"\u00A9"
CC = C.encode('utf8')

#=====================================Build Hamiltonian from projections in reciprocal space
def build_Hks():
    Hks = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            my_eigs=my_eigsmat[:,ik,ispin]
            #Building the Hamiltonian matrix
            E = np.diag(my_eigs)
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            norms = 1/np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
            UU[:,:nbnds_norm] = UU[:,:nbnds_norm]*norms[:nbnds_norm]
            eta=shift
            # Choose only the eigenvalues that are below the energy shift
            bnd_ik=0
            for n in range(bnd):
                if my_eigs[n] <= eta:
                    bnd_ik += 1
            if bnd_ik == 0: sys.exit('no eigenvalues in selected energy range')
            ac = UU[:,:bnd_ik]  # filtering: bnd is defined by the projectabilities
            ee1 = E[:bnd_ik,:bnd_ik]
            #if bnd == nbnds:
            #    bd = np.zeros((nawf,1))
            #    ee2 = 0
            #else:
            #    bd = UU[:,bnd:nbnds]
            #    ee2= E[bnd:nbnds,bnd:nbnds]
            if shift_type ==0:
                #option 1 (PRB 2013)
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(np.conj(ac).T))
            elif shift_type==1:
                #option 2 (PRB 2016)
                aux_p=LA.inv(np.dot(np.conj(ac).T,ac))
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(aux_p).dot(np.conj(ac).T))
            elif shift_type==2:
                # no shift
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T)
            else:
                sys.exit('shift_type not recognized')
    return(Hks)

#=====================================Build projections
def build_Pn():
    Pn = 0.0
    for ispin in range(nspin):
        for ik in range(nkpnts):
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            Pn += np.real(np.sum(np.conj(UU)*UU,axis=0))/nkpnts/nspin
    return Pn

#=====================================Diagonalize the TB Hamiltonian on the extended MP grid
def calc_TB_eigs_vecs():

    #=====================================Diagonalizer
    def diago():

        nawf = aux.shape[1]
        ekp = np.zeros((nsize,nawf),dtype=float)
        ekv = np.zeros((nsize,nawf,nawf),dtype=complex)

        for n in range(nsize):
            eigval,eigvec = LAN.eigh(aux[n,:,:],UPLO='U')
            ekp[n,:] = np.real(eigval)
            ekv[n,:,:] = eigvec

    return(ekp,ekv)

#    index = None
#
#    if rank == 0:
#        nktot,nawf,nawf,nspin = Hksp.shape
#        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}
#
#    index = comm.bcast(index,root=0)
#
#    nktot = index['nktot']
#    nawf = index['nawf']
#    nspin = index['nspin']

    if rank == 0:
        eall = np.zeros((nawf*nktot,nspin),dtype=float)
        E_k = np.zeros((nktot,nawf,nspin),dtype=float)
        v_k = np.zeros((nktot,nawf,nawf,nspin),dtype=complex)
    else:
        eall = None
        E_k = None
        v_k = None
        Hks_split = None
        E_k_split = None
        v_k_split = None

    for pool in range (npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool
        #if rank == 0: print('running on ',npool,' pools for nkpool = ',nkpool)

        if rank == 0:
            Hks_split = np.array_split(Hksp,npool,axis=0)[pool]
            E_k_split = np.array_split(E_k,npool,axis=0)[pool]
            v_k_split = np.array_split(v_k,npool,axis=0)[pool]

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)

        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)
        v_kaux = np.zeros((nsize,nawf,nawf,nspin),dtype=complex)
        aux = np.zeros((nsize,nawf,nawf),dtype=complex)

        comm.barrier()
        comm.Scatter(Hks_split,aux,root=0)

        E_kaux, v_kaux = diago()

        comm.barrier()
        comm.Gather(E_kaux,E_k_split,root=0)
        comm.Gather(v_kaux,v_k_split,root=0)

        if rank == 0:
            E_k[pool*nkpool:(pool+1)*nkpool,:] = E_k_split[:,:]
            v_k[pool*nkpool:(pool+1)*nkpool,:,:] = v_k_split[:,:,:]

    if rank == 0:
        #f=open('eig_'+str(ispin)+'.dat','w')
        nall=0
        for n in range(nktot):
            for m in range(nawf):
                eall[nall,ispin]=E_k[n,m,ispin]
                #f.write('%.5f  %.5f \n' %(n,E_k[n,m,ispin]))
                nall += 1
        #f.close()


    return(eall,E_k,v_k)

#=====================================Calculate bands in 1D
def do_bands_calc_1D(Hkaux):
    # FFT interpolation along a single directions in the BZ

    nawf = Hksp.shape[0]
    nk1 = Hksp.shape[2]
    nk2 = Hksp.shape[3]
    nk3 = Hksp.shape[4]
    nspin = Hksp.shape[5]

    # Count points along symmetry direction
    nL = 0
    for ik1 in range(nk1):
        for ik2 in range(nk2):
            for ik3 in range(nk3):
                nL += 1

    Hkaux  = np.zeros((nawf,nawf,nL,nspin),dtype=complex)
    for ispin in range(nspin):
        for i in range(nawf):
            for j in range(nawf):
                nL=0
                for ik1 in range(nk1):
                    for ik2 in range(nk2):
                        for ik3 in range(nk3):
                            Hkaux[i,j,nL,ispin]=Hksp[i,j,ik1,ik2,ik3,ispin]
                            nL += 1

    # zero padding interpolation
    # k to R
    npad = 500
    HRaux  = np.zeros((nawf,nawf,nL,nspin),dtype=complex)
    for ispin in range(nspin):
        for i in range(nawf):
            for j in range(nawf):
                HRaux[i,j,:,ispin] = FFT.ifft(Hkaux[i,j,:,ispin])

    Hkaux = None
    Hkaux  = np.zeros((nawf,nawf,npad+nL,nspin),dtype=complex)
    HRauxp  = np.zeros((nawf,nawf,npad+nL,nspin),dtype=complex)

    for ispin in range(nspin):
        for i in range(nawf):
            for j in range(nawf):
                HRauxp[i,j,:(nL/2),ispin]=HRaux[i,j,:(nL/2),ispin]
                HRauxp[i,j,(npad+nL/2):,ispin]=HRaux[i,j,(nL/2):,ispin]
                Hkaux[i,j,:,ispin] = FFT.fft(HRauxp[i,j,:,ispin])

    # Print TB eigenvalues on interpolated mesh
    for ispin in range(nspin):
        write_TB_eigs(Hkaux,ispin)

    return()

#=====================================Calculate bands along a path in the BZ
def do_bands_calc(HRaux,SRaux,R_wght,R,idx,non_ortho,ibrav,alat,a_vectors,b_vectors,dkres):
    # Compute bands on a selected path in the BZ
    # Define k-point mesh for bands interpolation
    kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
    nkpi=kq.shape[1]
    for n in range(nkpi):
        kq [:,n]=kq[:,n].dot(b_vectors)

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nkpi)

    nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
    Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # final data arrays
    Hks_aux  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # read data arrays from tasks

    Hks_aux[:,:,:,:] = band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx)

    comm.Reduce(Hks_aux,Hks_int,op=MPI.SUM)

    Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
    if non_ortho:
        Sks_aux  = np.zeros((nawf,nawf,nkpi,1),dtype=complex)
        Sks_aux[:,:,:,0] = band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx)

        comm.Reduce(Sks_aux,Sks_int,op=MPI.SUM)

    if rank ==0:
        for ispin in range(nspin):
            write_TB_eigs(Hks_int,Sks_int,non_ortho,ispin)
    return()

#=====================================Calculate H on the path
def band_loop_H(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx):

    auxh = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)

    for ik in range(ini_ik,end_ik):
        for ispin in range(nspin):
            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                        auxh[:,:,ik,ispin] += HRaux[:,:,i,j,k,ispin]*phase

    return(auxh)

#=====================================Calculate S on the path
def band_loop_S(ini_ik,end_ik,nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx):

    auxs = np.zeros((nawf,nawf,nkpi),dtype=complex)

    for ik in range(ini_ik,end_ik):
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)

#=====================================Calculate Berry curvature and anomalous Hall conductivity
def do_Berry_curvature(iswitch):
    #----------------------
    # Compute Berry curvature on a selected path in the BZ
    #----------------------

    # Compute only Omega_z(k)

    if rank == 0:
        Om_znk = np.zeros((nk1*nk2*nk3,nawf),dtype=float)
    else:
        Om_znk = None

    for pool in range(npool):
        if nk1*nk2*nk3%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nk1*nk2*nk3/npool

        if rank == 0:
            pksp_split = np.array_split(pksp,npool,axis=0)[pool]
            E_k_split = np.array_split(E_k,npool,axis=0)[pool]
            Om_znk_split = np.array_split(Om_znk,npool,axis=0)[pool]
        else:
            pksp_split = None
            E_k_split = None
            Om_znk_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        E_kaux = np.zeros((nsize,nawf,nspin),dtype = float)
        Om_znkaux = np.zeros((nsize,nawf),dtype=float)

        comm.Barrier()
        comm.Scatter(pksp_split,pksaux,root=0)
        comm.Scatter(E_k_split,E_kaux,root=0)

        for nk in range(nsize):
            for n in range(nawf):
                for m in range(nawf):
                    if n!= m:
                        Om_znkaux[nk,n] += -2.0*np.imag(pksaux[nk,0,n,m,0]*pksaux[nk,1,m,n,0]) / \
                        (E_kaux[nk,n,0]**2 - E_kaux[nk,m,0]**2 + delta**2)
        comm.Barrier()
        comm.Gather(Om_znkaux,Om_znk_split,root=0)

        if rank == 0:
            Om_znk[pool*nkpool:(pool+1)*nkpool,:] = Om_znk_split[:,:]

    if rank == 0:
        Om_zk = np.zeros((nk1*nk2*nk3),dtype=float)
    else:
        Om_znk = None
        Om_zk = None

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nk1*nk2*nk3)
    nsize = end_ik-ini_ik

    Om_znkaux = np.zeros((nsize,nawf),dtype=float)
    Om_zkaux = np.zeros((nsize),dtype=float)
    E_kaux = np.zeros((nsize,nawf,nspin),dtype = float)

    comm.Barrier()
    comm.Scatter(Om_znk,Om_znkaux,root= 0)
    comm.Scatter(Om_zk,Om_zkaux,root= 0)
    comm.Scatter(E_k,E_kaux,root= 0)

    for nk in range(nsize):
        for n in range(nawf):
            if E_kaux[nk,n,0] <= 0.0:
                Om_zkaux[nk] += Om_znkaux[nk,n] #* 1.0/2.0 * 1.0/(1.0+np.cosh((E_k[n,nk,0]/temp)))/temp

    comm.Barrier()
    comm.Gather(Om_zkaux,Om_zk,root=0)

    ahc = None
    if rank == 0: ahc = -E2*np.sum(Om_zk)/float(nk1*nk2*nk3)

    if iswitch == 0:

        # Define k-point mesh for bands interpolation

        kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
        nkpi=kq.shape[1]
        for n in range(nkpi):
            kq [:,n]=kq[:,n].dot(b_vectors)


        if rank == 0:
            # Compute Om_zR
            Om_zR = np.zeros((nk1*nk2*nk3),dtype=float)
            Om_zRc = np.zeros((nk1,nk2,nk3),dtype=complex)
            Om_zk = np.reshape(Om_zk,(nk1,nk2,nk3),order='C')+1.j
            fft = pyfftw.FFTW(Om_zk,Om_zRc,axes=(0,1,2), direction='FFTW_BACKWARD',\
                        flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
            Om_zRc = fft()

            for i in range(nk1):
                for j in range(nk2):
                    for k in range(nk3):
                        n = k + j*nk3 + i*nk2*nk3
                        Om_zR[n] = np.real(Om_zRc[i,j,k])

            Om_zk_disp = np.zeros((nkpi),dtype=float)

        else:
            Om_zR = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nk1*nk2*nk3)
        nsize = end_ik-ini_ik

        for ik in range(nkpi):

            Om_zRaux = np.zeros(nsize,dtype=float)
            R_wghtaux = np.zeros(nsize,dtype=float)
            R_aux = np.zeros((nsize,3),dtype=float)
            Om_zk_sum = np.zeros(1,dtype=float)
            auxsum = np.zeros(1,dtype=float)

            comm.Barrier()
            comm.Scatter(R,R_aux,root=0)
            comm.Scatter(R_wght,R_wghtaux,root=0)
            comm.Scatter(Om_zR,Om_zRaux,root=0)

            for nk in range(nsize):
                phase=R_wghtaux[nk]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R_aux[nk,:])*1j)
                auxsum += np.real(Om_zRaux[nk]*phase)

            comm.Barrier()
            comm.Reduce(auxsum,Om_zk_sum,op=MPI.SUM)
            if rank == 0: Om_zk_disp[ik] = Om_zk_sum

        if rank == 0:
            f=open('Omega_z'+'.dat','w')
            for ik in range(nkpi):
                f.write('%3d  %.5f \n' %(ik,-Om_zk_disp[ik]))
            f.close()

    return(Om_zk,ahc)

#=====================================Compute Boltzmann tensor for transport
def do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin):
    # Compute the L_alpha tensors for Boltzmann transport

    global ene
    emin = -2.0 # To be read in input
    emax = 2.0
    de = (emax-emin)/500
    ene = np.arange(emin,emax,de,dtype=float)

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nktot)
    nsize = end_ik-ini_ik

    kq_wghtaux = np.zeros(nsize,dtype=float)
    velkpaux = np.zeros((nsize,3,nawf,nspin),dtype=float)
    E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)

    comm.Barrier()
    comm.Scatter(velkp,velkpaux,root=0)
    comm.Scatter(E_k,E_kaux,root=0)
    comm.Scatter(kq_wght,kq_wghtaux,root=0)

    L0 = np.zeros((3,3,ene.size),dtype=float)
    L0aux = np.zeros((3,3,ene.size),dtype=float)

    L1 = np.zeros((3,3,ene.size),dtype=float)
    L1aux = np.zeros((3,3,ene.size),dtype=float)

    L2 = np.zeros((3,3,ene.size),dtype=float)
    L2aux = np.zeros((3,3,ene.size),dtype=float)

    for i in range(3):
        for j in range(3):
            for ne in range(ene.size):
                for n in range(nawf):
                    L0aux[i,j,ne] += np.sum(1.0/temp * kq_wghtaux[:]*velkpaux[:,i,n,ispin]*velkpaux[:,j,n,ispin] * \
                        1.0/2.0 * 1.0/(1.0+np.cosh((E_kaux[:,n,ispin]-ene[ne])/temp)) * \
                        pow((E_kaux[:,n,ispin]-ene[ne]),0))
                    L1aux[i,j,ne] += np.sum(1.0/temp * kq_wghtaux[:]*velkpaux[:,i,n,ispin]*velkpaux[:,j,n,ispin] * \
                        1.0/2.0 * 1.0/(1.0+np.cosh((E_kaux[:,n,ispin]-ene[ne])/temp)) * \
                        pow((E_kaux[:,n,ispin]-ene[ne]),1))
                    L2aux[i,j,ne] += np.sum(1.0/temp * kq_wghtaux[:]*velkpaux[:,i,n,ispin]*velkpaux[:,j,n,ispin] * \
                        1.0/2.0 * 1.0/(1.0+np.cosh((E_kaux[:,n,ispin]-ene[ne])/temp)) * \
                        pow((E_kaux[:,n,ispin]-ene[ne]),2))

    comm.Barrier()
    comm.Reduce(L0aux,L0,op=MPI.SUM)
    comm.Reduce(L1aux,L1,op=MPI.SUM)
    comm.Reduce(L2aux,L2,op=MPI.SUM)

    return(ene,L0,L1,L2)

#=====================================Compute DOS
def do_dos_calc(eig,emin,emax,delta,netot,nawf,ispin):
    # DOS calculation with gaussian smearing

    #emin = np.min(eig)-1.0
    #emax = np.max(eig)-shift/2.0
    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1000
    ene = np.arange(emin,emax,de,dtype=float)

    # Load balancing
    ini_ie, end_ie = load_balancing(size,rank,netot)

    nsize = end_ie-ini_ie

    dos = np.zeros((ene.size),dtype=float)

    for ne in range(ene.size):

        dossum = np.zeros(1,dtype=float)
        aux = np.zeros(nsize,dtype=float)

        comm.Barrier()
        comm.Scatter(eig,aux,root=0)

        dosaux = np.sum(1.0/np.sqrt(np.pi)*np.exp(-((ene[ne]-aux)/delta)**2)/delta)

        comm.Barrier()
        comm.Reduce(dosaux,dossum,op=MPI.SUM)
        dos[ne] = dossum*float(nawf)/float(netot)

    if rank == 0:
        f=open('dos_'+str(ispin)+'.dat','w')
        for ne in range(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],dos[ne]))
        f.close()

    return

#=====================================Interpolate the Hamiltonian on the extended MP grid
def do_double_grid(nfft1,nfft2,nfft3,HRaux,nthread):
    # Fourier interpolation on extended grid (zero padding)
    if HRaux.shape[0] != 3 and HRaux.shape[1] == HRaux.shape[0]:
        nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
        nk1p = nfft1
        nk2p = nfft2
        nk3p = nfft3
        nfft1 = nfft1-nk1
        nfft2 = nfft2-nk2
        nfft3 = nfft3-nk3
        nktotp= nk1p*nk2p*nk3p

        # Extended R to k (with zero padding)
        HRauxp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
        Hksp  = np.zeros((nk1p,nk2p,nk3p,nawf,nawf,nspin),dtype=complex)
        aux = np.zeros((nk1,nk2,nk3),dtype=complex)

        for ispin in range(nspin):
            for i in range(nawf):
                for j in range(nawf):
                    aux = HRaux[i,j,:,:,:,ispin]
                    fft = pyfftw.FFTW(zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3),Hksp[:,:,:,i,j,ispin], axes=(0,1,2), direction='FFTW_FORWARD',\
                                flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                    Hksp[:,:,:,i,j,ispin] = fft()
    else:
        sys.exit('wrong dimensions in input array')

    nk1 = nk1p
    nk2 = nk2p
    nk3 = nk3p
    aux = None
    return(Hksp,nk1,nk2,nk3)


#=====================================Calculate the Re and Im part of the dielectric tensor
def do_epsilon(E_k,pksp,kq_wght,omega,delta,temp,ispin):
    # Compute the dielectric tensor

    emin = 0.1 # To be read in input
    emax = 10.0
    de = (emax-emin)/500
    ene = np.arange(emin,emax,de,dtype=float)

    #=======================
    # Im
    #=======================

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,kq_wght.size)

    epsi = np.zeros((3,3,ene.size),dtype=float)
    epsi_aux = np.zeros((3,3,ene.size,1),dtype=float)

    epsi_aux[:,:,:,0] = epsi_loop(ini_ik,end_ik,ene,E_k,pksp,kq_wght,omega,delta,temp,ispin)

    comm.Allreduce(epsi_aux,epsi,op=MPI.SUM)

    #=======================
    # Re
    #=======================

    # Load balancing
    ini_ie, end_ie = load_balancing(size,rank,ene.size)

    epsr = np.zeros((3,3,ene.size),dtype=float)
    epsr_aux = np.zeros((3,3,ene.size,1),dtype=float)

    epsr_aux[:,:,:,0] = epsr_kramkron(ini_ie,end_ie,ene,epsi)

    comm.Allreduce(epsr_aux,epsr,op=MPI.SUM)

    epsr += 1.0

    return(ene,epsi,epsr)

#=====================================Evaluate Im epsilon
def epsi_loop(ini_ik,end_ik,ene,E_k,pksp,kq_wght,omega,delta,temp,ispin):

    epsi = np.zeros((3,3,ene.size),dtype=float)

    arg = np.zeros((ene.size),dtype=float)
    raux = np.zeros((ene.size),dtype=float)

    for nk in range(ini_ik,end_ik):
        for n in range(pksp.shape[2]):
            arg2 = E_k[nk,n,ispin]/temp
            raux2 = 1.0/(np.exp(arg2)+1)
            for m in range(pksp.shape[2]):
                arg3 = E_k[nk,m,ispin]/temp
                raux3 = 1.0/(np.exp(arg3)+1)
                arg[:] = (ene[:] - ((E_k[nk,m,ispin]-E_k[nk,n,ispin])))/delta
                raux[:] = 1.0/np.sqrt(np.pi)*np.exp(-arg[:]**2)
                if n != m:
                    for i in range(3):
                        for j in range(3):
                            epsi[i,j,:] += 1.0/(ene[:]**2+delta**2) * \
                                    kq_wght[nk] /delta * raux[:] * (raux2 - raux3) * \
                                    abs(pksp[nk,i,n,m,ispin] * pksp[nk,j,m,n,ispin])
                else:
                    for i in range(3):
                        for j in range(3):
                            epsi[i,j,:] += 1.0/ene[:] * kq_wght[nk] * raux[:]/delta *  \
                                    1.0/2.0 * 1.0/(1.0+np.cosh((arg2)))/temp *    \
                                    abs(pksp[nk,i,n,m,ispin] * pksp[nk,j,m,n,ispin])

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)

    return(epsi)

#=====================================Evaluate Re epsilon via Kramers-Kroning
def epsr_kramkron(ini_ie,end_ie,ene,epsi):

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    for ie in range(ini_ie,end_ie):
        for i in range(3):
            for j in range(3):
                epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return(epsr)

#=====================================Compute the momentum operator in reciprocal space
def do_momentum():
    # calculate momentum vector

    if rank == 0:
        pksp = np.zeros((nktot,3,nawf,nawf,nspin),dtype=complex)
    else:
        pksp = None

    for pool in range(npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool

        if rank == 0:
            dHksp_split = np.array_split(dHksp,npool,axis=0)[pool]
            pks_split = np.array_split(pksp,npool,axis=0)[pool]
            vec_split = np.array_split(v_k,npool,axis=0)[pool]
        else:
            dHksp_split = None
            pks_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik
        if nkpool%nsize != 0: sys.exit('npool not compatible with nsize')

        dHkaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        pksaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype = complex)
        vecaux = np.zeros((nsize,nawf,nawf,nspin),dtype = complex)

        comm.Barrier()
        comm.Scatter(dHksp_split,dHkaux,root=0)
        comm.Scatter(pks_split,pksaux,root=0)
        comm.Scatter(vec_split,vecaux,root=0)

        for ik in range(nsize):
            for ispin in range(nspin):
                for l in range(3):
                    pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        comm.Gather(pksaux,pks_split,root=0)

        if rank == 0:
            pksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:] = pks_split[:,:,:,:,:,]

    return(pksp)

#=====================================Transform H to a non ortogonal basis
def do_non_ortho():
    # Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(1/2)*Hks*Sks^(1/2)+

    if len(Hks.shape) > 4:

        nawf = Hks.shape[0]
        nk1 = Hks.shape[2]
        nk2 = Hks.shape[3]
        nk3 = Hks.shape[4]
        nspin = Hks.shape[5]
        aux = np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
        saux = np.zeros((nawf,nawf,nk1*nk2*nk3),dtype=complex)
        idk = np.zeros((nk1,nk2,nk3),dtype=int)
        nkpnts = 0
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    aux[:,:,nkpnts,:] = Hks[:,:,i,j,k,:]
                    saux[:,:,nkpnts] = Sks[:,:,i,j,k]
                    idk[i,j,k] = nkpnts
                    nkpnts += 1

        S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in range(nkpnts):
            w, v = LAN.eigh(saux[:,:,ik],UPLO='U')
            #w, v = LA.eigh(saux[:,:,ik])
            w = np.sqrt(w)
            for j in range(nawf):
                for i in range(nawf):
                    S2k[i,j,ik] = v[i,j]*w[j]
            S2k[:,:,ik] = S2k[:,:,ik].dot(np.conj(v).T)

        Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
        for ispin in range(nspin):
            for ik in range(nkpnts):
                Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(aux[:,:,ik,ispin]).dot(S2k[:,:,ik])

        aux = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    aux[:,:,i,j,k,:]=Hks_no[:,:,idk[i,j,k],:]
        return(aux)

    else:

        nawf = Hks.shape[0]
        nkpnts = Hks.shape[2]
        nspin = Hks.shape[3]
        S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in range(nkpnts):
            w, v = LAN.eigh(Sks[:,:,ik],UPLO='U')
            #w, v = LA.eigh(Sks[:,:,ik])
            w = np.sqrt(w)
            for j in range(nawf):
                for i in range(nawf):
                    S2k[i,j,ik] = v[i,j]*w[j]
            S2k[:,:,ik] = S2k[:,:,ik].dot(np.conj(v).T)

        Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
        for ispin in range(nspin):
            for ik in range(nkpnts):
                Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])
        return(Hks_no)

#=====================================Transform H to an orthogonal basis
def do_ortho(Hks,Sks):
    # Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(1/2)*Hks*Sks^(1/2)+

    if len(Hks.shape) > 4:

        nawf = Hks.shape[0]
        nk1 = Hks.shape[2]
        nk2 = Hks.shape[3]
        nk3 = Hks.shape[4]
        nspin = Hks.shape[5]
        aux = np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
        saux = np.zeros((nawf,nawf,nk1*nk2*nk3),dtype=complex)
        idk = np.zeros((nk1,nk2,nk3),dtype=int)
        nkpnts = 0
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    aux[:,:,nkpnts,:] = Hks[:,:,i,j,k,:]
                    saux[:,:,nkpnts] = Sks[:,:,i,j,k]
                    idk[i,j,k] = nkpnts
                    nkpnts += 1

        S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in range(nkpnts):
            #w, v = LAN.eigh(saux[:,:,ik],UPLO='U')
            w, v = LA.eigh(saux[:,:,ik])
            w = 1.0/np.sqrt(w)
            for j in range(nawf):
                for i in range(nawf):
                    S2k[i,j,ik] = v[i,j]*w[j]
            S2k[:,:,ik] = S2k[:,:,ik].dot(np.conj(v).T)

        Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
        for ispin in range(nspin):
            for ik in range(nkpnts):
                Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(aux[:,:,ik,ispin]).dot(S2k[:,:,ik])

        aux = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    aux[:,:,i,j,k,:]=Hks_no[:,:,idk[i,j,k],:]
        return(aux)

    else:

        nawf = Hks.shape[0]
        nkpnts = Hks.shape[2]
        nspin = Hks.shape[3]
        S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in range(nkpnts):
            w, v = LAN.eigh(Sks[:,:,ik],UPLO='U')
            #w, v = LA.eigh(Sks[:,:,ik])
            w = 1.0/np.sqrt(w)
            for j in range(nawf):
                for i in range(nawf):
                    S2k[i,j,ik] = v[i,j]*w[j]
            S2k[:,:,ik] = S2k[:,:,ik].dot(np.conj(v).T)

        Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
        for ispin in range(nspin):
            for ik in range(nkpnts):
                Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])
        return(Hks_no)

#=====================================Add spin orbit correction
def do_spin_orbit_calc():

    nawf = HRaux.shape[0]
    nk1 = HRaux.shape[2]
    nk2 = HRaux.shape[3]
    nk3 = HRaux.shape[4]
    nspin = HRaux.shape[5]

    HR_double= np.zeros((2*nawf,2*nawf,nk1,nk2,nk3,nspin),dtype=complex)
    HR_soc_p = np.zeros((18,18),dtype=complex)  #Hardcoded do s,p,d only (18 orbitals per atom) - Must Change

    # nonmagnetic :  copy H at the upper (lower) left (right) of the double matrix HR_double
    if nspin == 1:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	      	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
    # magnetic :  copy H_up (H_down) at the upper (lower) left (right) of the double matrix 
    else:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	       	       =  HRaux[0:nawf,0:nawf,:,:,:,1]

    HR_soc_p =  soc_p(theta,phi)
    #HR_soc_d =  soc_d(theta,phi)

    M=9
    nt=natoms
    for n in range(nt):
        i=n*M
        j=(n+1)*M
        # Up-Up
        HR_double[i:j,i:j,0,0,0,0]                             = HR_double[i:j,i:j,0,0,0,0]                             + socStrengh[n,0]*HR_soc_p[0:9,0:9]
        # Down-Down
        HR_double[(i+nt*M):(j+nt*M),(i+nt*M):(j+nt*M),0,0,0,0] = HR_double[(i+nt*M):(j+nt*M),(i+nt*M):(j+nt*M),0,0,0,0] + socStrengh[n,0]*HR_soc_p[9:18,9:18]
        # Up-Down
        HR_double[i:j,(i+nt*M):(j+nt*M),0,0,0,0]               = HR_double[i:j,(i+nt*M):(j+nt*M),0,0,0,0]               + socStrengh[n,0]*HR_soc_p[0:9,9:18]
        # Down-Up
        HR_double[(i+nt*M):(j+nt*M),i:j,0,0,0,0]               = HR_double[(i+nt*M):(j+nt*M),i:j,0,0,0,0]               + socStrengh[n,0]*HR_soc_p[9:18,0:9]

    return(HR_double)


#=====================================SOC for p orbitals
def soc_p(theta,phi):

    # Hardcoded to s,p,d. This must change latter.
        HR_soc = np.zeros((18,18),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

	#Spin Up - Spin Up  part of the p-satets Hamiltonian
        HR_soc[1,2] = -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[1,3] =  0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[2,3] = -0.5*np.complex(0.0,cTheta)
        HR_soc[2,1]=np.conjugate(HR_soc[1,2])
        HR_soc[3,1]=np.conjugate(HR_soc[1,3])
        HR_soc[3,2]=np.conjugate(HR_soc[2,3])
	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[10:13,10:13] = - HR_soc[1:4,1:4] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[1,11] = -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[1,12] = -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[2,12] =  0.5*np.complex(0.0,sTheta)
        HR_soc[2,10] = -HR_soc[1,11]
        HR_soc[3,10] = -HR_soc[1,12]
        HR_soc[3,11] = -HR_soc[2,12]
	#Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[11,1]=np.conjugate(HR_soc[1,11])
        HR_soc[12,1]=np.conjugate(HR_soc[1,12])
        HR_soc[10,2]=np.conjugate(HR_soc[2,10])
        HR_soc[12,2]=np.conjugate(HR_soc[2,12])
        HR_soc[10,3]=np.conjugate(HR_soc[3,10])
        HR_soc[11,3]=np.conjugate(HR_soc[3,11])
	return(HR_soc)

#=====================================Generate fft grid of k vectors
def get_K_grid_fft(nk1,nk2,nk3,b_vectors):
    nktot = nk1*nk2*nk3
    Kint = np.zeros((3,nktot),dtype=float)
    K_wght = np.ones((nktot),dtype=float)
    K_wght /= nktot
    idk = np.zeros((nk1,nk2,nk3),dtype=int)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j*nk3 + i*nk2*nk3
                Rx = float(i)/float(nk1)
                Ry = float(j)/float(nk1)
                Rz = float(k)/float(nk1)
                if Rx >= 0.5: Rx=Rx-1.0
                if Ry >= 0.5: Ry=Ry-1.0
                if Rz >= 0.5: Rz=Rz-1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                idk[i,j,k]=n
                Kint[:,n] = Rx*b_vectors[0,:]+Ry*b_vectors[1,:]+Rz*b_vectors[2,:]

    return(Kint,K_wght,nktot,idk)

#=====================================Generate fft grid of R vectors
def get_R_grid_fft():
    nrtot = nk1*nk2*nk3
    R = np.zeros((nrtot,3),dtype=float)
    Rfft = np.zeros((nk1,nk2,nk3,3),dtype=float)
    R_wght = np.ones((nrtot),dtype=float)
    idx = np.zeros((nk1,nk2,nk3),dtype=int)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j*nk3 + i*nk2*nk3
                Rx = float(i)/float(nk1)
                Ry = float(j)/float(nk1)
                Rz = float(k)/float(nk1)
                if Rx >= 0.5: Rx=Rx-1.0
                if Ry >= 0.5: Ry=Ry-1.0
                if Rz >= 0.5: Rz=Rz-1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                R[n,:] = Rx*nk1*a_vectors[0,:]+Ry*nk2*a_vectors[1,:]+Rz*nk3*a_vectors[2,:]
                Rfft[i,j,k,:] = R[n,:]
                idx[i,j,k]=n

    Rfft = FFT.fftshift(Rfft,axes=(0,1,2))

    return(R,Rfft,R_wght,nrtot,idx)

#=====================================Utility for generating standard paths in the IBZ
# This modules are from AFLOWpi
def free2abc(cellparamatrix,cosine=True,degrees=True,string=True,bohr=False):
    '''
    Convert lattice vectors to a,b,c,alpha,beta,gamma of the primitive lattice

    Arguments:
          cellparamatrix (np.matrix): matrix of cell vectors

    Keyword Arguments:
          cosine (bool): If True alpha,beta,gamma are cos(alpha),cos(beta),cos(gamma),
          degrees (bool): If True return alpha,beta,gamma in degrees; radians if False
          string (bool): If True return a,b,c,alpha,beta,gamma as a string; if False return as a list
          bohr (bool): If True return a,b,c in bohr radii; if False return in angstrom

    Returns:
         paramArray (list): a list of the parameters a,b,c,alpha,beta,gamma generated from the input matrix

    '''

    ibrav = getIbravFromVectors(cellparamatrix)
    try:
        cellparamatrix=cellparamatrix.getA()
    except :
        pass
    try:
        a = np.sqrt(cellparamatrix[0].dot(cellparamatrix[0].T))
        b = np.sqrt(cellparamatrix[1].dot(cellparamatrix[1].T))
        c = np.sqrt(cellparamatrix[2].dot(cellparamatrix[2].T))
    except:
        cellparamatrix = np.array(cellparamatrix)
        a = np.sqrt(cellparamatrix[0].dot(cellparamatrix[0]))
        b = np.sqrt(cellparamatrix[1].dot(cellparamatrix[1]))
        c = np.sqrt(cellparamatrix[2].dot(cellparamatrix[2]))

    degree2radian = np.pi/180
    alpha,beta,gamma=(0.0,0.0,0.0)


    alpha = np.arccos(cellparamatrix[1].dot(cellparamatrix[2].T)/(b*c))
    beta  = np.arccos(cellparamatrix[0].dot(cellparamatrix[2].T)/(a*c))
    gamma = np.arccos(cellparamatrix[0].dot(cellparamatrix[1].T)/(a*b))

    if np.abs(alpha)<0.000001:
        alpha=0.0
    if np.abs(beta)<0.000001:
        beta=0.0
    if np.abs(gamma)<0.000001:
        gamma=0.0


    AngstromToBohr = 1.88971616463207
    BohrToAngstrom = 1/AngstromToBohr
    if bohr==False:
        a*=BohrToAngstrom
        b*=BohrToAngstrom
        c*=BohrToAngstrom

        a=float('%10.5e'%np.around(a,decimals=5))
        b=float('%10.5e'%np.around(b,decimals=5))
        c=float('%10.5e'%np.around(c,decimals=5))

    if cosine==True:
        cosBC=np.cos(alpha)
        cosAC=np.cos(beta)
        cosAB=np.cos(gamma)
        paramArray = [a,b,c,cosBC,cosAC,cosAB]

        param_list=[]
        for v in range(len(paramArray)):
            param_list.append(float('%10.5e'%np.around(paramArray[v],decimals=5)))
        paramArray=tuple(param_list)

        returnString = 'a=%s,b=%s,c=%s,cos(alpha)=%s,cos(beta)=%s,cos(gamma)=%s' % tuple(paramArray)

    if degrees==True:
        alpha/=degree2radian
        beta/= degree2radian
        gamma/=degree2radian

    if cosine!=True:
        paramArray = (a,b,c,alpha,beta,gamma)

        param_list=[]
        for v in range(len(paramArray)):
            param_list.append(float('%10.5e'%np.around(paramArray[v],decimals=5)))
        paramArray=tuple(param_list)

        returnString = 'A=%s,B=%s,C=%s,alpha=%s,beta=%s,gamma=%s' % tuple(paramArray)

    if string==True:
        return returnString
    else:

        return paramArray



def _getHighSymPoints(ibrav,alat,cellOld):
    '''
    Searching for the ibrav number in the input file for the calculation
    to determine the path for the band structure calculation

    Arguments:
          oneCalc (dict): a dictionary containing properties about the AFLOWpi calculation

    Keyword Arguments:
          ID (str): ID string for the particular calculation and step

    Returns:
          special_points (list): list of the HSP names
          band_path (str): path in string form

    '''

    def CUB(cellOld):
        special_points = {
          'gG'   : (0.0, 0.0, 0.0),
          'M'   : (0.5, 0.5, 0.0),
          'R'   : (0.5, 0.5, 0.5),
          'X'   : (0.0, 0.5, 0.0)
        }

        default_band_path = 'gG-X-M-gG-R-X|M-R'
        band_path = default_band_path
        return special_points, band_path
    def FCC(cellOld):
        special_points = {
          'gG'   : (0.0, 0.0, 0.0),
          'K'    : (0.375, 0.375, 0.750),
          'L'    : (0.5, 0.5, 0.5),
          'U'    : (0.625, 0.250, 0.625),
          'W'    : (0.5, 0.25, 0.75),
          'X'    : (0.5, 0.0, 0.5)
        }

        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[-1.0, 0.0, 1.0,],
                                    [ 0.0, 1.0, 1.0],
                                    [-1.0, 1.0, 0.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())



        default_band_path = 'gG-X-W-K-gG-L-U-W-L-K|U-X'
        band_path = default_band_path
        return special_points, band_path
    def BCC(cellOld):

        special_points = {
            'gG'  : [0, 0, 0],
            'H'   : [0.5, -0.5, 0.5],
            'P'   : [0.25, 0.25, 0.25],
            'N'   : [0.0, 0.0, 0.5]
            }

        #convert HSP from aflow to qe convention brillouin zones
        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 1.0,],
                                    [-1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-H-N-gG-P-H|P-N'
        band_path = default_band_path
        return special_points, band_path


    def HEX(cellOld):
        special_points = {
          'gG'    : (0, 0, 0),
          'A'    : (0.0, 0.0, 0.5),
          'H'    : (1.0/3.0, 1.0/3.0, 0.5),
          'K'    : (1.0/3.0, 1.0/3.0, 0.0),
          'L'    : (0.5, 0.0, 0.5),
          'M'    : (0.5, 0.0, 0.0)
        }

        default_band_path = 'gG-M-K-gG-A-L-H-A|L-M|K-H'
        band_path = default_band_path
        return special_points, band_path


    def RHL1(cellOld):

        tx = cellOld[0][0]
        c = (((tx**2)*2)-1.0)
        c = -c
        alpha = np.arccos(c)
        eta1 = 1.0 + 4.0*np.cos(alpha)
        eta2 = 2.0 + 4.0*np.cos(alpha)
        eta = eta1/eta2
        nu = 0.75 - eta/2.0
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'B'    : (eta, 0.5, 1.0-eta),
            'B1'    : (0.5, 1.0-eta, eta-1.0),
            'F'    : (0.5, 0.5, 0.0),
            'L'    : (0.5, 0.0, 0.0),
            'L1'    : (0.0, 0.0, -0.5),
            'P'    : (eta, nu, nu),
            'P1'    : (1.0-nu, 1.0-nu, 1.0-eta),
            'P2'    : (nu, nu, eta-1.0),
            'Q'    : (1.0-nu, nu, 0.0),
            'X'    : (nu, 0.0, -nu),
            'Z'    : (0.5, 0.5, 0.5)
          }
        default_band_path = 'gG-L-B1|B-Z-gG-X|Q-F-P1-Z|L-P'
        band_path = default_band_path
        return special_points, band_path

    def RHL2(cellOld):

        tx = cellOld[0][0]
        c = (((tx**2)*2)-1.0)
        c = -c
        alpha = np.arccos(c)
        eta = 1.0/(2*np.tan(alpha/2.0)**2)
        nu = 0.75 - eta/2.0

        special_points = {
           'gG'    : (0.0, 0.0, 0.0),
           'F'    : (0.5, -0.5, 0.0),
           'L'    : (0.5, 0.0, 0.0),
           'P'    : (1.0-nu, -nu, 1.0-nu),
           'P1'    : (nu, nu-1.0, nu-1.0),
           'Q'    : (eta, eta, eta),
           'Q1'    : (1.0-eta, -eta, -eta),
           'Z'    : (0.5, -0.5, 0.5)
         }
        default_band_path = 'gG-P-Z-Q-gG-F-P1-Q1-L-Z'
        band_path = default_band_path
        return special_points, band_path



    def TET(cellOld):

        special_points = {
          'gG'    : (0.0, 0.0, 0.0),
          'A'    : (0.5, 0.5, 0.5),
          'M'    : (0.5, 0.5, 0.0),
          'R'    : (0.0, 0.5, 0.5),
          'X'    : (0.0, 0.5, 0.0),
          'Z'    : (0.0, 0.0, 0.5)
        }
        default_band_path = 'gG-X-M-gG-Z-R-A-Z|X-R|M-A'
        band_path = default_band_path
        return special_points, band_path

    def BCT1(cellOld):
        a = cellOld[1][0]*2
        c = cellOld[1][2]*2

        if a==c:
            return BCC(cellOld)

        eta = (1 + c**2/a**2)/4

        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'M'    : (-0.5, 0.5, 0.5),
            'N'    : (0.0, 0.5, 0.0),
            'P'    : (0.25, 0.25, 0.25),
            'X'    : (0.0, 0.0, 0.5),
            'Z'    : (eta, eta, -eta),
            'Z1'    : (-eta, 1.0-eta, eta)
          }

        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-X-M-gG-Z-P-N-Z1-M|X-P'
        band_path = default_band_path
        return special_points, band_path


    def BCT2(cellOld):

        a = cellOld[1][0]*2
        c = cellOld[1][2]*2
        if a==c:
            return BCC(cellOld)

        eta = (1 + a**2/c**2)/4.0
        zeta = a**2/(2.0*c**2)
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'N'    : (0.0, 0.5, 0.0),
            'P'    : (0.25, 0.25, 0.25),
            'gS'    : (-eta, eta, eta),
            'gS1'    : (eta, 1-eta, -eta),
            'X'    : (0.0, 0.0, 0.5),
            'Y'    : (-zeta, zeta, 0.5),
            'Y1'   : (0.5, 0.5, -zeta),
            'Z'    : (0.5, 0.5, -0.5)
          }

        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-X-Y-gS-gG-Z-gS1-N-P-Y1-Z|X-P'
        band_path = default_band_path
        return special_points, band_path



    def ORC(cellOld):
        special_points = {
          'gG'    : (0.0, 0.0, 0.0),
          'R'    : (0.5, 0.5, 0.5),
          'S'    : (0.5, 0.5, 0.0),
          'T'    : (0.0, 0.5, 0.5),
          'U'    : (0.5, 0.0, 0.5),
          'X'    : (0.5, 0.0, 0.0),
          'Y'    : (0.0, 0.5, 0.0),
          'Z'    : (0.0, 0.0, 0.5)
        }

        default_band_path = 'gG-X-S-Y-gG-Z-U-R-T-Z|Y-T|U-X|S-R'
        band_path = default_band_path
        return special_points, band_path

    def ORCF1(cellOld):

        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        eta = (1 + a**2/b**2 + a**2/c**2)/4
        zeta = (1 + a**2/b**2 - a**2/c**2)/4
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'A'    : (0.5, 0.5 + zeta, zeta),
            'A1'    : (0.5, 0.5-zeta, 1.0-zeta),
            'L'    : (0.5, 0.5, 0.5),
            'T'    : (1.0, 0.5, 0.5),
            'X'    : (0.0, eta, eta),
            'X1'    : (1.0, 1.0-eta, 1.0-eta),
            'Y'    : (0.5, 0.0, 0.5),
            'Z'    : (0.5, 0.5, 0.0)
          }



        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0],
                                    [ 0.0, 1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-Y-T-Z-gG-X-A1-Y|T-X1|X-A-Z|L-gG'
        band_path = default_band_path
        return special_points, band_path

    def ORCF2(cellOld):

        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        eta = (1 + a**2/b**2 - a**2/c**2)/4
        phi = (1 + c**2/b**2 - c**2/a**2)/4
        delta = (1 + b**2/a**2 - b**2/c**2)/4
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'C'    : (0.5, 0.5-eta, 1.0-eta),
            'C1'    : (0.5, 0.5+eta, eta),
            'D'    : (0.5-delta, 0.5, 1.0-delta),
            'D1'    : (0.5+delta, 0.5, delta),
            'L'    : (0.5, 0.5, 0.5),
            'H'    : (1.0-phi, 0.5-phi, 0.5),
            'H1'    : (phi, 0.5+phi, 0.5),
            'X'    : (0.0, 0.5, 0.5),
            'Y'    : (0.5, 0.0, 0.5),
            'Z'    : (0.5, 0.5, 0.0),
            }

        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0],
                                    [ 0.0, 1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())
#           print( k,special_points[k])
        default_band_path = 'gG-Y-C-D-X-gG-Z-D1-H-C|C1-Z|X-H1|H-Y|L-gG'
        band_path = default_band_path
        return special_points, band_path


    def ORCF3(cellOld):

        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        eta = (1 + a**2/b**2 + a**2/c**2)/4
        zeta = (1 + a**2/b**2 - a**2/c**2)/4
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'A'    : (0.5, 0.5 + zeta, zeta),
            'A1'    : (0.5, 0.5-zeta, 1.0-zeta),
            'L'    : (0.5, 0.5, 0.5),
            'T'    : (1.0, 0.5, 0.5),
            'X'    : (0.0, eta, eta),
            'X1'    : (1.0, 1.0-eta, 1.0-eta),
            'Y'    : (0.5, 0.0, 0.5),
            'Z'    : (0.5, 0.5, 0.0)
            }

        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0],
                                    [ 0.0, 1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-Y-T-Z-gG-X-A1-Y|X-A-Z|L-R'
        band_path = default_band_path
        return special_points, band_path


    def ORCC(cellOld):

        try:
            cellOld=cellOld.getA()
        except:
            pass
        a = cellOld[0][0]*2
        b = cellOld[1][1]*2
        c = cellOld[2][2]
        myList = [a, b,c]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        zeta = (1 + a**2/b**2)/4.0
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'A'    : (zeta, zeta, 0.5),
            'A1'    : (-zeta, 1.0-zeta, 0.5),
            'R'    : (0.0, 0.5, 0.5),
            'S'    : (0.0, 0.5, 0.0),
            'T'    : (-0.5, 0.5, 0.5),
            'X'    : (zeta, zeta, 0.0),
            'X1'    : (-zeta, 1.0-zeta, 0.0),
            'Y'    : (-0.5, 0.5, 0.0),
            'Z'    : (0.0, 0.0, 0.5)
          }

        aflow_conv = np.asarray([[ 1.0,-1.0, 0.0,],
                                    [ 1.0, 1.0, 0.0,],
                                    [ 0.0, 0.0, 2.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 0.0,],
                                    [-1.0, 1.0, 0.0],
                                    [ 0.0, 0.0, 2.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-X-S-R-A-Z-gG-Y-X1-A1-T-Y|Z-T'
        band_path = default_band_path
        return special_points, band_path

    def ORCI(cellOld):

        try:
            cellOld=cellOld.getA()
        except:
            pass
        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)

        myList.remove(a)
        myList.remove(c)
        b = myList[0]
        print(abs(a-b),abs(a-c),abs(c-b))
        if abs(1.0-a/b)<0.01:
            if abs(1.0-a/c)<0.01:

                return BCC(cellOld)
            if abs(1.0-c/b)<0.01:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)
            else:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)
        if abs(1.0-a/c)<0.01:
            if abs(1.0-a/b)<0.01:
                return BCC(cellOld)
            if abs(1.0-c/b)<0.01:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)


            else:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)


        chi = (1.0 + (a/c)**2)/4.0
        eta = (1.0 + (b/c)**2)/4.0
        delta = (b*b - a*a)/(4*c*c)
        mu = (b*b + a*a)/(4*c*c)
        special_points = {
          'gG'    : (0, 0, 0),
          'L'    : (-mu, mu, 0.5-delta),
          'L1'   : (mu, -mu, 0.5+delta),
          'L2'   : (0.5-delta, 0.5+delta, -mu),
          'R'    : (0.0, 0.5, 0.0),
          'S'    : (0.5, 0.0, 0.0),
          'T'    : (0.0, 0.0, 0.5),
          'W'    : (0.25,0.25,0.25),
          'X'    : (-chi, chi, chi),
          'X1'   : (chi, 1.0-chi, -chi),
          'Y'    : (eta, -eta, eta),
          'Y1'   : (1.0-eta, eta, -eta),
          'Z'    : (0.5, 0.5, -0.5)
        }

        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 1.0,],
                                    [-1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-X-L-T-W-R-X1-Z-gG-Y-S-W|L1-Y|Y1-Z'
        band_path = default_band_path
        return special_points, band_path


    def MCLC(cellOld):
        try:
            cellOld=cellOld.getA()
        except:
            pass

        a,b,c,cos_alpha,cos_beta_cos_gamma = free2abc(cellparamatrix,cosine=True,degrees=False,string=False,bohr=False)

        sin_gamma = np.sin(np.arccos(cos_gamma))

        mu        = (1+(b/a)**2.0)/4.0
        delta     = b*c*cos_gamma/(2.0*a**2.0)
        xi        = mu -0.25*+(1.0 - b*cos_gamma/c)*(4.0*(sin_gamma**2.0))
        eta       = 0.5 + 2.0*xi*c*cos_gamma/b
        phi       = 1.0 + xi - 2.0*mu
        psi       = eta - 2.0*delta

    def MCLC12(cellOld):
        pass

    def MCLC5(cellOld):
        pass


    def TRI1A(cellOld):
        special_points = {
           'gG'   : (0.0,0.0,0.0),
           'L'    : (0.5,0.5,0.0),
           'M'    : (0.0,0.5,0.5),
           'N'    : (0.5,0.0,0.5),
           'R'    : (0.5,0.5,0.5),
           'X'    : (0.5,0.0,0.0),
           'Y'    : (0.0,0.5,0.0),
           'Z'    : (0.0,0.0,0.5),
           }

        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'
        return special_points, band_path
    def TRI2A(cellOld):
        special_points = {
           'gG'   : (0.0,0.0,0.0),
           'L'    : (0.5,0.5,0.0),
           'M'    : (0.0,0.5,0.5),
           'N'    : (0.5,0.0,0.5),
           'R'    : (0.5,0.5,0.5),
           'X'    : (0.5,0.0,0.0),
           'Y'    : (0.0,0.5,0.0),
           'Z'    : (0.0,0.0,0.5),
           }

        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'

        return special_points, band_path

    def TRI1B(cellOld):
        special_points = {
           'gG'   : ( 0.0, 0.0,0.0),
           'L'    : ( 0.5,-0.5,0.0),
           'M'    : ( 0.0, 0.0,0.5),
           'N'    : (-0.5,-0.5,0.5),
           'R'    : ( 0.0,-0.5,0.5),
           'X'    : ( 0.0,-0.5,0.0),
           'Y'    : ( 0.5, 0.0,0.0),
           'Z'    : (-0.5, 0.0,0.5),
           }


        band_path = str("X-gG-Y|L-gG-Z|N-gG-M|R-gG")
        return special_points, band_path

    def TRI2B(cellOld):
        special_points = {
           'gG'   : ( 0.0, 0.0,0.0),
           'L'    : ( 0.5,-0.5,0.0),
           'M'    : ( 0.0, 0.0,0.5),
           'N'    : (-0.5,-0.5,0.5),
           'R'    : ( 0.0,-0.5,0.5),
           'X'    : ( 0.0,-0.5,0.0),
           'Y'    : ( 0.5, 0.0,0.0),
           'Z'    : (-0.5, 0.0,0.5),
           }

        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'
        return special_points, band_path

    def choose_lattice(ibrav, cellOld):
        if(int(ibrav)==1):
            var1, var2 = CUB(cellOld)
            return var1, var2

        elif(int(ibrav)==2):
            var1, var2 = FCC(cellOld)
            return var1, var2

        elif(int(ibrav)==3):
            var1, var2 = BCC(cellOld)
            return var1, var2

        elif(int(ibrav)==4):
            var1, var2 = HEX(cellOld)
            return var1, var2

        elif(int(ibrav)==5):
            try:
                cellOld = cellOld.getA()
            except:
                pass


            tx = cellOld[0][0]

            c = (((tx**2)*2)-1.0)
            c = -c



            alpha=np.arccos(c)


            if alpha < np.pi/2.0:
                var1, var2 = RHL1(cellOld)
                return var1, var2

            elif alpha > np.pi/2.0:
                var1, var2 = RHL2(cellOld)
                return var1, var2


        elif(int(ibrav)==6):
            var1, var2 = TET(cellOld)
            return var1, var2

        elif(int(ibrav)==7):
            try:
                cellOld = cellOld.getA()
            except:
                pass
            a = cellOld[1][0]*2
            c = cellOld[1][2]*2

            if(c < a):
                var1, var2 = BCT1(cellOld)
                return var1, var2


            elif(c > a):
                var1, var2 = BCT2(cellOld)
                return var1, var2

            else:
                var1, var2 = BCC(cellOld)
                return var1, var2

        elif(int(ibrav)==8):
            var1, var2 = ORC(cellOld)
            return var1, var2

        elif(int(ibrav)==9):
            var1, var2 = ORCC(cellOld)
            return var1, var2

        elif(int(ibrav)==10):
            try:
                cellOld = cellOld.getA()
            except:
                pass
            a1 = cellOld[0][0]*2
            b1 = cellOld[1][1]*2
            c1 = cellOld[2][2]*2
            myList = [a1, b1, c1]
            c = max(myList)
            a = min(myList)
            myList.remove(a)
            myList.remove(c)
            b = myList[0]

            if(1.0/a**2 > 1.0/b**2 + 1.0/c**2):
                var1, var2 = ORCF1(cellOld)
                return var1, var2

            elif(1.0/a**2 < 1.0/b**2 + 1.0/c**2):
                var1, var2 = ORCF2(cellOld)
                return var1, var2

            elif(1.0/a**2 == 1.0/b**2 + 1.0/c**2):
                '''
                var1, var2 = ORCF3(cellOld)
                return var1, var2
                '''
                raise Exception
        elif(int(ibrav)==11):

            var1, var2 = ORCI(cellOld)
            return var1, var2
        elif(int(ibrav)==14):

            a,b,c,alpha,beta,gamma =  free2abc(cellOld,cosine=False,bohr=False,string=False)

            minAngle = min([float(alpha),float(beta),float(gamma)])
            maxAngle = max([float(alpha),float(beta),float(gamma)])
            if alpha==90.0 or beta==90.0 or gamma==90.0:
                if alpha>=90.0 or beta>=90.0 or gamma>=90.0:
                    var1,var2 = TRI2A(cellOld)
                    return var1,var2
                if alpha<=90.0 or beta<=90.0 or gamma<=90.0:
                    var1,var2 = TRI2B(cellOld)
                    return var1,var2
            elif minAngle>90.0:
                var1,var2 = TRI1A(cellOld)
                return var1,var2
            elif maxAngle<90:
                var1,var2 = TRI1B(cellOld)
                return var1,var2

        '''
        elif(int(ibrav)==12):
            var1, var2 = MCL(cellOld)
            return var1, var2
        '''
    if ibrav==0:
        sys.exit('IBRAV = 0 not permitted')
    if ibrav < 0:
        print('Lattice type %s is not implemented' % ibrav)
        logging.error('The ibrav value from expresso has not yet been implemented to the framework')
        raise Exception

    if ibrav==5:
        cellOld/=alat

    special_points, band_path = choose_lattice(ibrav, cellOld)

    return special_points, band_path



#=====================================Generate the k path
def kpnts_interpolation_mesh(ibrav,alat,cell,dk):
    '''
    Get path between HSP

    Arguments:
          dk (float): distance between points

    Returns:
          kpoints : array of arrays kx,ky,kz
          numK    : Total no. of k-points

    '''

    def kdistance(hs, p1, p2):
        g = np.dot(hs.T, hs)
        p1, p2 = np.array(p1), np.array(p2)
        d = p1 - p2
        dist2 = np.dot(d.T, np.dot(g, d).T)
        return np.sqrt(dist2)

    def getSegments(path):
        segments = path.split('|')
        return segments

    def getPoints(pathSegment):
        pointsList = pathSegment.split('-')
        return pointsList
    def getNumPoints(path):
        list1 = getSegments(path)
        numPts = 0
        for index in (list1):
            numPts += len(getPoints(index))
        return numPts

    if ibrav==0:
        sys.exit('IBRAV = 0 not permitted')
    if ibrav<0:
        print('Lattice type %s is not implemented') % ibrav
        logging.error('The ibrav value from QE has not yet been implemented')
        raise Exception

    totalK=0
    special_points, band_path = _getHighSymPoints(ibrav,alat,cell)
    hs = 2*np.pi*np.linalg.inv(cell)  # reciprocal lattice
    segs = getSegments(band_path)

    kx = np.array([])
    ky = np.array([])
    kz = np.array([])

    for index in segs:

        a = getPoints(index) #gets the points in each segment of path separated by |
        point1 = None
        point2 = None

        for index2 in range(len(a)-1):
            try:
                point1 = a[index2]
                point2 = a[index2+1]
                p1 = special_points[point1]
                p2 = special_points[point2]

                newDK = (2*np.pi/alat)*dk
                numK = int(np.ceil((kdistance(hs, p1, p2)/newDK)))
                totalK+=numK

                numK = str(numK)

                a0 = np.linspace(p1[0],p2[0],numK).astype(np.float16)
                a1 = np.linspace(p1[1],p2[1],numK).astype(np.float16)
                a2 = np.linspace(p1[2],p2[2],numK).astype(np.float16)

                kx = np.concatenate((kx,a0))
                ky = np.concatenate((ky,a1))
                kz = np.concatenate((kz,a2))

            except Exception as e:
                print(e)

        kpoints = np.array([kx,ky,kz])

    return (kpoints)

#=====================================Load balancing for MPI
def load_balancing(size,rank,n):

    # Load balancing
    ini = np.zeros((size),dtype=int)
    end = np.zeros((size),dtype=int)
    splitsize = 1.0/size*n
    for i in range(size):
        ini[i] = int(round(i*splitsize))
        end[i] = int(round((i+1)*splitsize))
    start = ini[rank]
    stop = end[rank]

    return(start,stop)

#=====================================Generate plot with comparison of DFT eigenvaluaes vs. projections
def plot_compare_TB_DFT_eigs():

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))

    ispin = 0 #plots only 1 spin channel
    #for ispin in range(nspin):
    for ik in range(nkpnts):
        if non_ortho:
            eigval,_ = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
        else:
            eigval,_ = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    fig=plt.figure
    nbnds_dft,_,_=my_eigsmat.shape
    for i in range(nbnds_dft):
        #print("{0:d}".format(i))
        yy = my_eigsmat[i,:,ispin]
        if i==0:
            plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime',label='DFT')
        else:
            plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime')

    for i in range(nbnds_tb):
        yy = E_k[i,:,ispin]
        if i==0:
            plt.plot(yy,'ok',markersize=2,markeredgecolor='None',label='TB')
        else:
            plt.plot(yy,'ok',markersize=2,markeredgecolor='None')

    plt.xlabel('k-points')
    plt.ylabel('Energy - E$_F$ (eV)')
    plt.legend()
    plt.title('Comparison of TB vs. DFT eigenvalues')
    plt.savefig('comparison.pdf',format='pdf')
    return()

#=====================================Read input file
def read_input():

    non_ortho  = False
    shift_type = 2
    shift      = 20
    pthr       = 0.9
    do_comparison = False
    double_grid = False
    do_bands = False
    onedim = False
    do_dos = False
    emin = -10.
    emax = 2
    nfft1 = 0
    nfft2 = 0
    nfft3 = 0
    ibrav = 0
    dkres = 0.1
    Boltzmann = False
    epsilon = False
    do_spin_orbit = False
    theta = 0.0 
    phi = 0.0
    lambda_p = 0.0
    lambda_d = 0.0
    Berry = False
    npool = 1

    f = open(input_file)
    lines=f.readlines()
    f.close
    for line in lines:
        line = line.strip()
        if re.search('fpath',line):
            p = line.split()
            fpath = p[1]
        if re.search('non_ortho',line):
            p = line.split()
            non_ortho = p[1]
            if non_ortho == 'False':
                non_ortho = (1 == 2)
            else:
                non_ortho = (1 == 1)
        if re.search('do_comparison',line):
            p = line.split()
            do_comparison = p[1]
            if do_comparison == 'False':
                do_comparison = (1 == 2)
            else:
                do_comparison = (1 == 1)
        if re.search('double_grid',line):
            p = line.split()
            double_grid = p[1]
            if double_grid == 'False':
                double_grid = (1 == 2)
            else:
                double_grid = (1 == 1)
        if re.search('do_bands',line):
            p = line.split()
            do_bands = p[1]
            if do_bands == 'False':
                do_bands = (1 == 2)
            else:
                do_bands = (1 == 1)
        if re.search('onedim',line):
            p = line.split()
            onedim = p[1]
            if onedim == 'False':
                onedim = (1 == 2)
            else:
                onedim = (1 == 1)
        if re.search('do_dos',line):
            p = line.split()
            do_dos = p[1]
            if do_dos == 'False':
                do_dos = (1 == 2)
            else:
                do_dos = (1 == 1)
            emin = p[2]
            emax = p[3]
        if re.search('delta',line):
            p = line.split()
            delta = float(p[1])
        if re.search('do_spin_orbit',line):
            p = line.split()
            do_spin_orbit = p[1]
            if do_spin_orbit == 'False':
                do_spin_orbit = (1 == 2)
            else:
                do_spin_orbit = (1 == 1)
        if re.search('theta',line):
            p = line.split()
            theta = float(p[1])
            theta=theta*DEGTORAD
        if re.search('phi',line):
            p = line.split()
            phi = float(p[1])
            phi=phi*DEGTORAD
        if re.search('lambda_p',line):
            p = line.split('#')[0].split()
            lambda_p = np.array(p[1:],dtype='float') 
        if re.search('lambda_d',line):
            p = line.split('#')[0].split()
            lambda_d = np.array(p[1:],dtype='float') 
        if re.search('shift_type',line):
            p = line.split()
            shift_type = int(p[1])
        if re.search('shift',line):
            p = line.split()
            shift = float(p[1])
        if re.search('pthr',line):
            p = line.split()
            pthr = float(p[1])
        if re.search('nfft123',line):
            p = line.split()
            nfft1 = int(p[1])
            nfft2 = int(p[2])
            nfft3 = int(p[3])
        if re.search('ibrav',line):
            p = line.split()
            ibrav = int(p[1])
        if re.search('dkres',line):
            p = line.split()
            dkres = float(p[1])
        if re.search('Boltzmann',line):
            p = line.split()
            Boltzmann = p[1]
            if Boltzmann == 'False':
                Boltzmann = (1 == 2)
            else:
                Boltzmann = (1 == 1)
        if re.search('epsilon',line):
            p = line.split()
            epsilon = p[1]
            if epsilon == 'False':
                epsilon = (1 == 2)
            else:
                epsilon = (1 == 1)
        if re.search('Berry',line):
            p = line.split()
            Berry = p[1]
            if Berry == 'False':
                Berry = (1 == 2)
            else:
                Berry = (1 == 1)
        if re.search('npool',line):
            p = line.split()
            npool = int(p[1])
    if fpath == '':
        sys.exit('missing path to _.save')

    return(non_ortho, shift_type, fpath, shift, pthr, do_comparison, double_grid, \
            do_bands, onedim, do_dos, emin, emax, delta, do_spin_orbit, nfft1, nfft2, nfft3, \
            ibrav, dkres, Boltzmann, epsilon,theta,phi,lambda_p,lambda_d, Berry,npool)

#=====================================Read output of QE and projwfc.x
def read_QE_output_xml():
    atomic_proj = fpath+'/atomic_proj.xml'
    data_file   = fpath+'/data-file.xml'

# Reading data-file.xml
    tree  = ET.parse(data_file)
    root  = tree.getroot()

    alatunits  = root.findall("./CELL/LATTICE_PARAMETER")[0].attrib['UNITS']
    alat   = float(root.findall("./CELL/LATTICE_PARAMETER")[0].text.split()[0])

    if rank == 0 and verbose == True: print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

    aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a1")[0].text.split()
    a1=np.array(aux,dtype="float32")

    aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a2")[0].text.split()
    a2=np.array(aux,dtype="float32")

    aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a3")[0].text.split()
    a3=np.array(aux,dtype="float32")

    a_vectors = np.array([a1,a2,a3])/alat #in units of alat
    aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b1")[0].text.split()
    b1=np.array(aux,dtype='float32')

    aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b2")[0].text.split()
    b2=np.array(aux,dtype='float32')

    aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b3")[0].text.split()
    b3=np.array(aux,dtype='float32')

    b_vectors = np.array([b1,b2,b3]) #in units of 2pi/alat

    # numbor of atoms
    natoms=int(float(root.findall("./IONS/NUMBER_OF_ATOMS")       [0].text.split()[0]))

    # Monkhorst&Pack grid
    nk1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk1'])
    nk2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk2'])
    nk3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk3'])
    k1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k1'])
    k2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k2'])
    k3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k3'])
    if rank == 0 and  verbose == True: print('Monkhorst&Pack grid',nk1,nk2,nk3,k1,k2,k3)

    if rank == 0 and  verbose == True: print('reading data-file.xml in ',time.clock(),' sec')
    reset=time.clock()

    # Reading atomic_proj.xml
    tree  = ET.parse(atomic_proj)
    root  = tree.getroot()

    nkpnts = int(root.findall("./HEADER/NUMBER_OF_K-POINTS")[0].text.strip())
    #if rank == 0: print('Number of kpoints: {0:d}'.format(nkpnts))

    nspin  = int(root.findall("./HEADER/NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
    #if rank == 0: print('Number of spin components: {0:d}'.format(nspin))

    kunits = root.findall("./HEADER/UNITS_FOR_K-POINTS")[0].attrib['UNITS']
    #if rank == 0: print('Units for the kpoints: {0:s}'.format(kunits))

    aux = root.findall("./K-POINTS")[0].text.split()
    kpnts  = np.array(aux,dtype="float32").reshape((nkpnts,3))
    #if rank == 0: print('Read the kpoints')

    aux = root.findall("./WEIGHT_OF_K-POINTS")[0].text.split()
    kpnts_wght  = np.array(aux,dtype='float32')

    if kpnts_wght.shape[0] != nkpnts:
        sys.exit('Error in size of the kpnts_wght vector')


    nbnds  = int(root.findall("./HEADER/NUMBER_OF_BANDS")[0].text.split()[0])
    if rank == 0 and  verbose == True: print('Number of bands: {0:d}'.format(nbnds))

    aux    = root.findall("./HEADER/UNITS_FOR_ENERGY")[0].attrib['UNITS']
    #if rank == 0: print('The units for energy are {0:s}'.format(aux))

    Efermi = float(root.findall("./HEADER/FERMI_ENERGY")[0].text.split()[0])*RYTOEV
    if rank == 0 and  verbose == True: print('Fermi energy: {0:f} eV '.format(Efermi))

    nawf   =int(root.findall("./HEADER/NUMBER_OF_ATOMIC_WFC")[0].text.split()[0])
    if rank == 0 and  verbose == True: print('Number of atomic wavefunctions: {0:d}'.format(nawf))

    #Read eigenvalues and projections

    U = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)  # final data array
    my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
    Uaux = np.zeros((nbnds,nawf,nkpnts,nspin,1),dtype=complex) # read data from task
    my_eigsmataux = np.zeros((nbnds,nkpnts,nspin,1))
    Uaux1 = np.zeros((nbnds,nawf,nkpnts,nspin,1),dtype=complex) # receiving data array
    my_eigsmataux1 = np.zeros((nbnds,nkpnts,nspin,1))

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nkpnts)

    Uaux[:,:,:,:,0] = read_proj(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi)

    if rank == 0:
        U[:,:,:,:]=Uaux[:,:,:,:,0]
        for i in range(1,size):
            comm.Recv(Uaux1,ANY_SOURCE)
            U[:,:,:,:] += Uaux1[:,:,:,:,0]
    else:
        comm.Send(Uaux,0)
    U = comm.bcast(U)

    my_eigsmataux[:,:,:,0] = read_eig(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi)

    if rank == 0:
        my_eigsmat[:,:,:]=my_eigsmataux[:,:,:,0]
        for i in range(1,size):
            comm.Recv(my_eigsmataux1,ANY_SOURCE)
            my_eigsmat[:,:,:] += my_eigsmataux1[:,:,:,0]
    else:
        comm.Send(my_eigsmataux,0)
    my_eigsmat = comm.bcast(my_eigsmat)

    if rank == 0 and  verbose == True: print('reading eigenvalues and projections in ',time.clock()-reset,' sec')
    reset=time.clock()

    if non_ortho:
        Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in range(nkpnts):
            #There will be nawf projections. Each projector of size nbnds x 1
            ovlp_type = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].attrib['type']
            aux = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].text
            #aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')
            aux = np.array([float(i) for i in re.split(',|\n',aux.strip())])

            if ovlp_type !='complex':
                sys.exit('the overlaps are assumed to be complex numbers')
            if len(aux) != nawf**2*2:
                sys.exit('wrong number of elements when reading the S matrix')

            aux = aux.reshape((nawf**2,2))
            ovlp_vector = aux[:,0]+1j*aux[:,1]
            Sks[:,:,ik] = ovlp_vector.reshape((nawf,nawf))

        return(U,Sks, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, \
             nk1, nk2, nk3, natoms)

    else:
        return(U, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, \
             nk1, nk2, nk3, natoms)

#=====================================Read eigenvalues
def read_eig(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi):

    my_eigsmat_p = np.zeros((nbnds,nkpnts,nspin))

    for ik in range(ini_ik,end_ik):
        for ispin in range(nspin):
        #Reading eigenvalues
            if nspin==1:
                eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].attrib['type']
            else:
                eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].attrib['type']
            if eigk_type != 'real':
                sys.exit('Reading eigenvalues that are not real numbers')
            if nspin==1:
                eigk_file=np.array(root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].text.split(),dtype='float32')
            else:
                eigk_file=np.array(root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].text.split(),dtype='float32')
            my_eigsmat_p[:,ik,ispin] = np.real(eigk_file)*RYTOEV-Efermi #meigs in eVs and wrt Ef

    return(my_eigsmat_p)


#=====================================Read projections
def read_proj(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi):

    U_p = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)

    for ik in range(ini_ik,end_ik):
        for ispin in range(nspin):
            #Reading projections
            for iin in range(nawf): #There will be nawf projections. Each projector of size nbnds x 1
                if nspin==1:
                    wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].attrib['type']
                    aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].text
                else:
                    wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,ispin+1,iin+1))[0].attrib['type']
                    aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,ispin+1,iin+1))[0].text

                aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')

                if wfc_type=='real':
                    wfc = aux.reshape((nbnds,1))#wfc = nbnds x 1
                    U_p[:,iin,ik,ispin] = wfc[:,0]
                elif wfc_type=='complex':
                    wfc = aux.reshape((nbnds,2))
                    U_p[:,iin,ik,ispin] = wfc[:,0]+1j*wfc[:,1]
                else:
                    sys.exit('neither real nor complex??')
    return(U_p)

#=====================================Diagonalizer
def write_TB_eigs(Hks,Sks,non_ortho,ispin):

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))
    E_kaux = np.zeros((nbnds_tb,nkpnts,nspin))

    for ik in range(nkpnts):
        if non_ortho:
            eigval,_ = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
        else:
            eigval,_ = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    ipad = False
    if ipad:
        f=open('bands_'+str(ispin)+'.dat','w')
        for ik in range(nkpnts):
            for nb in range(nawf):
                f.write('%3d  %.5f \n' %(ik,E_k[nb,ik,ispin]))
        f.close()
    else:
        f=open('bands_'+str(ispin)+'.dat','w')
        for ik in range(nkpnts):
            s="%d\t"%ik
            for  j in E_k[:,ik,ispin]:s += "%3.5f\t"%j
            s+="\n"
            f.write(s)
        f.close()

    return()

#=====================================Zero padding for H interpolation
def zero_pad(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    # zero padding for FFT interpolation in 3D
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=complex)
    auxp1[:,:,:(nk3/2)]=aux[:,:,:(nk3/2)]
    auxp1[:,:,(nfft3+nk3/2):]=aux[:,:,(nk3/2):]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=complex)
    auxp2[:,:(nk2/2),:]=auxp1[:,:(nk2/2),:]
    auxp2[:,(nfft2+nk2/2):,:]=auxp1[:,(nk2/2):,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=complex)
    auxp3[:(nk1/2),:,:]=auxp2[:(nk1/2),:,:]
    auxp3[(nfft1+nk1/2):,:,:]=auxp2[(nk1/2):,:,:]

    return(auxp3)

#########################################################################################
#                                                                                       #
#                                    AFLOWpi_TB main code                               #
#                                                                                       #
#########################################################################################

if __name__=='__main__':

    #----------------------
    # initialize time
    #----------------------
    if rank == 0: start = time.time()

    #----------------------
    # Print header
    #----------------------
    if rank == 0:
        print('          ')
        print('#############################################################################################')
        print('#                                                                                           #')
        print('#                                       ',AFLOWPITB,'                                        #')
        print('#                                                                                           #')
        print('#                 Utility to construct and operate on TB Hamiltonians from                  #')
        print('#               the projections of DFT wfc on the pseudoatomic orbital basis                #')
        print('#                                                                                           #')
        print('#                        ',str('%1s' %CC),'2016 ERMES group (http://ermes.unt.edu)                         #')
        print('#############################################################################################')
        print('          ')

    #----------------------
    # Initialize n. of threads for multiprocessing (FFTW)
    #----------------------
    #nthread = multiprocessing.cpu_count()
    nthread = size

    ir rank == 0:
    #----------------------
    # Read input and DFT data
    #----------------------
    input_file = str(sys.argv[1])

    non_ortho, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
            do_bands, onedim, do_dos,emin,emax, delta, do_spin_orbit,nfft1, nfft2, \
            nfft3, ibrav, dkres, Boltzmann, epsilon, theta, phi,        \
            lambda_p, lambda_d, Berry, npool = read_input()

    if size >  1:
        if rank == 0 and npool == 1: print('parallel execution on ',size,' processors, ',nthread,' threads and ',npool,' pool')
        if rank == 0 and npool > 1: print('parallel execution on ',size,' processors, ',nthread,' threads and ',npool,' pools')
    else:
        if rank == 0: print('serial execution')
    if rank == 0: print('   ')

    verbose = False

    if rank == 0:
    if (not non_ortho):
        U, my_eigsmat, alat, a_vectors, b_vectors, \
        nkpnts, nspin, kpnts, kpnts_wght, \
        nbnds, Efermi, nawf, nk1, nk2, nk3,natoms  =  read_QE_output_xml()
        Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        sumk = np.sum(kpnts_wght)
        kpnts_wght /= sumk
        for ik in range(nkpnts):
            Sks[:,:,ik]=np.identity(nawf)
        if rank == 0 and verbose == True: print('...using orthogonal algorithm')
    else:
        U, Sks, my_eigsmat, alat, a_vectors, b_vectors, \
        nkpnts, nspin, kpnts, kpnts_wght, \
        nbnds, Efermi, nawf, nk1, nk2, nk3,natoms  =  read_QE_output_xml()
        if rank == 0 and verbose == True: print('...using non-orthogonal algorithm')

    if rank == 0: print('reading in                       %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
    reset=time.time()

    #----------------------
    # Building the Projectability
    #----------------------
    Pn = build_Pn()

    if rank == 0 and verbose == True: print('Projectability vector ',Pn)

    # Check projectability and decide bnd

    bnd = 0
    for n in range(nbnds):
        if Pn[n] > pthr:
            bnd += 1
    if rank == 0 and verbose == True: print('# of bands with good projectability (>',pthr,') = ',bnd)

    #----------------------
    # Building the TB Hamiltonian
    #----------------------
    nbnds_norm = nawf
    Hks = build_Hks()
    U = None
    my_eigsmat = None

    if rank == 0: print('building Hks in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

    # NOTE: Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to 
    # apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
    # non_ortho flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
    # non_ortho flag == 1 - makes H orthogonal (rotated basis) 
    #    Hks = do_non_ortho(Hks,Sks)
    #    Hks = do_ortho(Hks,Sks)

    if non_ortho:
        Hks = do_non_ortho()

    #----------------------
    # Plot the TB and DFT eigevalues. Writes to comparison.pdf
    #----------------------
    if do_comparison:
        plot_compare_TB_DFT_eigs()
        quit()

    #----------------------
    # Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)
    #----------------------

    # Original k grid to R grid
    reset=time.time()
    Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
    Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)

    Hkaux = np.reshape(Hks,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
    if non_ortho:
        Skaux = np.reshape(Hks,(nawf,nawf,nk1,nk2,nk3),order='C')

    HRaux = np.zeros_like(Hks)
    SRaux = np.zeros_like(Sks)

    HRaux = FFT.ifftn(Hkaux,axes=[2,3,4])
    if non_ortho:
        SRaux = FFT.ifftn(Skaux,axes=[2,3,4])

    # NOTE: Naming convention (from here):
    # Hks = k-space Hamiltonian on original MP grid
    # HRs = R-space Hamiltonian on original MP grid
    Hks = None
    Sks = None
    Hks = Hkaux
    Sks = Skaux
    HRs = HRaux
    SRs = SRaux
    Hkaux = None
    Skaux = None
    HRaux = None
    SRaux = None

    if rank == 0: print('k -> R in                        %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

    if do_spin_orbit:
        #----------------------
        # Compute bands with spin-orbit coupling
        #----------------------

        socStrengh = np.zeros((natoms,2),dtype=float)
        socStrengh [:,0] =  lambda_p[:]
        socStrengh [:,1] =  lambda_d[:]

        HRs = do_spin_orbit_calc()
        nawf=2*nawf

    if do_bands and not(onedim):
        #----------------------
        # Compute bands on a selected path in the BZ
        #----------------------

        alat /= ANGSTROM_AU

        if non_ortho:
            # now we orthogonalize the Hamiltonian again
            Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
            Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
            Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
            Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
            Hkaux = do_ortho(Hkaux,Skaux)

            HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
            non_ortho = False

        # Define real space lattice vectors
        R,_,R_wght,nrtot,idx = get_R_grid_fft()

        do_bands_calc(HRs,SRs,R_wght,R,idx,non_ortho,ibrav,alat,a_vectors,b_vectors,dkres)

        alat *= ANGSTROM_AU

        if rank == 0: print('bands in                         %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    elif do_bands and onedim:
        #----------------------
        # FFT interpolation along a single directions in the BZ
        #----------------------
        if rank == 0 and verbose == True: print('... computing bands along a line')
        do_bands_calc_1D(Hks)

        if rank ==0: print('bands in                          %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    #----------------------
    # Start master-slaves communication
    #----------------------

    Hksp = None
    if rank == 0:
        if double_grid:

            if non_ortho:
                # now we orthogonalize the Hamiltonian again
                Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
                Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
                Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
                Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
                Hkaux = do_ortho(Hkaux,Skaux)

                HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
                non_ortho = False

            #----------------------
            # Fourier interpolation on extended grid (zero padding)
            #----------------------
            Hksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRs,nthread)
            # Naming convention (from here): 
            # Hksp = k-space Hamiltonian on interpolated grid
            if rank == 0 and verbose == True: print('Grid of k vectors for zero padding Fourier interpolation ',nk1,nk2,nk3),

            kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

            if rank ==0: print('R -> k zero padding in           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
        else:
            kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
            Hksp = Hks

    Hks =None
    Sks =None
    HRs = None
    SRs = None

    if do_dos or Boltzmann or epsilon or Berry:
        #----------------------
        # Compute eigenvalues of the interpolated Hamiltonian
        #----------------------

        eig = None
        E_k = None
        v_k = None
        if rank == 0:
            Hksp = np.reshape(Hksp,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
        for ispin in range(nspin):
            eig, E_k, v_k = calc_TB_eigs_vecs()
        if rank == 0:
            Hksp = np.reshape(Hksp,(nk1,nk2,nk3,nawf,nawf,nspin),order='C')

        index = None
        if rank == 0:
            nk1,nk2,nk3,_,_,_ = Hksp.shape
            index = {'nk1':nk1,'nk2':nk2,'nk3':nk3}
        index = comm.bcast(index,root=0)
        nk1 = index['nk1']
        nk2 = index['nk2']
        nk3 = index['nk3']
        nktot = nk1*nk2*nk3

        if rank ==0: print('eigenvalues in                   %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    if do_dos:
        #----------------------
        # DOS calculation with gaussian smearing on double_grid Hksp
        #----------------------

        index = None
        if rank == 0:
            index = {'eigtot':eig.shape[0]}
        index = comm.bcast(index,root=0)
        eigtot = index['eigtot']

        eigup = None
        eigdw = None

        if nspin == 1 or nspin == 2: 
            if rank == 0: eigup = eig[:,0]
            do_dos_calc(eigup,emin,emax,delta,eigtot,nawf,0)
            eigup = None
        if nspin == 2:
            if rank == 0: eigdw = eig[:,1]
            do_dos_calc(eigdw,emin,emax,delta,eigtot,nawf,1)
            eigdw = None

        if rank ==0: print('dos in                           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    R = None
    if Boltzmann or epsilon or Berry:
        if rank == 0:
            #----------------------
            # Compute the gradient of the k-space Hamiltonian
            #----------------------

            # fft grid in R shifted to have (0,0,0) in the center
            R,Rfft,R_wght,nrtot,idx = get_R_grid_fft()

            HRaux  = np.zeros_like(Hksp)
            for ispin in range(nspin):
                for n in range(nawf):
                    for m in range(nawf):
                        fft = pyfftw.FFTW(Hksp[:,:,:,n,m,ispin],HRaux[:,:,:,n,m,ispin],axes=(0,1,2),direction='FFTW_BACKWARD',\
                              flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                        HRaux[:,:,:,n,m,ispin] = fft()
            HRaux = FFT.fftshift(HRaux,axes=(0,1,2))

            Hksp = None

            dHksp  = np.zeros((nk1,nk2,nk3,3,nawf,nawf,nspin),dtype=complex)
            Rfft = np.reshape(Rfft,(nk1*nk2*nk3,3),order='C')
            HRaux = np.reshape(HRaux,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
            for l in range(3):
                # Compute R*H(R)
                dHRaux  = np.zeros_like(HRaux)
                for ispin in range(nspin):
                    for n in range(nawf):
                        for m in range(nawf):
                            dHRaux[:,n,m,ispin] = 1.0j*alat*Rfft[:,l]*HRaux[:,n,m,ispin]
                dHRaux = np.reshape(dHRaux,(nk1,nk2,nk3,nawf,nawf,nspin),order='C')

                # Compute dH(k)/dk
                for ispin in range(nspin):
                    for n in range(nawf):
                        for m in range(nawf):
                            fft = pyfftw.FFTW(dHRaux[:,:,:,n,m,ispin],dHksp[:,:,:,l,n,m,ispin],axes=(0,1,2), \
                            direction='FFTW_FORWARD',flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
                            dHksp[:,:,:,l,n,m,ispin] = fft()
                dHRaux = None

            HRaux = None

            print('gradient in                      %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()

        #----------------------
        # Compute the momentum operator p_n,m(k)
        #----------------------

        if rank != 0: 
            dHksp = None
            v_k = None
            pksp = None
        if rank == 0:
            dHksp = np.reshape(dHksp,(nk1*nk2*nk3,3,nawf,nawf,nspin),order='C')
        pksp = do_momentum()

        dHksp = None

        if rank == 0: print('momenta in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

        index = None
        if rank == 0:
            index = {'nawf':E_k.shape[1],'nktot':E_k.shape[0]}
        index = comm.bcast(index,root=0)
        nawf = index['nawf']
        nktot = index['nktot']

        #if rank != 0: E_k = np.zeros((nktot,nawf,nspin),dtype=float)
        #comm.Bcast(E_k,root=0)

        kq_wght = np.ones((nktot),dtype=float)
        kq_wght /= float(nktot)

    velkp = None
    if rank == 0:
        if Boltzmann:
            #----------------------
            # Compute velocities for Boltzmann transport
            #----------------------
            velkp = np.zeros((nk1*nk2*nk3,3,nawf,nspin),dtype=float)
            for n in range(nawf):
                velkp[:,:,n,:] = np.real(pksp[:,:,n,n,:])

    if Berry:
        #----------------------
        # Compute Berry curvature... (only the z component for now - Anomalous Hall Conductivity (AHC))
        #----------------------

        temp = 0.025852  # set room temperature in eV
        alat /= ANGSTROM_AU

        if do_bands:
            #----------------------
            # ...on a path in the BZ or...
            #----------------------
            Om_zk,ahc = do_Berry_curvature(0)
        else:
            #----------------------
            # ...in the full BZ
            #----------------------
            Om_zk,ahc = do_Berry_curvature(1)

        alat *= ANGSTROM_AU

        if rank == 0:
            f=open('ahc.dat','w')
            ahc = ahc*EVTORY*AU_TO_OHMCMM1
            f.write(' Anomalous Hall conductivity sigma_xy = %.6f\n' %ahc)
            f.close()

        if rank == 0: print('Berry curvature in               %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    if Boltzmann:
        #----------------------
        # Compute transport quantities (conductivity, Seebeck and thermal electrical conductivity)
        #----------------------
        temp = 0.025852  # set room temperature in eV

        ene = None

        for ispin in range(nspin):
            ene,L0,L1,L2 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin)

            #----------------------
            # Conductivity (in units of 1.e21/Ohm/m/s)
            #----------------------

            L0 *= ELECTRONVOLT_SI**2/(4.0*np.pi**3)* \
                  (ELECTRONVOLT_SI/(H_OVER_TPI**2*BOHR_RADIUS_SI))*1.0e-21
            if rank == 0:
                f=open('sigma_'+str(ispin)+'.dat','w')
                for n in range(ene.size):
                    f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],L0[0,0,n],L0[1,1,n],L0[2,2,n],L0[0,1,n],L0[0,2,n],L0[1,2,n]))
                f.close()

            #----------------------
            # Seebeck (in units of 1.e-4 V/K)
            #----------------------

            S = np.zeros((3,3,ene.size),dtype=float)

            L0 *= 1.0e21
            L1 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**2/(H_OVER_TPI**2*BOHR_RADIUS_SI))

            if rank == 0:
                for n in range(ene.size):
                    S[:,:,n] = LAN.inv(L0[:,:,n])*L1[:,:,n]*(-K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**2))*1.e4

                f=open('Seebeck_'+str(ispin)+'.dat','w')
                for n in range(ene.size):
                    f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                            %(ene[n],S[0,0,n],S[1,1,n],S[2,2,n],S[0,1,n],S[0,2,n],S[1,2,n]))
                f.close()

            #----------------------
            # Electron thermal conductivity ((in units of 1.e15 W/m/K/s)
            #----------------------

            kappa = np.zeros((3,3,ene.size),dtype=float)

            L2 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**3/(H_OVER_TPI**2*BOHR_RADIUS_SI))

            if rank == 0:
                for n in range(ene.size):
                    kappa[:,:,n] = (L2[:,:,n] - L1[:,:,n]*LAN.inv(L0[:,:,n])*L1[:,:,n])* \
                    (K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**3))*1.e-15

                f=open('kappa_'+str(ispin)+'.dat','w')
                for n in range(ene.size):
                    f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                            %(ene[n],kappa[0,0,n],kappa[1,1,n],kappa[2,2,n],kappa[0,1,n],kappa[0,2,n],kappa[1,2,n]))
                f.close()

        velkp = None
        L0 = None
        L1 = None
        L2 = None
        S = None
        kappa = None

        if rank ==0: print('transport in                     %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    if epsilon:

        #----------------------
        # Compute dielectric tensor (Re and Im epsilon)
        #----------------------

        temp = 0.025852  # set room temperature in eV

        omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

        for ispin in range(nspin):

            ene, epsi, epsr = do_epsilon(E_k,pksp,kq_wght,omega,delta,temp,ispin)

            if rank == 0:
                f=open('epsi_'+str(ispin)+'.dat','w')
                for n in range(ene.size):
                    f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                            %(ene[n],epsi[0,0,n],epsi[1,1,n],epsi[2,2,n],epsi[0,1,n],epsi[0,2,n],epsi[1,2,n]))
                f.close()
                f=open('epsr_'+str(ispin)+'.dat','w')
                for n in range(ene.size):
                    f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                            %(ene[n],epsr[0,0,n],epsr[1,1,n],epsr[2,2,n],epsr[0,1,n],epsr[0,2,n],epsr[1,2,n]))
                f.close()


        if rank ==0: print('epsilon in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))

    # Timing
    if rank ==0: print('   ')
    if rank ==0: print('Total CPU time =                 %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
