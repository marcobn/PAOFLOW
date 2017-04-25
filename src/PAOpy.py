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

# import general modules
from __future__ import print_function
from scipy import fftpack as FFT
from scipy import linalg as LA
from numpy import linalg as LAN
import xml.etree.ElementTree as ET
import numpy as np
#import numexpr as ne
import sys, time
from mpi4py import MPI
import multiprocessing

# Define paths
sys.path.append(sys.path[0]+'/defs')

# Import TB specific functions
from read_input import *
from build_Pn import *
from build_Hks import *
from do_non_ortho import *
from do_ortho import *
from get_R_grid_fft import *
from get_K_grid_fft import *
from do_bands_calc import *
from do_bands_calc_1D import *
from do_double_grid import *
from do_spin_orbit import *
from constants import *
from read_QE_output_xml_parse import *
from write3Ddatagrid import *
#----------------------
# initialize parallel execution
#----------------------
comm=MPI.COMM_WORLD
size=comm.Get_size()

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
    print('#                                          ',PAOPY,'                                           #')
    print('#                                                                                           #')
    print('#                  Utility to construct and operate on Hamiltonians from                    #')
    print('#                 the Projections of DFT wfc on Atomic Orbital bases (PAO)                  #')
    print('#                                                                                           #')
    print('#                     ',str('%1s' %CC),'2016,2017 ERMES group (http://ermes.unt.edu)                       #')
    print('#############################################################################################')
    print('          ')

#----------------------
# Initialize n. of threads for multiprocessing (FFTW)
#----------------------
nthread = multiprocessing.cpu_count()
#ne.set_num_threads(nthread)

#----------------------
# Check for fftw libraries
#----------------------
scipyfft = False
try:
    import pyfftw
except:
    if rank == 0: print('using scipy FFT')
    scipyfft = True

#----------------------
# Read input
#----------------------
from input_default import *
try:
    from inputfile import *
except:
    if rank == 0: print('missing inputfile.py ...')
    if rank == 0: print('using default input module')
    pass

if size >  1:
    if rank == 0 and npool == 1: print('parallel execution on ',size,' processors, ',nthread,' threads and ',npool,' pool')
    if rank == 0 and npool > 1: print('parallel execution on ',size,' processors, ',nthread,' threads and ',npool,' pools')
else:
    if rank == 0: print('serial execution')

#----------------------
# Do dimension checks
#----------------------

nktot=nfft1*nfft2*nfft3
if nktot%npool != 0: 
    if rank == 0 : print('npool not compatible with MP mesh',nktot,npool)
    sys.exit()
nkpool = nktot/npool
ini_ik, end_ik = load_balancing(size,rank,nkpool)
nsize = end_ik-ini_ik
if nkpool%nsize != 0: 
    if rank == 0 : print('npool not compatible with nsize',nkpool,nsize)
    sys.exit()

#----------------------
# Read DFT data
#----------------------

if (not non_ortho):
    U, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, dftSO, kpnts, kpnts_wght, \
    nelec, nbnds, Efermi, nawf, nk1, nk2, nk3,natoms  =  read_QE_output_xml(fpath, verbose, non_ortho)
    Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
    sumk = np.sum(kpnts_wght)
    kpnts_wght /= sumk
    for ik in xrange(nkpnts):
        Sks[:,:,ik]=np.identity(nawf)
    if rank == 0 and verbose: print('...using orthogonal algorithm')
else:
    U, Sks, my_eigsmat, alat, a_vectors, b_vectors, \
    nkpnts, nspin, dftSO, kpnts, kpnts_wght, \
    nelec, nbnds, Efermi, nawf, nk1, nk2, nk3,natoms  =  read_QE_output_xml(fpath,verbose,non_ortho)
    if rank == 0 and verbose: print('...using non-orthogonal algorithm')

if nk1%2. != 0 or nk2%2. != 0 or nk3%2. != 0:
    if rank == 0: print('CAUTION! nk1 or nk2 or nk3 not even!')

#----------------------
# Do memory checks 
#----------------------

gbyte = nawf**2*nfft1*nfft2*nfft3*3*2*16./1.e9
if rank == 0: print('estimated maximum array size: %5.2f GBytes' %(gbyte))
if rank == 0: print('   ')

if rank == 0: print('reading in                       %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
reset=time.time()

#----------------------
# Building the Projectability
#----------------------
Pn = build_Pn(nawf,nbnds,nkpnts,nspin,U)

if rank == 0 and verbose: print('Projectability vector ',Pn)

# Check projectability and decide bnd

bnd = 0
for n in xrange(nbnds):
    if Pn[n] > pthr:
        bnd += 1
if rank == 0 and verbose: print('# of bands with good projectability (>',pthr,') = ',bnd)
if rank == 0 and verbose and bnd < nbnds: print('Range of suggested shift ',np.amin(my_eigsmat[bnd,:,:]),' , ', \
                                np.amax(my_eigsmat[bnd,:,:]))
if shift == 'auto': shift = np.amin(my_eigsmat[bnd,:,:])

#----------------------
# Building the TB Hamiltonian
#----------------------
nbnds_norm = nawf
Hks,Sks = build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U,Sks)

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
    Hks = do_non_ortho(Hks,Sks)

if rank == 0 and write2file:
    #----------------------
    # write to file Hks,Sks,kpnts,kpnts_wght
    #----------------------                                                                                                                           
    if write_binary:# or whatever you want to call it            
        if nspin==1:#postfix .npy just to make it clear what they are
            np.save('kham.npy',np.ravel(Hks[...,0],order='C'))
        if nspin==2:
            np.save('kham_up.npy',np.ravel(Hks[...,0],order='C'))
            np.save('kham_dn.npy',np.ravel(Hks[...,1],order='C'))
        np.save('kovp.npy',np.ravel(Sks,order='C'))
    else:
        f=open('kham.txt','w')
        for ik in xrange(nkpnts):
            for i in xrange(nawf):
                for j in xrange(nawf):
                    f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
        f.close()
        f=open('kovp.txt','w')
        for ik in xrange(nkpnts):
            for i in xrange(nawf):
                for j in xrange(nawf):
                    f.write('%20.13f %20.13f \n' %(np.real(Sks[i,j,ik]),np.imag(Sks[i,j,ik])))
        f.close()
        f=open('k.txt','w')
        for ik in xrange(nkpnts):
            f.write('%20.13f %20.13f %20.13f \n' %(kpnts[ik,0],kpnts[ik,1],kpnts[ik,2]))
        f.close()
        f=open('wk.txt','w')
        for ik in xrange(nkpnts):
            f.write('%20.13f \n' %(kpnts_wght[ik]))
        f.close()
    print('H(k),S(k),k,wk written to file')


#----------------------
# Plot the TB and DFT eigevalues. Writes to comparison.pdf
#----------------------
if rank == 0 and do_comparison:
    from plot_compare_TB_DFT_eigs import *
    plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,non_ortho)
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

HRaux = np.zeros_like(Hkaux)
SRaux = np.zeros_like(Skaux)

HRaux = FFT.ifftn(Hkaux,axes=[2,3,4])
if non_ortho:
    SRaux = FFT.ifftn(Skaux,axes=[2,3,4])

# NOTE: Naming convention (from here):
# Hks = k-space Hamiltonian on original MP grid
# HRs = R-space Hamiltonian on original MP grid
if non_ortho:
    if Boltzmann or epsilon:
        Sks_long = Sks
Hks_long = Hks
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

    HRs = do_spin_orbit_calc(HRs,natoms,theta,phi,socStrengh)
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
        Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
        Skaux = np.reshape(Skaux,(nawf,nawf,nk1*nk2*nk3),order='C')
        Hkaux = do_ortho(Hkaux,Skaux)
        Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
        Skaux = np.reshape(Skaux,(nawf,nawf,nk1,nk2,nk3),order='C')
        HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
        non_ortho = False

    # Define real space lattice vectors
    R,Rfft,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

    # Define k-point mesh for bands interpolation
    kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,dkres)
    nkpi=kq.shape[1]
    for n in xrange(nkpi):
        kq[:,n]=np.dot(kq[:,n],b_vectors)

    # Compute the bands along the path in the IBZ
    E_kp,v_kp = do_bands_calc(HRs,SRs,kq,R_wght,R,idx,non_ortho)

    if rank == 0: print('bands in                         %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

    if band_topology:
        # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry curvature operators along the path in the IBZ
        from do_topology_calc import *
        do_topology_calc(HRs,SRs,non_ortho,kq,E_kp,v_kp,R,Rfft,R_wght,idx,alat,b_vectors,nelec,bnd,Berry,ipol,jpol,spin_Hall,spol,do_spin_orbit,sh,nl)
        if rank == 0: print('band topology in                 %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    if double_grid == False:
        if rank ==0: print('   ')
        if rank ==0: print('Total CPU time =                 %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
        sys.exit()


    alat *= ANGSTROM_AU

elif do_bands and onedim:
    #----------------------
    # FFT interpolation along a single directions in the BZ
    #----------------------
    if rank == 0 and verbose: print('... computing bands along a line')
    do_bands_calc_1D(Hks)

    if rank ==0: print('bands in                          %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

#----------------------
# Initialize Read/Write restart data
#----------------------
checkpoint = 0
if rank == 0:
    if restart:
        try:
            datadump = np.load(fpath+'PAOdump.npz')
            checkpoint = datadump['checkpoint']
            print('reading data from dump at checkpoint ',checkpoint)
        except:
            pass
checkpoint = comm.bcast(checkpoint,root=0)

#----------------------
# Start master-slaves communication
#----------------------

Hksp = None
if rank == 0:
    if double_grid and checkpoint == 0:

        if non_ortho:
            # now we orthogonalize the Hamiltonian again
            Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
            Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
            Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
            Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
            Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
            Skaux = np.reshape(Skaux,(nawf,nawf,nk1*nk2*nk3),order='C')
            Hkaux = do_ortho(Hkaux,Skaux)
            Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
            Skaux = np.reshape(Skaux,(nawf,nawf,nk1,nk2,nk3),order='C')
            HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
            non_ortho = False

        #----------------------
        # Fourier interpolation on extended grid (zero padding)
        #----------------------
        Hksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRs,nthread,scipyfft)
        # Naming convention (from here): 
        # Hksp = k-space Hamiltonian on interpolated grid
        if rank == 0 and verbose: print('Grid of k vectors for zero padding Fourier interpolation ',nk1,nk2,nk3),

        kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

        if rank ==0: print('R -> k zero padding in           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()
    else:
        kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
        Hksp = Hks

#----------------------
# Read/Write restart data
#----------------------
if rank == 0:
    if restart:
        if checkpoint == 0:
            checkpoint += 1
            np.savez(fpath+'PAOdump.npz',checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3)
        elif checkpoint > 0:
            Hksp = datadump['Hksp']
            kq = datadump['kq']
            kq_wght = datadump['kq_wght']
            idk = datadump['idk']
            nk1 = datadump['nk1']
            nk2 = datadump['nk2']
            nk3 = datadump['nk3']
else:
    Hksp = None
    kq = None
    kq_wght = None
    idk = None
    nk1 = None
    nk2 = None
    nk3 = None
checkpoint = comm.bcast(checkpoint,root=0)

#----------------------
# Compute eigenvalues of the interpolated Hamiltonian
#----------------------
if checkpoint < 2:
    from calc_TB_eigs_vecs import *

    eig = None
    E_k = None
    v_k = None
    if rank == 0:
        Hksp = np.reshape(Hksp,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
    for ispin in xrange(nspin):
        eig, E_k, v_k = calc_TB_eigs_vecs(Hksp,ispin,npool)
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

    if rank ==0: print('eigenvalues in                   %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

#----------------------
# Read/Write restart data
#----------------------
if rank == 0:
    if restart:
        if checkpoint == 1:
            checkpoint += 1
            np.savez(fpath+'PAOdump.npz',checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3, \
                    eig=eig,E_k=E_k,v_k=v_k)
        elif checkpoint > 1:
            Hksp = datadump['Hksp']
            kq = datadump['kq']
            kq_wght = datadump['kq_wght']
            idk = datadump['idk']
            nk1 = datadump['nk1']
            nk2 = datadump['nk2']
            nk3 = datadump['nk3']
            eig = datadump['eig']
            E_k = datadump['E_k']
            v_k = datadump['v_k']
else:
    Hksp = None
    kq = None
    kq_wght = None
    idk = None
    nk1 = None
    nk2 = None
    nk3 = None
    eig = None
    E_k = None
    v_k = None
checkpoint = comm.bcast(checkpoint,root=0)
nk1 = comm.bcast(nk1,root=0)
nk2 = comm.bcast(nk2,root=0)
nk3 = comm.bcast(nk3,root=0)

if do_dos or do_pdos:
    #----------------------
    # DOS calculation with gaussian smearing on double_grid Hksp
    #----------------------
    from do_dos_calc import *

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

    if do_pdos:
        #----------------------
        # PDOS calculation
        #----------------------
        from do_pdos_calc import *

        if nspin == 1 or nspin == 2:
            if rank == 0: eigup = E_k[:,:,0]
            do_pdos_calc(eigup,emin,emax,delta,v_k,nk1,nk2,nk3,nawf,0)
            eigup = None
        if nspin == 2:
            if rank == 0: eigdw = E_k[:,:,1]
            do_pdos_calc(eigdw,emin,emax,delta,v_k,nk1,nk2,nk3,nawf,1)
            eigdw = None

    if rank ==0: print('dos in                           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

if do_fermisurf or do_spintexture:
    #----------------------
    # Fermi surface calculation
    #----------------------
    from do_fermisurf import *

    if nspin == 1 or nspin == 2:
        if rank == 0:
            eigup = E_k[:,:,0]
            do_fermisurf(fermi_dw,fermi_up,eigup,alat,b_vectors,nk1,nk2,nk3,nawf,0)
        eigup = None
    if nspin == 2:
        if rank == 0:
            eigdw = E_k[:,:,1]
            do_fermisurf(fermi_dw,fermi_up,eigdw,alat,b_vectors,nk1,nk2,nk3,nawf,0)
        eigdw = None
    if do_spintexture and nspin == 1:
        from do_spin_texture import *
        do_spin_texture(fermi_dw,fermi_up,E_k,v_k,sh,nl,nk1,nk2,nk3,nawf,nspin,do_spin_orbit,npool)

    if rank ==0: print('FermiSurf in                     %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

pksp = None
jksp = None
if Boltzmann or epsilon or Berry or spin_Hall or critical_points or smearing != None:
    if checkpoint < 3:
        #----------------------
        # Compute the gradient of the k-space Hamiltonian
        #----------------------
        from do_gradient import *
        dHksp = do_gradient(Hksp,a_vectors,alat,nthread,npool,scipyfft)

        if rank == 0:
            print('gradient in                      %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()

    #----------------------
    # Read/Write restart data
    #----------------------
    if rank == 0:
        if restart:
            if checkpoint == 2:
                checkpoint += 1
                np.savez(fpath+'PAOdump.npz',checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3, \
                        eig=eig,E_k=E_k,v_k=v_k,dHksp=dHksp)
            elif checkpoint > 2:
                Hksp = datadump['Hksp']
                kq = datadump['kq']
                kq_wght = datadump['kq_wght']
                idk = datadump['idk']
                nk1 = datadump['nk1']
                nk2 = datadump['nk2']
                nk3 = datadump['nk3']
                eig = datadump['eig']
                E_k = datadump['E_k']
                v_k = datadump['v_k']
                dHksp = datadump['dHksp']
    else:
        Hksp = None
        kq = None
        kq_wght = None
        idk = None
        nk1 = None
        nk2 = None
        nk3 = None
        eig = None
        E_k = None
        v_k = None
        dHksp = None
    checkpoint = comm.bcast(checkpoint,root=0)

    if checkpoint < 4:
        #----------------------
        # Compute the momentum operator p_n,m(k)
        #----------------------
        from do_momentum import *

        if rank != 0:
            dHksp = None
            v_k = None
            pksp = None
        if rank == 0:
            dHksp = np.reshape(dHksp,(nk1*nk2*nk3,3,nawf,nawf,nspin),order='C')
        pksp = do_momentum(v_k,dHksp,npool)

        if rank == 0: print('momenta in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    #----------------------
    # Read/Write restart data
    #----------------------
    if rank == 0:
        if restart:
            if checkpoint == 3:
                checkpoint += 1
                np.savez(fpath+'PAOdump.npz',checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3, \
                        eig=eig,E_k=E_k,v_k=v_k,dHksp=dHksp,pksp=pksp)
            elif checkpoint > 3:
                Hksp = datadump['Hksp']
                kq = datadump['kq']
                kq_wght = datadump['kq_wght']
                idk = datadump['idk']
                nk1 = datadump['nk1']
                nk2 = datadump['nk2']
                nk3 = datadump['nk3']
                eig = datadump['eig']
                E_k = datadump['E_k']
                v_k = datadump['v_k']
                dHksp = datadump['dHksp']
                pksp = datadump['pksp']
    else:
        Hksp = None
        kq = None
        kq_wght = None
        idk = None
        nk1 = None
        nk2 = None
        nk3 = None
        eig = None
        E_k = None
        v_k = None
        dHksp = None
        pksp = None
    checkpoint = comm.bcast(checkpoint,root=0)
    nk1 = comm.bcast(nk1,root=0)
    nk2 = comm.bcast(nk2,root=0)
    nk3 = comm.bcast(nk3,root=0)

    if spin_Hall:
        #----------------------
        # Compute the spin current operator j^l_n,m(k)
        #----------------------
        jksp = None
        spincheck = 0
        if restart and rank == 0:
            try:
                spindump = np.load(fpath+'PAOspin'+str(spol)+'.npz')
                jksp = spindump['jksp']
                spincheck += 1
                print('reading spin current for polarization ',spol)
            except:
                pass
        spincheck=comm.bcast(spincheck,root=0)
        if spincheck == 0:
            from do_spin_current import *
            jksp = do_spin_current(v_k,dHksp,spol,npool,do_spin_orbit,sh,nl)
            if restart and rank == 0:
                np.savez(fpath+'PAOspin'+str(spol)+'.npz',jksp=jksp)
            if rank == 0: print('spin current in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()

    dHksp = None

    index = None
    if rank == 0:
        index = {'nawf':E_k.shape[1],'nktot':E_k.shape[0]}
    index = comm.bcast(index,root=0)
    nawf = index['nawf']
    nktot = index['nktot']

    kq_wght = np.ones((nktot),dtype=float)
    kq_wght /= float(nktot)

deltakp = None
deltakp2 = None
if rank == 0:
    if smearing != None:
        #----------------------
        # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
        #----------------------
        deltakp = np.zeros((nk1*nk2*nk3,nawf,nspin),dtype=float)
        deltakp2 = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin),dtype=float)
        omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
        dk = (8.*np.pi**3/omega/(nk1*nk2*nk3))**(1./3.)
        if smearing == 'gauss':
            afac = 0.7
        elif smearing == 'm-p':
            afac = 1.0

        for n in xrange(nawf):
            deltakp[:,n,:] = afac*LAN.norm(np.real(pksp[:,:,n,n,:]),axis=1)*dk
            for m in xrange(nawf):
                deltakp2[:,n,m,:] = afac*LAN.norm(np.real(np.absolute(pksp[:,:,n,n,:]-pksp[:,:,m,m,:])),axis=1)*dk

        if restart:
            np.savez(fpath+'PAOdelta'+str(nspin)+'.npz',deltakp=deltakp,deltakp2=deltakp2)

velkp = None
if rank == 0:
    if Boltzmann or critical_points:
        #----------------------
        # Compute velocities for Boltzmann transport
        #----------------------
        velkp = np.zeros((nk1*nk2*nk3,3,nawf,nspin),dtype=float)
        for n in xrange(nawf):
            velkp[:,:,n,:] = np.real(pksp[:,:,n,n,:])

        if critical_points:
            #----------------------
            # Find critical points (grad(E_kn)=0)
            #----------------------
            f=open('critical_points.dat','w')
            for ik in xrange(nk1*nk2*nk3):
                for n in xrange(bnd):
                    for ipin in xrange(nspin):
                        if  np.abs(velkp[ik,0,n,ispin]) < 1.e-2 and \
                            np.abs(velkp[ik,1,n,ispin]) < 1.e-2 and \
                            np.abs(velkp[ik,2,n,ispin]) < 1.e-2:
                            f.write('band %5d at %.5f %.5f %.5f \n' %(n,kq[0,ik],kq[1,ik],kq[2,ik]))
            f.close()

if do_dos and smearing != None:
    #----------------------
    # DOS calculation with adaptive gaussian smearing on double_grid Hksp - TESTING
    #----------------------
    from do_dos_calc_adaptive import *

    index = None
    if rank == 0:
        index = {'eigtot':eig.shape[0]}
    index = comm.bcast(index,root=0)
    eigtot = index['eigtot']

    eigup = None
    eigdw = None
    deltakpup = None
    deltakpdw = None

    if rank == 0: deltakp = np.reshape(deltakp,(nk1*nk2*nk3*nawf,nspin),order='C')

    if nspin == 1 or nspin == 2:
        if rank == 0:
            eigup = eig[:,0]
            deltakpup = deltakp[:,0]
        do_dos_calc_adaptive(eigup,emin,emax,deltakpup,eigtot,nawf,0,smearing)
        eigup = None
        deltakpup = None
    if nspin == 2:
        if rank == 0:
            eigdw = eig[:,1]
            deltakpdw = deltakp[:,1]
        do_dos_calc_adaptive(eigdw,emin,emax,deltakpdw,eigtot,nawf,1,smearing)
        eigdw = None
        deltakpdw = None
    if rank == 0: deltakp = np.reshape(deltakp,(nk1*nk2*nk3,nawf,nspin),order='C')

if Berry:
    #----------------------
    # Compute Berry curvature... (only the z component for now - Anomalous Hall Conductivity (AHC))
    #----------------------
    from do_Berry_curvature import *
    from do_Berry_conductivity import *

    Om_k = np.zeros((nk1,nk2,nk3,2),dtype=float)
    ene,ahc,Om_k[:,:,:,0] = do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltakp,smearing)

    if rank == 0 and writedata:
        from write2bxsf import *
        x0 = np.zeros(3,dtype=float)
        ind_plot = np.zeros(2)
        Om_k[:,:,:,1] = Om_k[:,:,:,0]
        write2bxsf(fermi_dw,fermi_up,Om_k,nk1,nk2,nk3,2,ind_plot,0.0,alat,x0,b_vectors,'Berry_'+str(LL[ipol])+str(LL[jpol])+'.bxsf')

        np.savez('Berry_'+str(LL[ipol])+str(LL[jpol])+'.npz',kq=kq,Om_k=Om_k[:,:,:,0])

    if ac_cond_Berry:
        ene_ac,sigxy = do_Berry_conductivity(E_k,pksp,temp,ispin,npool,ipol,jpol,deltakp,deltakp2,smearing)
        ahc0 = np.real(sigxy[0])

    omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

    if rank == 0 and ene.size == 1:
        f=open('ahc.dat','w')
        ahc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
        if ac_cond_Berry:
            ahc0 *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
            f.write(' Anomalous Hall conductivity sigma_xy = %.6f (%.6f)\n' %(ahc,ahc0))
        else:
            f.write(' Anomalous Hall conductivity sigma_xy = %.6f \n' %ahc)
        f.close()
    elif rank == 0:
        ahc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
        f=open('ahcEf_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
        for n in xrange(ene.size):
            f.write('%.5f %9.5e \n' %(ene[n],ahc[n]))
        f.close()

    if rank == 0 and ac_cond_Berry:
        sigxy *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
        f=open('sigxyi.dat','w')
        for n in xrange(ene.size):
            f.write('%.5f %9.5e \n' %(ene_ac[n],np.imag(ene_ac[n]*sigxy[n]/105.4571)))  #convert energy in freq (1/hbar in cgs units)
        f.close()
        f=open('sigxyr.dat','w')
        for n in xrange(ene.size):
            f.write('%.5f %9.5e \n' %(ene_ac[n],np.real(sigxy[n])))
        f.close()

    if rank == 0: print('Berry module in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

if spin_Hall:
    #----------------------
    # Compute spin Berry curvature... 
    #----------------------
    from do_spin_Berry_curvature import *
    from do_spin_Hall_conductivity import *

    Om_k = np.zeros((nk1,nk2,nk3,2),dtype=float)
    ene,shc,Om_k[:,:,:,0] = do_spin_Berry_curvature(E_k,jksp,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltakp,smearing)

    if rank == 0 and writedata:
        from write2bxsf import *
        x0 = np.zeros(3,dtype=float)
        ind_plot = np.zeros(2)
        Om_k[:,:,:,1] = Om_k[:,:,:,0]
        write2bxsf(fermi_dw,fermi_up,Om_k,nk1,nk2,nk3,2,ind_plot,0.0,alat,x0,b_vectors,'spin_Berry_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.bxsf')

    if ac_cond_spin:
        ene,sigxy = do_spin_Hall_conductivity(E_k,jksp,pksp,temp,ispin,npool,ipol,jpol)
        shc0 = np.real(sigxy[0])

    omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

    if rank == 0 and ene.size == 1:
        f=open('shc.dat','w')
        shc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
        if ac_cond_spin:
            shc0 *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
            f.write(' spin Hall conductivity sigma^z_xy = %.6f (%.6f)\n' %(shc,shc0))
        else:
            f.write(' spin Hall conductivity sigma^z_xy = %.6f \n' %shc)
        f.close()
    elif rank == 0:
        shc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
        f=open('shcEf_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
        for n in xrange(ene.size):
            f.write('%.5f %9.5e \n' %(ene[n],shc[n]))
        f.close()

    if rank == 0: print('spin Hall module in              %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    reset=time.time()

if Boltzmann:
    #----------------------
    # Compute transport quantities (conductivity, Seebeck and thermal electrical conductivity)
    #----------------------
    from do_Boltz_tensors import *

    for ispin in xrange(nspin):
        ene,L0,L1,L2 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin)

        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------

        L0 *= ELECTRONVOLT_SI**2/(4.0*np.pi**3)* \
              (ELECTRONVOLT_SI/(H_OVER_TPI**2*BOHR_RADIUS_SI))*1.0e-21
        if rank == 0:
            f=open('sigma_'+str(ispin)+'.dat','w')
            for n in xrange(ene.size):
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
            for n in xrange(ene.size):
                S[:,:,n] = LAN.inv(L0[:,:,n])*L1[:,:,n]*(-K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**2))*1.e4

            f=open('Seebeck_'+str(ispin)+'.dat','w')
            for n in xrange(ene.size):
                f.write('%.5f %9.5e %9.5e %9.5e %9.5e %9.5e %9.5e \n' \
                        %(ene[n],S[0,0,n],S[1,1,n],S[2,2,n],S[0,1,n],S[0,2,n],S[1,2,n]))
            f.close()

        #----------------------
        # Electron thermal conductivity ((in units of 1.e15 W/m/K/s)
        #----------------------

        kappa = np.zeros((3,3,ene.size),dtype=float)

        L2 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**3/(H_OVER_TPI**2*BOHR_RADIUS_SI))

        if rank == 0:
            for n in xrange(ene.size):
                kappa[:,:,n] = (L2[:,:,n] - L1[:,:,n]*LAN.inv(L0[:,:,n])*L1[:,:,n])*(K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**3))*1.e-15

            f=open('kappa_'+str(ispin)+'.dat','w')
            for n in xrange(ene.size):
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
    from do_epsilon import *

    omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

    for ispin in xrange(nspin):

        ene, epsi, epsr = do_epsilon(E_k,pksp,kq_wght,omega,delta,temp,ipol,jpol,ispin,metal,ne,epsmin,epsmax,deltakp,deltakp2,smearing)

        if rank == 0:
            f=open('epsi_'+str(LL[ipol])+str(LL[jpol])+'_'+str(ispin)+'.dat','w')
            for n in xrange(ene.size):
                f.write('%.5f %9.5e \n' \
                        %(ene[n],epsi[ipol,jpol,n]))
            f.close()
            f=open('epsr_'+str(LL[ipol])+str(LL[jpol])+'_'+str(ispin)+'.dat','w')
            for n in xrange(ene.size):
                f.write('%.5f %9.5e \n' \
                        %(ene[n],epsr[ipol,jpol,n]))
            f.close()


    if rank ==0: print('epsilon in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))

# Timing
if rank ==0: print('   ')
if rank ==0: print('Total CPU time =                 %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
quit()
