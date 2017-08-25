# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2017      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2017 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
#
#  This file is part of AFLOW software.
#
#  AFLOW is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************************

# future imports
from __future__ import print_function

try:
    # import general modules
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

    # Import PAO specific functions
    from build_Pn import *
    from build_Hks import *
    from do_non_ortho import *
    from do_ortho import *
    from add_ext_field import *
    from get_R_grid_fft import *
    from get_K_grid_fft import *
    from do_bands_calc import *
    from do_bands_calc_1D import *
    from do_double_grid import *
    from do_spin_orbit import *
    from constants import *
    from read_QE_output_xml_parse import *
    from write3Ddatagrid import *
except Exception as e:
    print('Exception in Imports')
    print(e)
    quit()

try:
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
        print('#                                          PAOFLOW                                          #')
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
except Exception as e:
    print('Rank %d: Exception in Initialization'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
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

    #----------------------
    # Check for fftw libraries
    #----------------------
    scipyfft = False
    if rank == 0:
        if use_cuda:
            from cuda_fft import *
            print('using skcuda FFT')
        else:
            try:
                import pyfftw
                print('using pyFFTW')
            except:
                print('using scipy FFT')
                scipyfft = True

    if size >  1:
        if rank == 0 and npool == 1: print('parallel execution on ',size,' processors, ',nthread,' threads and ',npool,' pool')
        if rank == 0 and npool > 1: print('parallel execution on ',size,' processors, ',nthread,' threads and ',npool,' pools')
    else:
        if rank == 0: print('serial execution')

    #----------------------
    # Do dimension checks
    #----------------------

    wrongdim = False
    if rank == 0:
        nktot=nfft1*nfft2*nfft3
        if nktot%npool != 0:
            print('npool not compatible with MP mesh',nktot,npool)
            wrongdim = True
    comm.bcast(wrongdim,root=0)
    if wrongdim: quit()

    comm.Barrier()
except Exception as e:
    print('Rank %d: Exception in FFT Library or Dimension Check'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Read DFT data
    #----------------------

    # Initialize variables
    U = None
    my_eigsmat = None
    alat = None
    a_vectors = b_vectors = None
    nkpnts = None
    nspin = None
    dftSO = None
    kpnts_wght = None
    nelec = None
    nbnd = None
    Efermi = None
    nawf = None
    nk1 = nk2 = nk3 = None
    natoms = None
    tau = None
    Sks = None

    if rank == 0:

        if (not non_ortho):
            U,my_eigsmat,alat,a_vectors,b_vectors, \
            nkpnts,nspin,dftSO,kpnts,kpnts_wght, \
            nelec,nbnds,Efermi,nawf,nk1,nk2,nk3,natoms,tau  =  read_QE_output_xml(fpath, verbose, non_ortho)
            Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
            sumk = np.sum(kpnts_wght)
            kpnts_wght /= sumk
            for ik in xrange(nkpnts):
                Sks[:,:,ik]=np.identity(nawf)
            if verbose: print('...using orthogonal algorithm')
        else:
            U,Sks,my_eigsmat,alat,a_vectors,b_vectors, \
            nkpnts,nspin,dftSO,kpnts,kpnts_wght, \
            nelec,nbnds,Efermi,nawf,nk1,nk2,nk3,natoms,tau  =  read_QE_output_xml(fpath,verbose,non_ortho)
            if verbose: print('...using non-orthogonal algorithm')

        if nk1%2. != 0 or nk2%2. != 0 or nk3%2. != 0:
            print('CAUTION! nk1 or nk2 or nk3 not even!')

    # Broadcast data
    alat = comm.bcast(alat,root=0)
    a_vectors = comm.bcast(a_vectors,root=0)
    b_vectors = comm.bcast(b_vectors,root=0)
    nkpnts = comm.bcast(nkpnts,root=0)
    nspin = comm.bcast(nspin,root=0)
    dftSO = comm.bcast(dftSO,root=0)
    kpnts_wght = comm.bcast(kpnts_wght,root=0)
    nelec = comm.bcast(nelec,root=0)
    nbnd = comm.bcast(nbnd,root=0)
    Efermi = comm.bcast(Efermi,root=0)
    nawf = comm.bcast(nawf,root=0)
    nk1 = comm.bcast(nk1,root=0)
    nk2 = comm.bcast(nk2,root=0)
    nk3 = comm.bcast(nk3,root=0)
    natoms = comm.bcast(natoms,root=0)
    tau = comm.bcast(tau,root=0)

    #----------------------
    # Do memory checks 
    #----------------------

    if rank == 0:
        gbyte = nawf**2*nfft1*nfft2*nfft3*3*2*16./1.e9
        print('estimated maximum array size: %5.2f GBytes' %(gbyte))
        print('   ')

    comm.Barrier()
    if rank == 0:
        print('reading in                       %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
        reset=time.time()
except Exception as e:
    print('Rank %d: Exception in Reading DFT Data'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Building the Projectability
    #----------------------
    bnd = None
    if rank != 0: shift = None
    if rank == 0:
        Pn = build_Pn(nawf,nbnds,nkpnts,nspin,U)

        if verbose: print('Projectability vector ',Pn)

        # Check projectability and decide bnd

        bnd = 0
        for n in xrange(nbnds):
            if Pn[n] > pthr:
                bnd += 1
        if verbose: print('# of bands with good projectability (>',pthr,') = ',bnd)
        if verbose and bnd < nbnds: print('Range of suggested shift ',np.amin(my_eigsmat[bnd,:,:]),' , ', \
                                        np.amax(my_eigsmat[bnd,:,:]))
        if shift == 'auto': shift = np.amin(my_eigsmat[bnd,:,:])

    # Broadcast 
    bnd = comm.bcast(bnd,root=0)
    shift = comm.bcast(shift,root=0)
except Exception as e:
    print('Rank %d: Exception in Building Projectability'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Building the PAO Hamiltonian
    #----------------------

    Hks = None
    if rank == 0:
        Hks,Sks = build_Hks(nawf,bnd,nkpnts,nspin,shift,my_eigsmat,shift_type,U,Sks)

    comm.Barrier()
    if rank == 0:
        print('building Hks in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()
        nawf = Hks.shape[0]
except Exception as e:
    print('Rank %d: Exception in Building PAO Hamiltonian'%rank)
    print(e)
    comm.Abort()
    raise Exception

# NOTE: Take care of non-orthogonality, if needed
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to 
# apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
# non_ortho flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
# non_ortho flag == 1 - makes H orthogonal (rotated basis) 
#    Hks = do_non_ortho(Hks,Sks)
#    Hks = do_ortho(Hks,Sks)

try:
    if rank == 0 and non_ortho:
        Hks = do_non_ortho(Hks,Sks)
except Exception as e:
    print('Rank %d: Exception in Orthogonality'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
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
            if nspin == 1:
                f=open('kham.txt','w')
                for ik in xrange(nkpnts):
                    for i in xrange(nawf):
                        for j in xrange(nawf):
                            f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
                f.close()
            elif nspin == 2:
                f=open('kham_up.txt','w')
                for ik in xrange(nkpnts):
                    for i in xrange(nawf):
                        for j in xrange(nawf):
                            f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
                f.close()
                f=open('kham_down.txt','w')
                for ik in xrange(nkpnts):
                    for i in xrange(nawf):
                        for j in xrange(nawf):
                            f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,1]),np.imag(Hks[i,j,ik,1])))
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
    if write2file: quit()
except Exception as e:
    print('Rank %d: Exception in Write to File'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Plot the PAO and DFT eigevalues. Writes to comparison.pdf
    #----------------------
    if rank == 0 and do_comparison:
        from plot_compare_PAO_DFT_eigs import *
        plot_compare_PAO_DFT_eigs(Hks,Sks,my_eigsmat,non_ortho)
        quit()
except Exception as e:
    print('Rank %d: Exception in Do Comparison'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)
    #----------------------
    HRs = SRs = None
    if rank ==0:
        # Original k grid to R grid
        reset=time.time()
        Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
        Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)

        Hkaux = np.reshape(Hks,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
        if non_ortho:
            Skaux = np.reshape(Hks,(nawf,nawf,nk1,nk2,nk3),order='C')

        HRaux = np.zeros_like(Hkaux)
        SRaux = np.zeros_like(Skaux)

        if use_cuda:
            HRaux = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
        else:
            HRaux = FFT.ifftn(Hkaux,axes=[2,3,4])

        if non_ortho:
            if use_cuda:
                SRaux = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
            else:
                SRaux = FFT.ifftn(Skaux,axes=[2,3,4])

        # NOTE: Naming convention (from here):
        # Hks = k-space Hamiltonian on original MP grid
        # HRs = R-space Hamiltonian on original MP grid
        if non_ortho:
            if Boltzmann or epsilon:
                Sks_long = Sks
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

    comm.Barrier()
    if rank == 0:
        print('k -> R in                        %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()
except Exception as e:
    print('Rank %d: Exception in k->R'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    if rank == 0 and (Efield.any() != 0.0 or Bfield.any() != 0.0 or HubbardU.any() != 0.0):
        # Add external fields or non scf ACBN0 correction
        tau_wf = np.zeros((nawf,3),dtype=float)
        l=0
        for n in xrange(natoms):
            for i in xrange(naw[n]):
                tau_wf[l,:] = tau[n,:]
                l += 1

        # Define real space lattice vectors
        R,Rfft,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

        HRs = add_ext_field(HRs,tau_wf,R,alat,Efield,Bfield,HubbardU)
except Exception as e:
    print('Rank %d: Exception in Adding External Fields'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    if rank == 0 and do_spin_orbit:
        #----------------------
        # Compute bands with spin-orbit coupling
        #----------------------

        socStrengh = np.zeros((natoms,2),dtype=float)
        socStrengh [:,0] =  lambda_p[:]
        socStrengh [:,1] =  lambda_d[:]

        HRs = do_spin_orbit_calc(HRs,natoms,theta,phi,socStrengh)
        nawf=2*nawf
except Exception as e:
    print('Rank %d: Exception in Do Spin Orbit'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    if do_bands and not(onedim):
        #----------------------
        # Compute bands on a selected path in the BZ
        #----------------------

        alat /= ANGSTROM_AU

        if rank == 0 and non_ortho:
            # now we orthogonalize the Hamiltonian again
            if use_cuda:
                Hkaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
                Hkaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(HRs,[0,1],[3,4]),axes=[0,1,2])
                Skaux  = np.zeros((nk1,nk2,nk3,nawf,nawf),dtype=complex)
                Skaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(SRs,[0,1],[3,4]),axes=[0,1,2])
                Hkaux = np.reshape(np.moveaxis(Hkaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                Skaux = np.reshape(np.moveaxis(Skaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3),order='C')
            else:
                Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
                Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
                Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
                Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
                Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                Skaux = np.reshape(Skaux,(nawf,nawf,nk1*nk2*nk3),order='C')

            Hkaux = do_ortho(Hkaux,Skaux)
            Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
            Skaux = np.reshape(Skaux,(nawf,nawf,nk1,nk2,nk3),order='C')
            if use_cuda:
                HRs = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
            else:
                HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
            non_ortho = False

        # Broadcast HRs and SRs
        HRs = comm.bcast(HRs,root=0)
        if non_ortho: SRs = comm.bcast(SRs,root=0)

        # Define real space lattice vectors
        R,Rfft,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)

        # Define k-point mesh for bands interpolation
        kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,b_vectors,nk)
        nkpi=kq.shape[1]
        for n in xrange(nkpi):
            kq[:,n]=np.dot(kq[:,n],b_vectors)

        # Compute the bands along the path in the IBZ
        E_kp = v_kp = None
        E_kp,v_kp = do_bands_calc(HRs,SRs,kq,R_wght,R,idx,non_ortho)

        comm.Barrier()
        if rank == 0:
            print('bands in                         %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()

        if band_topology:
            # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry curvature operators along the path in the IBZ
            #from do_topology_calc import *
            from do_topology_calc_new import *
            do_topology_calc(HRs,SRs,non_ortho,kq,E_kp,v_kp,R,Rfft,R_wght,idx,alat,b_vectors,nelec,bnd,Berry,ipol,jpol,spin_Hall,spol,do_spin_orbit,sh,nl)

            comm.Barrier()
            if rank == 0:
                print('band topology in                 %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()

        alat *= ANGSTROM_AU

    elif do_bands and onedim:
        #----------------------
        # FFT interpolation along a single directions in the BZ
        #----------------------
        if rank == 0 and verbose: print('... computing bands along a line')
        if rank == 0: do_bands_calc_1D(Hks)

        comm.Barrier()
        if rank ==0:
            print('bands in                          %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
except Exception as e:
    print('Rank %d: Exception in Do Bands or Band Topology'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
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
except Exception as e:
    print('Rank %d: Exception in Checkpoint Initialization'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Start master-slaves communication
    #----------------------
    Hksp = None
    if rank == 0:
        if double_grid and checkpoint == 0:

            if non_ortho:
                # now we orthogonalize the Hamiltonian again
                if use_cuda:
                    Hkaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
                    Hkaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(HRs,[0,1],[3,4]),axes=[0,1,2])
                    Skaux  = np.zeros((nk1,nk2,nk3,nawf,nawf),dtype=complex)
                    Skaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(SRs,[0,1],[3,4]),axes=[0,1,2])
                    Hkaux = np.reshape(np.moveaxis(Hkaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                    Skaux = np.reshape(np.moveaxis(Skaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3),order='C')
                else:
                    Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
                    Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
                    Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
                    Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
                    Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                    Skaux = np.reshape(Skaux,(nawf,nawf,nk1*nk2*nk3),order='C')
    
                Hkaux = do_ortho(Hkaux,Skaux)
                Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
                Skaux = np.reshape(Skaux,(nawf,nawf,nk1,nk2,nk3),order='C')
                if use_cuda:
                    HRs = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
                else:
                    HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
                non_ortho = False

            #----------------------
            # Fourier interpolation on extended grid (zero padding)
            #----------------------
            Hksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRs,nthread)
            # Naming convention (from here): 
            # Hksp = k-space Hamiltonian on interpolated grid
            if rank == 0 and verbose: print('Grid of k vectors for zero padding Fourier interpolation ',nk1,nk2,nk3),
            kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)

        else:
            kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
            Hksp = np.moveaxis(Hks,(0,1),(3,4))
            Hksp = Hksp.copy(order='C')

    comm.Barrier()
    if rank ==0:
        print('R -> k zero padding in           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

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
except Exception as e:
    print('Rank %d: Exception in Do Double Grid'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Compute eigenvalues of the interpolated Hamiltonian
    #----------------------
    if checkpoint < 2:
        from calc_PAO_eigs_vecs import *

        eig = None
        E_k = None
        v_k = None
        if rank == 0:
            Hksp = np.reshape(Hksp,(nk1*nk2*nk3,nawf,nawf,nspin),order='C')
        eig, E_k, v_k = calc_PAO_eigs_vecs(Hksp,bnd,npool)
        if rank == 0:
            Hksp = np.reshape(Hksp,(nk1,nk2,nk3,nawf,nawf,nspin),order='C')

        if rank == 0 and HubbardU.any() != 0.0:
            eig -= np.amax(E_k[:,bval,:])
            E_k -= np.amax(E_k[:,bval,:])

        index = None
        if rank == 0:
            nk1,nk2,nk3,_,_,_ = Hksp.shape
            index = {'nk1':nk1,'nk2':nk2,'nk3':nk3}
        index = comm.bcast(index,root=0)
        nk1 = index['nk1']
        nk2 = index['nk2']
        nk3 = index['nk3']

        comm.Barrier()
        if rank ==0:
            print('eigenvalues in                   %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
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
except Exception as e:
    print('Rank %d: Exception in Eigenvalues'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    if (do_dos or do_pdos) and smearing == None:
        #----------------------
        # DOS calculation with gaussian smearing on double_grid Hksp
        #----------------------
        from do_dos_calc import *

        index = None
        if rank == 0:
            index = {'eigtot':eig.shape[0]}
        index = comm.bcast(index,root=0)
        eigtot = index['eigtot']

        eigup = eigdw = None

        if nspin == 1 or nspin == 2:
            if rank == 0: eigup = np.array(eig[:,0])
            do_dos_calc(eigup,emin,emax,delta,eigtot,bnd,0)
            eigup = None
        if nspin == 2:
            if rank == 0: eigdw = np.array(eig[:,1])
            do_dos_calc(eigdw,emin,emax,delta,eigtot,bnd,1)
            eigdw = None

        if do_pdos:
            #----------------------
            # PDOS calculation
            #----------------------
            from do_pdos_calc import *

            v_kup = v_kdw = None
            if nspin == 1 or nspin == 2:
                if rank == 0:
                    eigup = np.array(E_k[:,:,0])
                    v_kup = np.array(v_k[:,:,:,0])
                do_pdos_calc(eigup,emin,emax,delta,v_kup,nk1,nk2,nk3,nawf,0)
                eigup = None
                v_kup = None
            if nspin == 2:
                if rank == 0:
                    eigdw = np.array(E_k[:,:,1])
                    v_kdw = np.array(v_k[:,:,:,1])
                do_pdos_calc(eigdw,emin,emax,delta,v_kdw,nk1,nk2,nk3,nawf,1)
                eigdw = None
                v_kdw = None

        comm.Barrier()
        if rank ==0:
            print('dos in                           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
except Exception as e:
    print('Rank %d: Exception in DOS Calculation'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
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

        comm.Barrier()
        if rank ==0:
            print('FermiSurf in                     %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
except Exception as e:
    print('Rank %d: Exception in Fermi Surface'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    pksp = None
    jksp = None
    if Boltzmann or epsilon or Berry or spin_Hall or critical_points or smearing != None:
        if checkpoint < 3:
            #----------------------
            # Compute the gradient of the k-space Hamiltonian
            #----------------------
            from do_gradient import *
            dHksp = do_gradient(Hksp,a_vectors,alat,nthread,npool)
            #from do_gradient_d2 import *
            #dHksp,d2Hksp = do_gradient(Hksp,a_vectors,alat,nthread,npool,scipyfft)

            comm.Barrier()
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
        # Compute the momentum operator p_n,m(k) (and kinetic energy operator)
        #----------------------
        from do_momentum import *
        #from do_momentum_d2 import *

        if rank != 0:
            dHksp = None
            v_k = None
            pksp = None
            tksp = None
        if rank == 0:
            dHksp = np.reshape(dHksp,(nk1*nk2*nk3,3,nawf,nawf,nspin),order='C')
        pksp = do_momentum(v_k,dHksp,npool)
        #if rank == 0:
        #    d2Hksp = np.reshape(d2Hksp,(nk1*nk2*nk3,3,3,nawf,nawf,nspin),order='C')
        #pksp,tksp = do_momentum(v_k,dHksp,d2Hksp,npool)

        comm.Barrier()
        if rank == 0:
            print('momenta in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
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

    index = None
    if rank == 0:
        index = {'nawf':E_k.shape[1],'nktot':E_k.shape[0]}
    index = comm.bcast(index,root=0)
    nawf = index['nawf']
    nktot = index['nktot']

    kq_wght = np.ones((nktot),dtype=float)
    kq_wght /= float(nktot)
except Exception as e:
    print('Rank %d: Exception in Gradient or Momenta'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
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
                    if smearing == 'gauss':
                        afac = 0.7
                    elif smearing == 'm-p':
                        afac = 1.0
                    deltakp2[:,n,m,:] = afac*LAN.norm(np.real(np.absolute(pksp[:,:,n,n,:]-pksp[:,:,m,m,:])),axis=1)*dk

            if restart:
                np.savez(fpath+'PAOdelta'+str(nspin)+'.npz',deltakp=deltakp,deltakp2=deltakp2)

    comm.Barrier()
    if rank == 0 and smearing != None:
        print('adaptive smearing in             %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()
except Exception as e:
    print('Rank %d: Exception in Adaptive Smearing'%rank)
    print(e)
    comm.Abort()
    raise Exception

# to test formula 23 in the Graf & Vogl paper...
#if rank == 0:
#    from smearing import *
#    nawf = bnd
#    Ef = 0.0
#    effterm = np.zeros((nk1*nk2*nk3,nawf),dtype=complex)
#    for n in xrange(nawf):
#        for m in xrange(nawf):
#            if m != n:
#                fm = intgaussian(E_k[:,m,0],Ef,deltakp[:,m,0])
#                effterm[:,n] += 1.0/(E_k[:,m,0]-E_k[:,n,0] + 1.0j*deltakp2[:,n,m,0]) * \
#                                (pksp[:,ipol,n,m,0] * pksp[:,jpol,m,n,0] + pksp[:,jpol,n,m,0] * pksp[:,ipol,m,n,0])
#
#    f=open('Tnn'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
#    for n in xrange(nawf):
#        f.write('%.5f %9.5e %9.5e \n' \
#                %(n,np.sum(np.real(kq_wght[:]*tksp[:,ipol,jpol,n,n,0])), np.sum(np.real(kq_wght[:]*effterm[:,n]))))
#    f.close()
#
#    Tsum = 0.0
#    Psum = 0.0
#    for n in xrange(nawf):
#        fn = intgaussian(E_k[:,n,0],Ef,deltakp[:,n,0])
#        Tsum += np.sum(np.real(kq_wght[:]*tksp[:,ipol,jpol,n,n,0])*fn)
#        Psum += np.sum(np.real(kq_wght[:]*effterm[:,n])*fn)
#    print (Tsum, Psum)
#quit()

try:
    velkp = None
    if rank == 0:
        if Boltzmann or critical_points:
            #----------------------
            # Compute velocities for Boltzmann transport
            #----------------------
            velkp = np.zeros((nk1*nk2*nk3,3,bnd,nspin),dtype=float)
            for n in xrange(bnd):
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
except Exception as e:
    print('Rank %d: Exception computing velocities for Boltzmann Transport'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    if (do_dos or do_pdos) and smearing != None:
        #----------------------
        # DOS calculation with adaptive smearing on double_grid Hksp
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
    
        if nspin == 1 or nspin == 2:
            if rank == 0:
                eigup = np.array(eig[:,0])
                deltakpup = np.array(np.reshape(np.delete(deltakp,np.s_[bnd:],axis=1),(nk1*nk2*nk3*bnd,nspin),order='C')[:,0])
            do_dos_calc_adaptive(eigup,emin,emax,deltakpup,eigtot,bnd,0,smearing)
            eigup = None
            deltakpup = None
        if nspin == 2:
            if rank == 0:
                eigdw = np.array(eig[:,1])
                deltakpdw = np.array(np.reshape(np.delete(deltakp,np.s_[bnd:],axis=1),(nk1*nk2*nk3*bnd,nspin),order='C')[:,1])
            do_dos_calc_adaptive(eigdw,emin,emax,deltakpdw,eigtot,bnd,1,smearing)
            eigdw = None
            deltakpdw = None
    
        if rank == 0: deltakp = np.reshape(deltakp,(nk1*nk2*nk3,nawf,nspin),order='C')
    
        if do_pdos:
            v_kup = v_kdw = None
            #----------------------
            # PDOS calculation
            #----------------------
            from do_pdos_calc_adaptive import *

            if nspin == 1 or nspin == 2:
                if rank == 0:
                    eigup = np.array(E_k[:,:,0])
                    deltakpup = np.array(deltakp[:,:,0])
                    v_kup = np.array(v_k[:,:,:,0])
                do_pdos_calc_adaptive(eigup,emin,emax,deltakpup,v_kup,nk1,nk2,nk3,nawf,0,smearing)
                eigup = None
            if nspin == 2:
                if rank == 0:
                    eigdw = np.array(E_k[:,:,1])
                    deltakpdw = np.array(deltakp[:,:,1])
                    v_kdw = np.array(v_k[:,:,:,1])
                do_pdos_calc_adaptive(eigdw,emin,emax,deltakpdw,v_kdw,nk1,nk2,nk3,nawf,1,smearing)
                eigdw = None
 
        comm.Barrier()
        if rank ==0:
            print('dos (adaptive smearing) in       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
except Exception as e:
    print('Rank %d: Exception in DOS (Adaptive Smearing)'%rank)
    print(e)
    comm.Abort()
    raise Exception

#----------------------
# Memory reduction
#----------------------
# Reduce memory requirements and improve performance by reducing nawf to bnd (states with good projectability)
try:
    if rank == 0:
        pksp = np.delete(np.delete(pksp,np.s_[bnd:],axis=2),np.s_[bnd:],axis=3)
        E_k = np.delete(E_k,np.s_[bnd:],axis=1)
        deltakp = np.delete(deltakp,np.s_[bnd:],axis=1)
        deltakp2 = np.delete(np.delete(deltakp2,np.s_[bnd:],axis=1),np.s_[bnd:],axis=2)
except Exception as e:
    print('Rank %d: Exception in Memory Reduction'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Spin Hall calculation
    #----------------------
    if spin_Hall:
        if dftSO == False: sys.exit('full relativistic calculation with SO needed')

        from do_spin_Berry_curvature import *
        from do_spin_Hall_conductivity import *
        from do_spin_current import *

        for n in xrange(s_tensor.shape[0]):
            ipol = s_tensor[n][0]
            jpol = s_tensor[n][1]
            spol = s_tensor[n][2]
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
                jksp = np.delete(np.delete(do_spin_current(v_k,dHksp,spol,npool,do_spin_orbit,sh,nl),np.s_[bnd:],axis=2),np.s_[bnd:],axis=3)
                if restart and rank == 0:
                    np.savez(fpath+'PAOspin'+str(spol)+'.npz',jksp=jksp)
 
                comm.Barrier()
                if rank == 0:
                    print('spin current in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                    reset=time.time()
            #----------------------
            # Compute spin Berry curvature... 
            #----------------------
            Om_k = np.zeros((nk1,nk2,nk3,2),dtype=float)
            ene,shc,Om_k[:,:,:,0] = do_spin_Berry_curvature(E_k,jksp,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminSH,emaxSH,fermi_dw,fermi_up,deltakp,smearing)

            if rank == 0 and writedata:
                from write2bxsf import *
                x0 = np.zeros(3,dtype=float)
                ind_plot = np.zeros(2)
                Om_k[:,:,:,1] = Om_k[:,:,:,0]
                write2bxsf(fermi_dw,fermi_up,Om_k,nk1,nk2,nk3,2,ind_plot,0.0,alat,x0,b_vectors,'spin_Berry_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.bxsf')

            if ac_cond_spin:
                ene_ac,sigxy = do_spin_Hall_conductivity(E_k,jksp,pksp,temp,0,npool,ipol,jpol,shift,deltakp,deltakp2,smearing)
                shc0 = np.real(sigxy[0])

            omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

            if rank == 0:
                shc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                f=open('shcEf_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
                for n in xrange(ene.size):
                    f.write('%.5f %9.5e \n' %(ene[n],shc[n]))
                f.close()

            if rank == 0 and ac_cond_spin:
                sigxy *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                f=open('SCDi_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
                for n in xrange(ene.size):
                    f.write('%.5f %9.5e \n' %(ene_ac[n],np.imag(ene_ac[n]*sigxy[n]/105.4571)))  #convert energy in freq (1/hbar in cgs units)
                f.close()
                f=open('SCDr_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
                for n in xrange(ene.size):
                    f.write('%.5f %9.5e \n' %(ene_ac[n],np.real(sigxy[n])))
                f.close()

        comm.Barrier()
        if rank == 0:
            print('spin Hall module in              %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()

    dHksp = None
except Exception as e:
    print('Rank %d: Exception in Spin Hall Module'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Compute Berry curvature and AHC
    #----------------------
    if Berry:
        if dftSO == False: sys.exit('full relativistic calculation with SO needed')

        from do_Berry_curvature import *
        from do_Berry_conductivity import *

        for n in xrange(a_tensor.shape[0]):
            ipol = a_tensor[n][0]
            jpol = a_tensor[n][1]
            Om_k = np.zeros((nk1,nk2,nk3,2),dtype=float)
            ene,ahc,Om_k[:,:,:,0] = do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,npool,ipol,jpol,eminAH,emaxAH,fermi_dw,fermi_up,deltakp,smearing)

            if rank == 0 and writedata:
                from write2bxsf import *
                x0 = np.zeros(3,dtype=float)
                ind_plot = np.zeros(2)
                Om_k[:,:,:,1] = Om_k[:,:,:,0]
                write2bxsf(fermi_dw,fermi_up,Om_k,nk1,nk2,nk3,2,ind_plot,0.0,alat,x0,b_vectors,'Berry_'+str(LL[ipol])+str(LL[jpol])+'.bxsf')

                np.savez('Berry_'+str(LL[ipol])+str(LL[jpol])+'.npz',kq=kq,Om_k=Om_k[:,:,:,0])

            if ac_cond_Berry:
                ene_ac,sigxy = do_Berry_conductivity(E_k,pksp,temp,0,npool,ipol,jpol,shift,deltakp,deltakp2,smearing)
                ahc0 = np.real(sigxy[0])

            omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

            if rank == 0:
                ahc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                f=open('ahcEf_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
                for n in xrange(ene.size):
                    f.write('%.5f %9.5e \n' %(ene[n],ahc[n]))
                f.close()

            if rank == 0 and ac_cond_Berry:
                sigxy *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                f=open('MCDi_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
                for n in xrange(ene.size):
                    f.write('%.5f %9.5e \n' %(ene_ac[n],np.imag(ene_ac[n]*sigxy[n]/105.4571)))  #convert energy in freq (1/hbar in cgs units)
                f.close()
                f=open('MCDr_'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
                for n in xrange(ene.size):
                    f.write('%.5f %9.5e \n' %(ene_ac[n],np.real(sigxy[n])))
                f.close()

        comm.Barrier()
        if rank == 0:
            print('Berry module in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
except Exception as e:
    print('Rank %d: Exception in Berry Module'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Compute transport quantities (conductivity, Seebeck and thermal electrical conductivity)
    #----------------------
    if Boltzmann:
        from do_Boltz_tensors import *

        for ispin in xrange(nspin):

            if smearing == None:
                ene,L0,L1,L2 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin,deltakp,smearing,t_tensor)
            else:
                ene,L0 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin,deltakp,smearing,t_tensor)

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

            if smearing == None:
                #----------------------
                # Seebeck (in units of 1.e-4 V/K)
                #----------------------

                S = np.zeros((3,3,ene.size),dtype=float)

                L0 *= 1.0e21
                L1 *= (ELECTRONVOLT_SI**2/(4.0*np.pi**3))*(ELECTRONVOLT_SI**2/(H_OVER_TPI**2*BOHR_RADIUS_SI))

                if rank == 0:
                    for n in xrange(ene.size):
                        try:
                            S[:,:,n] = LAN.inv(L0[:,:,n])*L1[:,:,n]*(-K_BOLTZMAN_SI/(temp*ELECTRONVOLT_SI**2))*1.e4
                        except:
                            sys.exit('check t_tensor components - matrix cannot be singular')

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

        comm.Barrier()
        if rank ==0:
            print('transport in                     %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
except Exception as e:
    print('Rank %d: Exception in Transport'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    #----------------------
    # Compute dielectric tensor (Re and Im epsilon)
    #----------------------
    if epsilon:
        from do_epsilon import *
        #from do_epsilon_d2 import *

        omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

        for n in xrange(d_tensor.shape[0]):
            ipol = d_tensor[n][0]
            jpol = d_tensor[n][1]
            for ispin in xrange(nspin):

                ene, epsi, epsr, jdos = do_epsilon(E_k,pksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,epsmin,epsmax,deltakp,deltakp2,smearing,kramerskronig)
                #ene, epsi, epsr, jdos = do_epsilon(E_k,pksp,tksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,epsmin,epsmax,deltakp,deltakp2,smearing,kramerskronig)

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
                    f=open('jdos_'+str(ispin)+'.dat','w')
                    for n in xrange(ene.size):
                        f.write('%.5f %9.5e \n' \
                                %(ene[n],jdos[n]))
                    f.close()

        comm.Barrier()
        if rank ==0:
            print('epsilon in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
except Exception as e:
    print('Rank %d: Exception in Epsilon'%rank)
    print(e)
    comm.Abort()
    raise Exception

try:
    # Timing
    if rank ==0:
        print('   ')
        print('Total CPU time =                 %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
    quit()
except Exception as e:
    print('Rank %d: Exception in Total Time'%rank)
    print(e)
    comm.Abort()
    raise Exception
