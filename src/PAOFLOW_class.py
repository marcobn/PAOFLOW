import os
import sys
import time
import traceback
import numpy as np
from mpi4py import MPI
from scipy import fftpack as FFT


sys.path.append(sys.path[0]+'/defs')
from DataController import DataController
from report_exception import report_exception


class PAOFLOW:
    comm = rank = size = None

    start_time = reset_time = None

    inputpath = inputfile = None

    data_controller = None

    def __init__ ( self,  inputpath='./', inputfile='inputfile.xml', verbose=False ):

        self.inputpath = inputpath
        self.inputfile = inputfile

        #-------------------------------
        # Initialize Parallel Execution
        #-------------------------------
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        #----------------------
        # Initialize Time
        #----------------------
        if self.rank == 0:
            self.start_time = time.time()

        #----------------------
        # Print Header
        #----------------------
        if self.rank == 0:
            print('')
            print('#############################################################################################')
            print('#                                                                                           #')
            print('#                                          PAOFLOW                                          #')
            print('#                                                                                           #')
            print('#                  Utility to construct and operate on Hamiltonians from                    #')
            print('#                 the Projections of DFT wfc on Atomic Orbital bases (PAO)                  #')
            print('#                                                                                           #')
            print('#                       (c)2016-2018 ERMES group (http://ermes.unt.edu)                     #')
            print('#############################################################################################\n')

        # Initialize Data Controller 
        self.data_controller = DataController(inputpath, inputfile)
        try:
            self.data_controller.read_external_files()
        except Exception as e:
            report_exception()
            self.comm.Abort()
        
        attributes = self.data_controller.data_attributes

        # Update Path
        attributes['fpath'] = os.path.join(inputpath, attributes['fpath'])

        # Update Input Arguments
        attributes['verbose'] = verbose

### MOVE TO DATA_CONTROLLER
        #------------------------------
        # Initialize Return Dictionary
        #------------------------------
        #if self.rank == 0 and attributes['out_vals'] is not None:
        #    outDict = {}
        #    if len(attributes['out_vals']) > 0:
        #        for i in attributes['out_vals']:
        #            outDict[i] = None

        #------------------------------
        # Check for CUDA FFT Libraries
        #-----------------------------
        attributes['scipyfft'] = True
        if attributes['use_cuda']:
            attributes['scipyfft'] = False

        # Ensure nffts are even
        if attributes['double_grid']:
            if attributes['nfft1']%2!=0 or attributes['nfft2']%2!=0 or attributes['nfft3']%2!=0:
                attributes['nfft1'] = 2*((attributes['nfft1']+1)//2)
                attributes['nfft2'] = 2*((attributes['nfft2']+1)//2)
                attributes['nfft3'] = 2*((attributes['nfft3']+1)//2)
                print('Warning: nfft grid has been modified to support double_grid,')
                print('Modified nfft grid to: %d %d %d\n'%(attributes['nfft1'],attributes['nfft2'],attributes['nfft3']))

        # set npool to minimum needed if npool isnt high enough
        int_max = 2147483647
        temp_pool = int(np.ceil((float(attributes['nawf']**2*attributes['nfft1']*attributes['nfft2']*attributes['nfft3']*3*attributes['nspin'])/float(int_max))))
        if temp_pool > attributes['npool']:
            if self.rank == 0:
                print("Warning: %s too low. Setting npool to %s"%(attributes['npool'],temp_pool))
            attributes['npool'] = temp_pool

        if self.rank == 0:
            if self.size >  1:
                print('Parallel execution on %d processors and %d pool'%(self.size,attributes['npool']) + ('' if attributes['npool']==1 else 's'))
            else:
                print('Serial execution')

        #----------------------
        # Do memory checks 
        #----------------------
        if self.rank == 0:
            gbyte = 3*3*2*16./1.E9*attributes['nawf']**2*attributes['nfft1']*attributes['nfft2']*attributes['nfft3']
            print('Estimated maximum array size: %5.2f GBytes' %(gbyte))
            print('   ')

        self.comm.Barrier()
        if self.rank == 0:
            print('reading in                       %5s sec ' %str('%.3f' %(time.time()-self.start_time)).rjust(10))
            self.reset_time = time.time()


    def calc_projectability ( self, pthr=None ):
        from do_projectability import do_projectability
        if pthr is not None:
            self.data_controller.data_attributes['pthr'] = pthr
        do_projectability(self.data_controller)


    def calc_pao_hamiltonian ( self ):
        from do_build_pao_hamiltonian import do_build_pao_hamiltonian

        if self.rank == 0:
            do_build_pao_hamiltonian(self.data_controller)

        # U not needed anymore
        del self.data_controller.data_arrays['U']

        self.comm.Barrier()
        if self.rank == 0:
            print('building Hks in                  %5s sec ' %str('%.3f' %(time.time()-self.reset_time)).rjust(10))
            self.reset_time = time.time()


    def orthogonalize_hamiltonian ( self ):
        from scipy import fftpack as FFT

        arrays = self.data_controller.data_arrays
        attributes = self.data_controller.data_attributes

        nawf,_,nk1,nk2,nk3,nspin = arrays['HRs'].shape

        if attributes['use_cuda']:
            from cuda_fft import cuda_fftn, cuda_ifftn
            arrays['Hks'] = cuda_fftn(np.moveaxis(arrays['HRs'],[0,1],[3,4]), axes=[0,1,2])
            arrays['Sks'] = cuda_fftn(np.moveaxis(arrays['SRs'],[0,1],[3,4]), axes=[0,1,2])
            arrays['Hks'] = np.reshape(np.moveaxis(arrays['Hks'],[3,4],[0,1]), (nawf,nawf,nktot,nspin), order='C')
            arrays['Sks'] = np.reshape(np.moveaxis(arrays['Sks'],[3,4],[0,1]), (nawf,nawf,nktot), order='C')
        else:
            arrays['Hks'] = FFT.fftn(arrays['HRs'],axes=[2,3,4])
            arrays['Sks'] = FFT.fftn(arrays['SRs'], axes=[2,3,4])
            arrays['Hks'] = np.reshape(arrays['Hks'], (nawf,nawf,nktot,nspin), order='C')
            arrays['Sks'] = np.reshape(arrays['Sks'], (nawf,nawf,nktot), order='C')

        arrays['Hks'] = do_ortho(arrays['Hks'], arrays['Sks'])
        arrays['Hks'] = np.reshape(arrays['Hks'], (nawf,nawf,nk1,nk2,nk3,nspin), order='C')
        arrays['Sks'] = np.reshape(arrays['Sks'], (nawf,nawf,nk1,nk2,nk3), order='C')
        if attributes['use_cuda']:
            arrays['HRs'] = np.moveaxis(cuda_ifftn(np.moveaxis(arrays['Hks'],[0,1],[3,4]), axes=[0,1,2]),[3,4],[0,1])
        else:
            arrays['HRs'] = FFT.ifftn(arrays['Hks'], axes=[2,3,4])

        del arrays['Sks']
        del arrays['SRs']


    def calc_k_to_R ( self ):
        from get_K_grid_fft import get_K_grid_fft

        arrays = self.data_controller.data_arrays
        attributes = self.data_controller.data_attributes

        #----------------------
        # Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)
        #----------------------
        if self.rank == 0:
            # Original k grid to R grid
            arrays['HRs'] = np.zeros_like(arrays['Hks'])
            arrays['HRs'] = FFT.ifftn(arrays['Hks'], axes=[2,3,4])

            if attributes['non_ortho']:
                arrays['SRs'] = np.zeros_like(arrays['Sks'])
                arrays['SRs'] = FFT.ifftn(arrays['Sks'], axes=[2,3,4])
                del arrays['Sks']

        # Save Hks if the interpolated Hamiltonian will not be computed.
        if not attributes['double_grid']:
            from communication import scatter_full
            if self.rank == 0:
                nawf,_,nk1,nk2,nk3,nspin = arrays['Hks'].shape
                arrays['Hks'] = np.reshape(arrays['Hks'], (nawf**2,nk1,nk2,nk3,nspin), order='C')
            else:
                arrays['Hks'] = None
            arrays['Hksp'] = scatter_full(arrays['Hks'], attributes['npool'])
            del arrays['Hks']

        get_K_grid_fft(self.data_controller)

        self.comm.Barrier()
        if self.rank == 0:
            print('k -> R in                        %5s sec ' %str('%.3f' %(time.time()-self.reset_time)).rjust(10))
            self.reset_time = time.time()


    def add_external_fields ( self ):
        from add_ext_field import add_ext_field

        arrays = self.data_controller.data_arrays

        # Add external fields or non scf ACBN0 correction
        if self.rank == 0 and (arrays['Efield'].any() != 0.0 or arrays['Bfield'].any() != 0.0 or arrays['HubbardU'].any() != 0.0):
            add_ext_field(self.data_controller)
        self.comm.Barrier()



# ----- Z2 Pack ----- #



    def calc_bands ( self ):
        from do_bands import do_bands

        arrays = self.data_controller.data_arrays
        attributes = self.data_controller.data_attributes

        #----------------------
        # Compute bands with spin-orbit coupling
        #----------------------
        if self.rank == 0 and attributes['do_spin_orbit']:
            from do_spin_orbit import do_spin_orbit_bands

            socStrengh = np.zeros((attributes['natoms'],2), dtype=float)
            socStrengh [:,0] =  arrays['lambda_p'][:]
            socStrengh [:,1] =  arrays['lambda_d'][:]

            do_spin_orbit_bands(self.data_controller)

        do_bands(self.data_controller)

        self.comm.Barrier()
        if self.rank == 0:
            print('bands in                         %5s sec ' %str('%.3f' %(time.time()-self.reset_time)).rjust(10))
            self.reset_time = time.time()
    
        if not attributes['onedim'] and attributes['band_topology']:
            from do_topology_calc import do_topology_calc
            # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry
            # curvature operators along the path in the IBZ from do_topology_calc 

            attributes['eff_mass'] = True
            do_topology_calc(self.data_controller)
    
            self.comm.Barrier()
            if self.rank == 0:
                print('band topology in                 %5s sec ' %str('%.3f' %(time.time()-self.reset_time)).rjust(10))
                self.reset_time = time.time()

        del arrays['R']
        del arrays['idx']
        del arrays['Rfft']
        del arrays['R_wght']
#        self.data_controller.clean_data()


    def double_grid ( self ):
        from get_K_grid_fft import get_K_grid_fft
        from do_double_grid import do_double_grid

        arrays = self.data_controller.data_arrays
        attributes = self.data_controller.data_attributes

        #------------------------------------------------------
        # Fourier interpolation on extended grid (zero padding)
        #------------------------------------------------------
        do_double_grid(self.data_controller)

        nk1 = attributes['nk1']
        nk2 = attributes['nk2']
        nk3 = attributes['nk3']

        get_K_grid_fft(self.data_controller)

        if self.rank == 0:
            if attributes['verbose']:
                print('Grid of k vectors for zero padding Fourier interpolation ', nk1, nk2, nk3)
            print('R -> k zero padding in           %5s sec ' %str('%.3f' %(time.time()-self.reset_time)).rjust(10))
            self.reset_time = time.time()

        del arrays['HRs']

        if self.rank == 0:
            self.data_controller.clean_data()


        #do_double_grid(self.data_controller)
