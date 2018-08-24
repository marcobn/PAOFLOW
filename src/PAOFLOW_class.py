import os
import sys
import numpy as np
from time import time
from scipy import fftpack as FFT


sys.path.append(sys.path[0]+'/defs')
from DataController import DataController
from report_exception import report_exception


class PAOFLOW:

  data_controller = None

  comm = rank = size = None

  inputpath = inputfile = None

  start_time = reset_time = None


  def __init__ ( self, inputpath='./', inputfile='inputfile.xml', verbose=False ):
    from mpi4py import MPI

    self.inputpath = inputpath
    self.inputfile = inputfile

    #-------------------------------
    # Initialize Parallel Execution
    #-------------------------------
    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

    #-----------------
    # Initialize Time
    #-----------------
    if self.rank == 0:
      self.start_time = self.reset_time = time()

    #--------------
    # Print Header
    #--------------
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
    #  outDict = {}
    #  if len(attributes['out_vals']) > 0:
    #    for i in attributes['out_vals']:
    #      outDict[i] = None

    #------------------------------
    # Check for CUDA FFT Libraries
    #------------------------------
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
      if self.size > 1:
        print('Parallel execution on %d processors and %d pool'%(self.size,attributes['npool']) + ('' if attributes['npool']==1 else 's'))
      else:
        print('Serial execution')

    #------------------
    # Do memory checks 
    #------------------
    if self.rank == 0:
      dxdydz = 3
      B_to_GB = 1.E-9
      fudge_factor = 1.
      bytes_per_complex = 128//8
      spins = attributes['nspin']
      num_wave_functions = attributes['nawf']
      nd1 = (attributes['nfft1'] if attributes['nfft1']>attributes['nk1'] else attributes['nk1'])
      nd2 = (attributes['nfft2'] if attributes['nfft2']>attributes['nk2'] else attributes['nk2'])
      nd3 = (attributes['nfft3'] if attributes['nfft3']>attributes['nk3'] else attributes['nk3'])
      gbyte = num_wave_functions**2 * (nd1*nd2*nd3) * spins * dxdydz * bytes_per_complex * fudge_factor * B_to_GB
      print('Estimated maximum array size: %5.2f GBytes\n' %(gbyte))

    self.report_module_time('Reading in')


  def report_module_time ( self, mname ):

    spaces = 40
    lmn = len(mname)
    if len(mname) > spaces:
      print('Please use a shorter module tag.')
      quit()

    self.comm.Barrier()
    if self.rank == 0:
      lms = spaces-lmn
      dt = time() - self.reset_time
      print('%s: %s %.3f sec'%(mname,lms*' ',dt))
      self.reset_time = time()
    self.comm.Barrier()


  def calc_projectability ( self, pthr=None ):
    from do_projectability import do_projectability
    if pthr is not None:
      self.data_controller.data_attributes['pthr'] = pthr
    do_projectability(self.data_controller)



  def calc_pao_hamiltonian ( self ):
    from do_build_pao_hamiltonian import do_build_pao_hamiltonian
    from get_K_grid_fft import get_K_grid_fft

    arrays = self.data_controller.data_arrays
    attributes = self.data_controller.data_attributes

    if self.rank == 0:
      do_build_pao_hamiltonian(self.data_controller)

    # U not needed anymore
    del self.data_controller.data_arrays['U']

    self.report_module_time('Building Hks in')

    #----------------------------------------------------------
    # Define the Hamiltonian and overlap matrix in real space:
    #   HRs and SRs (noinv and nosym = True in pw.x)
    #----------------------------------------------------------
    if self.rank == 0:
      # Original k grid to R grid
      arrays['HRs'] = np.zeros_like(arrays['Hks'])
      arrays['HRs'] = FFT.ifftn(arrays['Hks'], axes=[2,3,4])

      if attributes['non_ortho']:
        arrays['SRs'] = np.zeros_like(arrays['Sks'])
        arrays['SRs'] = FFT.ifftn(arrays['Sks'], axes=[2,3,4])
        del arrays['Sks']

#### PARALLELIZATION
    #### MUST KNOW DOUBLE_GRID
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

    self.data_controller.broadcast_single_array('HRs')
    if attributes['non_ortho']:
      self.data_controller.broadcast_single_array('SRs')

    self.report_module_time('k -> R in')



  def orthogonalize_hamiltonian ( self ):

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

    self.data_controller.broadcast_single_array('HRs')

    del arrays['Sks']
    del arrays['SRs']



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

    #----------------------------------------
    # Compute bands with spin-orbit coupling
    #----------------------------------------
    if self.rank == 0 and attributes['do_spin_orbit']:
      from do_spin_orbit import do_spin_orbit_bands

      socStrengh = np.zeros((attributes['natoms'],2), dtype=float)
      socStrengh [:,0] = arrays['lambda_p'][:]
      socStrengh [:,1] = arrays['lambda_d'][:]

      do_spin_orbit_bands(self.data_controller)

    do_bands(self.data_controller)

    self.report_module_time('Bands in')

    if not attributes['onedim'] and attributes['band_topology']:
      from do_topology_calc import do_topology_calc
      # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry
      # curvature operators along the path in the IBZ from do_topology_calc 

      attributes['eff_mass'] = True
      do_topology_calc(self.data_controller)
  
      self.report_module_time('Band Topology in')

    del arrays['R']
    del arrays['idx']
    del arrays['Rfft']
    del arrays['R_wght']
#    self.data_controller.clean_data()



  def calc_double_grid ( self ):
    from get_K_grid_fft import get_K_grid_fft
    from do_double_grid import do_double_grid

    arrays = self.data_controller.data_arrays
    attributes = self.data_controller.data_attributes

    #------------------------------------------------------
    # Fourier interpolation on extended grid (zero padding)
    #------------------------------------------------------
    do_double_grid(self.data_controller)

    get_K_grid_fft(self.data_controller)

    if self.rank == 0 and attributes['verbose']:
      nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']
      print('Grid of k vectors for zero padding Fourier interpolation ', nk1, nk2, nk3)

    self.report_module_time('R -> k with Zero Padding in')

    del arrays['HRs']



  def calc_pao_eigh ( self ):
    from communication import scatter_full, gather_full
    from do_pao_eigh import do_pao_eigh

    arrays = self.data_controller.data_arrays
    attributes = self.data_controller.data_attributes

    #-----------------------------------------------------
    # Compute eigenvalues of the interpolated Hamiltonian
    #-----------------------------------------------------

    do_pao_eigh(self.data_controller)

### PARALLELIZATION
    ## Parallelize search for amax & subtract for all processes. Test time.
    if arrays['HubbardU'].any() != 0.0:
      if self.rank == 0 and attributes['verbose']:
        print('Shifting Eigenvalues to top of valence band.')
      arrays['E_k'] = gather_full(arrays['E_k'], attributes['npool'])
      if self.rank == 0:
        arrays['E_k'] -= np.amax(arrays['E_k'][:,attributes['bval'],:])
      self.comm.Barrier()
      arrays['E_k'] = scatter_full(arrays['E_k'], attributes['npool'])

    self.report_module_time('Eigenvalues in')



  def calc_gradient_and_momenta ( self ):
    from do_gradient import do_gradient
    from do_momentum import do_momentum
    from communication import gather_scatter

    arrays = self.data_controller.data_arrays
    attributes = self.data_controller.data_attributes

    snktot,nawf,_,nspin = arrays['Hksp'].shape
    arrays['Hksp'] = np.reshape(arrays['Hksp'], (snktot, nawf**2, nspin))
    arrays['Hksp'] = gather_scatter(arrays['Hksp'], 1, attributes['npool'])
    arrays['Hksp'] = np.moveaxis(arrays['Hksp'], 0, 1)
    snawf,_,nspin = arrays['Hksp'].shape
    arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf,attributes['nk1'],attributes['nk2'],attributes['nk3'],nspin))

    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------            
    do_gradient(self.data_controller)

########### PARALLELIZATION
    #gather dHksp on nawf*nawf and scatter on k points
    ################DISTRIBUTE ARRAYS ON KPOINTS#################
    #############################################################
    arrays['dHksp'] = np.reshape(arrays['dHksp'], (snawf,attributes['nkpnts'],3,nspin))
    arrays['dHksp'] = gather_scatter(arrays['dHksp'], 1, attributes['npool'])
    arrays['dHksp'] = np.moveaxis(arrays['dHksp'], 0, 2)
    arrays['dHksp'] = np.reshape(arrays['dHksp'], (snktot,3,nawf,nawf,nspin), order="C")

    self.report_module_time('Gradient in')

    #----------------------------------------------------------------------
    # Compute the momentum operator p_n,m(k) (and kinetic energy operator)
    #----------------------------------------------------------------------
    do_momentum(self.data_controller)

## NEED FOR A BETTER CLEANING METHODOLOGY! AGH!!!
#    if not attributes['spin_Hall']:
#      del arrays['dHksp']

    self.report_module_time('Momenta in')



  def calc_adaptive_smearing ( self, smearing=None ):
    from do_adaptive_smearing import do_adaptive_smearing

    attributes = self.data_controller.data_attributes

    if smearing != None and (smearing == 'gauss' or smearing == 'm-p'):
      attributes['smearing'] = smearing

    do_adaptive_smearing(self.data_controller)

    self.report_module_time('Adaptive Smearing in')



  def calc_dos ( self ):
    pass



  def trim_non_projectable_bands ( self ):

    arrays,attributes = self.data_controller.data_dicts()

    arrays['E_k'] = arrays['E_k'][:,:bnd]
    arrays['pksp'] = arrays['pksp'][:,:,:bnd,:bnd]
    if 'deltakp' in arrays:
      arrays['deltakp'] = arrays['deltakp'][:,:bnd]
      arrays['deltakp2'] = arrays['deltakp2'][:,:bnd]



  def calc_dos_adaptive ( self, smearing='gauss', do_dos=True, do_pdos=True ):
    from do_dos_calc_adaptive import do_dos_calc_adaptive
    from do_pdos_calc_adaptive import do_pdos_calc_adaptive

    attributes = self.data_controller.data_attributes

    if smearing != attributes['smearing']:
      attributes['smearing'] = smearing

    #------------------------------------------------------------
    # DOS calculation with adaptive smearing on double_grid Hksp
    #------------------------------------------------------------
    if do_dos:
      do_dos_calc_adaptive(self.data_controller)

    #----------------------
    # PDOS calculation ...
    #----------------------
    if do_pdos:
      do_pdos_calc_adaptive(self.data_controller)

    self.report_module_time('DoS (Adaptive Smearing) in')



  def calc_fermi_surface ( self ):
    from do_fermisurf import do_fermisurf
    #----------------------
    # Fermi surface calculation
    #----------------------
    do_fermisurf(self.data_controller)

    self.report_module_time('Fermi Surface in')



  def calc_spin_texture ( self ):
    from do_spin_texture import do_spin_texture

    attributes = self.data_controller.data_attributes

    if attributes['nspin'] == 1:
      do_spin_texture(self.data_controller)
      #do_spin_texture(fermi_dw,fermi_up,E_k,v_k,sh,nl,nk1,nk2,nk3,nawf,nspin,do_spin_orbit,npool,inputpath)

      self.report_module_time('Spin Texutre in')

    else:
      if self.rank == 0:
        print('Cannot compute spin texture with nspin=2')
        self.comm.Abort()
      self.comm.Barrier()



  def calc_spin_Hall ( self, shc=True, ahc=True ):
    from do_spin_current import do_spin_current
    from do_spin_Berry_curvature import do_spin_Berry_curvature
    from constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL

    arrays = self.data_controller.data_arrays
    attributes = self.data_controller.data_attributes

    s_tensor = arrays['s_tensor']

    #----------------------
    # Spin Hall calculation
    #----------------------
    if attributes['dftSO'] == False:
      if self.rank == 0:
        print('Relativistic calculation with SO required')
        self.comm.Abort()
      self.comm.Barrier()

    for n in range(s_tensor.shape[0]):
      ipol = s_tensor[n][0]
      jpol = s_tensor[n][1]
      spol = s_tensor[n][2]
      #----------------------
      # Compute the spin current operator j^l_n,m(k)
      #----------------------
      jksp = do_spin_current(self.data_controller, spol)

      #----------------------
      # Compute spin Berry curvature... 
      #----------------------

      ene,shc,Om_k = do_spin_Berry_curvature(self.data_controller, jksp, ipol, jpol)

      if self.rank == 0:
        shc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/attributes['omega']
      fshc = 'shcEf_%s_%s%s.dat'%(str(LL[spol]),str(LL[ipol]),str(LL[jpol]))
      self.data_controller.write_file_row_col(fshc, ene, shc)

    self.report_module_time('Spin Hall and Anomalous Hall in')
#if rank == 0:
#shc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
#f=open(os.path.join(inputpath,'shcEf_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
#for n in range(ene.size):
#f.write('%.5f %9.5e \n' %(ene[n],shc[n]))
#f.close()

#if  ac_cond_spin:
#sigxy *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
#f=open(os.path.join(inputpath,'SCDi_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
#for n in range(ene.size):
#f.write('%.5f %9.5e \n' %(ene_ac[n],np.imag(ene_ac[n]*sigxy[n]/105.4571)))  #convert energy in freq (1/hbar in cgs units)
#f.close()
#f=open(os.path.join(inputpath,'SCDr_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
#for n in range(ene.size):
#f.write('%.5f %9.5e \n' %(ene_ac[n],np.real(sigxy[n])))
#f.close()

