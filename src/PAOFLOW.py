#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np

class PAOFLOW:

  data_controller = None

  comm = rank = size = None

  workpath = inputfile = None

  start_time = reset_time = None

  # Overestimate factor for guessing memory requirements
  gb_fudge_factor = 4.

  # Function container for ErrorHandler's method
  report_exception = None


  def print_data_keys ( self ):
    '''
      Print's out the keys do the data_controller dictionaries, "arrays" and "attributes".
      Each stores the respective data required for the various calculation PAOFLOW can perform.

      Arguments:
          None

      Returns:
          None
    '''
    if self.rank == 0:
      self.data_controller.print_data()
    self.comm.Barrier()



  def __init__ ( self, workpath='./', outputdir='output', inputfile=None, savedir=None, smearing='gauss', npool=1, verbose=False ):
    '''
    Initialize the PAOFLOW class, either with a save directory with required QE output or with an xml inputfile

    Arguments:
        workpath (str): Path to the working directory
        outputdir (str): Name of the output directory (created in the working directory path)
        inputfile (str): (optional) Name of the xml inputfile
        savedir (str): QE .save directory
        smearing (str): Smearing type (None, m-p, gauss)
        npool (int): The number of pools to use. Increasing npool may reduce memory requirements.
        verbose (bool): False supresses debugging output

    Returns:
        None
    '''
    from time import time
    from mpi4py import MPI
    from .defs.header import header
    from .DataController import DataController

    self.workpath = workpath
    self.outputdir = outputdir
    self.inputfile = inputfile

    #-------------------------------
    # Initialize Parallel Execution
    #-------------------------------
    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

    #-----------------
    # Print Header
    # Initialize Time
    #--------------
    if self.rank == 0:
      header()
      self.start_time = self.reset_time = time()

    # Initialize Data Controller
    self.data_controller = DataController(workpath, outputdir, inputfile, savedir, smearing, npool, verbose)

    # Data Attributes
    attr = self.data_controller.data_attributes
    self.report_exception = self.data_controller.report_exception

    # Check for CUDA FFT Libraries
## CUDA not yet supported in PAOFLOW_CLASS
    attr['use_cuda'] = False
    attr['scipyfft'] = True
    if attr['use_cuda']:
      attr['scipyfft'] = False
    if self.rank == 0 and attr['verbose']:
      if attr['use_cuda']:
        print('CUDA will perform FFTs on %d GPUs'%1)
      else:
        print('SciPy will perform FFTs')

    # Report execution information
    if self.rank == 0:
      if self.size == 1:
        print('Serial execution')
      else:
        print('Parallel execution on %d processors and %d pool'%(self.size,attr['npool']) + ('' if attr['npool']==1 else 's'))

    # Do memory checks
    if self.rank == 0:
      gbyte = self.memory_check()
      print('Estimated maximum array size: %.2f GBytes\n' %(gbyte))

    self.report_module_time('Initialization')



  def report_module_time ( self, mname ):
    from time import time

    self.comm.Barrier()

    if self.rank == 0:

      # White spacing between module name and reported time
      spaces = 40
      lmn = len(mname)
      if len(mname) > spaces:
        print('DEBUG: Please use a shorter module tag.')
        self.comm.Abort()

      # Format string and print
      lms = spaces-lmn
      dt = time() - self.reset_time
      print('%s in: %s %.3f sec'%(mname,lms*' ',dt))
      self.reset_time = time()



  def memory_check ( self ):
    '''
    Estimate PAOFLOW's memory requirements with a "fudge factor" defined above

    Arguments:
        None

    Returns:
        gbyte (float): Estimated number of Gigabytes required in memory for PAOFLOW to run.
    '''
    dxdydz = 3
    B_to_GB = 1.E-9
    ff = self.gb_fudge_factor
    bytes_per_complex = 128//8
    arry,attr = self.data_controller.data_dicts()
    nd1,nd2,nd3 = attr['nk1'],attr['nk2'],attr['nk3']
    spins,num_wave_functions = attr['nspin'],attr['nawf']
    return num_wave_functions**2 * (nd1*nd2*nd3) * spins * dxdydz * bytes_per_complex * ff * B_to_GB



  def finish_execution ( self ):
    '''
    Finish the PAOFLOW execution. Print out run time and maximum memory usage

    Arguments:
        None

    Returns:
        None
    '''
    import resource
    from time import time
    from mpi4py import MPI

    if self.rank == 0:
      tt = time() - self.start_time
      print('Total CPU time =%s%.3f sec'%(27*' ',tt))

    verbose = self.data_controller.data_attributes['verbose']

    if verbose:

      # Add up memory usage from each core
      mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      mem = np.array([mem], dtype=float)
      mem0 = np.zeros(1, dtype=float) if self.rank==0 else None
      self.comm.Reduce(mem, mem0, op=MPI.SUM, root=0)

      if self.rank == 0:
        print("Memory usage on rank 0:  %6.4f GB"%(mem[0]/1024.0**2))
        print("Maximum concurrent memory usage:  %6.4f GB"%(mem0[0]/1024.0**2))

    MPI.Finalize()
    quit()



  def projectability ( self, pthr=0.95, shift='auto' ):
    '''
    Calculate the Projectability Matrix to determine how many states need to be shifted

    Arguments:
        pthr (float): The minimum allowed projectability for high projectability states
        shift (str or float): If 'auto' the shift will be automatic. Otherwise the shift will be the value of 'shift'

    Returns:
        None
    '''
    from .defs.do_projectability import do_projectability

    attr = self.data_controller.data_attributes

    if 'pthr' not in attr: attr['pthr'] = pthr
    if 'shift' not in attr: attr['shift'] = shift

    try:
      do_projectability(self.data_controller)
    except:
      self.report_exception('projectability')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Projectability')



  def pao_hamiltonian ( self, non_ortho=False, shift_type=1,expand_wedge=True,symmetrize=False,thresh=1.e-6,max_iter=16 ):
    '''
    Construct the Tight Binding Hamiltonian
    Yields 'HRs', 'Hks' and 'kq_wght'

    Arguments:
        non_ortho (bool): If True the Hamiltonian will be Orthogonalized after construction
        shift_type (int): Shift type [ 0-(PRB 2016), 1-(PRB 2013), 2-No Shift ] 

    Returns:
        None
    
    '''
    from .defs.get_K_grid_fft import get_K_grid_fft
    from .defs.do_build_pao_hamiltonian import do_build_pao_hamiltonian,do_Hks_to_HRs

    # Data Attributes and Arrays
    arrays,attr = self.data_controller.data_dicts()

    if 'non_ortho' not in attr: attr['non_ortho'] = non_ortho
    if 'shift_type' not in attr: attr['shift_type'] = shift_type

    attr['symm_thresh']=thresh
    attr['symm_max_iter']=max_iter
    attr['symmetrize']=symmetrize
    attr['expand_wedge']=expand_wedge

    try:
      do_build_pao_hamiltonian(self.data_controller)
    except:
      self.report_exception('pao_hamiltonian')
      if attr['abort_on_exception']:
        self.comm.Abort()
    self.report_module_time('Building Hks')

    # Done with U
    del arrays['U']

    try:
      do_Hks_to_HRs(self.data_controller)

      ### PARALLELIZATION
      self.data_controller.broadcast_single_array('HRs')
      if attr['non_ortho']:
        self.data_controller.broadcast_single_array('SRs')

      get_K_grid_fft(self.data_controller)
    except:
      self.report_exception('pao_hamiltonian')
      if attr['abort_on_exception']:
        self.comm.Abort()
    self.report_module_time('k -> R')

    # Orthogonalization
    if non_ortho:
      try:
        from .defs.do_ortho import do_orthogonalize

        do_orthogonalize(self.data_controller)
      except:
        self.report_exception('pao_hamiltonian')
        if attr['abort_on_exception']:
          self.comm.Abort()
      self.report_module_time('Orthogonalize')



  def add_external_fields ( self, Efield=[0.], Bfield=[0.], HubbardU=[0.] ):
    '''
    Add External Fields and Corrections to the Hamiltonian 'HRs'

    Arguments:
        Efield (ndarray): 1D array of Efield values
        Bfield (ndarray): 1D array of Bfield values
        HubbardU (ndarray): 1D array of Hubbard Correctional values

    Returns:
        None
    
    '''
    arry,attr = self.data_controller.data_dicts()

    if any(v != 0. for v in Efield): arry['Efield'] = np.array(Efield)
    if any(v != 0. for v in Bfield): arry['Bfield'] = np.array(Bfield)
    if any(v != 0. for v in HubbardU): arry['HubbardU'] = np.array(HubbardU)

    Efield,Bfield,HubbardU = arry['Efield'],arry['Bfield'],arry['HubbardU']

    try:
      # Add external fields or non scf ACBN0 correction
      if Efield.any() != 0. or Bfield.any() != 0. or HubbardU.any() != 0.:
        from .defs.add_ext_field import add_ext_field
        add_ext_field(self.data_controller)
        if self.rank == 0 and attr['verbose']:
          print('External Fields Added')
    except:
      self.report_exception('add_external_fields')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.comm.Barrier()
    

  def z2_pack ( self, fname='z2pack_hamiltonian.hm' ):
    '''
    Write 'HRs' to file for use with Z2 Pack

    Arguments:
        fname (str): File name for the Hamiltonian in the Z2 Pack format

    Returns:
        None

    '''
    try:
      self.data_controller.write_z2pack(fname)
    except:
      self.report_exception('z2_pack')
      if self.data_controller.data_attributes['abort_on_exception']:
        self.comm.Abort()


  def bands ( self, ibrav=None, band_path=None, high_sym_points=None, spin_orbit=False, fname='bands', nk=500 , theta=0., phi=0., lambda_p=[0.], lambda_d=[0.], orb_pseudo=['s'] ):
    '''
    Compute the electronic band structure

    Arguments:
        ibrav (int): Crystal structure (following the specifications of QE)
        band_path (str): A string representing the band path to follow
        high_sym_points (dictionary): A dictionary with symbols of high symmetry points as keys and length 3 numpy arrays containg the location of the symmetry points as values.
        spin_orbit (bool): If True the calculation includes relativistic spin orbit coupling
        fname (str): File name for the band output
        nk (int): Number of k-points to include in the path (High Symmetry points are currently included twice, increasing nk)
        theta (int): Spin orbit angle
        phi (int): Spin orbit azimuthal angle
        lambda_p (list of floats):
        lambda_d (list of float):
        orb_pseudo (list of str): Orbitals included in the Pseudopotential

    Returns:
        None

    '''
    from .defs.do_bands import do_bands
    from .defs.communication import gather_full

    arrays,attr = self.data_controller.data_dicts()

    if 'ibrav' not in attr:
      if ibrav is None and self.rank == 0:
        if self.rank == 0:
          print('Must specify \'ibrav\' in the inputfile or as an optional argument to \'calc_bands\'')
        quit()
      else:
        attr['ibrav'] = ibrav

    if 'nk' not in attr: attr['nk'] = nk
    if band_path is not None: attr['band_path'] = band_path
    if 'do_spin_orbit' not in attr: attr['do_spin_orbit'] = spin_orbit
    if high_sym_points is not None: arrays['high_sym_points'] = high_sym_points

    # Prepare HRs for band computation with spin-orbit coupling
    try:
      if self.rank == 0 and attr['do_spin_orbit']:
        from .defs.do_spin_orbit import do_spin_orbit_bands

        natoms = attr['natoms']
        if 'phi' not in attr: attr['phi'] = phi
        if 'theta' not in attr: attr['theta'] = theta
        if 'lambda_p' not in arrays: arrays['lambda_p'] = lambda_p[:]
        if 'lambda_d' not in arrays: arrays['lambda_d'] = lambda_d[:]
        if 'orb_pseudo' not in arrays: arrays['orb_pseudo'] = orb_pseudo[:]
        if len(arrays['lambda_p']) != natoms or len(arrays['lambda_p']) != natoms:
          print('\'lambda_p\' and \'lambda_d\' must contain \'natoms\' (%d) elements each.'%natoms)
          self.comm.Abort()

        do_spin_orbit_bands(self.data_controller)

      # Calculate the bands
      do_bands(self.data_controller)

      if self.rank == 0 and arrays['kq'].shape[1] == attr['nkpnts']:
        print('WARNING: The bands kpath and nscf calculations have the same size.')
        print('Spin Texture calculation should be performed after \'pao_eigh\' to ensure integration across the entire BZ.\n')

      E_kp = gather_full(arrays['E_k'], attr['npool'])
      self.data_controller.write_bands(fname, E_kp)
      E_kp = None
    except:
      self.report_exception('bands')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Bands')



  def wave_function_projection ( self, dimension=3 ):
    '''
    Marcio, can you write something here please?
    Please check the argument descriptions also, I half guessed.

    Arguments:
        dimension (int): Dimensionality of the system

    Returns:
        None
mo    '''
    from .defs.do_wave_function_site_projection import wave_function_site_projection

    try:
      wave_function_site_projection(self.data_controller)
    except:
      self.report_exception('wave_function_projection')
      if self.data_controller.data_attributes['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('wave_function_projection')


  def doubling_Hamiltonian ( self, nx , ny, nz ):
    '''
    Marcio, can you write something here please?
    Please check the argument descriptions also

    Arguments:
        nx (bool): Number of doubles in first dimension
        ny (bool): Number of doubles in second dimension
        nz (bool): Number of doubles in third dimension

    Returns:
        None
    '''
    from .defs.do_doubling import doubling_HRs
    
    arrays,attr = self.data_controller.data_dicts()
    attr['nx'],attr['ny'],attr['nz'] = nx,ny,nz
    
    try:
      doubling_HRs(self.data_controller)
    except:
      self.report_exception('doubling_Hamiltonian')
      if attr['abort_on_exception']:
        self.comm.Abort()

    # Broadcasting the modified arrays
    #self.data_controller.broadcast_single_array('HRs')
    #self.data_controller.broadcast_single_array('tau')
    #self.data_controller.broadcast_single_array('a_vectors')
    #self.data_controller.broadcast_single_array('naw')
    #self.data_controller.broadcast_single_array('sh')
    #if (attr['do_spin_orbit']):
    #    self.data_controller.broadcast_single_array('lambda_p')
    #    self.data_controller.broadcast_single_array('lambda_d')
    #    self.data_controller.broadcast_single_array('orb_pseudo')

    self.report_module_time('doubling_Hamiltonian')
  


  def cutting_Hamiltonian ( self, x=False , y=False, z=False ):
    '''
    Marcio, can you write something here please?

    Arguments:
        x (bool): If True, cut along the first dimension
        y (bool): If True, cut along the second dimension
        z (bool): If True, cut along the third dimension

    Returns:
        None
    '''
    arry,attr = self.data_controller.data_dicts()

    try:
      if x:
        for i in range(attr['nk1']-1,0,-1):
          arry['HRs'] = np.delete(arry['HRs'],i,2)
      if y:
        for i in range(attr['nk2']-1,0,-1):
          arry['HRs'] = np.delete(arry['HRs'],i,3)
      if z:
        for i in range(attr['nk3']-1,0,-1):
          arry['HRs'] = np.delete(arry['HRs'],i,4)

      _,_,attr['nk1'],attr['nk2'],attr['nk3'],_ = arry['HRs'].shape
      attr['nkpnts'] = attr['nk1']*attr['nk2']*attr['nk3']
    except:
      self.report_exception('cutting_Hamiltonian')
      if attr['abort_on_exception']:
        self.comm.Abort()



  def spin_operator ( self, spin_orbit=False, sh_l=None, sh_j=None):
    '''
    Calculate the Spin Operator for calculations involving spin
      Requires: None
      Yeilds: 'Sj'

    Arguments:
        spin_orbit (bool): If True the calculation includes relativistic spin orbit coupling
        fnscf (string): Filename for the QE nscf inputfile, from which to read shell data
        sh (list of ints): The Shell levels
        nl (list of ints): The Shell level occupations

    Returns:
        None
    '''
    arrays,attr = self.data_controller.data_dicts()

    if 'do_spin_orbit' not in attr: attr['do_spin_orbit'] = spin_orbit

    if 'sh_l' not in arrays and 'sh_j' not in arrays:
      if sh_l is None and sh_j is None:
        from .defs.read_sh_nl import read_sh_nl
        arrays['sh_l'],arrays['sh_j'] = read_sh_nl(self.data_controller)
      else:
        arrays['sh_l'],arrays['sh_j'] = sh_l,sh_j
    try:
      nawf = attr['nawf']

      # Compute spin operators
      # Pauli matrices (x,y,z)
      Sj = np.zeros((3,nawf,nawf), dtype=complex)
      sP = 0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
      if spin_orbit:
        # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
        for spol in range(3):
          for i in range(nawf//2):
            Sj[spol,i,i] = sP[spol][0,0]
            Sj[spol,i,i+1] = sP[spol][0,1]
          for i in range(nawf//2, nawf):
            Sj[spol,i,i-1] = sP[spol][1,0]
            Sj[spol,i,i] = sP[spol][1,1]
      else:
        from .defs.clebsch_gordan import clebsch_gordan
        # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
        for spol in range(3):
          Sj[spol,:,:] = clebsch_gordan(nawf, arrays['sh_l'], arrays['sh_j'], spol)

      arrays['Sj'] = Sj
    except:
      self.report_exception('spin_operator')
      if attr['abort_on_exception']:
        self.comm.Abort()




  def topology ( self, eff_mass=False, Berry=False, spin_Hall=False, spin_orbit=False, spol=None, ipol=None, jpol=None ):
    '''
    Calculate the Band Topology along the k-path 'kq'

    Arguments:
        eff_mass (bool): If True calculate the Effective Mass Tensor
        Berry (bool): If True calculate the Berry Curvature
        spin_Hall (bool): If True calculate Spin Hall Conductivity
        spin_orbit (bool): If True the calculation includes spin_orbit effects for topology.
        spol (int): Spin polarization
        ipol (int): In plane dimension 1
        jpol (int): In plane dimension 2

    Returns:
        None
    '''
    from .defs.do_topology import do_topology
    # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry
    # curvature operators along the path in the IBZ from do_topology_calc

    arrays,attr = self.data_controller.data_dicts()

    if 'Berry' not in attr: attr['Berry'] = Berry
    if 'eff_mass' not in attr: attr['eff_mass'] = eff_mass
    if 'spin_Hall' not in attr: attr['spin_Hall'] = spin_Hall
    if 'do_spin_orbit' not in attr: attr['do_spin_orbit'] = spin_orbit

    if 'spol' not in attr: attr['spol'] = spol
    if 'ipol' not in attr: attr['ipol'] = ipol
    if 'jpol' not in attr: attr['jpol'] = jpol

    if attr['spol'] is None or attr['ipol'] is None or attr['jpol'] is None:
      if self.rank == 0:
        print('Must specify \'spol\', \'ipol\', and \'jpol\'')
      quit()

    if spin_Hall and 'Sj' not in arrays:
      self.spin_operator(spin_orbit=attr['do_spin_orbit'])

    try:
      do_topology(self.data_controller)
    except:
      self.report_exception('topology')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Band Topology')

    del arrays['R']
    del arrays['idx']
    del arrays['Rfft']
    del arrays['R_wght']



  def interpolated_hamiltonian ( self, nfft1=0, nfft2=0, nfft3=0 ):
    '''
    Calculate the interpolated Hamiltonian with the method of zero padding
    Yields 'Hksp'

    Arguments:
        nfft1 (int): Desired size of the interpolated Hamiltonian's first dimension
        nfft2 (int): Desired size of the interpolated Hamiltonian's second dimension
        nfft3 (int): Desired size of the interpolated Hamiltonian's third dimension

    Returns:
        None
    '''
    from .defs.get_K_grid_fft import get_K_grid_fft
    from .defs.do_double_grid import do_double_grid
    from .defs.communication import gather_scatter

    arrays,attr = self.data_controller.data_dicts()

    try:

      if 'HRs' not in arrays:
        raise KeyError('HRs')

      nawf = attr['nawf']
      nko1,nko2,nko3 = attr['nk1'],attr['nk2'],attr['nk3']

      # Automatically doubles grid in any direction with unspecified nfft value
      if nfft1 == 0: nfft1 = 2*nko1
      if nfft2 == 0: nfft2 = 2*nko2
      if nfft3 == 0: nfft3 = 2*nko3

      attr['nfft1'],attr['nfft2'],attr['nfft3'] = nfft1,nfft2,nfft3

      # Adjust 'npool' if arrays exceed MPI maximum
      int_max = 2147483647
      temp_pool = int(np.ceil((float(nawf**2*nfft1*nfft2*nfft3*3*attr['nspin'])/float(int_max))))
      if temp_pool > attr['npool']:
        if self.rank == 0:
          print("Warning: %s too low. Setting npool to %s"%(attr['npool'],temp_pool))
        attr['npool'] = temp_pool

      # Fourier interpolation on extended grid (zero padding)
      do_double_grid(self.data_controller)

      snawf,_,_,_,nspin = arrays['Hksp'].shape
      arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf,attr['nkpnts'],nspin))
      arrays['Hksp'] = gather_scatter(arrays['Hksp'], 1, attr['npool'])
      snktot = arrays['Hksp'].shape[1]
      arrays['Hksp'] = np.reshape(np.moveaxis(arrays['Hksp'],0,1), (snktot,nawf,nawf,nspin))

      get_K_grid_fft(self.data_controller)

      # Report new memory requirements
      if self.rank == 0:
        gbyte = self.memory_check()
        if attr['verbose']:
          print('Performing Fourier interpolation on a larger grid.')
          print('d : nk -> nfft\n1 : %d -> %d\n2 : %d -> %d\n3 : %d -> %d'%(nko1,nfft1,nko2,nfft2,nko3,nfft3))
        print('New estimated maximum array size: %.2f GBytes'%gbyte)

    except:
      self.report_exception('interpolated_hamiltonian')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('R -> k with Zero Padding')



  def pao_eigh ( self, bval=0 ):
    '''
    Calculate the Eigen values and vectors of k-space Hamiltonian 'Hksp'
    Yields 'E_k' and 'v_k'

    Arguments:
        bval (int): Top valence band number (nelec/2) to correctly shift Eigenvalues

    Returns:
        None
    '''
    from .defs.do_eigh import do_pao_eigh
    from .defs.communication import gather_scatter,scatter_full,gather_full

    arrays,attr = self.data_controller.data_dicts()

    if 'bval' not in attr: attr['bval'] = bval

    # HRs and Hks are replaced with Hksp
    del arrays['HRs']

    try:
      if 'Hksp' not in arrays:
        if self.rank == 0:
          nktot = attr['nkpnts']
          nawf,_,nk1,nk2,nk3,nspin = arrays['Hks'].shape
          arrays['Hks'] = np.moveaxis(np.reshape(arrays['Hks'],(nawf,nawf,nktot,nspin),order='C'), 2, 0)
        else:
          arrays['Hks'] = None
        arrays['Hksp'] = scatter_full(arrays['Hks'], attr['npool'])
        del arrays['Hks']

      do_pao_eigh(self.data_controller)

      ### PARALLELIZATION
      ## DEV: Sample RunTime Here
      ## DEV: Parallelize search for amax & subtract for all processes.
      if 'HubbardU' in arrays and arrays['HubbardU'].any() != 0.0:
        arrays['E_k'] = gather_full(arrays['E_k'], attr['npool'])
        if self.rank == 0:
          if attr['verbose']:
            print('Shifting Eigenvalues to top of valence band.')
          arrays['E_k'] -= np.amax(arrays['E_k'][:,attr['bval'],:])
        self.comm.Barrier()
        arrays['E_k'] = scatter_full(arrays['E_k'], attr['npool'])
    except:
      self.report_exception('pao_eigh')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Eigenvalues')



  def gradient_and_momenta ( self,band_curvature=False ):
    '''
    Calculate the Gradient of the k-space Hamiltonian, 'Hksp'
    Requires 'Hksp'
    Yields 'dHksp'

    Arguments:
        None

    Returns:
        None
    '''
    from .defs.do_gradient import do_gradient
    from .defs.do_momentum import do_momentum
    from .defs.communication import gather_scatter
    import numpy as np 

    arrays,attr = self.data_controller.data_dicts()

    try:
      snktot,nawf,_,nspin = arrays['Hksp'].shape

      for ik in range(snktot):
        for ispin in range(nspin):
          #make sure Hksp is hermitian (it should be)
          arrays['Hksp'][ik,:,:,ispin] = (np.conj(arrays['Hksp'][ik,:,:,ispin].T) + arrays['Hksp'][ik,:,:,ispin])/2.

      arrays['Hksp'] = np.reshape(arrays['Hksp'], (snktot, nawf**2, nspin))
      arrays['Hksp'] = np.moveaxis(gather_scatter(arrays['Hksp'],1,attr['npool']), 0, 1)
      snawf,_,nspin = arrays['Hksp'].shape
      arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf,attr['nk1'],attr['nk2'],attr['nk3'],nspin))

      do_gradient(self.data_controller)

      if not band_curvature:
        # No more need for k-space Hamiltonian
        del arrays['Hksp']

      ### PARALLELIZATION
      #gather dHksp on nawf*nawf and scatter on k points
      arrays['dHksp'] = np.reshape(arrays['dHksp'], (snawf,attr['nkpnts'],3,nspin))
      arrays['dHksp'] = np.moveaxis(gather_scatter(arrays['dHksp'],1,attr['npool']), 0, 2)
      arrays['dHksp'] = np.reshape(arrays['dHksp'], (snktot,3,nawf,nawf,nspin), order="C")

      if band_curvature:
        from .defs.do_band_curvature import do_band_curvature
        do_band_curvature (self.data_controller)
      else:
        # No more need for k-space Hamiltonian
        del arrays['Hksp']


    except:
      self.report_exception('gradient_and_momenta')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Gradient')

    ### DEV: Proposed to remove this and calculate pksp or velkp when required
    # Compute the momentum operator p_n,m(k) (and kinetic energy operator)
    do_momentum(self.data_controller)
    self.report_module_time('Momenta')



  def adaptive_smearing ( self, smearing='gauss' ):
    '''
    Calculate the Adaptive Smearing parameters
    Yields 'deltakp' and 'deltakp2'

    Arguments:
        smearing (str): Smearing type (m-p and gauss)

    Returns:
        None
    '''
    from .defs.do_adaptive_smearing import do_adaptive_smearing

    attr = self.data_controller.data_attributes

    if smearing != 'gauss' and 'smearing' != 'm-p':
      raise ValueError('Smearing type %s not supported.\nSmearing types are \'gauss\' and \'m-p\''%str(smearing))
      #if self.rank == 0:
      #  print('Smearing type %s not supported.\nSmearing types are \'gauss\' and \'m-p\''%str(attr['smearing']))
      #quit()

    try:
      do_adaptive_smearing(self.data_controller, smearing)
    except:
      self.report_exception('adaptive_smearing')
      if attr['abort_on_exception']:
        self.comm.Abort()
    self.report_module_time('Adaptive Smearing')



  def dos ( self, do_dos=True, do_pdos=True, delta=0.01, emin=-10., emax=2., ne=1000 ):
    '''
    Calculate the Density of States and Projected Density of States
      If Adaptive Smearing has been performed, the Adaptive DoS will be calculated

    Arguments:
        do_dos (bool): Perform Density of States calculation
        do_pdos (bool): Perform Projected Density of States calculation
        delta (float): The gaussian width
        emin (float): The minimum energy in the range to be computed
        emax (float): The maximum energy in the range to be computed
        ne (int): The number of points to place in the range [emin,emax]

    Returns:
        None
    '''
    arrays,attr = self.data_controller.data_dicts()

    if 'smearing' not in attr: attr['smearing'] = None

    try:
      if attr['smearing'] is None:
        if do_dos:
          from .defs.do_dos import do_dos
          do_dos(self.data_controller, emin, emax, ne, delta)
        if do_pdos:
          from .defs.do_pdos import do_pdos
          do_pdos(self.data_controller, emin, emax, ne, delta)
      else:
        if 'deltakp' not in arrays:
          if self.rank == 0:
            print('Perform calc_adaptive_smearing() to calculate \'deltakp\' before calling calc_dos_adaptive()')
          quit()

        if do_dos:
          from .defs.do_dos import do_dos_adaptive
          do_dos_adaptive(self.data_controller, emin, emax, ne)

        if do_pdos:
          from .defs.do_pdos import do_pdos_adaptive
          do_pdos_adaptive(self.data_controller, emin, emax, ne)
    except:
      self.report_exception('dos')
      if attr['abort_on_exception']:
        self.comm.Abort()

    mname = 'DoS%s'%('' if attr['smearing'] is None else ' (Adaptive Smearing)')
    self.report_module_time(mname)



  def trim_non_projectable_bands ( self ):

    arrays = self.data_controller.data_arrays

    bnd = attributes['nawf'] = attributes['bnd']

    arrays['E_k'] = arrays['E_k'][:,:bnd]
    arrays['pksp'] = arrays['pksp'][:,:,:bnd,:bnd]
    if 'deltakp' in arrays:
      arrays['deltakp'] = arrays['deltakp'][:,:bnd]
      arrays['deltakp2'] = arrays['deltakp2'][:,:bnd]



  def fermi_surface ( self, fermi_up=1., fermi_dw=-1. ):
    '''
    Calculate the Fermi Surface

    Arguments:
        fermi_up (float): The upper limit of the occupied energy range
        fermi_dw (float): The lower limit of the occupied energy range

    Returns:
        None
    '''
    from .defs.do_fermisurf import do_fermisurf

    attr = self.data_controller.data_attributes

    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    try:
      do_fermisurf(self.data_controller)
    except:
      self.report_exception('fermi_surface')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Fermi Surface')



  def spin_texture ( self, fermi_up=1., fermi_dw=-1. ):
    '''
    Calculate the Spin Texture

    Arguments:
        fermi_up (float): The upper limit of the occupied energy range
        fermi_dw (float): The lower limit of the occupied energy range

    Returns:
        None
    '''
    from .defs.do_spin_texture import do_spin_texture

    arry,attr = self.data_controller.data_dicts()

    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    try:
      if attr['nspin'] == 1:
        if 'Sj' not in arry:
          self.spin_operator()
        do_spin_texture(self.data_controller)
        self.report_module_time('Spin Texture')
      else:
        if self.rank == 0:
          print('Cannot compute spin texture with nspin=2')
    except:
      self.report_exception('spin_texture')
      if attr['abort_on_exception']:
        self.comm.Abort()


    self.comm.Barrier()



  def spin_Hall ( self, twoD=False, do_ac=False, emin=-1., emax=1., fermi_up=1., fermi_dw=-1., s_tensor=None ):
    '''
    Calculate the Spin Hall Conductivity
      Currently this module does not possess the "spin_orbit" capability of do_topology, because I(Frank) do not know what this modification entails.
      Thus, do_spin_orbit defaults to False here. If anybody needs the spin_orbit capability here, please contact me and we'll sort it out.

    Arguments:
        twoD (bool): True to output in 2D units of Ohm^-1, neglecting the sample height in the z direction
        do_ac (bool): True to calculate the Spic Circular Dichroism
        emin (float): The minimum energy in the range
        emax (float): The maximum energy in the range
        fermi_up (float): The upper limit of the occupied energy range
        fermi_dw (float): The lower limit of the occupied energy range
        s_tensor (list): List of tensor elements to calculate (e.g. To calculate xxx and zxy use [[0,0,0],[0,1,2]])

    Returns:
        None
    '''
    from .defs.do_Hall import do_spin_Hall

    arrays,attr = self.data_controller.data_dicts()

    attr['eminH'],attr['emaxH'] = emin,emax

    if s_tensor is not None: arrays['s_tensor'] = np.array(s_tensor)
    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    if 'Sj' not in arrays:
      self.spin_operator()

    try:
      do_spin_Hall(self.data_controller, twoD, do_ac)
    except:
      self.report_exception('spin_Hall')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Spin Hall Conductivity')



  def anomalous_Hall ( self, do_ac=False, emin=-1., emax=1., fermi_up=1., fermi_dw=-1., a_tensor=None ):
    '''
    Calculate the Anomalous Hall Conductivity

    Arguments:
        do_ac (bool): True to calculate the Magnetic Circular Dichroism
        emin (float): The minimum energy in the range
        emax (float): The maximum energy in the range
        fermi_up (float): The upper limit of the occupied energy range
        fermi_dw (float): The lower limit of the occupied energy range
        a_tensor (list): List of tensor elements to calculate (e.g. To calculate xx and yz use [[0,0],[1,2]])

    Returns:
        None
    '''
    from .defs.do_Hall import do_anomalous_Hall

    arrays,attr = self.data_controller.data_dicts()

    attr['eminH'] = emin
    attr['emaxH'] = emax

    if 'a_tensor' is not None: arrays['a_tensor'] = np.array(a_tensor)
    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    try:
      do_anomalous_Hall(self.data_controller, do_ac)
    except:
      self.report_exception('anomalous_Hall')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Anomalous Hall Conductivity')



  def transport ( self, tmin=300, tmax=300, tstep=1, emin=0., emax=10., ne=500, t_tensor=None, carrier_conc=False):
    '''
    Calculate the Transport Properties

    Arguments:
        tmin (float): The minimum temperature in the range
        tmax (float): The maximum temperature in the range
        tstep (float): The step size for temperature increments
        emin (float): The minimum energy in the range
        emax (float): The maximum energy in the range
        ne (float): The number of energy increments
        t_tensor (list): List of tensor elements to calculate (e.g. To calculate xx and yz use [[0,0],[1,2]])

    Returns:
        None
    '''
    from .defs.do_transport import do_transport

    arrays,attr = self.data_controller.data_dicts()

    ene = np.linspace(emin, emax, ne)
    temps = np.arange(tmin, tmax+1.e-10, tstep)

    if t_tensor is not None: arrays['t_tensor'] = np.array(t_tensor)

    try:
      # Compute Velocities for Spin 0 Only
      bnd = attr['bnd']
      velkp = np.zeros((arrays['pksp'].shape[0],3,bnd,attr['nspin']))
      for n in range(bnd):
        velkp[:,:,n,:] = np.real(arrays['pksp'][:,:,n,n,:])

      do_transport(self.data_controller, temps, ene, velkp, carrier_conc)


      if carrier_conc:
        from .defs.do_transport import do_carrier_conc        
        do_carrier_conc(self.data_controller,velkp,ene,temps)

      velkp = None

    except:
      self.report_exception('transport')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Transport')



  def dielectric_tensor ( self, metal=False, temp=None, delta=0.01, emin=0., emax=10., ne=500., d_tensor=None ):
    '''
    Calculate the Dielectric Tensor

    Arguments:
        metal (bool): True if system is metallic
        temp (float): Temperature (default is Room Temperature)
        delta (float): Smearing width for gaussian (if smearing is None)
        emin (float): The minimum value of energy
        emax (float): The maximum value of energy
        ne (float): Number of energy values between emin and emax
        d_tensor (list): List of tensor elements to calculate (e.g. To calculate xx and yz use [[0,0],[1,2]])

    Returns:
        None
    '''
    from .defs.do_epsilon import do_dielectric_tensor

    arrays,attr = self.data_controller.data_dicts()

    if temp is not None: attr['temp'] = temp
    if 'delta' not in attr: attr['delta'] = delta
    if 'metal' not in attr: attr['metal'] = metal
    if d_tensor is not None: arrays['d_tensor'] = np.array(d_tensor)

    #-----------------------------------------------
    # Compute dielectric tensor (Re and Im epsilon)
    #-----------------------------------------------

    try:
      ene = np.linspace(emin, emax, ne)
      do_dielectric_tensor(self.data_controller, ene)
    except:
      self.report_exception('dielectric_tensor')
      if attr['abort_on_exception']:
        self.comm.Abort()

    self.report_module_time('Dielectric Tensor')



