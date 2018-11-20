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
  gb_fudge_factor = 10.


  def __init__ ( self, workpath='./', outputdir='output', inputfile=None, savedir=None, smearing=None, npool=1, verbose=False ):
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
      if self.size > 1:
        print('Parallel execution on %d processors and %d pool'%(self.size,attr['npool']) + ('' if attr['npool']==1 else 's'))
      else:
        print('Serial execution')

    # Do memory checks 
    if self.rank == 0:
      dxdydz = 3
      B_to_GB = 1.E-9
      spins = attr['nspin']
      ff = self.gb_fudge_factor
      bytes_per_complex = 128//8
      num_wave_functions = attr['nawf']
      nd1,nd2,nd3 = attr['nk1'],attr['nk2'],attr['nk3']
      gbyte = num_wave_functions**2 * (nd1*nd2*nd3) * spins * dxdydz * bytes_per_complex * ff * B_to_GB
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
        quit()

      # Format string and print
      lms = spaces-lmn
      dt = time() - self.reset_time
      print('%s in: %s %.3f sec'%(mname,lms*' ',dt))
      self.reset_time = time()



  def finish_execution ( self ):
    import resource
    from time import time
    from mpi4py import MPI

    verbose = self.data_controller.data_attributes['verbose']

    if self.rank == 0:
      tt = time() - self.start_time
      print('Total CPU time =%s%.3f sec'%(27*' ',tt))
      if verbose:
        mem = np.asarray(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print("Memory usage on rank 0:  %6.4f GB"%(mem/1024.0**2))
    self.comm.Barrier()
    if self.rank == 1 and verbose:
      mem = self.size*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      print("Approximate maximum concurrent memory usage:  %6.4f GB"%(mem/1024.0**2))
    MPI.Finalize()
    quit()



  def projectability ( self, pthr=0.95, shift='auto' ):
    from .defs.do_projectability import do_projectability

    attr = self.data_controller.data_attributes

    if 'pthr' not in attr: attr['pthr'] = pthr
    if 'shift' not in attr: attr['shift'] = shift

    do_projectability(self.data_controller)
    self.report_module_time('Projectability')



  def pao_hamiltonian ( self, non_ortho=False, shift_type=1 ):
    from .defs.get_K_grid_fft import get_K_grid_fft
    from .defs.do_build_pao_hamiltonian import do_build_pao_hamiltonian,do_Hks_to_HRs

    # Data Attributes and Arrays
    arrays,attr = self.data_controller.data_dicts()

    if 'non_ortho' not in attr: attr['non_ortho'] = non_ortho
    if 'shift_type' not in attr: attr['shift_type'] = shift_type

    do_build_pao_hamiltonian(self.data_controller)
    self.report_module_time('Building Hks')

    # Done with U
    del arrays['U']

    do_Hks_to_HRs(self.data_controller)
#### PARALLELIZATION
    self.data_controller.broadcast_single_array('HRs')
    if attr['non_ortho']:
      self.data_controller.broadcast_single_array('SRs')

    get_K_grid_fft(self.data_controller)
    self.report_module_time('k -> R')

    if non_ortho:
      from .defs.do_ortho import do_orthogonalize

      do_orthogonalize(self.data_controller)
      self.report_module_time('Orthogonalize')



  def add_external_fields ( self, Efield=[0.], Bfield=[0.], HubbardU=[0.] ):

    arry,attr = self.data_controller.data_dicts()

    if 'Efield' not in arry: arry['Efield'] = np.array(Efield)
    if 'Bfield' not in arry: arry['Bfield'] = np.array(Bfield)
    if 'HubbardU' not in arry: arry['HubbardU'] = np.array(HubbardU)

    Efield,Bfield,HubbardU = arry['Efield'],arry['Bfield'],arry['HubbardU']

    # Add external fields or non scf ACBN0 correction
    if self.rank == 0 and (Efield.any() != 0. or Bfield.any() != 0. or HubbardU.any() != 0.):
      from .defs.add_ext_field import add_ext_field
      add_ext_field(self.data_controller)
      if attr['verbose']:
        print('External Fields Added')
    self.comm.Barrier()



  def z2_pack ( self, fname='z2pack_hamiltonian.dat' ):

    attr,arry = self.data_controller.data_dicts()

    if 'HRs' not in arry:
      print('HRs must first be calculated with \'build_pao_hamiltonian\'')
    else:
      self.data_controller.write_z2pack(fname)



  def bands ( self, ibrav=None, spin_orbit=False, fname='bands', nk=500., theta=0., phi=0., lambda_p=[0.], lambda_d=[0.] ):
    from .defs.do_bands import do_bands

    arrays,attr = self.data_controller.data_dicts()

    if 'ibrav' not in attr:
      if ibrav is None:
        if self.rank == 0:
          print('Must specify \'ibrav\' in the inputfile or as an optional argument to \'calc_bands\'')
        quit()
      else:
        attr['ibrav'] = ibrav

    if 'nk' not in attr: attr['nk'] = nk
    if 'do_spin_orbit' not in attr: attr['do_spin_orbit'] = spin_orbit

    #----------------------------------------
    # Compute bands with spin-orbit coupling
    #----------------------------------------
    if self.rank == 0 and attr['do_spin_orbit']:
      from .defs.do_spin_orbit import do_spin_orbit_bands

      natoms = attr['natoms']
      if len(lambda_p) != natoms or len(lambda_d) != natoms:
        if self.rank == 0:
          print('\'lambda_p\' and \'lambda_d\' must contain \'natoms\' (%d) elements each.'%natoms)
        quit()

      if 'phi' not in attr: attr['phi'] = phi
      if 'theta' not in attr: attr['theta'] = theta
      if 'lambda_p' not in arrays: arrays['lambda_p'] = lambda_p[:]
      if 'lambda_d' not in arrays: arrays['lambda_d'] = lambda_d[:]

      do_spin_orbit_bands(self.data_controller)

    do_bands(self.data_controller)

    E_kp = gather_full(arrays['E_k'], attr['npool'])
    data_controller.write_bands(fname, E_kp)
    E_kp = None

    self.report_module_time('Bands')



  def spin_operator ( self, spin_orbit=False, sh=[0,1,2,0,1,2], nl=[2,1,1,1,1,1]):

    arrays,attr = self.data_controller.data_dicts()

    if 'do_spin_orbit' not in attr: attr['do_spin_orbit'] = spin_orbit
    if 'sh' not in arrays: arrays['sh'] = sh
    if 'nl' not in arrays: arrays['nl'] = nl

    nawf = attr['nawf']

    # Compute spin operators
    # Pauli matrices (x,y,z)
    sP = 0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
    if spin_orbit:
      # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
      Sj = np.zeros((3,nawf,nawf), dtype=complex)
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
      Sj = np.zeros((3,nawf,nawf), dtype=complex)
      for spol in range(3):
        Sj[spol,:,:] = clebsch_gordan(nawf, arrays['sh'], arrays['nl'], spol)

    arrays['Sj'] = Sj



  def topology ( self, eff_mass=False, Berry=False, spin_Hall=False, spol=None, ipol=None, jpol=None ):
    from .defs.do_topology import do_topology
    # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry
    # curvature operators along the path in the IBZ from do_topology_calc 

    arrays,attr = self.data_controller.data_dicts()

    if 'Berry' not in attr: attr['Berry'] = Berry
    if 'eff_mass' not in attr: attr['eff_mass'] = eff_mass
    if 'spin_Hall' not in attr: attr['spin_Hall'] = spin_Hall

    if 'spol' not in attr: attr['spol'] = spol
    if 'ipol' not in attr: attr['ipol'] = ipol
    if 'jpol' not in attr: attr['jpol'] = jpol

    if attr['spol'] is None or attr['ipol'] is None or attr['jpol'] is None:
      if self.rank == 0:
        print('Must specify \'spol\', \'ipol\', and \'jpol\'')
      quit()

    if attr['spin_Hall'] and 'Sj' not in arrays:
      if rank == 0:
        print('Spin operator \'Sj\' must be calculated before Band Topology can be computed.')
      quit()

    do_topology(self.data_controller)
    self.report_module_time('Band Topology')

    del arrays['R']
    del arrays['idx']
    del arrays['Rfft']
    del arrays['R_wght']



  def interpolated_hamiltonian ( self, nfft1=None, nfft2=None, nfft3=None ):
    from .defs.get_K_grid_fft import get_K_grid_fft
    from .defs.do_double_grid import do_double_grid
    from .defs.communication import gather_scatter,scatter_full

    arrays,attr = self.data_controller.data_dicts()

    # Automatically doubles grid in all directions
    if nfft1 is None: nfft1 = 2*attr['nk1']
    if nfft2 is None: nfft2 = 2*attr['nk2']
    if nfft3 is None: nfft3 = 2*attr['nk3']

    if 'nfft1' not in attr: attr['nfft1'] = nfft1
    if 'nfft2' not in attr: attr['nfft2'] = nfft2
    if 'nfft3' not in attr: attr['nfft3'] = nfft3

    nfft1,nfft2,nfft3 = attr['nfft1'],attr['nfft2'],attr['nfft3']

    # Ensure FFT grid is even
    if nfft1%2!=0 or nfft2%2!=0 or nfft3%2!=0:
      nfft1 = 2*((nfft1+1)//2)
      nfft2 = 2*((nfft2+1)//2)
      nfft3 = 2*((nfft3+1)//2)
      if self.rank == 0:
        print('Warning: nfft grid has been modified to support double_grid\nModified nfft grid to: %d %d %d'%(nfft1,nfft2,nfft3))

    # Adjust 'npool' if arrays exceed MPI maximum
    int_max = 2147483647
    temp_pool = int(np.ceil((float(attr['nawf']**2*nfft1*nfft2*nfft3*3*attr['nspin'])/float(int_max))))
    if temp_pool > attr['npool']:
      if self.rank == 0:
        print("Warning: %s too low. Setting npool to %s"%(attr['npool'],temp_pool))
      attr['npool'] = temp_pool

    if self.rank == 0:
      dxdydz = 3
      B_to_GB = 1.E-9
      spins = attr['nspin']
      bytes_per_complex = 128//8
      nd1,nd2,nd3 = nfft1,nfft2,nfft3
      num_wave_functions = attr['nawf']
      ff = self.gb_fudge_factor
      nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']
      gbyte = num_wave_functions**2 * (nd1*nd2*nd3) * spins * dxdydz * bytes_per_complex * ff * B_to_GB
      if attr['verbose']:
        print('Performing Fourier interpolation on a larger grid.')
        print('d : nk -> nfft\n1 : %d -> %d\n2 : %d -> %d\n3 : %d -> %d'%(nk1,nfft1,nk2,nfft2,nk3,nfft3))
      print('New estimated maximum array size: %.2f GBytes'%gbyte)

    #------------------------------------------------------
    # Fourier interpolation on extended grid (zero padding)
    #------------------------------------------------------

    if self.rank == 0:
      nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']
      arrays['HRs'] = np.reshape(arrays['HRs'], (attr['nawf']**2,nk1,nk2,nk3,attr['nspin']))
    arrays['HRs'] = scatter_full((arrays['HRs'] if self.rank==0 else None), attr['npool'])

    do_double_grid(self.data_controller)

    snawf,_,_,_,nspin = arrays['Hksp'].shape
    arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf,attr['nkpnts'],nspin))
    arrays['Hksp'] = gather_scatter(arrays['Hksp'], 1, attr['npool'])
    nawf = attr['nawf']
    snktot = arrays['Hksp'].shape[1]
    arrays['Hksp'] = np.reshape(np.moveaxis(arrays['Hksp'],0,1), (snktot,nawf,nawf,nspin))

    get_K_grid_fft(self.data_controller)

    self.report_module_time('R -> k with Zero Padding')



  def pao_eigh ( self, bval=0 ):
    from .defs.do_eigh import do_pao_eigh
    from .defs.communication import gather_scatter,scatter_full,gather_full

    arrays,attr = self.data_controller.data_dicts()

    del arrays['HRs']
    if 'bval' not in attr:
      attr['bval'] = bval

    if 'Hksp' not in arrays:
      if self.rank == 0:
        nktot = attr['nkpnts']
        nawf,_,nk1,nk2,nk3,nspin = arrays['Hks'].shape
        arrays['Hks'] = np.moveaxis(np.reshape(arrays['Hks'],(nawf,nawf,nktot,nspin),order='C'), 2, 0)
      else:
        arrays['Hks'] = None
      arrays['Hksp'] = scatter_full(arrays['Hks'], attr['npool'])
      del arrays['Hks']

    #-----------------------------------------------------
    # Compute eigenvalues of the interpolated Hamiltonian
    #-----------------------------------------------------
    do_pao_eigh(self.data_controller)

### PARALLELIZATION
### Sample RunTime Here
    ## Parallelize search for amax & subtract for all processes.
    if 'HubbardU' in arrays and arrays['HubbardU'].any() != 0.0:
      if self.rank == 0 and attr['verbose']:
        print('Shifting Eigenvalues to top of valence band.')
      arrays['E_k'] = gather_full(arrays['E_k'], attr['npool'])
      if self.rank == 0:
        arrays['E_k'] -= np.amax(arrays['E_k'][:,attr['bval'],:])
      self.comm.Barrier()
      arrays['E_k'] = scatter_full(arrays['E_k'], attr['npool'])

    self.report_module_time('Eigenvalues')



  def gradient_and_momenta ( self ):
    from .defs.do_gradient import do_gradient
    from .defs.do_momentum import do_momentum
    from .defs.communication import gather_scatter

    arrays,attr = self.data_controller.data_dicts()

    snktot,nawf,_,nspin = arrays['Hksp'].shape

    for ik in range(snktot):
      for ispin in range(nspin):
        arrays['Hksp'][ik,:,:,ispin] = (np.conj(arrays['Hksp'][ik,:,:,ispin].T) + arrays['Hksp'][ik,:,:,ispin])/2.

    arrays['Hksp'] = np.reshape(arrays['Hksp'], (snktot, nawf**2, nspin))
    arrays['Hksp'] = np.moveaxis(gather_scatter(arrays['Hksp'],1,attr['npool']), 0, 1)
    snawf,_,nspin = arrays['Hksp'].shape
    arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf,attr['nk1'],attr['nk2'],attr['nk3'],nspin))

    #----------------------
    # Compute the gradient of the k-space Hamiltonian
    #----------------------            
    do_gradient(self.data_controller)

########### PARALLELIZATION
    #gather dHksp on nawf*nawf and scatter on k points
    arrays['dHksp'] = np.reshape(arrays['dHksp'], (snawf,attr['nkpnts'],3,nspin))
    arrays['dHksp'] = np.moveaxis(gather_scatter(arrays['dHksp'],1,attr['npool']), 0, 2)
    arrays['dHksp'] = np.reshape(arrays['dHksp'], (snktot,3,nawf,nawf,nspin), order="C")

    self.report_module_time('Gradient')

    #----------------------------------------------------------------------
    # Compute the momentum operator p_n,m(k) (and kinetic energy operator)
    #----------------------------------------------------------------------
    do_momentum(self.data_controller)
    self.report_module_time('Momenta')



  def adaptive_smearing ( self, smearing='gauss' ):
    from .defs.do_adaptive_smearing import do_adaptive_smearing

    attr = self.data_controller.data_attributes

    if 'smearing' not in attr or attr['smearing'] is None:
      attr['smearing'] = smearing
    if attr['smearing'] != 'gauss' and attr['smearing'] != 'm-p':
      if self.rank == 0:
        print('Smearing type %s not supported.\nSmearing types are \'gauss\' and \'m-p\''%str(attr['smearing']))
      quit()

    do_adaptive_smearing(self.data_controller)
    self.report_module_time('Adaptive Smearing')



  def dos ( self, do_dos=True, do_pdos=True, delta=0.01, emin=-10., emax=2. ):

    arrays,attr = self.data_controller.data_dicts()

    if 'delta' not in attr: attr['delta'] = delta
    if 'smearing' not in attr: attr['smearing'] = None

    if attr['smearing'] is None:
      if do_dos:
        from .defs.do_dos import do_dos
        do_dos(self.data_controller, emin=emin, emax=emax)
      if do_pdos:
        from .defs.do_pdos import do_pdos
        do_pdos(self.data_controller, emin=emin, emax=emax)
    else:
      if 'deltakp' not in arrays:
        if self.rank == 0:
          print('Perform calc_adaptive_smearing() to calculate \'deltakp\' before calling calc_dos_adaptive()')
        quit()
      #------------------------------------------------------------
      # DOS calculation with adaptive smearing on double_grid Hksp
      #------------------------------------------------------------
      if do_dos:
        from .defs.do_dos import do_dos_adaptive
        do_dos_adaptive(self.data_controller, emin=emin, emax=emax)

      #----------------------
      # PDOS calculation ...
      #----------------------
      if do_pdos:
        from .defs.do_pdos import do_pdos_adaptive
        do_pdos_adaptive(self.data_controller, emin=emin, emax=emax)

    mname = 'DoS%s'%('' if attr['smearing'] is None else ' (Adaptive Smearing)')
    self.report_module_time(mname)



  def trim_non_projectable_bands ( self ):

    arrays,_ = self.data_controller.data_dicts()

    bnd = attributes['bnd']
    attributes['nawf'] = bnd

    arrays['E_k'] = arrays['E_k'][:,:bnd]
    arrays['pksp'] = arrays['pksp'][:,:,:bnd,:bnd]
    if 'deltakp' in arrays:
      arrays['deltakp'] = arrays['deltakp'][:,:bnd]
      arrays['deltakp2'] = arrays['deltakp2'][:,:bnd]



  def fermi_surface ( self, fermi_up=1., fermi_dw=-1. ):
    from .defs.do_fermisurf import do_fermisurf

    attr = self.data_controller.data_attributes

    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    #---------------------------
    # Fermi surface calculation
    #---------------------------
    do_fermisurf(self.data_controller)
    self.report_module_time('Fermi Surface')



  def spin_texture ( self, fermi_up=1., fermi_dw=-1. ):
    from .defs.do_spin_texture import do_spin_texture

    arry,attr = self.data_controller.data_dicts()

    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    if attr['nspin'] == 1:
      if 'Sj' not in arry:
        if rank == 0:
          print('Spin operator \'Sj\' must be calculated before Spin Texture can be computed.')
        quit()
      do_spin_texture(self.data_controller)
      self.report_module_time('Spin Texutre')
    else:
      if self.rank == 0:
        print('Cannot compute spin texture with nspin=2')
        self.comm.Abort()
      self.comm.Barrier()



  def spin_Hall ( self, do_ac=True, emin=-1., emax=1., fermi_up=1., fermi_dw=-1., s_tensor=None ):
    from .defs.do_Hall import do_spin_Hall

    arrays,attr = self.data_controller.data_dicts()

    attr['eminH'] = emin
    attr['emaxH'] = emax

    if s_tensor is not None: arrays['s_tensor'] = np.array(s_tensor)
    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    if 'Sj' not in arrays:
      if rank == 0:
        print('Spin operator \'Sj\' must be calculated before Spin Hall Conductivity can be computed.')
      quit()

    do_spin_Hall(self.data_controller, do_ac)
    self.report_module_time('Spin Hall Conductivity')



  def anomalous_Hall ( self, do_ac=True, emin=-1., emax=1., fermi_up=1., fermi_dw=-1., a_tensor=None ):
    from .defs.do_Hall import do_anomalous_Hall

    arrays,attr = self.data_controller.data_dicts()

    attr['eminH'] = emin
    attr['emaxH'] = emax

    if 'a_tensor' is not None: arrays['a_tensor'] = np.array(a_tensor)
    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    do_anomalous_Hall(self.data_controller, do_ac)
    self.report_module_time('Anomalous Hall Conductivity')



  def transport ( self, tmin=300, tmax=300, tstep=1, emin=0., emax=10., ne=500, t_tensor=None ):
    from .defs.do_transport import do_transport

    arrays,attr = self.data_controller.data_dicts()

    temps = np.arange(tmin, tmax+1.e-10, tstep)
    ene = np.arange(emin, emax, (emax-emin)/float(ne))

    bnd = attr['bnd']
    nspin = attr['nspin']
    snktot = arrays['pksp'].shape[0]

    if t_tensor is not None:
      arrays['t_tensor'] = np.array(t_tensor)

    # Compute Velocities
    velkp = np.zeros((snktot,3,bnd,nspin), dtype=float)
    for n in range(bnd):
      velkp[:,:,n,:] = np.real(arrays['pksp'][:,:,n,n,:])

    do_transport(self.data_controller, temps, ene, velkp)
    velkp = None
    self.report_module_time('Transport')



  def dielectric_tensor ( self, metal=False, kramerskronig=True, temp=None, delta=0.01, emin=0., emax=10., ne=500., d_tensor=None ):
    from .defs.do_epsilon import do_dielectric_tensor

    arrays,attr = self.data_controller.data_dicts()

    #-----------------------------------------------
    # Compute dielectric tensor (Re and Im epsilon)
    #-----------------------------------------------

    if temp is not None: attr['temp'] = temp
    if 'delta' not in attr: attr['delta'] = delta
    if 'metal' not in attr: attr['metal'] = metal
    if 'kramerskronig' not in attr: attr['kramerskronig'] = kramerskronig

    if d_tensor is not None:
      arrays['d_tensor'] = np.array(d_tensor)

    ene = np.arange(emin, emax, (emax-emin)/ne)
    do_dielectric_tensor(self.data_controller, ene)
    self.report_module_time('Dielectric Tensor')
