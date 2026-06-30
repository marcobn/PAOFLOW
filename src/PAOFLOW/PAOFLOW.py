import numpy as np


class PAOFLOW:
    """Post-processing engine for Pseudo-Atomic Orbital (PAO) electronic structure calculations.

    ``PAOFLOW`` reads the output of a plane-wave DFT code (Quantum ESPRESSO or VASP),
    projects the Bloch eigenstates onto a compact PAO basis, and exposes a high-level
    Python API for a broad range of electronic, topological, and transport properties.
    All heavy numerics are MPI-parallelised through ``mpi4py`` and optionally distributed
    across k-point pools to keep per-process memory bounded.

    Typical workflow
    ----------------
    ::

        from PAOFLOW import PAOFLOW

        pf = PAOFLOW(workpath='./', savedir='prefix.save', outputdir='output')
        pf.projections()           # build PAO projections (or read_atomic_proj_QE)
        pf.projectability()        # drop low-projectability bands
        pf.pao_hamiltonian()       # construct H(R) and H(k) in the PAO basis
        pf.interpolated_hamiltonian()  # Fourier-interpolate onto a denser k-grid
        pf.pao_eigh()              # diagonalise → E(k), v(k)
        pf.gradient_and_momenta()  # ∇_k H, momentum matrix
        pf.dos()                   # density of states / projected DOS
        pf.transport()             # Boltzmann transport tensors
        pf.finish_execution()      # print timings and memory usage

    Parameters (constructor)
    ------------------------
    workpath : str, default ``'./'``
        Path to the working directory.
    outputdir : str, default ``'output'``
        Name of the output sub-directory created under ``workpath``.
    inputfile : str, optional
        Path to an XML input file that configures the run.
    savedir : str, optional
        Path to the QE ``.save`` directory (required when not using ``inputfile``).
    model : dict, optional
        Parameters for building the Hamiltonian from a tight-binding model instead
        of a DFT calculation.  Must contain at least the key ``'label'``.
    npool : int, default 1
        Number of k-point pools.  Increasing ``npool`` distributes k-point work
        across MPI ranks and reduces per-process memory.
    smearing : str, optional
        Global smearing type for BZ integration (``None``, ``'m-p'``, or ``'gauss'``).
    save_overlaps : bool, default ``False``
        Retain the wavefunction overlap matrices ``Sks`` in the ``DataController``
        after Hamiltonian construction.  Required for ACBN0.
    acbn0 : bool, default ``False``
        Orthogonalise the PAO Hamiltonian using the ACBN0 procedure.
    verbose : bool, default ``False``
        Enable detailed debugging output.
    restart : bool, default ``False``
        Resume from a previously saved ``.json`` dump (see :meth:`restart_dump`).
    dft : str, default ``'QE'``
        DFT back-end: ``'QE'`` (Quantum ESPRESSO) or ``'VASP'``.

    Key attributes
    --------------
    data_controller : DataController
        Central data store; all arrays (``HRs``, ``Hks``, ``E_k``, …) and
        scalar attributes live in its ``data_arrays`` and ``data_attributes``
        dictionaries.
    comm, rank, size : MPI communicator and process identifiers.

    Methods — PAO Hamiltonian
    -------------------------
    projections(\**kw)
        Compute PAO projections from pseudopotential or all-electron basis sets,
        replacing ``projwfc.x``.
    read_atomic_proj_QE()
        Read pre-computed projections from QE's ``atomic_proj.xml``.
    projectability(pthr, shift)
        Identify and optionally shift low-projectability bands.
    pao_hamiltonian(shift_type, insulator, write_binary, expand_wedge, symmetrize, …)
        Build the real-space Hamiltonian ``H(R)`` (stored as ``HRs``) and the
        k-space Hamiltonian ``H(k)`` (stored as ``Hks``).
    add_external_fields(Efield, Bfield, HubbardU)
        Apply electric field, magnetic field, or Hubbard-U corrections to ``HRs``.
    write_Hamiltonian(fname)
        Dump ``HRs`` in the Z2Pack format.

    Methods — Hamiltonian manipulation
    ------------------------------------
    interpolated_hamiltonian(nfft1, nfft2, nfft3, reshift_Ef)
        Fourier-interpolate onto a denser k-grid via zero-padding, producing ``Hksp``.
    doubling_Hamiltonian(nx, ny, nz)
        Double the supercell in one or more directions.
    cutting_Hamiltonian(x, y, z)
        Trim periodic images from ``HRs`` along selected axes.
    adhoc_spin_orbit(naw, phi, theta, lambda_p, lambda_d, soc_strengh, soc_species)
        Add phenomenological spin-orbit coupling to ``HRs`` without a DFT+SOC run.

    Methods — Band structure and eigenvalues
    -----------------------------------------
    bands(ibrav, band_path, high_sym_points, spin_orbit, fname, nk)
        Compute and write the band structure along a high-symmetry path.
    pao_eigh(bval)
        Diagonalise ``Hksp`` to obtain eigenvalues ``E_k`` and eigenvectors ``v_k``.
    gradient_and_momenta(band_curvature)
        Compute ∇_k H (``dHksp``) and the momentum matrix ``pksp``.  Optionally
        compute the band-curvature tensor.
    adaptive_smearing(smearing, afac)
        Compute adaptive smearing widths ``deltakp`` for BZ integration.
    effective_mass(emin, emax, ne)
        Calculate effective mass tensor components along kx, ky, kz.

    Methods — Spectral and spatial quantities
    -------------------------------------------
    dos(do_dos, do_pdos, delta, emin, emax, ne)
        Total and projected density of states (supports adaptive smearing).
    density(nr1, nr2, nr3)
        Real-space electron density on a uniform grid.
    fermi_surface(fermi_up, fermi_dw)
        Extract the Fermi surface within an energy window.
    spin_texture(fermi_up, fermi_dw)
        Map the spin expectation value ⟨S⟩ across the Fermi surface.
    wave_function_projection(dimension)
        Site-projected wave-function weights.
    site_projected_bands(site_proj)
        Band structure weighted by on-site probability density.
    ipr(fname)
        Inverse participation ratio (IPR) of PAO eigenstates.
    doping(tmin, tmax, nt, delta, emin, emax, ne, doping_conc, core_electrons, fname)
        Chemical potential as a function of carrier doping and temperature.

    Methods — Topology and Berry physics
    ----------------------------------------
    topology(eff_mass, Berry, spin_Hall, spin_orbit, spol, ipol, jpol)
        Berry curvature, effective mass, and spin Berry curvature along the
        k-path, including Z2 invariant support.
    berry_phase(kspace_method, berry_path, high_sym_points, …)
        Berry/Zak phase via the discretized product formula (Resta 1994).
    anomalous_Hall(do_ac, emin, emax, fermi_up, fermi_dw, ne, delta, a_tensor)
        Anomalous Hall conductivity (and optionally magnetic circular dichroism).
    spin_Hall(twoD, do_ac, emin, emax, ne, delta, fermi_up, fermi_dw, s_tensor, …)
        Spin Hall conductivity and spin circular dichroism.
    rashba_edelstein(emin, emax, ne, temps, twoD, lt, st, …)
        Rashba–Edelstein (inverse spin galvanic effect) tensor.
    find_weyl_points(symmetrize, test_rad, search_grid)
        Locate Weyl nodes in the BZ via Chern-number integration.
    spin_operator(spin_orbit, sh_l, sh_j)
        Build the spin operator matrix ``Sj`` in the PAO basis.

    Methods — Transport
    --------------------
    transport(tmin, tmax, nt, emin, emax, ne, scattering_channels, …)
        Boltzmann semi-classical transport tensors (conductivity, Seebeck, thermal
        conductivity) over a temperature and energy grid.
    dielectric_tensor(delta, intrasmear, emin, emax, ne, d_tensor, degauss)
        Frequency-dependent real and imaginary parts of the dielectric tensor.
    jdos(delta, emin, emax, ne, jdos_smeartype)
        Joint density of states.

    Methods — Utilities
    --------------------
    print_data_keys()
        Print all keys stored in the DataController dictionaries.
    memory_check()
        Estimate peak memory usage in GBytes.
    restart_dump(fname_prefix)
        Pickle the current DataController state for later resumption.
    restart_load(fname_prefix)
        Restore a previously pickled DataController state.
    report_module_time(mname)
        Print wall-clock time elapsed since the last timer reset.
    finish_execution()
        Print total run time and (if verbose) aggregate memory usage across ranks.

    Notes
    -----
    - The ``DataController`` (``self.data_controller``) is the single source of
      truth for all intermediate results.  Its ``data_arrays`` dict holds NumPy
      arrays (e.g. ``HRs``, ``E_k``, ``pksp``); ``data_attributes`` holds scalar
      configuration values (e.g. ``nawf``, ``nspin``, ``nkpnts``).
    - Each public method guards its core computation in a ``try/except`` block and
      calls ``self.report_exception`` on failure.  Setting
      ``abort_on_exception = True`` in the DataController attributes causes the
      exception to be re-raised immediately.
    - MPI barriers are inserted automatically at the start and end of every
      timed module via :meth:`report_module_time`.
    """

    data_controller = None

    comm = rank = size = None

    start_time = reset_time = None

    # Overestimate factor for guessing memory requirements
    gb_fudge_factor = 4.0

    # Function container for ErrorHandler's method
    report_exception = None

    def print_data_keys(self):
        """
        Print's out the keys do the data_controller dictionaries, "arrays" and "attributes".
        Each stores the respective data required for the various calculation PAOFLOW can perform.

        Arguments:
            None

        Returns:
            None
        """
        if self.rank == 0:
            self.data_controller.print_data()
        self.comm.Barrier()

    def __init__(
        self,
        workpath='./',
        outputdir='output',
        inputfile=None,
        savedir=None,
        model=None,
        npool=1,
        smearing=None,
        save_overlaps=False,
        acbn0=False,
        verbose=False,
        restart=False,
        dft='QE',
    ):
        """
        Initialize the PAOFLOW class, either with a save directory with required QE output or with an xml inputfile
        Arguments:
            workpath (str): Path to the working directory
            outputdir (str): Name of the output directory (created in the working directory path)
            inputfile (str): (optional) Name of the xml inputfile
            savedir (str): QE .save directory
            model (dict): Dictionary with 'label' key and parameters to build Hamiltonian from TB model
            npool (int): The number of pools to use. Increasing npool may reduce memory requirements.
            smearing (str): Smearing type (None, m-p, gauss)
            save_overlaps (bool): If True the wavefunction overlaps will be saved to the DataController
            acbn0 (bool): If True the Hamiltonian will be Orthogonalized after construction
            verbose (bool): False supresses debugging output
            restart (bool): True if the run is being restarted from a .json data dump.
            dft (str): 'QE' or 'VASP'
        Returns:
            None
        """
        from time import time

        from mpi4py import MPI

        from .DataController import DataController
        from .utils.header import header

        # -------------------------------
        # Initialize Parallel Execution
        # -------------------------------
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # -----------------
        # Print Header
        # Initialize Time
        # --------------
        if self.rank == 0:
            header()
            self.start_time = self.reset_time = time()

        # Initialize Data Controller
        self.data_controller = DataController(
            workpath,
            outputdir,
            inputfile,
            model,
            savedir,
            npool,
            smearing,
            save_overlaps,
            acbn0,
            False,
            1.0e-6,
            verbose,
            restart,
            dft,
        )

        self.report_exception = self.data_controller.report_exception

        if not restart:
            # Data Attributes
            attr = self.data_controller.data_attributes

            attr['scipyfft'] = True

        # Report execution information
        if self.rank == 0:
            if restart:
                print('Run starting from Restart data.')
            else:
                if self.size == 1:
                    print('Serial execution')
                else:
                    print(
                        'Parallel execution on %d processors and %d pool'
                        % (self.size, attr['npool'])
                        + ('' if attr['npool'] == 1 else 's')
                    )

        # Do memory checks
        if model is None and not restart and self.rank == 0:
            gbyte = self.memory_check()
            print('Estimated maximum array size: %.2f GBytes\n' % (gbyte))

        self.report_module_time('Initialization')

    def report_module_time(self, mname):
        from time import time

        self.comm.Barrier()

        if self.rank == 0:
            # White spacing between module name and reported time
            spaces = 35
            lmn = len(mname)
            if len(mname) > spaces:
                print('DEBUG: Please use a shorter module tag.')
                self.comm.Abort()

            # Format string and print
            lms = spaces - lmn
            dt = time() - self.reset_time
            print('%s in: %s %8.3f sec' % (mname, lms * ' ', dt), flush=True)
            self.reset_time = time()

    def restart_dump(self, fname_prefix='paoflow_dump'):
        """
        Saves the necessary information to restart a PAOFLOW run from any step in calculation.

        Arguments:
            fname_prefix (str): Prefix of the filenames which will be written. Files are written to the directory housing the python script which instantiates PAOFLOW, unless otherwise specified in this argument.

        Returns:
            None
        """
        from pickle import HIGHEST_PROTOCOL, dump

        fname = fname_prefix + '_%d' % self.rank + '.json'

        arry, attr = self.data_controller.data_dicts()
        with open(fname, 'wb') as f:
            dump([arry, attr], f, HIGHEST_PROTOCOL)

        self.report_module_time('Restart DUMP')

    def restart_load(self, fname_prefix='paoflow_dump'):
        """
        Loads the previously dumped save files and populates the DataController with said data.

        Arguments:
            fname_prefix (str): Prefix of the filenames which will be written. Files are written to the directory housing the python script which instantiates PAOFLOW, unless otherwise specified in this argument.

        Returns:
            None
        """
        from os.path import exists
        from pickle import load

        fname = fname_prefix + '_%d' % self.rank + '.json'
        if not exists(fname):
            print('Restart file named %s does not exist.' % fname)
            raise OSError('File: %s not found.' % fname)

        arry, attr = None, None
        with open(fname, 'rb') as f:
            arry, attr = load(f)

        if self.size != attr['mpisize']:
            print('Restarted runs must use the same number of cores as the original run.')
            raise ValueError('Number of processors does not match that of the previous run.')

        self.data_controller.data_arrays = arry
        self.data_controller.data_attributes = attr

        self.report_module_time('Restart LOAD')

    def memory_check(self):
        """
        Estimate PAOFLOW's memory requirements with a "fudge factor" defined above

        Arguments:
            None

        Returns:
            gbyte (float): Estimated number of Gigabytes required in memory for PAOFLOW to run.
        """
        dxdydz = 3
        B_to_GB = 1.0e-9
        ff = self.gb_fudge_factor
        bytes_per_complex = 128 // 8
        arry, attr = self.data_controller.data_dicts()
        nd1, nd2, nd3 = attr['nk1'], attr['nk2'], attr['nk3']
        spins, num_wave_functions = attr['nspin'], attr['nawf']
        return (
            num_wave_functions**2
            * (nd1 * nd2 * nd3)
            * spins
            * dxdydz
            * bytes_per_complex
            * ff
            * B_to_GB
        )

    def finish_execution(self):
        """
        Finish the PAOFLOW execution. Print out run time and maximum memory usage

        Arguments:
            None

        Returns:
            None
        """
        import resource
        from time import time

        from mpi4py import MPI

        if self.rank == 0:
            tt = time() - self.start_time
            print('Total CPU time =%s%8.3f sec' % (25 * ' ', tt))

        verbose = self.data_controller.data_attributes['verbose']

        if verbose:
            # Add up memory usage from each core
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem = np.array([mem], dtype=float)
            mem0 = np.zeros(1, dtype=float) if self.rank == 0 else None
            self.comm.Reduce(mem, mem0, op=MPI.SUM, root=0)

            if self.rank == 0:
                print('Memory usage on rank 0:  %6.4f GB' % (mem[0] / 1024.0**2))
                print('Maximum concurrent memory usage:  %6.4f GB' % (mem0[0] / 1024.0**2))

    def projections(self, internal=False, basispath=None, configuration=None):
        """
        Calculate the projections on the atomic basis provided by the pseudopotential or
        on the all-electron internal basis sets.
        Replaces projwfc.

        Parameters
        ----------
        internal : bool, optional
            If ``True``, use the all-electron internal basis (loaded from
            ``basispath``) instead of the pseudopotential basis.  Always
            forced ``True`` for VASP calculations.
        basispath : str, optional
            Directory containing the per-element ``BASIS/<elem>/*.dat``
            files.  Required when ``internal`` is ``True`` or when
            ``configuration`` is a preset string.
        configuration : dict, str, or None, optional
            How to build the projection basis:

            * ``"minimal"`` — use the pseudo-atomic wavefunctions
              shipped in each species' UPF file (smooth, matches the
              default QE projwfc behaviour).  ``internal`` is ignored.
              Spans the valence bands well; conduction states need
              ``"standard"``, ``"extended"`` or an explicit
              configuration dict.
            * ``"standard"`` — AE basis built from ``basispath``: the
              minimal valence set augmented with (a) the next missing
              angular-momentum channel at ``nmax`` (e.g. ``3D`` for
              Si) and (b) ``(n+1)L`` for each occupied shell — see
              :func:`PAOFLOW.inputs.basis_presets.standard_augmentation`.
              Provides a moderate set of conduction states without the
              full ``"extended"`` polarization.  ``internal`` is
              ignored.
            * ``"extended"`` — AE basis built from ``basispath``: the
              UPF valence shells plus a generous rule-based set of
              polarization shells (see
              :func:`PAOFLOW.inputs.basis_presets.extended_augmentation`).
              ``internal`` is ignored.  Equivalent to the
              ``internal=True`` legacy path with an auto-generated
              configuration dict.
            * ``dict`` — explicit per-element mapping
              ``{element: spec}``.  Each ``spec`` may be a list of shell
              labels (``['3S', '3P', '4S']`` — used verbatim) or a preset
              name string (``'standard'``, ``'extended'`` …) resolved for
              that element.  This lets you ask for a curated preset on
              some species while hand-picking orbitals on others, e.g.
              ``{'Ga': 'standard', 'As': ['4S', '4P', '3D']}``.  Consumed
              by the AE-only builder (pass ``internal=True``).
              Backwards-compatible with previous releases.
            * ``None`` — keep whatever is already stored in
              ``arry['configuration']`` (legacy behaviour).
        """

        from .inputs.basis_presets import (
            resolve_configuration,
            resolve_configuration_dict,
        )
        from .projection.do_atwfc_proj import (
            build_aewfc_basis,
            build_pswfc_basis_all,
            calc_proj_k,
        )
        from .utils.communication import gather_array, load_balancing

        arry, attr = self.data_controller.data_dicts()

        if basispath is not None:
            attr['basispath'] = basispath

        preset = None
        if configuration is not None:
            if isinstance(configuration, str):
                preset = configuration.lower()
                arry['configuration'] = resolve_configuration(self.data_controller, configuration)
                if attr.get('verbose') and self.rank == 0:
                    print("Resolved configuration preset '%s':" % configuration)
                    for elem, shells in arry['configuration'].items():
                        print('  %-3s : %s' % (elem, ', '.join(shells)))
            elif isinstance(configuration, dict):
                # Per-element dict: each value may be an explicit list of
                # shells or a preset name ('standard', 'extended', ...), so
                # the user can ask for a curated preset on some species
                # while hand-picking orbitals on others.
                arry['configuration'] = resolve_configuration_dict(
                    self.data_controller, configuration
                )
                if attr.get('verbose') and self.rank == 0:
                    print('Resolved configuration dict:')
                    for elem, shells in arry['configuration'].items():
                        print('  %-3s : %s' % (elem, ', '.join(shells)))
            else:
                raise TypeError(
                    'configuration must be a dict, a preset string '
                    "('minimal', 'standard' or 'extended'), or None; got %r"
                    % type(configuration).__name__
                )

        # Dispatch to the correct basis builder.
        #   'minimal'  -> pseudo PSWFC from UPF (smooth, matches QE bands).
        #   'standard' -> AE basis from BASIS/ (valence + same-L next-n
        #                 polarization shells; ~2× minimal).
        #   'extended' -> AE basis from BASIS/ (valence + generous
        #                 rule-based polarization shells).
        # Presets override the ``internal`` flag because they imply a
        # specific scheme.
        if preset == 'minimal':
            basis, arry['shells'] = build_pswfc_basis_all(self.data_controller)
        elif preset in ('standard', 'extended'):
            basis, arry['shells'] = build_aewfc_basis(self.data_controller)
        elif internal or attr['dft'] == 'VASP':
            # Legacy AE-only path (explicit dict configuration).
            basis, arry['shells'] = build_aewfc_basis(self.data_controller)
        else:
            basis, arry['shells'] = build_pswfc_basis_all(self.data_controller)

        # Expose the per-orbital atomic-basis records (r, wfc, l, m, atom,
        # tau, label) so that downstream modules can reconstruct the
        # actual PAO radial functions used by the projection.  Required
        # by ``hamiltonian.nonlocal_velocity.load_pao_orbitals`` when the
        # production run uses an AE / extended basis that does not match
        # the UPF pswfc set.
        arry['atomic_basis'] = basis

        nkpnts = len(arry['kpnts'])
        nbnds = attr['nbnds']
        nspin = attr['nspin']
        natwfc = len(basis)
        attr['nawf'] = natwfc

        arry['Dnm'] = np.empty((attr['nawf'], attr['nawf'], 3))
        for i in range(3):
            for n in range(attr['nawf']):
                for m in range(attr['nawf']):
                    arry['Dnm'][n, m, i] = basis[n]['tau'][i] - basis[m]['tau'][i]

        ini_ik, end_ik = load_balancing(self.size, self.rank, nkpnts)
        Unewaux = np.zeros((end_ik - ini_ik, nbnds, natwfc, nspin), dtype=complex)
        for ispin in range(nspin):
            for ik in range(ini_ik, end_ik):
                Unewaux[ik - ini_ik, :, :, ispin] = calc_proj_k(
                    self.data_controller, basis, ik, ispin
                )

        Unew = np.zeros((nkpnts, nbnds, natwfc, nspin), dtype=complex) if self.rank == 0 else None
        gather_array(Unew, Unewaux)
        if self.rank == 0:
            Unew = np.moveaxis(Unew, 0, 2)
        Unew = self.comm.bcast(Unew, root=0)

        arry['U'] = Unew
        arry['basis'] = basis

        self.report_module_time('Projections')

    def read_atomic_proj_QE(self):
        """
        Read the wavefunctions and overlaps from atomic-proj.xml, written by Quantum Espresso
        in the .save directory specified in PAOFLOW's constructos. They are saved to the
        DataController's arrays dictionary with keys 'U' and 'Sks', respectively.
        """
        from os.path import exists, join

        from .inputs.read_upf import UPF
        from .projection.do_atwfc_proj import build_pswfc_basis_all

        arry, attr = self.data_controller.data_dicts()
        fpath = attr['fpath']
        if exists(join(fpath, 'atomic_proj.xml')):
            from .inputs.read_QE_xml import parse_qe_atomic_proj

            if attr['acbn0'] and not attr['save_overlaps']:
                if self.rank == 0:
                    print(
                        'WARNING: ACBN0 requires wavefunction overlaps. Setting save_overlaps to True.'
                    )
                attr['save_overlaps'] = True
            parse_qe_atomic_proj(self.data_controller, join(fpath, 'atomic_proj.xml'))
        else:
            raise Exception('atomic_proj.xml was not found.\n')

        arry['jchia'] = {}
        arry['shells'] = {}
        for at, pseudo in arry['species']:
            fname = join(attr['fpath'], pseudo)
            if exists(fname):
                upf = UPF(fname)
                arry['shells'][at] = upf.shells
                arry['jchia'][at] = upf.jchia
            else:
                raise Exception('Pseudopotential not found: %s' % fname)

        # Silencing verbose for a moment
        verbose_status = attr['verbose']
        if verbose_status == True:
            attr['verbose'] = False
        basis, _ = build_pswfc_basis_all(self.data_controller)
        # Restoring verbose
        if verbose_status == True:
            attr['verbose'] = True

        arry['Dnm'] = np.empty((attr['nawf'], attr['nawf'], 3))
        for i in range(3):
            for n in range(attr['nawf']):
                for m in range(attr['nawf']):
                    arry['Dnm'][n, m, i] = basis[n]['tau'][i] - basis[m]['tau'][i]

        arry['basis'] = basis

    def projectability(self, pthr=0.95, shift='auto'):
        """
        Calculate the Projectability Matrix to determine how many states need to be shifted

        Arguments:
            pthr (float): The minimum allowed projectability for high projectability states
            shift (str or float): If 'auto' the shift will be automatic. Otherwise the shift will be the value of 'shift'

        Returns:
            None
        """
        from .projection.do_projectability import do_projectability

        attr = self.data_controller.data_attributes

        if 'pthr' not in attr:
            attr['pthr'] = pthr
        if 'shift' not in attr:
            attr['shift'] = shift

        try:
            do_projectability(self.data_controller)
        except Exception as e:
            self.report_exception('projectability')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Projectability')

    def pao_hamiltonian(
        self,
        shift_type=1,
        insulator=False,
        write_binary=False,
        expand_wedge=True,
        symmetrize=False,
        thresh=1.0e-6,
        max_iter=16,
    ):
        """
        Construct the Tight Binding Hamiltonian
        Populates DataController with 'HRs', 'Hks' and 'kq_wght'

        Arguments:
            shift_type (int): Shift type [ 0-(PRB 2016), 1-(PRB 2013), 2-No Shift ]

        Returns:
            None

        """
        from .hamiltonian.do_build_pao_hamiltonian import (
            do_build_pao_hamiltonian,
            do_Hks_to_HRs,
        )
        from .utils.get_K_grid_fft import get_K_grid_fft

        # Data Attributes and Arrays
        arrays, attr = self.data_controller.data_dicts()

        if insulator:
            attr['insulator'] = True
        if 'shift_type' not in attr:
            attr['shift_type'] = shift_type
        if 'write_binary' not in attr:
            attr['write_binary'] = write_binary

        attr['symm_thresh'] = thresh
        attr['symmetrize'] = symmetrize
        attr['symm_max_iter'] = max_iter
        attr['expand_wedge'] = expand_wedge
        #  Skip open_grid when all k-points are included (no symmetry)
        #  Note expand_wedge is still required for VASP even not using symmetry.
        #  This is because we need find_equiv_k() in paosym to have the correct k-point ordering.

        if attr['symmetrize'] and attr['acbn0']:
            if self.rank == 0:
                print(
                    'WARNING: Non-ortho is currently not supported with pao_sym. Use nosym=.true., noinv=.true.'
                )

        try:
            do_build_pao_hamiltonian(self.data_controller)
            self.data_controller.broadcast_single_array('Hks')

        except Exception as e:
            self.report_exception('pao_hamiltonian')
            if attr['abort_on_exception']:
                raise e
        self.report_module_time('Building Hks')

        # Done with U and Sks
        del arrays['U']

        try:
            do_Hks_to_HRs(self.data_controller)

            ### PARALLELIZATION
            self.data_controller.broadcast_single_array('HRs')

            get_K_grid_fft(self.data_controller)
        except Exception as e:
            self.report_exception('pao_hamiltonian')
            if attr['abort_on_exception']:
                raise e
        self.report_module_time('k -> R')

    def minimal(self, first_band=None, R=False):
        from .projection.do_minimal import do_minimal

        raise Exception('ONLY FOR ARCHIVAL PUTPOSES - DO NOT USE')
        do_minimal(self.data_controller, first_band)
        if R:
            from .hamiltonian.do_build_pao_hamiltonian import do_Hks_to_HRs

            do_Hks_to_HRs(self.data_controller)
            self.data_controller.broadcast_single_array('HRs')

    def add_external_fields(self, Efield=[0.0], Bfield=[0.0], HubbardU=[0.0]):
        """
        Add External Fields and Corrections to the Hamiltonian 'HRs'

        Arguments:
            Efield (ndarray): 1D array of Efield values
            Bfield (ndarray): 1D array of Bfield values
            HubbardU (ndarray): 1D array of Hubbard Correctional values

        Returns:
            None

        """
        arry, attr = self.data_controller.data_dicts()

        if any(v != 0.0 for v in Efield):
            arry['Efield'] = np.array(Efield)
        if any(v != 0.0 for v in Bfield):
            arry['Bfield'] = np.array(Bfield)
        if any(v != 0.0 for v in HubbardU):
            arry['HubbardU'] = np.array(HubbardU)

        Efield, Bfield, HubbardU = arry['Efield'], arry['Bfield'], arry['HubbardU']

        try:
            # Add external fields or non scf ACBN0 correction
            if Efield.any() != 0.0 or Bfield.any() != 0.0 or HubbardU.any() != 0.0:
                from .hamiltonian.add_ext_field import add_ext_field

                add_ext_field(self.data_controller)
                if self.rank == 0 and attr['verbose']:
                    print('External Fields Added')
        except Exception as e:
            self.report_exception('add_external_fields')
            if attr['abort_on_exception']:
                raise e

        self.comm.Barrier()

    def write_Hamiltonian(self, fname='hamiltonian.dat'):
        """
        Write 'HRs' to file for use with Z2 Pack

        Arguments:
            fname (str): File name for the Hamiltonian in the Z2 Pack format

        Returns:
            None

        """
        try:
            self.data_controller.write_HRs(fname)
        except Exception as e:
            self.report_exception('write_Hamiltonian')
            if self.data_controller.data_attributes['abort_on_exception']:
                raise e
        self.report_module_time('write_Hamiltonian')

    def bands(
        self,
        ibrav=None,
        band_path=None,
        high_sym_points=None,
        spin_orbit=False,
        fname='bands',
        nk=500,
    ):
        """
        Compute the electronic band structure

        Arguments:
            ibrav (int): Crystal structure (following the specifications of QE)
            band_path (str): A string representing the band path to follow
            high_sym_points (dictionary): A dictionary with symbols of high symmetry points as keys and length 3 numpy arrays containg the location of the symmetry points as values.
            spin_orbit (bool): If True the calculation includes relativistic spin orbit coupling
            fname (str): File name for the band output
            nk (int): Number of k-points to include in the path (High Symmetry points are currently included twice, increasing nk)

        Returns:
            None

        """
        from .spectrum.do_bands import do_bands
        from .utils.communication import gather_full

        arrays, attr = self.data_controller.data_dicts()

        if ibrav is not None:
            attr['ibrav'] = ibrav

        if 'ibrav' not in attr and 'kq' not in arrays:
            if band_path is None or high_sym_points is None:
                if self.rank == 0:
                    print("Must specify the high-symmetry path, 'kq', or 'ibrav'")

        if 'nk' not in attr:
            attr['nk'] = nk
        if band_path is not None:
            attr['band_path'] = band_path
        if 'do_spin_orbit' not in attr:
            attr['do_spin_orbit'] = spin_orbit
        if high_sym_points is not None:
            arrays['high_sym_points'] = high_sym_points

        # Prepare HRs for band computation with spin-orbit coupling
        try:
            do_bands(self.data_controller)
            if self.rank == 0 and 'nkpnts' in attr and arrays['kq'].shape[1] == attr['nkpnts']:
                print('WARNING: The bands kpath and nscf calculations have the same size.')
                print(
                    "Spin Texture calculation should be performed after 'pao_eigh' to ensure integration across the entire BZ.\n"
                )

            E_kp = gather_full(arrays['E_k'], attr['npool'])
            self.data_controller.write_bands(fname, E_kp)
            E_kp = None
        except Exception as e:
            self.report_exception('bands')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Bands')

    def adhoc_spin_orbit(
        self,
        naw=[1],
        phi=0.0,
        theta=0.0,
        lambda_p=[0.0],
        lambda_d=[0.0],
        soc_strengh={},
        soc_species=True,
        soc_shell_weights=None,
    ):
        """
        Include spin-orbit coupling

        Arguments:
            theta (float)             :  Spin orbit angle
            phi (float)               :  Spin orbit azimuthal angle
            soc_strengh(dict)         :  p and d orbitals SOC strengh for each species
            If soc_species = False
            lambda_p (list of floats) :  p orbitals SOC strengh for each atom
            lambda_d (list of float)  :  d orbitals SOC strengh for each atom
            soc_shell_weights (dict, optional):
                Per-shell SOC weights for the ``'generic'`` builder
                (extended bases).  Keyed by species symbol, value is a
                list of bool or float of the same length as
                ``arry['shells'][species]``.  ``None`` (default) uses
                the builder's intrinsic ``1/k**2`` occurrence-index
                falloff for each l > 0 channel (valence weight 1,
                1st augmentation 0.25, 2nd 0.111, ...).

        Returns:
            None

        """
        import scipy.linalg as la

        from .hamiltonian.do_spin_orbit import do_spin_orbit_H

        arry, attr = self.data_controller.data_dicts()
        attr['do_spin_orbit'] = attr['adhoc_SO'] = True

        if 'phi' not in attr:
            attr['phi'] = phi
        if 'theta' not in attr:
            attr['theta'] = theta

        if soc_species == True:
            lambda_p = []
            lambda_d = []
            for i in range(len(arry['atoms'])):
                lambda_p.append(soc_strengh[arry['atoms'][i]][0])
                lambda_d.append(soc_strengh[arry['atoms'][i]][1])
            arry['lambda_p'] = lambda_p
            arry['lambda_d'] = lambda_d
        else:
            if 'lambda_p' not in arry:
                arry['lambda_p'] = lambda_p[:]
            if 'lambda_d' not in arry:
                arry['lambda_d'] = lambda_d[:]

        if soc_shell_weights is not None:
            arry['soc_shell_weights'] = soc_shell_weights

        self.data_controller.build_arrays_adhoc_soc()

        # Check if the pseudo potential or internal basis configuraton is implemented
        if len(arry['orb_pseudo']) == attr['natoms']:
            # add SOC
            do_spin_orbit_H(self.data_controller)
            # Rezising
            attr['bnd'] *= 2
            attr['dftSO'] = True
            attr['nspin'] = 1
            attr['nawf'] = arry['HRs'].shape[0]

            if 'Dnm' in arry:
                Dnm_double = np.empty((attr['nawf'], attr['nawf'], 3))
            for i in range(3):
                Dnm = arry['Dnm'][:, :, i]
                Dnm_double[:, :, i] = la.block_diag(*[Dnm, Dnm])
            arry['Dnm'] = Dnm_double
            Dnm_double = None
            Dnm = None

            # for write Hamiltonian
            if 'Hks' in arry:
                del arry['Hks']
            arry['Hks'] = np.fft.fftn(arry['HRs'], axes=(2, 3, 4))
        else:
            self.report_module_time('adhoc_spin_orbit')
            raise (
                NotImplementedError(
                    'Pseudo potential or internal basis configuration not implemented'
                )
            )

        self.report_module_time('adhoc_spin_orbit')

    def wave_function_projection(self, dimension=3):
        """
        Marcio, can you write something here please?
        Please check the argument descriptions also, I half guessed.

        Arguments:
            dimension (int): Dimensionality of the system

        Returns:
            None
        """
        from .topology.do_wave_function_site_projection import (
            wave_function_site_projection,
        )

        try:
            wave_function_site_projection(self.data_controller)
        except Exception as e:
            self.report_exception('wave_function_projection')
            if self.data_controller.data_attributes['abort_on_exception']:
                raise e

        self.report_module_time('wave_function_projection')

    def site_projected_bands(self, site_proj=[0]):
        """
        This routine calculates the wavefunction square wheights to produce a site projected band
        structure.

        Arguments:
            site_proj (list of ints): Incides of the atomic site to project the bands

        Returns:
            None
        """

        from .spectrum.do_site_projected_bands import site_projeted_bands

        arry, attr = self.data_controller.data_dicts()

        if 'site_proj' not in arry:
            arry['site_proj'] = site_proj

        if 'naw' not in arry:
            self.data_controller.build_arrays_adhoc_soc()

        try:
            site_projeted_bands(self.data_controller)
        except Exception as e:
            self.report_exception('site_projeted_bands')
            if self.data_controller.data_attributes['abort_on_exception']:
                raise e

        self.report_module_time('site_projeted_bands')

    def doubling_Hamiltonian(self, nx, ny, nz):
        """
        Marcio, can you write something here please?
        Please check the argument descriptions also

        Arguments:
            nx (bool): Number of doubles in first dimension
            ny (bool): Number of doubles in second dimension
            nz (bool): Number of doubles in third dimension

        Returns:
            None
        """
        from .hamiltonian.do_doubling import doubling_HRs

        arrays, attributes = self.data_controller.data_dicts()
        attributes['nx'], attributes['ny'], attributes['nz'] = nx, ny, nz

        try:
            if self.rank == 0:
                doubling_HRs(self.data_controller)

            array_list = [
                'HRs',
                'naw',
                'Dnm',
                'a_vectors',
                'tau',
                'atoms',
                'sh',
                'nl',
                'Sj',
                'lambda_p',
                'lambda_d',
                'orb_pseudo',
            ]

            for arry in array_list:
                has_key = self.comm.bcast((arry in arrays) if self.rank == 0 else None, root=0)
                if not has_key:
                    continue

                if self.rank == 0:
                    try:
                        arry_type = arrays[arry].dtype

                        if arry_type == 'float64':
                            bcast_mode = 'float'
                        elif arry_type == 'complex128':
                            bcast_mode = 'complex'
                        elif arry_type == 'int32':
                            bcast_mode = 'int'
                        else:
                            bcast_mode = 'list'
                    except AttributeError:
                        bcast_mode = 'list'
                else:
                    bcast_mode = None

                bcast_mode = self.comm.bcast(bcast_mode, root=0)
                if bcast_mode == 'float':
                    self.data_controller.broadcast_single_array(arry, dtype=float)
                elif bcast_mode == 'complex':
                    self.data_controller.broadcast_single_array(arry)
                elif bcast_mode == 'int':
                    self.data_controller.broadcast_single_array(arry, dtype=int)
                else:
                    self.data_controller.broadcast_single_list(arry)

            attr_list = [
                'nawf',
                'natoms',
                'nelec',
                'nbnds',
                'bnd',
                'omega',
            ]
            for attr in attr_list:
                if attr in attributes:
                    self.data_controller.broadcast_attribute(attr)

        except Exception as e:
            self.report_exception('doubling_Hamiltonian')
            if attributes['abort_on_exception']:
                raise e

        self.report_module_time('doubling_Hamiltonian')

    def cutting_Hamiltonian(self, x=False, y=False, z=False):
        """
        Marcio, can you write something here please?

        Arguments:
            x (bool): If True, cut along the first dimension
            y (bool): If True, cut along the second dimension
            z (bool): If True, cut along the third dimension

        Returns:
            None
        """
        arry, attr = self.data_controller.data_dicts()

        try:
            if x:
                for i in range(attr['nk1'] - 1, 0, -1):
                    arry['HRs'] = np.delete(arry['HRs'], i, 2)
            if y:
                for i in range(attr['nk2'] - 1, 0, -1):
                    arry['HRs'] = np.delete(arry['HRs'], i, 3)
            if z:
                for i in range(attr['nk3'] - 1, 0, -1):
                    arry['HRs'] = np.delete(arry['HRs'], i, 4)

            _, _, attr['nk1'], attr['nk2'], attr['nk3'], _ = arry['HRs'].shape
            attr['nkpnts'] = attr['nk1'] * attr['nk2'] * attr['nk3']
        except Exception as e:
            self.report_exception('cutting_Hamiltonian')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('cutting_Hamiltonian')

    def spin_operator(self, spin_orbit=False, sh_l=None, sh_j=None):
        """
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
        """
        arrays, attr = self.data_controller.data_dicts()

        if 'do_spin_orbit' not in attr:
            attr['do_spin_orbit'] = spin_orbit
        adhoc_SO = 'adhoc_SO' in attr and attr['adhoc_SO']

        if ('sh_l' not in arrays and 'sh_j' not in arrays) and not adhoc_SO:
            if sh_l is None and sh_j is None:
                sh = arrays['shells']
                shells, jchia = [], []
                for i, a in enumerate(arrays['atoms']):
                    ash = []
                    for v in sh[a]:
                        ash += [v, v] if v == 0 else [v]
                    shells += ash[::2]
                    for l in ash[::2]:
                        jchia += [0.5, 0.5] if l == 0 else [l - 0.5, l + 0.5]
                arrays['sh_j'] = np.array(jchia)[::2]
                arrays['sh_l'] = np.array(shells)
            else:
                arrays['sh_l'], arrays['sh_j'] = sh_l, sh_j
        try:
            nawf = attr['nawf']

            # Compute spin operators
            # Pauli matrices (x,y,z)
            Sj = np.zeros((3, nawf, nawf), dtype=complex)
            sP = 0.5 * np.array(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                    [[0.0, -1.0j], [1.0j, 0.0]],
                    [[1.0, 0.0], [0.0, -1.0]],
                ]
            )
            if spin_orbit:
                # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
                # for spol in range(3):
                #     if spol == 2:  # Sz
                #         for i in range(nawf // 2):
                #             Sj[spol, i, i] = sP[spol][0, 0]
                #             Sj[spol, i, i + 1] = sP[spol][0, 1]
                #         for i in range(nawf // 2, nawf):
                #             Sj[spol, i, i - 1] = sP[spol][1, 0]
                #             Sj[spol, i, i] = sP[spol][1, 1]
                #     else:  # Sx and Sy
                #         for i in range(nawf // 2):
                #             Sj[spol, i, i + (nawf // 2 - 1)] = sP[spol][0, 0]
                #             Sj[spol, i, i + 1 + (nawf // 2 - 1)] = sP[spol][0, 1]
                #         for i in range(nawf // 2, nawf):
                #             Sj[spol, i, i - 1 - (nawf // 2 - 1)] = sP[spol][1, 0]
                #             Sj[spol, i, i - (nawf // 2 - 1)] = sP[spol][1, 1]

                Sj = np.zeros((3, nawf, nawf), dtype=complex)
                sP = 0.5 * np.array(
                    [
                        [[0.0, 1.0], [1.0, 0.0]],
                        [[0.0, -1.0j], [1.0j, 0.0]],
                        [[1.0, 0.0], [0.0, -1.0]],
                    ]
                )
                for spol in range(3):
                    for i in range(nawf // 2):
                        i_up = i
                        i_dn = nawf // 2 + i
                        Sj[spol, i_up, i_up] = sP[spol][0, 0]
                        Sj[spol, i_up, i_dn] = sP[spol][0, 1]
                        Sj[spol, i_dn, i_up] = sP[spol][1, 0]
                        Sj[spol, i_dn, i_dn] = sP[spol][1, 1]
            else:
                from .topology.clebsch_gordan import clebsch_gordan

                # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
                for spol in range(3):
                    Sj[spol, :, :] = clebsch_gordan(nawf, arrays['sh_l'], arrays['sh_j'], spol)

            arrays['Sj'] = Sj
        except Exception as e:
            self.report_exception('spin_operator')
            if attr['abort_on_exception']:
                raise e

    def topology(
        self,
        eff_mass=False,
        Berry=False,
        spin_Hall=False,
        spin_orbit=False,
        spol=None,
        ipol=None,
        jpol=None,
    ):
        """
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
        """
        from .topology.do_topology import do_topology
        # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry
        # curvature operators along the path in the IBZ from do_topology_calc

        arrays, attr = self.data_controller.data_dicts()

        if 'Berry' not in attr:
            attr['Berry'] = Berry
        if 'eff_mass' not in attr:
            attr['eff_mass'] = eff_mass
        if 'spin_Hall' not in attr:
            attr['spin_Hall'] = spin_Hall
        if 'do_spin_orbit' not in attr:
            attr['do_spin_orbit'] = spin_orbit

        attr['spol'] = spol
        attr['ipol'] = ipol
        attr['jpol'] = jpol

        if attr['spol'] is None or attr['ipol'] is None or attr['jpol'] is None:
            if self.rank == 0:
                print("Must specify 'spol', 'ipol', and 'jpol'")
            quit()

        if spin_Hall and 'Sj' not in arrays:
            self.spin_operator(spin_orbit=attr['do_spin_orbit'])

        try:
            do_topology(self.data_controller)
        except Exception as e:
            self.report_exception('topology')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Band Topology')

        arrays.pop('R', None)
        arrays.pop('idx', None)
        arrays.pop('Rfft', None)
        arrays.pop('R_wght', None)

    def interpolated_hamiltonian(self, nfft1=0, nfft2=0, nfft3=0, reshift_Ef=False):
        """
        Calculate the interpolated Hamiltonian with the method of zero padding
        Populates DataController with 'Hksp'.

        Arguments:
            nfft1 (int): Desired size of the interpolated Hamiltonian's first dimension
            nfft2 (int): Desired size of the interpolated Hamiltonian's second dimension
            nfft3 (int): Desired size of the interpolated Hamiltonian's third dimension

        Returns:
            None
        """
        from .hamiltonian.do_double_grid import do_double_grid
        from .spectrum.do_Efermi import E_Fermi
        from .utils.communication import gather_scatter
        from .utils.get_K_grid_fft import get_K_grid_fft

        arrays, attr = self.data_controller.data_dicts()

        try:
            if 'HRs' not in arrays:
                raise KeyError('HRs')

            nawf = attr['nawf']
            nko1, nko2, nko3 = attr['nk1'], attr['nk2'], attr['nk3']

            # Automatically doubles grid in any direction with unspecified nfft value
            if nfft1 == 0:
                nfft1 = 2 * nko1
            if nfft2 == 0:
                nfft2 = 2 * nko2
            if nfft3 == 0:
                nfft3 = 2 * nko3

            attr['nfft1'], attr['nfft2'], attr['nfft3'] = nfft1, nfft2, nfft3

            # Adjust 'npool' if arrays exceed MPI maximum
            int_max = 2147483647
            temp_pool = int(
                np.ceil(
                    (float(nawf**2 * nfft1 * nfft2 * nfft3 * 3 * attr['nspin']) / float(int_max))
                )
            )
            if temp_pool > attr['npool']:
                if self.rank == 0:
                    print('Warning: %s too low. Setting npool to %s' % (attr['npool'], temp_pool))
                attr['npool'] = temp_pool

            # Fourier interpolation on extended grid (zero padding)
            do_double_grid(self.data_controller)
            snawf, _, _, _, nspin = arrays['Hksp'].shape
            arrays['Hksp'] = np.reshape(arrays['Hksp'], (snawf, attr['nkpnts'], nspin))
            arrays['Hksp'] = gather_scatter(arrays['Hksp'], 1, attr['npool'])

            snktot = arrays['Hksp'].shape[1]
            if reshift_Ef:
                Hksp = arrays['Hksp'].reshape((nawf, nawf, snktot, nspin))
                Ef = E_Fermi(Hksp, self.data_controller, parallel=True)
                dinds = np.diag_indices(nawf)
                Hksp[dinds[0], dinds[1]] -= Ef
                arrays['Hksp'] = np.moveaxis(Hksp, 2, 0)
            else:
                arrays['Hksp'] = np.reshape(
                    np.moveaxis(arrays['Hksp'], 0, 1), (snktot, nawf, nawf, nspin)
                )

            get_K_grid_fft(self.data_controller)

            # Report new memory requirements
            if self.rank == 0:
                gbyte = self.memory_check()
                if attr['verbose']:
                    print('Performing Fourier interpolation on a larger grid.')
                    print(
                        'd : nk -> nfft\n1 : %d -> %d\n2 : %d -> %d\n3 : %d -> %d'
                        % (nko1, nfft1, nko2, nfft2, nko3, nfft3)
                    )
                print('New estimated maximum array size: %.2f GBytes' % gbyte)

        except Exception as e:
            self.report_exception('interpolated_hamiltonian')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('R -> k with Zero Padding')

    def pao_eigh(self, bval=0):
        """
        Calculate the Eigen values and vectors of k-space Hamiltonian 'Hksp'
        Populates DataController with 'E_k' and 'v_k'

        Arguments:
            bval (int): Top valence band number (nelec/2) to correctly shift Eigenvalues

        Returns:
            None
        """
        from .spectrum.do_eigh import do_pao_eigh
        from .utils.communication import gather_full, scatter_full

        arrays, attr = self.data_controller.data_dicts()

        if 'bval' not in attr:
            attr['bval'] = bval

        # HRs and Hks are replaced with Hksp
        if 'HRs' in arrays:
            del arrays['HRs']

        try:
            if 'Hksp' not in arrays:
                if self.rank == 0:
                    nktot = attr['nkpnts']
                    nawf, _, nk1, nk2, nk3, nspin = arrays['Hks'].shape
                    arrays['Hks'] = np.moveaxis(
                        np.reshape(arrays['Hks'], (nawf, nawf, nktot, nspin), order='C'),
                        2,
                        0,
                    )
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
                    arrays['E_k'] -= np.amax(arrays['E_k'][:, attr['bval'], :])
                self.comm.Barrier()
                arrays['E_k'] = scatter_full(arrays['E_k'], attr['npool'])
        except Exception as e:
            self.report_exception('pao_eigh')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Eigenvalues')

    def gradient_and_momenta(
        self,
        band_curvature=False,
        nonlocal_velocity=None,
        nonlocal_velocity_inject=None,
        nonlocal_velocity_sign=None,
    ):
        """
        Calculate the gradient of the k-space Hamiltonian and momentum operator.

        Arguments:
          band_curvature (bool): also compute the band curvature.
          nonlocal_velocity (bool or None): enable the non-local
            pseudopotential velocity correction.  When ``None`` (default)
            the value falls back to ``attr['nonlocal_velocity']`` (False if
            unset), preserving the legacy DataController-driven behaviour.
            Pass ``True`` here to enable the correction directly from the
            call without touching the DataController.
          nonlocal_velocity_inject (bool or None): fold the correction into
            ``dHksp`` so downstream momenta/optics pick it up.  When ``None``
            it falls back to ``attr['nonlocal_velocity_inject']`` if set,
            otherwise defaults to the resolved ``nonlocal_velocity`` value
            (i.e. enabling the correction injects it by default; building
            without injecting is diagnostic-only).
          nonlocal_velocity_sign (int or None): injection sign convention.
            When ``None`` it falls back to ``attr['nonlocal_velocity_sign']``
            if set, otherwise the calibrated per-path default is used
            (+1 scalar / ad-hoc-SO, -1 fully-relativistic jm-kspace).

        Returns:
          None
        """
        import numpy as np

        from .hamiltonian.do_gradient import do_gradient
        from .hamiltonian.do_momentum import do_momentum
        from .utils.communication import gather_scatter

        arrays, attr = self.data_controller.data_dicts()

        try:
            snktot, nawf, _, nspin = arrays['Hksp'].shape

            for ik in range(snktot):
                for ispin in range(nspin):
                    # make sure Hksp is hermitian (it should be)
                    arrays['Hksp'][ik, :, :, ispin] = (
                        np.conj(arrays['Hksp'][ik, :, :, ispin].T) + arrays['Hksp'][ik, :, :, ispin]
                    ) / 2.0

            arrays['Hksp'] = np.reshape(arrays['Hksp'], (snktot, nawf**2, nspin))
            arrays['Hksp'] = np.moveaxis(gather_scatter(arrays['Hksp'], 1, attr['npool']), 0, 1)
            snawf, _, nspin = arrays['Hksp'].shape
            arrays['Hksp'] = np.reshape(
                arrays['Hksp'], (snawf, attr['nk1'], attr['nk2'], attr['nk3'], nspin)
            )

            do_gradient(self.data_controller)

            if not band_curvature:
                # No more need for k-space Hamiltonian
                del arrays['Hksp']

            ### PARALLELIZATION
            # gather dHksp on nawf*nawf and scatter on k points
            arrays['dHksp'] = np.reshape(arrays['dHksp'], (snawf, attr['nkpnts'], 3, nspin))
            arrays['dHksp'] = np.moveaxis(gather_scatter(arrays['dHksp'], 1, attr['npool']), 0, 2)
            arrays['dHksp'] = np.reshape(arrays['dHksp'], (snktot, 3, nawf, nawf, nspin), order='C')

            for nk in range(snktot):
                for i in range(3):
                    for s in range(nspin):
                        arrays['dHksp'][nk, i, :, :, s] = (
                            arrays['dHksp'][nk, i, :, :, s]
                            + np.conj(arrays['dHksp'][nk, i, :, :, s].T)
                        ) / 2.0
            if band_curvature:
                from .spectrum.do_band_curvature import do_band_curvature

                do_band_curvature(self.data_controller)
                # No more need for k-space Hamiltonian
                del arrays['Hksp']

        except Exception as e:
            self.report_exception('gradient_and_momenta')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Gradient')

        # ---- Optional non-local pseudopotential velocity correction ----
        # Enable via the ``nonlocal_velocity`` kwarg (preferred) or
        # ``attr['nonlocal_velocity'] = True`` (legacy).  Default off, so
        # this is a no-op for legacy runs.  See
        # ``TODOs/nonlocal_velocity_correction.md`` and
        # :mod:`PAOFLOW.hamiltonian.nonlocal_velocity` for the physics.
        # Explicit kwargs take precedence over the corresponding ``attr``
        # entries; the injection sign, when unset, resolves to the
        # calibrated per-path default (+1 scalar / ad-hoc-SO, -1
        # fully-relativistic jm-kspace) inside
        # :meth:`nonlocal_velocity_correction`.
        if nonlocal_velocity is None:
            nlv_enabled = bool(attr.get('nonlocal_velocity', False))
        else:
            nlv_enabled = bool(nonlocal_velocity)

        if nonlocal_velocity_inject is not None:
            nlv_inject = bool(nonlocal_velocity_inject)
        elif 'nonlocal_velocity_inject' in attr:
            nlv_inject = bool(attr['nonlocal_velocity_inject'])
        else:
            # Enabling the correction injects it by default; building Delta_p
            # without injecting is a diagnostic-only mode.
            nlv_inject = nlv_enabled

        if nonlocal_velocity_sign is not None:
            nlv_sign = int(nonlocal_velocity_sign)
        elif attr.get('nonlocal_velocity_sign', None) is not None:
            nlv_sign = int(attr['nonlocal_velocity_sign'])
        else:
            nlv_sign = None

        if nlv_enabled:
            # --- Preserve the BARE band group velocity for adaptive smearing ---
            # The Yates adaptive width uses |grad_k(E_n - E_m)|, taken from the
            # diagonal of the eigenbasis velocity matrix (the band group
            # velocity). By Hellmann-Feynman nabla_k E_n = <n|dH/dk|n> equals
            # that diagonal ONLY for the bare gradient dHksp. The non-local
            # velocity correction is an interband position-commutator term;
            # once it is folded into dHksp below, the diagonal of pksp is no
            # longer the bare group velocity, which inflates deltakp/deltakp2
            # and produces a spurious epsilon spike near omega -> 0. Build pksp
            # from the bare dHksp first and stash its diagonal so
            # do_adaptive_smearing can use the uncontaminated group velocity.
            do_momentum(self.data_controller)
            nb = arrays['pksp'].shape[2]
            bdiag = np.arange(nb)
            arrays['velkp_bare'] = np.ascontiguousarray(arrays['pksp'][:, :, bdiag, bdiag, :])

            self.nonlocal_velocity_correction(
                inject=nlv_inject,
                sign=nlv_sign,
            )

        ### DEV: Proposed to remove this and calculate pksp or velkp when required
        # Compute the momentum operator p_n,m(k) (and kinetic energy operator)
        do_momentum(self.data_controller)
        self.report_module_time('Momenta')

    def nonlocal_velocity_correction(
        self,
        *,
        q_max: float = 15.0,
        n_q: int = 300,
        pao_tol: float = 1.0e-3,
        units: str = 'rydberg',
        inject: bool = False,
        sign: int = None,
    ):
        """Assemble the non-local pseudopotential velocity correction.

        Computes the operator

        .. math::
           \\Delta p_\\alpha(\\mathbf{k})
             = \\frac{m}{i\\hbar}\\,\\sum_I\\!\\left[
                  P_I^{\\dagger}\\,D^I\\,P_I^{\\alpha}
                - (P_I^{\\alpha})^{\\dagger}\\,D^I\\,P_I
                \\right]

        in the PAO basis on the current FFT k-grid and stores it under
        ``arry['Delta_pksp']`` with shape ``(nktot, 3, nawf, nawf)``,
        complex.

        Parameters
        ----------
        q_max, n_q : float, int
            Radial Bessel transform quadrature parameters passed to
            :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.build_nl_real_space_tables`.
        pao_tol : float
            Cutoff tolerance used when enumerating the (β-site, PAO-site,
            :math:`\\Delta\\mathbf{R}`) pair list.
        units : {'hartree', 'rydberg'}
            Units of the raw operator stored in ``arry['Delta_pksp']``.
            ``'rydberg'`` (default) returns :math:`\\Delta p_\\alpha` in
            Ry/Bohr; ``'hartree'`` returns Ha/Bohr.
        inject : bool, optional
            **Experimental — sign provisional, awaiting integration
            calibration.**  When ``True`` the correction is additively
            applied to ``arry['dHksp']`` (which is in eV·Bohr) using

            .. math::
               \\Delta H_{\\mathrm{ksp}} \\mathrel{+}= -\\,\\lambda\\,
                  \\Delta p_\\alpha,

            with :math:`\\lambda = 13.605693122994` eV/Ry for ``'rydberg'``
            or :math:`2\\lambda` eV/Ha for ``'hartree'``.  The minus sign
            reflects PAOFLOW's internal convention
            ``pksp = -\\langle n|p|m\\rangle`` (confirmed by
            ``writers/write4bt2.py`` which exports ``mommat = -np.real(pksp)``
            and by the cubium Hellmann-Feynman test).  Pin in Phase 4
            will compare against ``epsilon.x`` on Si/Cu and may flip the
            sign; the conversion factor itself is exact.
        sign : int or None, optional
            ``+1``, ``-1``, or ``None`` (default).  When ``None`` the
            calibrated default for the active path is used: ``+1`` for the
            scalar / ad-hoc-SO path (calibrated against ``epsilon.x`` for
            Cu) and ``-1`` for the fully-relativistic ``(j, m_j)``-kspace
            path (the jm builder carries the opposite covariance/phase
            convention, validated against QE ``epsilon.x`` and SHC for
            Pt_REL).  Forwarded to
            :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.inject_into_dHksp`
            when ``inject=True``.  Also configurable via
            ``attr['nonlocal_velocity_sign']`` for the
            ``gradient_and_momenta`` gate; set it explicitly to override
            the per-path default.

        Notes
        -----
        Norm-conserving (KB) pseudopotentials only.  Caches the loaded
        :class:`~PAOFLOW.hamiltonian.nonlocal_velocity.BetaCatalog`,
        :class:`~PAOFLOW.hamiltonian.nonlocal_velocity.PAOCatalog`, and
        :class:`~PAOFLOW.hamiltonian.nonlocal_velocity.NLRealSpaceTables`
        on the data controller (keys ``'_NL_beta_catalog'``,
        ``'_NL_pao_catalog'``, ``'_NL_pairs'``, ``'_NL_tables'``) so a
        repeat call is cheap.
        """
        import numpy as np

        from .hamiltonian.nonlocal_velocity import (
            build_jm_transformation_matrix,
            build_nl_real_space_tables,
            compute_nonlocal_velocity_jm_on_grid,
            compute_nonlocal_velocity_on_grid,
            enumerate_nl_pairs,
            inject_into_dHksp,
            load_beta_projectors,
            load_pao_orbitals,
            rotate_dp_to_jm,
        )

        arry, attr = self.data_controller.data_dicts()
        import os as _os

        # Path selection for fully-relativistic UPFs (dftSO=True, not the
        # ad-hoc SOC path).  Default is Option A: build Delta_p directly in
        # the (j, m_j)-coupled basis used by dHksp, preserving the
        # j-dependence of the NL operator.  Two legacy paths remain
        # reachable for comparison:
        #   * NL_SCALAR_TILE=1  -- inject the scalar Delta_p
        #     block-diagonally in spin (the previous default; reproduces
        #     QE epsilon.x for cubic Pt_REL but is only an approximation).
        #   * NL_JM_ROTATION=1  -- Phase D scalar->jm spin-trace rotation
        #     (build_jm_transformation_matrix + rotate_dp_to_jm); breaks
        #     cubic isotropy, retained only for diagnostics.
        fully_rel = bool(attr.get('dftSO', False)) and not bool(attr.get('adhoc_SO', False))
        scalar_tile = _os.environ.get('NL_SCALAR_TILE') == '1'
        jm_rotation = _os.environ.get('NL_JM_ROTATION') == '1'
        jm_kspace = fully_rel and not scalar_tile and not jm_rotation

        # Resolve the calibrated default injection sign per path when the
        # caller left ``sign`` unset.  The fully-relativistic jm-kspace
        # builder carries the opposite covariance/phase convention from the
        # scalar path (see TODOs/nonlocal_velocity_correction.md), so it
        # needs ``-1`` to match QE epsilon.x and SHC for Pt_REL; the scalar
        # / ad-hoc-SO path keeps the Cu-calibrated ``+1``.
        if sign is None:
            sign = -1 if jm_kspace else +1
        sign = int(sign)
        try:
            if '_NL_beta_catalog' not in arry:
                arry['_NL_beta_catalog'] = load_beta_projectors(self.data_controller)
            if '_NL_pao_catalog' not in arry:
                arry['_NL_pao_catalog'] = load_pao_orbitals(self.data_controller)
            beta_cat = arry['_NL_beta_catalog']
            pao_cat = arry['_NL_pao_catalog']

            if '_NL_tables' not in arry:
                a_cart = np.asarray(arry['a_vectors']) * float(attr['alat'])
                pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=pao_tol)
                arry['_NL_pairs'] = pairs
                arry['_NL_tables'] = build_nl_real_space_tables(
                    beta_cat, pao_cat, pairs, q_max=q_max, n_q=n_q
                )
            tables = arry['_NL_tables']

            def _build_dP(kgrid_slice):
                if jm_kspace:
                    # Option A: Delta_p directly in the (j, m_j) basis
                    # (shape (nk, 3, n_rel, n_rel), n_rel = 2 * nawf_scalar).
                    return compute_nonlocal_velocity_jm_on_grid(
                        beta_cat,
                        pao_cat,
                        tables,
                        kgrid_slice,
                        float(attr['alat']),
                        units=units,
                    )
                return compute_nonlocal_velocity_on_grid(
                    beta_cat,
                    pao_cat,
                    tables,
                    kgrid_slice,
                    float(attr['alat']),
                    units=units,
                )

            # Under MPI, ``dHksp`` is k-scattered along axis 0 (see
            # :meth:`gradient_and_momenta`).  ``dP`` must end up partitioned
            # the same way via :func:`scatter_full`.  Rather than building the
            # whole grid serially on rank 0, each rank assembles a contiguous
            # slice of the global k-grid in parallel; the slices are gathered
            # on rank 0 (in global order, matching the un-scattered dHksp), and
            # then re-scattered with :func:`scatter_full` so the per-rank
            # layout matches dHksp exactly.  The k-grid assembly is independent
            # per k-point, so a sliced build is bit-identical to the full one.
            from .utils.communication import (
                gather_array,
                load_balancing,
                scatter_full,
            )

            if attr.get('mpisize', 1) > 1:
                kgrid = np.asarray(arry['kgrid'])
                nktot = kgrid.shape[1]
                ks, ke = load_balancing(self.size, self.rank, nktot)
                dP_part = np.ascontiguousarray(_build_dP(kgrid[:, ks:ke]))
                if self.rank == 0:
                    dP = np.zeros((nktot,) + dP_part.shape[1:], dtype=dP_part.dtype)
                else:
                    dP = None
                gather_array(dP, dP_part)
                if self.rank == 0:
                    dP_local = scatter_full(np.ascontiguousarray(dP), attr['npool'])
                else:
                    dP_local = scatter_full(None, attr['npool'])
            else:
                dP_local = _build_dP(arry['kgrid'])

            # Legacy Phase D: rotate the scalar Delta_p into the (j, m_j)
            # relativistic basis used by ``dHksp`` via a spin-trace (gated
            # behind NL_JM_ROTATION=1; the Option A jm-kspace builder above
            # is the default and does not use this path).
            if jm_rotation:
                if '_NL_jm_T' not in arry:
                    if 'atomic_basis' not in arry:
                        raise RuntimeError(
                            'nonlocal_velocity_correction: jm rotation requires '
                            "arry['atomic_basis'] (the relativistic basis); "
                            'run projections() before this method.'
                        )
                    basis_scalar = [
                        {
                            'atom': pao_cat.sites[o.site_index].label,
                            'label': o.label,
                            'l': o.l,
                        }
                        for o in pao_cat.basis
                    ]
                    arry['_NL_jm_T'] = build_jm_transformation_matrix(
                        arry['atomic_basis'], basis_scalar
                    )
                dP_local = rotate_dp_to_jm(dP_local, arry['_NL_jm_T'])

            arry['Delta_pksp'] = dP_local

            if inject:
                inject_into_dHksp(arry['dHksp'], dP_local, units=units, sign=sign)
        except Exception as e:
            self.report_exception('nonlocal_velocity_correction')
            if attr.get('abort_on_exception', True):
                raise e

        self.report_module_time('NL velocity correction')

    def adaptive_smearing(self, smearing='gauss', afac=None):
        """
        Calculate the Adaptive Smearing parameters
        Populates DataController with 'deltakp' and 'deltakp2'

        Arguments:
            smearing (str): Smearing type (m-p and gauss)
            afac   (float): Smearing control. Small values (<0.01) are equivalent to no smearing

        Returns:
            None
        """
        from .spectrum.do_adaptive_smearing import do_adaptive_smearing

        arrays, attr = self.data_controller.data_dicts()

        attr['smearing'] = smearing
        if smearing != 'gauss' and smearing != 'm-p':
            raise ValueError(
                "Smearing type %s not supported.\nSmearing types are 'gauss' and 'm-p'"
                % str(smearing)
            )
        try:
            do_adaptive_smearing(self.data_controller, smearing, afac)
        except Exception as e:
            self.report_exception('adaptive_smearing')
            if attr['abort_on_exception']:
                raise e
        self.report_module_time('Adaptive Smearing')

    def dos(self, do_dos=True, do_pdos=True, delta=0.01, emin=-10.0, emax=2.0, ne=1000):
        """
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
        """
        arrays, attr = self.data_controller.data_dicts()

        if 'smearing' not in attr:
            attr['smearing'] = None

        try:
            if attr['smearing'] is None:
                if do_dos:
                    from .spectrum.do_dos import do_dos

                    do_dos(self.data_controller, emin, emax, ne, delta)
                if do_pdos:
                    from .spectrum.do_pdos import do_pdos

                    do_pdos(self.data_controller, emin, emax, ne, delta)
            else:
                if 'deltakp' not in arrays:
                    if do_dos:
                        from .spectrum.do_dos import do_dos

                        do_dos(self.data_controller, emin, emax, ne, delta)
                    if do_pdos:
                        from .spectrum.do_pdos import do_pdos

                        do_pdos(self.data_controller, emin, emax, ne, delta)
                else:
                    if do_dos:
                        from .spectrum.do_dos import do_dos_adaptive

                        do_dos_adaptive(self.data_controller, emin, emax, ne)
                    if do_pdos:
                        from .spectrum.do_pdos import do_pdos_adaptive

                        do_pdos_adaptive(self.data_controller, emin, emax, ne)
        except Exception as e:
            self.report_exception('dos')
            if attr['abort_on_exception']:
                raise e

        mname = 'DoS%s' % (
            '' if attr['smearing'] is None or 'deltakp' not in arrays else ' (Adaptive Smearing)'
        )
        self.report_module_time(mname)

    def density(self, nr1=48, nr2=48, nr3=48):
        """
        Calculate the Electron Density in real space
        Arguments:
            nr1,nr2,nr3: real space grid

        Returns:
            None
        """
        from .hamiltonian.do_real_space import do_density

        do_density(self.data_controller, nr1, nr2, nr3)

        self.report_module_time('Density')

    def trim_non_projectable_bands(self):
        arrays, attributes = self.data_controller.data_dicts()

        bnd = attributes['nawf'] = attributes['bnd']

        arrays['E_k'] = arrays['E_k'][:, :bnd]
        arrays['pksp'] = arrays['pksp'][:, :, :bnd, :bnd]
        if 'deltakp' in arrays:
            arrays['deltakp'] = arrays['deltakp'][:, :bnd]
            arrays['deltakp2'] = arrays['deltakp2'][:, :bnd]

    def fermi_surface(self, fermi_up=1.0, fermi_dw=-1.0):
        """
        Calculate the Fermi Surface

        Arguments:
            fermi_up (float): The upper limit of the occupied energy range
            fermi_dw (float): The lower limit of the occupied energy range

        Returns:
            None
        """
        from .topology.do_fermisurf import do_fermisurf

        attr = self.data_controller.data_attributes

        if 'fermi_up' not in attr:
            attr['fermi_up'] = fermi_up
        if 'fermi_dw' not in attr:
            attr['fermi_dw'] = fermi_dw

        try:
            do_fermisurf(self.data_controller)
        except Exception as e:
            self.report_exception('fermi_surface')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Fermi Surface')

    def spin_texture(self, fermi_up=1.0, fermi_dw=-1.0):
        """
        Calculate the Spin Texture

        Arguments:
            fermi_up (float): The upper limit of the occupied energy range
            fermi_dw (float): The lower limit of the occupied energy range

        Returns:
            None
        """
        from .topology.do_spin_texture import do_spin_texture

        arry, attr = self.data_controller.data_dicts()

        if 'fermi_up' not in attr:
            attr['fermi_up'] = fermi_up
        if 'fermi_dw' not in attr:
            attr['fermi_dw'] = fermi_dw

        try:
            if attr['nspin'] == 1:
                if 'Sj' not in arry:
                    self.spin_operator()
                do_spin_texture(self.data_controller)
                self.report_module_time('Spin Texture')
            else:
                if self.rank == 0:
                    print('Cannot compute spin texture with nspin=2')
        except Exception as e:
            self.report_exception('spin_texture')
            if attr['abort_on_exception']:
                raise e

        self.comm.Barrier()

    def spin_Hall(
        self,
        twoD=False,
        do_ac=False,
        emin=-1.0,
        emax=1.0,
        ne=501,
        delta=0.05,
        fermi_up=1.0,
        fermi_dw=-1.0,
        s_tensor=None,
        shc_proj=None,
    ):
        """
        Calculate the Spin Hall Conductivity
          Currently this module does not possess the "spin_orbit" capability of do_topology, because I(Frank) do not know what this modification entails.
          Thus, do_spin_orbit defaults to False here. If anybody needs the spin_orbit capability here, please contact me and we'll sort it out.

        Arguments:
            twoD (bool): True to output in 2D units of Ohm^-1, neglecting the sample height in the z direction
            do_ac (bool): True to calculate the Spic Circular Dichroism
            emin (float): The minimum energy in the range
            emax (float): The maximum energy in the range
            ne (float): The number of energy increments
            delta (float) : small imaginary part added to the eigenvalue difference
            fermi_up (float): The upper limit of the occupied energy range
            fermi_dw (float): The lower limit of the occupied energy range
            s_tensor (list): List of tensor elements to calculate (e.g. To calculate xxx and zxy use [[0,0,0],[0,1,2]])

        Returns:
            None
        """
        from .projection.projection_operator import (
            do_projection_operator,
            orbital_array,
        )
        from .response.do_Hall import do_spin_Hall

        arrays, attr = self.data_controller.data_dicts()

        attr['eminH'], attr['emaxH'] = emin, emax
        attr['deltaH'] = delta
        attr['esizeH'] = ne

        if s_tensor is not None:
            arrays['s_tensor'] = np.array(s_tensor)
        if 'fermi_up' not in attr:
            attr['fermi_up'] = fermi_up
        if 'fermi_dw' not in attr:
            attr['fermi_dw'] = fermi_dw

        if shc_proj is not None:
            arrays['shc_proj'] = np.array(shc_proj)

        if 'Sj' not in arrays:
            self.spin_operator(spin_orbit=attr['do_spin_orbit'])

        try:
            if shc_proj == None:
                P = np.eye(attr['nawf'])
                do_spin_Hall(self.data_controller, twoD, do_ac, P)
            else:
                arrays['naw'] = orbital_array(self.data_controller)
                P = do_projection_operator(self.data_controller, arrays['shc_proj'])
                do_spin_Hall(self.data_controller, twoD, do_ac, P)

        except Exception as e:
            self.report_exception('spin_Hall')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Spin Hall Conductivity')

    def rashba_edelstein(
        self,
        emin=-2,
        emax=2,
        ne=500,
        temps=0.0,
        reg=1e-30,
        twoD=False,
        lt=1.0,
        st=1.0,
        write_to_file=True,
        delta=0.05,
    ):
        """
        Calculate the Rashba-Edelstein tensor
        Arguments:
            emin (float): The minimum energy in the range
            emax (float): The maximum energy in the range
            ne (float): The number of energy increments in [emin,emax]
            temps (float): Smearing temperature in eV
            reg (float): The regularization number that is applicable only for materials with band gap (in the order of 1e-30)
            twoD (bool): set True for two dimnesional materials
            lt (float): The lattice height of the twoD structure (in cm)
            st (float): The structure 'effectivce' thickness of the twoD structure (in cm)
            write_to_file (bool): Set True to write tensors to file

        Returns:
            None
        """
        from .response.do_rashba_edelstein import do_rashba_edelstein

        arrays, attr = self.data_controller.data_dicts()
        attr['deltaH'] = delta
        attr['esizeH'] = ne

        ene = np.linspace(emin, emax, ne)
        try:
            do_rashba_edelstein(self.data_controller, ene, temps, reg, twoD, lt, st, write_to_file)

        except Exception as e:
            self.report_exception('rashba_edelstein')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Rashba_Edelstein')

    def anomalous_Hall(
        self,
        do_ac=False,
        emin=-1.0,
        emax=1.0,
        fermi_up=1.0,
        fermi_dw=-1.0,
        ne=501,
        delta=0.05,
        a_tensor=None,
    ):
        """
        Calculate the Anomalous Hall Conductivity

        Arguments:
            do_ac (bool): True to calculate the Magnetic Circular Dichroism
            emin (float): The minimum energy in the range
            emax (float): The maximum energy in the range
            ne (float): The number of energy increments
            delta (float) : small imaginary part added to the eigenvalue difference
            fermi_up (float): The upper limit of the occupied energy range
            fermi_dw (float): The lower limit of the occupied energy range
            a_tensor (list): List of tensor elements to calculate (e.g. To calculate xx and yz use [[0,0],[1,2]])

        Returns:
            None
        """
        from .response.do_Hall import do_anomalous_Hall

        arrays, attr = self.data_controller.data_dicts()

        attr['eminH'] = emin
        attr['emaxH'] = emax
        attr['deltaH'] = delta
        attr['esizeH'] = ne

        if a_tensor is not None:
            arrays['a_tensor'] = np.array(a_tensor)
        if 'fermi_up' not in attr:
            attr['fermi_up'] = fermi_up
        if 'fermi_dw' not in attr:
            attr['fermi_dw'] = fermi_dw

        try:
            do_anomalous_Hall(self.data_controller, do_ac)
        except Exception as e:
            self.report_exception('anomalous_Hall')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Anomalous Hall Conductivity')

    def effective_mass(self, emin=-1.0, emax=1.0, ne=1000):
        """
        Calculate effective mass components for kx,ky and kz

        Arguments:
            tmin (float): The minimum temperature in the range
            tmax (float): The maximum temperature in the range
            nt (float): The number of temperature increments
            emin (float): The minimum energy in the range
            emax (float): The maximum energy in the range
            ne (float): The number of energy increments

        Returns:
            None
        """
        from .spectrum.do_effective_mass import do_effective_mass

        _, attr = self.data_controller.data_dicts()

        try:
            do_effective_mass(self.data_controller)

        except Exception as e:
            self.report_exception('effective_mass')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Effective mass')

    def doping(
        self,
        tmin=300,
        tmax=300,
        nt=1,
        delta=0.01,
        emin=-1.0,
        emax=1.0,
        ne=1000,
        doping_conc=0.0,
        core_electrons=0.0,
        fname='doping_',
    ):
        """
        Calculate the chemical potential that corresponds to specified doping for different temperatures

        Arguments:
            tmin (float): The minimum temperature in the range
            tmax (float): The maximum temperature in the range
            nt (float): The number of temperature increments
            emin (float): The minimum energy in the range
            emax (float): The maximum energy in the range
            ne (float): The number of energy increments
            doping_conc(float): The amount of doping in 1/cm^3. Positive value for p type and negative value for n type
            core_electrons(float): The number of core electrons in a system. Adding this to integrated dos allows for integration over a narrower energy window
            fname (str): Prefix for the filename containg Doping vs Temperature

        Returns:
            None
        """
        from .boltzmann.do_doping import do_doping
        from .spectrum.do_dos import do_dos, do_dos_adaptive

        arrays, attr = self.data_controller.data_dicts()

        if 'delta' not in attr:
            attr['delta'] = delta
        if 'doping_conc' not in attr:
            attr['doping_conc'] = doping_conc
        if 'core_electrons' not in attr:
            attr['core_electrons'] = core_electrons

        ene = np.linspace(emin, emax, ne)
        temps = np.linspace(tmin, tmax, nt)
        if attr['smearing'] == None:
            do_dos(self.data_controller, emin, emax, ne, delta)
        else:
            do_dos_adaptive(self.data_controller, emin, emax, ne, delta)
        do_doping(self.data_controller, temps, ene, fname)

        self.report_module_time('Doping')

    def transport(
        self,
        tmin=300.0,
        tmax=300.0,
        nt=1,
        emin=-2.0,
        emax=2.0,
        ne=500,
        scattering_channels=[],
        scattering_weights=[],
        tau_dict={},
        do_hall=False,
        write_to_file=True,
        save_tensors=False,
    ):
        """
        Calculate the Transport Properties

        Arguments:
            tmin (float): The minimum temperature in the range
            tmax (float): The maximum temperature in the range
            nt (float): The number of temperatures to evaluate in [tmin,tmax]
            emin (float): The minimum energy in the range
            emax (float): The maximum energy in the range
            ne (float): The number of energy increments in [emin,emax]
            fit (bool): fit paoflow output to experiments
            scattering_channels (list): List of strings specifying a built in model and/or TauModel objects to evaluate
            scattering_weights (list): List of coefficients weighting components of the harmonic sum when computing tau
            tau_dict (dict): Dictionary housing parameters required for calculating Tau with built-in models
            do_hall (bool): Set True to calculate hall coefficient
            write_to_file (bool): Set True to write tensors to file
            save_tensors (bool): Set True to save the tensors into the data controller

        Returns:
            None
        """
        from .boltzmann.do_transport import do_transport

        arrays, attr = self.data_controller.data_dicts()
        if 'tau_dict' not in attr:
            attr['tau_dict'] = tau_dict

        ene = np.linspace(emin, emax, ne)
        temps = np.linspace(tmin, tmax, nt)
        sc, sw = scattering_channels, scattering_weights
        try:
            bnd = attr['bnd']
            if 'pksp' in arrays:
                velkp = np.zeros((arrays['pksp'].shape[0], 3, bnd, attr['nspin']))
                for n in range(bnd):
                    velkp[:, :, n, :] = np.real(arrays['pksp'][:, :, n, n, :])
            elif 'velkp' in arrays:
                velkp = np.ascontiguousarray(arrays['velkp'][:, :, :bnd, :])

            do_transport(
                self.data_controller,
                temps,
                ene,
                velkp,
                sc,
                sw,
                do_hall,
                write_to_file,
                save_tensors,
            )

        except Exception as e:
            self.report_exception('transport')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Transport')

    def dielectric_tensor(
        self,
        delta=0.1,
        intrasmear=0.05,
        emin=0.0,
        emax=10.0,
        ne=501,
        d_tensor=None,
        degauss=0.1,
        emissivity=False,
        emis_angles=(0.0, 30.0, 60.0),
        emis_ntheta=90,
        emis_temperature=300.0,
    ):
        r"""Compute the frequency-dependent dielectric tensor.

        Evaluates :math:`\varepsilon_{\alpha\beta}(\omega) =
        \varepsilon_1 + i\varepsilon_2` in the independent-particle
        (RPA, no local-field) approximation using the Drude-Lorentz form
        of Quantum ESPRESSO's ``epsilon.x`` (Calandra & Mauri / QE
        manual, ``epsilon.x`` user guide, eq. 8):

        * Interband (:math:`n \neq n'`): prefactor :math:`8\pi e^2/(\Omega N_k m^2)`.
        * Intraband / Drude (metals only): prefactor :math:`4\pi e^2/(\Omega N_k m^2)`,
          i.e. one half of the interband prefactor. PAOFLOW applies a
          single common normalization built for the interband case and
          rescales the intraband contribution internally by ``1/2`` —
          see ``response/do_epsilon.py``.

        Outputs (written by ``data_controller``):

        * ``epsi_<a><b>.dat`` — :math:`\varepsilon_2(\omega)` (imag part)
        * ``epsr_<a><b>.dat`` — :math:`\varepsilon_1(\omega)` (real part)
        * ``ieps_<a><b>.dat`` — :math:`\varepsilon(i\omega)`, the Kramers-Kronig
          transform onto the imaginary frequency axis
        * ``eels_<a><b>.dat`` — :math:`-\mathrm{Im}\,\varepsilon^{-1}(\omega)`,
          electron energy-loss function. **Only written for diagonal pairs**
          (``ipol == jpol``); the EELS = :math:`\varepsilon_2/(\varepsilon_1^2+\varepsilon_2^2)`
          formula is not physically meaningful for off-diagonal components.
        * ``nref_<a><a>.dat``, ``kref_<a><a>.dat`` — real (refractive index)
          and imaginary (extinction coefficient) parts of the complex
          refractive index :math:`\tilde n = n + i\kappa`, with
          :math:`\tilde n^2 = \varepsilon`. **Diagonal pairs only**
          (Option A: per principal axis; off-diagonal complex permittivity
          requires tensor diagonalisation).
        * ``alpha_<a><a>.dat`` — absorption coefficient
          :math:`\alpha = 2\omega\kappa/c` (1/m). **Diagonal pairs only**.
        * ``refl_<a><a>.dat`` — normal-incidence reflectivity
          :math:`R = ((n-1)^2+\kappa^2)/((n+1)^2+\kappa^2)`. **Diagonal pairs only**.

        **Emissivity** (written only when ``emissivity=True``, diagonal pairs):

        * ``refl_th<deg>_<a><a>.dat`` — Fresnel directional reflectivity
          :math:`R(\theta, \omega)`, polarization-averaged over s and p
          waves, at each incidence angle in ``emis_angles``.
        * ``emis_th<deg>_<a><a>.dat`` — directional emissivity
          :math:`\varepsilon(\theta, \omega) = 1 - R(\theta, \omega)`
          (Kirchhoff's law, opaque medium).
        * ``emish_<a><a>.dat`` — spectral hemispherical emissivity
          :math:`\varepsilon(\omega) = 2\int_0^{\pi/2}\varepsilon(\theta,\omega)
          \cos\theta\sin\theta\,d\theta`.
        * ``emist_<a><a>.dat`` — total hemispherical emissivity
          :math:`\varepsilon(T)`, the Planck-weighted spectral average, as a
          two-column (temperature, emissivity) file.

        On rank 0 the f-sum-rule plasmon frequency
        :math:`\omega_p = \sqrt{(2/\pi)\int_0^{\omega_{\max}}\omega\varepsilon_2 d\omega}`
        is printed per diagonal component and can be compared against the
        equivalent value from ``epsilon.x``.

        Parameters
        ----------
        delta : float, optional
            Interband broadening :math:`\Gamma` (eV). QE input
            ``intersmear``. Default 0.1.
        intrasmear : float, optional
            Intraband Drude broadening :math:`\eta` (eV). QE input
            ``intrasmear``. Default 0.05.
        emin, emax : float, optional
            Frequency window (eV).
        ne : int, optional
            Number of frequency samples in ``[emin, emax]``.
        d_tensor : list or {'all', 'diag', 'offdiag'}, optional
            Tensor components to compute. ``'diag'`` -> ``[[0,0],[1,1],[2,2]]``,
            ``'offdiag'`` -> the six off-diagonal pairs, or pass an explicit
            list such as ``[[0,0],[1,2]]``.
        degauss : float, optional
            Fermi-Dirac smearing width (eV) used to evaluate occupations
            and (for metals) the Drude :math:`-\partial f/\partial E`
            delta-function approximation. Should match the QE SCF/NSCF
            smearing for direct benchmarks.
        emissivity : bool, optional
            If ``True``, additionally compute the Fresnel directional
            reflectivity, spectral hemispherical emissivity, and total
            hemispherical emissivity from the diagonal complex refractive
            index (see Outputs). Default ``False``.
        emis_angles : sequence of float, optional
            Incidence angles (degrees, relative to the surface normal) at
            which the directional reflectivity/emissivity are tabulated.
            Only used when ``emissivity=True``. Default ``(0, 30, 60)``.
        emis_ntheta : int, optional
            Number of polar-angle samples in :math:`[0, \pi/2]` for the
            hemispherical integral. Only used when ``emissivity=True``.
            Default 90.
        emis_temperature : float or sequence of float, optional
            Temperature(s) in kelvin at which the Planck-weighted total
            hemispherical emissivity is evaluated. Only used when
            ``emissivity=True``. Default 300 K.

            .. note::

               The total emissivity integral is taken over the supplied
               ``[emin, emax]`` grid only. The Planck weight peaks near
               :math:`k_B T`, so for a meaningful :math:`\varepsilon(T)` the
               window should start near zero and resolve the thermally
               relevant low-energy range; otherwise the truncated integral
               underestimates :math:`\varepsilon(T)`.

        Returns
        -------
        None
            Files are written through ``data_controller``.

        Notes
        -----
        Benchmarked against ``epsilon.x`` on Si (insulator) and Al fcc
        (metal); see ``examples/qe_examples/example15_Si_epsilon`` and
        ``example17_Al_epsilon``. Plasmon frequencies agree to within a
        few percent on a converged 16x16x16 k-grid.

        **Adaptive interband broadening** (Yates *et al.*, Phys. Rev. B
        **75**, 195121 (2007)): if :meth:`adaptive_smearing` has been called
        before this method (so the per-:math:`(k,n,m)` widths ``deltakp2``
        are present), the fixed scalar ``delta`` broadening is replaced by
        the local :math:`\eta_{nm}(\mathbf{k}) = \alpha\,
        |\nabla_k(E_n-E_m)|\,\delta k` in the interband Lorentzian. The width
        is floored by ``attr['adaptive_smearing_floor']`` (default: the
        frequency-grid spacing ``(emax-emin)/(ne-1)``) so every Lorentzian is
        at least one bin wide -- this removes single-point divergences where
        bands are locally parallel while keeping sharp van Hove singularities.
        Raise the floor (e.g. to the fixed ``delta``) for a smoother, purely
        additive broadening. The fixed-smearing path is used otherwise.
        Example::

            pf.gradient_and_momenta()
            pf.adaptive_smearing(smearing='gauss')   # builds deltakp/deltakp2
            pf.dielectric_tensor(d_tensor='diag')    # uses adaptive eta_nm
        """

        from .response.do_epsilon import do_dielectric_tensor

        arrays, attr = self.data_controller.data_dicts()

        if 'degauss' not in attr:
            attr['degauss'] = degauss
        if 'delta' not in attr:
            attr['delta'] = delta
        attr['intrasmear'] = intrasmear
        attr['emissivity'] = emissivity
        attr['emis_angles'] = np.atleast_1d(np.array(emis_angles, dtype=float))
        attr['emis_ntheta'] = int(emis_ntheta)
        attr['emis_temperature'] = np.atleast_1d(np.array(emis_temperature, dtype=float))
        if d_tensor == 'all':
            pass
        elif d_tensor == 'diag':
            arrays['d_tensor'] = np.array([[0, 0], [1, 1], [2, 2]])
        elif d_tensor == 'offdiag':
            arrays['d_tensor'] = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
        else:
            arrays['d_tensor'] = np.array(d_tensor)

        # -----------------------------------------------
        # Compute dielectric tensor (Re and Im epsilon)
        # -----------------------------------------------
        try:
            ene = np.linspace(emin, emax, ne)
            do_dielectric_tensor(self.data_controller, ene)
        except Exception as e:
            self.report_exception('dielectric_tensor')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Dielectric Tensor')

    def jdos(self, delta=0.1, emin=0.0, emax=10.0, ne=501, jdos_smeartype='gauss'):
        """
        Calculate the Dielectric Tensor

        Arguments:
            delta (float): broadening parameter in eV
            emin (float): The minimum value of energy
            emax (float): The maximum value of energy
            ne (float): Number of energy values between emin and emax
            jdos_smeartype: 'gauss' or "lorentz"

        Returns:
            None
        """
        from .response.do_epsilon import do_jdos

        _, attr = self.data_controller.data_dicts()
        if 'delta' not in attr:
            attr['delta'] = delta

        try:
            ene = np.linspace(emin, emax, ne)
            do_jdos(self.data_controller, ene, jdos_smeartype)
        except Exception as e:
            self.report_exception('joint density of states')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Joint density of states')

    def find_weyl_points(self, symmetrize=None, test_rad=0.01, search_grid=[8, 8, 8]):
        from .topology.do_find_Weyl import find_weyl

        try:
            if symmetrize is not None:
                self.data_controller.data_attributes['symmetrize'] = symmetrize
            find_weyl(self.data_controller, test_rad, search_grid)

        except Exception as e:
            self.report_exception('pao_hamiltonian')
            if self.data_controller.data_attributes['abort_on_exception']:
                raise e

        self.report_module_time('Weyl Search')

    def ipr(self, fname='ipr'):
        r"""
        Compute the inverse partiticipation ratio (IPR) from PAO eigenstates

                     \sum_n |v_nk|^4
        IPR_nk = -----------------------
                  ( \sum_n |v_nk|^2 )^2

        where n is the band index and k the k-point

        The final shape is (nspin,nkpts,nbands,3),
        where the last axis gives: 0 as the k-point coordinate,
                                   1 the band energy, and
                                   2 the inverse partition ratio

        The result in saved to a ipr.npy file.
        To open the file one should use:
        `ipr = np.load('ipr.npy')`

        """

        from os.path import join

        from .response.do_ipr import inverse_participation_ratio

        arry, attr = self.data_controller.data_dicts()

        try:
            arry['ipr'] = inverse_participation_ratio(self.data_controller)
            np.save(join(attr['opath'], fname + '.npy'), arry['ipr'])

        except Exception as e:
            self.report_exception('ipr')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Inverse Participation Ratio (IPR)')

    def berry_phase(
        self,
        kspace_method='path',
        berry_path=None,
        high_sym_points=None,
        kpath_funct=None,
        nk1=100,
        nk2=100,
        closed=True,
        method='berry',
        sub=None,
        occupied=True,
        kradius=None,
        kcenter=[0.0, 0.0, 0.0],
        kxlim=(-0.5, 0.5),
        kylim=(-0.5, 0.5),
        eigvals=False,
        fname='berry_phase',
        contin=False,
    ):
        """
        Compute the Berry phase using the discretized formula.
        See R. Resta, Rev. Mod. Phys. 66, 899 (1994) and R. Resta, J. Phys.: Condens. Matter 22, 123201 (2010)

        Arguments:
            kspace_method (str): method used to sample the BZ:
                                 *'path': 1D path along the BZ;
                                 *'track': 1D path along x direction for several points in the y direction;
                                 *'circle': circular path around a k-point, given center and radius;
                                 *'square': retangular region with nk1 points along x and nk2 points along y. Region defined given x and y start and end points or full BZ.
            berry_path (str): A string representing the band path to follow. The first and last k-point must not be the same.
            berry_high_sym_points (dictionary): A dictionary with symbols of high symmetry points as keys and length 3 numpy arrays containg the location of the symmetry points as values.
            nk (int): Number of k-points to include in the path
            closed (bool, optional): whether or not to include the connection of the last and first points in the loop
            method (str, {'berry','zak'}): 'berry' returns the usual berry phase. 'zak' includes the Zak phase for 1D systems which takes into account the Bloch factor exp(-iG.r)
                                            accumulated over a Brillouin zone. See J. Zak, Phys. Rev. Lett. 62, 2747 (1989)
            sub (None or list of int, optional): index of selected bands to calculate the Berry phase
            occupied (bool, optional): calculate the Berry phase over all occupied bands (if set to True, sub is set to None)
            kxlim (tuple, float): start and end points in x direction for sampling the BZ, used when kspace_method='square' .
            kylim (tuple, float): start and end points in y direction for sampling the BZ, used when kspace_method='square' .
            eigvals (bool, optional): write overlap matrix eigenvalues
            eigvecs (bool, optional): write overlap matrix eigenvectors
            save_prd (bool, optional): save overlap matrix
            fname (str, optional): filename to save the berry phase results
            contin (bool, optional): make berry phase continuous by fixing 2pi shifts

        Returns:
            Berry/Zak phase

        """
        from .topology.do_berry_phase import do_berry_phase

        arry, attr = self.data_controller.data_dicts()

        kspace_method = kspace_method.lower()
        attr['berry_kspace_method'] = kspace_method

        attr['berry_path'] = berry_path
        arry['berry_high_sym_points'] = high_sym_points

        attr['berry_nk'] = nk1
        attr['berry_nk1'] = nk1
        attr['berry_nk2'] = nk2

        attr['berry_kpath_funct'] = kpath_funct

        arry['berry_kxlim'] = kxlim
        arry['berry_kylim'] = kylim

        if kspace_method != 'square':
            attr['berry_eigvals'] = eigvals
        else:
            attr['berry_eigvals'] = False

        attr['berry_kradius'] = kradius
        arry['berry_kcenter'] = np.array(kcenter)
        attr['berry_eigvals'] = eigvals

        method = method.lower()
        if method in ['berry', 'zak']:
            attr['berry_method'] = method
        else:
            print("method should be either berry or zak. Falling back to method = 'berry'")
            attr['berry_method'] = 'berry'

        arry['berry_sub'] = sub
        if sub != None or (sub == None and not occupied):
            attr['berry_occupied'] = False
        else:
            attr['berry_occupied'] = occupied

        attr['berry_closed'] = closed
        attr['berry_contin'] = contin
        attr['berry_fname'] = fname

        try:
            do_berry_phase(self)

        except Exception as e:
            self.report_exception('berry_phase')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Berry phase')

    def phonon_setup(
        self,
        supercell_matrix,
        primitive_matrix=None,
        displacement_distance=0.01,
        q_mesh=None,
        q_path=None,
    ):
        """Initialise the phonopy interface (Stage 0: structure bridge).

        Converts the PAOFLOW structure into a ``phonopy`` unit cell, stores the
        phonon configuration on the ``DataController`` and creates the
        :class:`phonopy.Phonopy` object reused by subsequent phonon stages.

        Arguments:
            supercell_matrix: Supercell used for the finite-displacement
                force constants.  Accepts a scalar (isotropic diagonal),
                a length-3 sequence (anisotropic diagonal) or a 3x3 matrix.
            primitive_matrix (optional): Primitive-cell transformation passed
                to phonopy.  May be ``None``, ``'auto'`` or a 3x3 matrix.
            displacement_distance (float): Atomic displacement amplitude in
                Angstrom used to generate displaced supercells.
            q_mesh (optional): Default q-point mesh for DOS / thermal
                properties (used in later stages).
            q_path (optional): Default q-point path for the phonon dispersion
                (used in later stages).

        Returns:
            None
        """
        from .phonon.do_phonopy import init_phonopy

        arry, attr = self.data_controller.data_dicts()

        attr['phonon_supercell_matrix'] = supercell_matrix
        attr['phonon_primitive_matrix'] = primitive_matrix
        attr['phonon_displacement_distance'] = displacement_distance
        attr['phonon_q_mesh'] = q_mesh
        arry['phonon_q_path'] = q_path

        try:
            init_phonopy(self.data_controller)
        except Exception as e:
            self.report_exception('phonon_setup')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Phonon Setup')

    def phonons(
        self,
        supercell_matrix=None,
        primitive_matrix=None,
        displacement_distance=None,
        forces=None,
        phonon_dir='phonon',
        pp_dir=None,
        prefix=None,
        kgrid=None,
        ibrav=None,
        hubbard_file=None,
        hubbard_card=None,
        nac=False,
        born_file=None,
        born=None,
        dielectric=None,
        q_path=None,
        q_labels=None,
        q_npoints=101,
        mesh=None,
        do_bands=True,
        do_dos=True,
        do_thermal=False,
        t_min=0.0,
        t_max=1000.0,
        t_step=10.0,
        units='THz',
        fname='phonon',
    ):
        """Harmonic phonons via phonopy finite displacements (Stage 1).

        The routine operates in two phases:

        1. **Generate** (``forces=None``): build the phonopy object, create the
           displaced supercells and write a complete, ready-to-run Quantum
           ESPRESSO ``pw.x`` SCF input for each one
           (``<outputdir>/<phonon_dir>/supercell-NNN.in``).  Run ``pw.x`` on
           every input, then call this method again with ``forces=...``.

        2. **Analyse** (``forces`` provided): assemble the second-order force
           constants and compute the requested harmonic properties.

        Arguments:
            supercell_matrix: Supercell for the finite displacements (scalar,
                length-3 or 3x3).  Required on the first call; reused
                afterwards.
            primitive_matrix (optional): phonopy primitive transformation
                (``None`` -> identity, ``'auto'`` or a 3x3 matrix).
            displacement_distance (float, optional): Displacement amplitude in
                Bohr (default 0.01).
            forces: Force source for the analysis phase.  ``None`` -> only
                write the displaced-supercell inputs; ``'qe'`` -> harvest forces
                from ``supercell-NNN.out`` in ``phonon_dir``; a path string ->
                ingest an external ``FORCE_SETS``; an array
                ``(ndisp, natoms, 3)`` (Ry/au) -> use directly.
            phonon_dir (str): Sub-directory (under ``outputdir``) for the
                displaced supercells and ``FORCE_SETS``.
            pp_dir (str, optional): Pseudopotential directory written into the
                QE inputs (default: the DFT ``.save`` path).
            prefix (str, optional): QE ``prefix`` for the supercell runs.
            kgrid (optional): Explicit Monkhorst-Pack grid for the supercell
                (default: unit-cell grid scaled by the supercell multiplicity).
            ibrav (int, optional): Quantum ESPRESSO Bravais lattice index used
                to derive the default high-symmetry dispersion path when
                ``q_path`` is ``None`` (the QE ``.save`` does not record it).
            hubbard_file (str, optional): Path to a ``pw.x`` input whose
                new-style ``HUBBARD`` card is read and appended to every
                displaced-supercell input so the forces reflect the DFT+U
                electronic structure.  Only on-site ``U`` (manifold) parameters
                are kept; intersite ``V`` lines (cell-specific atom indices) are
                dropped.
            hubbard_card (str, optional): Explicit ``HUBBARD`` card text
                (overrides ``hubbard_file``).
            nac (bool): Apply the non-analytical term correction (LO-TO
                splitting near Gamma).  Requires ``born_file`` or both ``born``
                and ``dielectric``.
            born_file (str, optional): Path to a phonopy ``BORN`` file with the
                dielectric tensor and Born effective charges; takes precedence
                over ``born``/``dielectric`` when given.
            born (array_like, optional): Born effective charges
                ``(natom_prim, 3, 3)`` in units of the elementary charge.
            dielectric (array_like, optional): ``(3, 3)`` high-frequency
                dielectric tensor.
            q_path (optional): Dispersion path as a sequence of segments in
                fractional reciprocal coordinates; ``None`` -> path derived from
                ``ibrav`` (or automatic seekpath path if ``ibrav`` is unset).
            q_labels (optional): Tick labels matching ``q_path``.
            q_npoints (int): q-points per path segment.
            mesh (optional): q-mesh for DOS / thermal properties (default
                ``[20, 20, 20]``).
            do_bands, do_dos, do_thermal (bool): Properties to compute.
            t_min, t_max, t_step (float): Temperature grid (K) for thermal
                properties.
            units (str): Frequency units for outputs, ``'THz'`` or ``'cm-1'``.
            fname (str): Output filename prefix.

        Returns:
            None
        """
        from .phonon.do_phonopy import (
            attach_nac,
            compute_phonon_bands,
            compute_phonon_dos,
            compute_thermal_properties,
            generate_displacements,
            init_phonopy,
            produce_force_constants,
        )
        from .phonon.io import (
            harvest_qe_forces,
            ingest_force_sets,
            read_hubbard_card,
            write_displaced_supercells,
            write_force_sets,
        )

        arry, attr = self.data_controller.data_dicts()

        if supercell_matrix is not None:
            attr['phonon_supercell_matrix'] = supercell_matrix
        if primitive_matrix is not None:
            attr['phonon_primitive_matrix'] = primitive_matrix
        if displacement_distance is not None:
            attr['phonon_displacement_distance'] = displacement_distance
        if mesh is not None:
            attr['phonon_q_mesh'] = mesh
        if ibrav is not None:
            attr['ibrav'] = ibrav

        if hubbard_card is None and hubbard_file is not None:
            hubbard_card = read_hubbard_card(hubbard_file, include_v=False)

        try:
            # Deterministic rebuild: same structure + distance + supercell yield
            # the same displacement ordering, so a later analysis call lines up
            # with the inputs written in the generation phase.
            init_phonopy(self.data_controller)
            generate_displacements(self.data_controller)

            if forces is None:
                paths = write_displaced_supercells(
                    self.data_controller,
                    phonon_dir=phonon_dir,
                    pp_dir=pp_dir,
                    prefix=prefix,
                    kgrid=kgrid,
                    hubbard_card=hubbard_card,
                )
                if self.rank == 0:
                    print(
                        'Wrote %d displaced-supercell QE inputs. Run pw.x on each, '
                        "then re-call phonons(forces='qe')." % len(paths)
                    )
                self.report_module_time('Phonons (write inputs)')
                return

            if isinstance(forces, str) and forces.lower() == 'qe':
                harvest_qe_forces(self.data_controller, phonon_dir=phonon_dir)
                produce_force_constants(self.data_controller)
            elif isinstance(forces, str):
                ingest_force_sets(self.data_controller, forces)
                produce_force_constants(self.data_controller)
            else:
                produce_force_constants(self.data_controller, forces=forces)

            write_force_sets(self.data_controller, phonon_dir=phonon_dir)

            if nac:
                attach_nac(
                    self.data_controller,
                    born=born,
                    dielectric=dielectric,
                    born_file=born_file,
                )

            if do_bands:
                compute_phonon_bands(
                    self.data_controller,
                    q_path=q_path,
                    q_labels=q_labels,
                    npoints=q_npoints,
                    units=units,
                    fname=fname,
                )
            if do_dos:
                compute_phonon_dos(self.data_controller, mesh=mesh, units=units, fname=fname)
            if do_thermal:
                compute_thermal_properties(
                    self.data_controller,
                    mesh=mesh,
                    t_min=t_min,
                    t_max=t_max,
                    t_step=t_step,
                    fname=fname,
                )

        except Exception as e:
            self.report_exception('phonons')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Phonons')

    def born_charges(
        self,
        supercell_matrix=None,
        primitive_matrix=None,
        method='dfpt',
        forces=None,
        phonon_dir='phonon',
        pp_dir=None,
        prefix=None,
        outdir=None,
        kgrid=None,
        field_strength=0.001,
        nberrycyc=3,
        hubbard_file=None,
        hubbard_card=None,
        enforce_sum_rule=True,
        symmetrize=True,
    ):
        """Born effective charges and epsilon_inf (Stage 2b).

        Computes the macroscopic Born effective charge tensors and the
        high-frequency (clamped-ion) dielectric tensor, and writes a phonopy
        ``BORN`` file usable by :meth:`phonons` (``nac=True``, ``born_file=...``)
        for the LO-TO splitting.  Two back-ends are available:

        * ``method='dfpt'`` (default): a single Gamma-point ``ph.x`` run
          (``epsil=.true., trans=.false.``) gives both tensors directly.  Fast
          and accurate, but unavailable for DFT+U / hybrid functionals.
        * ``method='field'``: finite electric-field (``lelfield``) runs on the
          primitive cell (central differences of forces and polarization).
          Slower, but works whenever ``ph.x`` cannot be used.

        The routine operates in two phases:

        1. **Generate** (``forces=None``): build the phonopy object and write
           the QE input(s).  For ``method='dfpt'`` this is a single
           ``ph_epsil.in``; for ``method='field'`` it is seven primitive-cell
           ``field-*.in`` SCF inputs.  Run QE on the input(s), then re-call with
           ``forces='qe'``.

        2. **Analyse** (``forces='qe'``): parse the results, impose the acoustic
           sum rule, symmetrize, and write the ``BORN`` file.

        Arguments:
            supercell_matrix: Supercell used to build the phonopy object (only
                the primitive cell is used here); reused from a previous phonon
                call when omitted.
            primitive_matrix (optional): phonopy primitive transformation.
            method (str): ``'dfpt'`` (``ph.x``) or ``'field'`` (``lelfield``).
            forces: ``None`` -> write the QE input(s) only; ``'qe'`` -> harvest
                the QE output(s) and write ``BORN``.
            phonon_dir (str): Sub-directory (under ``outputdir``) for the QE
                runs and the ``BORN`` file.
            pp_dir (str, optional): Pseudopotential directory (``method='field'``
                QE inputs).
            prefix (str, optional): QE ``prefix``.  For ``method='dfpt'`` this
                defaults to the DFT ``.save`` prefix so ``ph.x`` reuses the
                existing self-consistent save.
            outdir (str, optional): QE ``outdir`` for ``method='dfpt'`` (default:
                the directory containing the DFT ``.save``).
            kgrid (optional): Monkhorst-Pack grid for the primitive cell
                (``method='field'``; default: the unit-cell grid).
            field_strength (float): ``efield_cart`` magnitude in QE atomic units
                (``method='field'``; 1 a.u. = 36.3609e10 V/m).
            nberrycyc (int): Berry-phase cycles per SCF step (``method='field'``).
            hubbard_file (str, optional): Path to a ``pw.x`` input whose
                new-style ``HUBBARD`` card is appended to the ``method='field'``
                QE inputs (on-site ``U`` only; intersite ``V`` dropped).  The
                ``method='dfpt'`` route instead reuses the Hubbard setup stored
                in the existing ``.save``.
            hubbard_card (str, optional): Explicit ``HUBBARD`` card text
                (overrides ``hubbard_file``).
            enforce_sum_rule (bool): Impose ``sum_k Z*_k = 0``.
            symmetrize (bool): Symmetrize the Born and dielectric tensors.

        Returns:
            None
        """
        from .phonon.do_born_charges import compute_born_and_epsilon
        from .phonon.do_phonopy import generate_displacements, init_phonopy
        from .phonon.io import read_hubbard_card, write_field_inputs, write_ph_epsil_input

        arry, attr = self.data_controller.data_dicts()

        method = str(method).lower()
        if method not in ('dfpt', 'field'):
            raise ValueError("born_charges method must be 'dfpt' or 'field'.")

        if supercell_matrix is not None:
            attr['phonon_supercell_matrix'] = supercell_matrix
        if primitive_matrix is not None:
            attr['phonon_primitive_matrix'] = primitive_matrix

        if hubbard_card is None and hubbard_file is not None:
            hubbard_card = read_hubbard_card(hubbard_file, include_v=False)

        try:
            init_phonopy(self.data_controller)
            generate_displacements(self.data_controller)

            if forces is None:
                if method == 'dfpt':
                    path = write_ph_epsil_input(
                        self.data_controller,
                        phonon_dir=phonon_dir,
                        prefix=prefix,
                        outdir=outdir,
                    )
                    if self.rank == 0:
                        print(
                            'Wrote ph.x input %s. Run ph.x on it, then re-call '
                            "born_charges(method='dfpt', forces='qe')." % path
                        )
                else:
                    paths = write_field_inputs(
                        self.data_controller,
                        field_strength=field_strength,
                        phonon_dir=phonon_dir,
                        pp_dir=pp_dir,
                        prefix=prefix,
                        kgrid=kgrid,
                        nberrycyc=nberrycyc,
                        hubbard_card=hubbard_card,
                    )
                    if self.rank == 0:
                        print(
                            'Wrote %d lelfield QE inputs. Run pw.x on each, then '
                            "re-call born_charges(method='field', forces='qe')." % len(paths)
                        )
                self.report_module_time('Born Charges (write inputs)')
                return

            if isinstance(forces, str) and forces.lower() == 'qe':
                compute_born_and_epsilon(
                    self.data_controller,
                    method=method,
                    phonon_dir=phonon_dir,
                    enforce_sum_rule=enforce_sum_rule,
                    symmetrize=symmetrize,
                    write_born=True,
                )
            else:
                raise ValueError(
                    "born_charges forces must be None (write inputs) or 'qe' "
                    '(harvest the QE output).'
                )

        except Exception as e:
            self.report_exception('born_charges')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Born Charges')

    def ir_spectrum(
        self,
        supercell_matrix=None,
        primitive_matrix=None,
        forces='qe',
        phonon_dir='phonon',
        born_file=None,
        born=None,
        dielectric=None,
        freq_min=None,
        freq_max=None,
        npoints=2000,
        gamma=4.0,
        units='cm-1',
        fname='phonon',
    ):
        """Infrared spectrum from Born charges and zone-centre eigenvectors (Stage 3).

        For each zone-centre mode the *mode effective charge vector*
        ``Zbar_v,a = sum_k,b Z*_k,a,b e_v,k,b / sqrt(M_k)`` is formed from the
        Born effective charges, the (mass-weighted) Gamma-point eigenvectors and
        the atomic masses; the IR intensity is ``I_v = sum_a |Zbar_v,a|^2``.  A
        Lorentzian broadening of each mode yields the continuous spectrum.  The
        transverse-optical eigenvectors at exactly Gamma are used, so the
        non-analytical (LO-TO) correction is not required.

        The harmonic force constants are rebuilt deterministically from the
        displaced-supercell forces (as in :meth:`phonons`), and the Born charges
        are taken from ``born``/``born_file`` or from a preceding
        :meth:`born_charges` call.

        Arguments:
            supercell_matrix: Supercell used to build the phonopy object;
                reused from a previous phonon call when omitted.
            primitive_matrix (optional): phonopy primitive transformation.
            forces: Force source for the harmonic force constants, as in
                :meth:`phonons` (``'qe'`` -> harvest ``supercell-NNN.out``; a
                path -> ingest ``FORCE_SETS``; an array -> use directly).
            phonon_dir (str): Sub-directory (under ``outputdir``) with the
                displaced supercells / ``FORCE_SETS`` and the ``BORN`` file.
            born_file (str, optional): Path to a phonopy ``BORN`` file with the
                Born effective charges; takes precedence over ``born``.
            born (array_like, optional): Born effective charges
                ``(natom_prim, 3, 3)`` in units of the elementary charge.
            dielectric (array_like, optional): ``(3, 3)`` high-frequency
                dielectric tensor (read alongside ``born_file``; unused by the
                oscillator strengths).
            freq_min, freq_max (float, optional): Frequency-axis limits of the
                broadened spectrum (in ``units``).
            npoints (int): Number of points on the broadened-spectrum grid.
            gamma (float): Lorentzian full width at half maximum (in ``units``).
            units (str): Frequency units for outputs, ``'cm-1'`` or ``'THz'``.
            fname (str): Output filename prefix; writes ``<fname>_ir_modes.dat``
                and ``<fname>_ir_spectrum.dat``.

        Returns:
            None
        """
        from .phonon.do_ir_raman import compute_ir_spectrum
        from .phonon.do_phonopy import (
            generate_displacements,
            init_phonopy,
            produce_force_constants,
        )
        from .phonon.io import harvest_qe_forces, ingest_force_sets

        arry, attr = self.data_controller.data_dicts()

        if supercell_matrix is not None:
            attr['phonon_supercell_matrix'] = supercell_matrix
        if primitive_matrix is not None:
            attr['phonon_primitive_matrix'] = primitive_matrix

        try:
            init_phonopy(self.data_controller)
            generate_displacements(self.data_controller)

            if isinstance(forces, str) and forces.lower() == 'qe':
                harvest_qe_forces(self.data_controller, phonon_dir=phonon_dir)
                produce_force_constants(self.data_controller)
            elif isinstance(forces, str):
                ingest_force_sets(self.data_controller, forces)
                produce_force_constants(self.data_controller)
            else:
                produce_force_constants(self.data_controller, forces=forces)

            compute_ir_spectrum(
                self.data_controller,
                born=born,
                dielectric=dielectric,
                born_file=born_file,
                freq_min=freq_min,
                freq_max=freq_max,
                npoints=npoints,
                gamma=gamma,
                units=units,
                fname=fname,
            )

        except Exception as e:
            self.report_exception('ir_spectrum')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('IR Spectrum')

    def _raman_cell_epsilon(
        self,
        savedir,
        workpath,
        basispath,
        configuration='extended',
        pthr=0.95,
        nonlocal_velocity=True,
        nfft=None,
        e_static=0.05,
        eps_outdir='eps',
        energies=None,
        lifetime=0.1,
        e_window=None,
        e_ne=None,
        return_static=False,
    ):
        """Dielectric tensor of one displaced cell via the PAO pipeline.

        Mirrors the standard PAOFLOW optical workflow (internal projections ->
        PAO Hamiltonian -> non-local velocity -> dielectric tensor) on the
        displaced-cell SCF save.

        When ``energies`` is ``None`` the static ``(3, 3)`` dielectric tensor
        (``epsilon_1`` at ``omega -> 0``) is returned (non-resonant Raman).
        Otherwise the **complex** dielectric tensor is evaluated on a frequency
        grid spanning ``energies`` (with a finite ``lifetime`` broadening) and
        the interpolated value at each requested photon energy (eV) is returned,
        stacked as ``(len(energies), 3, 3)`` (resonance Raman).  When
        ``return_static`` is ``True`` (only meaningful with ``energies``), the
        static tensor is harvested from the same grid (its ``omega -> 0`` row)
        and the method returns the tuple ``(stack, static)`` -- letting the
        ``method='all'`` workflow obtain both spectra from a single SCF/optics
        run per cell.
        """
        from .phonon.io import read_epsilon_at, read_static_epsilon

        _, attr = self.data_controller.data_dicts()
        pf = type(self)(
            workpath=workpath,
            outputdir=eps_outdir,
            savedir=savedir,
            smearing=attr.get('smearing', 'gauss'),
            npool=attr.get('npool', 1),
            verbose=False,
        )
        pf.projections(basispath=basispath, configuration=configuration)
        pf.projectability(pthr=pthr)
        pf.pao_hamiltonian()
        if nfft is not None:
            pf.interpolated_hamiltonian(nfft1=nfft[0], nfft2=nfft[1], nfft3=nfft[2])
        pf.pao_eigh()
        pf.gradient_and_momenta(nonlocal_velocity=nonlocal_velocity)
        pf.adaptive_smearing()

        d_tensor = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
        if energies is None:
            pf.dielectric_tensor(emin=0.0, emax=e_static, ne=2, d_tensor=d_tensor)
            _, pat = pf.data_controller.data_dicts()
            return read_static_epsilon(pat['opath'], nspin=int(pat.get('nspin', 1)))

        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        grid_max = float(e_window) if e_window else float(np.max(energies)) * 1.25 + 5.0 * lifetime
        grid_ne = (
            int(e_ne)
            if e_ne
            else max(201, int(np.ceil(grid_max / max(lifetime / 4.0, 1.0e-3))) + 1)
        )
        pf.dielectric_tensor(delta=lifetime, emin=0.0, emax=grid_max, ne=grid_ne, d_tensor=d_tensor)
        _, pat = pf.data_controller.data_dicts()
        nspin = int(pat.get('nspin', 1))
        stack = np.stack([read_epsilon_at(pat['opath'], energy=e, nspin=nspin) for e in energies])
        if return_static:
            return stack, read_static_epsilon(pat['opath'], nspin=nspin)
        return stack

    def raman_spectrum(
        self,
        supercell_matrix=None,
        primitive_matrix=None,
        forces='qe',
        phonon_dir='phonon',
        raman_dir='raman',
        delta=0.05,
        nbnd=None,
        basispath=None,
        configuration='extended',
        pthr=0.95,
        nonlocal_velocity=True,
        nfft=None,
        e_static=0.05,
        method='static',
        lifetime=0.1,
        e_window=None,
        e_ne=None,
        dielectric_callback=None,
        laser_nm=None,
        temperature=300.0,
        freq_min=None,
        freq_max=None,
        npoints=2000,
        gamma=4.0,
        units='cm-1',
        fname='phonon',
        generate=None,
    ):
        """Raman spectrum by finite differences of the dielectric tensor (Stage 4).

        Two flavours, selected with ``method``:

        * ``'static'`` (default) -- **non-resonant (Placzek)** Raman from the
          finite-difference derivative of the *static* dielectric tensor
          (``omega -> 0``).  The Raman tensor is real.
        * ``'resonance'`` -- **resonance Raman** from the derivative of the
          *complex* dielectric tensor evaluated at the laser frequency
          ``omega_L`` (set by ``laser_nm``).  In the independent-particle limit
          this is the Albrecht A+B sum (Franck-Condon and Herzberg-Teller
          captured together, since the finite difference differentiates the
          transition energies *and* the transition dipoles), with resonance
          enhancement as ``omega_L`` approaches the interband transitions.  The
          Raman tensor is complex and intensities use ``45|a|^2 + 7 gamma^2``.

        Two-phase workflow (identical for both flavours):

        * **generate** -- displace the primitive cell by ``+/-delta`` along each
          optical zone-centre eigenvector and write a ready-to-run ``pw.x`` SCF
          input per displacement under
          ``<raman_dir>/mode-NNN-{plus,minus}/<prefix>.scf.in``.  Run these SCF
          calculations (any tool); no NSCF/projwfc are needed.
        * **analyse** -- for every displaced cell run the PAOFLOW optical
          pipeline (internal projections -> dielectric tensor) to obtain the
          dielectric tensor (static, or at ``omega_L`` for resonance), build the
          Raman tensor ``R^v = (eps(+delta) - eps(-delta)) / (2 delta)`` per
          mode, and form the orientationally-averaged (powder) Stokes
          intensities.

        The phase is chosen automatically (``generate=None``): *analyse* when
        every displaced-cell SCF save is present, otherwise *generate*.  Pass
        ``generate=True``/``False`` to force a phase.

        Arguments:
            supercell_matrix, primitive_matrix: As in :meth:`phonons`; reused
                from a previous call when omitted.
            forces: Force source for the harmonic force constants, as in
                :meth:`phonons` (used to obtain the zone-centre eigenvectors).
            phonon_dir (str): Sub-directory with the displaced supercells /
                ``FORCE_SETS``.
            raman_dir (str): Sub-directory for the displaced primitive cells and
                their dielectric outputs.
            delta (float): Mass-weighted normal-coordinate displacement
                amplitude (Bohr*sqrt(amu) convention; the same value is used to
                finite-difference).
            nbnd (int, optional): Bands for the displaced SCF inputs (empty
                states are needed for the optical response).
            basispath (str, optional): Pseudo-atomic basis directory for the
                internal PAO projections in the analyse phase.
            configuration (str): PAO basis configuration (default ``'extended'``).
            pthr (float): Projectability threshold for the analyse phase.
            nonlocal_velocity (bool): Use the non-local velocity correction in
                the dielectric tensor (recommended).
            nfft (tuple, optional): Double-grid interpolation ``(n1, n2, n3)``.
            e_static (float): Upper energy (eV) of the 2-point grid used to read
                the static dielectric tensor (``epsilon_1`` at ``omega -> 0``;
                ``method='static'`` only).
            method (str): ``'static'`` (non-resonant), ``'resonance'`` (at the
                laser frequency), or ``'all'`` (both, harvested from a single
                optics run per displaced cell; the static spectrum is written to
                ``<fname>_static_raman_*.dat`` and each laser to
                ``<fname>_<nm>nm_raman_*.dat``).
            lifetime (float): Lorentzian lifetime broadening (eV) of the complex
                dielectric tensor in the resonance case (the ``delta`` of
                :meth:`dielectric_tensor`); keeps the response finite on
                resonance.
            e_window (float, optional): Upper energy (eV) of the per-cell
                dielectric grid in the resonance case (default: just above the
                largest laser energy).
            e_ne (int, optional): Number of grid points for the per-cell
                dielectric grid in the resonance case (default: chosen so the
                spacing resolves ``lifetime``).
            dielectric_callback (callable, optional): ``f(savedir, workpath) ->
                ndarray`` returning the dielectric tensor of a displaced cell --
                ``(3, 3)`` for ``method='static'`` or
                ``(len(laser_nm), 3, 3)`` (complex) for ``method='resonance'``.
                Overrides the built-in PAO pipeline.
            laser_nm (float or sequence, optional): Excitation wavelength(s) in
                nm.  Required for ``method='resonance'`` (sets ``omega_L`` and
                enables the ``(omega_L - omega_v)^4`` prefactor).  A sequence
                produces one spectrum per wavelength (a Raman *excitation
                profile*), each written to ``<fname>_<nm>nm_raman_*.dat``.
            temperature (float): Temperature (K) for the Bose ``(n+1)`` factor.
            freq_min, freq_max, npoints, gamma, units, fname: Spectrum grid and
                output options, as in :meth:`ir_spectrum`.  Writes
                ``<fname>_raman_modes.dat`` and ``<fname>_raman_spectrum.dat``.

        Returns:
            None
        """
        from .phonon.do_ir_raman import compute_raman_spectrum
        from .phonon.do_phonopy import (
            generate_displacements,
            init_phonopy,
            produce_force_constants,
        )
        from .phonon.io import (
            harvest_qe_forces,
            ingest_force_sets,
            raman_cell_dirs,
            write_raman_displaced_inputs,
        )

        arry, attr = self.data_controller.data_dicts()

        if supercell_matrix is not None:
            attr['phonon_supercell_matrix'] = supercell_matrix
        if primitive_matrix is not None:
            attr['phonon_primitive_matrix'] = primitive_matrix

        try:
            init_phonopy(self.data_controller)
            generate_displacements(self.data_controller)

            if isinstance(forces, str) and forces.lower() == 'qe':
                harvest_qe_forces(self.data_controller, phonon_dir=phonon_dir)
                produce_force_constants(self.data_controller)
            elif isinstance(forces, str):
                ingest_force_sets(self.data_controller, forces)
                produce_force_constants(self.data_controller)
            else:
                produce_force_constants(self.data_controller, forces=forces)

            # Decide phase: analyse when every displaced-cell save is present.
            run_analyse = generate is False
            if generate is None:
                try:
                    _, entries = raman_cell_dirs(self.data_controller, raman_dir=raman_dir)
                    run_analyse = bool(entries) and all(
                        os.path.isdir(save) for _, _, _, save in entries
                    )
                except FileNotFoundError:
                    run_analyse = False
            elif generate is True:
                run_analyse = False
            else:
                run_analyse = True

            if not run_analyse:
                write_raman_displaced_inputs(
                    self.data_controller,
                    delta,
                    raman_dir=raman_dir,
                    nbnd=nbnd,
                )
                if self.rank == 0 and attr.get('verbose', False):
                    print(
                        'Raman: displaced-cell SCF inputs written; run them, then re-run to analyse.'
                    )
                self.report_module_time('Raman Spectrum')
                return

            # --- analyse phase -------------------------------------------------
            phonon = arry['phonopy']
            nmodes = 3 * len(phonon.primitive)
            optical, entries = raman_cell_dirs(self.data_controller, raman_dir=raman_dir)

            # Laser wavelength(s) / photon energies for the resonance harvest.
            method_l = str(method).lower()
            if method_l not in ('static', 'resonance', 'all'):
                raise ValueError(
                    "raman_spectrum: method must be 'static', 'resonance' or 'all' "
                    '(got %r).' % method
                )
            want_static = method_l in ('static', 'all')
            want_resonance = method_l in ('resonance', 'all')

            if want_resonance:
                if laser_nm is None:
                    raise ValueError(
                        'raman_spectrum(method=%r) requires laser_nm=... '
                        '(the excitation wavelength(s) in nm).' % method_l
                    )
                laser_list = [float(x) for x in np.atleast_1d(laser_nm)]
                # E(eV) = h c / lambda, with h c = 1239.841984 eV*nm.
                res_energies = [1239.841984 / nm for nm in laser_list]
            else:
                laser_list = []
                res_energies = []

            if dielectric_callback is not None and method_l == 'all':
                raise ValueError(
                    'raman_spectrum: dielectric_callback is not supported with '
                    "method='all'; call method='static' and method='resonance' "
                    'separately with your callback instead.'
                )

            # One output channel per spectrum to produce.  Each carries its own
            # +/- dielectric arrays, the laser used for the Stokes prefactor and
            # the output basename.
            channels = []
            static_channel = None
            if want_static:
                # Bare Placzek for 'all'; pass the (scalar) laser through for a
                # plain static run to preserve the historical behaviour.
                static_laser = laser_nm if method_l == 'static' else None
                static_channel = {
                    'fname': fname + '_static' if method_l == 'all' else fname,
                    'laser': static_laser,
                    'plus': np.zeros((nmodes, 3, 3), dtype=float),
                    'minus': np.zeros((nmodes, 3, 3), dtype=float),
                }
                channels.append(static_channel)

            res_channels = []
            multi = method_l == 'all' or len(laser_list) > 1
            for i, nm in enumerate(laser_list):
                ch = {
                    'fname': '%s_%dnm' % (fname, int(round(nm))) if multi else fname,
                    'laser': nm,
                    'plus': np.zeros((nmodes, 3, 3), dtype=complex),
                    'minus': np.zeros((nmodes, 3, 3), dtype=complex),
                    'index': i,
                }
                channels.append(ch)
                res_channels.append(ch)

            computed = np.zeros(nmodes, dtype=bool)
            for v, sign, cell_dir, save in entries:
                res_stack = None
                stat = None
                if dielectric_callback is not None:
                    out = np.asarray(dielectric_callback(save, cell_dir))
                    if want_resonance:
                        res_stack = out[None, ...] if out.ndim == 2 else out
                    else:
                        stat = np.real(out)
                elif want_resonance:
                    if basispath is None:
                        raise ValueError(
                            'raman_spectrum analyse phase needs basispath=... '
                            '(or a dielectric_callback) for the internal PAO projections.'
                        )
                    out = self._raman_cell_epsilon(
                        save,
                        cell_dir,
                        basispath,
                        configuration=configuration,
                        pthr=pthr,
                        nonlocal_velocity=nonlocal_velocity,
                        nfft=nfft,
                        energies=res_energies,
                        lifetime=lifetime,
                        e_window=e_window,
                        e_ne=e_ne,
                        return_static=want_static,
                    )
                    if want_static:
                        res_stack, stat = out
                    else:
                        res_stack = out
                else:
                    if basispath is None:
                        raise ValueError(
                            'raman_spectrum analyse phase needs basispath=... '
                            '(or a dielectric_callback) for the internal PAO projections.'
                        )
                    stat = self._raman_cell_epsilon(
                        save,
                        cell_dir,
                        basispath,
                        configuration=configuration,
                        pthr=pthr,
                        nonlocal_velocity=nonlocal_velocity,
                        nfft=nfft,
                        e_static=e_static,
                    )

                key = 'plus' if sign == '+' else 'minus'
                if static_channel is not None and stat is not None:
                    static_channel[key][v] = np.real(stat)
                if res_stack is not None:
                    res_stack = np.asarray(res_stack)
                    for ch in res_channels:
                        ch[key][v] = res_stack[ch['index']]
                computed[v] = True

            for ch in channels:
                compute_raman_spectrum(
                    self.data_controller,
                    ch['plus'],
                    ch['minus'],
                    delta,
                    computed=computed,
                    laser_nm=ch['laser'],
                    temperature=temperature,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    npoints=npoints,
                    gamma=gamma,
                    units=units,
                    fname=ch['fname'],
                )

        except Exception as e:
            self.report_exception('raman_spectrum')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Raman Spectrum')
