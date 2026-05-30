#
# PAOFLOW sparse frontend
#
# This file keeps sparse orchestration outside PAOFLOW.py.  Dense methods that
# do not need a sparse twin are inherited unchanged from PAOFLOW.PAOFLOW.

from __future__ import annotations

from .PAOFLOW import PAOFLOW


class SparsePAOFLOW(PAOFLOW):
    r"""Sparse PAOFLOW workflow frontend.

    The object represents the same PAO Hamiltonian workflow as ``PAOFLOW``, but
    Hamiltonian construction and sparse-aware post-processing are routed through
    sparse-native routines.  A sparse Hamiltonian block is interpreted as the
    matrix :math:`H_{ij}(k)` or :math:`H_{ij}(R)` in a PAO basis, with the row
    and column indices labeling PAO orbitals.

    Parameters
    ----------
    workpath : str, optional
        Working directory containing input and output paths.
    outputdir : str, optional
        Directory where PAOFLOW output is written.
    inputfile : str or None, optional
        Input XML file, when the workflow is initialized from a file rather
        than a saved DFT directory.
    savedir : str or None, optional
        Quantum ESPRESSO ``.save`` directory.
    model : dict or None, optional
        Tight-binding model description used instead of a DFT save directory.
    npool : int, optional
        Number of MPI pools used for k-point distribution.
    smearing : str or None, optional
        Smearing model used by downstream integrations.
    save_overlaps : bool, optional
        Store wave-function overlaps in the data controller.
    acbn0 : bool, optional
        Enable the ACBN0 orthogonalization workflow.
    sparse_threshold : float, optional
        Magnitude cutoff used by sparse builders when pruning matrix elements.
    verbose : bool, optional
        Print diagnostic information.
    restart : bool, optional
        Restart from a serialized PAOFLOW state.
    dft : {'QE', 'VASP'}, optional
        DFT interface used to read input quantities.

    Returns
    -------
    None
        The initialized object stores workflow state in ``self.data_controller``.
    """

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
        sparse_threshold=1.0e-6,
        verbose=False,
        restart=False,
        dft='QE',
    ):
        from time import time

        from mpi4py import MPI

        from .DataController import DataController
        from .defs.header import header

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank == 0:
            header()
            self.start_time = self.reset_time = time()

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
            True,
            sparse_threshold,
            verbose,
            restart,
            dft,
        )

        self.report_exception = self.data_controller.report_exception

        if not restart:
            attr = self.data_controller.data_attributes
            attr['use_cuda'] = False
            attr['scipyfft'] = True
            if attr['use_cuda']:
                attr['scipyfft'] = False
            if self.rank == 0 and attr['verbose']:
                if attr['use_cuda']:
                    print('CUDA will perform FFTs on %d GPUs' % 1)
                else:
                    print('SciPy will perform FFTs')

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

        if model is None and not restart and self.rank == 0:
            gbyte = self.memory_check()
            print('Estimated maximum array size: %.2f GBytes\n' % (gbyte))

        self.report_module_time('Initialization')

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
        r"""Construct sparse PAO Hamiltonians in k and real space.

        The sparse path constructs :math:`H(k)` and transforms it into sparse
        real-space blocks :math:`H(R)` without materializing the dense tensor
        :math:`H_{ij}(R)` for all PAO pairs and lattice vectors.

        Parameters
        ----------
        shift_type : int, optional
            Projectability shift convention used by the PAO Hamiltonian builder.
        insulator : bool, optional
            Mark the system as insulating for downstream routines.
        write_binary : bool, optional
            Request binary Hamiltonian output where supported.
        expand_wedge : bool, optional
            Expand symmetry-reduced k-points before building the Hamiltonian.
        symmetrize : bool, optional
            Request symmetry restoration before sparse construction.
        thresh : float, optional
            Symmetry threshold stored in the data controller.
        max_iter : int, optional
            Maximum number of symmetry iterations.

        Returns
        -------
        None
            Sparse Hamiltonian objects are stored in the data controller.
        """
        from .defs.get_K_grid_fft import get_K_grid_fft
        from .sparse.build_hk import do_build_pao_hamiltonian
        from .sparse.build_hr import do_Hks_to_HRs

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

        try:
            do_build_pao_hamiltonian(self.data_controller)
            do_Hks_to_HRs(self.data_controller)
            get_K_grid_fft(self.data_controller)
        except Exception as e:
            self.report_exception('pao_hamiltonian_sparse')
            if attr['abort_on_exception']:
                raise e

        arrays.pop('U', None)
        self.report_module_time('Sparse Hk -> R boundary')

    def doubling_Hamiltonian(self, nx, ny, nz):
        r"""Double sparse real-space Hamiltonian blocks.

        Parameters
        ----------
        nx, ny, nz : int
            Repetition factors along the three lattice directions.  Sparse
            doubling maps :math:`H(R)` into the enlarged PAO basis while keeping
            each block sparse.

        Returns
        -------
        None
            Doubled sparse Hamiltonian metadata are stored and broadcast.
        """
        arrays, attributes = self.data_controller.data_dicts()
        attributes['nx'], attributes['ny'], attributes['nz'] = nx, ny, nz

        try:
            if self.rank == 0:
                from .sparse.doubling import doubling_HRs

                doubling_HRs(self.data_controller)

            if self.rank != 0:
                arrays.pop('HRs', None)

            array_list = [
                'SparseHRs',
                'naw',
                'a_vectors',
                'tau',
                'atoms',
                'sh',
                'nl',
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
                'sparse_doubling_physical_rcut',
                'sparse_doubling_uses_automatic_physical_rcut',
                'Dnm_invalid_after_sparse_doubling',
            ]
            for attr in attr_list:
                if attr in attributes:
                    self.data_controller.broadcast_attribute(attr)

        except Exception as e:
            self.report_exception('doubling_Hamiltonian')
            if attributes['abort_on_exception']:
                raise e

        self.report_module_time('doubling_Hamiltonian')

    def bands(
        self,
        ibrav=None,
        band_path=None,
        high_sym_points=None,
        spin_orbit=False,
        fname='bands',
        nk=500,
        target='near_fermi',
        nbands=None,
        return_eigenvectors=False,
        sigma=None,
        tol=0.0,
        real_tol=1.0e-10,
        fermi_window=6.0,
        profile_timing=False,
        near_fermi_initial=None,
        near_fermi_step=None,
        near_fermi_max_candidates=None,
        dense_local=False,
        dense_density_threshold=0.5,
    ):
        r"""Compute selected sparse band eigenvalues along a k-path.

        For each sampled k-point, the sparse solver forms the Hamiltonian
        action associated with :math:`H(k)` and computes only the requested
        eigenvalues, rather than diagonalizing the full dense PAO matrix.

        Parameters
        ----------
        ibrav : int or None, optional
            Bravais lattice index used by the path generator.
        band_path : str or None, optional
            Symbolic high-symmetry path.
        high_sym_points : dict or None, optional
            Mapping from high-symmetry labels to fractional k coordinates.
        spin_orbit : bool, optional
            Whether the band path corresponds to a spin-orbit calculation.
        fname : str, optional
            Output filename prefix.
        nk : int, optional
            Number of k-points on the path.
        target : {'lowest', 'highest', 'near_fermi', 'near_energy'}, optional
            Part of the spectrum requested from the sparse eigensolver.
        nbands : int or None, optional
            Number of eigenvalues requested.  If ``None``, a conservative value
            based on ``attr['bnd']`` and ``attr['nawf']`` is used.
        return_eigenvectors : bool, optional
            Store selected eigenvectors when the sparse solver returns them.
        sigma : float or None, optional
            Target energy for ``target='near_energy'``.
        tol : float, optional
            Iterative eigensolver tolerance.
        real_tol : float, optional
            Tolerance for treating tiny imaginary eigenvalue parts as numerical
            noise.
        fermi_window : float, optional
            Half-width in electron-volts used to accept states around the Fermi
            energy when ``target='near_fermi'``.
        profile_timing : bool, optional
            Store and print sparse band timing split between sparse ``H(k)``
            assembly and eigensolver time.
        near_fermi_initial : int or None, optional
            Initial candidate count for ``target='near_fermi'``.
        near_fermi_step : int or None, optional
            Linear increment applied when additional near-Fermi candidates are
            needed.
        near_fermi_max_candidates : int or None, optional
            Upper bound for candidate count in ``target='near_fermi'``.
        dense_local : bool, optional
            Explicitly allow one assembled :math:`H(k)` block to be converted to
            a dense matrix when its structural density exceeds
            ``dense_density_threshold``. The dense block is diagonalized and
            discarded before moving to the next k-point.
        dense_density_threshold : float, optional
            Minimum density :math:`\mathrm{nnz}(H)/N^2` needed before the
            opt-in local dense backend is used.

        Returns
        -------
        None
            Eigenvalues are stored in ``arrays['E_k']`` and written to disk.
        """
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

        if nbands is None:
            nbands = min(max(1, int(attr['bnd'])), int(attr['nawf']) - 2)
        attr['sparse_bands_target'] = str(target).lower()
        attr['sparse_bands_nbands'] = int(nbands)
        attr['sparse_bands_return_eigenvectors'] = bool(return_eigenvectors)
        attr['sparse_bands_tol'] = float(tol)
        attr['sparse_bands_real_tol'] = float(real_tol)
        attr['sparse_bands_fermi_window'] = float(fermi_window)
        attr['sparse_bands_profile_timing'] = bool(profile_timing)
        attr['sparse_bands_dense_local'] = bool(dense_local)
        attr['sparse_bands_dense_density_threshold'] = float(dense_density_threshold)
        if sigma is not None:
            attr['sparse_bands_sigma'] = float(sigma)
        if near_fermi_initial is not None:
            attr['sparse_bands_near_fermi_initial'] = int(near_fermi_initial)
        if near_fermi_step is not None:
            attr['sparse_bands_near_fermi_step'] = int(near_fermi_step)
        if near_fermi_max_candidates is not None:
            attr['sparse_bands_near_fermi_max_candidates'] = int(near_fermi_max_candidates)

        try:
            from .sparse.bands import do_bands

            do_bands(self.data_controller)
            self.data_controller.write_bands(fname, arrays['E_k'])
        except Exception as e:
            self.report_exception('bands')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Bands')

    def pao_eigh(
        self,
        bval=0,
        target='near_fermi',
        nbands=None,
        return_eigenvectors=False,
        sigma=None,
        tol=0.0,
        real_tol=1.0e-10,
    ):
        r"""Solve selected sparse eigenproblems on the stored k-point mesh.

        Parameters
        ----------
        bval : int, optional
            Valence-band index used when eigenvalues are shifted after Hubbard-U
            corrections.
        target : {'lowest', 'highest', 'near_fermi', 'near_energy'}, optional
            Spectral region requested from the sparse eigensolver.
        nbands : int or None, optional
            Number of selected eigenvalues requested.  If ``None``, a
            conservative value based on ``attr['bnd']`` and ``attr['nawf']`` is
            used.
        return_eigenvectors : bool, optional
            Store selected eigenvectors in ``arrays['v_k']``.
        sigma : float or None, optional
            Target energy for ``target='near_energy'``.
        tol : float, optional
            Iterative eigensolver tolerance.
        real_tol : float, optional
            Tolerance for treating tiny imaginary eigenvalue parts as numerical
            noise.

        Returns
        -------
        None
            Selected eigenvalues are stored in ``arrays['E_k']``.
            Selected eigenvectors are stored in ``arrays['v_k']`` only
            when requested.
        """
        from .sparse.eigh import do_pao_eigh as do_pao_eigh_sparse

        arrays, attr = self.data_controller.data_dicts()

        if 'bval' not in attr:
            attr['bval'] = bval
        arrays.pop('HRs', None)

        if nbands is None:
            nbands = min(max(1, int(attr['bnd'])), int(attr['nawf']) - 2)

        attr['sparse_eigh_target'] = str(target).lower()
        attr['sparse_eigh_nbands'] = int(nbands)
        attr['sparse_eigh_return_eigenvectors'] = bool(return_eigenvectors)
        attr['sparse_eigh_tol'] = float(tol)
        attr['sparse_eigh_real_tol'] = float(real_tol)
        if sigma is not None:
            attr['sparse_eigh_sigma'] = float(sigma)
        else:
            attr.pop('sparse_eigh_sigma', None)

        try:
            if 'Hks_sparse' in arrays and 'Hksp' not in arrays:
                e_k_sparse, v_k_sparse = do_pao_eigh_sparse(self.data_controller)
                arrays['E_k'] = e_k_sparse
                arrays.pop('E_k', None)
                arrays.pop('v_k', None)
                arrays.pop('degen', None)
                if v_k_sparse is None:
                    arrays.pop('v_k', None)
                else:
                    arrays['v_k'] = v_k_sparse
            else:
                super().pao_eigh(bval=bval)
                return
        except Exception as e:
            self.report_exception('pao_eigh')
            if attr['abort_on_exception']:
                raise e

        self.report_module_time('Eigenvalues')
