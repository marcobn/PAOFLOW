"""SparsePAOFLOW — a purely sparse driver for the PAOFLOW workflow.

``SparsePAOFLOW`` subclasses :class:`PAOFLOW.PAOFLOW.PAOFLOW` and reuses its
backend-agnostic stages verbatim (``read_atomic_proj_QE``, ``projectability``,
``finish_execution``, ``memory_check``, the QE readers).  It overrides only the
stages that would otherwise materialise dense ``(nawf, nawf, nkpnts, nspin)``
tensors, replacing them with the sparse kernels in :mod:`PAOFLOW.sparse`.

The public method names and signatures mirror the dense driver so an example
reads as the same sequence of physics actions:

    paoflow = SparsePAOFLOW(savedir='silicon.save', smearing='gauss', verbose=True)
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.doubling_Hamiltonian(nx=2, ny=2, nz=2)   # or interpolated_hamiltonian()
    paoflow.bands(ibrav=2, nk=2000)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)
    paoflow.finish_execution()
"""

import numpy as np

from .PAOFLOW import PAOFLOW
from .sparse import stats
from .sparse.driver import SparseOrchestrationMixin


class SparsePAOFLOW(SparseOrchestrationMixin, PAOFLOW):
    """Sparse PAOFLOW driver (no dense fine-grid materialisation).

    Parameters
    ----------
    All parameters of :class:`PAOFLOW.PAOFLOW.PAOFLOW`, plus:

    sparse_threshold : float
        Real-space Hamiltonian entries with ``abs(H) < sparse_threshold`` are
        discarded when the coarse Hamiltonian is converted to the sparse
        hopping list.  Default ``1e-6``.
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
        verbose=False,
        restart=False,
        dft='QE',
        sparse_threshold=1.0e-6,
    ):
        from time import time

        from mpi4py import MPI

        from .DataController import DataController
        from .utils.header import header

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank == 0:
            header()
            self.start_time = self.reset_time = time()

        # The only substantive difference from the dense constructor: the
        # DataController is built with sparse=True and the requested threshold,
        # activating the dormant sparse hook that dense PAOFLOW leaves at False.
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
            attr['scipyfft'] = True

        if self.rank == 0:
            if restart:
                print('Run starting from Restart data.')
            elif self.size == 1:
                print('Serial execution (sparse backend)')
            else:
                print(
                    'Parallel execution on %d processors and %d pool'
                    % (self.size, self.data_controller.data_attributes['npool'])
                    + ('' if self.data_controller.data_attributes['npool'] == 1 else 's')
                )

        if model is None and not restart and self.rank == 0:
            gbyte = self.memory_check()
            print('Estimated maximum (dense) array size: %.2f GBytes\n' % gbyte)

        self.report_module_time('Initialization')

    # ------------------------------------------------------------------
    #  Hamiltonian construction (sparse)
    # ------------------------------------------------------------------
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
        """Build the coarse PAO Hamiltonian (bounded) — sparse conversion deferred.

        The dense coarse ``Hks``/``HRs`` are the intrinsically-dense PAO
        projection outer product on the QE mesh (a bounded, size-gated input
        stage).  They are held only until the first stage that needs the sparse
        Hamiltonian (``doubling_Hamiltonian`` operates on the coarse ``HRs``;
        ``interpolated_hamiltonian``/``bands``/``pao_eigh`` then finalize it into
        the sparse hopping list and free the dense array).
        """
        from .sparse.hamiltonian_builder import build_coarse_hamiltonian
        from .utils.get_K_grid_fft import get_K_grid_fft

        arrays, attr = self.data_controller.data_dicts()

        if insulator:
            attr['insulator'] = True
        attr.setdefault('shift_type', shift_type)
        attr.setdefault('write_binary', write_binary)
        attr['symm_thresh'] = thresh
        attr['symmetrize'] = symmetrize
        attr['symm_max_iter'] = max_iter
        attr['expand_wedge'] = expand_wedge

        self._guard('pao_hamiltonian', lambda: build_coarse_hamiltonian(self.data_controller))
        # U is consumed; drop it (mirrors the dense eviction point).
        if 'U' in arrays:
            del arrays['U']
        self._guard('pao_hamiltonian', lambda: get_K_grid_fft(self.data_controller))

        nkpnts = attr['nk1'] * attr['nk2'] * attr['nk3']
        self.sparse_log(
            'Coarse PAO H(R) built on %dx%dx%d grid (bounded input stage, '
            'dense=%s)'
            % (
                attr['nk1'],
                attr['nk2'],
                attr['nk3'],
                stats.human_bytes(
                    stats.estimate_dense_grid_bytes(attr['nawf'], nkpnts, attr['nspin'])
                ),
            )
        )
        self.report_module_time('Building Hks')

    def doubling_Hamiltonian(self, nx, ny, nz):
        """Double the cell on the coarse Hamiltonian (delegates to dense kernel)."""
        from .sparse.doubling import sparse_doubling

        self._guard(
            'doubling_Hamiltonian',
            lambda: sparse_doubling(self.data_controller, nx, ny, nz),
        )
        self.report_module_time('doubling_Hamiltonian')

    def interpolated_hamiltonian(self, nfft1=0, nfft2=0, nfft3=0, reshift_Ef=False):
        """Record the finer interpolation grid and finalize the sparse Hamiltonian.

        Sparse interpolation is matrix-free: the thresholded ``H(R)`` is the
        interpolant and ``H(k)`` is assembled on the finer mesh on demand.  No
        dense ``(nawf, nawf, nfft1, nfft2, nfft3, nspin)`` array is formed.
        """
        from .sparse.interpolation import set_interpolation_grid

        _, attr = self.data_controller.data_dicts()

        sparse_h = self._guard(
            'interpolated_hamiltonian',
            lambda: set_interpolation_grid(
                self.data_controller, nfft1, nfft2, nfft3, attr['sparse_threshold']
            ),
        )
        if sparse_h is not None:
            self.sparse_log(stats.hamiltonian_stats(sparse_h))
            self.sparse_log(
                'Interpolation: sparse (matrix-free), grid %dx%dx%d -> %dx%dx%d'
                % (
                    attr['nk1'],
                    attr['nk2'],
                    attr['nk3'],
                    attr['nfft1'],
                    attr['nfft2'],
                    attr['nfft3'],
                )
            )
        self.report_module_time('R -> k (sparse, matrix-free)')

    def _ensure_sparse_H(self):
        """Finalize the sparse Hamiltonian from the coarse/doubled ``HRs`` if needed."""
        from .sparse.hamiltonian_builder import finalize_sparse_hamiltonian

        arrays, attr = self.data_controller.data_dicts()
        if arrays.get('sparse_H') is None:
            sparse_h = finalize_sparse_hamiltonian(self.data_controller, attr['sparse_threshold'])
            self.sparse_log(stats.hamiltonian_stats(sparse_h))
        return arrays['sparse_H']

    # ------------------------------------------------------------------
    #  Spectral stages (sparse eigsh)
    # ------------------------------------------------------------------
    def bands(self, ibrav=None, band_path=None, high_sym_points=None, nk=500, n_bands=None):
        """Selected band structure along a k-path via sparse ``eigsh``."""
        from .sparse.eigensolvers import solve_path
        from .spectrum.kpnts_interpolation_mesh import kpnts_interpolation_mesh
        from .utils.constants import ANGSTROM_AU

        arrays, attr = self.data_controller.data_dicts()
        if ibrav is not None:
            attr['ibrav'] = ibrav
        attr['nk'] = nk
        attr['band_path'] = band_path
        if high_sym_points is not None:
            arrays['high_sym_points'] = high_sym_points
        arrays.setdefault('high_sym_points', {})

        def _run():
            self._ensure_sparse_H()
            # Build the crystal k-path exactly as the dense bands stage does
            # (including its Bohr<->Angstrom alat handling for point density).
            attr['alat'] /= ANGSTROM_AU
            kpnts_interpolation_mesh(self.data_controller)
            attr['alat'] *= ANGSTROM_AU
            kq_cart = arrays['kq'].T @ arrays['b_vectors']  # (nkpi, 3) Cartesian

            n_eig = n_bands if n_bands is not None else self._auto_band_count()
            eig = solve_path(
                self.data_controller,
                kq_cart,
                n_eig,
                progress_callback=lambda done, total: self.sparse_log(
                    'Sparse bands progress: %d/%d k-points' % (done, total)
                ),
            )
            self.sparse_log(
                stats.eigensolver_stats(
                    'bands', None, eig.n_sel, kq_cart.shape[0], eig.converged, eig.solver
                )
            )
            self.data_controller.write_bands('bands', eig.E_k)

        self._guard('bands', _run)
        self.report_module_time('Bands')

    def _auto_band_count(self):
        """Default number of lowest bands to compute (occupied + conduction buffer)."""
        _, attr = self.data_controller.data_dicts()
        nawf = attr['nawf']
        n_occ = int(round(attr.get('nelec', nawf))) // 2
        selected_cap = max(1, int(np.floor(0.75 * nawf)) - 1)
        return int(min(nawf - 2, selected_cap, n_occ + max(4, n_occ // 2)))

    def pao_eigh(self, bval=0, emin=None, emax=None, n_bands=None, solver='eigsh'):
        """Selected-window eigenpairs over the full BZ grid via sparse ``eigsh``.

        Computes the lowest bands (with eigenvectors, needed for velocities) on
        the interpolation/coarse BZ mesh.  ``n_bands``/``emax`` control the
        window; by default enough bands are computed to cover the occupied
        states plus a conduction buffer, which downstream ``dos``/``transport``
        verify against their own energy windows.
        """
        from .sparse.eigensolvers import build_bz_kgrid, solve_window
        from .sparse.observables import store_eigenpairs

        arrays, _ = self.data_controller.data_dicts()

        def _run():
            self._ensure_sparse_H()
            kcart = build_bz_kgrid(self.data_controller)
            arrays['sparse_kgrid'] = kcart
            n_eig = n_bands if n_bands is not None else self._auto_band_count()
            e_lo = -1.0e3 if emin is None else emin
            eig = solve_window(
                self.data_controller,
                kcart,
                e_lo,
                emax if emax is not None else -np.inf,
                want_vectors=True,
                n_eigs=n_eig,
                solver=solver,
            )
            store_eigenpairs(self.data_controller, eig)
            self.sparse_log(
                stats.eigensolver_stats(
                    'pao_eigh',
                    eig.window if emax is not None else None,
                    eig.n_sel,
                    kcart.shape[0],
                    eig.converged,
                    eig.solver,
                )
            )

        self._guard('pao_eigh', _run)
        self.report_module_time('Eigenvalues')

    def gradient_and_momenta(self, band_curvature=False):
        """Band-diagonal group velocities via Hellmann–Feynman (sparse ``dH/dk``)."""
        from .sparse.observables import compute_velocities

        arrays, attr = self.data_controller.data_dicts()

        def _run():
            compute_velocities(self.data_controller, arrays['sparse_kgrid'])
            self.sparse_log(stats.velocity_stats(attr['nkpnts'], attr['bnd'], attr['nspin']))
            # Eigenvectors are no longer needed once velocities are formed.
            arrays['sparse_v_k'] = None

        self._guard('gradient_and_momenta', _run)
        self.report_module_time('Momenta')

    # ------------------------------------------------------------------
    #  Observables (reuse band-diagonal dense kernels)
    # ------------------------------------------------------------------
    def adaptive_smearing(self, smearing='gauss', afac=None):
        """Yates adaptive smearing widths from the sparse band velocities."""
        from .sparse.observables import adaptive_smearing as _adaptive

        _, attr = self.data_controller.data_dicts()
        if smearing not in ('gauss', 'm-p'):
            raise ValueError("Smearing must be 'gauss' or 'm-p'.")
        attr['smearing'] = smearing

        self._guard(
            'adaptive_smearing',
            lambda: _adaptive(self.data_controller, smearing, afac),
        )
        self.report_module_time('Adaptive Smearing')

    def dos(self, do_dos=True, do_pdos=False, delta=0.01, emin=-10.0, emax=2.0, ne=1000):
        """Density of states from the selected-window spectrum (reuses dense DOS)."""
        from .spectrum.do_dos import do_dos as _do_dos
        from .spectrum.do_dos import do_dos_adaptive as _do_dos_adaptive

        arrays, attr = self.data_controller.data_dicts()
        attr.setdefault('smearing', None)

        def _run():
            if attr['smearing'] is None or 'deltakp' not in arrays:
                _do_dos(self.data_controller, emin, emax, ne, delta)
            else:
                _do_dos_adaptive(self.data_controller, emin, emax, ne)

        self._guard('dos', _run)
        tag = (
            'DoS'
            if (attr['smearing'] is None or 'deltakp' not in arrays)
            else 'DoS (Adaptive Smearing)'
        )
        self.report_module_time(tag)

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
        """Boltzmann transport tensors from the sparse band-diagonal spectrum."""
        from .sparse.transport import run_transport

        _, attr = self.data_controller.data_dicts()
        attr['transport_ne'] = ne

        self._guard(
            'transport',
            lambda: run_transport(
                self.data_controller,
                tmin,
                tmax,
                nt,
                emin,
                emax,
                scattering_channels,
                scattering_weights,
                tau_dict,
                do_hall,
                write_to_file,
                save_tensors,
            ),
        )
        self.report_module_time('Transport')
