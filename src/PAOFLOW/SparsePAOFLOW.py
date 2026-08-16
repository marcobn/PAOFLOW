"""Sparse driver mirroring the dense :class:`PAOFLOW.PAOFLOW` API.

``SparsePAOFLOW`` wraps a dense ``PAOFLOW`` instance for the input stages
(QE parsing, projectability, base-cell Hamiltonian construction — the one
sanctioned dense stage, at the small pre-doubling ``nawf``) and switches
to the purely sparse backend (:mod:`PAOFLOW.sparse`) from the moment the
bond list exists.  Method names and signatures mirror the dense driver so
an example script differs only in the import and constructor.

Dense methods without a sparse counterpart raise ``NotImplementedError``
loudly — there is no silent fallback to dense arrays.

Memory contract (see :mod:`PAOFLOW.sparse`): after ``pao_hamiltonian``
returns, no array of size O(nawf^2 * nk) exists; per-k dense workspace is
limited to one ``(nawf, nev)`` eigenvector block.
"""

import numpy as np
from mpi4py import MPI

from .PAOFLOW import PAOFLOW
from .sparse.hamiltonian import SparseHamiltonian


class SparsePAOFLOW:
    def __init__(
        self,
        workpath='./',
        outputdir='output_sparse',
        inputfile=None,
        savedir=None,
        npool=1,
        smearing='gauss',
        verbose=False,
        threshold=1.0e-3,
        restart=False,
        dft='QE',
    ):
        """
        Arguments mirror :class:`PAOFLOW.PAOFLOW`; additionally:
            threshold (float): magnitude (eV) below which H(R) matrix
            elements are dropped when the dense base-cell Hamiltonian is
            converted to the sparse bond list.  The conversion prints a
            rigorous bound on the eigenvalue error this truncation can
            cause at any k-point.
        """
        self._pao = PAOFLOW(
            workpath=workpath,
            outputdir=outputdir,
            inputfile=inputfile,
            savedir=savedir,
            npool=npool,
            smearing=smearing,
            verbose=verbose,
            restart=restart,
            dft=dft,
        )
        self.data_controller = self._pao.data_controller
        self.comm = self._pao.comm
        self.rank = self._pao.rank
        self.threshold = float(threshold)
        self.H = None  # SparseHamiltonian, set by pao_hamiltonian
        self._mesh_plan = {}  # parameters recorded for the fused mesh pass

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        if hasattr(PAOFLOW, name):
            raise NotImplementedError(
                "SparsePAOFLOW does not implement '%s'. Only the example01 "
                'pipeline (bands, dos, transport) has a sparse backend so far; '
                'use the dense PAOFLOW driver for other features.' % name
            )
        raise AttributeError(name)

    def _guard(self, tag, func):
        """Mirror the dense try/except + abort_on_exception convention."""
        attr = self.data_controller.data_attributes
        try:
            return func()
        except Exception as e:
            self._pao.report_exception(tag)
            if attr.get('abort_on_exception', True):
                raise e

    def _require_H(self, caller):
        if self.H is None:
            raise RuntimeError(
                'SparsePAOFLOW.%s requires the sparse Hamiltonian; call '
                'pao_hamiltonian() first.' % caller
            )

    # ------------------------------------------------------------------
    # Input stages (delegated to the dense driver, base cell only)
    # ------------------------------------------------------------------

    def read_atomic_proj_QE(self):
        self._pao.read_atomic_proj_QE()

    def projectability(self, pthr=0.95, shift='auto'):
        self._pao.projectability(pthr=pthr, shift=shift)

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
        """Build the base-cell PAO Hamiltonian (dense, sanctioned input
        stage) and immediately convert it to the sparse bond list; the
        dense ``HRs``/``Hks`` are deleted before returning."""
        self._pao.pao_hamiltonian(
            shift_type=shift_type,
            insulator=insulator,
            write_binary=write_binary,
            expand_wedge=expand_wedge,
            symmetrize=symmetrize,
            thresh=thresh,
            max_iter=max_iter,
        )

        def _convert():
            arrays, _ = self.data_controller.data_dicts()
            self.H = SparseHamiltonian.from_data_controller(self.data_controller, self.threshold)
            # the dense source must not outlive the conversion
            del arrays['HRs']
            arrays.pop('Hks', None)
            arrays.pop('Dnm', None)  # carried per bond by the container
            if self.rank == 0:
                print(self.H.stats_line())

        self._guard('sparse_conversion', _convert)
        self._pao.report_module_time('Sparse conversion')

    # ------------------------------------------------------------------
    # Doubling (purely sparse)
    # ------------------------------------------------------------------

    def doubling_Hamiltonian(self, nx, ny, nz):
        """Double the cell ``nx``/``ny``/``nz`` times along each lattice
        vector by index arithmetic on the bond list (never dense), then
        Hermitize once — the bond-level equivalent of the per-k
        Hermitizations the dense pipeline applies downstream."""
        from .hamiltonian.do_doubling import doubling_attr_arry
        from .sparse.doubling import double_axis

        self._require_H('doubling_Hamiltonian')
        arrays, attr = self.data_controller.data_dicts()
        attr['nx'], attr['ny'], attr['nz'] = nx, ny, nz

        def _double():
            # deterministic and replicated on every rank: no broadcast needed
            for axis, reps in ((0, nx), (1, ny), (2, nz)):
                for _ in range(reps):
                    self.H = double_axis(self.H, axis)
                    arrays['tau'] = np.append(
                        arrays['tau'],
                        arrays['tau'] + arrays['a_vectors'][axis, :] * attr['alat'],
                        axis=0,
                    )
                    arrays['a_vectors'][axis, :] *= 2
                    doubling_attr_arry(self.data_controller)
            self.H = self.H.hermitize()
            if self.rank == 0:
                print(self.H.stats_line())

        self._guard('doubling_Hamiltonian', _double)
        self._pao.report_module_time('doubling_Hamiltonian')

    # ------------------------------------------------------------------
    # Bands along a high-symmetry path
    # ------------------------------------------------------------------

    def bands(
        self, ibrav=None, band_path=None, high_sym_points=None, fname='bands', nk=500, nsel=None
    ):
        """Band structure along a path; computes only the lowest ``nsel``
        bands (default ``attr['bnd']``) iteratively.  Output format matches
        the dense ``bands_{ispin}.dat`` with ``nsel`` value columns."""
        from .sparse.bands import do_bands_sparse
        from .utils.communication import gather_full

        self._require_H('bands')
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
        if high_sym_points is not None:
            arrays['high_sym_points'] = high_sym_points
        if nsel is None:
            nsel = attr['bnd']

        def _bands():
            do_bands_sparse(self.data_controller, self.H, nsel, verbose=attr['verbose'])
            E_kp = gather_full(arrays['E_k'], attr['npool'])
            self.data_controller.write_bands(fname, E_kp)

        self._guard('bands', _bands)
        self._pao.report_module_time('Bands')

    # ------------------------------------------------------------------
    # Fourier interpolation to a finer k-mesh (pure metadata)
    # ------------------------------------------------------------------

    def interpolated_hamiltonian(self, nfft1=0, nfft2=0, nfft3=0):
        """Interpolate onto a finer k-mesh.  Zero-padding H(R) adds only
        zero hoppings, and the bond-list assembly already uses the
        Hermiticity-preserving Nyquist-split convention of
        ``utils.zero_pad`` — so sparse interpolation is exact and free:
        only the mesh dimensions change, no new data is created.
        Arguments of 0 default to twice the current grid (as dense)."""
        self._require_H('interpolated_hamiltonian')
        arrays, attr = self.data_controller.data_dicts()
        nfft = [
            nfft1 if nfft1 > 0 else 2 * attr['nk1'],
            nfft2 if nfft2 > 0 else 2 * attr['nk2'],
            nfft3 if nfft3 > 0 else 2 * attr['nk3'],
        ]
        attr['nk1'], attr['nk2'], attr['nk3'] = nfft
        attr['nkpnts'] = nfft[0] * nfft[1] * nfft[2]
        if self.rank == 0 and attr['verbose']:
            print(
                'Sparse interpolation: property mesh set to %d x %d x %d '
                '(no new data — bond list evaluated on the finer grid)' % tuple(nfft)
            )
        self._pao.report_module_time('R -> k with Zero Padding')

    # ------------------------------------------------------------------
    # Fused mesh pass (eigenvalues + velocities + smearing widths + PDOS)
    # ------------------------------------------------------------------

    def pao_eigh(self, bval=0):
        """Recorded for API compatibility: the mesh eigensolve is fused
        with velocities/PDOS into one pass, executed by the first property
        call that needs it (``dos`` or ``transport``)."""
        self._mesh_plan['eigh'] = True

    def gradient_and_momenta(self, **kwargs):
        """Recorded for API compatibility: band-diagonal velocities are
        computed inside the fused mesh pass; the full momentum tensor
        ``pksp`` is never formed."""
        self._mesh_plan['velocities'] = True

    def adaptive_smearing(self, smearing='gauss', afac=None):
        """Record the adaptive-smearing prefactor for the fused mesh pass
        (Yates widths, as ``do_adaptive_smearing``; the interband
        ``deltakp2`` is not needed by the sparse pipeline)."""
        self._mesh_plan['smearing'] = smearing
        self._mesh_plan['afac'] = afac

    def _ensure_mesh(self, pdos_spec=None):
        """Run the fused mesh pass if its results are not yet available
        (or if PDOS accumulation is requested but was not part of the
        earlier pass, in which case the mesh is recomputed)."""
        from .sparse.mesh import run_mesh
        from .sparse.pdos import PdosConsumer

        arrays, attr = self.data_controller.data_dicts()
        have = self._mesh_plan.get('executed', False)
        need_pdos = pdos_spec is not None and not self._mesh_plan.get('pdos_done', False)
        if have and not need_pdos:
            return
        if have and need_pdos and self.rank == 0 and attr['verbose']:
            print(
                'Sparse mesh: re-running the fused pass to accumulate PDOS '
                '(eigenvectors are never stored).'
            )

        consumers = []
        if pdos_spec is not None:
            consumers.append(PdosConsumer(self.data_controller, *pdos_spec))

        nev = attr['bnd']

        def _mesh():
            run_mesh(
                self.data_controller,
                self.H,
                nev,
                consumers=consumers,
                afac=self._mesh_plan.get('afac'),
                smearing=self._mesh_plan.get('smearing', attr.get('smearing', 'gauss')),
                verbose=attr['verbose'],
            )

        self._guard('sparse_mesh', _mesh)
        self._mesh_plan['executed'] = True
        if pdos_spec is not None:
            self._mesh_plan['pdos_done'] = True
        self._pao.report_module_time('Sparse mesh (eigh + velocities)')

    # ------------------------------------------------------------------
    # Properties (dense band-diagonal kernels reused verbatim)
    # ------------------------------------------------------------------

    def dos(self, do_dos=True, do_pdos=True, delta=0.01, emin=-10.0, emax=2.0, ne=1000):
        """DOS via the dense ``do_dos_adaptive`` (consumes only band-diagonal
        arrays); PDOS accumulated streaming inside the mesh pass."""
        from .spectrum.do_dos import do_dos_adaptive

        self._require_H('dos')
        self._ensure_mesh(pdos_spec=(emin, emax, ne) if do_pdos else None)

        def _dos():
            if do_dos:
                do_dos_adaptive(self.data_controller, emin, emax, ne)

        self._guard('dos', _dos)
        self._pao.report_module_time('DoS')

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
        """Boltzmann transport via the dense stack (it consumes only the
        band-diagonal ``velkp``/``E_k``/``deltakp`` the mesh produced)."""
        self._require_H('transport')
        self._ensure_mesh()

        arrays, attr = self.data_controller.data_dicts()
        top = self.comm.allreduce(float(np.min(arrays['E_k'][:, -1, :])), op=MPI.MIN)
        if emax > top:
            raise RuntimeError(
                'sparse transport: requested emax=%.3f eV exceeds the lowest '
                'computed top band (%.3f eV); the %d-band window does not cover '
                'the energy range.' % (emax, top, attr['bnd'])
            )

        self._pao.transport(
            tmin=tmin,
            tmax=tmax,
            nt=nt,
            emin=emin,
            emax=emax,
            ne=ne,
            scattering_channels=scattering_channels,
            scattering_weights=scattering_weights,
            tau_dict=tau_dict,
            do_hall=do_hall,
            write_to_file=write_to_file,
            save_tensors=save_tensors,
        )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def finish_execution(self):
        self._pao.finish_execution()
