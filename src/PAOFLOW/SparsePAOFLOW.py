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


def _available_memory_bytes():
    """Free memory from ``/proc/meminfo``, or ``None`` where unreadable.

    ``MemAvailable`` rather than ``MemFree``: it accounts for reclaimable
    page cache, which is what a large allocation can actually take over.
    """
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _node_local_ranks(comm):
    """Ranks sharing this node, which all hold their own copy of the bond
    list.  Falls back to the world size (the pessimistic reading) if the
    MPI build has no shared-memory split."""
    try:
        node = comm.Split_type(MPI.COMM_TYPE_SHARED)
        size = node.Get_size()
        node.Free()
        return size
    except (AttributeError, MPI.Exception):
        return comm.Get_size()


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
        rcut=None,
        solver='auto',
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

            rcut (float or None): optional real-space cutoff in Bohr on
            the physical bond length, applied together with ``threshold``
            at the base cell.  This is a second, physically different
            truncation axis (bond length rather than matrix-element
            magnitude) and the two interact, so results are not directly
            comparable with an ``rcut=None`` run; the ``eig_bound``
            printed at conversion covers both.  Default ``None``.

            solver ('auto' | 'dense' | 'arpack'): per-k eigensolver
            branch.  ``'auto'`` dispatches on ``(nawf, nev)`` — see
            :func:`PAOFLOW.sparse.solver.select_method`.  The explicit
            values force one branch for A/B validation.
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
        self.rcut = None if rcut is None else float(rcut)
        self.solver = solver
        self.H = None  # SparseHamiltonian, set by pao_hamiltonian
        self._mesh_plan = {}  # parameters recorded for the fused mesh pass
        self._window = None  # (emin, emax, margin, ehi) once energy_window ran

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
            self.H = SparseHamiltonian.from_data_controller(
                self.data_controller, self.threshold, rcut=self.rcut
            )
            # the dense source must not outlive the conversion
            del arrays['HRs']
            arrays.pop('Hks', None)
            arrays.pop('Dnm', None)  # carried per bond by the container
            if self.rank == 0:
                if self.rcut is not None:
                    print(
                        'Sparse conversion: real-space cutoff rcut = %.3f Bohr applied at the '
                        'base cell, together with threshold = %.1e eV (both are folded into '
                        'the eigenvalue bound below).' % (self.rcut, self.threshold)
                    )
                print(self.H.stats_line())

        self._guard('sparse_conversion', _convert)
        self._pao.report_module_time('Sparse conversion')

    # ------------------------------------------------------------------
    # Doubling (purely sparse)
    # ------------------------------------------------------------------

    def _preflight_doubling(self, nx, ny, nz, mem_budget_gb=None, force=False):
        """Project the cost of doubling before allocating anything.

        ``nx, ny, nz`` are doubling *exponents*: the cell multiplier is
        ``N = 2**(nx+ny+nz)``, so 4,4,4 is 64x the size of 2,2,2, not 2x.
        That is easy to misjudge, and the failure mode without a gate is an
        OOM kill minutes into an HPC job with no diagnostic. This raises in
        under a second instead, with the projected numbers and the exits.

        The bond list is replicated on every rank (doubling is deterministic
        and needs no communication), so the budget is per rank and *more
        ranks on a node makes the fit worse, not better*.
        """
        from .sparse.solver import DENSE_RATIO, select_method

        attr = self.data_controller.data_attributes
        proj = self.H.project_doubling(nx, ny, nz)
        gb = 1024.0**3

        local_ranks = _node_local_ranks(self.comm)
        avail = _available_memory_bytes()
        if mem_budget_gb is not None:
            budget = float(mem_budget_gb) * gb
            budget_src = 'mem_budget_gb=%.1f (caller)' % mem_budget_gb
        elif avail is not None:
            budget = 0.8 * avail / local_ranks
            budget_src = '80%% of MemAvailable (%.1f GB) over %d rank(s) on this node' % (
                avail / gb,
                local_ranks,
            )
        else:
            budget = 8.0 * gb
            budget_src = 'default 8.0 GB (MemAvailable unreadable)'

        # nev if energy_window() is never called: doubling_attr_arry doubles
        # attr['bnd'] once per doubling.
        bnd_final = int(attr['bnd']) * proj['N']
        try:
            solver_note = 'dispatch: %s' % (
                select_method(proj['nawf'], bnd_final, method=self.solver)[0].upper()
            )
        except NotImplementedError:
            solver_note = (
                'solve REFUSED at nev=bnd=%d (%.0f%% of n, past the %.0f%% iterative '
                'regime, and n > dense_n_max) — energy_window() would have to bring '
                'nev under %d for this to run'
                % (
                    bnd_final,
                    100.0 * bnd_final / proj['nawf'],
                    100.0 * DENSE_RATIO,
                    int(DENSE_RATIO * proj['nawf']),
                )
            )

        report = (
            'Doubling projection for nx,ny,nz = %d,%d,%d  (N = 2^%d = %d cells)\n'
            '  nawf        %d -> %d\n'
            '  bonds       %.3gM -> %.3gM  (doubling replicates each bond exactly 2x per step)\n'
            '  peak/rank   %.2f GB during hermitize   [budget %.2f GB: %s]\n'
            '  steady/rank %.2f GB after compact()\n'
            '  dense H(k)  %.2f GB per k-point;  %s'
            % (
                nx,
                ny,
                nz,
                proj['d'],
                proj['N'],
                self.H.nawf,
                proj['nawf'],
                self.H.nnz / 1e6,
                proj['nnz'] / 1e6,
                proj['peak_bytes'] / gb,
                budget / gb,
                budget_src,
                proj['steady_bytes'] / gb,
                proj['dense_hk_bytes'] / gb,
                solver_note,
            )
        )

        if proj['peak_bytes'] > budget and not force:
            exits = []
            if self.rcut is None:
                exits.append(
                    'set rcut (Bohr) in the constructor: the bond list is currently '
                    'untruncated in real space, which is usually the largest single factor'
                )
            if proj['d'] > 1:
                exits.append(
                    'reduce the doubling count — d = nx+ny+nz is an exponent, so d-1 '
                    'halves every number above (%.2f GB peak)' % (proj['peak_bytes'] / 2 / gb)
                )
            if local_ranks > 1:
                exits.append(
                    'run fewer ranks per node: the bond list is replicated, so %d ranks '
                    'here each need the full %.2f GB' % (local_ranks, proj['peak_bytes'] / gb)
                )
            exits.append(
                'raise the budget explicitly with '
                'doubling_Hamiltonian(..., mem_budget_gb=...) or bypass with force=True '
                'if this projection is wrong for your machine'
            )
            raise RuntimeError(
                '%s\n\nProjected peak exceeds the budget by %.1fx. Refusing to start; '
                'nothing has been allocated.\nExits:\n  - %s'
                % (report, proj['peak_bytes'] / budget, '\n  - '.join(exits))
            )

        if self.rank == 0 and attr['verbose']:
            print(report, flush=True)
        return proj

    def doubling_Hamiltonian(self, nx, ny, nz, mem_budget_gb=None, force=False):
        """Double the cell ``nx``/``ny``/``nz`` times along each lattice
        vector by index arithmetic on the bond list (never dense), then
        Hermitize once — the bond-level equivalent of the per-k
        Hermitizations the dense pipeline applies downstream.

        ``nx``/``ny``/``nz`` are doubling counts, so the cell multiplier is
        ``2**(nx+ny+nz)``.  A pre-flight projection refuses sizes that
        cannot fit rather than letting them OOM part-way; see
        :meth:`_preflight_doubling` for ``mem_budget_gb`` and ``force``.
        """
        from .hamiltonian.do_doubling import doubling_attr_arry
        from .sparse.doubling import double_axis

        self._require_H('doubling_Hamiltonian')
        arrays, attr = self.data_controller.data_dicts()
        if self._window is not None:
            raise RuntimeError(
                'SparsePAOFLOW: energy_window() ran before doubling_Hamiltonian(). '
                "doubling_attr_arry doubles attr['bnd'] on every call, which would scale the "
                'window-sized nev by the cell multiplier. Call energy_window() after doubling.'
            )
        self._preflight_doubling(nx, ny, nz, mem_budget_gb=mem_budget_gb, force=force)
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
            # the bond list is final here: release the raw arrays the
            # assembly plan duplicates (about half the steady-state bytes)
            self.H.compact()
            if self.rank == 0:
                print(self.H.stats_line())

        self._guard('doubling_Hamiltonian', _double)
        self._pao.report_module_time('doubling_Hamiltonian')

    # ------------------------------------------------------------------
    # Energy window: size nev from the property range instead of bnd
    # ------------------------------------------------------------------

    def energy_window(self, emin, emax, margin=1.0, nprobe=16, nev=None):
        """Size the per-k solve from the property energy range.

        MUST be called after ``doubling_Hamiltonian()`` and before
        ``bands()`` / ``dos()`` / ``transport()``.  Sets ``attr['bnd']``,
        which every downstream band-diagonal consumer reads, so the band
        path and the mesh both pick the new width up automatically.

        The window top is ``ehi = emax + margin``.  ``margin`` (eV) has to
        cover the adaptive smearing tail (Yates widths here are
        <~ 0.22 eV, so 4 sigma is <~ 0.9 eV) and the transport occupation
        derivative at 300 K (~0.1 eV); 1.0 eV covers both.

        ``nev`` is probed by counting eigenvalues below ``ehi`` at
        ``nprobe`` deterministic k-points (Gamma, the supercell-BZ
        corners, then strided mesh points) and padding by
        ``max(8, 2%)``.  Pass ``nev`` explicitly to skip the probe.

        Caveat, stated because it is easy to misread: this narrows the
        solve but does not make the workload iterative again.  The
        fraction of the spectrum below a fixed ``emax`` is scale
        invariant under folding, so ``nev/nawf`` stays put as the cell
        grows.  For a DOS-from-``emin`` run the dense branch is
        permanent; only an *interior* window (shift-invert near E_F,
        transport only) would change that, and that is a different
        solver.

        Note also that ``attr['bnd']`` changes meaning here, from "bands
        with projectability > pthr, times the cell multiplier" to "bands
        inside the property window".  The downstream normalizations are
        unaffected (``do_dos_adaptive``'s two ``bnd`` factors cancel;
        transport slices are ``bnd``-independent), but ``bands_*.dat``
        gains or loses columns, so band files are not column-comparable
        across runs with and without a window.
        """
        import itertools

        from .sparse.solver import count_below

        self._require_H('energy_window')
        arrays, attr = self.data_controller.data_dicts()
        ehi = float(emax) + float(margin)
        nawf = self.H.nawf

        def _window():
            if nev is not None:
                chosen, probed = int(nev), None
            else:
                from .utils.get_K_grid_fft import get_K_grid_fft_crystal

                kprobe = np.array(list(itertools.product((0.0, 0.5), repeat=3)))  # Gamma + corners
                extra = nprobe - len(kprobe)
                if extra > 0:
                    kgrid = get_K_grid_fft_crystal(attr['nk1'], attr['nk2'], attr['nk3'])
                    stride = max(1, len(kgrid) // extra)
                    kprobe = np.vstack((kprobe, kgrid[::stride][:extra]))

                probed = 0
                for ispin in range(self.H.nspin):
                    for kf in kprobe:
                        hk = self.H.assemble_hk(kf, ispin=ispin, sign=-1)
                        probed = max(probed, count_below(hk, ehi))
                chosen = min(nawf, probed + max(8, int(np.ceil(0.02 * probed))))

            old = attr.get('bnd', nawf)
            attr['bnd'] = chosen
            self._window = (float(emin), float(emax), float(margin), ehi)
            if self.rank == 0:
                print(
                    'Sparse energy window: [%.3f, %.3f] eV + %.3f margin -> ehi = %.3f eV.\n'
                    "  nev = %d of nawf = %d (was bnd = %d)%s.  attr['bnd'] now means "
                    "'bands inside the window', not 'projectable bands x cell multiplier'."
                    % (
                        emin,
                        emax,
                        margin,
                        ehi,
                        chosen,
                        nawf,
                        old,
                        ''
                        if probed is None
                        else '; probe found %d over %d k-points' % (probed, len(kprobe)),
                    ),
                    flush=True,
                )

        self._guard('energy_window', _window)
        self._pao.report_module_time('Energy window')

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
            do_bands_sparse(
                self.data_controller,
                self.H,
                nsel,
                verbose=attr['verbose'],
                method=self.solver,
                ehi=None if self._window is None else self._window[3],
            )
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

    def plan_pdos(self, emin=-10.0, emax=2.0, ne=1000):
        """Register PDOS accumulation *before* the mesh pass runs.

        The mesh is fused and streaming, so a PDOS consumer can only join
        while the pass is executing.  Asking for PDOS after some other
        property already triggered the mesh forces a full second pass.
        Call this ahead of the first property (or simply call ``dos()``
        before ``transport()``) to avoid that."""
        self._mesh_plan['pdos_spec'] = (emin, emax, ne)

    def _ensure_mesh(self, pdos_spec=None):
        """Run the fused mesh pass if its results are not yet available
        (or if PDOS accumulation is requested but was not part of the
        earlier pass, in which case the mesh is recomputed)."""
        from .sparse.mesh import run_mesh
        from .sparse.pdos import PdosConsumer

        arrays, attr = self.data_controller.data_dicts()
        if pdos_spec is None:
            pdos_spec = self._mesh_plan.get('pdos_spec')
        have = self._mesh_plan.get('executed', False)
        need_pdos = pdos_spec is not None and not self._mesh_plan.get('pdos_done', False)
        if have and not need_pdos:
            return
        if have and need_pdos and self.rank == 0:
            # loud and unconditional: this doubles the run time
            print(
                'WARNING: Sparse mesh is being re-run from scratch to accumulate PDOS, '
                'because the first property call did not request it. This costs a second '
                'full pass over the k-mesh. Call plan_pdos(emin, emax, ne) before the '
                'first property (or dos() before transport()) to fold PDOS into the '
                'original pass.',
                flush=True,
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
                method=self.solver,
                ehi=None if self._window is None else self._window[3],
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
