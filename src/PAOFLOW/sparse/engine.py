"""Sparse engine behind ``PAOFLOW(..., sparse=True)``.

:class:`SparseEngine` is not a driver.  :class:`PAOFLOW.PAOFLOW` stays the
only user-facing class and routes the methods marked ``@sparse_override``
here (see :mod:`PAOFLOW.sparse.dispatch`); the methods have the same names
and accept the same arguments as their dense counterparts.  The pipeline
stages (Hamiltonian, doubling, interpolation, bands) are methods of the
engine; the properties are :class:`~PAOFLOW.sparse.properties.MeshProperty`
classes, which the engine finds by name in the registry.  Features that
exist only in sparse mode (``fused``, the bond list ``H``) are reached as
``pao.sparse.<name>``; the energy windows are the ``'energy_window'`` and
``'interior_window'`` options of ``sparse_config``.

The DFT input stages (QE parsing, projectability, base-cell Hamiltonian
construction — the one sanctioned dense stage, at the small pre-doubling
``nawf``) run on the dense code; the engine takes over from the moment the
bond list exists.  Dense methods without a sparse counterpart raise
``NotImplementedError`` loudly — there is no silent fallback to dense
arrays.  For those, hand the base-cell model to the dense pipeline with
``pao.to_dense()``, or restart a dense run from the same archive with
``load_sparse_hamiltonian``.

The mesh stages ``pao_eigh``, ``gradient_and_momenta`` and
``adaptive_smearing`` are fused into one streaming pass that the first
property runs on demand, so they are optional here; calling them only
records their parameters.  A later property that needs something the pass
did not provide (a streaming consumer, the band curvature) costs another
pass; ``with pao.sparse.fused():`` collects several properties into one.

Memory contract (see :mod:`PAOFLOW.sparse`): after ``pao_hamiltonian``
returns, no array of size O(nawf^2 * nk) exists; per-k dense workspace is
one eigenvector block, ``(nawf, nev)``, or ``(nawf, nawf)`` with the
per-k matrices built from it when a property needs the full spectrum
(only while ``nawf <= DENSE_N_MAX``).
"""

import functools
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

from .bridge import init_restart_session, sparsify
from .config import resolve_threshold
from .solver import DENSE_N_MAX
from .dispatch import call_dense
from .log import get_sparse_log


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


class SparseEngine:
    def __init__(self, host, config):
        """
        Arguments:
            host (PAOFLOW): the driver that owns this engine.  The engine
            uses its ``data_controller``, ``comm``, ``rank``, timing and
            exception reporting, and the dense bodies of the stages it
            extends (:func:`~PAOFLOW.sparse.dispatch.call_dense`).

            config (SparseConfig): truncation, solver and resource settings,
            parsed from the ``sparse_config=`` dict by
            :meth:`~PAOFLOW.sparse.config.SparseConfig.parse`.
        """
        self.host = host
        self.config = config
        self.data_controller = host.data_controller
        self.comm = host.comm
        self.rank = host.rank
        if self.data_controller.data_attributes is None:
            # restart=True: the log needs the session before the archive is read
            init_restart_session(self.data_controller, self.comm, **host._session)
        attr = self.data_controller.data_attributes
        if attr.get('smearing') is None:
            # the mesh pass always produces adaptive widths
            attr['smearing'] = 'gauss'

        self._interior = None  # (elo, ehi) when an interior window is active
        self._skipped = []  # properties skipped because the window cannot support them
        self.H = None  # SparseHamiltonian, set by pao_hamiltonian
        self._Dnm = None  # base-cell Dnm, kept only for release_to_dense()
        self._mesh_plan = {}  # parameters recorded for the fused mesh pass
        self._mesh_passes = 0  # mesh passes run so far
        self._mesh_products = set()  # products ('d2Ed2k') stored by those passes
        self._fusing = None  # properties queued inside fused(), else None
        self._window = None  # the EnergyWindow once applied

        cfg = self.config
        self.log = get_sparse_log(self.data_controller)
        self.log.header(
            'Sparse run configuration',
            (
                ('output directory', attr['opath']),
                ('MPI ranks', self.comm.Get_size()),
                ('k-point pools', attr['npool']),
                ('hopping_threshold (eV)', _threshold_label(cfg)),
                ('rcut (Bohr)', 'none' if cfg.rcut is None else '%.3f' % cfg.rcut),
                ('bond_order', 'none' if cfg.bond_order is None else cfg.bond_order),
                ('H(k) solver', cfg.hk_solver),
                ('energy window', _window_label(cfg)),
                ('smearing', attr['smearing']),
                ('verbose', attr['verbose']),
            ),
        )

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _guard(self, tag, func):
        """Mirror the dense try/except + abort_on_exception convention."""
        attr = self.data_controller.data_attributes
        try:
            return func()
        except Exception as e:
            self.host.report_exception(tag)
            if attr.get('abort_on_exception', True):
                raise e

    def _require_H(self, caller):
        if self.H is None:
            raise RuntimeError(
                'sparse %s requires the sparse Hamiltonian; call pao_hamiltonian() or '
                'load_sparse_hamiltonian() first.' % caller
            )

    def _time(self, mname):
        self.host.report_module_time(mname)

    def require_base_cell(self, caller):
        """Raise if the bond list was already doubled.

        Stages that build PAO-basis data from the base-cell orbital map
        (``spin_operator``, ``orbital_operator``) must run before
        ``doubling_Hamiltonian``, which then extends their results; the
        orbital map is not doubled with the cell.
        """
        if self.H is not None and self.H._doubled:
            raise RuntimeError(
                'sparse %s: doubling_Hamiltonian() already ran. Call %s() at the base cell, '
                'before doubling; the doubling then extends its result block-diagonally, as '
                'in the dense pipeline.' % (caller, caller)
            )

    def require_operator(self, key, builder, caller):
        """The sparse operator ``arrays[key]``, building it at the base cell
        with ``builder`` if absent, or raising after doubling."""
        arrays = self.data_controller.data_arrays
        if key not in arrays:
            if self.H is not None and self.H._doubled:
                raise RuntimeError(
                    'sparse %s needs %r, which was not built before doubling_Hamiltonian(). '
                    'Call %s() at the base cell, before doubling.' % (caller, key, builder.__name__)
                )
            builder()
        return arrays[key]

    # ------------------------------------------------------------------
    # Base-cell Hamiltonian (dense input stage, then the bond list)
    # ------------------------------------------------------------------

    def pao_hamiltonian(self, *args, **kwargs):
        """Build the base-cell PAO Hamiltonian with the dense
        ``pao_hamiltonian`` (sanctioned input stage, same arguments) and
        immediately convert it to the sparse bond list; the dense
        ``HRs``/``Hks`` are deleted before returning."""
        call_dense(self.host, 'pao_hamiltonian', *args, **kwargs)
        cfg = self.config

        def _convert():
            arrays, _ = self.data_controller.data_dicts()
            self.H = sparsify(
                self.data_controller,
                cfg.hopping_threshold,
                rcut=cfg.rcut,
                bond_order=cfg.bond_order,
            )
            # the dense source must not outlive the conversion
            del arrays['HRs']
            arrays.pop('Hks', None)
            # carried per bond by the container; the (nawf, nawf, 3) base-cell
            # copy is kept aside only so release_to_dense() can hand it back
            self._Dnm = arrays.pop('Dnm', None)
            self.log.section('Base-cell conversion (dense H(R) -> sparse bond list)')
            self._log_truncation()

        self._guard('sparse_conversion', _convert)
        self._time('Sparse conversion')

    def _log_truncation(self):
        """Describe the truncation carried by ``self.H`` and its error bound."""
        report = self.H.drop_report
        rcut = report.get('rcut')
        if rcut is not None:
            how = (
                'neighbour shell %d -> rcut = %.3f Bohr' % (report['bond_order'], rcut)
                if report.get('bond_order') is not None
                else 'rcut = %.3f Bohr, snapped to the shell gap at %.4f Bohr'
                % (report.get('rcut_requested', rcut), rcut)
            )
            if report.get('threshold', 0.0) > 0.0:
                how += ', threshold = %.1e eV' % report['threshold']
            self.log.write(
                'Real-space cutoff (%s) applied at the base cell. It is folded\n'
                'into the eigenvalue bound below.' % how
            )
        else:
            self.log.write(
                'Element threshold = %.1e eV applied at the base cell; no real-space '
                'cutoff. An element threshold does not preserve symmetry: degenerate\n'
                'bands can split by up to the eigenvalue bound below.'
                % report.get('threshold', self.H.threshold)
            )
        self.log.write(self.H.stats_line())

    # ------------------------------------------------------------------
    # Persistence of the base-cell bond list
    # ------------------------------------------------------------------

    def save_sparse_hamiltonian(
        self, fname='sparse_hamiltonian.npz', hopping_threshold=None, bond_order=None, rcut=None
    ):
        """Write the base-cell bond list and run metadata to ``fname``.

        Must be called after ``pao_hamiltonian()`` (or
        ``load_sparse_hamiltonian()``) and before ``doubling_Hamiltonian()``:
        only the base cell has a well-defined bond geometry and orbital
        map.  Relative names resolve inside the output directory.  The
        archive also serves as a labelled dataset; see
        :func:`PAOFLOW.sparse.io.bond_table`.

        The truncation arguments of the dense method are refused: the bond
        list already carries the truncation set by the ``sparse_config`` options.
        """
        from .io import write_sparse_hamiltonian

        if (hopping_threshold, bond_order, rcut) != (None, None, None):
            raise ValueError(
                'save_sparse_hamiltonian: in a sparse run the bond list is already truncated '
                'by the sparse_config options (hopping_threshold/rcut/bond_order); do not pass '
                'them here.'
            )
        self._require_H('save_sparse_hamiltonian')

        def _save():
            if self.rank == 0:
                path = write_sparse_hamiltonian(self.data_controller, self.H, fname, Dnm=self._Dnm)
                self.log.section('Saved sparse Hamiltonian')
                self.log.field('file', path)
                self.log.field('bonds', self.H.nnz)

        self._guard('save_sparse_hamiltonian', _save)
        self.comm.Barrier()
        self._time('save_sparse_hamiltonian')

    def load_sparse_hamiltonian(self, fname='sparse_hamiltonian.npz'):
        """Restart from an archive written by ``save_sparse_hamiltonian``.

        Replaces the input stages (``read_atomic_proj_QE`` / ``projections``,
        ``projectability``, ``pao_hamiltonian``): the bond list is read
        directly, no dense ``HRs`` is rebuilt, and the run continues with
        ``doubling_Hamiltonian`` or any property.  Use on a driver
        created with ``restart=True``.  The truncation is the one recorded
        in the file; the ``sparse_config`` truncation options do not apply.
        Every rank reads the file.
        """
        from os.path import exists, isabs, join

        from .bridge import archive_Dnm
        from .io import read_sparse_hamiltonian, restore_data_controller

        attr = self.data_controller.data_attributes

        def _load():
            path = fname
            if not isabs(fname) and not exists(fname):
                path = join(attr['opath'], fname)
            self.H, bundle = read_sparse_hamiltonian(path)
            restore_data_controller(self.data_controller, bundle)
            self._Dnm = archive_Dnm(bundle)
            self.log.section('Loaded sparse Hamiltonian (%s)' % path)
            self._log_truncation()

        self._guard('load_sparse_hamiltonian', _load)
        self._time('load_sparse_hamiltonian')

    def run_base_cell(self, name, body, modifies=True):
        """Run a dense base-cell ``H(R)`` transformation on the bond list.

        Parameters
        ----------
        name : str
            The PAOFLOW method (for messages).
        body : callable
            The dense method body, bound to its arguments.
        modifies : bool, optional
            Whether ``body`` changes ``HRs``; if not, the bond list is kept.

        Returns
        -------
        object
            Whatever ``body`` returns.

        Raises
        ------
        RuntimeError
            After doubling, interpolation or an energy window: the dense
            body would act on a base-cell ``HRs`` that no longer describes
            the run.

        Notes
        -----
        The bond list is scattered into a dense base-cell ``HRs`` (with
        ``Hks`` and ``Dnm``) by :func:`~PAOFLOW.sparse.bridge.densify`, the
        dense body runs unchanged, and the result is converted back by
        :func:`~PAOFLOW.sparse.bridge.sparsify` with the same
        ``sparse_config`` truncation, which is applied again to the
        transformed ``H(R)`` (logged).  The dense arrays are deleted
        afterwards.  The operators ``Sj``/``Lj`` are handed to the body as
        ndarrays and converted back; if the body changed ``nawf`` (ad-hoc
        spin-orbit), operators of the old basis are dropped.  A mesh pass
        that already ran described the old Hamiltonian, so its results are
        discarded.
        """
        from .bridge import densify
        from .operators import OPERATOR_KEYS, to_dense_operator, to_sparse_operator

        self._require_H(name)
        arrays, attr = self.data_controller.data_dicts()
        grid = (attr.get('nk1'), attr.get('nk2'), attr.get('nk3'))
        if (
            self.H._doubled
            or grid != tuple(self.H.nk_grid)
            or self._window is not None
            or self._interior is not None
        ):
            raise RuntimeError(
                'sparse %s transforms the base-cell H(R): call it after pao_hamiltonian() or '
                'load_sparse_hamiltonian() and before doubling_Hamiltonian(), '
                'interpolated_hamiltonian() and the energy windows.' % name
            )
        nnz_before = self.H.nnz
        nawf_before = self.H.nawf
        for key in OPERATOR_KEYS:
            if key in arrays:
                arrays[key] = to_dense_operator(arrays[key])
        densify(self.data_controller, self.H, Dnm=self._Dnm)
        cfg = self.config

        def _run():
            try:
                return body()
            finally:
                if modifies:
                    self.H = sparsify(
                        self.data_controller,
                        cfg.hopping_threshold,
                        rcut=cfg.rcut,
                        bond_order=cfg.bond_order,
                    )
                arrays.pop('HRs', None)
                arrays.pop('Hks', None)
                self._Dnm = arrays.pop('Dnm', None)
                for key in OPERATOR_KEYS:
                    if key not in arrays:
                        continue
                    if np.shape(arrays[key])[-1] == self.H.nawf:
                        arrays[key] = to_sparse_operator(arrays[key])
                    else:
                        del arrays[key]

        result = _run()
        if modifies:
            self._mesh_passes = 0
            self._mesh_products = set()
            for key in ('E_k', 'velkp', 'deltakp', 'd2Ed2k'):
                arrays.pop(key, None)
            self.log.section('%s (base cell, through the dense body)' % name)
            self.log.write(
                'The %d-bond list was scattered into a dense HRs, %s ran on it, and the result was\n'
                'converted back with the same truncation: %d -> %d bonds, nawf %d -> %d.'
                % (nnz_before, name, nnz_before, self.H.nnz, nawf_before, self.H.nawf)
            )
            self._log_truncation()
        return result

    def release_to_dense(self):
        """Scatter the base-cell bond list into the dense arrays and give it up.

        Called by ``PAOFLOW.to_dense()``, which then drops this engine.
        Leaves the shared ``DataController`` as dense ``pao_hamiltonian()``
        would with the truncated model (``HRs``, ``Hks``, ``Dnm``).
        """
        from .bridge import densify

        self._require_H('to_dense')
        if self.H._doubled:
            raise RuntimeError(
                'to_dense: doubling_Hamiltonian() already ran. Hand the base cell over before '
                'doubling and call doubling_Hamiltonian() on the dense pipeline.'
            )
        attr = self.data_controller.data_attributes
        grid = (attr['nk1'], attr['nk2'], attr['nk3'])
        if grid != self.H.nk_grid or self._window is not None or self._interior is not None:
            # sparse interpolation and the windows only rewrite attributes the
            # dense pipeline would then read against a base-grid HRs
            raise RuntimeError(
                'to_dense: call it before interpolated_hamiltonian() and the first solve; '
                'the mesh is now %dx%dx%d but the bond list is on the '
                '%dx%dx%d base grid. Interpolate on the dense pipeline instead.'
                % (grid + self.H.nk_grid)
            )

        cfg = self.config
        if cfg.energy_window is not None or cfg.interior_window is not None:
            message = (
                'WARNING: to_dense: the %s of sparse_config does not apply to the dense '
                'pipeline; the run continues without it.'
                % ('energy_window' if cfg.energy_window is not None else 'interior_window')
            )
            if self.rank == 0:
                print(message, flush=True)
            self.log.write(message)
        nnz = self.H.nnz
        densify(self.data_controller, self.H, Dnm=self._Dnm)
        from .operators import OPERATOR_KEYS, to_dense_operator

        arrays = self.data_controller.data_arrays
        for key in OPERATOR_KEYS:
            if key in arrays:
                arrays[key] = to_dense_operator(arrays[key])
        self.H, self._Dnm = None, None
        self.log.section('Handoff to the dense pipeline')
        self.log.write(
            'The %d-bond list was scattered into a dense HRs (%d x %d x %d x %d x %d x %d);\n'
            'the run continues on the dense pipeline of the same PAOFLOW object.'
            % ((nnz,) + self.data_controller.data_arrays['HRs'].shape)
        )
        self._time('Handoff to dense')

    # ------------------------------------------------------------------
    # Doubling (purely sparse)
    # ------------------------------------------------------------------

    def _preflight_doubling(self, nx, ny, nz):
        """Project the cost of doubling before allocating anything.

        ``nx, ny, nz`` are doubling *exponents*: the cell multiplier is
        ``N = 2**(nx+ny+nz)``, so 4,4,4 is 64x the size of 2,2,2, not 2x.
        That is easy to misjudge, and without a projection the first sign
        of it is an OOM kill minutes into an HPC job.  The projection is
        always logged; when the peak exceeds 80% of ``MemAvailable`` shared
        over the ranks of the node, a warning with the ways to shrink it is
        printed and doubling goes ahead.

        The bond list is replicated on every rank (doubling is deterministic
        and needs no communication), so the memory is per rank and *more
        ranks on a node makes the fit worse, not better*.
        """
        from .solver import DENSE_RATIO, select_hk_solver

        cfg = self.config
        attr = self.data_controller.data_attributes
        proj = self.H.project_doubling(nx, ny, nz)
        gb = 1024.0**3

        local_ranks = _node_local_ranks(self.comm)
        avail = _available_memory_bytes()
        if avail is not None:
            budget = 0.8 * avail / local_ranks
            budget_note = '[%.2f GB available: 80%% of MemAvailable (%.1f GB) over %d rank(s)]' % (
                budget / gb,
                avail / gb,
                local_ranks,
            )
        else:
            budget = None
            budget_note = '[available memory unknown: MemAvailable unreadable]'

        # nev without an 'energy_window' option: doubling_attr_arry doubles
        # attr['bnd'] once per doubling.
        bnd_final = int(attr['bnd']) * proj['N']
        try:
            solver_note = 'dispatch: %s' % (
                select_hk_solver(proj['nawf'], bnd_final, hk_solver=cfg.hk_solver)[0].upper()
            )
        except NotImplementedError:
            solver_note = (
                'solve REFUSED at nev=bnd=%d (%.0f%% of n, past the %.0f%% iterative '
                'regime, and n > dense_n_max) — an energy_window in sparse_config would '
                'have to bring nev under %d for this to run'
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
            '  peak/rank   %.2f GB during hermitize   %s\n'
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
                budget_note,
                proj['steady_bytes'] / gb,
                proj['dense_hk_bytes'] / gb,
                solver_note,
            )
        )

        self.log.section('Doubling pre-flight projection')
        self.log.write(report)

        if budget is not None and proj['peak_bytes'] > budget:
            remedies = []
            if cfg.rcut is None and cfg.bond_order is None:
                remedies.append(
                    "set 'rcut' (Bohr) or 'bond_order' in sparse_config: the bond list is currently "
                    'untruncated in real space, which is usually the largest single factor'
                )
            if proj['d'] > 1:
                remedies.append(
                    'reduce the doubling count — d = nx+ny+nz is an exponent, so d-1 '
                    'halves every number above (%.2f GB peak)' % (proj['peak_bytes'] / 2 / gb)
                )
            if local_ranks > 1:
                remedies.append(
                    'run fewer ranks per node: the bond list is replicated, so %d ranks '
                    'here each need the full %.2f GB' % (local_ranks, proj['peak_bytes'] / gb)
                )
            message = (
                'WARNING: doubling_Hamiltonian may run out of memory: the projected peak of '
                '%.2f GB per rank is %.1fx the memory available to each rank. Doubling '
                'continues; to reduce the memory:%s'
                % (
                    proj['peak_bytes'] / gb,
                    proj['peak_bytes'] / budget,
                    ''.join('\n  - ' + r for r in remedies),
                )
            )
            if self.rank == 0:
                print(report + '\n' + message, flush=True)
            self.log.write('\n' + message)

        return proj

    def doubling_Hamiltonian(self, nx, ny, nz):
        """Double the cell ``nx``/``ny``/``nz`` times along each lattice
        vector by index arithmetic on the bond list (never dense), then
        Hermitize once — the bond-level equivalent of the per-k
        Hermitizations the dense pipeline applies downstream.

        ``nx``/``ny``/``nz`` are doubling counts, so the cell multiplier is
        ``2**(nx+ny+nz)``.  A pre-flight projection of the memory is
        logged first, and printed with a warning when it exceeds what the
        node has available; see :meth:`_preflight_doubling`.
        """
        from ..hamiltonian.do_doubling import doubling_attr_arry
        from .doubling import double_axis

        self._require_H('doubling_Hamiltonian')
        arrays, attr = self.data_controller.data_dicts()
        if self._window is not None:
            raise RuntimeError(
                "sparse: the 'energy_window' of sparse_config was already applied by an earlier "
                "solve. doubling_attr_arry doubles attr['bnd'] on every call, which would scale "
                'the window-sized nev by the cell multiplier. Call doubling_Hamiltonian() before '
                'bands() and the properties.'
            )
        self._preflight_doubling(nx, ny, nz)
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
            # assembly plan duplicates (most of the steady-state bytes)
            self.H.compact()
            self.log.section('Doubling (%d,%d,%d)' % (nx, ny, nz))
            self.log.write(self.H.stats_line())

        self._guard('doubling_Hamiltonian', _double)
        self._time('doubling_Hamiltonian')

    # ------------------------------------------------------------------
    # Energy windows (sparse_config options), applied before the first solve
    # ------------------------------------------------------------------

    def _ensure_window(self):
        """Apply the ``'energy_window'`` or ``'interior_window'`` option of
        ``sparse_config`` once, right before the first solve (``bands`` or
        the first property).

        Running it lazily puts it after ``doubling_Hamiltonian`` and
        ``interpolated_hamiltonian`` whatever the call order of the script,
        and before anything reads ``attr['bnd']`` or the interior state.  A
        no-op without either option or once applied.  See
        :class:`~PAOFLOW.sparse.config.EnergyWindow` and
        :class:`~PAOFLOW.sparse.config.InteriorWindow` for the semantics.
        """
        if self._window is not None or self._interior is not None:
            return
        if self.config.energy_window is not None:
            self._apply_energy_window(self.config.energy_window)
        elif self.config.interior_window is not None:
            self._apply_interior_window(self.config.interior_window)

    def _apply_energy_window(self, window):
        """Size ``nev`` (``attr['bnd']``) from the bottom of the spectrum up
        to ``window.ehi``, probing ``count_below`` unless ``window.nev`` is set."""
        import itertools

        from .solver import count_below

        arrays, attr = self.data_controller.data_dicts()
        ehi = window.ehi
        nawf = self.H.nawf

        def _window():
            if window.nev is not None:
                chosen, probed = window.nev, None
            else:
                from ..utils.get_K_grid_fft import get_K_grid_fft_crystal

                kprobe = np.array(list(itertools.product((0.0, 0.5), repeat=3)))  # Gamma + corners
                extra = window.nprobe - len(kprobe)
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
            self._window = window
            self.log.section('Energy window')
            self.log.field(
                'window (eV)',
                'bottom of the spectrum to %.3f + %.3f margin' % (window.emax, window.margin),
            )
            self.log.field('ehi (eV)', '%.3f' % ehi)
            self.log.field('nev', '%d of nawf = %d (was bnd = %d)' % (chosen, nawf, old))
            self.log.field(
                'probe',
                'skipped (nev given)'
                if probed is None
                else '%d bands found over %d k-points' % (probed, len(kprobe)),
            )
            self.log.write(
                "attr['bnd'] now means 'bands inside the window', not 'projectable bands\n"
                "x cell multiplier'; band files are not column-comparable with runs that\n"
                'have no window.'
            )

        self._guard('energy_window', _window)
        self._time('Energy window')

    def _apply_interior_window(self, window):
        """Solve inside ``[window.elo, window.ehi]`` from here on; properties
        that need states below the window are skipped with a warning."""
        self._interior = (window.elo, window.ehi)
        self._kT_margin = window.kT_margin_eV
        self._smear_margin = window.smear_margin_eV
        self.log.section('Interior energy window')
        self.log.field('window (eV)', '[%.3f, %.3f]' % self._interior)
        self.log.field('transport margin (eV)', '%.3f each side' % self._kT_margin)
        self.log.field('DoS smearing margin (eV)', '%.3f each side' % self._smear_margin)
        self.log.write(
            'Only states inside the window are computed; nothing below elo exists.\n'
            'DoS/PDoS ranges are clamped to the window, the transport chemical-potential\n'
            'scan is clamped to [elo+margin, ehi-margin], and properties that need the\n'
            'full occupied manifold (carrier density, Hall) are skipped with a warning.'
        )
        self._time('Interior window')

    def _skip(self, prop, reason):
        """Warn loudly, record, and let the caller move to the next property."""
        self._skipped.append((prop, reason))
        message = (
            'WARNING: sparse %s SKIPPED under the interior window [%.3f, %.3f] -- %s\n'
            '         No output was written for it; the run continues.'
            % (prop, self._interior[0], self._interior[1], reason)
        )
        if self.rank == 0:
            print(message, flush=True)
        self.log.write('\n' + message)

    def _check_smearing_margin(self, prop, emin, emax):
        """Warn if adaptive widths reach further past the window edge than the
        margin allowed -- i.e. if uncomputed states below ``elo`` are leaking
        into what was plotted."""
        attr = self.data_controller.data_attributes
        dmax = float(attr.get('sparse_interior_dmax', 0.0))
        if dmax <= 0.0:
            return
        elo, ehi = self._interior
        reach = 4.0 * dmax  # 4 sigma -> ~1e-7 relative tail
        gap = min(float(emin) - elo, ehi - float(emax))
        if reach > gap:
            message = (
                'WARNING: sparse %s may be contaminated near the interior-window edges.\n'
                '         Measured max adaptive width %.3f eV reaches %.3f eV (4 sigma), but '
                'the plotted\n         range [%.3f, %.3f] is only %.3f eV inside the window '
                '[%.3f, %.3f].\n         States below elo were never computed, so their '
                'smearing tails are missing.\n'
                '         Widen the window or raise smear_margin_eV to at least %.3f.'
                % (prop, dmax, reach, emin, emax, gap, elo, ehi, reach)
            )
            if self.rank == 0:
                print(message, flush=True)
            self.log.write('\n' + message)

    def _check_window_covers(self, prop, emax):
        """Raise if ``emax`` lies above the lowest top band of an
        energy window: some k-point would be missing states inside the
        requested range.  Collective.

        A no-op without an energy window, where the solve keeps the dense
        ``bnd`` bands and every dense kernel reused here sums over the same
        ones, and under an interior window, whose solve is complete inside
        it by construction."""
        if self._window is None or self._interior is not None:
            return
        arrays, attr = self.data_controller.data_dicts()
        top = self.comm.allreduce(float(np.min(arrays['E_k'][:, -1, :])), op=MPI.MIN)
        if float(emax) > top:
            raise RuntimeError(
                'sparse %s: requested emax=%.3f eV exceeds the lowest computed top band '
                '(%.3f eV); the %d-band window does not cover the energy range. Raise emax or '
                "margin in the 'energy_window' of sparse_config." % (prop, emax, top, attr['bnd'])
            )

    def _clamp_to_window(self, prop, emin, emax, margin=0.0):
        """Intersect a requested range with the interior window.

        Returns ``(emin, emax)`` clamped, or ``None`` if nothing usable is
        left -- in which case the caller skips the property.
        """
        elo, ehi = self._interior
        lo, hi = max(float(emin), elo + margin), min(float(emax), ehi - margin)
        if hi <= lo:
            self._skip(
                prop,
                'the requested range [%.3f, %.3f] does not overlap the usable part of the '
                'window [%.3f, %.3f]' % (emin, emax, elo + margin, ehi - margin),
            )
            return None
        if (lo, hi) != (float(emin), float(emax)):
            message = (
                'WARNING: sparse %s range clamped from [%.3f, %.3f] to [%.3f, %.3f] by '
                'the interior window; states outside it were never computed.'
                % (prop, emin, emax, lo, hi)
            )
            if self.rank == 0:
                print(message, flush=True)
            self.log.write('\n' + message)
        return lo, hi

    # ------------------------------------------------------------------
    # Bands along a high-symmetry path
    # ------------------------------------------------------------------

    def bands(
        self,
        ibrav=None,
        band_path=None,
        high_sym_points=None,
        adhoc_SO=False,
        fname='bands',
        nk=500,
    ):
        """Band structure along a path; computes only the lowest
        ``attr['bnd']`` bands (set by the ``'energy_window'`` option).  Output
        format matches the dense ``bands_{ispin}.dat``."""
        from ..utils.communication import gather_full
        from .bands import do_bands_sparse

        if adhoc_SO:
            raise NotImplementedError(
                'sparse bands: adhoc_SO has no sparse implementation; call to_dense() on the '
                'base cell first.'
            )
        self._require_H('bands')
        self._ensure_window()
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

        def _bands():
            do_bands_sparse(
                self.data_controller,
                self.H,
                attr['bnd'],
                verbose=attr['verbose'],
                hk_solver=self.config.hk_solver,
                ehi=None if self._window is None else self._window.ehi,
                interior=self._interior,
            )
            E_kp = gather_full(arrays['E_k'], attr['npool'])
            self.data_controller.write_bands(fname, E_kp)

        self._guard('bands', _bands)
        self._time('Bands')

    # ------------------------------------------------------------------
    # Fourier interpolation to a finer k-mesh (pure metadata)
    # ------------------------------------------------------------------

    def interpolated_hamiltonian(self, nfft1=0, nfft2=0, nfft3=0, reshift_Ef=False, free_HRs=True):
        """Interpolate onto a finer k-mesh.  Zero-padding H(R) adds only
        zero hoppings, and the bond-list assembly already uses the
        Hermiticity-preserving Nyquist-split convention of
        ``utils.zero_pad`` — so sparse interpolation is exact and free:
        only the mesh dimensions change, no new data is created.
        Arguments of 0 default to twice the current grid (as dense).
        ``free_HRs`` has no meaning here (there is no ``HRs``)."""
        if reshift_Ef:
            raise NotImplementedError(
                'sparse interpolated_hamiltonian: reshift_Ef needs the Fermi level of the whole '
                'mesh before any eigensolve, which the fused sparse pass does not provide.'
            )
        self._require_H('interpolated_hamiltonian')
        arrays, attr = self.data_controller.data_dicts()
        nfft = [
            nfft1 if nfft1 > 0 else 2 * attr['nk1'],
            nfft2 if nfft2 > 0 else 2 * attr['nk2'],
            nfft3 if nfft3 > 0 else 2 * attr['nk3'],
        ]
        attr['nk1'], attr['nk2'], attr['nk3'] = nfft
        attr['nkpnts'] = nfft[0] * nfft[1] * nfft[2]
        self.log.section('Interpolation')
        self.log.write(
            'Property mesh set to %d x %d x %d (no new data — the bond list is simply\n'
            'evaluated on the finer grid; zero-padding H(R) is exact here).' % tuple(nfft)
        )
        self._time('R -> k with Zero Padding')

    # ------------------------------------------------------------------
    # Fused mesh pass (eigenvalues + velocities + smearing widths + PDOS)
    # ------------------------------------------------------------------

    def pao_eigh(self, bval=0):
        """Optional: the mesh eigensolve is fused with velocities and every
        streaming property into one pass, executed by the first property
        call that needs it.  Only ``bval`` is recorded."""
        self.data_controller.data_attributes.setdefault('bval', bval)
        self._mesh_plan['eigh'] = True
        self.log.write('pao_eigh(): fused into the mesh pass run by the first property.')

    def gradient_and_momenta(
        self,
        band_curvature=False,
        nonlocal_velocity=None,
        nonlocal_velocity_inject=None,
        nonlocal_velocity_sign=None,
    ):
        """Optional: band-diagonal velocities are computed inside the fused
        mesh pass, and the momentum matrix is formed per k-point only for
        the properties that need it; the k-indexed ``pksp`` never exists.

        ``band_curvature=True`` records the ``d2Ed2k`` product, so the first
        mesh pass computes the band curvature as the dense
        ``gradient_and_momenta`` does (its interband sum needs the full
        spectrum per k-point; see :func:`~PAOFLOW.sparse.mesh.run_mesh`).
        """
        if nonlocal_velocity:
            raise NotImplementedError(
                'sparse gradient_and_momenta: nonlocal_velocity needs the pseudopotential '
                'projectors on the full k-grid, which the sparse engine does not build; call '
                'to_dense() on the base cell first.'
            )
        self._mesh_plan['velocities'] = True
        if band_curvature:
            if self._interior is not None:
                self._skip(
                    'gradient_and_momenta band_curvature',
                    'the band curvature sums over interband pairs that include every state, '
                    'and an interior window computes none outside it',
                )
            else:
                self._mesh_plan.setdefault('products', set()).add('d2Ed2k')
        self.log.write(
            'gradient_and_momenta(): fused into the mesh pass (band-diagonal velocities%s).'
            % (' + band curvature' if band_curvature and self._interior is None else '')
        )

    def adaptive_smearing(self, smearing='gauss', afac=None):
        """Record the adaptive-smearing type and prefactor for the fused mesh
        pass (Yates widths, as ``do_adaptive_smearing``; the interband
        widths ``deltakp2`` are formed per k-point for the properties that
        need them).  Optional: the mesh defaults to the run's smearing type."""
        if smearing not in ('gauss', 'm-p'):
            raise ValueError(
                "Smearing type %s not supported.\nSmearing types are 'gauss' and 'm-p'"
                % str(smearing)
            )
        self.data_controller.data_attributes['smearing'] = smearing
        self._mesh_plan['smearing'] = smearing
        self._mesh_plan['afac'] = afac

    @contextmanager
    def fused(self):
        """Run every property called inside the block in one mesh pass.

        Properties are queued instead of run; on a normal exit one mesh
        pass runs with the union of what they need (streaming consumers,
        stored products such as ``d2Ed2k``, the full spectrum), and then
        each property's post-pass step (file writes, the dense
        band-diagonal body) runs in call order.  If the block raises,
        nothing queued runs.  Outside a block every property call that needs
        something the last pass did not provide costs a full pass of its
        own::

            with pao.sparse.fused():
                pao.dos(emin=-12.0, emax=2.2)
                pao.transport(do_hall=True)
        """
        if self._fusing is not None:
            raise RuntimeError('sparse.fused(): blocks cannot be nested.')
        self._fusing = []
        try:
            yield self
        except BaseException:
            self._fusing = None
            raise
        queued, self._fusing = self._fusing, None
        if queued:
            self._execute(queued)

    def __getattr__(self, name):
        """Resolve a property method through the registry, so the
        :mod:`~PAOFLOW.sparse.properties` modules need no engine edits."""
        if name.startswith('_'):
            raise AttributeError(name)
        from .properties import _load

        cls = _load().get(name)
        if cls is None:
            raise AttributeError(
                "'SparseEngine' has no attribute %r (no sparse property of that name)" % name
            )
        return functools.partial(self._run_property, cls)

    def _run_property(self, cls, *args, **kwargs):
        """Build a registered property with the dense arguments and run it,
        or queue it inside :meth:`fused`."""
        self._require_H(cls.method)
        self._ensure_window()
        arrays, attr = self.data_controller.data_dicts()
        before = (dict(attr), dict(arrays))
        prop = cls(self, *args, **kwargs)
        if self._interior is not None:
            reason = prop.interior_reason
            if reason is not None:
                self._skip(cls.method, reason)
                return
        if not prop.prepare():
            return
        if self._interior is not None and 'd2Ed2k' in prop.products:
            # prepare() may drop it (transport skips only its Hall term)
            self._skip(
                cls.method,
                'it needs the band curvature, whose interband sum runs over every state, '
                'and an interior window computes none outside it',
            )
            return
        prop.remember_context(*before)
        if self._fusing is not None:
            self._fusing.append(prop)
            self.log.write('%s(): queued for the fused mesh pass.' % cls.method)
            return
        self._execute([prop])

    def _execute(self, props):
        """One mesh pass and the path passes ``props`` need, then their
        post-pass steps in call order."""
        mesh = [p for p in props if not p.path and p.uses_mesh]
        if mesh:
            consumers = [p for p in mesh if p.streaming]
            products = set().union(*(p.products for p in mesh))
            self._ensure_mesh(products=products, consumers=consumers)
        for with_dnm in (True, False):
            group = [p for p in props if p.path and p.with_dnm == with_dnm]
            if group:
                self._run_path(group, with_dnm)
        for p in props:
            p.activate()
            self._guard(p.method, lambda p=p: p.finalize(self.data_controller))
            if p.label:
                self._time(p.label)

    def _run_path(self, consumers, with_dnm):
        """One pass over the band path for path properties (never cached:
        nothing is stored per path point)."""
        from .bands import run_path

        attr = self.data_controller.data_attributes
        full = any('full_spectrum' in c.needs for c in consumers)
        if full and self.H.nawf > DENSE_N_MAX:
            raise NotImplementedError(
                'sparse %s: the interband sums run over every state, which needs the full '
                'spectrum at each path point (nawf = %d > DENSE_N_MAX = %d). A Sternheimer '
                'solve for them is not implemented yet.'
                % (', '.join(sorted({c.method for c in consumers})), self.H.nawf, DENSE_N_MAX)
            )
        nsolve = self.H.nawf if full else int(attr['bnd'])

        def _path():
            run_path(
                self.data_controller,
                self.H,
                consumers,
                nsolve,
                with_dnm=with_dnm,
                hk_solver=self.config.hk_solver,
                verbose=attr['verbose'],
            )

        self._guard('sparse_path', _path)
        self._time('Sparse path pass')

    def _ensure_mesh(self, products=(), consumers=()):
        """Run the fused mesh pass unless the last one already provides
        everything asked for.

        A pass is needed if none has run, if a requested product (or one
        recorded by ``gradient_and_momenta``) is missing, or if there are
        streaming consumers, which can only see the k-points while a pass
        runs.  A second pass is announced loudly, since it costs as much as
        the first; :meth:`fused` avoids it.  Products of earlier passes stay
        valid and are not recomputed.
        """
        from .mesh import run_mesh

        arrays, attr = self.data_controller.data_dicts()
        wanted = set(products) | self._mesh_plan.get('products', set())
        if self._interior is not None and 'd2Ed2k' in self._mesh_plan.get('products', set()):
            # recorded by gradient_and_momenta(band_curvature=True) before
            # the interior window was applied
            self._mesh_plan['products'].discard('d2Ed2k')
            wanted.discard('d2Ed2k')
            self._skip(
                'gradient_and_momenta band_curvature',
                'the band curvature sums over interband pairs that include every state, '
                'and an interior window computes none outside it',
            )
        missing = wanted - self._mesh_products
        if self._mesh_passes and not missing and not consumers:
            return
        if self._mesh_passes:
            why = [p.method for p in consumers] + sorted(missing)
            message = (
                'WARNING: Sparse mesh pass %d is being run from scratch for %s, which the '
                'earlier pass did not provide. It costs as much as the first pass. Put the '
                'property calls in one `with pao.sparse.fused():` block to compute them in a '
                'single pass.' % (self._mesh_passes + 1, ', '.join(why))
            )
            if self.rank == 0:
                print(message, flush=True)
            self.log.write('\n' + message)

        nev = attr['bnd']

        def _mesh():
            run_mesh(
                self.data_controller,
                self.H,
                nev,
                consumers=consumers,
                products=missing,
                afac=self._mesh_plan.get('afac'),
                smearing=self._mesh_plan.get('smearing', attr['smearing']),
                verbose=attr['verbose'],
                hk_solver=self.config.hk_solver,
                ehi=None if self._window is None else self._window.ehi,
                interior=self._interior,
            )

        self._guard('sparse_mesh', _mesh)
        self._mesh_passes += 1
        self._mesh_products |= missing
        self._time('Sparse mesh pass')

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _report_skips(self):
        """Restate every skipped property at the end, where it cannot be lost
        in the scroll of a long run."""
        if not self._skipped:
            return
        lines = [
            '%d propert%s SKIPPED under the interior window:'
            % (len(self._skipped), 'y was' if len(self._skipped) == 1 else 'ies were')
        ]
        lines += ['  - %s: %s' % (p, r) for p, r in self._skipped]
        text = '\n'.join(lines)
        if self.rank == 0:
            print('\n' + text, flush=True)
        self.log.section('Skipped properties')
        self.log.write(text)

    def finish_execution(self):
        self._report_skips()
        call_dense(self.host, 'finish_execution')


def _window_label(cfg):
    """Header entry for the window option of a config."""
    if cfg.energy_window is not None:
        w = cfg.energy_window
        return 'from the bottom up to %.3f + %.3f eV%s' % (
            w.emax,
            w.margin,
            '' if w.nev is None else ' (nev = %d)' % w.nev,
        )
    if cfg.interior_window is not None:
        return 'interior [%.3f, %.3f] eV' % (cfg.interior_window.elo, cfg.interior_window.ehi)
    return 'none'


def _threshold_label(cfg):
    """Header entry for the element threshold a config actually applies."""
    threshold = resolve_threshold(cfg.hopping_threshold, cfg.rcut, cfg.bond_order)
    if threshold == 0.0 and (cfg.rcut is not None or cfg.bond_order is not None):
        return 'none (real-space cutoff)'
    return '%.3e' % threshold
