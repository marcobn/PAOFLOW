"""Sparse engine behind ``PAOFLOW(..., sparse=True)``.

:class:`SparseEngine` is not a driver.  :class:`PAOFLOW.PAOFLOW` stays the
only user-facing class and routes the methods marked ``@sparse_override``
here (see :mod:`PAOFLOW.sparse.dispatch`); the methods have the same names
and accept the same arguments as their dense counterparts.  The pipeline
stages (Hamiltonian, doubling, interpolation, bands) are methods of the
engine; the properties are :class:`~PAOFLOW.sparse.properties.MeshProperty`
classes, which the engine finds by name in the registry.  Features that
exist only in sparse mode (``fused``, the bond list ``H``) are reached as
``pao.sparse.<name>``.  The energy window of each pass is derived from the
energy ranges of its properties (:mod:`~PAOFLOW.sparse.window`); the
``'nev'`` and ``'interior_window'`` options of ``sparse_config`` override it.

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

Every property sees only well-projected states.  ``pao_hamiltonian`` keeps
the bands whose projectability passes ``pthr`` with their DFT energies, as
long as they lie below the shift ``eta`` (``attr['shift']``), and places
everything else in the PAO space at ``eta`` itself (around it off the DFT
grid).  The states from ``eta - shift_margin`` up are therefore treated as
placeholders, not bands.  The engine drops them at
every k-point (:meth:`SparseEngine._projectable_top`), caps every window at
``eta - shift_margin``, pulls a property range that reaches past it back
below it with a warning, and
leaves them only as the intermediate states of the full-spectrum sums,
which the dense kernels include as well.

Memory contract (see :mod:`PAOFLOW.sparse`): after ``pao_hamiltonian``
returns, no array of size O(nawf^2 * nk) exists; per-k dense workspace is
one eigenvector block, ``(nawf, nev)``, or ``(nawf, nawf)`` with the
per-k matrices built from it when a property needs the full spectrum
(only while ``nawf <= dense_n_max`` of ``sparse_config``).
"""

import functools
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

from . import window as _window
from .bridge import init_restart_session, sparsify
from .config import resolve_threshold
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

        self._interior = None  # (elo, ehi) of the last interior window
        self._skipped = []  # properties skipped because the window cannot support them
        self.H = None  # SparseHamiltonian, set by pao_hamiltonian
        self._Dnm = None  # base-cell Dnm, kept only for release_to_dense()
        self._mesh_plan = {}  # parameters recorded for the fused mesh pass
        self._mesh_passes = 0  # mesh passes run so far
        self._path_passes = 0  # band and path passes run so far
        self._mesh_products = set()  # products ('d2Ed2k') stored by those passes
        self._fusing = None  # properties queued inside fused(), else None
        self._bnd_projectable = None  # attr['bnd'] before the first window
        self._mesh_window = None  # PassWindow of the last mesh pass
        self._mesh_bnd = None  # attr['bnd'] the last mesh pass left (its E_k width)
        self._last_window = None  # PassWindow of the last pass of any kind
        self._probe_cache = {}  # window probe results, by probe kind
        self._warned = set()  # one-time warnings already issued
        self._ptop = None  # top of the well-projected range, set by _projectable_top
        self._ptop_known = False

        cfg = self.config
        self.log = get_sparse_log(self.data_controller)
        self.log.header(
            'Run',
            (
                ('output directory', attr['opath']),
                ('MPI ranks', self.comm.Get_size()),
                ('k-point pools', attr['npool']),
                ('smearing', attr['smearing']),
                ('verbose', attr['verbose']),
            ),
        )
        self.log.header('Sparse configuration', _config_fields(cfg))

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
            self.log.section('Hamiltonian (base cell)')
            self._log_truncation()

        self._guard('sparse_conversion', _convert)
        self._time('Sparse conversion')

    def _log_truncation(self):
        """Log the truncation carried by ``self.H`` and what it costs.

        The eigenvalue bound is the Gershgorin bound of everything the
        truncation dropped: no eigenvalue at any k-point moves by more.  A
        real-space cutoff keeps or drops whole atom-pair blocks by bond
        length and so preserves the crystal symmetry; an element threshold
        does not, and can split symmetry-protected degeneracies by up to
        the bound (see :func:`~PAOFLOW.sparse.config.resolve_threshold`).
        """
        report = self.H.drop_report
        st = self.H.stats()
        rows = []
        rcut = report.get('rcut')
        if rcut is not None:
            order = report.get('bond_order')
            requested = report.get('rcut_requested')
            if order is not None:
                value = '%.3f  (bond length of neighbour shell %d)' % (rcut, order)
            elif requested is not None and abs(float(requested) - rcut) > 5.0e-4:
                value = '%.3f requested, %.3f applied (moved into the gap above the last shell)' % (
                    float(requested),
                    rcut,
                )
            else:
                value = '%.3f' % rcut
            rows.append(('cutoff radius (Bohr)', value))
        threshold = float(report.get('threshold', self.H.threshold) or 0.0)
        if threshold > 0.0:
            rows.append(
                (
                    'hopping threshold (eV)',
                    '%.1e  (can split degeneracies by up to the eigenvalue shift below)'
                    % threshold,
                )
            )
        else:
            rows.append(('hopping threshold (eV)', 'none'))
        rows += [
            ('orbitals (nawf)', st['nawf']),
            ('R vectors', st['nR']),
            (
                'bonds kept',
                '%.3gM of %.3gM  (%.1f%%)'
                % (st['nnz'] / 1e6, st['total'] / 1e6, 100.0 * st['density']),
            ),
            ('bond list (MB)', '%.1f' % st['mbytes']),
            ('dense H(R) (MB)', '%.1f' % st['dense_mbytes']),
            ('max eigenvalue shift from truncation (eV)', '%.3g' % st['eig_bound']),
        ]
        self.log.rows(rows)

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
            self.log.section('Hamiltonian (loaded)')
            self.log.field('file', path)
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
            After doubling, interpolation or a solve: the dense body would
            act on a base-cell ``HRs`` that no longer describes the run.

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
        if self.H._doubled or grid != tuple(self.H.nk_grid) or self._solved():
            raise RuntimeError(
                'sparse %s transforms the base-cell H(R): call it after pao_hamiltonian() or '
                'load_sparse_hamiltonian() and before doubling_Hamiltonian(), '
                'interpolated_hamiltonian(), bands() and the properties.' % name
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
            self._bnd_projectable = None
            self._probe_cache = {}
            self._mesh_window = self._last_window = self._mesh_bnd = None
            self._interior = None
            for key in ('E_k', 'velkp', 'deltakp', 'd2Ed2k'):
                arrays.pop(key, None)
            self.log.section('%s (base cell)' % name)
            self.log.rows(
                (
                    ('bonds', '%d -> %d' % (nnz_before, self.H.nnz)),
                    ('orbitals (nawf)', '%d -> %d' % (nawf_before, self.H.nawf)),
                )
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
        if grid != self.H.nk_grid or self._solved():
            # sparse interpolation and the windows only rewrite attributes the
            # dense pipeline would then read against a base-grid HRs
            raise RuntimeError(
                'to_dense: call it before interpolated_hamiltonian() and the first solve; '
                'the mesh is now %dx%dx%d but the bond list is on the '
                '%dx%dx%d base grid. Interpolate on the dense pipeline instead.'
                % (grid + self.H.nk_grid)
            )

        cfg = self.config
        if cfg.nev is not None or cfg.interior_window is not None:
            self.log.warning(
                "to_dense: the '%s' option of sparse_config does not apply to the dense "
                'pipeline; the run continues without it'
                % ('nev' if cfg.nev is not None else 'interior_window')
            )
        nnz = self.H.nnz
        densify(self.data_controller, self.H, Dnm=self._Dnm)
        from .operators import OPERATOR_KEYS, to_dense_operator

        arrays = self.data_controller.data_arrays
        for key in OPERATOR_KEYS:
            if key in arrays:
                arrays[key] = to_dense_operator(arrays[key])
        self.H, self._Dnm = None, None
        self.log.section('Handoff to the dense pipeline')
        self.log.rows(
            (
                ('bonds', nnz),
                (
                    'dense H(R) shape',
                    ' x '.join(str(d) for d in self.data_controller.data_arrays['HRs'].shape),
                ),
            )
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
        from .solver import select_hk_solver

        cfg = self.config
        attr = self.data_controller.data_attributes
        proj = self.H.project_doubling(nx, ny, nz)
        gb = 1024.0**3

        local_ranks = _node_local_ranks(self.comm)
        avail = _available_memory_bytes()
        budget = None if avail is None else 0.8 * avail / local_ranks

        # doubling_attr_arry doubles attr['bnd'] once per doubling
        bnd_final = cfg.nev if cfg.nev is not None else int(attr['bnd']) * proj['N']
        if cfg.interior_window is not None:
            solver_note = 'interior solve'
        else:
            try:
                chosen = select_hk_solver(
                    proj['nawf'], bnd_final, hk_solver=cfg.hk_solver, **cfg.limits
                )[0]
                solver_note = 'dense (LAPACK)' if chosen == 'dense' else 'sparse (ARPACK)'
            except NotImplementedError:
                solver_note = (
                    'refused: nev = %d is %.0f%% of n (limit %.0f%% above dense_n_max); give '
                    "'nev' below %d or use 'interior_window'"
                    % (
                        bnd_final,
                        100.0 * bnd_final / proj['nawf'],
                        100.0 * cfg.dense_ratio,
                        int(cfg.dense_ratio * proj['nawf']),
                    )
                )

        if avail is not None:
            available = '%.2f  (80%% of MemAvailable %.1f GB, shared by %d rank%s on this node)' % (
                budget / gb,
                avail / gb,
                local_ranks,
                '' if local_ranks == 1 else 's',
            )
        else:
            available = 'unknown (MemAvailable unreadable)'
        self.log.section('Doubling (nx, ny, nz = %d, %d, %d)' % (nx, ny, nz))
        self.log.rows(
            (
                ('cells', '%d  (2^%d)' % (proj['N'], proj['d'])),
                ('orbitals (nawf)', '%d -> %d' % (self.H.nawf, proj['nawf'])),
                ('bonds', '%.3gM -> %.3gM' % (self.H.nnz / 1e6, proj['nnz'] / 1e6)),
                ('peak memory per rank (GB)', '%.2f' % (proj['peak_bytes'] / gb)),
                ('available per rank (GB)', available),
                ('steady memory per rank (GB)', '%.2f' % (proj['steady_bytes'] / gb)),
                ('dense H(k) per k-point (GB)', '%.2f' % (proj['dense_hk_bytes'] / gb)),
                ('H(k) solver after doubling', solver_note),
            )
        )

        if budget is not None and proj['peak_bytes'] > budget:
            remedies = []
            if cfg.rcut is None and cfg.bond_order is None:
                remedies.append("set 'rcut' or 'bond_order' in sparse_config")
            if proj['d'] > 1:
                remedies.append('reduce the doubling count (one step less halves the memory)')
            if local_ranks > 1:
                remedies.append('run fewer ranks per node (each rank holds the whole bond list)')
            self.log.warning(
                'doubling_Hamiltonian may run out of memory: peak %.2f GB per rank, %.2f GB '
                'available per rank; doubling continues%s'
                % (
                    proj['peak_bytes'] / gb,
                    budget / gb,
                    ''.join('; ' + r for r in remedies),
                )
            )

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
        if self._solved():
            raise RuntimeError(
                'sparse: call doubling_Hamiltonian() before bands() and the properties: a solve '
                "already sized attr['bnd'] for the undoubled cell."
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
            st = self.H.stats()
            self.log.rows(
                (
                    ('bonds after doubling', '%.3gM' % (st['nnz'] / 1e6)),
                    ('bond list after doubling (MB)', '%.1f' % st['mbytes']),
                    ('max eigenvalue shift from truncation (eV)', '%.3g' % st['eig_bound']),
                )
            )

        self._guard('doubling_Hamiltonian', _double)
        self._time('doubling_Hamiltonian')

    # ------------------------------------------------------------------
    # Energy window of each pass (see PAOFLOW.sparse.window)
    # ------------------------------------------------------------------

    def _solved(self):
        """Whether a mesh, path or band pass already ran.  The first pass
        fixes ``attr['bnd']`` for the cell it was run on."""
        return self._mesh_passes > 0 or self._path_passes > 0

    def _projectable_top(self):
        """Top of the well-projected energy range (eV), or ``None`` without one.

        ``pao_hamiltonian`` builds H(k) from the bands that pass the
        projectability threshold and lie below the shift ``eta``
        (``attr['shift']``), and puts the rest of the PAO space at ``eta``
        (``shift_type`` 0 and 1).  On the DFT k-grid those placeholders sit
        at ``eta`` exactly; at interpolated k-points they spread around it,
        so the real states of each k-point are taken to be the ones below
        ``eta - shift_margin`` (``sparse_config``).  States above it are
        dropped at every k-point, and windows and property ranges are kept
        below it.  With ``shift_type`` 2 the rest sits at zero instead,
        inside the spectrum, and cannot be told apart by energy; there, and
        for a Hamiltonian without a shift (a model, or a restart that did
        not record one), nothing is dropped.
        """
        if self._ptop_known:
            return self._ptop
        attr = self.data_controller.data_attributes
        shift = attr.get('shift')
        shift_type = attr.get('shift_type', 1)
        self._ptop_known = True
        self.log.section('Well-projected range')
        if shift_type == 2 or not isinstance(shift, (int, float, np.floating)):
            reason = (
                'shift_type 2 leaves the non-projectable states inside the spectrum'
                if shift_type == 2
                else 'no PAO shift is known for this Hamiltonian'
            )
            self.log.warning(
                'sparse: every solved state is used (%s), so states outside the well-projected '
                'bands cannot be told apart by energy' % reason
            )
            return None
        self._ptop = float(shift) - self.config.shift_margin
        self.log.rows(
            (
                ('PAO shift (eV)', '%.3f' % float(shift)),
                ('pthr', attr.get('pthr', 'n/a')),
                ('usable top (eV)', '%.3f  (PAO shift - shift_margin)' % self._ptop),
            )
        )
        return self._ptop

    def _warn_once(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            self.log.warning(message)

    def _clamp_to_projectable(self, prop, emin, emax, margin=0.0):
        """Keep the top of a property's energy range in the well-projected range.

        Parameters
        ----------
        prop : str
            Property name, for the message.
        emin, emax : float
            The requested range (eV).
        margin : float, optional
            How far below the projectable top ``emax`` has to stay: the
            states the property needs above ``emax`` (the transport
            occupation derivative, say) must be real too.

        Returns
        -------
        (float, float)
            ``(emin, emax)``, with ``emax`` lowered, and a warning printed,
            if it reached past ``eta - margin``.  Above it the PAO spectrum
            holds only the non-projectable states placed at the shift.

        Raises
        ------
        ValueError
            If nothing of the range is left below the projectable top.
        """
        top = self._projectable_top()
        if top is None or float(emax) <= top - margin:
            return emin, emax
        hi = top - margin
        if hi <= float(emin):
            raise ValueError(
                'sparse %s: the requested range [%.3f, %.3f] eV lies above the well-projected '
                'bands, which end at %.3f eV (the PAO shift; %.3f eV margin). Lower the range, '
                'or raise the shift with more bands or a lower pthr in projectability().'
                % (prop, emin, emax, top, margin)
            )
        self.log.warning(
            'sparse %s: emax lowered from %.3f to %.3f eV, the top of the well-projected bands'
            '%s; more bands or a lower pthr in projectability() raise it'
            % (prop, emax, hi, '' if margin == 0 else ' minus a %.3f eV margin' % margin)
        )
        return emin, hi

    def _afac_dk(self):
        """Adaptive-smearing prefactor and mesh spacing of the next mesh pass,
        as :func:`~PAOFLOW.sparse.mesh.run_mesh` computes them."""
        attr = self.data_controller.data_attributes
        afac = self._mesh_plan.get('afac')
        if afac is None:
            smearing = self._mesh_plan.get('smearing', attr['smearing'])
            afac = 1.0 if smearing == 'm-p' else 0.7
        nk = attr['nk1'] * attr['nk2'] * attr['nk3']
        dk = (8.0 * np.pi**3 / attr['omega'] / nk) ** (1.0 / 3.0)
        return float(afac), float(dk)

    def _probe(self, ptop, interior=None):
        """Eigenvalues and band-velocity norms at the probe k-points.

        :func:`~PAOFLOW.sparse.window.probe_points` (Gamma, the zone corners
        and generic points for the band velocities), shared over the ranks; each is solved by :func:`~PAOFLOW.sparse.mesh.probe_k`, from
        the bottom up to ``ptop`` or, with ``interior``, inside it.  A
        failure on one rank is raised on all, so none waits in the
        reduction.  Collective; cached until the Hamiltonian changes.

        Returns
        -------
        dict or None
            ``{'points': [(E, vnorm), ...], 'nk': int}`` over all ranks, or
            ``None`` when a from-the-bottom probe would densify H(k) above
            ``dense_n_max``.
        """
        from .mesh import probe_k

        key = 'bottom' if interior is None else ('interior',) + tuple(interior)
        if key in self._probe_cache:
            return self._probe_cache[key]
        cfg = self.config
        if interior is None and self.H.nawf > cfg.dense_n_max:
            self._probe_cache[key] = None
            return None
        kprobe = np.array(_window.probe_points())

        local, error = [], None
        nranks = self.comm.Get_size()
        try:
            for ispin in range(self.H.nspin):
                for kf in kprobe[self.rank :: nranks]:
                    local.append(
                        probe_k(
                            self.H,
                            kf,
                            ispin,
                            ptop,
                            interior=interior,
                            hk_solver=cfg.hk_solver,
                            dense_n_max=cfg.dense_n_max,
                        )
                    )
        except Exception as exc:
            error = exc
        if self.comm.allreduce(error is not None, op=MPI.LOR):
            raise error or RuntimeError(
                'sparse window probe: the probe failed on another rank; see its error.'
            )
        points = [point for part in self.comm.allgather(local) for point in part]
        result = {'points': points, 'nk': len(kprobe)}
        self._probe_cache[key] = result
        return result

    def _resolve_window(self, props, tag):
        """The energy window of a pass over ``props``.

        Collective.  The window covers the energy range of every property
        of the pass plus the margin it needs (see :mod:`PAOFLOW.sparse.window`),
        and never reaches past the well-projected top.  The smearing margin
        is applied only to a window the engine places itself: with a given
        ``nev`` or explicit interior bounds the property ranges are clipped
        to the window as given, and the widths measured on the mesh are
        checked after the pass.  From the bottom of
        the spectrum the band count is the projectable count when the window
        reaches that top, ``sparse_config['nev']`` when given, and otherwise
        the largest number of states below the window top at the probe
        k-points, padded; it is capped at the projectable count, since no
        k-point has more real states than that.  Nothing is written to the
        run data: the pass that uses the window sets ``attr['bnd']``.

        Parameters
        ----------
        props : sequence of MeshProperty
            The properties of the pass.
        tag : str
            Name of the pass, for the log.

        Returns
        -------
        PassWindow
        """
        cfg = self.config
        attr = self.data_controller.data_attributes
        ptop = self._projectable_top()
        if self._bnd_projectable is None:
            self._bnd_projectable = int(attr['bnd'])
        bnd = self._bnd_projectable
        nawf = self.H.nawf
        iw = cfg.interior_window

        raw = []
        for p in props:
            r = p.energy_range()
            if r is not None:
                emin, emax, kind, tmax = r
                raw.append((p.method, float(emin), float(emax), kind, tmax))
        if iw is not None and iw.auto and not raw:
            if self._last_window is not None and self._last_window.mode == 'interior':
                return self._last_window
            _window.derive_interior([], ptop)  # raises, naming the way out

        # the smearing margin needs the widths at the probe points; it places
        # a window the engine chooses, so it is not estimated with a given
        # nev or explicit interior bounds (the widths are still checked
        # after the pass)
        smeared = [r for r in raw if r[3] == _window.SMEARING]
        probe = dmax = None
        placed = cfg.nev is None if iw is None else iw.auto
        if smeared and placed:
            lo = min(r[1] for r in smeared)
            hi = max(r[2] for r in smeared)
            if ptop is not None:
                hi = min(hi, ptop)
            if iw is not None and nawf > cfg.dense_n_max:
                probe = self._probe(ptop, interior=(lo, hi)) if hi > lo else None
            else:
                probe = self._probe(ptop)
            if probe is not None:
                # the steepest band of every probed state, as the mesh pass
                # measures the widths of every state it keeps
                vmax = max((float(vn.max()) for _, vn in probe['points'] if len(vn)), default=0.0)
                afac, dk = self._afac_dk()
                dmax = afac * dk * vmax

        ranges = []
        for method, emin, emax, kind, tmax in raw:
            if kind == _window.THERMAL:
                margin = _window.kt_margin(tmax)
            elif kind == _window.SMEARING and dmax is not None:
                margin = _window.smearing_margin(dmax)
            else:
                margin = 0.0
            ranges.append(_window.Range(method, emin, emax, kind, margin, tmax))

        if iw is None:
            ehi, reaches_top = _window.derive_bottom(ranges, ptop)
            probed = None
            if cfg.nev is not None:
                nev, source = min(cfg.nev, nawf), 'given'
            elif reaches_top or ehi is None:
                nev, source = bnd, 'projectable'
            else:
                if probe is None:
                    probe = self._probe(ptop)
                if probe is None:
                    # past dense_n_max: solve every real state
                    nev, source, ehi = bnd, 'projectable', ptop
                else:
                    probed = max(
                        (int(np.searchsorted(E, ehi, side='right')) for E, _ in probe['points']),
                        default=0,
                    )
                    nev, source = min(bnd, nawf, _window.padded_count(probed)), 'probe'
            window = _window.PassWindow(
                'bottom', None, ehi, nev, nev >= bnd, tuple(ranges), source, probed
            )
        else:
            if iw.auto:
                elo, ehi = _window.derive_interior(ranges, ptop)
            else:
                elo, ehi = iw.elo, iw.ehi
                if ptop is not None and ehi > ptop:
                    if elo >= ptop:
                        raise ValueError(
                            'sparse interior_window: [%.3f, %.3f] eV lies above the '
                            'well-projected bands, which end at %.3f eV (the PAO shift). Move '
                            'the window down, or raise the shift with more bands or a lower '
                            'pthr in projectability().' % (elo, ehi, ptop)
                        )
                    self._warn_once(
                        'interior ehi',
                        'sparse interior_window: ehi lowered from %.3f to %.3f eV, the top of '
                        'the well-projected bands' % (ehi, ptop),
                    )
                    ehi = ptop
            window = _window.PassWindow(
                'interior', elo, ehi, None, False, tuple(ranges), 'interior'
            )
            self._interior = (elo, ehi)
        self._log_window(window, tag, props, probe, dmax)
        return window

    def _log_window(self, window, tag, props, probe, dmax):
        """The ``Solve window`` section of ``sparse.log``."""
        log = self.log
        log.section('Solve window (%s)' % tag)
        log.field('properties', ', '.join(p.method for p in props) or 'none')
        if probe is not None and dmax is not None:
            afac, dk = self._afac_dk()
            log.field(
                'probe',
                '%d k-points; largest adaptive width %.3f eV (afac %.2f, k spacing %.4f 1/Bohr)'
                % (probe['nk'], dmax, afac, dk),
            )
        elif probe is not None:
            log.field('probe', '%d k-points' % probe['nk'])
        if dmax is not None:
            log.field(
                'smearing margin (eV)',
                '%.3f  (%g x the largest adaptive width)'
                % (_window.smearing_margin(dmax), _window.SIGMA_FACTOR),
            )
        elif any(r.kind == _window.SMEARING for r in window.ranges):
            log.field(
                'smearing margin (eV)',
                'not applied (%s; widths checked after the pass)'
                % ("'nev' given" if window.mode == 'bottom' else 'explicit window'),
            )
        temperatures = sorted({r.tmax for r in window.ranges if r.kind == _window.THERMAL})
        if temperatures:
            log.field(
                'thermal margin (eV)',
                ', '.join('%.3f at %g K' % (_window.kt_margin(tmax), tmax) for tmax in temperatures)
                + '  (%g kT)' % _window.KT_FACTOR,
            )
        for r in window.ranges:
            lo = 'bottom' if not np.isfinite(r.emin) else '%.3f' % r.emin
            if r.kind == _window.THERMAL:
                extra = '  + thermal margin at %g K' % r.tmax
            elif r.kind == _window.SMEARING:
                extra = '  + smearing margin'
            else:
                extra = ''
            log.field('%s range (eV)' % r.method, '[%s, %.3f]%s' % (lo, r.emax, extra))
        nawf = self.H.nawf
        if window.mode == 'interior':
            log.field(
                'window (eV)',
                '[%.3f, %.3f]  (interior: states outside it are not computed)'
                % (window.elo, window.ehi),
            )
            log.field('bands per k-point', 'varies with k (interior solve)')
            return
        log.field(
            'window (eV)',
            'whole spectrum'
            if window.ehi is None
            else 'bottom of the spectrum to %.3f' % window.ehi,
        )
        bnd = self._bnd_projectable
        if window.nev_source == 'given':
            note = 'given'
        elif window.nev_source == 'projectable':
            note = 'all %d projectable bands' % bnd
        else:
            padded = _window.padded_count(window.probed)
            note = '%d states below the window top at the probe points, padded to %d%s' % (
                window.probed,
                padded,
                '; capped at the %d projectable bands' % bnd if padded > window.nev else '',
            )
        log.field('bands per k-point', '%d of %d  (%s)' % (window.nev, nawf, note))

    def _skip(self, prop, reason):
        """Warn, record, and let the caller move to the next property."""
        self._skipped.append((prop, reason))
        self.log.warning(
            'sparse %s SKIPPED under the interior window: %s; no output was written for it'
            % (prop, reason)
        )

    def _check_smearing_margin(self, prop, emin, emax):
        """Warn if the adaptive widths measured on the mesh reach past the
        solved states: a smeared property plotted that close to the window
        edge misses the tails of states that were never computed.

        The window margin is ``SIGMA_FACTOR`` widths estimated at the probe
        points; the mesh can hold wider states, so the warning fires when the
        solved states end less than ``SIGMA_CHECK`` measured widths past the
        range (where a Gaussian tail has dropped to about 1%).
        """
        window = self._mesh_window
        if window is None or window.at_top:
            return
        attr = self.data_controller.data_attributes
        dmax = float(attr.get('sparse_dmax', 0.0))
        if dmax <= 0.0:
            return
        # no real states above the well-projected top: nothing missing there
        ptop = self._projectable_top()
        if window.mode == 'interior':
            gap = float(emin) - window.elo
            if ptop is None or window.ehi < ptop - 1e-9:
                gap = min(gap, window.ehi - float(emax))
            remedy = 'use a finer k-mesh or give wider interior_window bounds'
        else:
            top = attr.get('sparse_solved_top')
            if top is None or (ptop is not None and top >= ptop):
                return
            gap = top - float(emax)
            remedy = "use a finer k-mesh or give a larger 'nev'"
        reach = _window.SIGMA_CHECK * dmax
        if reach > gap + 1e-9:
            self.log.warning(
                'sparse %s may be contaminated near the window edge: the largest adaptive width '
                'on the mesh is %.3f eV, but the solved states end %.3f eV past the range '
                '[%.3f, %.3f]; %s' % (prop, dmax, gap, emin, emax, remedy)
            )

    def _check_window_covers(self, prop, emax):
        """Raise if ``emax`` lies above the lowest solved top band of the last
        mesh pass: some k-point would be missing states inside the requested
        range.  Collective.

        A no-op without a mesh pass, under an interior window (complete
        inside it by construction), and when the pass solved every
        projectable band, as the dense pipeline does.  A property of the
        same pass is always covered; this catches a later property that
        reuses the pass with a wider range.
        """
        window = self._mesh_window
        if window is None or window.mode == 'interior' or window.at_top:
            return
        arrays, attr = self.data_controller.data_dicts()
        top = attr.get('sparse_solved_top')
        if top is None:
            top = self.comm.allreduce(float(np.min(arrays['E_k'][:, -1, :])), op=MPI.MIN)
        if float(emax) > top + 1e-9:
            raise RuntimeError(
                'sparse %s: requested emax = %.3f eV lies above the lowest solved top band '
                '(%.3f eV) of the %d-band mesh pass. Call the properties inside one '
                '`with pao.sparse.fused():` block, or call the one with the widest range '
                "first, or give a larger 'nev' in sparse_config." % (prop, emax, top, window.nev)
            )

    def _clamp_to_window(self, prop, emin, emax, margin, window):
        """Intersect a requested range with an interior window, ``margin``
        inside each edge.

        At a window top on the well-projected top no margin is kept: there
        are no real states above it whose tails could be missing.

        Returns ``(emin, emax)`` clamped, or ``None`` if nothing usable is
        left, in which case the property is skipped.
        """
        elo, ehi = window.elo, window.ehi
        emin, emax = float(emin), float(emax)
        tol = 1e-9
        ptop = self._projectable_top()
        top_margin = 0.0 if ptop is not None and ehi >= ptop - tol else margin
        lo = emin if emin >= elo + margin - tol else elo + margin
        hi = emax if emax <= ehi - top_margin + tol else ehi - top_margin
        if hi <= lo:
            self._skip(
                prop,
                'the requested range [%.3f, %.3f] does not overlap the usable part of the '
                'window [%.3f, %.3f]' % (emin, emax, elo + margin, ehi - top_margin),
            )
            return None
        if (lo, hi) != (emin, emax):
            self.log.warning(
                'sparse %s range clamped from [%.3f, %.3f] to [%.3f, %.3f] by the interior '
                'window; states outside it are not computed' % (prop, emin, emax, lo, hi)
            )
        return lo, hi

    def _fit_to_window(self, prop, window):
        """Clamp a property's range to an interior window; ``False`` if it
        was skipped."""
        if window is None or window.mode != 'interior':
            return True
        r = prop.energy_range()
        if r is None:
            return True
        emin, emax, kind, tmax = r
        clamped = self._clamp_to_window(prop.method, emin, emax, window.margin(kind, tmax), window)
        if clamped is None:
            return False
        if clamped != (float(emin), float(emax)):
            prop.apply_range(*clamped)
        return True

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
        write_QE_path=False,
    ):
        """Band structure along a path: every well-projected band (or the
        ``'nev'`` of ``sparse_config``, or the states inside the interior
        window).  Output format matches the dense ``bands_{ispin}.dat``."""
        from ..utils.communication import gather_full
        from .bands import do_bands_sparse

        if adhoc_SO:
            raise NotImplementedError(
                'sparse bands: adhoc_SO has no sparse implementation; call to_dense() on the '
                'base cell first.'
            )
        self._require_H('bands')
        arrays, attr = self.data_controller.data_dicts()
        window = self._resolve_window((), 'bands')

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
        # read by the path writer, as the dense method sets it
        attr['write_QE_bands'] = write_QE_path

        def _bands():
            do_bands_sparse(
                self.data_controller,
                self.H,
                window.nev if window.nev is not None else attr['bnd'],
                verbose=attr['verbose'],
                hk_solver=self.config.hk_solver,
                ehi=None if window.at_top or window.mode == 'interior' else window.ehi,
                interior=window.interior,
                ptop=self._projectable_top(),
                **self.config.limits,
            )
            self._path_passes += 1
            self._last_window = window
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
        self.log.field(
            'property mesh', '%d x %d x %d  (%d k-points)' % (tuple(nfft) + (attr['nkpnts'],))
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
            if self.config.interior_window is not None:
                self._skip(
                    'gradient_and_momenta band_curvature',
                    'needs the band curvature (interband sum over every state)',
                )
            else:
                self._mesh_plan.setdefault('products', set()).add('d2Ed2k')

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
        band-diagonal body) runs in call order.  The energy window of that
        pass covers the ranges of every property in the block.  If the
        block raises, nothing queued runs.  Outside a block every property
        call that needs something the last pass did not provide costs a full
        pass of its own, and a later property whose range reaches above the
        bands of an earlier pass raises::

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
        arrays, attr = self.data_controller.data_dicts()
        before = (dict(attr), dict(arrays))
        prop = cls(self, *args, **kwargs)
        if self.config.interior_window is not None:
            reason = prop.interior_reason
            if reason is not None:
                self._skip(cls.method, reason)
                return
        prop.remember_context(*before)
        if self._fusing is not None:
            self._fusing.append(prop)
            return
        self._execute([prop])

    def _mesh_needed(self, products, consumers):
        """Whether a mesh pass must run for these products and consumers."""
        wanted = set(products) | self._mesh_plan.get('products', set())
        return not self._mesh_passes or bool(wanted - self._mesh_products) or bool(consumers)

    def _execute(self, props):
        """Resolve the window, prepare ``props``, run the mesh pass and the
        path passes they need, then their post-pass steps in call order.

        ``prepare`` runs only once the window of the pass is known, since
        under an interior window a range may have to be clamped to it; the
        run data it writes is added to what ``activate`` restores.  A
        property that reuses an earlier mesh pass gets that pass's window.
        """
        arrays, attr = self.data_controller.data_dicts()
        mesh = [p for p in props if not p.path and p.uses_mesh]
        paths = [p for p in props if p.path]
        products = set().union(*(p.products for p in mesh)) if mesh else set()
        consumers = [p for p in mesh if p.streaming]
        if mesh and self._mesh_needed(products, consumers):
            mesh_window = self._resolve_window(
                mesh + paths, 'mesh pass %d' % (self._mesh_passes + 1)
            )
            path_window = mesh_window
        else:
            mesh_window = self._mesh_window
            path_window = self._resolve_window(paths, 'path pass') if paths else None

        ready = []
        for p in props:
            window = path_window if p.path else mesh_window
            p.activate()
            snapshot = (dict(attr), dict(arrays))
            if not self._fit_to_window(p, window):
                continue
            if not p.prepare():
                continue
            if self.config.interior_window is not None and 'd2Ed2k' in p.products:
                # prepare() may drop it (transport skips only its Hall term)
                self._skip(p.method, 'needs the band curvature (interband sum over every state)')
                continue
            p.merge_context(*snapshot)
            ready.append(p)

        mesh = [p for p in ready if not p.path and p.uses_mesh]
        if mesh:
            self._ensure_mesh(
                products=set().union(*(p.products for p in mesh)),
                consumers=[p for p in mesh if p.streaming],
                window=mesh_window,
            )
        for with_dnm in (True, False):
            group = [p for p in ready if p.path and p.with_dnm == with_dnm]
            if group:
                self._run_path(group, with_dnm, path_window)
        for p in ready:
            p.activate()
            # the band count of the pass this property reads
            if p.path and path_window is not None and path_window.nev is not None:
                attr['bnd'] = path_window.nev
            elif self._mesh_bnd is not None:
                attr['bnd'] = self._mesh_bnd
            self._guard(p.method, lambda p=p: p.finalize(self.data_controller))
            r = p.energy_range()
            if p.uses_mesh and not p.path and r is not None and r[2] == _window.SMEARING:
                self._check_smearing_margin(p.method, r[0], r[1])
            if p.label:
                self._time(p.label)
        if self._mesh_bnd is not None:
            attr['bnd'] = self._mesh_bnd

    def _run_path(self, consumers, with_dnm, window=None):
        """One pass over the band path for path properties (never cached:
        nothing is stored per path point)."""
        from .bands import run_path

        attr = self.data_controller.data_attributes
        if window is None:
            window = self._resolve_window(consumers, 'path pass')
        full = any('full_spectrum' in c.needs for c in consumers)
        dense_n_max = self.config.dense_n_max
        if full and self.H.nawf > dense_n_max:
            raise NotImplementedError(
                'sparse %s: the interband sums run over every state, which needs the full '
                'spectrum at each path point (nawf = %d > dense_n_max = %d). A Sternheimer '
                'solve for them is not implemented yet.'
                % (', '.join(sorted({c.method for c in consumers})), self.H.nawf, dense_n_max)
            )
        # run_path and the path finalizers read the band count from attr['bnd']
        attr['bnd'] = window.nev
        nsolve = self.H.nawf if full else int(window.nev)

        def _path():
            run_path(
                self.data_controller,
                self.H,
                consumers,
                nsolve,
                with_dnm=with_dnm,
                hk_solver=self.config.hk_solver,
                verbose=attr['verbose'],
                ptop=self._projectable_top(),
                **self.config.limits,
            )
            self._path_passes += 1
            self._last_window = window

        self._guard('sparse_path', _path)
        self._time('Sparse path pass')

    def _ensure_mesh(self, products=(), consumers=(), window=None):
        """Run the fused mesh pass unless the last one already provides
        everything asked for.

        A pass is needed if none has run, if a requested product (or one
        recorded by ``gradient_and_momenta``) is missing, or if there are
        streaming consumers, which can only see the k-points while a pass
        runs.  A second pass is announced loudly, since it costs as much as
        the first; :meth:`fused` avoids it.  Products of earlier passes stay
        valid and are not recomputed, unless the new window solves a
        different number of bands, which they would no longer match.
        """
        from .mesh import run_mesh

        arrays, attr = self.data_controller.data_dicts()
        wanted = set(products) | self._mesh_plan.get('products', set())
        missing = wanted - self._mesh_products
        if self._mesh_passes and not missing and not consumers:
            return
        if window is None:
            window = self._resolve_window(consumers, 'mesh pass %d' % (self._mesh_passes + 1))
        if self._mesh_passes:
            why = [p.method for p in consumers] + sorted(missing)
            self.log.warning(
                'Sparse mesh pass %d recomputes the whole mesh for %s, which the earlier pass '
                'did not provide; put the property calls in one `with pao.sparse.fused():` '
                'block to compute them in one pass' % (self._mesh_passes + 1, ', '.join(why))
            )
            old = self._mesh_window
            if old is not None and (old.nev, old.interior) != (window.nev, window.interior):
                self._mesh_products = set()
                missing = wanted

        if window.mode == 'bottom':
            attr['bnd'] = window.nev
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
                ehi=None if window.at_top or window.mode == 'interior' else window.ehi,
                interior=window.interior,
                ptop=self._projectable_top(),
                label='Mesh pass %d' % (self._mesh_passes + 1),
                **self.config.limits,
            )

        self._guard('sparse_mesh', _mesh)
        self._mesh_passes += 1
        self._mesh_products |= missing
        self._mesh_window = self._last_window = window
        self._mesh_bnd = attr['bnd']
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
        if self.rank == 0:
            print('\n' + '\n'.join(lines), flush=True)
        self.log.section('Skipped properties')
        self.log.rows(self._skipped)

    def finish_execution(self):
        self._report_skips()
        call_dense(self.host, 'finish_execution')


def _config_fields(cfg):
    """The ``Sparse configuration`` section of ``sparse.log``."""
    iw = cfg.interior_window
    fields = [('hopping_threshold (eV)', _threshold_label(cfg))]
    if cfg.bond_order is not None:
        fields.append(('bond_order', cfg.bond_order))
    if cfg.rcut is not None:
        fields.append(('rcut (Bohr)', '%.3f' % cfg.rcut))
    if iw is not None:
        bands = 'varies with k (interior solve)'
    elif cfg.nev is not None:
        bands = '%d (given)' % cfg.nev
    else:
        bands = 'automatic (from the property ranges)'
    if iw is None:
        interior = 'none'
    elif iw.auto:
        interior = 'automatic bounds (from the property ranges)'
    else:
        interior = '[%.3f, %.3f] eV' % (iw.elo, iw.ehi)
    fields += [
        ('bands per k-point', bands),
        ('interior window', interior),
        ('H(k) solver', cfg.hk_solver),
        ('dense_n_max', cfg.dense_n_max),
        ('dense_ratio', cfg.dense_ratio),
        ('shift_margin (eV)', '%.3f' % cfg.shift_margin),
    ]
    return fields


def _threshold_label(cfg):
    """Header entry for the element threshold a config actually applies."""
    threshold = resolve_threshold(cfg.hopping_threshold, cfg.rcut, cfg.bond_order)
    if threshold == 0.0 and (cfg.rcut is not None or cfg.bond_order is not None):
        return 'none (a real-space cutoff is used instead)'
    return '%.1e' % threshold
