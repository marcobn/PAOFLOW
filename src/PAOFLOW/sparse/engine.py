"""Sparse engine behind ``PAOFLOW(..., sparse=True)``.

:class:`SparseEngine` is not a driver: :class:`PAOFLOW.PAOFLOW` stays the only user-facing class
and routes its ``@sparse_override`` methods here (:mod:`PAOFLOW.sparse.dispatch`), with the same
names and arguments as the dense ones.  Properties are
:class:`~PAOFLOW.sparse.properties.MeshProperty` classes found by name in the registry;
sparse-only features (``fused``, the bond list ``H``) are reached as ``pao.sparse.<name>``.  The
window of each pass is derived from its property ranges (:mod:`~PAOFLOW.sparse.window`); the
``'nev'`` and ``'interior_window'`` options of ``sparse_config`` override it.

The DFT input stages run on the dense code, up to the base-cell ``pao_hamiltonian`` (the one
sanctioned dense stage, at the small pre-doubling ``nawf``).  Dense methods without a sparse
counterpart raise ``NotImplementedError``, never fall back silently; use ``pao.to_dense()``, or
restart dense from a ``save_sparse_hamiltonian`` archive.

``pao_eigh``, ``gradient_and_momenta`` and ``adaptive_smearing`` only record parameters: the first
property runs them as one fused streaming mesh pass.  A later property needing something that
pass did not provide costs another pass; ``pao.sparse.fused()`` avoids it.

Placeholder states: ``pao_hamiltonian`` keeps the bands that pass ``pthr`` and lie below the
shift ``eta`` (``attr['shift']``) at their DFT energies and puts the rest of the PAO space at
``eta`` (around it off the DFT grid).  States from ``eta - shift_margin`` up are not bands: they
are dropped at every k-point, windows are capped and property ranges pulled (with a warning)
below that top, and they remain only as intermediate states of the full-spectrum sums, as in the
dense kernels.  See :meth:`SparseEngine._projectable_top`.

Memory contract (see :mod:`PAOFLOW.sparse`): after ``pao_hamiltonian`` no O(nawf^2 * nk) array
exists; per-k dense workspace is one ``(nawf, nev)`` eigenvector block, or ``(nawf, nawf)`` when
a property needs the full spectrum (only while ``nawf <= dense_n_max``).
"""

import functools
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

from . import window as _window
from .bridge import init_restart_session, sparsify
from .config import resolve_threshold
from .dispatch import _TO_DENSE, call_dense
from .log import get_sparse_log


def _available_memory_bytes():
    """``MemAvailable`` from ``/proc/meminfo`` in bytes, or ``None`` where unreadable.  Not
    ``MemFree``: it counts reclaimable page cache, which a large allocation can take over."""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _node_local_ranks(comm):
    """Ranks on this node (each holds a bond-list copy); world size if MPI cannot split by node."""
    try:
        node = comm.Split_type(MPI.COMM_TYPE_SHARED)
        size = node.Get_size()
        node.Free()
        return size
    except (AttributeError, MPI.Exception):
        return comm.Get_size()


class SparseEngine:
    def __init__(self, host, config):
        """``host``: the owning PAOFLOW driver; ``config``: the parsed ``SparseConfig``."""
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

        self._skipped = []  # properties skipped because the window cannot support them
        self.H = None  # SparseHamiltonian, set by pao_hamiltonian
        self._Dnm = None  # base-cell Dnm, kept only for release_to_dense()
        self._mesh_plan = {}  # parameters recorded for the fused mesh pass
        self._interior = None  # (elo, ehi) of the last interior window
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
        """Raise if doubled: ``spin_operator``/``orbital_operator`` read the base-cell orbital map,
        which is not doubled, so they run before ``doubling_Hamiltonian``, which extends their
        results."""
        if self.H is not None and self.H._doubled:
            raise RuntimeError(
                'sparse %s: doubling_Hamiltonian() already ran. Call %s() at the base cell, '
                'before doubling; the doubling then extends its result block-diagonally, as '
                'in the dense pipeline.' % (caller, caller)
            )

    def require_operator(self, key, caller, **kwargs):
        """Sparse operator ``arrays[key]`` (``'Sj'``/``'Lj'``), built with ``kwargs`` if absent;
        raises if it is absent after doubling."""
        arrays = self.data_controller.data_arrays
        if key not in arrays:
            builder = {'Sj': 'spin_operator', 'Lj': 'orbital_operator'}[key]
            if self.H is not None and self.H._doubled:
                raise RuntimeError(
                    'sparse %s needs %r, which was not built before doubling_Hamiltonian(). '
                    'Call %s() at the base cell, before doubling.' % (caller, key, builder)
                )
            getattr(self.host, builder)(**kwargs)
        return arrays[key]

    def _require_base_cell_state(self, name, way_out=''):
        """Raise unless ``H`` is the undoubled base cell on its own grid and nothing was solved."""
        attr = self.data_controller.data_attributes
        grid = (attr.get('nk1'), attr.get('nk2'), attr.get('nk3'))
        if self.H._doubled or grid != tuple(self.H.nk_grid) or self._solved():
            raise RuntimeError(
                'sparse %s acts on the base-cell H(R), so it must come before '
                'doubling_Hamiltonian(), before interpolated_hamiltonian() and before bands() '
                'and the properties.%s' % (name, way_out)
            )

    def _to_bond_list(self, rebuild=True):
        """Drop dense ``HRs``/``Hks`` (sparsified first if ``rebuild``); keep ``Dnm`` aside for
        :meth:`release_to_dense`."""
        arrays = self.data_controller.data_arrays
        if rebuild:
            cfg = self.config
            self.H = sparsify(
                self.data_controller,
                cfg.hopping_threshold,
                rcut=cfg.rcut,
                bond_order=cfg.bond_order,
            )
        arrays.pop('HRs', None)
        arrays.pop('Hks', None)
        self._Dnm = arrays.pop('Dnm', None)

    def _to_dense(self):
        """Scatter the bond list into dense ``HRs``/``Hks``/``Dnm``; operators become ndarrays."""
        from .bridge import densify
        from .operators import OPERATOR_KEYS, to_dense_operator

        densify(self.data_controller, self.H, Dnm=self._Dnm)
        arrays = self.data_controller.data_arrays
        for key in OPERATOR_KEYS:
            if key in arrays:
                arrays[key] = to_dense_operator(arrays[key])

    # ------------------------------------------------------------------
    # Base-cell Hamiltonian (dense input stage, then the bond list)
    # ------------------------------------------------------------------

    def pao_hamiltonian(self, *args, **kwargs):
        """Dense ``pao_hamiltonian`` (same arguments), then convert to the bond list; the dense
        ``HRs``/``Hks`` are deleted."""
        call_dense(self.host, 'pao_hamiltonian', *args, **kwargs)

        def _convert():
            self._to_bond_list()
            self.log.section('Hamiltonian (base cell)')
            self._log_truncation()

        self._guard('sparse_conversion', _convert)
        self._time('Sparse conversion')

    def _log_truncation(self):
        """Log the truncation of ``self.H`` and its cost.

        The eigenvalue bound is the Gershgorin bound of the dropped part: no eigenvalue at any k
        moves more.  A real-space cutoff drops whole atom-pair blocks and keeps the crystal
        symmetry; an element threshold can split symmetry-protected degeneracies by up to the bound
        (:func:`~PAOFLOW.sparse.config.resolve_threshold`).
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
        """Write the base-cell bond list and run metadata to ``fname`` (relative: output directory).

        Only before doubling: only the base cell has a well-defined bond geometry and orbital map.
        The dense truncation arguments are refused (``sparse_config`` already truncated the bond
        list). Also a labelled dataset: :func:`PAOFLOW.sparse.io.bond_table`.
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
        """Restart a ``restart=True`` driver from a ``save_sparse_hamiltonian`` archive.

        Replaces ``read_atomic_proj_QE``/``projections``, ``projectability`` and ``pao_hamiltonian``
        without rebuilding ``HRs``.  The file's truncation applies, not ``sparse_config``'s.  Every
        rank reads the file.
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
        """Apply a dense base-cell ``H(R)`` transformation to the bond list; returns ``body()``.

        Densify, run ``body`` unchanged, re-sparsify with the ``sparse_config`` truncation (applied
        again, logged).  ``Sj``/``Lj`` pass as ndarrays; old-basis ones are dropped if ``nawf``
        changed (ad-hoc SOC).  Results of earlier mesh passes are discarded.

        Parameters
        ----------
        name : str
            PAOFLOW method name, for messages.
        body : callable
            Dense method body, bound to its arguments.
        modifies : bool, optional
            Whether ``body`` changes ``HRs``; if not, the bond list is kept.

        Raises
        ------
        RuntimeError
            After doubling, interpolation or a solve: the base-cell ``HRs`` would be stale.
        """
        from .operators import OPERATOR_KEYS, to_sparse_operator

        self._require_H(name)
        self._require_base_cell_state(name)
        arrays = self.data_controller.data_arrays
        nnz_before = self.H.nnz
        nawf_before = self.H.nawf
        self._to_dense()

        def _run():
            try:
                return body()
            finally:
                self._to_bond_list(rebuild=modifies)
                for key in OPERATOR_KEYS:
                    if key not in arrays:
                        continue
                    if np.shape(arrays[key])[-1] == self.H.nawf:
                        arrays[key] = to_sparse_operator(arrays[key])
                    else:
                        del arrays[key]

        result = _run()
        if modifies:
            self._forget_windows()
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
        """For ``PAOFLOW.to_dense()``: scatter the bond list into dense ``HRs``/``Hks``/``Dnm`` as
        dense ``pao_hamiltonian()`` would leave them (truncated model), and give it up."""
        self._require_H('to_dense')
        # sparse interpolation and the windows only rewrite attributes the
        # dense pipeline would then read against a base-grid HRs
        self._require_base_cell_state(
            'to_dense', ' Hand the base cell over first and run those steps on the dense pipeline.'
        )

        cfg = self.config
        if cfg.nev is not None or cfg.interior_window is not None:
            self.log.warning(
                "to_dense: the '%s' option of sparse_config does not apply to the dense "
                'pipeline; the run continues without it'
                % ('nev' if cfg.nev is not None else 'interior_window')
            )
        nnz = self.H.nnz
        self._to_dense()
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
        """Log the projected doubling memory before allocating; warn, and go on, above 80% of
        ``MemAvailable`` per node-local rank.

        ``nx, ny, nz`` are exponents (``N = 2**(nx+ny+nz)``: 4,4,4 is 64x 2,2,2), easily misjudged
        into an OOM kill minutes into a job.  The bond list is replicated on every rank
        (deterministic, no communication), so more ranks per node make the fit worse, not better.
        """
        from .solver import GUARD, select_hk_solver

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
                # the same numbers as the check: ARPACK solves GUARD pairs beyond nev
                solver_note = (
                    'refused: nev + %d = %d is %.0f%% of n (limit %.0f%% above dense_n_max); give '
                    "'nev' up to %d or use 'interior_window'"
                    % (
                        GUARD,
                        bnd_final + GUARD,
                        100.0 * (bnd_final + GUARD) / proj['nawf'],
                        100.0 * cfg.dense_ratio,
                        int(cfg.dense_ratio * proj['nawf']) - GUARD,
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
        """Double the cell ``nx``/``ny``/``nz`` times (exponents) on the bond list, never dense.
        Hermitizes once at the end, the bond-level equivalent of the dense per-k Hermitizations."""
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

        self._forget_windows()
        self._guard('doubling_Hamiltonian', _double)
        self._time('doubling_Hamiltonian')

    # ------------------------------------------------------------------
    # Energy window of each pass (see PAOFLOW.sparse.window)
    # ------------------------------------------------------------------

    def _forget_windows(self):
        """Drop what window resolution cached for the current H(R); it also runs when a
        property then skips itself, so no pass guards these caches."""
        self._bnd_projectable = None
        self._probe_cache = {}
        self._interior = None

    def _solved(self):
        """Whether a mesh, path or band pass already ran.  The first pass
        fixes ``attr['bnd']`` for the cell it was run on."""
        return self._mesh_passes > 0 or self._path_passes > 0

    def _projectable_top(self):
        """Top of the well-projected range, ``eta - shift_margin`` (eV), or ``None``.

        Placeholders sit at ``eta`` on the DFT grid but spread around it at interpolated k, hence
        the margin.  With ``shift_type`` 2 they sit at zero, inside the spectrum, and without a
        recorded shift (a model, an old restart) there is none; then nothing is dropped.
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
        """Lower ``emax``, with a warning, to stay ``margin`` below :meth:`_projectable_top`.

        ``margin``: states the property needs above ``emax`` (e.g. the transport occupation
        derivative) must be real too.  Returns ``(emin, emax)``; raises ``ValueError`` if nothing of
        the range is left below the top.
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
        """Eigenvalues and band-velocity norms at :func:`~PAOFLOW.sparse.window.probe_points`.

        Collective: points are shared over the ranks and solved by
        :func:`~PAOFLOW.sparse.mesh.probe_k` from the bottom up to ``ptop``, or inside ``interior``;
        a failure on one rank is raised on all so none hangs in the reduction.  Cached until the
        Hamiltonian changes.  Returns ``{'points': [(E, vnorm), ...], 'nk': int}`` over all ranks,
        or ``None`` if a from-the-bottom probe would densify H(k) above ``dense_n_max``.
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
        """The ``PassWindow`` of a pass over ``props``; ``tag`` names the pass in the log.

        Collective; writes nothing (the pass sets ``attr['bnd']``).  Covers every property range
        plus margin (:mod:`PAOFLOW.sparse.window`), capped at :meth:`_projectable_top`.  The
        smearing margin applies only to a window the engine places: with a given ``nev`` or interior
        bounds ranges are clipped and mesh widths checked after the pass.  Band count capped at the
        projectable count: no k-point has more real states.
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
                # the mesh pass bounds every width the same way
                lo, hi = self._mesh_plan.get('min_width'), self._mesh_plan.get('max_width')
                dmax = max(dmax, 0.0 if lo is None else lo)
                dmax = dmax if hi is None else min(dmax, hi)

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
        """Warn if measured mesh widths reach past the solved states (tails missing near the edge).

        The window margin used ``SIGMA_FACTOR`` widths estimated at the probe points, and the mesh
        can hold wider states; this fires when the solved states end less than ``SIGMA_CHECK``
        measured widths past the range (Gaussian tail about 1%).
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
        """Raise if a later property reusing the last mesh pass has ``emax`` above its lowest solved
        top band.  Collective.  No-op under an interior window (complete by construction) or when
        every projectable band was solved."""
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
        """Clip a range to an interior window, ``margin`` inside each edge; ``None`` = skipped.  No
        top margin at the projectable top: no real states above it can lose tails."""
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
        """Bands along a path: well-projected bands, the given ``'nev'``, or the interior window's.
        Output format matches the dense ``bands_{ispin}.dat``."""
        from ..spectrum.do_bands import set_band_path
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

        set_band_path(self.data_controller, ibrav, band_path, high_sym_points, nk, write_QE_path)

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
        """Interpolate onto a finer k-mesh: only the mesh dimensions change.

        Zero-padding H(R) adds only zero hoppings and the bond-list assembly already uses the
        Hermitian Nyquist split of ``utils.zero_pad``, so this is exact and free.  Arguments of 0
        mean twice the current grid; ``free_HRs`` is meaningless here.
        """
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
        """Optional: only records ``bval``; the eigensolve runs in the fused mesh pass."""
        self.data_controller.data_attributes.setdefault('bval', bval)

    def gradient_and_momenta(
        self,
        band_curvature=False,
        nonlocal_velocity=None,
        nonlocal_velocity_inject=None,
        nonlocal_velocity_sign=None,
        position_dipoles=None,
        position_dipole_level='full',
    ):
        """Optional: velocities come from the fused mesh pass; momenta are formed per k if needed.

        The k-indexed ``pksp`` never exists.  ``band_curvature=True`` adds ``d2Ed2k`` to the next
        mesh pass (full spectrum per k; see :func:`~PAOFLOW.sparse.mesh.run_mesh`).
        """
        attr = self.data_controller.data_attributes
        if (
            nonlocal_velocity
            or nonlocal_velocity_inject
            or nonlocal_velocity_sign is not None
            or attr.get('nonlocal_velocity')
        ):
            raise ValueError(
                'The non-local pseudopotential velocity correction was removed in October 2026: '
                'the PAO velocity i[H, r] already contains V_NL and the term double counted it. '
                'Use gradient_and_momenta(position_dipoles=True) instead '
                '(docs/internals/position-dipoles.md).'
            )
        if position_dipoles or (position_dipoles is None and attr.get('position_dipoles')):
            raise NotImplementedError(
                'sparse gradient_and_momenta: position_dipoles adds -i[H(k), d(k)] to the dense '
                'dHksp on the full k-grid, which the sparse engine never forms; ' + _TO_DENSE
            )
        if band_curvature:
            if self.config.interior_window is not None:
                self._skip(
                    'gradient_and_momenta band_curvature',
                    'needs the band curvature (interband sum over every state)',
                )
            else:
                self._mesh_plan.setdefault('products', set()).add('d2Ed2k')

    def adaptive_smearing(
        self,
        smearing: str = 'gauss',
        afac: float | None = None,
        min_width: float | None = None,
        max_width: float | None = None,
    ) -> None:
        """Optional: record smearing type, ``afac`` and the width bounds for the fused mesh pass
        (Yates widths, as ``do_adaptive_smearing``; interband ``deltakp2`` formed per k where
        needed).  Bad bounds raise when the pass runs, as in the dense pipeline."""
        if smearing not in ('gauss', 'm-p'):
            raise ValueError(
                "Smearing type %s not supported.\nSmearing types are 'gauss' and 'm-p'"
                % str(smearing)
            )
        self.data_controller.data_attributes['smearing'] = smearing
        self._mesh_plan['smearing'] = smearing
        self._mesh_plan['afac'] = afac
        self._mesh_plan['min_width'] = min_width
        self._mesh_plan['max_width'] = max_width

    @contextmanager
    def fused(self):
        """Run every property called inside the block in one mesh pass, e.g.
        ``with pao.sparse.fused(): pao.dos(); pao.transport()``.

        Calls are queued; on normal exit one pass runs with the union of their needs and a window
        covering all their ranges, then each post-pass step in call order.  If the block raises,
        nothing runs.  Outside a block, a property needing something the last pass lacks costs a
        full pass, and one whose range reaches above an earlier pass's bands raises.
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
        prop.merge_context(*before)
        if self._fusing is not None:
            self._fusing.append(prop)
            return
        self._execute([prop])

    def _mesh_needed(self, products, consumers):
        """Whether a mesh pass must run for these products and consumers."""
        wanted = set(products) | self._mesh_plan.get('products', set())
        return not self._mesh_passes or bool(wanted - self._mesh_products) or bool(consumers)

    def _execute(self, props):
        """Resolve the window, prepare ``props``, run mesh and path passes, then post-pass steps.
        ``prepare`` waits for the window, which may clamp its range; the run data it writes is added
        to what ``activate`` restores.  A property reusing an earlier mesh pass gets that pass's
        window."""
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
        """Run the fused mesh pass unless the last one provides everything asked for.

        Streaming consumers always need a pass (they see k-points only while it runs).  A repeat
        pass is warned about (it costs as much as the first; :meth:`fused` avoids it); earlier
        products stay valid unless the new window solves a different band count.
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
                min_width=self._mesh_plan.get('min_width'),
                max_width=self._mesh_plan.get('max_width'),
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
