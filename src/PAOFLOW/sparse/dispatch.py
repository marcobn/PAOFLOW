"""Route :class:`PAOFLOW.PAOFLOW` methods to the active engine.

``PAOFLOW`` is the only driver; ``self._engine`` is a
:class:`~PAOFLOW.sparse.engine.SparseEngine` under ``sparse=True``, else ``None``. Each public
method declares one role:

- ``@sparse_override``: the engine's same-name method (or a registered property class, see
  :mod:`PAOFLOW.sparse.properties`) replaces the dense body.
- ``@sparse_shared``: same code in both modes (DFT input stages before any bond list exists,
  bookkeeping).
- ``@sparse_base_cell``: a base-cell ``H(R)`` transform; the bond list is scattered into a dense
  ``HRs``, the dense body runs, and the result is re-truncated with ``sparse_config``. Only
  before doubling and interpolation.
- unmarked: dense-only; :func:`sparse_aware` makes it raise in a sparse run with the reason in
  :data:`DENSE_ONLY_REASONS` rather than fail later on a missing dense array.

In a dense run the wrappers only test ``self._engine``, so the dense pipeline is unchanged.
"""

import functools
import inspect

_ROLE = '_sparse_role'


def sparse_override(method):
    """Run the engine's method of the same name when the run is sparse."""
    name = method.__name__

    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        if getattr(self, '_engine', None) is None:
            return method(self, *args, **kwargs)
        return getattr(self._engine, name)(*args, **kwargs)

    setattr(routed, _ROLE, 'override')
    return routed


def sparse_shared(method):
    """Mark a method whose body is correct in both modes."""
    setattr(method, _ROLE, 'shared')
    return method


def sparse_base_cell(method=None, *, modifies=True):
    """Run a base-cell ``H(R)`` transformation through the dense body.

    Use bare or as ``@sparse_base_cell(modifies=False)`` for methods that only read ``HRs``
    (writers), which keeps the bond list instead of rebuilding it. See
    ``SparseEngine.run_base_cell``.
    """
    if method is None:
        return functools.partial(sparse_base_cell, modifies=modifies)
    name = method.__name__

    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        if getattr(self, '_engine', None) is None:
            return method(self, *args, **kwargs)
        return self._engine.run_base_cell(
            name, lambda: method(self, *args, **kwargs), modifies=modifies
        )

    setattr(routed, _ROLE, 'base_cell')
    return routed


_TO_DENSE = (
    'call to_dense() after pao_hamiltonian() or load_sparse_hamiltonian() (base cell, before '
    'doubling or interpolation), or restart a dense run (sparse=False) with '
    'load_sparse_hamiltonian().'
)

DENSE_ONLY_REASONS = {
    'restart_dump': (
        'it pickles the DataController, which does not hold the bond list or the engine '
        'state; save the base cell with save_sparse_hamiltonian() and restart with '
        'load_sparse_hamiltonian() instead.'
    ),
    'restart_load': (
        'a pickled DataController carries no bond list; restart with load_sparse_hamiltonian().'
    ),
    'memory_check': (
        'it estimates the dense O(nawf^2 nk) arrays, which a sparse run never forms; the '
        'sparse log prints the bond-list size and the doubling pre-flight projection instead.'
    ),
    'minimal': 'it is archival and disabled in the dense pipeline as well.',
    'mirror_chern_number': (
        'it rotates the dense HRs into mirror sectors and hands them to Z2Pack/tbmodels as '
        '_hr.dat files; ' + _TO_DENSE
    ),
    'find_weyl_points': (
        'its scipy minimizer evaluates the dense HRs at arbitrary k and integrates Chern '
        'numbers around each candidate; it has not been ported to the bond list yet; ' + _TO_DENSE
    ),
    'position_dipole_correction': (
        'it adds -i[H(k), d(k)] to dHksp on the whole k-grid, as dense (nk, nawf, nawf) '
        'blocks, which the sparse engine never forms; ' + _TO_DENSE
    ),
    'arpes': (
        'it evaluates photoemission matrix elements of the PAO Bloch states from the dense HRs; '
        'it has not been ported to the bond list yet; ' + _TO_DENSE
    ),
    'nonlinear_optics': (
        'it Wigner-Seitz interpolates the dense HRs on its own k-grid and sums r_nm and its '
        'generalized derivative over all PAO bands; ' + _TO_DENSE
    ),
    'trim_non_projectable_bands': (
        'it slices the stored dense pksp/E_k of a dense mesh; the sparse mesh keeps only the '
        'bands of attr["bnd"] in the first place.'
    ),
    'phonon_setup': 'the phonon workflow drives finite-displacement DFT runs, not H(R); '
    + _TO_DENSE,
    'phonons': 'the phonon workflow drives finite-displacement DFT runs, not H(R); ' + _TO_DENSE,
    'born_charges': 'Born charges come from finite-field DFT runs, not H(R); ' + _TO_DENSE,
    'ir_spectrum': 'it is built on the phonon workflow (DFT force constants); ' + _TO_DENSE,
    'raman_spectrum': 'it is built on the phonon workflow (DFT force constants); ' + _TO_DENSE,
    'vibrational_dielectric': 'it is built on the phonon workflow (DFT force constants); '
    + _TO_DENSE,
    'quasi_harmonic': 'it is built on the phonon workflow (DFT force constants); ' + _TO_DENSE,
    'electron_phonon': (
        'it combines the phonon workflow with the dense deformation potentials of HRs; ' + _TO_DENSE
    ),
}
"""Why each dense-only method has no sparse counterpart, and the way out."""


def _dense_only(method):
    name = method.__name__

    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if getattr(self, '_engine', None) is not None:
            reason = DENSE_ONLY_REASONS.get(name)
            if reason is None:
                reason = 'it has no sparse implementation yet; ' + _TO_DENSE
            raise NotImplementedError(
                'PAOFLOW.%s is dense-only, and this run was built with sparse=True: %s'
                % (name, reason)
            )
        return method(self, *args, **kwargs)

    setattr(guarded, _ROLE, 'dense')
    return guarded


def sparse_aware(cls):
    """Class decorator: guard every unmarked public method as dense-only."""
    for name, attr in list(vars(cls).items()):
        if name.startswith('_') or not inspect.isfunction(attr) or hasattr(attr, _ROLE):
            continue
        setattr(cls, name, _dense_only(attr))
    return cls


def call_dense(host, name, *args, **kwargs):
    """Call the dense body of a routed method on ``host``, bypassing the engine.

    For engine methods that extend the dense stage rather than replace it
    (the base-cell ``pao_hamiltonian``, the Boltzmann stack in ``transport``).
    """
    return getattr(type(host), name).__wrapped__(host, *args, **kwargs)
