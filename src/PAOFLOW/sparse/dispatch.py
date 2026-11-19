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

_PHONON = 'it drives finite-displacement DFT runs and their force constants, not H(R); ' + _TO_DENSE

DENSE_ONLY_REASONS = {
    'coupled_optical_response': (
        'it builds the periodic parts of the Bloch functions from the PAO basis on the FFT grid '
        'of the cell and Wigner-Seitz interpolates the dense HRs for a GMRES density response '
        'over all bands; ' + _TO_DENSE
    ),
    'doped_transport': (
        'it reads the chemical potentials that doping() returns and the sigma and S tensors that '
        'transport() leaves in the run data, and the sparse doping and transport only write '
        'their files; call doping() and transport() themselves in a sparse run.'
    ),
    'kubo_response': (
        'the Kubo-Bastin kernels read the stored dHksp and v_k of the whole grid, (nk, nawf, '
        'nawf) arrays the sparse pass never forms, and no per-k form of them is registered yet; '
        'linear_response() has the tutorial03 limits in a sparse run, or ' + _TO_DENSE
    ),
    'kubo_optical_response': (
        'the Kubo-Greenwood kernel reads the stored dHksp and v_k of the whole grid, (nk, nawf, '
        'nawf) arrays the sparse pass never forms, and no per-k form of it is registered yet; '
        'dielectric_tensor() and the do_ac option of the Hall methods run sparse, or ' + _TO_DENSE
    ),
    'magnetic_field': (
        'it adds mu_B (L + g_s S).B to the dense HRs and, in a scalar basis, first calls '
        'spinor_basis(), which a sparse run cannot follow (see there); ' + _TO_DENSE
    ),
    'orbital_magnetization': (
        'it integrates the orbital_moment and berry_curvature_vector arrays of orbital_moment(), '
        'which is dense-only; ' + _TO_DENSE
    ),
    'orbital_moment': (
        'it sums interband terms of the stored pksp, a (nk, 3, nawf, nawf) array the sparse pass '
        'never forms, and stores the moment of every band at every k; a full-spectrum '
        'MeshProperty over moment_and_curvature is the missing piece; ' + _TO_DENSE
    ),
    'peierls_field': (
        'it multiplies the dense HRs by Peierls phases, and a periodic cell needs a magnetic '
        'supercell (doubling first), where a sparse run has no HRs: the phases would have to be '
        'applied per bond of the doubled bond list, which is not written; ' + _TO_DENSE
    ),
    'self_consistent_spin': (
        'its self-consistency loop diagonalizes H(k) of the whole grid from the dense HRs at '
        'every iteration and updates the on-site exchange field in HRs; ' + _TO_DENSE
    ),
    'special_displacement': 'it builds ZG supercells for DFT runs from the phonon workflow; '
    + _PHONON,
    'spin_susceptibility': (
        'it differentiates the magnetization of self_consistent_spin(), which is dense-only; '
        + _TO_DENSE
    ),
    'spinor_basis': (
        'it doubles nawf by rewriting HRs in block-ordered spinors, and the sparse run keeps the '
        'band count of the scalar basis (through the base-cell round trip, Si gave 18 of its 36 '
        'bands); build the spinor Hamiltonian in a dense run, save_sparse_hamiltonian(), and '
        'load_sparse_hamiltonian() in the sparse one.'
    ),
    'sscha_setup': _PHONON,
    'sscha_population': _PHONON,
    'sscha_minimize': _PHONON,
    'sscha_hessian': _PHONON,
    'sscha_force_constants': _PHONON,
    'sscha_status': _PHONON,
    'stoner_parameters': (
        'it groups the PAO shells from the orbital count of the dense HRs and integrates the '
        'LSDA spin kernel over the charge density; it feeds self_consistent_spin(), which is '
        'dense-only; ' + _TO_DENSE
    ),
    'z2_invariant': (
        'it writes the dense HRs as an _hr.dat file for Z2Pack/tbmodels, which track the Wannier '
        'charge centres themselves; ' + _TO_DENSE
    ),
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
