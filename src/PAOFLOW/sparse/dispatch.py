"""Route :class:`PAOFLOW.PAOFLOW` methods to the active engine.

``PAOFLOW`` is the only driver.  A run built with ``sparse=...`` holds a
:class:`~PAOFLOW.sparse.engine.SparseEngine` in ``self._engine``; a dense
run holds ``None``.  Every public method of the class has exactly one of
three roles, declared where the method is defined:

- ``@sparse_override``: the engine has its own implementation of the
  same name and signature, which replaces the dense body in a sparse run.
- ``@sparse_shared``: the same code runs in both modes (the DFT input
  stages, which build the base cell before any bond list exists, and
  bookkeeping).
- anything unmarked is dense-only: :func:`sparse_aware` wraps it in a
  guard that raises in a sparse run instead of failing later on a
  missing dense array.

In a dense run the wrappers only test ``self._engine`` and call the
original method, so the dense pipeline is unchanged.
"""

import functools
import inspect

_ROLE = '_sparse_role'


def sparse_override(method):
    """Run the engine's method of the same name when the run is sparse."""
    name = method.__name__

    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        if self._engine is None:
            return method(self, *args, **kwargs)
        return getattr(self._engine, name)(*args, **kwargs)

    setattr(routed, _ROLE, 'override')
    return routed


def sparse_shared(method):
    """Mark a method whose body is correct in both modes."""
    setattr(method, _ROLE, 'shared')
    return method


def _dense_only(method):
    name = method.__name__

    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if self._engine is not None:
            routed = sorted(
                n for n, f in vars(type(self)).items() if getattr(f, _ROLE, None) == 'override'
            )
            raise NotImplementedError(
                'PAOFLOW.%s has no sparse implementation, and this run was built with sparse=. '
                'The sparse engine covers: %s.\n'
                'For dense-only features, call to_dense() after pao_hamiltonian() or '
                'load_sparse_hamiltonian() (base cell, before doubling or interpolation), or '
                'restart a dense run (no sparse=) with load_sparse_hamiltonian().'
                % (name, ', '.join(routed))
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
