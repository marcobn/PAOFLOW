"""Sparse implementations of PAOFLOW properties: one protocol for all of them.

Every property the sparse engine computes is a :class:`MeshProperty`
subclass registered under the name of the PAOFLOW method it implements.
The engine resolves an unknown attribute through :data:`REGISTRY`
(``SparseEngine.__getattr__``), so ``pao.dos(...)`` on a sparse run builds
``REGISTRY['dos'](engine, ...)`` with the dense arguments and runs it; a
new property needs no engine edits.

A property declares what it needs from the fused mesh pass
(:func:`~PAOFLOW.sparse.mesh.run_mesh`) and does its work in two hooks:

- ``on_k(kp)`` receives each k-point's :class:`~PAOFLOW.sparse.kpoint.KPoint`
  while the eigenvectors are live, and must reduce what it needs there
  (only when ``streaming`` is true);
- ``finalize(data_controller)`` runs after the pass: MPI reduction, file
  writes, or the call to a dense band-diagonal body, which then reads the
  stored ``E_k``/``velkp``/``deltakp`` (and products such as ``d2Ed2k``)
  exactly as the dense pipeline would.

A property with ``path = True`` sees the band path instead
(:func:`~PAOFLOW.sparse.bands.run_path`, the ``bands()`` convention), which
is solved afresh for every call; there are no stored arrays on a path.

``needs`` names per-k requirements: ``'full_spectrum'`` makes the pass solve
every state (interband sums run over all of them; admitted only while
``nawf <= DENSE_N_MAX``).  ``products`` names extra band-diagonal arrays the
pass must store (``'d2Ed2k'``).  ``interior`` is ``True`` if the property is
meaningful under ``interior_window``, or a string giving the reason it is
not; a ``full_spectrum`` need implies the latter.  The engine skips such
properties under an interior window with that reason.

Inside ``with pao.sparse.fused():`` the engine queues properties instead of
running them, then runs one pass with the union of their needs and calls
every ``finalize`` in call order.

Adding a property
-----------------
1. Split the dense kernel into a per-k function (the body of its k-loop,
   taking one k-point's eigenpairs and operators) and a reduce/write
   function, with the dense kernel calling both.  The split must leave the
   dense outputs byte-identical; diff them before and after.
2. Write a ``MeshProperty`` subclass in ``sparse/properties/<name>.py`` whose
   ``__init__`` takes the engine and then the dense method's signature, and
   decorate it with :func:`register`.  ``on_k`` calls the per-k function on
   the ``KPoint`` members; ``finalize`` calls the reduce/write function.
   Import the module in :func:`_load` below.
3. Mark the PAOFLOW method ``@sparse_override``.
4. Add a parity test against the dense pipeline (the example01 base cell at
   threshold 0 is cheap and runs the real dense code in-test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint

REGISTRY: dict[str, type[MeshProperty]] = {}

_P = TypeVar('_P', bound='type[MeshProperty]')


def register(cls: _P) -> _P:
    """Class decorator: make ``cls`` the sparse implementation of ``cls.method``."""
    if not cls.method:
        raise TypeError(f'{cls.__name__} must set the PAOFLOW method name it implements')
    if cls.method in REGISTRY and REGISTRY[cls.method] is not cls:
        raise TypeError(f'sparse property {cls.method!r} is registered twice')
    REGISTRY[cls.method] = cls
    return cls


class MeshProperty:
    """Base class of every sparse property; see the module docstring.

    Parameters
    ----------
    engine : SparseEngine
        The engine running the property.  Subclasses take the dense
        method's arguments after it.

    Attributes
    ----------
    method : str
        Name of the PAOFLOW method implemented (class attribute).
    label : str
        Module tag for the timing report.
    needs : frozenset of str
        Per-k requirements; ``'full_spectrum'`` is the only one the pass
        acts on.
    products : frozenset of str
        Band-diagonal arrays the pass must store, beyond ``E_k``,
        ``velkp`` and ``deltakp``.
    interior : bool or str
        ``True`` if meaningful under an interior window, else the reason.
    streaming : bool
        Whether ``on_k`` must see the k-points.
    path : bool
        Whether the k-points are the band path (``sparse.bands.run_path``)
        rather than the BZ mesh.
    with_dnm : bool
        Path properties only: whether ``dH/dk`` carries the ``Dnm`` terms,
        following the dense kernel the property replicates.
    uses_mesh : bool
        Whether the property needs the mesh pass at all (``False`` for
        properties that solve their own k-points in ``finalize``).
    """

    method: str = ''
    label: str = ''
    needs: frozenset[str] = frozenset()
    products: frozenset[str] = frozenset()
    interior: bool | str = True
    streaming: bool = True
    path: bool = False
    with_dnm: bool = True
    uses_mesh: bool = True

    def __init__(self, engine: SparseEngine) -> None:
        self.engine = engine
        self.data_controller = engine.data_controller

    def prepare(self) -> bool:
        """Last checks before the pass; ``False`` means the property skipped itself."""
        return True

    _context: tuple[dict, dict] | None = None

    def remember_context(self, attributes: dict, arrays: dict) -> None:
        """Keep the run data this property's set-up wrote (``attributes`` and
        ``arrays`` entries that are new or replaced), for :meth:`activate`."""
        data_arrays, data_attributes = self.data_controller.data_dicts()
        self._context = (
            {
                k: v
                for k, v in data_attributes.items()
                if k not in attributes or attributes[k] is not v
            },
            {k: v for k, v in data_arrays.items() if k not in arrays or arrays[k] is not v},
        )

    def activate(self) -> None:
        """Restore the run data of this property's set-up.

        The dense kernels read their parameters from the run data
        (``attr['eminH']``, ``attr['response']``, ``arrays['s_tensor']``...),
        which the dense methods set right before calling them.  Inside
        ``fused()`` every property is set up before any of them runs, so a
        later call would otherwise overwrite an earlier one's parameters;
        the engine calls this before each ``on_k`` and ``finalize``.
        """
        if self._context is not None:
            data_arrays, data_attributes = self.data_controller.data_dicts()
            data_attributes.update(self._context[0])
            data_arrays.update(self._context[1])

    def on_k(self, kp: KPoint) -> None:
        """Accumulate one k-point's contribution (streaming properties only)."""

    def finalize(self, data_controller: DataController) -> None:
        """Reduce across ranks and write the outputs, after the pass."""

    @property
    def interior_reason(self) -> str | None:
        """Why this property cannot run under an interior window, or ``None``."""
        if isinstance(self.interior, str):
            return self.interior
        if 'full_spectrum' in self.needs:
            return (
                'it sums over interband pairs that include every state, and an interior '
                'window computes none outside [elo, ehi]'
            )
        return None


class DenseBody(MeshProperty):
    """A band-diagonal property: the dense method body, run after the pass.

    The dense kernel reads only arrays the mesh stores (``E_k``,
    ``velkp``, ``deltakp``, plus ``products``), so ``finalize`` simply calls
    the dense method with the arguments it was given.
    """

    streaming = False

    def __init__(self, engine: SparseEngine, *args, **kwargs) -> None:
        import inspect

        super().__init__(engine)
        dense = inspect.unwrap(getattr(type(engine.host), self.method))
        # fail on a bad argument now, not after a whole mesh pass
        inspect.signature(dense).bind(engine.host, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def finalize(self, data_controller: DataController) -> None:
        from ..dispatch import call_dense

        call_dense(self.engine.host, self.method, *self.args, **self.kwargs)


def _load() -> dict[str, type[MeshProperty]]:
    """Import every property module so the registry is complete."""
    from . import (  # noqa: F401
        band_diagonal,
        dielectric,
        dos,
        hall,
        linear_response,
        path,
        rashba_edelstein,
        states,
        texture,
        transport,
    )

    return REGISTRY
