"""Sparse implementations of PAOFLOW properties: one protocol for all of them.

Each property is a :class:`MeshProperty` registered under the PAOFLOW method it implements;
``SparseEngine.__getattr__`` resolves ``pao.dos(...)`` to ``REGISTRY['dos'](engine, ...)``, so a
new property needs no engine edits.

Protocol, per pass (:func:`~PAOFLOW.sparse.mesh.run_mesh`, or
:func:`~PAOFLOW.sparse.bands.run_path` if ``path = True``; a path is solved afresh per call and
stores no arrays):

- ``energy_range()`` gives the range to solve and its margin (``'smearing'``, ``'thermal'``,
  ``'none'``); the engine builds the pass window from these (:mod:`~PAOFLOW.sparse.window`) and
  returns a range clipped to an interior window via ``apply_range``.  ``prepare()`` follows.
- ``on_k(kp)`` (only if ``streaming``) sees each :class:`~PAOFLOW.sparse.kpoint.KPoint` while
  the eigenvectors are live and must reduce what it needs there.
- ``finalize(data_controller)``: MPI reduction, writes, or a dense band-diagonal body reading the
  stored ``E_k``/``velkp``/``deltakp`` (and ``products`` such as ``d2Ed2k``) as dense would.

``needs = {'full_spectrum'}`` solves every state (interband sums; only while
``nawf <= DENSE_N_MAX``) and implies ``interior`` is a reason string: such properties are skipped
under an interior window.  Inside ``with pao.sparse.fused():`` properties are queued, run in one
pass with the union of their needs, and finalized in call order.

Adding a property
-----------------
1. Split the dense kernel into a per-k function and a reduce/write function, both called by the
   dense kernel; the dense outputs must stay byte-identical (diff them).
2. Subclass ``MeshProperty`` in ``sparse/properties/<name>.py``, ``__init__(engine, <dense
   signature>)``, decorate with :func:`register`, return any energy range from ``energy_range``
   (and take the clipped one in ``apply_range``).  Import it in :func:`_load`.
3. Mark the PAOFLOW method ``@sparse_override``.
4. Add a dense-parity test (example01 base cell at threshold 0 is cheap).
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

    Attributes
    ----------
    method : str
        PAOFLOW method implemented (class attribute).
    label : str
        Module tag for the timing report.
    needs : frozenset of str
        Per-k requirements; the pass acts only on ``'full_spectrum'``.
    products : frozenset of str
        Band-diagonal arrays to store beyond ``E_k``, ``velkp``, ``deltakp``.
    interior : bool or str
        ``True`` if meaningful under an interior window, else the reason.
    streaming : bool
        Whether ``on_k`` must see the k-points.
    path : bool
        k-points are the band path (``sparse.bands.run_path``), not the BZ mesh.
    with_dnm : bool
        Path only: whether ``dH/dk`` carries the ``Dnm`` terms, as the replicated dense kernel.
    uses_mesh : bool
        ``False`` if the property solves its own k-points in ``finalize``.
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

    def dense_arguments(self, args: tuple, kwargs: dict) -> dict:
        """The PAOFLOW method's arguments by name, its defaults applied; a bad argument fails
        here, not after a mesh pass."""
        import inspect

        dense = inspect.unwrap(getattr(type(self.engine.host), self.method))
        bound = inspect.signature(dense).bind(self.engine.host, *args, **kwargs)
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k != 'self'}

    def prepare(self) -> bool:
        """Last checks before the pass, after the window of the pass is
        known; ``False`` means the property skipped itself."""
        return True

    def energy_range(self) -> tuple[float, float, str, float | None] | None:
        """``(emin, emax, kind, tmax)``: range in eV (``emin`` may be ``-inf``), margin kind
        (:mod:`PAOFLOW.sparse.window`), highest temperature (K) for ``'thermal'``; or ``None``."""
        return None

    def apply_range(self, emin: float, emax: float) -> None:
        """Take a range clipped to an interior window (eV)."""

    _context: tuple[dict, dict] | None = None

    def merge_context(self, attributes: dict, arrays: dict) -> None:
        """Add the run data written since the snapshot ``attributes`` /
        ``arrays`` (by the set-up or :meth:`prepare`) to what :meth:`activate`
        restores."""
        data_arrays, data_attributes = self.data_controller.data_dicts()
        new_attributes = {
            k: v
            for k, v in data_attributes.items()
            if k not in attributes or attributes[k] is not v
        }
        new_arrays = {k: v for k, v in data_arrays.items() if k not in arrays or arrays[k] is not v}
        if self._context is None:
            self._context = ({}, {})
        self._context[0].update(new_attributes)
        self._context[1].update(new_arrays)

    def activate(self) -> None:
        """Restore the run data of this property's set-up, before each ``on_k``/``finalize``.

        Dense kernels read parameters from run data (``attr['eminH']``, ``arrays['s_tensor']``...);
        under ``fused()`` all set-ups run first, so a later one would overwrite earlier ones.
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
            return 'needs every state (interband sum)'
        return None


class DenseBody(MeshProperty):
    """A band-diagonal property: ``finalize`` calls the dense method on the stored arrays."""

    streaming = False

    def __init__(self, engine: SparseEngine, *args, **kwargs) -> None:
        super().__init__(engine)
        self.kwargs = self.dense_arguments(args, kwargs)

    def finalize(self, data_controller: DataController) -> None:
        from ..dispatch import call_dense

        call_dense(self.engine.host, self.method, **self.kwargs)


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
