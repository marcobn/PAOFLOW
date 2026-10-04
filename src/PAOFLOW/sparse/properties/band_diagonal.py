"""Band-diagonal properties: dense bodies on the stored mesh arrays.

``effective_mass``, ``doping`` and ``fermi_surface`` read only ``E_k``,
``deltakp`` and the ``d2Ed2k`` product, so the dense method runs unchanged
after the pass (:class:`~PAOFLOW.sparse.properties.DenseBody`).
``conductivity`` needs one more band-diagonal quantity per k-point, the
product of the two velocity components in a common degenerate-subspace
basis, which it streams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import DenseBody, MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint

_BAND_INDICES = (
    'it selects bands by index, and under an interior window the number of states '
    'in the window varies across the BZ, so a band index has no meaning'
)


@register
class EffectiveMass(DenseBody):
    """``effective_mass``: the dense writer on the ``d2Ed2k`` product.

    Notes
    -----
    The band curvature is a mesh product (``run_mesh(products={'d2Ed2k'})``);
    its interband sum needs every state, so the pass solves the full
    spectrum.  Masses are written for the window bands ``attr['bnd']``.
    """

    method = 'effective_mass'
    products = frozenset({'d2Ed2k'})
    interior = (
        'it needs the band curvature, whose interband sum runs over every state, and an '
        'interior window computes none outside it'
    )


@register
class Doping(DenseBody):
    """``doping``: chemical potential versus doping from the adaptive DOS.

    Notes
    -----
    The dense body integrates the DOS from ``emin`` and adds
    ``core_electrons``, so the window must reach ``emax``.  Under an interior
    window it is skipped: the doping level fixes the total electron count,
    which needs every occupied state.
    """

    method = 'doping'
    interior = (
        'the chemical potential at a doping level is fixed by the total electron count, '
        'which needs every occupied state; an interior window has none below elo'
    )

    def __init__(
        self,
        engine: SparseEngine,
        tmin: float = 300,
        tmax: float = 300,
        nt: int = 1,
        delta: float = 0.01,
        emin: float = -1.0,
        emax: float = 1.0,
        ne: int = 1000,
        doping_conc: float = 0.0,
        core_electrons: float = 0.0,
        fname: str = 'doping_',
    ) -> None:
        super().__init__(
            engine,
            tmin=tmin,
            tmax=tmax,
            nt=nt,
            delta=delta,
            emin=emin,
            emax=emax,
            ne=ne,
            doping_conc=doping_conc,
            core_electrons=core_electrons,
            fname=fname,
        )

    def finalize(self, data_controller: DataController) -> None:
        self.engine._check_window_covers('doping', self.kwargs['emax'])
        super().finalize(data_controller)


@register
class FermiSurface(DenseBody):
    """``fermi_surface``: BXSF files of the window bands crossing the Fermi window."""

    method = 'fermi_surface'
    interior = _BAND_INDICES

    def __init__(self, engine: SparseEngine, fermi_up: float = 1.0, fermi_dw: float = -1.0) -> None:
        super().__init__(engine, fermi_up=fermi_up, fermi_dw=fermi_dw)

    def finalize(self, data_controller: DataController) -> None:
        self.engine._check_window_covers('fermi_surface', self.kwargs['fermi_up'])
        super().finalize(data_controller)


@register
class Conductivity(MeshProperty):
    """``conductivity``: smeared band-diagonal :math:`v_i v_j` per tensor component.

    Parameters
    ----------
    engine : SparseEngine
    delta, emin, emax, ne, cond_tensor
        As ``PAOFLOW.conductivity``.

    Notes
    -----
    Per k-point the window bands' :func:`~PAOFLOW.response.do_conductivity
    .velocity_products_k` are stored, O(nk·nev) per component like
    ``velkp``; after the pass the dense
    :func:`~PAOFLOW.response.do_conductivity.conductivity_from_products`
    smears, reduces and writes them.  The dense kernel sums over all
    ``nawf`` bands; here the sum is over the window, as for the DOS, so
    ``emax`` must lie inside it.  Under an interior window the range is
    clamped like the DOS, and padding states carry zero weight.
    """

    method = 'conductivity'
    label = 'Conductivity'

    def __init__(
        self,
        engine: SparseEngine,
        delta: float = 0.01,
        emin: float = -10.0,
        emax: float = 2.0,
        ne: int = 1000,
        cond_tensor: list | None = None,
    ) -> None:
        super().__init__(engine)
        arrays = self.data_controller.data_arrays
        if cond_tensor is not None:
            arrays['cond_tensor'] = np.array(cond_tensor)
        self.pairs = [tuple(int(x) for x in row[:2]) for row in arrays['cond_tensor']]
        self.delta, self.emin, self.emax, self.ne = delta, emin, emax, ne
        self._products: dict[tuple[int, int], np.ndarray] = {}

    def prepare(self) -> bool:
        engine = self.engine
        if engine._interior is not None:
            clamped = engine._clamp_to_window(
                'conductivity', self.emin, self.emax, margin=engine._smear_margin
            )
            if clamped is None:
                return False
            self.emin, self.emax = clamped
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...response.do_conductivity import velocity_products_k

        V = kp.V[:, : kp.bnd]
        for i, j in self.pairs:
            self._products[(kp.ispin, kp.ik, i, j)] = velocity_products_k(
                kp.dhk[i], kp.dhk[j], V, kp.degen
            )

    def finalize(self, data_controller: DataController) -> None:
        from ...response.do_conductivity import conductivity_from_products

        arrays, attr = data_controller.data_dicts()
        emax = min(float(attr['shift']), float(self.emax))
        self.engine._check_window_covers('conductivity', emax)
        nk_local, nbands, nspin = arrays['E_k'].shape
        for i, j in self.pairs:
            for ispin in range(nspin):
                # padded interior states (and nothing else) stay at zero weight
                products = np.zeros((nk_local, nbands), dtype=complex)
                for ik in range(nk_local):
                    value = self._products.pop((ispin, ik, i, j))
                    products[ik, : len(value)] = value
                conductivity_from_products(
                    data_controller,
                    products,
                    self.emin,
                    self.emax,
                    self.ne,
                    self.delta,
                    i,
                    j,
                    ispin,
                )
