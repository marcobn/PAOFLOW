"""Band-diagonal properties: dense bodies on the stored mesh arrays.

``effective_mass``, ``doping`` and ``fermi_surface`` read only ``E_k``, ``deltakp`` and
``d2Ed2k``, so the dense method runs unchanged after the pass (:class:`DenseBody`).
``conductivity`` also streams ``v_i v_j`` in a common degenerate-subspace basis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import DenseBody, MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint

_BAND_INDICES = 'needs absolute band indices'


@register
class EffectiveMass(DenseBody):
    """``effective_mass``: the dense writer on the ``d2Ed2k`` product, for bands ``attr['bnd']``.

    ``d2Ed2k`` is an interband sum over every state, so the pass solves the full spectrum.
    """

    method = 'effective_mass'
    products = frozenset({'d2Ed2k'})
    interior = 'needs the band curvature (interband sum over every state)'


@register
class Doping(DenseBody):
    """``doping``: chemical potential versus doping from the adaptive DOS.

    Integrates the DOS from ``emin`` plus ``core_electrons``, so the window must reach ``emax``;
    skipped under an interior window, as the total electron count needs every occupied state.
    """

    method = 'doping'
    interior = 'needs the electron count below the window'

    def energy_range(self):
        kw = self.kwargs
        return kw['emin'], kw['emax'], 'thermal', kw['tmax']

    def apply_range(self, emin: float, emax: float) -> None:
        self.kwargs['emin'], self.kwargs['emax'] = emin, emax

    def prepare(self) -> bool:
        kw = self.kwargs
        kw['emin'], kw['emax'] = self.engine._clamp_to_projectable('doping', kw['emin'], kw['emax'])
        return True

    def finalize(self, data_controller: DataController) -> None:
        self.engine._check_window_covers('doping', self.kwargs['emax'])
        super().finalize(data_controller)


@register
class FermiSurface(DenseBody):
    """``fermi_surface``: BXSF files of the window bands crossing the Fermi window."""

    method = 'fermi_surface'
    interior = _BAND_INDICES

    def energy_range(self):
        return self.kwargs['fermi_dw'], self.kwargs['fermi_up'], 'none', None

    def apply_range(self, emin: float, emax: float) -> None:
        self.kwargs['fermi_dw'], self.kwargs['fermi_up'] = emin, emax

    def prepare(self) -> bool:
        kw = self.kwargs
        _, kw['fermi_up'] = self.engine._clamp_to_projectable(
            'fermi_surface', kw['fermi_dw'], kw['fermi_up']
        )
        return True

    def finalize(self, data_controller: DataController) -> None:
        self.engine._check_window_covers('fermi_surface', self.kwargs['fermi_up'])
        super().finalize(data_controller)


@register
class Conductivity(MeshProperty):
    """``conductivity``: smeared band-diagonal :math:`v_i v_j` per tensor component.

    Stores :func:`~PAOFLOW.response.do_conductivity.velocity_products_k` of the window bands per
    k (O(nk·nev) per component, like ``velkp``); the dense
    :func:`~PAOFLOW.response.do_conductivity.conductivity_from_products` smears, reduces, writes.
    Dense sums all ``nawf`` bands; here the window covers ``[emin, emax]`` plus a smearing margin,
    as the DOS (clamped likewise under interior bounds; padding states carry zero weight).
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

    def energy_range(self):
        return self.emin, self.emax, 'smearing', None

    def apply_range(self, emin: float, emax: float) -> None:
        self.emin, self.emax = emin, emax

    def prepare(self) -> bool:
        self.emin, self.emax = self.engine._clamp_to_projectable(
            'conductivity', self.emin, self.emax
        )
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
