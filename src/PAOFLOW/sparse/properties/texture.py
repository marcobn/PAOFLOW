"""``spin_texture`` / ``orbital_texture``: band-diagonal operator expectation values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


class _Texture(MeshProperty):
    """Shared body of the two texture properties.

    Notes
    -----
    The dense kernel picks the bands crossing ``[fermi_dw, fermi_up]`` from
    the whole mesh before projecting, which a single streaming pass cannot
    do.  So :func:`~PAOFLOW.topology.texture.texture_k` is stored for every
    window band, O(nk·nev) like ``velkp``, and the selection and the dense
    writer (:func:`~PAOFLOW.topology.texture.write_texture`) run after the
    pass.  Inside a degenerate group the values are gauge-dependent, in the
    dense kernel as much as here.
    """

    kind = ''
    key = ''
    result = ''
    builder = ''
    interior = (
        'it selects bands by index, and under an interior window the number of states '
        'in the window varies across the BZ, so a band index has no meaning'
    )

    def __init__(self, engine: SparseEngine, fermi_up: float = 1.0, fermi_dw: float = -1.0) -> None:
        super().__init__(engine)
        attr = self.data_controller.data_attributes
        # this routine's own window, as the dense method sets it
        attr['fermi_up'] = fermi_up
        attr['fermi_dw'] = fermi_dw
        self.fermi_up = fermi_up
        self._values: dict[tuple[int, int], np.ndarray] = {}

    def prepare(self) -> bool:
        attr = self.data_controller.data_attributes
        if attr['nspin'] != 1:
            if self.engine.rank == 0:
                print('Cannot compute %s texture with nspin=2' % self.kind)
            return False
        host = self.engine.host
        self.operator = self.engine.require_operator(
            self.key, getattr(host, self.builder), self.method
        )
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...topology.texture import texture_k

        self._values[kp.ik] = texture_k(kp.V[:, : kp.bnd], self.operator)

    def finalize(self, data_controller: DataController) -> None:
        from ...topology.texture import fermi_window_bands, write_texture
        from ...utils.communication import gather_full

        arrays, attr = data_controller.data_dicts()
        self.engine._check_window_covers(self.method, self.fermi_up)
        nk_local, nbands, _ = arrays['E_k'].shape
        txtaux = np.zeros((nk_local, 3, nbands), dtype=complex)
        for ik in range(nk_local):
            txtaux[ik] = self._values.pop(ik)
        E_k_full = gather_full(arrays['E_k'], attr['npool'])
        ind_plot = fermi_window_bands(data_controller, E_k_full)
        txtaux = np.take(txtaux, ind_plot, axis=2)
        # this rank's values of the selected bands, for rashba_edelstein
        self.selected = (ind_plot, txtaux)
        arrays[self.result] = write_texture(data_controller, self.kind, E_k_full, txtaux, ind_plot)


@register
class SpinTexture(_Texture):
    """``spin_texture``: :math:`\\langle n|S_l|n\\rangle` of the Fermi-window bands."""

    method = 'spin_texture'
    label = 'Spin Texture'
    kind = 'spin'
    key = 'Sj'
    result = 'sktxt'
    builder = 'spin_operator'


@register
class OrbitalTexture(_Texture):
    """``orbital_texture``: :math:`\\langle n|L_l|n\\rangle` of the Fermi-window bands."""

    method = 'orbital_texture'
    label = 'Orbital Texture'
    kind = 'orbital'
    key = 'Lj'
    result = 'oktxt'
    builder = 'orbital_operator'
