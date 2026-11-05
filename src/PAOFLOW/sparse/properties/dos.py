"""``dos``: total DOS from the stored mesh arrays, PDOS streamed in the pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


@register
class Dos(MeshProperty):
    """Density of states and projected density of states.

    Total DOS: dense ``do_dos_adaptive`` on the stored ``E_k``/``deltakp``.  The PDOS needs the
    eigenvectors, so :class:`~PAOFLOW.sparse.properties.pdos.PdosAccumulator` streams it (not
    streaming if ``do_pdos=False``).  ``delta`` is unused: the sparse mesh always has adaptive
    widths.  A reused pass whose lowest top band is below ``emax`` raises; interior bounds clamp
    the range; the mesh widths are checked against the window edges afterwards.
    """

    method = 'dos'
    label = 'DoS'

    def __init__(
        self,
        engine: SparseEngine,
        do_dos: bool = True,
        do_pdos: bool = True,
        delta: float = 0.01,
        emin: float = -10.0,
        emax: float = 2.0,
        ne: int = 1000,
    ) -> None:
        super().__init__(engine)
        self.do_dos = do_dos
        self.do_pdos = do_pdos
        self.emin, self.emax, self.ne = emin, emax, ne
        self.streaming = bool(do_pdos)
        self._pdos = None

    def energy_range(self):
        return self.emin, self.emax, 'smearing', None

    def apply_range(self, emin: float, emax: float) -> None:
        self.emin, self.emax = emin, emax

    def prepare(self) -> bool:
        self.emin, self.emax = self.engine._clamp_to_projectable('dos', self.emin, self.emax)
        if self.do_pdos:
            from .pdos import PdosAccumulator

            self._pdos = PdosAccumulator(self.data_controller, self.emin, self.emax, self.ne)
        return True

    def on_k(self, kp: KPoint) -> None:
        self._pdos.on_k(kp)

    def finalize(self, data_controller: DataController) -> None:
        from ...spectrum.do_dos import do_dos_adaptive

        self.engine._check_window_covers('dos', self.emax)
        if self._pdos is not None:
            self._pdos.finalize(data_controller)
        if self.do_dos:
            do_dos_adaptive(data_controller, self.emin, self.emax, self.ne)
