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

    Parameters
    ----------
    engine : SparseEngine
    do_dos, do_pdos, delta, emin, emax, ne
    As ``PAOFLOW.dos``.  ``delta`` is unused: the sparse mesh always
    produces adaptive widths.

    Notes
    -----
    The total DOS is the dense ``do_dos_adaptive`` reading the stored ``E_k``/``deltakp``.  The PDOS needs the eigenvector weights, so it is accumulated per k-point by :class:`~PAOFLOW.sparse.properties.pdos.PdosAccumulator`
    and makes the property streaming; with ``do_pdos=False`` none is needed.
    Under an energy window, ``emax`` above the lowest computed top band raises; under an interior window the range is clamped to the window minus ``smear_margin_eV`` and checked against the measured widths afterwards.
    """

    method = 'dos'
    label = 'DoS'

    def __init__(
        self,
        engine: SparseEngine,
        do_dos: bool = True,
        do_pdos: bool = True,
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

    def prepare(self) -> bool:
        engine = self.engine
        if engine._interior is not None:
            clamped = engine._clamp_to_window(
                'dos', self.emin, self.emax, margin=engine._smear_margin
            )
            if clamped is None:
                return False
            self.emin, self.emax = clamped
        if self.do_pdos:
            from .pdos import PdosAccumulator

            self._pdos = PdosAccumulator(self.data_controller, self.emin, self.emax, self.ne)
        return True

    def on_k(self, kp: KPoint) -> None:
        self._pdos.on_k(kp)

    def finalize(self, data_controller: DataController) -> None:
        from ...spectrum.do_dos import do_dos_adaptive

        self.engine._check_window_covers('dos', self.emax)
        if self.engine._interior is not None:
            self.engine._check_smearing_margin('dos', self.emin, self.emax)
        if self._pdos is not None:
            self._pdos.finalize(data_controller)
        if self.do_dos:
            do_dos_adaptive(data_controller, self.emin, self.emax, self.ne)
