"""``transport``: the dense Boltzmann stack on the stored mesh arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import DenseBody, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine


@register
class Transport(DenseBody):
    """Boltzmann transport, including the Hall and Nernst tensors.

    The dense body runs unchanged (``velkp`` branch, ``E_k``/``deltakp``).  ``do_hall`` adds the
    ``d2Ed2k`` product (read by ``L_loop_hall``), an interband sum over every state, so the pass
    solves the full spectrum.  Under an interior window the mu scan is clamped inside by the
    occupation-derivative margin and only the Hall term is skipped.
    """

    method = 'transport'

    def __init__(self, engine: SparseEngine, *args, **kwargs) -> None:
        super().__init__(engine, *args, **kwargs)
        self.products = frozenset({'d2Ed2k'}) if self.kwargs['do_hall'] else frozenset()

    def energy_range(self):
        kw = self.kwargs
        return kw['emin'], kw['emax'], 'thermal', kw['tmax']

    def apply_range(self, emin: float, emax: float) -> None:
        self.kwargs['emin'], self.kwargs['emax'] = emin, emax

    def prepare(self) -> bool:
        from ..window import kt_margin

        engine = self.engine
        kwargs = self.kwargs
        # the occupation derivative needs states within ~10 kT of every mu
        # on the scan, so the real states have to reach past the scan
        kwargs['emin'], kwargs['emax'] = engine._clamp_to_projectable(
            'transport', kwargs['emin'], kwargs['emax'], margin=kt_margin(kwargs['tmax'])
        )
        if engine.config.interior_window is not None and kwargs['do_hall']:
            engine._skip(
                'transport Hall term',
                'needs the band curvature (interband sum over every state); the rest of the '
                'transport tensor is computed',
            )
            kwargs['do_hall'] = False
            self.products = frozenset()
        return True

    def finalize(self, data_controller: DataController) -> None:
        self.engine._check_window_covers('transport', self.kwargs['emax'])
        super().finalize(data_controller)
