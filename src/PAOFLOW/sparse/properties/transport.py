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

    Parameters
    ----------
    engine : SparseEngine
    tmin, tmax, nt, emin, emax, ne, scattering_channels, scattering_weights,
    tau_dict, do_hall, write_to_file, save_tensors
        As ``PAOFLOW.transport``.

    Notes
    -----
    The dense ``transport`` body runs unchanged after the pass: it takes the
    ``velkp`` branch and reads ``E_k``/``deltakp``.  ``do_hall`` adds the
    band-curvature product ``d2Ed2k`` to the pass, which ``L_loop_hall``
    reads; its interband sum needs every state, so the pass solves the full
    spectrum when it is requested.

    Under an interior window the chemical-potential scan is clamped inside
    the window by the occupation-derivative margin, and the Hall term is
    skipped (the rest of the tensor is still computed): the curvature sums
    over all the states outside the window, which an interior solve never
    computes.
    """

    method = 'transport'

    def __init__(
        self,
        engine: SparseEngine,
        tmin: float = 300.0,
        tmax: float = 300.0,
        nt: int = 1,
        emin: float = -2.0,
        emax: float = 2.0,
        ne: int = 500,
        scattering_channels: list = [],
        scattering_weights: list = [],
        tau_dict: dict = {},
        do_hall: bool = False,
        write_to_file: bool = True,
        save_tensors: bool = False,
    ) -> None:
        super().__init__(
            engine,
            tmin=tmin,
            tmax=tmax,
            nt=nt,
            emin=emin,
            emax=emax,
            ne=ne,
            scattering_channels=scattering_channels,
            scattering_weights=scattering_weights,
            tau_dict=tau_dict,
            do_hall=do_hall,
            write_to_file=write_to_file,
            save_tensors=save_tensors,
        )
        self.products = frozenset({'d2Ed2k'}) if do_hall else frozenset()

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
