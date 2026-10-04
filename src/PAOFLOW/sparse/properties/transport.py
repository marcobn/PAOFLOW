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

    def prepare(self) -> bool:
        engine = self.engine
        if engine._interior is None:
            return True
        kwargs = self.kwargs
        # the occupation derivative needs states within ~10 kT of every mu
        # on the scan, so the window has to exceed the scan on both sides
        margin = max(engine._kT_margin, 10.0 * 8.617333e-5 * float(kwargs['tmax']))
        clamped = engine._clamp_to_window(
            'transport', kwargs['emin'], kwargs['emax'], margin=margin
        )
        if clamped is None:
            return False
        kwargs['emin'], kwargs['emax'] = clamped
        if kwargs['do_hall']:
            engine._skip(
                'transport Hall term',
                'the Hall and Nernst tensors need the band curvature, whose interband sum '
                'runs over every state, and an interior window computes none outside it. '
                'The rest of the transport tensor is still computed',
            )
            kwargs['do_hall'] = False
            self.products = frozenset()
        return True

    def finalize(self, data_controller: DataController) -> None:
        self.engine._check_window_covers('transport', self.kwargs['emax'])
        super().finalize(data_controller)
