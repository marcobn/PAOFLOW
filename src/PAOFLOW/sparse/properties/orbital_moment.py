"""``orbital_moment`` and ``orbital_magnetization``: interband sums of the momenta, per k.

:func:`~PAOFLOW.response.orbital_moment.moment_and_curvature` runs on each
:class:`~PAOFLOW.sparse.kpoint.KPoint` over every state (full spectrum, as the dense sum over all
``nawf`` bands); only the band-diagonal moment and curvature of the stored bands are kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


@register
class OrbitalMoment(MeshProperty):
    """``orbital_moment``: arrays ``orbital_moment`` and ``berry_curvature_vector``,
    ``(nk_local, bnd, 3, nspin)`` on the bands of ``E_k`` (zero for the padded states)."""

    method = 'orbital_moment'
    label = 'Orbital moment'
    needs = frozenset({'full_spectrum'})

    def __init__(self, engine: SparseEngine, degeneracy_tol: float = 1.0e-6) -> None:
        super().__init__(engine)
        self.degeneracy_tol = degeneracy_tol
        self._rows: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def on_k(self, kp: KPoint) -> None:
        from ...response.orbital_moment import moment_and_curvature

        # one eigenbasis for the three directions: KPoint.pksp rotates each degenerate group per
        # direction (perturb_split), which breaks the products V^a_nm V^b_mn
        velocities = np.array([kp.V.conj().T @ (kp.dhk[l] @ kp.V) for l in range(3)])
        moment, curvature = moment_and_curvature(kp.E, velocities, self.degeneracy_tol)
        self._rows[(kp.ispin, kp.ik)] = (moment[: kp.bnd], curvature[: kp.bnd])

    def finalize(self, data_controller: DataController) -> None:
        arrays = data_controller.data_arrays
        nk, nb, nspin = arrays['E_k'].shape
        moment = np.zeros((nk, nb, 3, nspin))
        curvature = np.zeros((nk, nb, 3, nspin))
        for (ispin, ik), (m, c) in self._rows.items():
            moment[ik, : len(m), :, ispin] = m
            curvature[ik, : len(c), :, ispin] = c
        self._rows = {}
        arrays['orbital_moment'], arrays['berry_curvature_vector'] = moment, curvature


@register
class OrbitalMagnetization(OrbitalMoment):
    """``orbital_magnetization``: the moments of this pass, then the dense sum over the stored
    bands.  The window reaches ``emax`` plus the thermal margin, so every occupied state counts."""

    method = 'orbital_magnetization'
    label = 'Orbital magnetization'

    def __init__(
        self,
        engine: SparseEngine,
        emin: float = -1.0,
        emax: float = 1.0,
        ne: int = 201,
        temperature: float = 0.0,
        degeneracy_tol: float = 1.0e-6,
        fname: str = 'orbital_magnetization',
    ) -> None:
        super().__init__(engine, degeneracy_tol)
        self.kwargs = dict(
            emin=emin,
            emax=emax,
            ne=ne,
            temperature=temperature,
            degeneracy_tol=degeneracy_tol,
            fname=fname,
        )

    def energy_range(self):
        kw = self.kwargs
        return kw['emin'], kw['emax'], 'thermal', kw['temperature']

    def prepare(self) -> bool:
        kw = self.kwargs
        kw['emin'], kw['emax'] = self.engine._clamp_to_projectable(
            self.method, kw['emin'], kw['emax']
        )
        return True

    def finalize(self, data_controller: DataController) -> None:
        from ..dispatch import call_dense

        super().finalize(data_controller)
        self.engine._check_window_covers(self.method, self.kwargs['emax'])
        # the dense body finds the moments just stored and only sums them
        self.result = call_dense(self.engine.host, self.method, **self.kwargs)
