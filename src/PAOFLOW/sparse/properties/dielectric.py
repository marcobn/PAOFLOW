"""``dielectric_tensor``: Kubo--Greenwood sum over the window bands, streamed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


@register
class DielectricTensor(MeshProperty):
    """Frequency-dependent dielectric tensor.

    Dense sums only window bands (``pksp[:bnd, :bnd]``), so no full spectrum: per k the momentum
    matrix (:attr:`~PAOFLOW.sparse.kpoint.KPoint.pksp`) and, if ``adaptive_smearing()`` was called
    (dense tests ``deltakp2``), the widths ``delta2`` feed the dense
    :func:`~PAOFLOW.response.do_epsilon.eps_accumulate`; else the fixed ``delta``.  Skipped under an
    interior window: transitions from the occupied states below ``elo`` are missing.
    """

    method = 'dielectric_tensor'
    label = 'Dielectric Tensor'
    interior = 'needs the occupied states below the window (interband transitions)'

    def __init__(
        self,
        engine: SparseEngine,
        delta: float = 0.1,
        intrasmear: float = 0.05,
        emin: float = 0.0,
        emax: float = 10.0,
        ne: int = 501,
        d_tensor: Any = None,
        degauss: float = 0.1,
        emissivity: bool = False,
        emis_angles: Any = (0.0, 30.0, 60.0),
        emis_ntheta: int = 90,
        emis_temperature: Any = 300.0,
        intraband: bool = True,
    ) -> None:
        from ...response.do_epsilon import dielectric_settings

        super().__init__(engine)
        dielectric_settings(
            self.data_controller,
            delta,
            intrasmear,
            d_tensor,
            degauss,
            emissivity,
            emis_angles,
            emis_ntheta,
            emis_temperature,
            intraband,
        )
        arrays = self.data_controller.data_arrays
        self.components = [(int(row[0]), int(row[1])) for row in arrays['d_tensor']]
        self.ene = np.linspace(emin, emax, ne)
        # do_epsilon shifts a zero frequency on its first call, for every
        # component after it
        if self.ene[0] == 0.0:
            self.ene[0] = 0.00001
        self.emax = emax

    def energy_range(self):
        # photon energies: transitions from the occupied states reach about
        # emax above E_F
        return -np.inf, self.emax, 'none', None

    def prepare(self) -> bool:
        from ...response.do_epsilon import eps_settings, report_dielectric_smearing

        attr = self.data_controller.data_attributes
        report_dielectric_smearing(attr)
        self.adaptive = 'smearing' in self.engine._mesh_plan
        self.settings = eps_settings(attr, self.ene, self.adaptive)
        nspin = attr['nspin']
        self._sums = {
            (c, ispin): [np.zeros(self.ene.size), np.zeros(self.ene.size), 0.0]
            for c in self.components
            for ispin in range(nspin)
        }
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...response.do_epsilon import eps_accumulate, eps_occupations

        bnd = kp.bnd
        Ek = kp.E[None, :bnd]
        fn, fnF = eps_occupations(Ek, self.settings)
        deltakp2 = kp.delta2[None, :bnd, :bnd] if self.adaptive else None
        pksp = kp.pksp
        with np.errstate(over='raise'):
            for ipol, jpol in self.components:
                epsi, epsr, drude = eps_accumulate(
                    Ek,
                    fn,
                    fnF,
                    pksp[ipol][None, :bnd, :bnd],
                    pksp[jpol][None, :bnd, :bnd],
                    deltakp2,
                    self.ene,
                    self.settings,
                )
                acc = self._sums[((ipol, jpol), kp.ispin)]
                acc[0] += epsi
                acc[1] += epsr
                acc[2] += drude

    def finalize(self, data_controller: DataController) -> None:
        from ...response.do_epsilon import eps_finish, epsilon_from_partials, write_dielectric

        attr = data_controller.data_attributes
        self._warn_if_truncated(data_controller)
        for ipol, jpol in self.components:
            per_spin = []
            for ispin in range(attr['nspin']):
                epsi, epsr, drude = self._sums.pop(((ipol, jpol), ispin))
                epsi, epsr = eps_finish(epsi, epsr, drude, self.ene, self.settings)
                per_spin.append(
                    epsilon_from_partials(data_controller, self.ene, epsi, epsr, ipol, jpol)
                )
            write_dielectric(data_controller, self.ene, ipol, jpol, per_spin)

    def _warn_if_truncated(self, data_controller: DataController) -> None:
        """Warn if photon energies pass the top solved band (dense cuts at ``bnd`` silently)."""
        from mpi4py import MPI

        engine = self.engine
        arrays, attr = data_controller.data_dicts()
        top = attr.get('sparse_solved_top')
        if top is None:
            top = engine.comm.allreduce(float(np.min(arrays['E_k'][:, -1, :])), op=MPI.MIN)
        # no transition reaches a state above the well-projected bands
        ptop = engine._projectable_top()
        if ptop is not None:
            top = min(top, ptop)
        if self.emax > top:
            engine.log.warning(
                'sparse dielectric_tensor: transitions up to emax = %.3f eV reach past the '
                'lowest top band of the %d solved bands (%.3f eV); transitions into the bands '
                "above are missing, as in the dense kernel at bnd; give a larger 'nev' if they "
                'matter' % (self.emax, arrays['E_k'].shape[1], top)
            )
