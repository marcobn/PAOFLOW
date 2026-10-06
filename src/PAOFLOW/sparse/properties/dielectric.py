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

    Parameters
    ----------
    engine : SparseEngine
    delta, intrasmear, emin, emax, ne, d_tensor, degauss, emissivity,
    emis_angles, emis_ntheta, emis_temperature
        As ``PAOFLOW.dielectric_tensor``.

    Notes
    -----
    The dense kernel sums over the window bands only (``pksp[:bnd, :bnd]``),
    so no full spectrum is needed: per k-point the momentum matrix
    (:attr:`~PAOFLOW.sparse.kpoint.KPoint.pksp`) and, when adaptive smearing
    was requested, the interband widths (``delta2``) feed the dense
    :func:`~PAOFLOW.response.do_epsilon.eps_accumulate`.  Only its partial
    sums survive the k-point; after the pass the dense reduction and writer
    produce the same files.

    As in the dense workflow, the adaptive interband broadening is used
    only if ``adaptive_smearing()`` was called (the dense kernel tests for
    ``deltakp2``), the fixed ``delta`` otherwise.  Under an interior window
    the property is skipped: transitions from the occupied states below
    ``elo`` are missing.
    """

    method = 'dielectric_tensor'
    label = 'Dielectric Tensor'
    interior = (
        'interband transitions start from occupied states below the window, which an '
        'interior window never computes'
    )

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
    ) -> None:
        super().__init__(engine)
        arrays, attr = self.data_controller.data_dicts()
        # the attributes the dense method sets before its kernel
        if 'degauss' not in attr:
            attr['degauss'] = degauss
        if 'delta' not in attr:
            attr['delta'] = delta
        attr['intrasmear'] = intrasmear
        attr['emissivity'] = emissivity
        attr['emis_angles'] = np.atleast_1d(np.array(emis_angles, dtype=float))
        attr['emis_ntheta'] = int(emis_ntheta)
        attr['emis_temperature'] = np.atleast_1d(np.array(emis_temperature, dtype=float))
        if d_tensor == 'all':
            pass
        elif d_tensor == 'diag':
            arrays['d_tensor'] = np.array([[0, 0], [1, 1], [2, 2]])
        elif d_tensor == 'offdiag':
            arrays['d_tensor'] = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
        else:
            arrays['d_tensor'] = np.array(d_tensor)
        self.components = [(int(row[0]), int(row[1])) for row in arrays['d_tensor']]
        self.ene = np.linspace(emin, emax, ne)
        # do_epsilon shifts a zero frequency on its first call, for every
        # component after it
        if self.ene[0] == 0.0:
            self.ene[0] = 0.00001
        self.emax = emax

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
        """Warn when photon energies reach past the window top.

        The dense kernel truncates at ``bnd`` bands in the same way, without
        a message; here the window may have been narrowed by
        ``energy_window``, so say so.
        """
        from mpi4py import MPI

        engine = self.engine
        arrays = data_controller.data_arrays
        top = engine.comm.allreduce(float(np.min(arrays['E_k'][:, -1, :])), op=MPI.MIN)
        if self.emax > top:
            message = (
                'WARNING: sparse dielectric_tensor: transitions up to emax=%.3f eV reach past '
                'the lowest top band of the %d-band window (%.3f eV above E_F); the spectrum '
                'misses transitions into the bands above it, as the dense kernel does at bnd. '
                "Widen the 'energy_window' of sparse_config if they matter."
                % (self.emax, arrays['E_k'].shape[1], top)
            )
            if engine.rank == 0:
                print(message, flush=True)
            engine.log.write('\n' + message)
