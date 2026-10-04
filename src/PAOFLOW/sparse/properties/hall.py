"""Anomalous, spin and orbital Hall conductivities from the Berry curvature.

Per k-point each tensor component needs one pair of interband matrices,
``perturb_split(J, dH/dk_j)`` with ``J = dH/dk_i`` (anomalous) or the
symmetrized current :math:`\\tfrac12\\{O, \\partial_i H\\}` of the spin or
orbital operator, projected onto selected sites.  The pair feeds the dense
per-k kernels of ``response.do_Hall`` — :func:`berry_curvature_k` and
:func:`berry_occupation_sum` for the Fermi-energy scan, and
:func:`smear_sigma_block` for the AC conductivity — and only their
reductions survive the k-point: ``(nk_local, esize)`` for the scan (it is
gathered for the Berry-curvature bxsf, as dense) and ``(esize,)`` for the AC
term.  After the pass the dense writers produce the same files.

The interband sums run over every state, so these properties need the full
spectrum (``needs = {'full_spectrum'}``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


class _BerryHall(MeshProperty):
    """Shared body of the three Hall properties; see the module docstring."""

    needs = frozenset({'full_spectrum'})
    kind = ''
    tensor_key = ''
    operator_key = None
    builder = None

    def _setup(
        self,
        emin: float,
        emax: float,
        ne: int,
        delta: float,
        fermi_up: float,
        fermi_dw: float,
        tensor: Any,
        proj: Any,
        twoD: bool,
        do_ac: bool,
    ) -> None:
        """Mirror the attributes the dense method sets before its kernel."""
        arrays, attr = self.data_controller.data_dicts()
        attr['eminH'], attr['emaxH'] = emin, emax
        attr['deltaH'] = delta
        attr['esizeH'] = ne
        if tensor is not None:
            arrays[self.tensor_key] = np.array(tensor)
        # this routine's own window, as the dense method sets it
        attr['fermi_up'] = fermi_up
        attr['fermi_dw'] = fermi_dw
        self.proj = None if proj is None else np.array(proj)
        self.twoD = twoD
        self.do_ac = do_ac
        self.components = [tuple(int(x) for x in row) for row in arrays[self.tensor_key]]

    def prepare(self) -> bool:
        from ...response.do_Hall import ac_frequency_grid, berry_energy_grid

        arrays, attr = self.data_controller.data_dicts()
        if attr['dftSO'] == False:
            # the dense kernel calls comm.Abort() here
            raise RuntimeError('sparse %s: relativistic calculation with SO required' % self.method)
        self.operator = None
        if self.operator_key is not None:
            host = self.engine.host
            builder = getattr(host, self.builder)
            self.operator = self.engine.require_operator(
                self.operator_key, lambda: builder(adhoc_SO=attr['adhoc_SO']), self.method
            )
        from ..operators import projection_operator

        self.P = None if self.proj is None else projection_operator(self.data_controller, self.proj)
        self.ene = berry_energy_grid(attr)
        self.ene_ac = ac_frequency_grid(attr) if self.do_ac else None
        self._om = {c: {} for c in self.components}
        self._sigma = {c: np.zeros(attr['esizeH'], dtype=complex) for c in self.components}
        return True

    def _pairs(self, kp: KPoint, component: tuple[int, ...]):
        """``(J, J_ac)`` for one component; ``J_ac`` is the unprojected current."""
        from ...response.do_Hall import current_operator, project_current

        i = component[0]
        if self.operator is None:
            return kp.dhk[i], kp.dhk[i]
        current = current_operator(self.operator[component[2]], kp.dhk[i])
        projected = current if self.P is None else project_current(self.P, current)
        return projected, current

    def on_k(self, kp: KPoint) -> None:
        from ...response.do_Hall import (
            berry_curvature_k,
            berry_occupation_sum,
            occupations,
            smear_sigma_block,
        )
        from ...utils.perturb_split import perturb_split

        attr = self.data_controller.data_attributes
        E = kp.E[None]
        widths = kp.delta_all[None]
        for component in self._om:
            j = component[1]
            current, current_ac = self._pairs(kp, component)
            jk, pk = perturb_split(current, kp.dhk[j], kp.V, kp.degen)
            Om_n = berry_curvature_k(kp.E, jk, pk, attr['deltaH'])
            self._om[component][(kp.ispin, kp.ik)] = berry_occupation_sum(
                E, widths, Om_n[None], self.ene, attr['smearing']
            )[0]
            if self.do_ac:
                if current_ac is not current:
                    jk, pk = perturb_split(current_ac, kp.dhk[j], kp.V, kp.degen)
                fn = occupations(E, widths, attr)
                self._sigma[component] += smear_sigma_block(
                    E, fn, jk[None], pk[None], self.ene_ac, kp.delta2[None]
                )

    def finalize(self, data_controller: DataController) -> None:
        from ...response.do_Hall import (
            ac_conductivity_reduce,
            berry_reduce,
            hall_conversion,
            hall_file_names,
            write_ac_outputs,
            write_berry_outputs,
        )

        arrays = data_controller.data_arrays
        nk_local = arrays['E_k'].shape[0]
        for component in self.components:
            per_k = self._om.pop(component)
            # the dense kernel reads spin 0 only
            Om_zkaux = np.array([per_k[(0, ik)] for ik in range(nk_local)]).reshape(
                nk_local, self.ene.size
            )
            ene, value, Om_k = berry_reduce(data_controller, Om_zkaux, self.ene)
            cgs_conv = hall_conversion(data_controller, self.twoD)
            names = hall_file_names(self.kind, *component)
            write_berry_outputs(data_controller, names, ene, value, Om_k, cgs_conv)
            if self.do_ac:
                ene, sigxy = ac_conductivity_reduce(
                    data_controller, self._sigma[component], self.ene_ac
                )
                write_ac_outputs(data_controller, names, ene, sigxy, cgs_conv)


@register
class AnomalousHall(_BerryHall):
    """``anomalous_Hall``: Berry curvature of ``dH/dk_i, dH/dk_j``."""

    method = 'anomalous_Hall'
    label = 'Anomalous Hall Conductivity'
    kind = 'anomalous'
    tensor_key = 'a_tensor'

    def __init__(
        self,
        engine: SparseEngine,
        do_ac: bool = False,
        emin: float = -1.0,
        emax: float = 1.0,
        fermi_up: float = 1.0,
        fermi_dw: float = -1.0,
        ne: int = 501,
        delta: float = 0.05,
        a_tensor: Any = None,
    ) -> None:
        super().__init__(engine)
        self._setup(emin, emax, ne, delta, fermi_up, fermi_dw, a_tensor, None, False, do_ac)


@register
class SpinHall(_BerryHall):
    """``spin_Hall``: Berry curvature of the spin current :math:`\\tfrac12\\{S, \\partial_i H\\}`."""

    method = 'spin_Hall'
    label = 'Spin Hall Conductivity'
    kind = 'spin'
    tensor_key = 's_tensor'
    operator_key = 'Sj'
    builder = 'spin_operator'

    def __init__(
        self,
        engine: SparseEngine,
        twoD: bool = False,
        do_ac: bool = False,
        emin: float = -1.0,
        emax: float = 1.0,
        ne: int = 501,
        delta: float = 0.05,
        fermi_up: float = 1.0,
        fermi_dw: float = -1.0,
        s_tensor: Any = None,
        shc_proj: Any = None,
    ) -> None:
        super().__init__(engine)
        self._setup(emin, emax, ne, delta, fermi_up, fermi_dw, s_tensor, shc_proj, twoD, do_ac)
        if shc_proj is not None:
            self.data_controller.data_arrays['shc_proj'] = np.array(shc_proj)


@register
class OrbitalHall(_BerryHall):
    """``orbital_Hall``: Berry curvature of the orbital current :math:`\\tfrac12\\{L, \\partial_i H\\}`."""

    method = 'orbital_Hall'
    label = 'Orbital Hall Conductivity'
    kind = 'orbital'
    tensor_key = 'o_tensor'
    operator_key = 'Lj'
    builder = 'orbital_operator'

    def __init__(
        self,
        engine: SparseEngine,
        twoD: bool = False,
        do_ac: bool = False,
        emin: float = -1.0,
        emax: float = 1.0,
        ne: int = 501,
        delta: float = 0.05,
        fermi_up: float = 1.0,
        fermi_dw: float = -1.0,
        o_tensor: Any = None,
        ohc_proj: Any = None,
    ) -> None:
        super().__init__(engine)
        self._setup(emin, emax, ne, delta, fermi_up, fermi_dw, o_tensor, ohc_proj, twoD, do_ac)
        if ohc_proj is not None:
            self.data_controller.data_arrays['ohc_proj'] = np.array(ohc_proj)
