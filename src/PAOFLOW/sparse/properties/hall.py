"""Anomalous, spin and orbital Hall conductivities from the Berry curvature.

Per k and component: ``perturb_split(J, dH/dk_j)`` with ``J = dH/dk_i`` (anomalous) or the
spin/orbital current :math:`\\tfrac12\\{O, \\partial_i H\\}` (optionally site-projected), to
``response.do_Hall``'s :func:`berry_curvature_k` + :func:`berry_occupation_sum` (Fermi scan, kept
``(nk_local, esize)`` for the bxsf, as dense) and :func:`smear_sigma_block` (AC, kept
``(esize,)``).  Interband sums over every state: ``needs = {'full_spectrum'}``.
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
    proj_key = None

    def __init__(self, engine: SparseEngine, *args: Any, **kwargs: Any) -> None:
        from ...response.do_Hall import hall_settings

        super().__init__(engine)
        kw = self.dense_arguments(args, kwargs)
        hall_settings(
            self.data_controller,
            self.tensor_key,
            kw[self.tensor_key],
            kw['emin'],
            kw['emax'],
            kw['ne'],
            kw['delta'],
            kw['fermi_up'],
            kw['fermi_dw'],
        )
        self.emin, self.emax = kw['emin'], kw['emax']
        proj = kw.get(self.proj_key)
        self.proj = None if proj is None else np.array(proj)
        self.twoD = kw.get('twoD', False)
        self.do_ac = kw['do_ac']
        arrays = self.data_controller.data_arrays
        self.components = [tuple(int(x) for x in row) for row in arrays[self.tensor_key]]

    def energy_range(self):
        return self.emin, self.emax, 'none', None

    def prepare(self) -> bool:
        from ...response.do_Hall import ac_frequency_grid, berry_energy_grid

        arrays, attr = self.data_controller.data_dicts()
        if attr['dftSO'] == False:
            # the dense kernel calls comm.Abort() here
            raise RuntimeError('sparse %s: relativistic calculation with SO required' % self.method)
        self.operator = None
        if self.operator_key is not None:
            self.operator = self.engine.require_operator(
                self.operator_key, self.method, adhoc_SO=attr['adhoc_SO']
            )
        from ..operators import projection_operator

        self.P = None if self.proj is None else projection_operator(self.data_controller, self.proj)
        attr['eminH'], attr['emaxH'] = self.engine._clamp_to_projectable(
            self.method, attr['eminH'], attr['emaxH']
        )
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


@register
class SpinHall(_BerryHall):
    """``spin_Hall``: curvature of the spin current :math:`\\tfrac12\\{S, \\partial_i H\\}`."""

    method = 'spin_Hall'
    label = 'Spin Hall Conductivity'
    kind = 'spin'
    tensor_key = 's_tensor'
    operator_key = 'Sj'
    proj_key = 'shc_proj'


@register
class OrbitalHall(_BerryHall):
    """``orbital_Hall``: curvature of orbital current :math:`\\tfrac12\\{L, \\partial_i H\\}`."""

    method = 'orbital_Hall'
    label = 'Orbital Hall Conductivity'
    kind = 'orbital'
    tensor_key = 'o_tensor'
    operator_key = 'Lj'
    proj_key = 'ohc_proj'
