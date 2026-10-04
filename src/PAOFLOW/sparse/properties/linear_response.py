"""``linear_response``: spin Hall, Rashba--Edelstein, conductivity and AHC tensors.

The dense method picks one of three kernels (``response.linear_response_eqn1``,
``..._eqn3``, ``..._eqn245``) from the response type and the ``t_odd``,
``full_chi2``, ``intraband`` and ``interband`` flags.  Each kernel builds one
pair of interband matrices per k-point and tensor component
(``perturb_split`` of a current or spin operator against ``dH/dk_i``),
reduces it to a band vector or an energy profile, and sums over the
Brillouin zone.  Here the same per-k functions run on each
:class:`~PAOFLOW.sparse.kpoint.KPoint`, the block sums accumulate one
k-point at a time, and the dense finishing functions reduce and write.

Every kernel sums over all bands, so the full spectrum is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


def _jobs(response: str, t_odd: bool, full_chi2: bool, intraband: bool, interband: bool):
    """The ``(kind, tensor key)`` pairs the dense dispatch would run.

    ``kind`` is ``'eqn1'``, ``'eqn3'`` or one of the equation (2)/(4)/(5)
    functions ``'chi2'``, ``'surf'``, ``'sea'`` of ``linear_response_eqn245``.
    """
    split = [k for k, flag in (('surf', intraband), ('sea', interband)) if flag]
    if response == 'shc':
        if t_odd:
            return [(k, 's_tensor') for k in split] if split else [('eqn1', 'shc_tensor')]
        return [('chi2', 's_tensor')] if full_chi2 else [('eqn3', 'shc_tensor')]
    if response == 'ree':
        if t_odd:
            return [('chi2', 'ree_tensor')] if full_chi2 else [('eqn3', 'ree_tensor')]
        return [(k, 'ree_tensor') for k in split] if split else [('eqn1', 'ree_tensor')]
    if response == 'cond':
        return [(k, 'ree_tensor') for k in split] if split else [('eqn1', 'ree_tensor')]
    if response == 'ahc':
        return [('chi2', 'ree_tensor')] if full_chi2 else [('eqn3', 'ree_tensor')]
    raise ValueError(f'Unknown response type: {response}')


@register
class LinearResponse(MeshProperty):
    """Linear-response tensors (see the module docstring).

    Parameters
    ----------
    engine : SparseEngine
    response, gamma, twoD, t_odd, full_chi2, intraband, interband, s_tensor,
    a_tensor, eminH, emaxH, esize
        As ``PAOFLOW.linear_response``.
    """

    method = 'linear_response'
    label = 'Linear response completed'
    needs = frozenset({'full_spectrum'})

    def __init__(
        self,
        engine: SparseEngine,
        response: str = 'shc',
        gamma: float = 0.01,
        twoD: bool = False,
        t_odd: bool = False,
        full_chi2: bool = False,
        intraband: bool = False,
        interband: bool = False,
        s_tensor: Any = None,
        a_tensor: Any = None,
        eminH: float = -1.0,
        emaxH: float = 1.0,
        esize: int = 200,
    ) -> None:
        super().__init__(engine)
        arry, attr = self.data_controller.data_dicts()
        # the attributes and tensors the dense method sets, quirks included
        attr['response'] = response
        attr['gamma'] = gamma
        attr['twoD'] = twoD
        attr['eminH'] = eminH
        attr['emaxH'] = np.amin(np.array([attr['shift'], emaxH]))
        attr['esize'] = esize
        attr['intraband'] = intraband
        attr['interband'] = interband
        attr['t_odd'] = t_odd
        attr['full_chi2'] = full_chi2
        arry['ree_tensor'] = a_tensor if s_tensor is not None else arry['a_tensor']
        arry['shc_tensor'] = s_tensor if s_tensor is not None else arry['s_tensor']
        self.response = response
        self.gamma = gamma
        self.jobs = []
        for kind, key in _jobs(response, t_odd, full_chi2, intraband, interband):
            for tensor in arry[key]:
                self.jobs.append((kind, tuple(int(x) for x in tensor)))

    def prepare(self) -> bool:
        from ...utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI

        arry, attr = self.data_controller.data_dicts()
        if self.response in ('ree', 'shc'):
            if attr['dftSO'] == False:
                # the dense kernels call comm.Abort() here
                raise RuntimeError(
                    'sparse linear_response: relativistic calculation with SO required'
                )
            self.Sj = self.engine.require_operator(
                'Sj', self.engine.host.spin_operator, self.method
            )
        if attr['twoD']:
            av0 = arry['a_vectors'][0, :]
            av1 = arry['a_vectors'][1, :]
            attr['cgs_conv'] = 1.0 / (np.linalg.norm(np.cross(av0, av1)) * attr['alat'] ** 2)
        else:
            attr['cgs_conv'] = (
                1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])
            )
        self.ene = np.linspace(attr['eminH'], attr['emaxH'], attr['esize'])
        nspin = attr['nspin']
        self._acc = {}
        for kind, tensor in self.jobs:
            if kind in ('eqn1', 'eqn3'):
                self._acc[(kind, tensor)] = np.zeros((self.ene.size, nspin))
            else:
                for ispin in self._spins(kind, nspin):
                    self._acc[(kind, tensor, ispin)] = np.zeros(self.ene.size)
        return True

    def _spins(self, kind: str, nspin: int) -> list[int]:
        from ...response.linear_response_eqn245 import eqn245_spins

        return eqn245_spins(self.response, nspin)

    def _operators(self, kp: KPoint, tensor: tuple[int, ...]):
        """The PAO-basis operator pair of one component, as the dense kernels build it."""
        from ...response.do_Hall import current_operator

        dhk = kp.dhk
        if self.response == 'shc':
            spol, jpol, ipol = tensor[0], tensor[1], tensor[2]
            return current_operator(self.Sj[spol], dhk[jpol]), dhk[ipol]
        if self.response == 'ree':
            spol, ipol = tensor[0], tensor[1]
            return self.Sj[spol], dhk[ipol]
        cpol, ipol = tensor[0], tensor[1]
        return dhk[cpol], dhk[ipol]

    def on_k(self, kp: KPoint) -> None:
        from ...response.linear_response_eqn1 import eqn1_response_k
        from ...response.linear_response_eqn3 import eqn3_berry_k, eqn3_occupied_sum
        from ...response.linear_response_eqn245 import (
            chi2_k,
            occupied_sum,
            sea_k,
            surf_k,
            surface_sum,
        )
        from ...utils.perturb_split import perturb_split

        attr = self.data_controller.data_attributes
        E, ispin, ene = kp.E, kp.ispin, self.ene
        E_block, widths = E[None], kp.delta_all[None]
        pairs = {}
        for kind, tensor in self.jobs:
            if tensor not in pairs:
                A, B = self._operators(kp, tensor)
                pairs[tensor] = perturb_split(A, B, kp.V, kp.degen)
            op1, op2 = pairs[tensor]
            if kind == 'eqn1':
                self._acc[(kind, tensor)][:, ispin] += eqn1_response_k(op1, op2, E, ene, self.gamma)
            elif kind == 'eqn3':
                Om = eqn3_berry_k(op1, op2, E, 0.05)
                self._acc[(kind, tensor)][:, ispin] += eqn3_occupied_sum(
                    E_block, widths, Om[None], ene, attr['smearing']
                )
            elif (kind, tensor, ispin) in self._acc:
                if kind == 'chi2':
                    value = occupied_sum(
                        E_block, widths, chi2_k(op1, op2, E, self.gamma, self.response)[None], ene
                    )
                elif kind == 'sea':
                    value = occupied_sum(
                        E_block, widths, sea_k(op1, op2, E, self.gamma, self.response)[None], ene
                    )
                else:
                    value = surface_sum(E_block, widths, surf_k(op1, op2, self.response)[None], ene)
                self._acc[(kind, tensor, ispin)] += value

    def finalize(self, data_controller: DataController) -> None:
        from ...response.linear_response_eqn1 import eqn1_finish
        from ...response.linear_response_eqn3 import eqn3_finish
        from ...response.linear_response_eqn245 import eqn245_finish

        nspin = data_controller.data_attributes['nspin']
        for kind, tensor in self.jobs:
            if kind == 'eqn1':
                eqn1_finish(data_controller, self._acc[(kind, tensor)], self.ene, tensor)
            elif kind == 'eqn3':
                eqn3_finish(data_controller, self._acc[(kind, tensor)], self.ene, tensor)
            else:
                for ispin in self._spins(kind, nspin):
                    eqn245_finish(
                        data_controller,
                        self._acc[(kind, tensor, ispin)],
                        self.ene,
                        kind,
                        self.response,
                        tensor,
                        ispin,
                    )
