"""``kubo_response`` and ``kubo_optical_response``: the Kubo kernels, one k-point at a time.

The dense drivers (:mod:`PAOFLOW.response.kubo_bastin`, :mod:`~PAOFLOW.response.kubo_optical`)
build the band matrices of the response and field operators per k from the stored ``dHksp`` and
``v_k``; here :func:`~PAOFLOW.response.kubo_operators.band_matrices_k` takes them from the live
:class:`~PAOFLOW.sparse.kpoint.KPoint` (full spectrum) and the same kernels and finishing
functions follow.

Band sums run over the bands below the well-projected top (``SparseEngine._projectable_top``), at
most the projectable count, where the dense default takes ``bnd`` bands at every k-point; an
explicit ``bands`` is used as given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


class _Kubo(MeshProperty):
    """What the two Kubo properties share: operators, components and the band count."""

    needs = frozenset({'full_spectrum'})

    def _setup(
        self,
        observable: str,
        components: Any,
        projection: list | None,
        operator: np.ndarray | None,
        bands: int | None,
    ) -> None:
        from ...response.kubo_operators import OBSERVABLES, parse_component
        from ..operators import projection_operator

        if observable not in OBSERVABLES:
            raise ValueError(
                f'unknown observable {observable!r}; choose from {sorted(OBSERVABLES)}'
            )
        self.observable = observable
        # the mesh of the pass, as run_mesh sets it (the unit factor divides by it)
        attr = self.data_controller.data_attributes
        attr['nkpnts'] = attr['nk1'] * attr['nk2'] * attr['nk3']
        if isinstance(components, str):
            components = [components]
        self.indices = [parse_component(observable, c) for c in components]
        self.operator = operator
        self.bands = bands
        self.projector = (
            None
            if projection is None
            else projection_operator(self.data_controller, np.array(projection))
        )
        self.base = None

    def prepare(self) -> bool:
        from ...response.kubo_operators import OBSERVABLES

        base = OBSERVABLES[self.observable].base
        if base is not None:
            adhoc = self.data_controller.data_attributes['adhoc_SO']
            self.base = self.engine.require_operator(base, self.method, adhoc_SO=adhoc)
        return True

    def _nbands(self, kp: KPoint) -> int:
        """Bands of the sums at this k-point (see the module docstring)."""
        if self.bands is not None:
            return max(1, min(int(self.bands), kp.nstates))
        nb = min(int(self.engine._bnd_projectable), kp.nstates)
        ptop = self.engine._projectable_top()
        return nb if ptop is None else min(nb, int(np.searchsorted(kp.E, ptop)))

    def _band_matrices(self, kp: KPoint, indices: tuple[int, ...], nb: int):
        from ...response.kubo_operators import band_matrices_k

        response, field = band_matrices_k(
            self.observable,
            indices,
            kp.dhk,
            kp.V,
            kp.degen,
            kp.E,
            self.base,
            self.projector,
            self.operator,
        )
        return response[:nb, :nb], field[:nb, :nb]


@register
class KuboResponse(_Kubo):
    """``kubo_response``: Kubo-Bastin terms against the chemical potential."""

    method = 'kubo_response'
    label = 'Kubo-Bastin response'

    def __init__(
        self,
        engine: SparseEngine,
        observable: str = 'charge_current',
        components: list | tuple = ('xy',),
        terms: list | tuple = ('total', 'surface', 'sea'),
        eta: float = 0.01,
        emin: float = -1.0,
        emax: float = 1.0,
        ne: int = 201,
        temperature: float = 0.0,
        regulariser: float = 0.0,
        projection: list | None = None,
        operator: np.ndarray | None = None,
        twoD: bool = False,
        write: bool = True,
        bands: int | None = None,
    ) -> None:
        from ...response.kubo_bastin import bastin_setup

        super().__init__(engine)
        self._setup(observable, components, projection, operator, bands)
        attr = self.data_controller.data_attributes
        # the dense method clips emax at the shift; here with a warning
        emin, emax = engine._clamp_to_projectable(self.method, emin, emax)
        self.mu = np.linspace(emin, emax, ne)
        self.terms, self.eta, self.regulariser = terms, eta, regulariser
        self.temperature, self.twoD, self.write = temperature, twoD, write
        self.grid, self.kT, self.exact, self.thermal, self.scale = bastin_setup(
            self.data_controller, observable, self.mu, eta, terms, temperature, twoD, attr['nspin']
        )
        self._local = [
            {
                'surface': np.zeros(self.grid.size),
                'intraband': np.zeros(self.grid.size),
                'sea': np.zeros(self.grid.size),
                'intrinsic': np.zeros(self.mu.size),
            }
            for _ in self.indices
        ]

    def on_k(self, kp: KPoint) -> None:
        from ...response import kubo_kernels as kernels
        from ...response.kubo_bastin import occupation_sum

        nb = self._nbands(kp)
        energies = kp.E[None, :nb]
        smearing = self.data_controller.data_attributes['smearing']
        for local, indices in zip(self._local, self.indices):
            response, field = self._band_matrices(kp, indices, nb)
            pairs = (response * field.T)[None]
            if self.exact:
                surface, intraband = kernels.surface_kernel(energies, pairs, self.grid, self.eta)
                local['surface'] += surface
                local['intraband'] += intraband
                local['sea'] += kernels.sea_kernel(energies, pairs, self.grid, self.eta)
            if 'intrinsic' in self.terms:
                omega = -kernels.berry_curvature(
                    kp.E[:nb], response, field, self.regulariser, 1.0e-12
                )
                local['intrinsic'] += occupation_sum(
                    energies, kp.delta_all[None, :nb], omega[None], self.mu, self.kT, smearing
                )

    def finalize(self, data_controller: DataController) -> None:
        from ...response.kubo_bastin import bastin_finish, write_kubo_response
        from ...response.kubo_operators import component_label

        self.result = {
            component_label(self.observable, indices): bastin_finish(
                local, self.grid, self.mu, self.kT, self.thermal, self.terms, self.scale
            )
            for local, indices in zip(self._local, self.indices)
        }
        if self.write:
            header = (
                f'eta = {self.eta} eV; T = {self.temperature} K; '
                f'regulariser = {self.regulariser} eV'
            )
            write_kubo_response(data_controller, self.observable, self.result, self.twoD, header)


@register
class KuboOpticalResponse(_Kubo):
    """``kubo_optical_response``: Kubo-Greenwood terms against the photon energy.

    ``omega_max`` must be given: the dense default is the band range of the mesh, which a sparse
    run knows only after the pass that needs the frequencies.
    """

    method = 'kubo_optical_response'
    label = 'Kubo-Greenwood optical response'

    def __init__(
        self,
        engine: SparseEngine,
        observable: str = 'charge_current',
        components: list | tuple = ('xx', 'xy'),
        terms: list | tuple = ('total', 'interband', 'intraband'),
        eta: float = 0.05,
        omega_min: float = 0.0,
        omega_max: float | None = None,
        nomega: int = 501,
        mu: float = 0.0,
        temperature: float = 0.0,
        degenerate_threshold: float = 1.0e-4,
        projection: list | None = None,
        operator: np.ndarray | None = None,
        twoD: bool = False,
        write: bool = True,
        bands: int | None = None,
    ) -> None:
        from ...response.kubo_bastin import K_BOLTZMANN_EV
        from ...response.kubo_operators import OBSERVABLES, unit_factor
        from ...response.kubo_optical import OPTICAL_TERMS

        super().__init__(engine)
        self._setup(observable, components, projection, operator, bands)
        attr = self.data_controller.data_attributes
        if omega_max is None:
            raise NotImplementedError(
                'sparse kubo_optical_response: the default omega_max is the band range of the '
                'whole mesh, known only after the pass that needs the frequencies; give '
                'omega_max (eV).'
            )
        # the checks of the dense driver, before the pass
        unknown = set(terms) - set(OPTICAL_TERMS)
        if unknown:
            raise ValueError(f'unknown terms {sorted(unknown)}; choose from {OPTICAL_TERMS}')
        if eta <= 0.0:
            raise ValueError('eta must be positive')
        if OBSERVABLES[observable].base == 'Sj' and not attr['dftSO']:
            raise ValueError(f'{observable} needs a spinor basis (spin-orbit or adhoc_SO)')
        self.omegas = np.linspace(omega_min, omega_max, nomega)
        self.terms, self.eta, self.mu = terms, eta, mu
        self.temperature, self.twoD, self.write = temperature, twoD, write
        self.kT = K_BOLTZMANN_EV * temperature
        self.degenerate_threshold = degenerate_threshold
        spin_multiplicity = 1.0 if (attr['dftSO'] or attr['nspin'] == 2) else 2.0
        self.scale = spin_multiplicity * unit_factor(self.data_controller, observable, twoD)
        self.scale /= float(attr['nkpnts'])
        self._local = [
            [np.zeros(nomega, dtype=complex), np.zeros(nomega, dtype=complex)] for _ in self.indices
        ]

    def on_k(self, kp: KPoint) -> None:
        from ...response import kubo_kernels as kernels
        from ...response.kubo_optical import band_occupations

        nb = self._nbands(kp)
        energies = kp.E[None, :nb]
        occupation, window = band_occupations(
            self.data_controller, energies, kp.delta_all[None, :nb], self.mu, self.kT, self.eta
        )
        for local, indices in zip(self._local, self.indices):
            response, field = self._band_matrices(kp, indices, nb)
            interband, intraband = kernels.optical_kernel(
                energies,
                (response * field.T)[None],
                occupation,
                window,
                self.omegas,
                self.eta,
                self.degenerate_threshold,
            )
            local[0] += interband
            local[1] += intraband

    def finalize(self, data_controller: DataController) -> None:
        from ...response.kubo_operators import component_label
        from ...response.kubo_optical import optical_finish, write_kubo_optical_response

        self.result = {
            component_label(self.observable, indices): optical_finish(
                local[0], local[1], self.omegas, self.terms, self.scale
            )
            for local, indices in zip(self._local, self.indices)
        }
        if self.write:
            header = f'eta = {self.eta} eV; mu = {self.mu} eV; T = {self.temperature} K'
            write_kubo_optical_response(
                data_controller, self.observable, self.result, self.twoD, header
            )
