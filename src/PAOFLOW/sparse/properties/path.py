"""Band-path properties (:func:`~PAOFLOW.sparse.bands.run_path`, the ``bands()`` convention).

Each calls the per-point bodies extracted from the dense kernels.  The dense kernels read the
eigenvectors ``bands()`` left behind, so must run between ``bands()`` and ``pao_eigh``; here each
property solves the path itself and has no ordering constraint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint

_PATH_INTERIOR = 'needs every state (interband sum)'


@register
class BerryCurvature(MeshProperty):
    """``berry_curvature``: spin/orbital Berry curvature along the band path.

    Full-spectrum and ``with_dnm=False`` like the dense kernel (``dH/dk`` from ``Rfft`` alone, sum
    over all ``nawf`` bands); writes path velocities and ``Omegaj_<kind>_<s>_<i><j>`` per kind.
    Both kinds share one pass.
    """

    method = 'berry_curvature'
    path = True
    with_dnm = False
    needs = frozenset({'full_spectrum'})
    interior = _PATH_INTERIOR

    def __init__(
        self,
        engine: SparseEngine,
        spin_Hall: bool = False,
        orbital_Hall: bool = False,
        spol: int | None = None,
        ipol: int | None = None,
        jpol: int | None = None,
    ) -> None:
        super().__init__(engine)
        attr = self.data_controller.data_attributes
        attr['spol'], attr['ipol'], attr['jpol'] = spol, ipol, jpol
        if spol is None or ipol is None or jpol is None:
            raise ValueError("sparse berry_curvature: must specify 'spol', 'ipol', and 'jpol'")
        self.kinds = [
            kind for kind, wanted in (('Spin', spin_Hall), ('Orbital', orbital_Hall)) if wanted
        ]
        self.label = '%s Berry Curvature' % (self.kinds[-1] if self.kinds else '')

    def prepare(self) -> bool:
        attr = self.data_controller.data_attributes
        self.operators = {}
        for kind, key in (('Spin', 'Sj'), ('Orbital', 'Lj')):
            if kind in self.kinds:
                self.operators[kind] = self.engine.require_operator(
                    key, self.method, adhoc_SO=attr['adhoc_SO']
                )
        self._acc = {kind: {} for kind in self.kinds}
        if self.kinds:
            attr['curvature'] = self.kinds[-1]
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...topology.do_berry_curvature import path_berry_k, path_matrix_elements_k

        attr = self.data_controller.data_attributes
        nb = kp.nstates
        spol = attr['spol']
        for kind, operator in self.operators.items():
            pks = np.empty((3, nb, nb), dtype=complex)
            jks = np.empty((3, nb, nb), dtype=complex)
            for l in range(3):
                pks[l], jks[l] = path_matrix_elements_k(kp.V, kp.dhk[l], operator[spol], nb)
            Omj_znk, Omj_zk = path_berry_k(kp.E, pks, jks, attr, nb)
            velocity = np.real(np.einsum('lnn->ln', pks))
            self._acc[kind][(kp.ispin, kp.ik)] = (velocity, Omj_znk, Omj_zk)

    def finalize(self, data_controller: DataController) -> None:
        from ...topology.do_berry_curvature import write_berry_path

        arrays, attr = data_controller.data_dicts()
        nkpi = arrays['kq'].shape[1]
        nspin = self.engine.H.nspin
        for kind in self.kinds:
            acc = self._acc.pop(kind)
            nk_local = max((ik for (_, ik) in acc), default=-1) + 1
            nb = next(iter(acc.values()))[0].shape[1] if acc else self.engine.H.nawf
            velk = np.zeros((nk_local, 3, nb, nspin), dtype=float)
            Omj_zk = np.zeros((nk_local, 1), dtype=float)
            Omj_znk = np.zeros((nk_local, nb), dtype=float)
            for (ispin, ik), (velocity, znk, zk) in acc.items():
                velk[ik, :, :, ispin] = velocity
                if ispin == 0:
                    Omj_znk[ik], Omj_zk[ik] = znk, zk
            write_berry_path(
                data_controller,
                velk,
                Omj_zk,
                Omj_znk,
                kind,
                attr['spol'],
                attr['ipol'],
                attr['jpol'],
                nkpi,
            )


def _time_reversal_operator(engine: SparseEngine):
    """The operator ``theta`` of ``do_topology``'s Z2 test, in sparse form.

    With DFT spin-orbit the dense ``-1j * clebsch_gordan(nawf, sh_l, sh_j, 1)`` is ``Sj[1]`` of
    ``spin_operator``, so the doubled ``Sj[1]`` serves at any cell size.  The ad-hoc spin-orbit
    pattern is rebuilt for the current ``nawf`` as the dense kernel writes it.
    """
    from scipy.sparse import coo_matrix

    arrays, attr = engine.data_controller.data_dicts()
    nawf = engine.H.nawf
    if attr.get('adhoc_SO', False) == True:
        sP = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
        rows, cols, vals = [], [], []
        for i in range(nawf // 2):
            rows += [i, i]
            cols += [i, i + 1]
            vals += [sP[0, 0], sP[0, 1]]
        for i in range(nawf // 2, nawf):
            rows += [i, i]
            cols += [i - 1, i]
            vals += [sP[1, 0], sP[1, 1]]
        Sjy = coo_matrix((vals, (rows, cols)), shape=(nawf, nawf), dtype=complex).tocsr()
        return -1.0j * Sjy
    Sj = engine.require_operator('Sj', 'topology')
    return -1.0j * Sj[1]


@register
class Topology(MeshProperty):
    """``topology``: Z2, path effective mass, Berry and spin Berry curvature.

    Window bands ``attr['bnd']``, ``dH/dk`` with ``Dnm`` but ``d2H/dk^2`` the bare lattice sum
    (no ``Dnm``), as the dense kernel.  Z2 (``nspin = 1`` with ``spin_Hall``) solves the lowest
    ``nelec`` states at the 16 TRIM points and evaluates the same Pfaffians.
    """

    method = 'topology'
    label = 'Band Topology'
    path = True
    with_dnm = True
    interior = 'needs the occupied bands by index'

    def __init__(
        self,
        engine: SparseEngine,
        eff_mass: bool = False,
        Berry: bool = False,
        spin_Hall: bool = False,
        adhoc_SO: bool = False,
        spol: int | None = None,
        ipol: int | None = None,
        jpol: int | None = None,
    ) -> None:
        super().__init__(engine)
        attr = self.data_controller.data_attributes
        # the dense method keeps the first call's flags
        if 'Berry' not in attr:
            attr['Berry'] = Berry
        if 'eff_mass' not in attr:
            attr['eff_mass'] = eff_mass
        if 'spin_Hall' not in attr:
            attr['spin_Hall'] = spin_Hall
        if 'adhoc_SO' not in attr:
            attr['adhoc_SO'] = adhoc_SO
        attr['spol'], attr['ipol'], attr['jpol'] = spol, ipol, jpol
        if spol is None or ipol is None or jpol is None:
            raise ValueError("sparse topology: must specify 'spol', 'ipol', and 'jpol'")

    def prepare(self) -> bool:
        from ...hamiltonian.do_d2Hd2k import IJ_PAIRS

        attr = self.data_controller.data_attributes
        self.Berry, self.eff_mass, self.spin_Hall = (
            attr['Berry'],
            attr['eff_mass'],
            attr['spin_Hall'],
        )
        self.Sj = None
        if self.spin_Hall:
            self.Sj = self.engine.require_operator('Sj', self.method, adhoc_SO=attr['adhoc_SO'])
        i, j = sorted((attr['ipol'], attr['jpol']))
        self.ij = [tuple(p) for p in IJ_PAIRS.tolist()].index((i, j))
        self._acc = {}
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...topology.do_berry_curvature import _project, path_matrix_elements_k
        from ...topology.do_topology import path_effective_mass_k, path_topology_berry_k

        attr = self.data_controller.data_attributes
        bnd = kp.bnd
        V = kp.V[:, :bnd]
        spol, ipol, jpol = attr['spol'], attr['ipol'], attr['jpol']
        pks = np.empty((3, bnd, bnd), dtype=complex)
        jks = np.empty((3, bnd, bnd), dtype=complex) if self.spin_Hall else None
        for l in range(3):
            pks[l], jks_l = path_matrix_elements_k(
                V, kp.dhk[l], self.Sj[spol] if self.spin_Hall else None, bnd
            )
            if self.spin_Hall:
                jks[l] = jks_l
        mass = None
        if self.eff_mass:
            d2hk = self.engine.H.assemble_derivatives(
                kp.kvec, ispin=kp.ispin, sign=+1, cart=True, order=2, with_dnm=False
            )[2]
            tks_ij = _project(V, d2hk[self.ij])[:bnd, :bnd]
            mass = path_effective_mass_k(kp.E, pks, tks_ij, ipol, jpol, bnd)
        om = omj = None
        if kp.ispin == 0 and (self.Berry or self.spin_Hall):
            om, omj = path_topology_berry_k(kp.E, pks, jks, ipol, jpol, bnd, self.Berry)
        self._acc[(kp.ispin, kp.ik)] = (np.real(np.einsum('lnn->ln', pks)), mass, om, omj)

    def finalize(self, data_controller: DataController) -> None:
        import os

        from ...topology.do_topology import trim_points, write_topology_path, write_z2, z2_deltas
        from ..solver import solve_lowest

        arrays, attr = data_controller.data_dicts()
        engine = self.engine
        nspin = engine.H.nspin
        bnd = int(attr['bnd'])

        if nspin == 1 and self.spin_Hall:
            nelec = int(attr['nelec'])
            ktrim = trim_points(arrays['b_vectors'])
            v_ktrim = np.array(
                [
                    solve_lowest(
                        engine.H.assemble_hk(k, sign=+1, cart=True), nelec, hk_solver='dense'
                    )[1]
                    for k in ktrim
                ]
            )
            delta_ik = z2_deltas(v_ktrim, _time_reversal_operator(engine), nelec)
            if engine.rank == 0:
                write_z2(os.path.join(attr['opath'], 'Z2' + '.dim'), delta_ik)

        nk_local = max((ik for (_, ik) in self._acc), default=-1) + 1
        velk = np.zeros((nk_local, 3, bnd, nspin), dtype=float)
        mkm1 = np.zeros((nk_local, bnd, 3, 3, nspin), dtype=complex) if self.eff_mass else None
        Om_zk = np.zeros((nk_local, 1)) if self.Berry else None
        Omj_zk = np.zeros((nk_local, 1)) if self.spin_Hall else None
        for (ispin, ik), (velocity, mass, om, omj) in self._acc.items():
            # a path point with placeholder states below bnd has fewer bands
            nb = velocity.shape[1]
            velk[ik, :, :nb, ispin] = velocity
            if mass is not None:
                mkm1[ik, :nb, attr['ipol'], attr['jpol'], ispin] = mass
            if ispin == 0 and Om_zk is not None:
                Om_zk[ik] = om
            if ispin == 0 and Omj_zk is not None:
                Omj_zk[ik] = omj
        nkpi = arrays['kq'].shape[1]
        write_topology_path(
            data_controller,
            velk,
            mkm1,
            Om_zk,
            Omj_zk,
            attr['spol'],
            attr['ipol'],
            attr['jpol'],
            nkpi,
        )


@register
class SiteProjectedBands(MeshProperty):
    """``site_projected_bands``: band weights on selected atoms along the path.

    Full spectrum because the dense kernel writes all ``nawf`` bands; keeps
    :func:`~PAOFLOW.spectrum.do_site_projected_bands.site_weights_k` per point (O(nkpi·nawf)).
    """

    method = 'site_projected_bands'
    label = 'site_projeted_bands'
    path = True
    needs = frozenset({'full_spectrum'})
    interior = 'it writes every band of the path by index, and an interior window has only some'

    def __init__(self, engine: SparseEngine, site_proj: list = [0]) -> None:
        super().__init__(engine)
        arry = self.data_controller.data_arrays
        if 'site_proj' not in arry:
            arry['site_proj'] = site_proj

    def prepare(self) -> bool:
        from ...spectrum.do_site_projected_bands import site_mask

        arry, attr = self.data_controller.data_dicts()
        if 'naw' not in arry:
            self.data_controller.build_arrays_adhoc_soc()
        self.mask = site_mask(
            arry['naw'], np.asarray(arry['site_proj']), self.engine.H.nawf, attr['do_spin_orbit']
        )
        self._acc = {}
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...spectrum.do_site_projected_bands import site_weights_k

        # the states above the well-projected bands are placeholders at the shift
        top = self.engine._projectable_top()
        E = kp.E if top is None else np.where(kp.E < top, kp.E, np.nan)
        self._acc[(kp.ispin, kp.ik)] = (E, site_weights_k(kp.V, self.mask))

    def finalize(self, data_controller: DataController) -> None:
        from ...spectrum.do_site_projected_bands import write_site_projected_bands

        nspin = self.engine.H.nspin
        nk_local = max((ik for (_, ik) in self._acc), default=-1) + 1
        nb = self.engine.H.nawf
        E_k = np.zeros((nk_local, nb, nspin))
        weights = np.zeros((nk_local, nb, nspin))
        for (ispin, ik), (E, w) in self._acc.items():
            E_k[ik, :, ispin] = E
            weights[ik, :, ispin] = w
        write_site_projected_bands(data_controller, E_k, weights)


@register
class WaveFunctionProjection(MeshProperty):
    """``wave_function_projection``: site weights of selected bands at one path point.

    The dense kernel reads stored eigenvectors at index ``k_proj`` of the last solved k-set; here
    ``k_proj`` indexes the band path (as the examples use it, after ``bands()``) and only that
    point is solved, for every state.  ``dimension`` is unused, as in dense (reads
    ``attr['dimension']``).
    """

    method = 'wave_function_projection'
    streaming = False
    uses_mesh = False
    interior = 'it selects bands by index, which an interior window does not provide'

    def __init__(self, engine: SparseEngine, dimension: int = 3) -> None:
        super().__init__(engine)

    def finalize(self, data_controller: DataController) -> None:
        from ...topology.do_wave_function_site_projection import wave_function_site_projection
        from ..bands import prepare_path
        from ..solver import solve_lowest

        arry, attr = data_controller.data_dicts()
        engine = self.engine
        prepare_path(data_controller)
        k = arry['kq'][:, int(attr['k_proj'])]
        hk = engine.H.assemble_hk(k, sign=+1, cart=True)
        _, V = solve_lowest(hk, engine.H.nawf, hk_solver='dense')
        if engine.rank == 0:
            wave_function_site_projection(data_controller, v_k=V[None, :, :, None], k_index=0)
        engine.comm.Barrier()
        engine._time('wave_function_projection')


@register
class BerryPhase(MeshProperty):
    """``berry_phase``: discretized Berry/Zak phase on a path, track, circle or square.

    Runs ``topology.do_berry_phase.do_berry_phase`` unchanged except the contour solve: instead of
    storing every eigenvector (``berry_v_k``), only the first and previous point's ``(nawf, nocc)``
    states are kept, all :func:`wilson_step`/:func:`wilson_close` need.  Every rank walks the whole
    (short) contour so the phase agrees everywhere, as the dense file writes expect.
    """

    method = 'berry_phase'
    streaming = False
    uses_mesh = False
    label = 'Berry phase'
    interior = 'needs the occupied states (Wilson loop)'

    def __init__(self, engine: SparseEngine, *args, **kwargs) -> None:
        super().__init__(engine)
        self.settings = self.dense_arguments(args, kwargs)

    def _contour_phase(self, data_controller: DataController):
        from ...topology.do_berry_phase import (
            prepare_contour,
            wilson_close,
            wilson_phase,
            wilson_settings,
            wilson_step,
        )
        from ...utils.constants import ANGSTROM_AU
        from ..solver import solve_lowest

        arry, attr = data_controller.data_dicts()
        H = self.engine.H
        attr['alat'] /= ANGSTROM_AU
        prepare_contour(data_controller)
        attr['alat'] *= ANGSTROM_AU
        settings = wilson_settings(data_controller)
        occ_idx = settings['occ_idx']
        nsolve = H.nawf if occ_idx is None else int(np.max(occ_idx)) + 1

        kq = arry['berry_kq']
        prd = np.eye(settings['dim'], dtype=complex)
        first = previous = None
        for ik in range(kq.shape[1]):
            hk = H.assemble_hk(kq[:, ik], ispin=0, sign=+1, cart=True)
            _, V = solve_lowest(
                hk, nsolve, hk_solver=self.engine.config.hk_solver, **self.engine.config.limits
            )
            if previous is None:
                first = V
            else:
                prd = wilson_step(prd, previous, V, settings)
            previous = V
        if settings['closed']:
            prd = wilson_close(prd, previous, first, settings)
        return wilson_phase(prd, settings)

    def finalize(self, data_controller: DataController) -> None:
        from ...topology.do_berry_phase import berry_phase_settings, do_berry_phase

        berry_phase_settings(data_controller, **self.settings)
        do_berry_phase(self.engine.host, contour_phase=self._contour_phase)
