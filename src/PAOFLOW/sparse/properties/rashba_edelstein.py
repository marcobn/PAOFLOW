"""``rashba_edelstein``: spin/orbital Rashba--Edelstein response."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint


@register
class RashbaEdelstein(MeshProperty):
    """Rashba--Edelstein tensor, inter-band (default) or intra-band.

    Parameters
    ----------
    engine : SparseEngine
    emin, emax, ne, delta, temps, reg, twoD, lt, st, write_to_file,
    intra_band, spin, orbital, ree_tensor, ree_proj
        As ``PAOFLOW.rashba_edelstein``.

    Notes
    -----
    The default path is band-diagonal: the dense method runs ``spin_texture``
    (``orbital_texture``) over ``[emin, emax]`` and sums the texture times
    the band velocities with adaptive smearing.  Here the texture is
    streamed by the same consumers as those properties (so their files are
    written too, as in the dense run), and
    :func:`~PAOFLOW.response.do_rashba_edelstein.ree_tensor` runs after the
    pass on each rank's own k-points with the stored ``velkp``/``deltakp``.

    ``intra_band=True`` streams, per tensor component, the band-diagonal
    products of :func:`~PAOFLOW.response.do_rashba_edelstein.ree_intra_products_k`
    for the window bands and hands them to
    :func:`~PAOFLOW.response.do_rashba_edelstein.ree_intra_from_products`.
    """

    method = 'rashba_edelstein'
    label = 'Rashba_Edelstein'

    def __init__(
        self,
        engine: SparseEngine,
        emin: float = -2,
        emax: float = 2,
        ne: int = 500,
        delta: float = 0.05,
        temps: float = 0.0,
        reg: float = 1e-30,
        twoD: bool = False,
        lt: float = 1.0,
        st: float = 1.0,
        write_to_file: bool = True,
        intra_band: bool = False,
        spin: bool = True,
        orbital: bool = False,
        ree_tensor: Any = None,
        ree_proj: Any = None,
    ) -> None:
        super().__init__(engine)
        attr = self.data_controller.data_attributes
        # expose the unit-defining factors to the intra-band routine, as the
        # dense method does
        attr['ree_reg'] = reg
        attr['ree_twoD'] = twoD
        attr['ree_lt'] = lt
        attr['ree_st'] = st
        self.emin, self.emax, self.ne = emin, emax, ne
        self.delta, self.temps, self.reg = delta, temps, reg
        self.twoD, self.lt, self.st = twoD, lt, st
        self.write_to_file = write_to_file
        self.intra_band = intra_band
        self.kinds = [kind for kind, wanted in (('spin', spin), ('orbital', orbital)) if wanted]
        self.ree_tensor = ree_tensor
        self.ree_proj = ree_proj
        if not intra_band:
            self.interior = (
                'it selects bands by index (through the texture), and under an interior '
                'window the number of states in the window varies across the BZ'
            )

    def prepare(self) -> bool:
        from .texture import OrbitalTexture, SpinTexture

        engine = self.engine
        arrays, attr = self.data_controller.data_dicts()
        if self.intra_band:
            if engine._interior is not None:
                clamped = engine._clamp_to_window(
                    'rashba_edelstein', self.emin, self.emax, margin=engine._smear_margin
                )
                if clamped is None:
                    return False
                self.emin, self.emax = clamped
            if self.ree_tensor is not None:
                arrays['ree_tensor'] = np.array(self.ree_tensor)
            self.components = [(int(r[0]), int(r[1])) for r in arrays['ree_tensor']]
            host = engine.host
            self.operators = {}
            for kind, key, builder in (
                ('spin', 'Sj', 'spin_operator'),
                ('orbital', 'Lj', 'orbital_operator'),
            ):
                if kind in self.kinds:
                    build = getattr(host, builder)
                    self.operators[kind] = engine.require_operator(
                        key, lambda build=build: build(adhoc_SO=attr['adhoc_SO']), self.method
                    )
            self.P = None
            if self.ree_proj is not None:
                from ..operators import projection_operator

                arrays['ree_proj'] = np.array(self.ree_proj)
                self.P = projection_operator(self.data_controller, arrays['ree_proj'])
            self._products = {}
        else:
            # the dense method calls spin_texture / orbital_texture over [emin, emax]
            self.textures = {}
            for kind, cls in (('spin', SpinTexture), ('orbital', OrbitalTexture)):
                if kind in self.kinds:
                    texture = cls(engine, fermi_up=self.emax, fermi_dw=self.emin)
                    if not texture.prepare():
                        return False
                    self.textures[kind] = texture
        self.ene = np.linspace(self.emin, self.emax, self.ne)
        return True

    def on_k(self, kp: KPoint) -> None:
        if not self.intra_band:
            for texture in self.textures.values():
                texture.on_k(kp)
            return
        from ...response.do_Hall import project_current
        from ...response.do_rashba_edelstein import ree_intra_products_k

        V = kp.V[:, : kp.bnd]
        for kind, operator in self.operators.items():
            for ipol, spol in self.components:
                op = operator[spol] if self.P is None else project_current(self.P, operator[spol])
                self._products[(kind, ipol, spol, kp.ispin, kp.ik)] = ree_intra_products_k(
                    op, kp.dhk[ipol], V, kp.degen
                )

    def finalize(self, data_controller: DataController) -> None:
        if self.intra_band:
            self._finalize_intra(data_controller)
        else:
            self._finalize_default(data_controller)

    def _finalize_default(self, data_controller: DataController) -> None:
        from ...response.do_rashba_edelstein import ree_tensor

        arrays = data_controller.data_arrays
        for kind, texture in self.textures.items():
            texture.finalize(data_controller)
            ind_plot, txtaux = texture.selected
            ree_tensor(
                data_controller,
                self.ene,
                np.real(txtaux),
                np.take(arrays['velkp'][:, :, :, 0], ind_plot, axis=2),
                np.take(arrays['deltakp'], ind_plot, axis=1)[:, :, 0],
                np.take(arrays['E_k'], ind_plot, axis=1)[:, :, 0],
                self.temps,
                self.reg,
                self.twoD,
                self.lt,
                self.st,
                self.write_to_file,
                '' if kind == 'spin' else 'orbital_',
            )

    def _finalize_intra(self, data_controller: DataController) -> None:
        from ...response.do_rashba_edelstein import ree_intra_from_products

        arrays, attr = data_controller.data_dicts()
        nk_local, nbands, nspin = arrays['E_k'].shape
        for kind in self.operators:
            for ipol, spol in self.components:
                for ispin in range(nspin):
                    spin_velocity = np.zeros((nk_local, nbands), dtype=complex)
                    velocity_squared = np.zeros((nk_local, nbands), dtype=complex)
                    for ik in range(nk_local):
                        sv, vv = self._products.pop((kind, ipol, spol, ispin, ik))
                        spin_velocity[ik, : len(sv)] = sv
                        velocity_squared[ik, : len(vv)] = vv
                    ree_intra_from_products(
                        data_controller,
                        kind,
                        self.ene,
                        self.delta,
                        ipol,
                        spol,
                        spin_velocity,
                        velocity_squared,
                        ispin,
                    )
