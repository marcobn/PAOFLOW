"""Eigenvector-shape properties: ``ipr`` and ``density``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import MeshProperty, register

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..engine import SparseEngine
    from ..kpoint import KPoint

_BAND_INDICES = (
    'it is tabulated by band index, and under an interior window the number of states '
    'in the window varies across the BZ'
)


@register
class Ipr(MeshProperty):
    """``ipr``: inverse participation ratio of the window bands on the mesh.

    Notes
    -----
    Per k-point :func:`~PAOFLOW.response.do_ipr.ipr_k` of the window
    eigenvectors; after the pass the values are gathered and saved in the
    dense layout ``(nspin, nkpts, nbands, 3)`` (k-point, energy, IPR).  The
    k-point field holds the mesh points in crystal coordinates, in the FFT
    order of ``E_k``.
    """

    method = 'ipr'
    label = 'Inverse Participation Ratio (IPR)'
    interior = _BAND_INDICES

    def __init__(self, engine: SparseEngine, fname: str = 'ipr') -> None:
        super().__init__(engine)
        self.fname = fname
        self._values: dict[tuple[int, int], np.ndarray] = {}

    def on_k(self, kp: KPoint) -> None:
        from ...response.do_ipr import ipr_k

        self._values[(kp.ispin, kp.ik)] = ipr_k(kp.V[:, : kp.bnd])

    def finalize(self, data_controller: DataController) -> None:
        from os.path import join

        from ...response.do_ipr import ipr_table
        from ...utils.communication import gather_full
        from ...utils.get_K_grid_fft import get_K_grid_fft_crystal

        arrays, attr = data_controller.data_dicts()
        nk_local, nbands, nspin = arrays['E_k'].shape
        values = np.zeros((nk_local, nbands, nspin), dtype=float)
        for (ispin, ik), value in self._values.items():
            values[ik, :, ispin] = value
        values = gather_full(values, attr['npool'])
        energies = gather_full(np.ascontiguousarray(arrays['E_k']), attr['npool'])
        arrays['ipr'] = None
        if self.engine.rank == 0:
            kpts = get_K_grid_fft_crystal(attr['nk1'], attr['nk2'], attr['nk3'])
            arrays['ipr'] = ipr_table(kpts, energies, values)
            np.save(join(attr['opath'], self.fname + '.npy'), arrays['ipr'])


@register
class Density(MeshProperty):
    """``density``: real-space electron density of the occupied window states.

    Notes
    -----
    The dense kernel projects each eigenvector onto the plane-wave basis of
    the DFT k-point with the same index
    (:func:`~PAOFLOW.hamiltonian.do_real_space.accumulate_density_k`), so the
    PAO mesh has to *be* the DFT grid: no interpolation and no doubling, and
    internal projections (``projections(internal=True)``, which provide
    ``basis``).  The same holds here, checked up front.  Per k-point the
    occupied states of the window are accumulated into one
    ``(nr1, nr2, nr3)`` grid per spin; the dense writer reduces and writes.
    """

    method = 'density'
    label = 'Density'
    interior = 'the density sums every occupied state, and an interior window has none below elo'

    def __init__(self, engine: SparseEngine, nr1: int = 48, nr2: int = 48, nr3: int = 48) -> None:
        super().__init__(engine)
        self.nr = (nr1, nr2, nr3)

    def prepare(self) -> bool:
        engine = self.engine
        arrays, attr = self.data_controller.data_dicts()
        mesh = (attr['nk1'], attr['nk2'], attr['nk3'])
        if engine.H._doubled or mesh != engine.H.nk_grid:
            raise RuntimeError(
                'sparse density: the density projects onto the plane waves of the DFT '
                'k-points by index, so the mesh must be the DFT grid %dx%dx%d of the base cell '
                '(now %dx%dx%d%s). Call density() before doubling_Hamiltonian() and '
                'interpolated_hamiltonian().'
                % (engine.H.nk_grid + mesh + (', doubled' if engine.H._doubled else '',))
            )
        if 'basis' not in arrays:
            raise RuntimeError(
                'sparse density needs the atomic basis of internal projections; run '
                'projections(internal=True) instead of read_atomic_proj_QE().'
            )
        self.rho = np.zeros(self.nr + (attr['nspin'],), dtype=complex, order='C')
        if engine.rank == 0 and attr['verbose']:
            print('Writing density files')
        return True

    def on_k(self, kp: KPoint) -> None:
        from ...hamiltonian.do_real_space import accumulate_density_k

        accumulate_density_k(
            self.rho[:, :, :, kp.ispin], self.data_controller, kp.kglobal, kp.E, kp.V, *self.nr
        )

    def finalize(self, data_controller: DataController) -> None:
        from ...hamiltonian.do_real_space import write_density

        write_density(data_controller, self.rho)
        self.rho = None
