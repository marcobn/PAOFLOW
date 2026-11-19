"""Streaming projected-DOS accumulator for the fused mesh pass (part of ``dos``).

Same output as ``spectrum.do_pdos.do_pdos_adaptive`` (energy grid, per-orbital file names,
normalization), but the weights ``|V_mn|^2`` are reduced per k-point, so the
``(nk, nawf, nawf)`` eigenvector tensor the dense kernel reads is never stored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from ..kpoint import KPoint

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class PdosAccumulator:
    """Mesh-pass accumulator of the projected DOS, adaptive (Yates) smearing.

    Parameters
    ----------
    data_controller : DataController
        Run state; supplies ``shift``, ``smearing``, ``nawf`` and ``nspin``.
    emin, emax : float
        Energy range (eV, PAO zero).  ``emax`` is clipped to ``shift``: no physical states above.
    ne : int
        Number of energy grid points.

    Attributes
    ----------
    ene : np.ndarray, shape (ne,)
    partial : np.ndarray, shape (nspin, nawf, ne)
        This rank's partial sum; :meth:`finalize` reduces it and divides by ``nkpnts``.
    """

    def __init__(self, data_controller: DataController, emin: float, emax: float, ne: int) -> None:
        arrays, attr = data_controller.data_dicts()
        emax = min(attr['shift'], emax)
        self.ene = np.linspace(emin, emax, ne)
        self.smearing = attr['smearing']
        self.nawf = attr['nawf']
        self.nspin = attr['nspin']
        self.partial = np.zeros((self.nspin, self.nawf, ne), dtype=float)

    def on_k(self, kp: KPoint) -> None:
        """Add ``|V_mn|^2 @ G(ene - E_n)^T`` for the window states ``E[:bnd]``."""
        from ...utils.smearing import gaussian, metpax

        nev = kp.bnd
        func = gaussian if self.smearing == 'gauss' else metpax
        G = func(self.ene[:, None], kp.E[None, :nev], kp.delta[None, :])
        self.partial[kp.ispin] += np.abs(kp.V[:, :nev]) ** 2 @ G.T

    def finalize(self, data_controller: DataController) -> None:
        """Reduce, normalize and write ``*_pdosdk_<ispin>.dat`` and ``pdosdk_sum_<ispin>.dat``.

        ``write_file_row_col`` is collective: every rank calls it, only rank 0 passes data.
        """
        from ...spectrum.do_pdos import _build_orbital_prefixes

        arrays, attr = data_controller.data_dicts()
        prefixes = _build_orbital_prefixes(arrays, self.nawf)

        for ispin in range(self.nspin):
            pdos = np.zeros((self.nawf, len(self.ene)), dtype=float) if rank == 0 else None
            comm.Reduce(np.ascontiguousarray(self.partial[ispin]), pdos, op=MPI.SUM)

            if rank == 0:
                pdos /= float(attr['nkpnts'])
                pdos_sum = np.zeros(len(self.ene), dtype=float)
                for m in range(self.nawf):
                    pdos_sum += pdos[m]
                    data_controller.write_file_row_col(
                        f'{prefixes[m]}_pdosdk_{ispin}.dat', self.ene, pdos[m]
                    )
            else:
                pdos_sum = None
                for m in range(self.nawf):
                    data_controller.write_file_row_col(
                        f'{prefixes[m]}_pdosdk_{ispin}.dat', self.ene, None
                    )

            data_controller.write_file_row_col(f'pdosdk_sum_{ispin}.dat', self.ene, pdos_sum)
