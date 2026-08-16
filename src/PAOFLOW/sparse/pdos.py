"""Streaming projected-DOS accumulator for the fused mesh pass.

Replicates ``spectrum.do_pdos.do_pdos_adaptive`` output exactly — same
energy grid (``emax`` clipped to ``min(shift, emax)``), same per-orbital
file names via ``_build_orbital_prefixes``, same normalization and MPI
reduction — but accumulates the orbital weights ``|V_mn|^2`` per k-point
inside the mesh loop, so the ``(nk, nawf, nawf)`` eigenvector tensor the
dense kernel reads is never stored.
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class PdosConsumer:
    def __init__(self, data_controller, emin, emax, ne):
        arrays, attr = data_controller.data_dicts()
        emax = min(attr['shift'], emax)
        self.ene = np.linspace(emin, emax, ne)
        self.smearing = attr['smearing']
        self.nawf = attr['nawf']
        self.nspin = attr['nspin']
        self.partial = np.zeros((self.nspin, self.nawf, ne), dtype=float)

    def on_k(self, ik, ispin, E, V, vel, delta):
        from ..utils.smearing import gaussian, metpax

        func = gaussian if self.smearing == 'gauss' else metpax
        # (ne, nev) smearing kernel; gaussian() is symmetric in (ene, eig)
        G = func(self.ene[:, None], E[None, :], delta[None, :])
        self.partial[ispin] += np.abs(V) ** 2 @ G.T

    def finalize(self, data_controller):
        from ..spectrum.do_pdos import _build_orbital_prefixes

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
                        '%s_pdosdk_%d.dat' % (prefixes[m], ispin), self.ene, pdos[m]
                    )
            else:
                pdos_sum = None
                for m in range(self.nawf):
                    data_controller.write_file_row_col(
                        '%s_pdosdk_%d.dat' % (prefixes[m], ispin), self.ene, None
                    )

            data_controller.write_file_row_col('pdosdk_sum_%d.dat' % ispin, self.ene, pdos_sum)
