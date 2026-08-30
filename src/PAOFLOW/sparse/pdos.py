"""Streaming projected-DOS accumulator for the fused mesh pass.

Replicates ``spectrum.do_pdos.do_pdos_adaptive`` output exactly — same
energy grid (``emax`` clipped to ``min(shift, emax)``), same per-orbital
file names via ``_build_orbital_prefixes``, same normalization and MPI
reduction — but accumulates the orbital weights ``|V_mn|^2`` per k-point
inside the mesh loop, so the ``(nk, nawf, nawf)`` eigenvector tensor the
dense kernel reads is never stored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class PdosConsumer:
    """Mesh consumer accumulating the projected density of states.

    Parameters
    ----------
    data_controller : DataController
        Run state; supplies ``shift``, ``smearing``, ``nawf`` and ``nspin``.
    emin, emax : float
        Requested energy range (eV, PAO zero of energy).  ``emax`` is
        clipped to the PAO ``shift``, above which the projected
        Hamiltonian carries no physical states.
    ne : int
        Number of points on the energy grid.

    Attributes
    ----------
    ene : np.ndarray, shape (ne,)
        Energy grid the PDOS is tabulated on.
    partial : np.ndarray, shape (nspin, nawf, ne)
        Running per-orbital accumulator, this rank's share of the mesh.

    Notes
    -----
    The projected DOS resolves the density of states by basis orbital,

    .. math::

        \\rho_m(E) = \\frac{1}{N_k} \\sum_{\\mathbf{k}, n}
            |V_{mn}(\\mathbf{k})|^2 \\, \\delta(E - E_n(\\mathbf{k})),

    where :math:`V_{mn}` is the weight of orbital :math:`m` in band
    :math:`n` and the delta function is replaced by a normalized smearing
    kernel of k- and band-dependent width (the Yates adaptive width the
    mesh pass computes from the band velocity).

    The dense kernel evaluates that sum after the fact, from a stored
    ``(nk, nawf, nawf)`` eigenvector tensor.  Here the sum is instead
    accumulated one k-point at a time, while the mesh loop still holds the
    eigenvector block: the weights ``|V_mn|^2`` are contracted with the
    smearing kernel immediately and only the ``(nspin, nawf, ne)`` result
    survives the k-point.  That result is independent of mesh size, so the
    memory cost of the PDOS does not grow with the k-mesh or with cell
    doubling.

    Because each rank walks its own share of the mesh, ``partial`` holds a
    partial sum until :meth:`finalize` reduces across ranks and divides by
    the total number of k-points.
    """

    def __init__(self, data_controller: DataController, emin: float, emax: float, ne: int) -> None:
        arrays, attr = data_controller.data_dicts()
        emax = min(attr['shift'], emax)
        self.ene = np.linspace(emin, emax, ne)
        self.smearing = attr['smearing']
        self.nawf = attr['nawf']
        self.nspin = attr['nspin']
        self.partial = np.zeros((self.nspin, self.nawf, ne), dtype=float)

    def on_k(
        self,
        ik: int,
        ispin: int,
        E: np.ndarray,
        V: np.ndarray,
        vel: np.ndarray,
        delta: np.ndarray,
    ) -> None:
        """Add one k-point's orbital weights to the accumulator.

        Parameters
        ----------
        ik : int
            Local k-point index; unused, the contribution is a plain sum.
        ispin : int
            Spin channel.
        E : np.ndarray, shape (nev,)
            Eigenvalues at this k-point (eV), ascending.
        V : np.ndarray, shape (nawf, nev)
            Eigenvector block.  Read here and not retained, per the mesh
            consumer contract.
        vel : np.ndarray, shape (3, nev)
            Band velocities; unused, they enter only through ``delta``.
        delta : np.ndarray, shape (nev,)
            Adaptive smearing width of each band at this k-point (eV).

        Notes
        -----
        The smearing kernel is evaluated on the outer product of the energy
        grid with the eigenvalues, giving an ``(ne, nev)`` matrix, and the
        orbital weights ``|V_mn|^2`` are contracted against it in one
        ``(nawf, nev) @ (nev, ne)`` matrix product.  Both smearing functions
        depend on ``ene`` and ``eig`` only through their difference, so the
        broadcast is symmetric in the two arguments.
        """
        from ..utils.smearing import gaussian, metpax

        func = gaussian if self.smearing == 'gauss' else metpax
        G = func(self.ene[:, None], E[None, :], delta[None, :])
        self.partial[ispin] += np.abs(V) ** 2 @ G.T

    def finalize(self, data_controller: DataController) -> None:
        """Reduce across ranks, normalize, and write the PDOS files.

        Parameters
        ----------
        data_controller : DataController
            Run state; supplies the orbital labels used for file names and
            performs the (collective) file writes.

        Notes
        -----
        Writes one ``*_pdosdk_<ispin>.dat`` file per basis orbital plus the
        summed ``pdosdk_sum_<ispin>.dat``, matching the dense kernel's file
        set name for name.  Every rank calls ``write_file_row_col`` for
        every file because that call is collective; only rank 0 passes data,
        the others pass ``None``.
        """
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
                        f'{prefixes[m]}_pdosdk_{ispin}.dat', self.ene, pdos[m]
                    )
            else:
                pdos_sum = None
                for m in range(self.nawf):
                    data_controller.write_file_row_col(
                        f'{prefixes[m]}_pdosdk_{ispin}.dat', self.ene, None
                    )

            data_controller.write_file_row_col(f'pdosdk_sum_{ispin}.dat', self.ene, pdos_sum)
