import numpy as np
from mpi4py import MPI

from ..utils.get_K_grid_fft import get_K_grid_fft_crystal

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def write4bt2(data_controller):
    """Write eigenvalues and structure in GENE format for BoltzTraP2.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpnts, nbnds, nspin)``),
        ``a_vectors`` (shape ``(3, 3)``), ``b_vectors`` (shape ``(3, 3)``),
        ``atoms`` (list of element symbols), ``tau``
        (shape ``(natoms, 3)``).  Optionally ``pksp`` for band derivatives.
        Required attributes: ``nkpnts``, ``nspin``, ``nbnds``, ``natoms``,
        ``alat``, ``nk1``, ``nk2``, ``nk3``, ``savedir``.

    Returns
    -------
    Optional[np.ndarray]
        If ``pksp`` is available, returns the momentum matrix diagonal
        converted to Rydberg units as shape ``(nkpnts, nbnds, 3)``.
        Returns ``None`` if ``pksp`` is not present.

    Notes
    -----
    Two files are written to the directory specified by ``savedir``
    (prefix obtained by removing the file extension):

    - ``{prefix}.energy``: k-point coordinates in crystal units, number
      of bands, and band eigenvalues converted from eV to Rydberg
      (``Ry2eV = 13.60569193``).
    - ``{prefix}.structure``: lattice vectors in Bohr, number of atoms,
      and atomic positions in crystal coordinates.

    Only MPI rank 0 performs file I/O.  The function must be called
    after :meth:`pao_hamiltonian`, :meth:`pao_eigh`, and
    :meth:`gradient_and_momenta` (the last only when band derivatives
    are needed).
    """

    if rank == 0:
        Ry2eV = 13.60569193
        arry, attr = data_controller.data_dicts()

        arry['kgrid'] = get_K_grid_fft_crystal(attr['nk1'], attr['nk2'], attr['nk3'])

        # write data in the GENE format for BoltzTrap2
        prefix = attr['savedir'].split('.')[0]
        fname_energy = prefix + '.energy'
        fname_struct = prefix + '.structure'

        f_energy = prefix + '\n'
        f_energy += str(attr['nkpnts']) + ' ' + str(int(attr['nspin'])) + ' ' + str(0) + '\n'
        for ik in range(attr['nkpnts']):
            f_energy += (
                str(arry['kgrid'][ik][0])
                + ' '
                + str(arry['kgrid'][ik][1])
                + ' '
                + str(arry['kgrid'][ik][2])
                + ' '
                + str(attr['nbnds'])
                + '\n'
            )
            for ib in range(attr['nbnds']):
                f_energy += str(arry['E_k'][ik, ib, 0] / Ry2eV) + '\n'

        f = open(fname_energy, 'w')
        f.write(f_energy)
        f.close()

        f_struct = prefix + '\n'
        for i in range(3):
            f_struct += (
                str(arry['a_vectors'][i][0] * attr['alat'])
                + ' '
                + str(arry['a_vectors'][i][1] * attr['alat'])
                + ' '
                + str(arry['a_vectors'][i][2] * attr['alat'])
                + '\n'
            )
        f_struct += str(attr['natoms']) + '\n'
        for ia in range(attr['natoms']):
            f_struct += str(arry['atoms'][ia]) + ' '
            f_struct += (
                str(arry['tau'][ia].dot(arry['b_vectors'].T)[0] / attr['alat'])
                + ' '
                + str(arry['tau'][ia].dot(arry['b_vectors'].T)[1] / attr['alat'])
                + ' '
                + str(arry['tau'][ia].dot(arry['b_vectors'].T)[2] / attr['alat'])
                + '\n'
            )

        f = open(fname_struct, 'w')
        f.write(f_struct)
        f.close()

        try:
            mommat = np.zeros((attr['nkpnts'], attr['nbnds'], 3), dtype=float)
            for ib in range(attr['nbnds']):
                mommat[:, ib, :] = -np.real(arry['pksp'][:, :, ib, ib, 0]) / (2 * Ry2eV)
            return mommat
        except:
            print('momentum matrix not available')
            return None
