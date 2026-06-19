#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np


# Compute Z2 invariant and topological properties on a selected path in the BZ
def do_berry_curvature(data_controller):
    from mpi4py import MPI
    from ..utils.constants import LL, ANGSTROM_AU
    from ..utils.get_R_grid_fft import get_R_grid_fft
    from ..utils.communication import scatter_full, gather_full
    from ..spectrum.kpnts_interpolation_mesh import kpnts_interpolation_mesh

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arrays, attributes = data_controller.data_dicts()

    npool = attributes['npool']

    if 'kq' not in arrays:
        kpnts_interpolation_mesh(data_controller)

    HRs = arrays['HRs']
    bnd = attributes['nawf']
    nkpi = arrays['kq'].shape[1]
    nawf, _, nk1, nk2, nk3, nspin = HRs.shape

    if 'Rfft' not in arrays:
        get_R_grid_fft(data_controller, nk1, nk2, nk3)

    ipol = attributes['ipol']
    jpol = attributes['jpol']
    spol = attributes['spol']

    curvature = attributes['curvature']

    alat = attributes['alat'] / ANGSTROM_AU
    b_vectors = arrays['b_vectors']

    # Compute momenta and kinetic energy
    kq_aux = scatter_full(arrays['kq'].T, npool)
    kq_aux = kq_aux.T

    # Compute R*H(R)
    Rfft = np.reshape(arrays['Rfft'], (nk1 * nk2 * nk3, 3), order='C')
    HRs = np.reshape(HRs, (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')
    HRs = np.moveaxis(HRs, 2, 0)

    HRs_aux = scatter_full(HRs, npool)
    Rfft_aux = scatter_full(Rfft, npool)

    HRs = np.reshape(np.moveaxis(HRs, 0, 2), (nawf, nawf, nk1, nk2, nk3, nspin), order='C')

    Oj = arrays['Oj']
    jks = np.zeros((kq_aux.shape[1], 3, bnd, bnd, nspin), dtype=complex)

    pks = np.zeros((kq_aux.shape[1], 3, bnd, bnd, nspin), dtype=complex)
    for l in range(3):
        dHRs = np.zeros((HRs_aux.shape[0], nawf, nawf, nspin), dtype=complex)
        for ispin in range(nspin):
            for n in range(nawf):
                for m in range(nawf):
                    dHRs[:, n, m, ispin] = (
                        1.0j * alat * ANGSTROM_AU * Rfft_aux[:, l] * HRs_aux[:, n, m, ispin]
                    )

        dHRs = gather_full(dHRs, npool)
        if rank != 0:
            dHRs = np.zeros((nk1 * nk2 * nk3, nawf, nawf, nspin), dtype=complex)
        comm.Bcast(dHRs)
        dHRs = np.moveaxis(dHRs, 0, 2)

        # Compute dH(k)/dk on the path
        dHks_aux = band_loop_H(dHRs, Rfft, kq_aux, nawf, nspin)

        dHRs = None

        # Compute momenta
        for ik in range(dHks_aux.shape[0]):
            for ispin in range(nspin):
                pks[ik, l, :, :, ispin] = (
                    np.conj(arrays['v_k'][ik, :, :, ispin].T)
                    .dot(dHks_aux[ik, :, :, ispin])
                    .dot(arrays['v_k'][ik, :, :, ispin])[:bnd, :bnd]
                )

        for ik in range(pks.shape[0]):
            for ispin in range(nspin):
                jks[ik, l, :, :, ispin] = (
                    np.conj(arrays['v_k'][ik, :, :, ispin].T)
                    .dot(
                        0.5
                        * (
                            np.dot(Oj[spol], dHks_aux[ik, :, :, ispin])
                            + np.dot(dHks_aux[ik, :, :, ispin], Oj[spol])
                        )
                    )
                    .dot(arrays['v_k'][ik, :, :, ispin])
                )[:bnd, :bnd]

    deltab = 0.05
    mu = -0.02  # chemical potential in eV)
    Omj_zk = np.zeros((pks.shape[0], 1), dtype=float)
    Omj_znk = np.zeros((pks.shape[0], bnd), dtype=float)
    for ik in range(pks.shape[0]):
        for n in range(bnd):
            for m in range(bnd):
                if m != n:
                    Omj_znk[ik, n] += (
                        -2.0
                        * np.imag(jks[ik, ipol, n, m, 0] * pks[ik, jpol, m, n, 0])
                        / ((arrays['E_k'][ik, m, 0] - arrays['E_k'][ik, n, 0]) ** 2 + deltab**2)
                    )
        Omj_zk[ik] = np.sum(
            Omj_znk[ik, :] * (0.5 * (1 - np.sign(arrays['E_k'][ik, :bnd, 0] - mu)))
        )  # T=0.0K

    indices = (curvature, LL[spol], LL[ipol], LL[jpol])
    lrng = list(range(nkpi)) if rank == 0 else None

    pks = gather_full(pks, npool)
    velk = np.zeros((nkpi, 3, bnd, nspin), dtype=float) if rank == 0 else None
    if rank == 0:
        for n in range(bnd):
            velk[:, :, n, :] = np.real(pks[:, :, n, n, :])
    for l in range(3):
        fvk = 'velocity_' + str(l)
        data_controller.write_bands(fvk, (velk[:, l, :bnd, :] if rank == 0 else None))
    pks = velk = None

    Omj_zk = gather_full(Omj_zk, npool)
    fOmj_zk = 'Omegaj_%s_%s_%s%s.dat' % (indices)

    data_controller.write_file_row_col(fOmj_zk, lrng, (Omj_zk[:, 0] if rank == 0 else None))
    Omj_zk = fOmj_zk = None


def band_loop_H(HRaux, R, kq, nawf, nspin):
    kdot = np.zeros((kq.shape[1], R.shape[0]), dtype=complex, order='C')
    kdot = np.tensordot(R, 2.0j * np.pi * kq, axes=([1], [0]))
    np.exp(kdot, kdot)

    Haux = np.zeros((nawf, nawf, kq.shape[1], nspin), dtype=complex, order='C')

    for ispin in range(nspin):
        Haux[:, :, :, ispin] = np.tensordot(HRaux[:, :, :, ispin], kdot, axes=([2], [0]))

    kdot = None
    Haux = np.transpose(Haux, (2, 0, 1, 3))
    return Haux
