#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
from mpi4py import MPI

from .communication import gather_scatter
from .do_eigh import get_degeneracies
from .perturb_split import perturb_split

# initialize parallel execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scipy import fftpack as FFT


def do_d2Hd2k_ij(Hksp, dHksp, Dnm, Rfft, alat, npool, v_kp, bnd, degen):
    # ----------------------
    # Compute the gradient of the k-space Hamiltonian
    # ----------------------
    Rfft = np.transpose(Rfft, (3, 0, 1, 2))

    num_n, nk1, nk2, nk3, nspin = Hksp.shape

    _, nk1, nk2, nk3, nspin = Hksp.shape

    M_ij = np.zeros((6, v_kp.shape[0], bnd, v_kp.shape[3]), dtype=float, order='C')
    ij_ind = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]], dtype=int)

    comm.Barrier()
    ########################################
    ### real space grid replaces k space ###
    ########################################
    # c1=c2=0
    # for ik in range(len(degen[0])):
    #     if len(degen[0][ik]) != 0:
    #         c1+=1
    #     else:
    #         c2+=1
    # print(c1+c2,c1,c2)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    num_n = Hksp.shape[0]

    dvec_list = []
    vel_degen = []
    degen_M = []

    for ij in range(M_ij.shape[0]):
        dir_tmp = []
        vel_degen_by_ij = []
        degen_M_by_ij = []
        d2Hksp = None
        d2Hksp = np.zeros((num_n, nk1, nk2, nk3, nspin), dtype=complex, order='C')

        ipol = ij_ind[ij][0]
        jpol = ij_ind[ij][1]

        RIJ = Rfft[ipol] * Rfft[jpol]

        for ispin in range(d2Hksp.shape[4]):
            for n in range(d2Hksp.shape[0]):
                # because of the way this is coded...Hksp is actually HR*1.0j*alat
                d2Hksp[n, :, :, :, ispin] = (
                    FFT.fftn(RIJ * Hksp[n, :, :, :, ispin] * 1.0j * alat)
                    + Dnm[n, ipol] * Dnm[n, jpol] * FFT.fftn(Hksp[n, :, :, :, ispin] * 1.0j / alat)
                    + Dnm[n, ipol] * FFT.fftn(Rfft[jpol] * Hksp[n, :, :, :, ispin] * 1.0j)
                    + Dnm[n, jpol] * FFT.fftn(Rfft[ipol] * Hksp[n, :, :, :, ispin] * 1.0j)
                )

        #############################################################################################
        #############################################################################################
        #############################################################################################

        # gather the arrays into flattened dHk
        d2Hksp = np.reshape(d2Hksp, (num_n, nk1 * nk2 * nk3, nspin), order='C')
        d2Hksp = gather_scatter(d2Hksp, 1, npool)
        nawf = int(np.sqrt(d2Hksp.shape[0]))

        d2Hksp = np.reshape(d2Hksp, (nawf, nawf, d2Hksp.shape[1], nspin), order='C')

        tksp = np.zeros_like(d2Hksp)

        # find non-degenerate set of psi(k) for d2H/d2k_ij
        for ispin in range(tksp.shape[3]):
            isp_tmp = []
            vel_degen_by_spin = []
            for ik in range(tksp.shape[2]):
                # we save dvec so that it can be used when calculating the second term in d2E/d2k
                v_aux, tksp[:, :, ik, ispin], dvec = perturb_split(
                    dHksp[ik, ipol, :, :, ispin],
                    d2Hksp[:, :, ik, ispin],
                    v_kp[ik, :, :, ispin],
                    degen[ispin][ik],
                    return_v_k=True,
                )

                vel_degen_by_kp = get_degeneracies(
                    v_aux.diagonal().reshape((1, len(v_aux.diagonal()), 1)), bnd
                )

                # vel_degen_by_spin.append(vel_degen_by_kp[vel_degen_by_kp == degen[ispin][ik]])
                vel_degen_by_spin.append(vel_degen_by_kp[0][0])

                isp_tmp.append(dvec)
            vel_degen.append(vel_degen_by_spin)
            dir_tmp.append(isp_tmp)
        vel_degen.append(vel_degen_by_ij)
        dvec_list.append(dir_tmp)

        # get the value for d2H/d2k
        for ispin in range(tksp.shape[3]):
            for n in range(bnd):
                M_ij[ij, :, n, ispin] = tksp[n, n, :, ispin].real

        for ispin in range(tksp.shape[3]):
            degen_M_by_spin = []
            for ik in range(tksp.shape[2]):
                degen_M_by_kp = []
                for i in range(len(vel_degen[ij][ispin][ik])):
                    # degenerate subspace indices upper and lower lim
                    ll = vel_degen[ij][ispin][ik][i][0]
                    ul = vel_degen[ij][ispin][ik][i][-1] + 1

                    degen_M_by_kp.append(tksp[ll:ul, ll:ul, ik, ispin])
                degen_M_by_spin.append(degen_M_by_kp)
            degen_M_by_ij.append(degen_M_by_spin)
        degen_M.append(degen_M_by_ij)
        comm.Barrier()

    d2Hksp = None

    return M_ij, vel_degen, degen_M, dvec_list
