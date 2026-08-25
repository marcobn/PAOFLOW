import numpy as np
from mpi4py import MPI
import time
from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI
from ..utils.perturb_split import perturb_split
from ..utils.communication import reduce_full
from ..utils.smearing import intgaussian, intmetpax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
bohr_to_cm = 5.29177249e-9


def calc_chi(data_controller):
    arry, attr = data_controller.data_dicts()
    # ------------------------------------------------------------
    # Check SO calculation
    # ------------------------------------------------------------
    if attr['response'] == 'ree' or attr['response'] == 'shc':
        if attr['dftSO'] == False:
            if rank == 0:
                print('Relativistic calculation with SO required')
                comm.Abort()
            comm.Barrier()
    # ------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------
    if attr['twoD']:
        av0 = arry['a_vectors'][0, :]
        av1 = arry['a_vectors'][1, :]
        attr['cgs_conv'] = 1.0 / (np.linalg.norm(np.cross(av0, av1)) * attr['alat'] ** 2)
    else:
        attr['cgs_conv'] = 1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])
    # ------------------------------------------------------------
    # Select tensors
    # ------------------------------------------------------------
    if attr['response'] == 'shc':
        my_tensor = arry['shc_tensor']

    if attr['response'] == 'ree' or attr['response'] == 'ahc':
        my_tensor = arry['ree_tensor']
    # ------------------------------------------------------------
    # Calculate each requested tensor
    # ------------------------------------------------------------
    for tensor in my_tensor:
        if rank == 0:
            start_time = time.time()
        calc_chi2(data_controller=data_controller, tensor=tensor)

        if rank == 0:
            end_time = time.time()
            total_time = (end_time - start_time) / 60.0

            if attr['response'] == 'shc':
                print(
                    f'EVEN SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.'
                )

            if attr['response'] == 'ree':
                print(f'ODD REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

            if attr['response'] == 'ahc':
                print(f'AHC [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')


def calc_chi2(data_controller=None, tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    deltap = 0.05
    attr['emaxH'] = np.amin(np.array([attr['shift'], attr['emaxH']]))

    ene = np.linspace(attr['eminH'], attr['emaxH'], attr['esize'])
    esize = attr['esize']

    # ============================================================
    # OPERATOR MATRICES
    # ============================================================

    oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
    oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
    # ------------------------------------------------------------
    # SHC
    # ------------------------------------------------------------
    if attr['response'] == 'shc':
        spol, jpol, ipol = tensor
        jksp_op = spin_current(data_controller=data_controller, tensor=tensor)
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin]) = perturb_split(
                    jksp_op[ik, :, :, ispin],
                    arry['dHksp'][ik, ipol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
        jksp_op = None
    # ------------------------------------------------------------
    # REE
    # ------------------------------------------------------------
    if attr['response'] == 'ree':
        spol, ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin]) = perturb_split(
                    arry['Sj'][spol, :, :],
                    arry['dHksp'][ik, ipol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
    # ------------------------------------------------------------
    # CONDUCTIVITY
    # ------------------------------------------------------------
    if attr['response'] == 'ahc':
        cpol, ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin]) = perturb_split(
                    arry['dHksp'][ik, cpol, :, :, ispin],
                    arry['dHksp'][ik, ipol, :, :, ispin],
                    arry['v_k'][ik, :, :, ispin],
                    arry['degen'][ispin][ik],
                )
    # ============================================================
    # ENERGY-INDEPENDENT BERRY-CURVATURE-LIKE QUANTITY
    # Shape:(local_nk, nbnd, nspin)
    # remove the much larger (nk, esize, nspin) array below.
    # ============================================================

    Om_znkaux = np.zeros((nk, nbnd, nspin), dtype=float)
    for ispin in range(nspin):
        for ik in range(nk):
            E = arry['E_k'][ik, :, ispin]
            # ----------------------------------------------------
            # E_nm = (E_n - E_m)^2 + deltap^2
            # ----------------------------------------------------
            E_nm = (E - E[:, None]) ** 2 + deltap**2
            E_nm[E_nm < 1.0e-4] = np.inf
            numerator = np.imag(oper_matrix1[ik, :, :, ispin] * oper_matrix2[ik, :, :, ispin].T)
            Om_znkaux[ik, :, ispin] = -2.0 * np.sum(numerator / E_nm, axis=1)
            numerator = None
            E_nm = None
    oper_matrix1 = None
    oper_matrix2 = None

    # ============================================================
    # IMPORTANT MEMORY OPTIMIZATION
    # OLD:
    # Om_zkaux = (nk, esize, nspin)
    # Then: gather_full()
    # Then sum over k-points.
    # NEW:
    # local_response = (esize, nspin)
    # perform the k-point summation immediately on each rank.
    # ============================================================

    local_response = np.zeros((esize, nspin), dtype=float)
    # ------------------------------------------------------------
    # Calculate energy-dependent response.
    # ------------------------------------------------------------
    for ispin in range(nspin):
        for i in range(esize):
            # ----------------------------------------------------
            # Calculate smearing function.
            # ----------------------------------------------------
            if attr['smearing'] == 'gauss':
                smear = intgaussian(arry['E_k'][:, :, ispin], ene[i], arry['deltakp'][:, :, ispin])
            elif attr['smearing'] == 'm-p':
                smear = intmetpax(arry['E_k'][:, :, ispin], ene[i], arry['deltakp'][:, :, ispin])
            else:
                smear = 0.5 * (-np.sign(arry['E_k'][:, :, ispin] - ene[i]) + 1)
            # ----------------------------------------------------
            # perform BOTH sums immediately: sum_k sum_n Omega_nk * smear_kn
            # ----------------------------------------------------
            local_response[i, ispin] = np.sum(Om_znkaux[:, :, ispin] * smear)
            smear = None
    Om_znkaux = None
    local_response = np.sum(local_response, axis=1)
    # ============================================================
    # MPI REDUCTION
    # local_response has shape: (esize, nspin)
    # Each rank contains only its local k-point contribution.
    # reduce_full() adds these contributions and places the
    # global result on rank 0.
    # ============================================================
    Om_zk = reduce_full(local_response, sroot=0)
    local_response = None
    # ============================================================
    # GLOBAL NORMALIZATION + UNIT CONVERSION + OUTPUT
    # ============================================================
    if rank == 0:
        # --------------------------------------------------------
        # SHC
        # --------------------------------------------------------
        if attr['response'] == 'shc':
            Om_zk *= attr['cgs_conv']
            Om_zk /= float(attr['nkpnts'])
        if attr['response'] == 'ree':
            Om_zk *= bohr_to_cm
            Om_zk /= float(attr['nkpnts'])
        if attr['response'] == 'ahc':
            Om_zk *= attr['cgs_conv']
            Om_zk /= float(attr['nkpnts'])

    xzy = ['x', 'y', 'z']
    if attr['response'] == 'shc':
        fname = f'SHC_EVEN_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Chi [(hbar/e)(S/cm)]'
    if attr['response'] == 'ree':
        fname = f'REE_ODD_{xzy[spol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Chi [hbar*(cm/V)]'
    if attr['response'] == 'ahc':
        fname = f'AHC_{xzy[cpol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'AH Conductivity [S/cm]'
    # data_controller.write_file_row_col(fname,ene,Om_zk)
    data_controller.write_file_row_col_units(fname, ene, Om_zk, unit1, unit2)
    ene = None
    Om_zk = None


def spin_current(data_controller=None, tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    spol, jspol, ipol = tensor
    Sj = arry['Sj'][spol]
    snktot, _, nawf, nawf, nspin = arry['dHksp'].shape
    jdHksp = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik, :, :, ispin] = 0.5 * (
                np.dot(Sj, arry['dHksp'][ik, jspol, :, :, ispin])
                + np.dot(arry['dHksp'][ik, jspol, :, :, ispin], Sj)
            )
    return jdHksp
