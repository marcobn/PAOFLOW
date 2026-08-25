import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI
from ..utils.perturb_split import perturb_split
from ..utils.communication import reduce_full
from ..utils.smearing import gaussian, intgaussian


bohr_to_cm = 5.29177249e-9


def calc_chi(data_controller):
    arry, attr = data_controller.data_dicts()
    if attr['response'] == 'ree' or attr['response'] == 'shc':
        if attr['dftSO'] == False:
            if rank == 0:
                print('Relativistic calculation with SO required')
                comm.Abort()
            comm.Barrier()
    nk, nbnd, nspin = arry['E_k'].shape

    if attr['response'] == 'shc' and attr['full_chi2'] and attr['t_odd'] == False:
        for tensor in arry['s_tensor']:
            spol, jpol, ipol = tensor[0], tensor[1], tensor[2]
            if rank == 0:
                start_time = time.time()
            jdHksp = do_spin_current(data_controller=data_controller, tensor=tensor)
            oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            for ik in range(jdHksp.shape[0]):
                for ispin in range(jdHksp.shape[3]):
                    oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = perturb_split(
                        jdHksp[ik, :, :, ispin],
                        arry['dHksp'][ik, ipol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            jdHksp = None
            calc_chi2(
                data_controller=data_controller,
                tensor=tensor,
                prop='shc',
                oper_matrix1=oper_matrix1,
                oper_matrix2=oper_matrix2,
            )
            oper_matrix1 = oper_matrix2 = None
            jdHksp = None
            if rank == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 60.0
                print(
                    f'EVEN SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.'
                )

    if attr['response'] == 'shc' and attr['t_odd'] == True:
        if attr['intraband'] or attr['interband']:
            for tensor in arry['s_tensor']:
                spol, jpol, ipol = tensor[0], tensor[1], tensor[2]
                if rank == 0:
                    start_time = time.time()
                jdHksp = do_spin_current(data_controller=data_controller, tensor=tensor)
                oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                for ik in range(jdHksp.shape[0]):
                    for ispin in range(jdHksp.shape[3]):
                        oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = (
                            perturb_split(
                                jdHksp[ik, :, :, ispin],
                                arry['dHksp'][ik, ipol, :, :, ispin],
                                arry['v_k'][ik, :, :, ispin],
                                arry['degen'][ispin][ik],
                            )
                        )
                jdHksp = None
                if attr['intraband']:
                    fermi_surf(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='shc',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                if attr['interband']:
                    fermi_sea(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='shc',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                oper_matrix1 = oper_matrix2 = None
                jdHksp = None
                if rank == 0:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 60.0
                    print(
                        f'ODD SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.'
                    )
        else:
            pass

    if attr['response'] == 'ree' and attr['full_chi2'] and attr['t_odd'] == True:
        for tensor in arry['ree_tensor']:
            if rank == 0:
                start_time = time.time()
            spol, ipol = tensor[0], tensor[1]
            oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            spol, ipol = tensor[0], tensor[1]
            for ik in range(nk):
                for ispin in range(nspin):
                    oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = perturb_split(
                        arry['Sj'][spol, :, :],
                        arry['dHksp'][ik, ipol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            calc_chi2(
                data_controller=data_controller,
                tensor=tensor,
                prop='ree',
                oper_matrix1=oper_matrix1,
                oper_matrix2=oper_matrix2,
            )
            oper_matrix1 = oper_matrix2 = None
            if rank == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 60.0
                print(f'ODD REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

    if attr['response'] == 'ree' and attr['t_odd'] == False:
        if attr['intraband'] or attr['interband']:
            for tensor in arry['ree_tensor']:
                if rank == 0:
                    start_time = time.time()
                spol, ipol = tensor[0], tensor[1]
                oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                spol, ipol = tensor[0], tensor[1]
                for ik in range(nk):
                    for ispin in range(nspin):
                        oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = (
                            perturb_split(
                                arry['Sj'][spol, :, :],
                                arry['dHksp'][ik, ipol, :, :, ispin],
                                arry['v_k'][ik, :, :, ispin],
                                arry['degen'][ispin][ik],
                            )
                        )
                if attr['intraband']:
                    fermi_surf(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='ree',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                if attr['interband']:
                    fermi_sea(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='ree',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                oper_matrix1 = oper_matrix2 = None
                if rank == 0:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 60.0
                    print(f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

    if attr['response'] == 'ahc' and attr['full_chi2']:
        for tensor in arry['ree_tensor']:
            if rank == 0:
                start_time = time.time()
            cpol, ipol = tensor[0], tensor[1]
            oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
            cpol, ipol = tensor[0], tensor[1]
            for ik in range(nk):
                for ispin in range(nspin):
                    oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = perturb_split(
                        arry['dHksp'][ik, cpol, :, :, ispin],
                        arry['dHksp'][ik, ipol, :, :, ispin],
                        arry['v_k'][ik, :, :, ispin],
                        arry['degen'][ispin][ik],
                    )
            calc_chi2(
                data_controller=data_controller,
                tensor=tensor,
                prop='ahc',
                oper_matrix1=oper_matrix1,
                oper_matrix2=oper_matrix2,
            )
            oper_matrix1 = oper_matrix2 = None
            if rank == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 60.0
                print(f'AHC [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')

    if attr['response'] == 'cond':
        if attr['intraband'] or attr['interband']:
            for tensor in arry['ree_tensor']:
                if rank == 0:
                    start_time = time.time()
                cpol, ipol = tensor[0], tensor[1]
                oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
                cpol, ipol = tensor[0], tensor[1]
                for ik in range(nk):
                    for ispin in range(nspin):
                        oper_matrix1[ik, :, :, ispin], oper_matrix2[ik, :, :, ispin] = (
                            perturb_split(
                                arry['dHksp'][ik, cpol, :, :, ispin],
                                arry['dHksp'][ik, ipol, :, :, ispin],
                                arry['v_k'][ik, :, :, ispin],
                                arry['degen'][ispin][ik],
                            )
                        )
                if attr['intraband']:
                    fermi_surf(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='cond',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                if attr['interband']:
                    fermi_sea(
                        data_controller=data_controller,
                        tensor=tensor,
                        prop='cond',
                        oper_matrix1=oper_matrix1,
                        oper_matrix2=oper_matrix2,
                    )
                oper_matrix1 = oper_matrix2 = None
                if rank == 0:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 60.0
                    print(f'Cond [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')


def calc_chi2(data_controller=None, tensor=None, prop=None, oper_matrix1=None, oper_matrix2=None):
    """EQUATION (2)"""
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    attr['emaxH'] = np.amin(np.array([attr['shift'], attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'], attr['esize'])
    gamma = attr['gamma']
    deltab = 0.001
    # ------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------
    if attr['twoD']:
        av0 = arry['a_vectors'][0, :]
        av1 = arry['a_vectors'][1, :]
        cgs_conv = 1.0 / (np.linalg.norm(np.cross(av0, av1)) * attr['alat'] ** 2)
    else:
        cgs_conv = 1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])

    if prop == 'shc' or prop == 'ree':
        # --------------------------------------------------------
        # aux1 array is only nk x nbnd
        # --------------------------------------------------------
        aux1 = np.zeros((nk, nbnd), dtype=float)
        for ik in range(nk):
            E = arry['E_k'][ik, :, 0]
            # E_nm = (E_n - E_m)^2
            E_nm = (E - E[:, None]) ** 2
            # ----------------------------------------------------
            # Numerator: oper_matrix1[n,m] * oper_matrix2[m,n]
            # ----------------------------------------------------
            aux = oper_matrix1[ik, :, :, 0] * oper_matrix2[ik, :, :, 0].T
            # Remove diagonal contribution
            np.fill_diagonal(aux, 0.0)
            # ----------------------------------------------------
            # Equation (2)
            # ----------------------------------------------------
            aux = 2.0 * np.imag(aux) * (gamma**2 - E_nm)
            aux /= (E_nm + gamma**2) ** 2 + deltab**2
            # Sum over m
            aux1[ik, :] = np.sum(aux, axis=1)
            aux = None
            E_nm = None
        # --------------------------------------------------------
        # Directly accumulate the total contribution for every energy:
        # response(E) = sum_k,n aux1[k,n] * smear[k,n]
        # local_response has only esize elements.
        # --------------------------------------------------------
        local_response = np.zeros(len(ene), dtype=float)
        for ie in range(len(ene)):
            smear = intgaussian(arry['E_k'][:, :, 0], ene[ie], arry['deltakp'][:, :, 0])
            # Sum over k and bands
            local_response[ie] = np.sum(aux1 * smear)
            smear = None
        aux1 = None
        # --------------------------------------------------------
        # Sum the contributions from all MPI ranks.
        # Each rank contains only the k-points assigned to it.
        # reduce_full performs:
        # response_total(E)
        #     = sum_rank response_rank(E)
        # Result exists on rank 0.
        # --------------------------------------------------------
        aux2_full = reduce_full(local_response, sroot=0)
        local_response = None
        if rank == 0:
            # Average over k-points
            aux2_full = aux2_full / aux2_full.shape[0] if False else aux2_full
            # Since reduce_full has already summed over k,now divide by total k-points.
            aux2_full /= attr['nkpnts']

            if prop == 'shc':
                aux2_full *= cgs_conv
            if prop == 'ree':
                aux2_full *= bohr_to_cm

        xzy = ['x', 'y', 'z']
        if prop == 'shc':
            spol, jpol, ipol = (tensor[0], tensor[1], tensor[2])
            fname = f'SHC_EVEN_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Chi [(hbar/e)(S/cm)]'

        if prop == 'ree':
            spol, ipol = tensor[0], tensor[1]
            fname = f'REE_ODD_{xzy[spol]}{xzy[ipol]}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Chi [hbar*(cm/V)]'

        # data_controller.write_file_row_col(fname,ene,aux2_full)
        data_controller.write_file_row_col_units(fname, ene, aux2_full, unit1, unit2)
        aux2_full = None
    if prop == 'ahc':
        for ispin in range(nspin):
            # ----------------------------------------------------
            # band-resolved quantity for every local k-point.
            # ----------------------------------------------------
            aux1 = np.zeros((nk, nbnd), dtype=float)
            for ik in range(nk):
                E = arry['E_k'][ik, :, ispin]
                # E_nm = (E_n - E_m)^2
                E_nm = (E - E[:, None]) ** 2
                aux = oper_matrix1[ik, :, :, ispin] * oper_matrix2[ik, :, :, ispin].T
                np.fill_diagonal(aux, 0.0)
                aux = -2.0 * np.imag(aux) * (gamma**2 - E_nm)
                aux /= (E_nm + gamma**2) ** 2 + deltab**2
                # Sum over m
                aux1[ik, :] = np.sum(aux, axis=1)
                aux = None
                E_nm = None
            # ----------------------------------------------------
            # Directly accumulate the energy-dependent result.
            # No (nk, esize) array is created.
            # ----------------------------------------------------
            local_response = np.zeros(len(ene), dtype=float)
            for ie in range(len(ene)):
                smear = intgaussian(arry['E_k'][:, :, ispin], ene[ie], arry['deltakp'][:, :, ispin])
                local_response[ie] = np.sum(aux1 * smear)
                smear = None
            aux1 = None
            # ----------------------------------------------------
            # MPI reduction over distributed k-points
            # ----------------------------------------------------
            aux2_full = reduce_full(local_response, sroot=0)
            local_response = None
            if rank == 0:
                # Average over total number of k-points
                aux2_full /= attr['nkpnts']
                # Conductivity unit conversion
                aux2_full *= cgs_conv
            xzy = ['x', 'y', 'z']
            cpol, ipol = (tensor[0], tensor[1])
            if nspin == 1:
                fname = f'AHC_{xzy[cpol]}{xzy[ipol]}.dat'
            else:
                fname = f'AHC_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'AH Conductivity [S/cm]'
            # data_controller.write_file_row_col(fname,ene,aux2_full)
            data_controller.write_file_row_col_units(fname, ene, aux2_full, unit1, unit2)
            aux2_full = None
    oper_matrix1 = None
    oper_matrix2 = None


def fermi_surf(data_controller=None, tensor=None, prop=None, oper_matrix1=None, oper_matrix2=None):
    """Equation (4)"""
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    gamma = attr['gamma']
    deltab = 0.0001
    attr['emaxH'] = np.amin(np.array([attr['shift'], attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'], attr['esize'])
    # ------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------
    if attr['twoD']:
        av0 = arry['a_vectors'][0, :]
        av1 = arry['a_vectors'][1, :]
        cgs_conv = 1.0 / (np.linalg.norm(np.cross(av0, av1)) * attr['alat'] ** 2)
    else:
        cgs_conv = 1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])
    # ============================================================
    # SHC / REE
    # ============================================================
    if prop == 'shc' or prop == 'ree':
        # --------------------------------------------------------
        # Extract diagonal matrix elements.
        # --------------------------------------------------------
        oper_matrix1 = np.diagonal(oper_matrix1[:, :, :, 0], axis1=1, axis2=2)
        oper_matrix2 = np.diagonal(oper_matrix2[:, :, :, 0], axis1=1, axis2=2)
        # --------------------------------------------------------
        # numerator = np.real(oper_matrix1 * oper_matrix2)
        # Then symmetrize: ##HACK: seems like symmetrization doesn't really change the result
        # 1/2 [A + A^T]
        # --------------------------------------------------------
        numerator = np.real(oper_matrix1 * oper_matrix2)
        for ik in range(nk):
            numerator[ik, :] = 0.5 * (numerator[ik, :] + np.conj(numerator[ik, :].T))
        oper_matrix1 = None
        oper_matrix2 = None
        # --------------------------------------------------------
        # Instead of creating: aux_diag = (nk, esize)
        # we directly accumulate the total contribution for each energy.
        # local_response has only:(esize,)
        # --------------------------------------------------------
        local_response = np.zeros(len(ene), dtype=float)
        # Flatten once so that the final contraction can be performed as a dot product.
        numerator_flat = numerator.ravel()
        for ie in range(len(ene)):
            smear = gaussian(arry['E_k'][:, :, 0], ene[ie], arry['deltakp'][:, :, 0])
            # Directly perform:
            # sum_k,n numerator[k,n] * smear[k,n]
            local_response[ie] = np.dot(numerator_flat, smear.ravel())
            smear = None
        numerator_flat = None
        numerator = None
        # --------------------------------------------------------
        # Sum contributions from all MPI ranks.
        # Each rank contains only its local k-points.
        # reduce_full gives the sum over ALL k-points on rank 0.
        # --------------------------------------------------------
        aux_full = reduce_full(local_response, sroot=0)
        local_response = None
        if rank == 0:
            aux_full /= attr['nkpnts']
            if prop == 'shc':
                aux_full *= cgs_conv
            if prop == 'ree':
                aux_full *= bohr_to_cm
            aux_full /= (-2.0 * gamma) + deltab
        xzy = ['x', 'y', 'z']
        if prop == 'shc':
            spol, jpol, ipol = (tensor[0], tensor[1], tensor[2])
            fname = f'SHC_ODD_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Chi [(hbar/e)(S/cm)]'
        if prop == 'ree':
            spol, ipol = (tensor[0], tensor[1])
            fname = f'REE_EVEN_{xzy[spol]}{xzy[ipol]}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Chi [hbar*(cm/V)]'
        # data_controller.write_file_row_col(fname,ene,aux_full)
        data_controller.write_file_row_col_units(fname, ene, aux_full, unit1, unit2)
        aux_full = None
    # ============================================================
    # CONDUCTIVITY
    # ============================================================
    if prop == 'cond':
        for ispin in range(nspin):
            # ----------------------------------------------------
            # Extract diagonal matrix elements for this spin.
            # ----------------------------------------------------
            oper_matrix1_spin = np.diagonal(oper_matrix1[:, :, :, ispin], axis1=1, axis2=2)
            oper_matrix2_spin = np.diagonal(oper_matrix2[:, :, :, ispin], axis1=1, axis2=2)
            # ----------------------------------------------------
            # Numerator
            # ----------------------------------------------------
            numerator = np.real(oper_matrix1_spin * oper_matrix2_spin)
            oper_matrix1_spin = None
            oper_matrix2_spin = None
            # ----------------------------------------------------
            # Direct energy accumulation.
            # No (nk, esize) aux_diag array.
            # ----------------------------------------------------
            local_response = np.zeros(len(ene), dtype=float)
            numerator_flat = numerator.ravel()
            for ie in range(len(ene)):
                smear = gaussian(arry['E_k'][:, :, ispin], ene[ie], arry['deltakp'][:, :, ispin])
                local_response[ie] = np.dot(numerator_flat, smear.ravel())
                smear = None
            numerator_flat = None
            numerator = None
            # ----------------------------------------------------
            # Sum over all MPI ranks / k-points.
            # ----------------------------------------------------
            aux_full = reduce_full(local_response, sroot=0)
            local_response = None
            # ----------------------------------------------------
            # Global normalization and conductivity factor.
            # ----------------------------------------------------
            if rank == 0:
                aux_full /= attr['nkpnts']
                aux_full *= cgs_conv
                # no minus sign for conductivity
                aux_full /= (2.0 * gamma) + deltab
            # ----------------------------------------------------
            # Output
            # ----------------------------------------------------
            cpol, ipol = (tensor[0], tensor[1])
            xzy = ['x', 'y', 'z']
            if nspin == 1:
                fname = f'Cond_{xzy[cpol]}{xzy[ipol]}.dat'
            else:
                fname = f'Cond_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Conductivity [S/cm]'

            # data_controller.write_file_row_col(fname,ene,aux_full)
            data_controller.write_file_row_col_units(fname, ene, aux_full, unit1, unit2)
            aux_full = None
        oper_matrix1 = None
        oper_matrix2 = None


def fermi_sea(data_controller=None, tensor=None, prop=None, oper_matrix1=None, oper_matrix2=None):
    """EQUATION (5)"""
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    deltab = 0.001
    gamma = attr['gamma']
    attr['emaxH'] = np.amin(np.array([attr['shift'], attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'], attr['esize'])
    # ------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------
    if attr['twoD']:
        av0 = arry['a_vectors'][0, :]
        av1 = arry['a_vectors'][1, :]
        cgs_conv = 1.0 / (np.linalg.norm(np.cross(av0, av1)) * attr['alat'] ** 2)
    else:
        cgs_conv = 1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * attr['omega'])
    # ============================================================
    # SHC / REE
    # ============================================================
    if prop == 'shc' or prop == 'ree':
        oper_matrix1 = oper_matrix1[:, :, :, 0]
        oper_matrix2 = oper_matrix2[:, :, :, 0]
        aux_odd = np.zeros((nk, nbnd), dtype=float)
        for ik in range(nk):
            E = arry['E_k'][ik, :, 0]
            # NO SQUARE HERE
            E_nm = E - E[:, None]
            aux = oper_matrix1[ik, :, :] * oper_matrix2[ik, :, :].T
            np.fill_diagonal(aux, 0.0)
            aux = 2.0 * np.real(aux) * gamma * E_nm
            aux /= (E_nm**2 + gamma**2) ** 2 + deltab**2
            aux_odd[ik, :] = np.sum(aux, axis=1)
            aux = None
        oper_matrix1 = None
        oper_matrix2 = None

        # --------------------------------------------------------
        # IMPORTANT OPTIMIZATION:
        # Don't need to store the response for every k-point.
        # Only need the total contribution for each energy.
        # local_response: (esize,)
        # --------------------------------------------------------
        local_response = np.zeros(len(ene), dtype=float)
        # --------------------------------------------------------
        # Calculate energy-dependent Fermi-surface smearing.
        # --------------------------------------------------------
        for ie in range(len(ene)):
            smear = intgaussian(arry['E_k'][:, :, 0], ene[ie], arry['deltakp'][:, :, 0])
            local_response[ie] = np.sum(aux_odd * smear)
            smear = None
        aux_odd = None

        # --------------------------------------------------------
        # MPI reduction.
        # Each rank has:
        # local_response = sum over its local k-points
        # reduce_full() gives rank 0: sum over ALL k-points
        # --------------------------------------------------------
        shc_aux_full = reduce_full(local_response, sroot=0)
        local_response = None
        # --------------------------------------------------------
        # Global k-point normalization and units.
        # --------------------------------------------------------
        if rank == 0:
            shc_aux_full /= attr['nkpnts']
            if prop == 'shc':
                shc_aux_full *= cgs_conv
            if prop == 'ree':
                shc_aux_full *= bohr_to_cm
        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------
        xzy = ['x', 'y', 'z']
        if prop == 'shc':
            spol, jpol, ipol = (tensor[0], tensor[1], tensor[2])
            fname = f'SHC_ODD_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Chi [(hbar/e)(S/cm)]'
        if prop == 'ree':
            spol, ipol = (tensor[0], tensor[1])
            fname = f'REE_EVEN_{xzy[spol]}{xzy[ipol]}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Chi [hbar*(cm/V)]'
        # data_controller.write_file_row_col(fname,ene,shc_aux_full)
        data_controller.write_file_row_col_units(fname, ene, shc_aux_full, unit1, unit2)
        shc_aux_full = None

    # ============================================================
    # CONDUCTIVITY
    # ============================================================
    if prop == 'cond':
        nk, nbnd, nspin = arry['E_k'].shape
        for ispin in range(nspin):
            # oper_matrix1 = oper_matrix1[:, :, :, ispin]
            # oper_matrix2 = oper_matrix2[:, :, :, ispin]
            # ----------------------------------------------------
            # Energy-independent part.
            # Shape:(local_nk, nbnd)
            # ----------------------------------------------------
            aux_odd = np.zeros((nk, nbnd), dtype=float)
            for ik in range(nk):
                E = arry['E_k'][ik, :, ispin]
                # NO SQUARE HERE
                E_nm = E - E[:, None]
                aux = oper_matrix1[ik, :, :, ispin] * oper_matrix2[ik, :, :, ispin].T
                np.fill_diagonal(aux, 0.0)
                # Minus sign for conductivity
                aux = -2.0 * np.real(aux) * gamma * E_nm
                aux /= (E_nm**2 + gamma**2) ** 2 + deltab**2
                aux_odd[ik, :] = np.sum(aux, axis=1)
                aux = None
            # oper_matrix1 = None
            # oper_matrix2 = None
            # ----------------------------------------------------
            # No (nk, esize) array.
            # ----------------------------------------------------
            local_response = np.zeros(len(ene), dtype=float)
            for ie in range(len(ene)):
                smear = intgaussian(arry['E_k'][:, :, ispin], ene[ie], arry['deltakp'][:, :, ispin])
                local_response[ie] = np.sum(aux_odd * smear)
                smear = None
            aux_odd = None
            # ----------------------------------------------------
            # MPI reduction.
            # ----------------------------------------------------
            shc_aux_full = reduce_full(local_response, sroot=0)
            local_response = None
            # ----------------------------------------------------
            # Global normalization and units.
            # ----------------------------------------------------
            if rank == 0:
                shc_aux_full /= attr['nkpnts']
                shc_aux_full *= cgs_conv
            # ----------------------------------------------------
            # Output
            # ----------------------------------------------------
            cpol, ipol = (tensor[0], tensor[1])
            xzy = ['x', 'y', 'z']
            if nspin == 1:
                fname = f'Cond_{xzy[cpol]}{xzy[ipol]}.dat'
            else:
                fname = f'Cond_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat'
            unit1 = 'Energy [eV]'
            unit2 = 'Conductivity [S/cm]'
            # data_controller.write_file_row_col(fname,ene,shc_aux_full)
            data_controller.write_file_row_col_units(fname, ene, shc_aux_full, unit1, unit2)
            shc_aux_full = None
        oper_matrix1 = None
        oper_matrix2 = None


def do_spin_current(data_controller=None, tensor=None):
    spol, jpol, ipol = tensor[0], tensor[1], tensor[2]
    arry, attr = data_controller.data_dicts()
    Sj = arry['Sj'][spol]
    snktot, _, nawf, nawf, nspin = arry['dHksp'].shape
    jdHksp = np.empty((snktot, nawf, nawf, nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik, :, :, ispin] = 0.5 * (
                np.dot(Sj, arry['dHksp'][ik, jpol, :, :, ispin])
                + np.dot(arry['dHksp'][ik, jpol, :, :, ispin], Sj)
            )
    return jdHksp
