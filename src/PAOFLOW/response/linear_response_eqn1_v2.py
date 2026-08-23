import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import sys
import time

from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI, LL
from ..utils.perturb_split import perturb_split
from ..utils.smearing import intgaussian, intmetpax


# bohr_to_m = 5.29177249e-11
bohr_to_cm = 5.29177249e-9


def linear_response_eqn1(data_controller):
    arry, attr = data_controller.data_dicts()
    # ------------------------------------------------------------
    # Check relativistic calculation requirement
    # ------------------------------------------------------------
    if attr['prop'] == 'ree' or attr['prop'] == 'shc':
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
        attr['cgs_conv'] = (1.0/ (np.linalg.norm(np.cross(av0, av1))* attr['alat']**2))
    else:
        attr['cgs_conv'] = (1.0e8* ANGSTROM_AU* ELECTRONVOLT_SI**2/ (H_OVER_TPI * attr['omega']))
        
    # ------------------------------------------------------------
    # Select tensor(s)
    # ------------------------------------------------------------
    if attr['prop'] == 'shc':
        my_tensor = arry['s_tensor']
    elif attr['prop'] == 'ree' or attr['prop'] == 'cond':
        my_tensor = arry['ree_tensor']

    # ------------------------------------------------------------
    # Calculate each tensor
    # ------------------------------------------------------------
    for tensor in my_tensor:
        if rank == 0:
            start_time = time.time()
        calc_chi1(data_controller=data_controller,tensor=tensor)
        if rank == 0:
            end_time = time.time()
            total_time = (end_time - start_time) / 60.0
            if attr['prop'] == 'shc':
                print(f'ODD SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'ree':
                print(
                    f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'cond':
                print(
                    f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')


def calc_chi1(data_controller=None, tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    # ------------------------------------------------------------
    # Energy grid
    # ------------------------------------------------------------
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'],attr['emaxH'],attr['esize'])
    nene = len(ene)
    gamma = attr['gamma']
    # ------------------------------------------------------------
    # use perturb_split() function to compute two matrices in numerator
    # ------------------------------------------------------------
    oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin),dtype=complex)
    oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin),dtype=complex)
    
    # ============================================================
    # CONSTRUCT OPERATORS
    # ============================================================
    
    # ------------------------------------------------------------
    # Spin Hall conductivity
    # ------------------------------------------------------------
    if attr['prop'] == 'shc':
        spol, jpol, ipol = tensor
        jksp_op = spin_current(data_controller=data_controller,tensor=tensor)
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik, :, :, ispin],oper_matrix2[ik, :, :, ispin]) = perturb_split(jksp_op[ik, :, :, ispin],arry['dHksp'][ik, ipol, :, :, ispin],arry['v_k'][ik, :, :, ispin],arry['degen'][ispin][ik])
        jksp_op = None
        
    # ------------------------------------------------------------
    # Rashba-Edelstein effect
    # ------------------------------------------------------------
    if attr['prop'] == 'ree':
        spol, ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik, :, :, ispin],oper_matrix2[ik, :, :, ispin]) = perturb_split(arry['Sj'][spol, :, :],arry['dHksp'][ik, ipol, :, :, ispin],arry['v_k'][ik, :, :, ispin],arry['degen'][ispin][ik])

    # ------------------------------------------------------------
    # Charge conductivity
    # ------------------------------------------------------------
    if attr['prop'] == 'cond':
        cpol, ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik, :, :, ispin],oper_matrix2[ik, :, :, ispin]) = perturb_split(arry['dHksp'][ik, cpol, :, :, ispin],arry['dHksp'][ik, ipol, :, :, ispin],arry['v_k'][ik, :, :, ispin],arry['degen'][ispin][ik])

    # ============================================================
    # RESPONSE CALCULATION
    # ============================================================
    # Original expression:
    #   sum_nm Re[O1_nm O2_mn] *gamma^2 / [((E-En)^2 + gamma^2)((E-Em)^2 + gamma^2)]
    # Define:
    #   W_n(E) = gamma /((E-En)^2 + gamma^2)
    #
    # Then:gamma^2 / (D_n D_m)= W_n(E) W_m(E)
    # Therefore: chi(E) = sum_nm A_nm W_n(E) W_m(E)
    # where A_nm = Re[O1_nm O2_mn]
    #
    # This allows us to completely eliminate the enormous
    # denominator array.
    #
    # ============================================================

    # ------------------------------------------------------------
    # Local accumulated response
    # IMPORTANT:
    # We accumulate directly over local k-points.
    # Therefore memory is: (nene × nspin) rather than: (nk × nene × nspin)
    # ------------------------------------------------------------
    
    prop_aux = np.zeros((nene, nspin), dtype=float)
    # ------------------------------------------------------------
    # Loop over spin
    # ------------------------------------------------------------
    for ispin in range(nspin):
        for ik in range(nk):
            E_k = arry['E_k'][ik, :, ispin]
            # ----------------------------------------------------
            # Construct Lorentzian weight matrix W[e,n] = gamma / ((E[e] - E_k[n])^2 + gamma^2)
            # shape = (nene, nbnd), this is very small compared with the old (nk × nene × nbnd × nbnd) denominator.
            # ----------------------------------------------------
            W = gamma / ((ene[:, None]- E_k[None, :])**2+ gamma**2)

            # ----------------------------------------------------
            # Construct numerator matrix:  shape = (nbnd, nbnd)
            # ----------------------------------------------------
            A = np.real(oper_matrix1[ik, :, :, ispin]* oper_matrix2[ik, :, :, ispin].T)

            # ----------------------------------------------------
            # Calculate:
            # response[e] = sum_nm W[e,n] A[n,m] W[e,m]
            #
            # First: Y = W @ A gives: Y[e,m] = sum_n W[e,n] A[n,m]
            # Then: sum_m Y[e,m] W[e,m] gives the desired quadratic form.
            # ----------------------------------------------------
            
            Y = W @ A
            response = np.real(np.sum(Y * W,axis=1))

            # ----------------------------------------------------
            # Accumulate this k-point
            # ----------------------------------------------------
            prop_aux[:, ispin] += response
            E_k = None
            W = None
            A = None
            Y = None
            response = None

    # ============================================================
    # MPI REDUCTION
    # prop_aux(rank) = sum over k-points belonging to that rank
    # and sum these arrays on rank 0.
    # ============================================================

    if rank == 0:
        prop_aux_full = np.zeros((nene, nspin),dtype=float)
    else:
        prop_aux_full = None
    comm.Reduce(
        prop_aux,
        prop_aux_full,
        op=MPI.SUM,
        root=0
    )

    # ------------------------------------------------------------
    # Global number of k-points
    # ------------------------------------------------------------
    nk_global = comm.allreduce(nk,op=MPI.SUM)

    # ============================================================
    # Average over k-points and convert units
    # ============================================================

    if rank == 0:
        # --------------------------------------------------------
        # SHC
        # --------------------------------------------------------
        if attr['prop'] == 'shc':
            prop_aux_full = (prop_aux_full[:, 0]/ nk_global* (attr['cgs_conv'] / np.pi)* (-1.0))

        # --------------------------------------------------------
        # REE
        # --------------------------------------------------------
        elif attr['prop'] == 'ree':
            prop_aux_full = (prop_aux_full[:, 0]/ nk_global* (bohr_to_cm / np.pi)* (-1.0))

        # --------------------------------------------------------
        # Charge conductivity
        #writes one file for each spin channel.
        # --------------------------------------------------------
        
        elif attr['prop'] == 'cond':
            prop_aux_full = (prop_aux_full/ nk_global* (attr['cgs_conv'] / np.pi))

    # ============================================================
    # Write output
    # ============================================================
    xzy = ['x', 'y', 'z']
    # ------------------------------------------------------------
    # SHC
    # ------------------------------------------------------------
    if attr['prop'] == 'shc':
        fname = (f'SHC_eqn1_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat')
        if rank == 0:
            data_controller.write_file_row_col(fname,ene,prop_aux_full)

    # ------------------------------------------------------------
    # REE
    # ------------------------------------------------------------
    elif attr['prop'] == 'ree':
        fname = (f'REE_eqn1_{xzy[spol]}{xzy[ipol]}.dat')
        if rank == 0:
            data_controller.write_file_row_col(fname,ene,prop_aux_full)
            
    # ------------------------------------------------------------
    # Conductivity
    # ------------------------------------------------------------
    elif attr['prop'] == 'cond':
        if rank == 0:
            for ispin in range(nspin):

                fname = (f'Cond_eqn1_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat')
                data_controller.write_file_row_col(fname,ene,prop_aux_full[:, ispin])
                
    oper_matrix1 = None
    oper_matrix2 = None
    prop_aux = None
    prop_aux_full = None


def spin_current(data_controller=None, tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    spol, jspol, ipol = tensor
    Sj = arry['Sj'][spol]
    snktot, _, nawf, nawf, nspin = arry['dHksp'].shape
    jdHksp = np.empty((snktot, nawf, nawf, nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik, :, :, ispin] = 0.5 * (np.dot(Sj,arry['dHksp'][ik,jspol,:, :,ispin])+np.dot(arry['dHksp'][ik,jspol,:, :,ispin],Sj))
    return jdHksp