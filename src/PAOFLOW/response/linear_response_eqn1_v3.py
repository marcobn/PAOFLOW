import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import sys
import time
from ..utils.constants import (ANGSTROM_AU,ELECTRONVOLT_SI,H_OVER_TPI,LL)
from ..utils.perturb_split import perturb_split
from ..utils.communication import reduce_full
from ..utils.smearing import (intgaussian,intmetpax)


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
        attr['cgs_conv'] = (1.0 /(np.linalg.norm(np.cross(av0, av1))*attr['alat']**2))
    else:
        attr['cgs_conv'] = (1.0e8* ANGSTROM_AU* ELECTRONVOLT_SI**2/(H_OVER_TPI* attr['omega']))
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
                print(f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'cond':
                print(f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')


def calc_chi1(data_controller=None,tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    # ------------------------------------------------------------
    # Energy grid
    # ------------------------------------------------------------
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'],attr['emaxH'],attr['esize'])
    nene = len(ene)
    gamma = attr['gamma']
    # ============================================================
    # use perturb_split() to compute the two numerator matrices
    # ============================================================
    oper_matrix1 = np.empty((nk,nbnd,nbnd,nspin),dtype=complex)
    oper_matrix2 = np.empty((nk,nbnd,nbnd,nspin),dtype=complex)
    # ============================================================
    # SHC
    # ============================================================
    if attr['prop'] == 'shc':
        spol, jpol, ipol = tensor
        jksp_op = spin_current(data_controller=data_controller,tensor=tensor)
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin]) = perturb_split(jksp_op[ik,:,:,ispin],arry['dHksp'][ik,ipol,:,:,ispin],arry['v_k'][ik,:,:,ispin],arry['degen'][ispin][ik])
        jksp_op = None

    # ============================================================
    # REE
    # ============================================================
    if attr['prop'] == 'ree':
        spol, ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin]) = perturb_split(arry['Sj'][spol,:,:],arry['dHksp'][ik,ipol,:,:,ispin],arry['v_k'][ik,:,:,ispin],arry['degen'][ispin][ik])
    # ============================================================
    # CONDUCTIVITY
    # ============================================================
    if attr['prop'] == 'cond':
        cpol, ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                (oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin]) = perturb_split(arry['dHksp'][ik,cpol,:,:,ispin],arry['dHksp'][ik,ipol,:,:,ispin],arry['v_k'][ik,:,:,ispin],arry['degen'][ispin][ik])
    # ============================================================
    #
    # RESPONSE CALCULATION
    # sum_nm Re[O1_nm O2_mn] gamma^2 / [((E-En)^2 + gamma^2)((E-Em)^2 + gamma^2)]
    # Define: W_n(E) = gamma /((E-En)^2 + gamma^2)
    # Therefore: gamma^2/(D_n D_m) = W_n W_m
    # ============================================================

    # ------------------------------------------------------------
    # Local accumulated response
    # IMPORTANT:
    # No nk dimension.
    # Each MPI rank accumulates its own local k-points.
    # ------------------------------------------------------------

    prop_aux = np.zeros((nene,nspin),dtype=float)
    for ispin in range(nspin):
        for ik in range(nk):
            E_k = arry['E_k'][ik,:,ispin]
            # Lorentzian weights W[e,n]
            W = gamma / ((ene[:, None]-E_k[None, :])**2+gamma**2) ##shape of W = (len(ene), nbnd)
            # ----------------------------------------------------
            # Numerator matrix: A[n,m]
            A = np.real(oper_matrix1[ik,:,:,ispin]*oper_matrix2[ik,:,:,ispin].T) ##shape of A = (nbnd, nbnd)
            # ----------------------------------------------------
            # Calculate: response[e] = sum_nm W[e,n] A[n,m] W[e,m]
            # First: Y = W @ A gives: Y[e,m] = sum_n W[e,n] A[n,m]; use dot product to sum over n
            # Then: sum_m Y[e,m] W[e,m] gives the desired quadratic form. elementwise multiplication
            # ----------------------------------------------------
            Y = W @ A ##shape of Y = (len(ene), nbnd)
            response = np.real(np.sum(Y * W,axis=1))
            # ----------------------------------------------------
            # Accumulate this local k-point
            # ----------------------------------------------------
            prop_aux[:,ispin] += response

    # ============================================================
    # MPI REDUCTION
    # IMPORTANT: This is a SUM, NOT a gather.
    # Every rank has:
        #prop_aux with shape (len(ene), nspin), and response is already sum over its LOCAL k-points
        #reduce_full() further adds prop_aux from all ranks and gives it to rank 0:
        #prop_aux_full = sum over ALL MPI ranks
    # ============================================================

    prop_aux_full = reduce_full(prop_aux,sroot=0)
    # ============================================================
    # Global number of k-points
    # ============================================================
    nk_global = comm.allreduce(nk,op=MPI.SUM)
    # ============================================================
    # Average over k-points and convert units
    # ============================================================
    if rank == 0:
        if attr['prop'] == 'shc':
            prop_aux_full = (prop_aux_full[:, 0]/ nk_global* (attr['cgs_conv']/ np.pi)* (-1.0))
        elif attr['prop'] == 'ree':
            prop_aux_full = (prop_aux_full[:, 0]/ nk_global* (bohr_to_cm/ np.pi)* (-1.0))
        elif attr['prop'] == 'cond':
            prop_aux_full = (prop_aux_full/ nk_global* (attr['cgs_conv']/ np.pi))
        

    # ============================================================
    # Write output
    # ============================================================

    xzy = ['x','y','z']
    # ------------------------------------------------------------
    # SHC
    # ------------------------------------------------------------
    if attr['prop'] == 'shc':
        fname = (f'SHC_eqn1_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat')
        data_controller.write_file_row_col(fname,ene,prop_aux_full)

    # ------------------------------------------------------------
    # REE
    # ------------------------------------------------------------

    elif attr['prop'] == 'ree':
        fname = (f'REE_eqn1_{xzy[spol]}{xzy[ipol]}.dat')
        data_controller.write_file_row_col(fname,ene,prop_aux_full)

    # ------------------------------------------------------------
    # Conductivity
    # ------------------------------------------------------------

    elif attr['prop'] == 'cond':
        for ispin in range(nspin):
            fname = (f'Cond_eqn1_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat')
            data_controller.write_file_row_col(fname,ene,prop_aux_full[:, ispin])

    # ============================================================
    # Free memory
    # ============================================================
    oper_matrix1 = None
    oper_matrix2 = None
    prop_aux = None
    prop_aux_full = None
    
def spin_current(data_controller=None,tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    spol, jspol, ipol = tensor
    Sj = arry['Sj'][spol]
    snktot, _, nawf, nawf, nspin = (arry['dHksp'].shape)
    jdHksp = np.empty((snktot,nawf,nawf,nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik,:,:,ispin] = 0.5 * (np.dot(Sj,arry['dHksp'][ik,jspol,:,:,ispin])+np.dot(arry['dHksp'][ik,jspol,:,:,ispin],Sj))
    return jdHksp