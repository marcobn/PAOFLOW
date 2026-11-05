import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import time
from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI
from ..utils.perturb_split import perturb_split
from ..utils.communication import reduce_full
from .kubo_bastin import BLOCK_ELEMENTS
from .kubo_kernels import surface_kernel
from .kubo_units import sheet_unit


# bohr_to_m = 5.29177249e-11
bohr_to_cm = 5.29177249e-9


def calc_chi(data_controller):
    arry, attr = data_controller.data_dicts()
    # ------------------------------------------------------------
    # Check relativistic calculation requirement
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
    # Select tensor(s)
    # ------------------------------------------------------------
    if attr['response'] == 'shc':
        my_tensor = arry['shc_tensor']
    elif attr['response'] == 'ree' or attr['response'] == 'cond':
        my_tensor = arry['ree_tensor']
    # ------------------------------------------------------------
    # Calculate each tensor
    # ------------------------------------------------------------
    for tensor in my_tensor:
        if rank == 0:
            start_time = time.time()
        calc_chi1(data_controller=data_controller, tensor=tensor)
        if rank == 0:
            end_time = time.time()
            total_time = (end_time - start_time) / 60.0
            if attr['response'] == 'shc':
                print(
                    f'ODD SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.'
                )
            if attr['response'] == 'ree':
                print(f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
            if attr['response'] == 'cond':
                print(f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')


def calc_chi1(data_controller=None, tensor=None):
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    # ------------------------------------------------------------
    # Energy grid
    # ------------------------------------------------------------
    ene = np.linspace(attr['eminH'], attr['emaxH'], attr['esize'])
    nene = len(ene)
    gamma = attr['gamma']
    # ============================================================
    # use perturb_split() to compute the two numerator matrices
    # ============================================================
    oper_matrix1 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
    oper_matrix2 = np.empty((nk, nbnd, nbnd, nspin), dtype=complex)
    # ============================================================
    # SHC
    # ============================================================
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

    # ============================================================
    # REE
    # ============================================================
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
    # ============================================================
    # CONDUCTIVITY
    # ============================================================
    if attr['response'] == 'cond':
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
    #
    # RESPONSE CALCULATION
    # sum_nm Re[O1_nm O2_mn] gamma^2 / [((E-En)^2 + gamma^2)((E-Em)^2 + gamma^2)]
    # Define: W_n(E) = gamma /((E-En)^2 + gamma^2)
    # Therefore: gamma^2/(D_n D_m) = W_n W_m
    # ============================================================

    # ------------------------------------------------------------
    # Local accumulated response
    # IMPORTANT:No nk dimension.
    # Each MPI rank accumulates its own local k-points.
    # ------------------------------------------------------------

    # sum_nm Re[O1_nm O2_mn] W_n W_m / pi with W = pi L: the Fermi-surface kernel pi sum Re P L L
    prop_aux = np.zeros((nene, nspin), dtype=float)
    block = max(1, BLOCK_ELEMENTS // (nene * nbnd + nbnd * nbnd))
    for ispin in range(nspin):
        for start in range(0, nk, block):
            ks = slice(start, min(start + block, nk))
            pair = oper_matrix1[ks, :, :, ispin] * np.transpose(
                oper_matrix2[ks, :, :, ispin], (0, 2, 1)
            )
            prop_aux[:, ispin] += surface_kernel(arry['E_k'][ks, :, ispin], pair, ene, gamma)[0]

    oper_matrix1 = None
    oper_matrix2 = None
    eqn1_finish(data_controller, prop_aux, ene, tensor)


def eqn1_response_k(op1, op2, E_k, ene, gamma):
    """Equation (1) contribution of one k-point and spin, on the energy grid.

    Parameters
    ----------
    op1, op2 : np.ndarray, shape ``(nbnd, nbnd)``, complex
        The two operators in the band basis (``perturb_split``).
    E_k : np.ndarray, shape ``(nbnd,)``
        Every eigenvalue at this k-point.
    ene : np.ndarray, shape ``(ne,)``
    gamma : float
        Lorentzian broadening.

    Returns
    -------
    np.ndarray, shape ``(ne,)``
        :math:`\\pi \\sum_{nm} L_n \\mathrm{Re}[O_{1,nm} O_{2,mn}] L_m`, the one-k
        :func:`~PAOFLOW.response.kubo_kernels.surface_kernel` of the dense loop.
    """
    return surface_kernel(E_k, op1 * op2.T, ene, gamma)[0]


def eqn1_finish(data_controller, prop_aux, ene, tensor):
    """Reduce, normalize and write the equation (1) response of one tensor component.

    Parameters
    ----------
    prop_aux : np.ndarray, shape ``(ne, nspin)``
        This rank's sum of :func:`eqn1_response_k` over its k-points.
    """
    arry, attr = data_controller.data_dicts()
    if attr['response'] == 'shc':
        spol, jpol, ipol = tensor
    else:
        cpol, ipol = tensor
        spol = cpol

    # ============================================================
    # MPI REDUCTION
    # IMPORTANT: This is a SUM, NOT a gather.
    # Every rank has:
    # prop_aux with shape (len(ene), nspin), and response is already sum over its LOCAL k-points
    # reduce_full() further adds prop_aux from all ranks and gives it to rank 0:
    # prop_aux_full = sum over ALL MPI ranks
    # ============================================================
    prop_aux = np.sum(prop_aux, axis=1)  ##sum over all spin channels
    prop_aux_full = reduce_full(prop_aux, sroot=0)
    prop_aux = None
    # ============================================================
    # Average over k-points and convert units
    # ============================================================
    if rank == 0:
        if attr['response'] == 'shc':
            prop_aux_full = prop_aux_full / attr['nkpnts'] * attr['cgs_conv'] * (-1.0)
        elif attr['response'] == 'ree':
            prop_aux_full = prop_aux_full / attr['nkpnts'] * bohr_to_cm * (-1.0)
        elif attr['response'] == 'cond':
            prop_aux_full = prop_aux_full / attr['nkpnts'] * attr['cgs_conv']

    # ============================================================
    # Write output
    # ============================================================
    xzy = ['x', 'y', 'z']
    if attr['response'] == 'shc':
        fname = f'SHC_ODD_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Chi [(hbar/e)(S/cm)]'
    elif attr['response'] == 'ree':
        fname = f'REE_EVEN_{xzy[spol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Chi [hbar*(cm/V)]'
    elif attr['response'] == 'cond':
        fname = f'Cond_{xzy[cpol]}{xzy[ipol]}.dat'
        unit1 = 'Energy [eV]'
        unit2 = 'Conductivity [S/cm]'
    if attr['twoD']:
        unit2 = sheet_unit(unit2)

    # data_controller.write_file_row_col(fname,ene,prop_aux_full)
    data_controller.write_file_row_col_units(fname, ene, prop_aux_full, unit1, unit2)
    # ============================================================
    # Free memory
    # ============================================================
    prop_aux_full = None


def spin_current(data_controller=None, tensor=None) -> np.ndarray:
    """Spin current {S_spin, dH_current}/2 of ``tensor = (spin, current, ...)`` at every local k."""
    from .do_Hall import do_spin_current as spin_current_operator

    return spin_current_operator(data_controller, tensor[0], tensor[1])


def linear_response_tensors(
    arry: dict,
    response: str,
    a_tensor: list | np.ndarray | None,
    s_tensor: list | np.ndarray | None,
) -> None:
    """Set ``ree_tensor`` and ``shc_tensor`` of ``PAOFLOW.linear_response`` (dense and sparse).

    Parameters
    ----------
    arry : dict
        DataController arrays; ``a_tensor``/``s_tensor`` of an earlier ``anomalous_Hall``/
        ``spin_Hall`` are the fallbacks.
    response : {'shc', 'ree', 'cond', 'ahc'}
    a_tensor, s_tensor : array_like or None
        The components given to ``linear_response``.

    Raises
    ------
    ValueError
        No tensor given for ``response`` and none stored by an earlier call.

    Notes
    -----
    Each tensor list falls back to the one stored by an earlier call only when it is not given
    here. Until 14 Nov 2026 ``a_tensor`` was read only when ``s_tensor`` was also given.
    """
    if a_tensor is not None:
        arry['ree_tensor'] = np.array(a_tensor)
    elif response != 'shc':
        if 'a_tensor' not in arry:
            raise ValueError(f"linear_response(response='{response}') needs a_tensor")
        arry['ree_tensor'] = arry['a_tensor']

    if s_tensor is not None:
        arry['shc_tensor'] = np.array(s_tensor)
    elif response == 'shc':
        if 's_tensor' not in arry:
            raise ValueError("linear_response(response='shc') needs s_tensor")
        # spin_Hall stores (current, field, spin); this method reads (spin, current, field)
        arry['shc_tensor'] = np.atleast_2d(arry['s_tensor'])[:, [2, 0, 1]]
