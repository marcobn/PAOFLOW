import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import sys
import time


from ..utils.constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI, LL
from ..utils.perturb_split import perturb_split
from ..utils.communication import gather_full
from ..utils.smearing import intgaussian, intmetpax

# bohr_to_m = 5.29177249e-11
bohr_to_cm = 5.29177249e-9

def linear_response_eqn1(data_controller):  
    arry,attr = data_controller.data_dicts()    
    if attr['prop'] == 'ree' or attr['prop'] == 'shc':
        if attr['dftSO'] == False:
            if rank == 0:
                print('Relativistic calculation with SO required')
                comm.Abort()
            comm.Barrier()   
    if attr['twoD']:
        av0,av1 = arry['a_vectors'][0,:],arry['a_vectors'][1,:]
        attr['cgs_conv'] = 1./(np.linalg.norm(np.cross(av0,av1))*attr['alat']**2)
    else: 
        attr['cgs_conv'] = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega']) ##only for Bulk
    

    if attr['prop'] == 'shc':
        my_tensor = arry['s_tensor']
    if attr['prop'] == 'ree' or attr['prop'] == 'cond':
        my_tensor = arry['ree_tensor']  
    for tensor in my_tensor:
        if rank == 0:
            start_time = time.time()
        calc_chi1(data_controller = data_controller,tensor=tensor)
        if rank == 0:
            end_time = time.time()
            total_time = (end_time-start_time)/60
            if attr['prop'] == 'shc':
                print(f'ODD SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'ree':
                print(f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'cond':
                print(f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
                
  
def calc_chi1(data_controller = None, tensor = None):
    arry,attr = data_controller.data_dicts()
    nk,nbnd,nspin = arry['E_k'].shape
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
    
    '''oper_matrix1 and oper_matrix2 are two matrices in the numerator'''
    oper_matrix1 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
    oper_matrix2 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
    if attr['prop'] == 'shc':
        '''spin current operator and corresponding matrix'''
        spol,jpol,ipol = tensor
        jksp_op = spin_current(data_controller = data_controller,tensor= tensor)
        for ispin in range(nspin):
            for ik in range(nk):
                oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(jksp_op[ik,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik]) 
        jksp_op = None   
                  
                  
    if attr['prop'] == 'ree':
        '''compute spin expectation value'''
        spol,ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(arry['Sj'][spol,:,:], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])           
    if attr['prop'] == 'cond':
        '''compute charge conductivity tensor'''
        cpol,ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(arry['dHksp'][ik,cpol,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik]) 
                
    
    prop_aux = np.zeros((nk,len(ene),nbnd, nspin), dtype=float)
    for ispin in range(nspin):
        for ie in range(len(ene)):
            if rank == 0:
                print(f'computing for {ie}/{len(ene)}: {ene[ie]}')
            for ik in range(nk):
                for n in range(nbnd):
                    for m in range(nbnd):
                        numerator = np.real(oper_matrix1[ik,n,m,ispin]*oper_matrix2[ik,m,n,ispin]*attr['gamma']**2)
                        denominator = ((ene[ie]-arry['E_k'][ik,n,ispin])**2 +attr['gamma']**2)*((ene[ie]-arry['E_k'][ik,m,ispin])**2 +attr['gamma']**2)
                        '''update prop_aux by summing index m for given band n, k, and ene index'''
                        prop_aux[ik,ie,n,ispin] += numerator/denominator
                        
    '''apply smearing function over bands now'''
    # smear = intgaussian(arry['E_k'][:,:,0], ene[ie], arry['deltakp'][:,:,0])
    # shc_aux[:,ie] = np.sum(aux*smear, axis=1)
    '''Marco- comment above two lines and uncomment one line below to not apply smearing'''
    '''no smearing added- only added over bands'''
    prop_aux = np.sum(prop_aux, axis=2) ##shape: (nk,len(ene) nspin)

                
    '''gather, fix unit and save'''
    if attr['prop'] != 'cond':
        prop_aux = prop_aux[:,:,0]
    prop_aux_full = gather_full(prop_aux, attr['npool'])  
    if rank == 0:
        if attr['prop'] == 'shc':
            prop_aux_full = np.sum(prop_aux_full, axis=0)*(1/prop_aux_full.shape[0]) *(attr['cgs_conv']/np.pi)*(-1)
        if attr['prop'] == 'ree':
            prop_aux_full = np.sum(prop_aux_full, axis=0)*(1/prop_aux_full.shape[0]) *(bohr_to_cm/np.pi)*(-1)
        if attr['prop'] == 'cond':
            prop_aux_full = np.sum(prop_aux_full, axis=0)*(1/prop_aux_full.shape[0]) *(attr['cgs_conv']/np.pi)
                    
    xzy = ['x','y','z'] 
    if attr['prop'] == 'shc':
        fname = f'SHC_eqn1_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
        data_controller.write_file_row_col(fname, ene, prop_aux_full)
        prop_aux_full = None
        prop_aux = None
    if attr['prop'] == 'ree':
        fname = f'REE_eqn1_{xzy[spol]}{xzy[ipol]}.dat' 
        data_controller.write_file_row_col(fname, ene, prop_aux_full)
        prop_aux_full = None
        prop_aux = None 
    if attr['prop'] == 'cond':
        for ispin in range(nspin):
            fname = f'Cond_eqn1_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat' 
            data_controller.write_file_row_col(fname, ene, prop_aux_full)
            prop_aux_full = None
            prop_aux = None
        
        
def spin_current(data_controller = None,tensor= None):
    arry,attr = data_controller.data_dicts()
    nk,nbnd,nspin = arry['E_k'].shape
    spol,jspol,ipol = tensor
    Sj = arry['Sj'][spol]
    snktot,_,nawf,nawf,nspin = arry['dHksp'].shape
    jdHksp = np.empty((snktot,nawf,nawf,nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik,:,:,ispin] = 0.5*(np.dot(Sj,arry['dHksp'][ik,jspol,:,:,ispin])+np.dot(arry['dHksp'][ik,jspol,:,:,ispin],Sj))
    return jdHksp
                


