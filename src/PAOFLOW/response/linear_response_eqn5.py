import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import sys
import time
from ..utils.constants import (ANGSTROM_AU,ELECTRONVOLT_SI,H_OVER_TPI,LL)
from ..utils.perturb_split import perturb_split
from ..utils.communication import (gather_full, reduce_full)
from ..utils.smearing import (gaussian,intgaussian,intmetpax)

bohr_to_cm = 5.29177249e-9

def do_chi2_simple(data_controller):  
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
           
    # for properties in arry['prop']:
    if attr['prop'] == 'shc':
        my_tensor = arry['s_tensor']
    if attr['prop'] == 'ree' or attr['prop'] == 'cond':
        my_tensor = arry['ree_tensor']            
    for tensor in my_tensor:
        if rank == 0:
            start_time = time.time()
        calc_chi2(data_controller = data_controller,tensor=tensor)
        if rank == 0:
            end_time = time.time()
            total_time = (end_time-start_time)/60
            if attr['prop'] == 'shc':
                print(f'EVEN SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'ree':
                print(f'ODD REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
            if attr['prop'] == 'cond':
                print(f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
  
def calc_chi2(data_controller = None,tensor = None):
    arry,attr = data_controller.data_dicts()
    nk,nbnd,nspin = arry['E_k'].shape
    deltap = 0.05
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
    esize = attr['esize']
    
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
    Om_znkaux = np.zeros((nk,nbnd, nspin), dtype=float)
    for ispin in range (nspin):
        for ik in range(nk):
            E_nm = (arry['E_k'][ik,arry['selected_bands'],ispin] - arry['E_k'][ik,arry['selected_bands'],ispin][:,None])**2+deltap**2
            E_nm[np.where(E_nm<1.e-4)] = np.inf
            Om_znkaux[ik,:,ispin] = -2.0*np.sum(np.imag(oper_matrix1[ik,:,:,ispin]*oper_matrix2[ik,:,:,ispin].T)/E_nm, axis=1)     
    oper_matrix1=oper_matrix2=None
    Om_zkaux = np.zeros((nk,esize, nspin), dtype=float) 
    
    for ispin in range(nspin):
        for i in range(esize):
            if attr['smearing'] == 'gauss':
                Om_zkaux[:,i,ispin] = np.sum((Om_znkaux[:,:,ispin]*intgaussian(arry['E_k'][:,:,ispin],ene[i],arry['deltakp'][:,:,ispin])), axis=1)
            elif attr['smearing'] == 'm-p':
                Om_zkaux[:,i,ispin] = np.sum(Om_znkaux[:,:,ispin]*intmetpax(arry['E_k'][:,:,ispin],ene[i],arry['deltakp'][:,:,ispin]), axis=1)
            else:
                Om_zkaux[:,i,ispin] = np.sum(Om_znkaux[:,:,ispin]*(0.5 * (-np.sign(arry['E_k'][:,:,ispin]-ene[i]) + 1)), axis=1)
       
    if attr['prop'] != 'cond':
        Om_zkaux = Om_zkaux[:,:,0]
    Om_zk = gather_full(Om_zkaux, attr['npool'])
    Om_zkaux = None
    if rank == 0:
        xzy = ['x','y','z']
        if attr['prop'] == 'shc':
            Om_zk *= attr['cgs_conv']
            Om_zk = np.sum(Om_zk, axis=0)/float(attr['nkpnts'])     
            fname = f'SHC_eqn5_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
            data_controller.write_file_row_col(fname, ene, Om_zk)
            
        if attr['prop'] == 'ree':
            Om_zk *= bohr_to_cm
            Om_zk = np.sum(Om_zk, axis=0)/float(attr['nkpnts'])     
            fname = f'REE_eqn5_{xzy[spol]}{xzy[ipol]}.dat' 
            data_controller.write_file_row_col(fname, ene, Om_zk)
            
        if attr['prop'] == 'cond':
            Om_zk *= attr['cgs_conv']
            Om_zk = np.sum(Om_zk, axis=0)/float(attr['nkpnts'])     
            fname = f'AHE_eqn5_{xzy[cpol]}{xzy[ipol]}.dat' 
            data_controller.write_file_row_col(fname, ene, Om_zk)          
    ene=Om_zk=None
        
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
                


