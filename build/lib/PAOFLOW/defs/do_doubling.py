#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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
import scipy.linalg as la

def doubling_HRs ( data_controller ):
    from scipy.fftpack import fftshift
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_rank() != 0:
       return

    arry,attr = data_controller.data_dicts()

    nx = attr['nx']   
    ny = attr['ny']   
    nz = attr['nz']   
    nk1 = attr['nk1']   
    nk2 = attr['nk2']   
    nk3 = attr['nk3']   
    nspin = attr['nspin']   

    nkpts=nk1*nk2*nk3

    cell_index=np.zeros((nk1,nk2,nk3,3),dtype=int)
    new_index=np.zeros((3,nkpts),dtype=int)
    
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j*nk3 + i*nk2*nk3
                Rx = float(i)/float(nk1)
                Ry = float(j)/float(nk2)
                Rz = float(k)/float(nk3)
                if Rx >= 0.5: Rx=Rx-1.0
                if Ry >= 0.5: Ry=Ry-1.0
                if Rz >= 0.5: Rz=Rz-1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                # the minus sign in Rx*nk1 is due to the Fourier transformation (Ri-Rj)
                ix=int(-round(Rx*nk1,0))
                iy=int(-round(Ry*nk2,0))
                iz=int(-round(Rz*nk3,0))
                
                cell_index[ix,iy,iz,0]=i
                cell_index[ix,iy,iz,1]=j
                cell_index[ix,iy,iz,2]=k           

                new_index[0,n]=ix
                new_index[1,n]=iy
                new_index[2,n]=iz

    # This construction is doubling along the X direction nx times    
    for dx in range(nx):
        
        HR_double= np.zeros((2*attr['nawf'],2*attr['nawf'],nk1,nk2,nk3,nspin),dtype=complex)
        for ix in range(min(new_index[0,:]),max(new_index[0,:])+1):
            for iy in range(min(new_index[1,:]),max(new_index[1,:])+1):
                for iz in range(min(new_index[2,:]),max(new_index[2,:])+1):
                                  
                    i,j,k = cell_index[ix,iy,iz,:] # doubled cell index

                    if ( ((2*ix) >= min(new_index[0,:])) and ((2*ix) <= (max(new_index[0,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[2*ix,iy,iz,:]                
                        # Upper left HR_double block                             
                        HR_double[0:attr['nawf'],0:attr['nawf'],i,j,k,:]           = arry['HRs'][:,:,m,n,l,:]
                        # Lower right HR_double block                             
                        HR_double[attr['nawf']:2*attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*ix+1) >= min(new_index[0,:])) and ((2*ix+1) <= (max(new_index[0,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[2*ix+1,iy,iz,:]
                        #Upper right HR_double block                
                        HR_double[0:attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*ix-1) >= min(new_index[0,:])) and ((2*ix-1) <= (max(new_index[0,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[2*ix-1,iy,iz,:]
                        #Lower left HR_double block
                        HR_double[attr['nawf']:2*attr['nawf'],0:attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]

        arry['HRs'] = HR_double
        HR_double = None
        arry['tau']   = np.append(arry['tau'],arry['tau'][:,:]+arry['a_vectors'][0,:]*attr['alat'],axis=0)
        arry['a_vectors'][0,:]=2*arry['a_vectors'][0,:]
        doubling_attr_arry(data_controller)



    # This construction is doubling along the Y direction ny times    
    for dy in range(ny):
        HR_double= np.zeros((2*attr['nawf'],2*attr['nawf'],nk1,nk2,nk3,nspin),dtype=complex)

        for ix in range(min(new_index[0,:]),max(new_index[0,:])+1):
            for iy in range(min(new_index[1,:]),max(new_index[1,:])+1):
                for iz in range(min(new_index[2,:]),max(new_index[2,:])+1):
                                  
                    i,j,k = cell_index[ix,iy,iz,:] # doubled cell index

                    if ( ((2*iy) >= min(new_index[1,:])) and ((2*iy) <= (max(new_index[1,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,2*iy,iz,:]                
                        # Upper left HR_double block                             
                        HR_double[0:attr['nawf'],0:attr['nawf'],i,j,k,:]           =  arry['HRs'][:,:,m,n,l,:]
                        # Lower right HR_double block                             
                        HR_double[attr['nawf']:2*attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] =  arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*iy+1) >= min(new_index[1,:])) and ((2*iy+1) <= (max(new_index[1,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,2*iy+1,iz,:]
                        #Upper right HR_double block                
                        HR_double[0:attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] =  arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*iy-1) >= min(new_index[1,:])) and ((2*iy-1) <= (max(new_index[1,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,2*iy-1,iz,:]
                        #Lower left HR_double block                
                        HR_double[attr['nawf']:2*attr['nawf'],0:attr['nawf'],i,j,k,:] =  arry['HRs'][:,:,m,n,l,:]
                

        arry['HRs'] = HR_double
        HR_double = None
        arry['tau']   = np.append(arry['tau'],arry['tau'][:,:]+arry['a_vectors'][1,:]*attr['alat'],axis=0)
        arry['a_vectors'][1,:]=2*arry['a_vectors'][1,:]
        doubling_attr_arry(data_controller)

    # This construction is doubling along the Z direction nz times    
    delete_index=0
    for dz in range(nz):

        HR_double= np.zeros((2*attr['nawf'],2*attr['nawf'],nk1,nk2,nk3,nspin),dtype=complex)
        
        for ix in range(min(new_index[0,:]),max(new_index[0,:])+1):
            for iy in range(min(new_index[1,:]),max(new_index[1,:])+1):
                for iz in range(min(new_index[2,:]),max(new_index[2,:])+1):
                                  
                    i,j,k = cell_index[ix,iy,iz,:] # doubled cell index

                    if ( ((2*iz) >= min(new_index[2,:])) and ((2*iz) <= (max(new_index[2,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,iy,2*iz,:]                
                        # Upper left HR_double block                             
                        HR_double[0:attr['nawf'],0:attr['nawf'],i,j,k,:]           = arry['HRs'][:,:,m,n,l,:]
                        # Lower right HR_double block                             
                        HR_double[attr['nawf']:2*attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*iz+1) >= min(new_index[2,:])) and ((2*iz+1) <= (max(new_index[2,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,iy,2*iz+1,:]
                        #Upper right HR_double block                
                        HR_double[0:attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*iz-1) >= min(new_index[2,:])) and ((2*iz-1) <= (max(new_index[2,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,iy,2*iz-1,:]
                        #Lower left HR_double block                
                        HR_double[attr['nawf']:2*attr['nawf'],0:attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
        
        arry['HRs'] = HR_double
        HR_double = None
        arry['tau']   = np.append(arry['tau'],arry['tau'][:,:]+arry['a_vectors'][2,:]*attr['alat'],axis=0)
        arry['a_vectors'][2,:]=2*arry['a_vectors'][2,:]
        doubling_attr_arry(data_controller)


def doubling_attr_arry ( data_controller ):

    arry,attr = data_controller.data_dicts()

    # Increassing nawf/natoms
    attr['nawf'] = 2*attr['nawf']
    if 'natoms' in attr: attr['natoms'] = 2*attr['natoms']
    if 'nelec' in attr: attr['nelec'] = 2*attr['nelec']
    if 'nbnds' in attr: attr['nbnds'] = 2*attr['nbnds']
    if 'bnd' in attr: attr['bnd'] = 2*attr['bnd']
    #doubling the atom number of orbitals / orbital character / multiplicity
    if 'naw' in arry: arry['naw']   = np.append(arry['naw'],arry['naw'])
    if 'sh' in arry: arry['sh']   = np.append(arry['sh'],arry['sh'])
    if 'nl' in arry: arry['nl']   = np.append(arry['nl'],arry['nl'])
    if 'atoms' in arry: arry['atoms']   = np.append(arry['atoms'],arry['atoms'])
    # if Sj is already computed, then double it
    if 'Sj' in arry:
        Sj_double = np.zeros((3,attr['nawf'],attr['nawf']),dtype=complex)
        for spol in range(3):
            Sj = arry['Sj'][spol]
            Sj_double[spol] = la.block_diag(*[Sj,Sj])

        arry['Sj'] = Sj_double
        Sj_double = None

    # If the SOC is included pertubative
    if 'do_spin_orbit' in attr and (attr['do_spin_orbit']):
        if 'lambda_p' in arry: arry['lambda_p']   = np.append(arry['lambda_p'],arry['lambda_p'])
        if 'lambda_d' in arry: arry['lambda_d']   = np.append(arry['lambda_d'],arry['lambda_d'])
        if 'orb_pseudo' in arry: arry['orb_pseudo'] = np.append(arry['orb_pseudo'],arry['orb_pseudo'])

