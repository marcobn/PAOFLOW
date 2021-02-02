#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
from .constants import ANGSTROM_AU

def doubling_HRs ( data_controller ):
    from scipy.fftpack import fftshift
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_rank() != 0:
       return

    arry,attr = data_controller.data_dicts()

    do_spin_orbit = attr['do_spin_orbit']    
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
                ix=int(round(Rx*nk1,0))
                iy=int(round(Ry*nk2,0))
                iz=int(round(Rz*nk3,0))
                
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
        arry['tau']   = np.append(arry['tau'],arry['tau'][:,:]+arry['a_vectors'][0,:]*ANGSTROM_AU,axis=0)
        arry['a_vectors'][0,:]=2*arry['a_vectors'][0,:]
        attr['omega'] = attr['alat']**3 * np.dot(arry['a_vectors'][0,:],np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))

        arry['b_vectors'][0,:] =  (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/attr['omega']*attr['alat']**3
        arry['b_vectors'][1,:] =  (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/attr['omega']*attr['alat']**3
        arry['b_vectors'][2,:] =  (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/attr['omega']*attr['alat']**3
        doubling_attr_arry(data_controller)



    # This construction is doubling along the X direction nx times    
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
                        HR_double[0:attr['nawf'],attr['nawf']:2*attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                
                    if ( ((2*iy-1) >= min(new_index[1,:])) and ((2*iy-1) <= (max(new_index[1,:])))):
                        i,j,k = cell_index[ix,iy,iz,:]
                        m,n,l = cell_index[ix,2*iy-1,iz,:]
                        #Lower left HR_double block                
                        HR_double[attr['nawf']:2*attr['nawf'],0:attr['nawf'],i,j,k,:] = arry['HRs'][:,:,m,n,l,:]
                

        arry['HRs'] = HR_double
        HR_double = None
        arry['tau']   = np.append(arry['tau'],arry['tau'][:,:]+arry['a_vectors'][1,:]*ANGSTROM_AU,axis=0)
        arry['a_vectors'][1,:]=2*arry['a_vectors'][1,:]
        attr['omega'] = attr['alat']**3 * np.dot(arry['a_vectors'][0,:],np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))
        arry['b_vectors'][0,:] =  (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/attr['omega']*attr['alat']**3
        arry['b_vectors'][1,:] =  (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/attr['omega']*attr['alat']**3
        arry['b_vectors'][2,:] =  (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/attr['omega']*attr['alat']**3
        doubling_attr_arry(data_controller)

    # This construction is doubling along the X direction nx times    
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
        arry['tau']   = np.append(arry['tau'],arry['tau'][:,:]+arry['a_vectors'][2,:]*ANGSTROM_AU,axis=0)
        arry['a_vectors'][2,:]=2*arry['a_vectors'][2,:]
        attr['omega'] = attr['alat']**3 * np.dot(arry['a_vectors'][0,:],np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))
        arry['b_vectors'][0,:] =  (np.cross(arry['a_vectors'][1,:],arry['a_vectors'][2,:]))/attr['omega']*attr['alat']**3
        arry['b_vectors'][1,:] =  (np.cross(arry['a_vectors'][2,:],arry['a_vectors'][0,:]))/attr['omega']*attr['alat']**3
        arry['b_vectors'][2,:] =  (np.cross(arry['a_vectors'][0,:],arry['a_vectors'][1,:]))/attr['omega']*attr['alat']**3
        doubling_attr_arry(data_controller)


def doubling_attr_arry ( data_controller ):

    arry,attr = data_controller.data_dicts()

    do_spin_orbit = attr['do_spin_orbit']    

    # Increassing nawf/natoms
    attr['nawf'] = 2*attr['nawf']
    attr['natoms'] = 2*attr['natoms']
    attr['nelec'] = 2*attr['nelec']
    attr['bnd'] = 2*attr['bnd']
    arry['species'] = np.append(arry['species'],arry['species'])
    # Doubling the atom number of orbitals / orbital character / multiplicity
    try:
      arry['naw']   = np.append(arry['naw'],arry['naw']) 
      arry['sh']   = np.append(arry['sh'],arry['sh']) 
      arry['nl']   = np.append(arry['nl'],arry['nl']) 
    except:
      pass
    # If the SOC is included pertubative
    if (attr['do_spin_orbit']):
        arry['lambda_p']   = np.append(arry['lambda_p'],arry['lambda_p'])
        arry['lambda_d']   = np.append(arry['lambda_d'],arry['lambda_d'])
        arry['orb_pseudo'] = np.append(arry['orb_pseudo'],arry['orb_pseudo'])
#
