#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import cmath
import numpy as np

def do_spin_orbit_H ( data_controller ):

  # construct TB spin orbit Hamiltonian (following Abate and Asdente, Phys. Rev. 140, A1303 (1965))

  arry,attr = data_controller.data_dicts()

  natoms = attr['natoms']
  theta,phi = attr['theta'],attr['phi']
  norb_array,orb_pseudo = arry['naw'],arry['orb_pseudo']

  nawf,_,nk1,nk2,nk3,nspin = arry['HRs'].shape

  socStrengh = np.zeros((natoms,2), dtype=float)
  socStrengh[:,0] = arry['lambda_p'][:]
  socStrengh[:,1] = arry['lambda_d'][:]

  HR_double = np.zeros((2*nawf,2*nawf,nk1,nk2,nk3,nspin), dtype=complex)

  # nonmagnetic :  copy H at the upper (lower) left (right) of the double matrix HR_double
  if nspin == 1:
    HR_double[:nawf,:nawf,:,:,:,0] = HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] = arry['HRs'][:nawf,:nawf,:,:,:,0]
  # magnetic :  copy H_up (H_down) at the upper (lower) left (right) of the double matrix 
  else:
    HR_double[:nawf,:nawf,:,:,:,0] = arry['HRs'][:nawf,:nawf,:,:,:,0]
    HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] = arry['HRs'][:nawf,:nawf,:,:,:,1]

  ##### Need clarification
  offset = np.zeros(natoms, dtype=int)
  for i in range(1, natoms):
    offset[i] = offset[i-1] + norb_array[i-1]

  for n,norb in enumerate(norb_array):
    HR_soc_p = np.zeros((2*norb,2*norb), dtype=complex)  
    HR_soc_d = np.zeros((2*norb,2*norb), dtype=complex)

    if (orb_pseudo[n] == 's'):
      pass
    if (orb_pseudo[n] == 'sp'):
      HR_soc_p[:,:] = soc_p_sp(theta, phi, norb)
    if (orb_pseudo[n] == 'spd'):
      HR_soc_p[:,:] = soc_p_spd(theta, phi, norb)
      HR_soc_d[:,:] = soc_d_spd(theta, phi, norb)
    if (orb_pseudo[n] == 'ps'):
      HR_soc_p[:,:] = soc_p_ps(theta, phi, norb)
    if (orb_pseudo[n] == 'sspd'):
      HR_soc_p[:,:] = soc_p_sspd(theta, phi, norb)
      HR_soc_d[:,:] = soc_d_sspd(theta, phi, norb)
    if (orb_pseudo[n] == 'ssp'):
      pass
    if (orb_pseudo[n] == 'ssppd'):
      HR_soc_p[:,:] = soc_p_ssppd(theta, phi, norb)
      HR_soc_d[:,:] = soc_d_ssppd(theta, phi, norb)

    uui = offset[n]
    uuj = uui + norb
    duj = udi = ddi = uui + nawf
    dui = udj = ddj = uuj + nawf

    # Up-Up
    HR_double[uui:uuj,uui:uuj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[0:norb,0:norb] + socStrengh[n,1]*HR_soc_d[0:norb,0:norb]
    # Down-Down
    HR_double[ddi:ddj,ddi:ddj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[norb:2*norb,norb:2*norb]  + socStrengh[n,1]*HR_soc_d[norb:2*norb,norb:2*norb]
    # Up-Down
    HR_double[uui:uuj,udi:udj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[0:norb,norb:2*norb] + socStrengh[n,1]*HR_soc_d[0:norb,norb:2*norb]
    # Down-Up
    HR_double[ddi:ddj,uui:uuj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[norb:2*norb,0:norb] + socStrengh[n,1]*HR_soc_d[norb:2*norb,0:norb]

    del arry['HRs']
    arry['HRs'] = HR_double
    attr['nawf'] = arry['HRs'].shape[0]

  arry['naw'] = np.append(arry['naw'],arry['naw'])

################### PSEUDOPOTENTIAL PS ##############################33
def soc_p_ps ( theta, phi, norb ):

  HR_soc = np.zeros((2*norb,2*norb), dtype=complex) 

  sTheta,sPhi = np.sin(theta),np.sin(phi)
  cTheta,cPhi = np.cos(theta),np.cos(phi)

  #Spin Up - Spin Up  part of the p-satets Hamiltonian
  HR_soc[0,1] = -0.5j * sTheta * sPhi
  HR_soc[0,2] =  0.5j * sTheta * cPhi
  HR_soc[1,2] = -0.5j * cTheta
  HR_soc[1,0] = np.conj(HR_soc[0,1])
  HR_soc[2,0] = np.conj(HR_soc[0,2])
  HR_soc[2,1] = np.conj(HR_soc[1,2])
  #Spin Down - Spin Down  part of the p-satets Hamiltonian
  HR_soc[5:8,5:8] = -HR_soc[1:4,1:4] 
  #Spin Up - Spin Down  part of the p-satets Hamiltonian
  HR_soc[0,5] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[0,6] = -0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[1,6] =  0.5j * sTheta
  HR_soc[1,4] = -HR_soc[0,5]
  HR_soc[2,4] = -HR_soc[0,6]
  HR_soc[2,5] = -HR_soc[1,6]
  #Spin Down - Spin Up  part of the p-satets Hamiltonian
  HR_soc[5,0] = np.conj(HR_soc[0,5])
  HR_soc[6,0] = np.conj(HR_soc[0,6])
  HR_soc[4,1] = np.conj(HR_soc[1,4])
  HR_soc[6,1] = np.conj(HR_soc[1,6])
  HR_soc[4,2] = np.conj(HR_soc[2,4])
  HR_soc[5,2] = np.conj(HR_soc[2,5])

  return HR_soc
################## END PSEUDOPOTENTIAL PS ##############################


################## PSEUDOPOTENTIAL SP ##############################
def soc_p_sp ( theta, phi, norb):

  HR_soc = np.zeros((2*norb,2*norb), dtype=complex) 

  sTheta,sPhi = np.sin(theta),np.sin(phi)
  cTheta,cPhi = np.cos(theta),np.cos(phi)

  # Spin Up - Spin Up  part of the p-satets Hamiltonian
  HR_soc[1,2] = -0.5j * sTheta * sPhi
  HR_soc[1,3] =  0.5j * sTheta * cPhi
  HR_soc[2,3] = -0.5j * cTheta
  HR_soc[2,1] = np.conj(HR_soc[1,2])
  HR_soc[3,1] = np.conj(HR_soc[1,3])
  HR_soc[3,2] = np.conj(HR_soc[2,3])
  # Spin Down - Spin Down  part of the p-satets Hamiltonian
  HR_soc[5:8,5:8] = -HR_soc[1:4,1:4] 
  # Spin Up - Spin Down  part of the p-satets Hamiltonian
  HR_soc[1,6] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[1,7] = -0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[2,7] =  0.5j * sTheta
  HR_soc[2,5] = -HR_soc[1,6]
  HR_soc[3,5] = -HR_soc[1,7]
  HR_soc[3,6] = -HR_soc[2,7]
  # Spin Down - Spin Up  part of the p-satets Hamiltonian
  HR_soc[6,1] = np.conj(HR_soc[1,6])
  HR_soc[7,1] = np.conj(HR_soc[1,7])
  HR_soc[5,2] = np.conj(HR_soc[2,5])
  HR_soc[7,2] = np.conj(HR_soc[2,7])
  HR_soc[5,3] = np.conj(HR_soc[3,5])
  HR_soc[6,3] = np.conj(HR_soc[3,6])

  return HR_soc
################## END PSEUDOPOTENTIAL SP ##############################


################## PSEUDOPOTENTIAL SPD ##############################
def soc_p_spd ( theta, phi, norb ):

  HR_soc = np.zeros((2*norb,2*norb), dtype=complex) 

  sTheta,sPhi = np.sin(theta),np.sin(phi)
  cTheta,cPhi = np.cos(theta),np.cos(phi)

  # Spin Up - Spin Up  part of the p-satets Hamiltonian
  HR_soc[1,2] = -0.5j * sTheta * sPhi
  HR_soc[1,3] =  0.5j * sTheta * cPhi
  HR_soc[2,3] = -0.5j * cTheta
  HR_soc[2,1] = np.conj(HR_soc[1,2])
  HR_soc[3,1] = np.conj(HR_soc[1,3])
  HR_soc[3,2] = np.conj(HR_soc[2,3])
  #Spin Down - Spin Down  part of the p-satets Hamiltonian
  HR_soc[10:13,10:13] = -HR_soc[1:4,1:4] 
  # Spin Up - Spin Down  part of the p-satets Hamiltonian
  HR_soc[1,11] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[1,12] = -0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[2,12] =  0.5j * sTheta
  HR_soc[2,10] = -HR_soc[1,11]
  HR_soc[3,10] = -HR_soc[1,12]
  HR_soc[3,11] = -HR_soc[2,12]
  # Spin Down - Spin Up  part of the p-satets Hamiltonian
  HR_soc[11,1] = np.conj(HR_soc[1,11])
  HR_soc[12,1] = np.conj(HR_soc[1,12])
  HR_soc[10,2] = np.conj(HR_soc[2,10])
  HR_soc[12,2] = np.conj(HR_soc[2,12])
  HR_soc[10,3] = np.conj(HR_soc[3,10])
  HR_soc[11,3] = np.conj(HR_soc[3,11])

  return HR_soc


def soc_d_spd(theta,phi,norb):

  HR_soc = np.zeros((2*norb,2*norb), dtype=complex) 

  sTheta,sPhi = np.sin(theta),np.sin(phi)
  cTheta,cPhi = np.cos(theta),np.cos(phi)

  s3 = np.sqrt(3.0)

  # Spin Up - Spin Up  part of the d-satets Hamiltonian
  HR_soc[4,5] = -s3 * 0.5j * sTheta * sPhi
  HR_soc[4,6] =  s3 * 0.5j * sTheta * cPhi
  HR_soc[5,6] = -0.5j * cTheta
  HR_soc[5,7] = -0.5j * sTheta * sPhi
  HR_soc[5,8] =  0.5j * sTheta * cPhi
  HR_soc[6,7] = -0.5j * sTheta * cPhi
  HR_soc[6,8] = -0.5j * sTheta * sPhi
  HR_soc[7,8] = -1.0j * cTheta
  HR_soc[5,4] = np.conj(HR_soc[4,5])
  HR_soc[6,4] = np.conj(HR_soc[4,6])
  HR_soc[6,5] = np.conj(HR_soc[5,6])
  HR_soc[7,5] = np.conj(HR_soc[5,7])
  HR_soc[8,5] = np.conj(HR_soc[5,8])
  HR_soc[7,6] = np.conj(HR_soc[6,7])
  HR_soc[8,6] = np.conj(HR_soc[6,8])

  # Spin Down - Spin Down  part of the p-satets Hamiltonian
  HR_soc[13:18,13:18] = -HR_soc[4:9,4:9] 
  # Spin Up - Spin Down  part of the p-satets Hamiltonian
  HR_soc[4,14] = -s3 * 0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[4,15] = -s3 * 0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[5,15] =  0.5j * sTheta
  HR_soc[5,16] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[5,17] = -0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[6,16] =  0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[6,17] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[7,17] =  1j * sTheta
  HR_soc[5,13] = -HR_soc[4,14] 
  HR_soc[6,13] = -HR_soc[4,15] 
  HR_soc[6,14] = -HR_soc[5,15]
  HR_soc[7,14] = -HR_soc[5,16] 
  HR_soc[8,14] = -HR_soc[5,17] 
  HR_soc[7,15] = -HR_soc[6,16] 
  HR_soc[8,15] = -HR_soc[6,17] 
  HR_soc[8,16] = -HR_soc[7,17] 
  # Spin Down - Spin Up  part of the p-satets Hamiltonian
  HR_soc[14,4] = np.conj(HR_soc[4,14]) 
  HR_soc[15,4] = np.conj(HR_soc[4,15])
  HR_soc[15,5] = np.conj(HR_soc[5,15])   
  HR_soc[16,5] = np.conj(HR_soc[5,16])   
  HR_soc[17,5] = np.conj(HR_soc[5,17])   
  HR_soc[16,6] = np.conj(HR_soc[6,16]) 
  HR_soc[17,6] = np.conj(HR_soc[6,17])    
  HR_soc[17,7] = np.conj(HR_soc[7,17])    
  HR_soc[13,5] = np.conj(HR_soc[5,13])
  HR_soc[13,6] = np.conj(HR_soc[6,13])
  HR_soc[14,6] = np.conj(HR_soc[6,14])
  HR_soc[14,7] = np.conj(HR_soc[7,14])
  HR_soc[14,8] = np.conj(HR_soc[8,14])
  HR_soc[15,7] = np.conj(HR_soc[7,15])
  HR_soc[15,8] = np.conj(HR_soc[8,15])

  return HR_soc
################## END PSEUDOPOTENTIAL SPD ##############################


################## PSEUDOPOTENTIAL SSPD ##############################
def soc_p_sspd ( theta, phi, norb ):

  # Hardcoded to s,p,d. This must change latter.
  HR_soc = np.zeros((2*norb,2*norb), dtype=complex) 

  sTheta,sPhi = np.sin(theta),np.sin(phi)
  cTheta,cPhi = np.cos(theta),np.cos(phi)

  # Spin Up - Spin Up  part of the p-satets Hamiltonian
  HR_soc[2,3] = -0.5j * sTheta*sPhi
  HR_soc[2,4] =  0.5j * sTheta*cPhi
  HR_soc[3,4] = -0.5j * cTheta
  HR_soc[3,2] = np.conj(HR_soc[2,3])
  HR_soc[4,2] = np.conj(HR_soc[2,4])
  HR_soc[4,3] = np.conj(HR_soc[3,4])
  # Spin Down - Spin Down  part of the p-satets Hamiltonian
  HR_soc[12:15,12:15] = -HR_soc[2:5,2:5] 
  # Spin Up - Spin Down  part of the p-satets Hamiltonian
  HR_soc[2,13] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[2,14] = -0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[3,14] =  0.5j * sTheta
  HR_soc[3,12] = -HR_soc[2,13]
  HR_soc[4,12] = -HR_soc[2,14]
  HR_soc[4,13] = -HR_soc[3,14]
  # Spin Down - Spin Up  part of the p-satets Hamiltonian
  HR_soc[13,2] = np.conj(HR_soc[2,13])
  HR_soc[14,2] = np.conj(HR_soc[2,14])
  HR_soc[12,3] = np.conj(HR_soc[3,12])
  HR_soc[14,3] = np.conj(HR_soc[3,14])
  HR_soc[12,4] = np.conj(HR_soc[4,12])
  HR_soc[13,4] = np.conj(HR_soc[4,13])

  return HR_soc


def soc_d_sspd ( theta, phi, norb ):

  # Hardcoded to s,p,d. This must change latter.
  HR_soc = np.zeros((2*norb,2*norb), dtype=complex)

  sTheta,sPhi = np.sin(theta),np.sin(phi)
  cTheta,cPhi = np.cos(theta),np.cos(phi)

  s3 = cmath.sqrt(3.0)

  #Spin Up - Spin Up  part of the d-satets Hamiltonian
  HR_soc[5,6] = -s3 * 0.5j * sTheta * sPhi
  HR_soc[5,7] =  s3 * 0.5j * sTheta * cPhi
  HR_soc[6,7] = -0.5j * cTheta
  HR_soc[6,8] = -0.5j * sTheta * sPhi
  HR_soc[6,9] =  0.5j * sTheta * cPhi
  HR_soc[7,8] = -0.5j * sTheta * cPhi
  HR_soc[7,9] = -0.5j * sTheta * sPhi
  HR_soc[8,9] = -1j * cTheta
  HR_soc[6,5] = np.conj(HR_soc[5,6])
  HR_soc[7,5] = np.conj(HR_soc[5,7])
  HR_soc[7,6] = np.conj(HR_soc[6,7])
  HR_soc[8,6] = np.conj(HR_soc[6,8])
  HR_soc[9,6] = np.conj(HR_soc[6,9])
  HR_soc[8,7] = np.conj(HR_soc[7,8])
  HR_soc[9,7] = np.conj(HR_soc[7,9])

  # Spin Down - Spin Down  part of the p-satets Hamiltonian
  HR_soc[15:20,15:20] = -HR_soc[5:10,5:10] 
  # Spin Up - Spin Down  part of the p-satets Hamiltonian
  HR_soc[5,16] = -s3 * 0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[5,17] = -s3 * 0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[6,17] =  0.5j * sTheta
  HR_soc[6,18] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[6,19] = -0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[7,18] =  0.5 * complex(sPhi, -cTheta*cPhi)
  HR_soc[7,19] = -0.5 * complex(cPhi, cTheta*sPhi)
  HR_soc[8,19] =  1j * sTheta
  HR_soc[6,15] = -HR_soc[5,16] 
  HR_soc[7,15] = -HR_soc[5,17] 
  HR_soc[7,16] = -HR_soc[6,17]
  HR_soc[8,16] = -HR_soc[6,18] 
  HR_soc[9,16] = -HR_soc[6,19] 
  HR_soc[8,17] = -HR_soc[7,18] 
  HR_soc[9,17] = -HR_soc[7,19] 
  HR_soc[9,18] = -HR_soc[8,19] 
  # Spin Down - Spin Up  part of the p-satets Hamiltonian
  HR_soc[16,5] = np.conj(HR_soc[5,16]) 
  HR_soc[17,5] = np.conj(HR_soc[5,17])
  HR_soc[17,6] = np.conj(HR_soc[6,17])   
  HR_soc[18,6] = np.conj(HR_soc[6,18])   
  HR_soc[19,6] = np.conj(HR_soc[6,19])   
  HR_soc[18,7] = np.conj(HR_soc[7,18]) 
  HR_soc[19,7] = np.conj(HR_soc[7,19])    
  HR_soc[19,8] = np.conj(HR_soc[8,19])    
  HR_soc[15,6] = np.conj(HR_soc[6,15])
  HR_soc[15,7] = np.conj(HR_soc[7,15])
  HR_soc[16,7] = np.conj(HR_soc[7,16])
  HR_soc[16,8] = np.conj(HR_soc[8,16])
  HR_soc[16,9] = np.conj(HR_soc[9,16])
  HR_soc[17,8] = np.conj(HR_soc[8,17])
  HR_soc[17,9] = np.conj(HR_soc[9,17])

  return HR_soc
 ################## END PSEUDOPOTENTIAL SSPD ##############################
