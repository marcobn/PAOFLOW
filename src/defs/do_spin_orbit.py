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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys, time

sys.path.append('./')

def do_spin_orbit_calc(HRaux,natoms,theta,phi,socStrengh):

    # construct TB spin orbit Hamiltonian (following Abate and Asdente, Phys. Rev. 140, A1303 (1965))

    nawf = HRaux.shape[0]
    nk1 = HRaux.shape[2]
    nk2 = HRaux.shape[3]
    nk3 = HRaux.shape[4]
    nspin = HRaux.shape[5]

    HR_double= np.zeros((2*nawf,2*nawf,nk1,nk2,nk3,nspin),dtype=complex)
    HR_soc_p = np.zeros((18,18),dtype=complex)  #Hardcoded do s,p,d only (18 orbitals per atom) - Must Change

    # nonmagnetic :  copy H at the upper (lower) left (right) of the double matrix HR_double
    if nspin == 1:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	      	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
    # magnetic :  copy H_up (H_down) at the upper (lower) left (right) of the double matrix 
    else:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	       	       =  HRaux[0:nawf,0:nawf,:,:,:,1]

    HR_soc_p =  soc_p(theta,phi)
    HR_soc_d =  soc_d(theta,phi)

    M=9
    nt=natoms
    for n in xrange(nt):
        i=n*M
        j=(n+1)*M
        # Up-Up
        HR_double[i:j,i:j,0,0,0,0]                             = HR_double[i:j,i:j,0,0,0,0] + socStrengh[n,0]*HR_soc_p[0:9,0:9] + socStrengh[n,1]*HR_soc_d[0:9,0:9]
        # Down-Down
        HR_double[(i+nt*M):(j+nt*M),(i+nt*M):(j+nt*M),0,0,0,0] = HR_double[(i+nt*M):(j+nt*M),(i+nt*M):(j+nt*M),0,0,0,0] + socStrengh[n,0]*HR_soc_p[9:18,9:18]  + socStrengh[n,1]*HR_soc_d[9:18,9:18]
        # Up-Down
        HR_double[i:j,(i+nt*M):(j+nt*M),0,0,0,0]               = HR_double[i:j,(i+nt*M):(j+nt*M),0,0,0,0] + socStrengh[n,0]*HR_soc_p[0:9,9:18] + socStrengh[n,1]*HR_soc_d[0:9,9:18]
        # Down-Up
        HR_double[(i+nt*M):(j+nt*M),i:j,0,0,0,0]               = HR_double[(i+nt*M):(j+nt*M),i:j,0,0,0,0] + socStrengh[n,0]*HR_soc_p[9:18,0:9] + socStrengh[n,1]*HR_soc_d[9:18,0:9]

    return(HR_double)


def soc_p(theta,phi):

    # Hardcoded to s,p,d. This must change latter.
        HR_soc = np.zeros((18,18),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

	#Spin Up - Spin Up  part of the p-satets Hamiltonian
        HR_soc[1,2] = -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[1,3] =  0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[2,3] = -0.5*np.complex(0.0,cTheta)
        HR_soc[2,1]=np.conjugate(HR_soc[1,2])
        HR_soc[3,1]=np.conjugate(HR_soc[1,3])
        HR_soc[3,2]=np.conjugate(HR_soc[2,3])
	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[10:13,10:13] = - HR_soc[1:4,1:4] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[1,11] = -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[1,12] = -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[2,12] =  0.5*np.complex(0.0,sTheta)
        HR_soc[2,10] = -HR_soc[1,11]
        HR_soc[3,10] = -HR_soc[1,12]
        HR_soc[3,11] = -HR_soc[2,12]
	#Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[11,1]=np.conjugate(HR_soc[1,11])
        HR_soc[12,1]=np.conjugate(HR_soc[1,12])
        HR_soc[10,2]=np.conjugate(HR_soc[2,10])
        HR_soc[12,2]=np.conjugate(HR_soc[2,12])
        HR_soc[10,3]=np.conjugate(HR_soc[3,10])
        HR_soc[11,3]=np.conjugate(HR_soc[3,11])
	return(HR_soc)
def soc_d(theta,phi):

    # Hardcoded to s,p,d. This must change latter.
        HR_soc = np.zeros((18,18),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

        s3 = cmath.sqrt(3.0)

	#Spin Up - Spin Up  part of the d-satets Hamiltonian
        HR_soc[4,5] = -s3*0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[4,6] =  s3*0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[5,6] =  -0.5*np.complex(0.0,cTheta)
        HR_soc[5,7] =  -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[5,8] =   0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[6,7] =  -0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[6,8] =  -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[7,8] =  -1.0*np.complex(0.0,cTheta)
        HR_soc[5,4] = np.conjugate(HR_soc[4,5])
        HR_soc[6,4] = np.conjugate(HR_soc[4,6])
        HR_soc[6,5] = np.conjugate(HR_soc[5,6])
        HR_soc[7,5] = np.conjugate(HR_soc[5,7])
        HR_soc[8,5] = np.conjugate(HR_soc[5,8])
        HR_soc[7,6] = np.conjugate(HR_soc[6,7])
        HR_soc[8,6] = np.conjugate(HR_soc[6,8])

	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[13:18,13:18] = - HR_soc[4:9,4:9] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[4,14] = -s3*0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[4,15] = -s3*0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[5,15] =     0.5*( np.complex(0.0,sTheta))
        HR_soc[5,16] =    -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[5,17] =    -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[6,16] =     0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[6,17] =    -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[7,17] =     1.0*( np.complex(0.0,sTheta))
        HR_soc[5,13] =  -HR_soc[4,14] 
        HR_soc[6,13] =  -HR_soc[4,15] 
        HR_soc[6,14] =  -HR_soc[5,15]
        HR_soc[7,14] =  -HR_soc[5,16] 
        HR_soc[8,14] =  -HR_soc[5,17] 
        HR_soc[7,15] =  -HR_soc[6,16] 
        HR_soc[8,15] =  -HR_soc[6,17] 
        HR_soc[8,16] =  -HR_soc[7,17] 
    #Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[14,4] = np.conjugate(HR_soc[4,14]) 
        HR_soc[15,4] = np.conjugate(HR_soc[4,15])
        HR_soc[15,5] = np.conjugate(HR_soc[5,15])   
        HR_soc[16,5] = np.conjugate(HR_soc[5,16])   
        HR_soc[17,5] = np.conjugate(HR_soc[5,17])   
        HR_soc[16,6] = np.conjugate(HR_soc[6,16]) 
        HR_soc[17,6] = np.conjugate(HR_soc[6,17])    
        HR_soc[17,7] = np.conjugate(HR_soc[7,17])    
        HR_soc[13,5] = np.conjugate(HR_soc[5,13])
        HR_soc[13,6] = np.conjugate(HR_soc[6,13])
        HR_soc[14,6] = np.conjugate(HR_soc[6,14])
        HR_soc[14,7] = np.conjugate(HR_soc[7,14])
        HR_soc[14,8] = np.conjugate(HR_soc[8,14])
        HR_soc[15,7] = np.conjugate(HR_soc[7,15])
        HR_soc[15,8] = np.conjugate(HR_soc[8,15])
	return(HR_soc)
