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

def do_spin_orbit_calc(HRaux,natoms,theta,phi,socStrengh,norb_array,orb_pseudo):

    # construct TB spin orbit Hamiltonian (following Abate and Asdente, Phys. Rev. 140, A1303 (1965))

    nawf  = HRaux.shape[0]
    nk1   = HRaux.shape[2]
    nk2   = HRaux.shape[3]
    nk3   = HRaux.shape[4]
    nspin = HRaux.shape[5]

    norb  = int(nawf/socStrengh.shape[0])
    HR_double = np.zeros((2*nawf,2*nawf,nk1,nk2,nk3,nspin),dtype=complex)

    # nonmagnetic :  copy H at the upper (lower) left (right) of the double matrix HR_double
    if nspin == 1:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	      	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
    # magnetic :  copy H_up (H_down) at the upper (lower) left (right) of the double matrix 
    else:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	       	       =  HRaux[0:nawf,0:nawf,:,:,:,1]

    natoms= int(socStrengh.shape[0])
    offset = np.zeros(natoms,dtype='int32')
    i=1
    for i in xrange(natoms):
        for j in xrange(i):
            offset[i]=offset[i-1] + norb_array[i-1]

    for n in xrange(natoms):
        HR_soc_p = np.zeros((2*norb_array[n],2*norb_array[n]),dtype=complex)  
    	HR_soc_d = np.zeros((2*norb_array[n],2*norb_array[n]),dtype=complex)
       
        norb=norb_array[n]

        if (orb_pseudo[n] == 's'):
            HR_soc_p[:,:] = 0.0
            HR_soc_d[:,:] = 0.0
        if (orb_pseudo[n] == 'sp'):
            HR_soc_p[:,:] = soc_p_sp(theta,phi,norb)
            HR_soc_d[:,:] = 0.0
        if (orb_pseudo[n] == 'spd'):
            HR_soc_p[:,:] = soc_p_spd(theta,phi,norb)
            HR_soc_d[:,:] = soc_d_spd(theta,phi,norb)
        if (orb_pseudo[n] == 'ps'):
            HR_soc_p[:,:] = soc_p_ps(theta,phi,norb)
            HR_soc_d[:,:] = 0.0
        if (orb_pseudo[n] == 'sspd'):
            HR_soc_p[:,:] = soc_p_sspd(theta,phi,norb)
            HR_soc_d[:,:] = soc_d_sspd(theta,phi,norb)
        if (orb_pseudo[n] == 'ssppd'):
            HR_soc_p[:,:] = soc_p_ssppd(theta,phi,norb)
            HR_soc_d[:,:] = soc_d_ssppd(theta,phi,norb)

        uui = offset[n]
        uuj = offset[n]+norb_array[n]
        ddi = offset[n]+nawf
        ddj = offset[n]+norb_array[n]+nawf

        udi = offset[n]+nawf
        udj = offset[n]+norb_array[n]+nawf

        dui = udj 
        duj = udi

        # Up-Up
        HR_double[uui:uuj,uui:uuj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[0:norb,0:norb] + socStrengh[n,1]*HR_soc_d[0:norb,0:norb]
        # Down-Down
        HR_double[ddi:ddj,ddi:ddj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[norb:2*norb,norb:2*norb]  + socStrengh[n,1]*HR_soc_d[norb:2*norb,norb:2*norb]
        # Up-Down
        HR_double[uui:uuj,udi:udj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[0:norb,norb:2*norb] + socStrengh[n,1]*HR_soc_d[0:norb,norb:2*norb]
        # Down-Up
        HR_double[ddi:ddj,uui:uuj,0,0,0,0] += socStrengh[n,0]*HR_soc_p[norb:2*norb,0:norb] + socStrengh[n,1]*HR_soc_d[norb:2*norb,0:norb]

    return(HR_double)



################### PSEUDOPOTENTIAL PS ##############################33
def soc_p_ps(theta,phi,norb):

        HR_soc = np.zeros((2*norb,2*norb),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

	#Spin Up - Spin Up  part of the p-satets Hamiltonian
        HR_soc[0,1] = -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[0,2] =  0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[1,2] = -0.5*np.complex(0.0,cTheta)
        HR_soc[1,0]=np.conjugate(HR_soc[0,1])
        HR_soc[2,0]=np.conjugate(HR_soc[0,2])
        HR_soc[2,1]=np.conjugate(HR_soc[1,2])
	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[5:8,5:8] = - HR_soc[1:4,1:4] 
  	#Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[0,5] = -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[0,6] = -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[1,6] =  0.5*np.complex(0.0,sTheta)
        HR_soc[1,4] = -HR_soc[0,5]
        HR_soc[2,4] = -HR_soc[0,6]
        HR_soc[2,5] = -HR_soc[1,6]
	#Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[5,0]=np.conjugate(HR_soc[0,5])
        HR_soc[6,0]=np.conjugate(HR_soc[0,6])
        HR_soc[4,1]=np.conjugate(HR_soc[1,4])
        HR_soc[6,1]=np.conjugate(HR_soc[1,6])
        HR_soc[4,2]=np.conjugate(HR_soc[2,4])
        HR_soc[5,2]=np.conjugate(HR_soc[2,5])

	return(HR_soc)
################### END PSEUDOPOTENTIAL PS ##############################33

################### PSEUDOPOTENTIAL SP ##############################33
def soc_p_sp(theta,phi,norb):

        HR_soc = np.zeros((2*norb,2*norb),dtype=complex) 

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
        HR_soc[5:8,5:8] = - HR_soc[1:4,1:4] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[1,6] = -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[1,7] = -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[2,7] =  0.5*np.complex(0.0,sTheta)
        HR_soc[2,5] = -HR_soc[1,6]
        HR_soc[3,5] = -HR_soc[1,7]
        HR_soc[3,6] = -HR_soc[2,7]
	#Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[6,1]=np.conjugate(HR_soc[1,6])
        HR_soc[7,1]=np.conjugate(HR_soc[1,7])
        HR_soc[5,2]=np.conjugate(HR_soc[2,5])
        HR_soc[7,2]=np.conjugate(HR_soc[2,7])
        HR_soc[5,3]=np.conjugate(HR_soc[3,5])
        HR_soc[6,3]=np.conjugate(HR_soc[3,6])
	return(HR_soc)
################### END PSEUDOPOTENTIAL SP ##############################33

################### PSEUDOPOTENTIAL SPD ##############################33
def soc_p_spd(theta,phi,norb):

        HR_soc = np.zeros((2*norb,2*norb),dtype=complex) 

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


def soc_d_spd(theta,phi,norb):

        HR_soc = np.zeros((2*norb,2*norb),dtype=complex) 

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

################### END PSEUDOPOTENTIAL SPD ##############################33


################### PSEUDOPOTENTIAL SSPD ##############################33
def soc_p_sspd(theta,phi,norb):

    # Hardcoded to s,p,d. This must change latter.
        HR_soc = np.zeros((2*norb,2*norb),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

	#Spin Up - Spin Up  part of the p-satets Hamiltonian
        HR_soc[2,3] = -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[2,4] =  0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[3,4] = -0.5*np.complex(0.0,cTheta)
        HR_soc[3,2]=np.conjugate(HR_soc[2,3])
        HR_soc[4,2]=np.conjugate(HR_soc[2,4])
        HR_soc[4,3]=np.conjugate(HR_soc[3,4])
	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[12:15,12:15] = - HR_soc[2:5,2:5] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[2,13] = -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[2,14] = -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[3,14] =  0.5*np.complex(0.0,sTheta)
        HR_soc[3,12] = -HR_soc[2,13]
        HR_soc[4,12] = -HR_soc[2,14]
        HR_soc[4,13] = -HR_soc[3,14]
	#Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[13,2]=np.conjugate(HR_soc[2,13])
        HR_soc[14,2]=np.conjugate(HR_soc[2,14])
        HR_soc[12,3]=np.conjugate(HR_soc[3,12])
        HR_soc[14,3]=np.conjugate(HR_soc[3,14])
        HR_soc[12,4]=np.conjugate(HR_soc[4,12])
        HR_soc[13,4]=np.conjugate(HR_soc[4,13])
	return(HR_soc)

	

def soc_d_sspd(theta,phi,norb):

    # Hardcoded to s,p,d. This must change latter.
        HR_soc = np.zeros((2*norb,2*norb),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

        s3 = cmath.sqrt(3.0)

	#Spin Up - Spin Up  part of the d-satets Hamiltonian
        HR_soc[5,6] = -s3*0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[5,7] =  s3*0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[6,7] =  -0.5*np.complex(0.0,cTheta)
        HR_soc[6,8] =  -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[6,9] =   0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[7,8] =  -0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[7,9] =  -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[8,9] =  -1.0*np.complex(0.0,cTheta)
        HR_soc[6,5] = np.conjugate(HR_soc[5,6])
        HR_soc[7,5] = np.conjugate(HR_soc[5,7])
        HR_soc[7,6] = np.conjugate(HR_soc[6,7])
        HR_soc[8,6] = np.conjugate(HR_soc[6,8])
        HR_soc[9,6] = np.conjugate(HR_soc[6,9])
        HR_soc[8,7] = np.conjugate(HR_soc[7,8])
        HR_soc[9,7] = np.conjugate(HR_soc[7,9])

	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[15:20,15:20] = - HR_soc[5:10,5:10] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[5,16] = -s3*0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[5,17] = -s3*0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[6,17] =     0.5*( np.complex(0.0,sTheta))
        HR_soc[6,18] =    -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[6,19] =    -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[7,18] =     0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[7,19] =    -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[8,19] =     1.0*( np.complex(0.0,sTheta))
        HR_soc[6,15] =  -HR_soc[5,16] 
        HR_soc[7,15] =  -HR_soc[5,17] 
        HR_soc[7,16] =  -HR_soc[6,17]
        HR_soc[8,16] =  -HR_soc[6,18] 
        HR_soc[9,16] =  -HR_soc[6,19] 
        HR_soc[8,17] =  -HR_soc[7,18] 
        HR_soc[9,17] =  -HR_soc[7,19] 
        HR_soc[9,18] =  -HR_soc[8,19] 
    #Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[16,5] = np.conjugate(HR_soc[5,16]) 
        HR_soc[17,5] = np.conjugate(HR_soc[5,17])
        HR_soc[17,6] = np.conjugate(HR_soc[6,17])   
        HR_soc[18,6] = np.conjugate(HR_soc[6,18])   
        HR_soc[19,6] = np.conjugate(HR_soc[6,19])   
        HR_soc[18,7] = np.conjugate(HR_soc[7,18]) 
        HR_soc[19,7] = np.conjugate(HR_soc[7,19])    
        HR_soc[19,8] = np.conjugate(HR_soc[8,19])    
        HR_soc[15,6] = np.conjugate(HR_soc[6,15])
        HR_soc[15,7] = np.conjugate(HR_soc[7,15])
        HR_soc[16,7] = np.conjugate(HR_soc[7,16])
        HR_soc[16,8] = np.conjugate(HR_soc[8,16])
        HR_soc[16,9] = np.conjugate(HR_soc[9,16])
        HR_soc[17,8] = np.conjugate(HR_soc[8,17])
        HR_soc[17,9] = np.conjugate(HR_soc[9,17])
	return(HR_soc)
################### END PSEUDOPOTENTIAL SSPD ##############################33
