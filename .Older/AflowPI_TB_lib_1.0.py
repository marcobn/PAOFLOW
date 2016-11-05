# 
# AflowPI_TB_lib.py
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
#
# Copyright (C) 2015 Luis A. Agapito, 2016 Marco Buongiorno Nardelli
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli, 
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).

# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli, 
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).

# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli, 
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).

from __future__ import print_function
#from scipy import linalg
from scipy import linalg as LA
from numpy import linalg as LAN
import numpy as np
import xml.etree.ElementTree as ET
import sys
import re
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#units
Ry2eV   = 13.60569193

def read_input(input_file):
	
        read_S     = False
        shift_type = 2
        shift      = 20
        pthr       = 0.9
        do_comparison = False
	#bnd        = 1
        #nbnds_norm = 1
        #bnd_g      = 1

	f = open(input_file)
	lines=f.readlines()
	f.close
	for line in lines:
    		line = line.strip()
    		if re.search('fpath',line):
       			p = line.split()
       			fpath = p[1]
    		if re.search('read_S',line):
       			p = line.split()
       			read_S = p[1]
                        if read_S == 'False':
				read_S = (1 == 2)
 			else:
				read_S = (1 == 1)
    		if re.search('do_comparison',line):
       			p = line.split()
       			do_comparison = p[1]
                        if do_comparison == 'False':
				do_comparison = (1 == 2)
 			else:
				do_comparison = (1 == 1)
    		#if re.search('nbnds_norm',line):
       		#	p = line.split()
       		#	nbnds_norm = int(p[1])
    		if re.search('shift_type',line):
       			p = line.split()
       			shift_type = int(p[1])
    		if re.search('shift',line):
       			p = line.split()
       			shift = float(p[1])
    		if re.search('pthr',line):
       			p = line.split()
       			pthr = float(p[1])
    		#if re.search('bnd',line):
       		#	p = line.split()
       		#	bnd = int(p[1])
    		#if re.search('bnd_g',line):
       		#	p = line.split()
       		#	bnd_g = int(p[1])
	if fpath == '':
		sys.exit('missing path to _.save')

	return(read_S, shift_type, fpath, shift, pthr, do_comparison) #, bnd, nbnds_norm, bnd_g)


def build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U):
    Hks = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            my_eigs=my_eigsmat[:,ik,ispin]
            #Building the Hamiltonian matrix
            E = np.diag(my_eigs)
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            norms = 1/np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
            UU[:,:nbnds_norm] = UU[:,:nbnds_norm]*norms[:nbnds_norm]
            eta=shift
	    # Choose only the eigenvalues that are below the energy shift
	    bnd_ik=0
            for n in range(bnd):
		if my_eigs[n] <= eta:
			bnd_ik += 1
	    if bnd_ik == 0: sys.exit('no eigenvalues in selected energy range')
            ac = UU[:,:bnd_ik]  # filtering: bnd is defined by the projectabilities
            ee1 = E[:bnd_ik,:bnd_ik] 
            #if bnd == nbnds:
            #    bd = np.zeros((nawf,1))
            #    ee2 = 0
            #else:
            #    bd = UU[:,bnd:nbnds]
            #    ee2= E[bnd:nbnds,bnd:nbnds]
            if shift_type ==0:
                #option 1 (PRB 2013)
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(np.conj(ac).T))
            elif shift_type==1:
                #option 2 (PRB 2016)
                aux_p=LA.inv(np.dot(np.conj(ac).T,ac))
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(aux_p).dot(np.conj(ac).T))
            elif shift_type==2:
                # no shift 
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T)
            else:
                sys.exit('shift_type not recognized')
    return Hks

def build_Pn(nawf,nbnds,nkpnts,nspin,U):
    Pn = 0.0
    for ispin in range(nspin):
        for ik in range(nkpnts):
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            Pn += np.real(np.sum(np.conj(UU)*UU,axis=0))/nkpnts
    return Pn

def get_R_grid(nk1,nk2,nk3,a_vectors):
	# Generate a regular grid in real space (crystal coordinates)
	# This is the algorithm in WanT
	nr = 0
        nrtot = 100000
	R = np.zeros((nrtot,3),dtype=float)
	R_wght = np.zeros((nrtot),dtype=float)
	for i in range(nk1):
        	for j in range(nk2):
                	for k in range(nk3):
				ri=i-int((nk1+1)/2)
				rj=j-int((nk2+1)/2)
				rk=k-int((nk3+1)/2)
                        	#R[nr,:] = np.array([ri,rj,rk]) #ri*a_vectors[0,:]+rj*a_vectors[1,:]+rk*a_vectors[2,:]
                        	R[nr,:] = ri*a_vectors[0,:]+rj*a_vectors[1,:]+rk*a_vectors[2,:]
 				R_wght[nr] = 1.0
                        	nr += 1
	nrtot = nr
	# Check that -R is always present
	for ir in range(nrtot):
		for ir2 in range(nrtot):
			test_opp = all(R[ir2,:] == -R[ir,:])
			if test_opp == True:
				break
		if test_opp == False:
			nr=nr+1
			R[nr,:] = -R[ir,:]
			R_wght[nr] = R_wght[ir]/2.0
			R_wght[ir] = R_wght[ir]/2.0
	nrtot = nr
	return(R,R_wght,nrtot)

def get_R_grid_WS(nk1,nk2,nk3,a_vectors):
	# Generate a regular grid of points that fall inside and on the surface of the Wigner-Seitz
	# cell centered in the origin of the lattice (as in Wannier90)

	nrtot=100000
        R = np.zeros((nrtot,3),dtype=float)
        R_wght = np.zeros((nrtot),dtype=float)
	
	# Define metric in real space
	metric = np.zeros((3,3),dtype=float)
	metric = a_vectors.T.dot(a_vectors)
	nrtot=0
	nR=2
	nRn=(2*nR+1)**3
	for n1 in range(-nk1,nk1+1):
		for n2 in range(-nk2,nk2+1):
			for n3 in range(-nk3,nk3+1):
			# loop over the 125 points R. R=0 corresponds to i1=i2=i3=0
				icnt=0
				dist = np.zeros((nRn),dtype=float)
				for i1 in range(-nR,nR+1):
					for i2 in range(-nR,nR+1):
						for i3 in range(-nR,nR+1):
							ndiff = np.zeros((3),dtype=float)
							if i1==i2==i3==0: i0=icnt
							# calculate the distance squared
							ndiff[0]=n1-i1*nk1		
							ndiff[1]=n2-i2*nk2		
							ndiff[2]=n3-i3*nk3		
							dist[icnt] += ndiff.T.dot(metric).dot(ndiff)
							icnt += 1
				dist_min = min(dist)
				if abs(dist[i0] - dist_min) < 1.0e-7:
					nrtot += 1
					degen=0.0
					for i in range(nRn):
						if abs(dist[i]-dist_min) < 1.0e-7: 
							degen += 1.0
					R[nrtot,:] = n1*a_vectors[0,:]+n2*a_vectors[1,:]+n3*a_vectors[2,:]
					R_wght[nrtot] = 1.0/float(degen)	
	return(R,R_wght,nrtot)

def kpnts_interpolation_mesh():
	# To be made general reading boundary points from input. For now:
	# define k vectors (L-Gamma-X-K-Gamma) by hand

	# L - Gamma
	nk=60
	kx=np.linspace(-0.5, 0.0, nk)
	ky=np.linspace(0.5, 0.0, nk)
	kz=np.linspace(0.5, 0.0, nk)
	k1=np.array([kx,ky,kz])
	# Gamma - X
	kx=np.linspace(0.0, -0.75, nk)
	ky=np.linspace(0.0, 0.75, nk)
	kz=np.zeros(nk)
	k2=np.array([kx,ky,kz])
	# X - K 
	kx=np.linspace(-0.75, -1.0, nk)
	ky=np.linspace(0.75,  0.0, nk)
	kz=np.zeros(nk)
	k3=np.array([kx,ky,kz])
	# K - Gamma
	kx=np.linspace(-1.0, 0.0, nk)
	ky=np.zeros(nk)
	kz=np.zeros(nk)
	k4=np.array([kx,ky,kz])

	k=np.concatenate((k1,k2,k3,k4),1)

	# Define path for plotting
	nkpi = 0
	kmod = np.zeros(k.shape[1],dtype=float)
	for ik in range(k.shape[1]):
        	if ik < nk:
                	kmod[nkpi]=-np.sqrt(np.absolute(np.dot(k[:,ik],k[:,ik])))
        	elif ik >= nk and ik < 2*nk:
                	kmod[nkpi]=np.sqrt(np.absolute(np.dot(k[:,ik],k[:,ik])))
        	elif ik >= 2*nk:
                	kmod[nkpi]=1+np.sqrt(2)-np.sqrt(np.absolute(np.dot(k[:,ik],k[:,ik])))
		nkpi += 1
	return (k,kmod,nkpi)

def read_QE_output_xml(fpath,read_S):
 atomic_proj = fpath+'/atomic_proj.xml'
 data_file   = fpath+'/data-file.xml'

 # Reading data-file.xml
 tree  = ET.parse(data_file)
 root  = tree.getroot()

 alatunits  = root.findall("./CELL/LATTICE_PARAMETER")[0].attrib['UNITS']
 alat   = float(root.findall("./CELL/LATTICE_PARAMETER")[0].text.split()[0])

 print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a1")[0].text.split()
 a1=[float(i) for i in aux]

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a2")[0].text.split()
 a2=[float(i) for i in aux]

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a3")[0].text.split()
 a3=[float(i) for i in aux]

 a_vectors = np.array([a1,a2,a3])/alat #in units of alat
# print(a_vectors.shape)
# print(a_vectors)

 # Monkhorst&Pack grid
 nk1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk1'])
 nk2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk2'])
 nk3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk3'])
 k1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k1'])
 k2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k2'])
 k3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k3'])
 print('Monkhorst&Pack grid',nk1,nk2,nk3,k1,k2,k3)

 # Reading atomic_proj.xml
 tree  = ET.parse(atomic_proj)
 root  = tree.getroot()

 nkpnts = int(root.findall("./HEADER/NUMBER_OF_K-POINTS")[0].text.strip())
 print('Number of kpoints: {0:d}'.format(nkpnts))

 nspin  = int(root.findall("./HEADER/NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
 print('Number of spin components: {0:d}'.format(nspin))

 kunits = root.findall("./HEADER/UNITS_FOR_K-POINTS")[0].attrib['UNITS']
 print('Units for the kpoints: {0:s}'.format(kunits))

 aux = root.findall("./K-POINTS")[0].text.split()
 kpnts  = np.array([float(i) for i in aux]).reshape((nkpnts,3))
 print('Read the kpoints')

 aux = root.findall("./WEIGHT_OF_K-POINTS")[0].text.split()
 kpnts_wght  = np.array([float(i) for i in aux])

 if kpnts_wght.shape[0] != nkpnts:
 	sys.exit('Error in size of the kpnts_wght vector')
 else:
 	print('Read the weight of the kpoints')


 nbnds  = int(root.findall("./HEADER/NUMBER_OF_BANDS")[0].text.split()[0])
 print('Number of bands: {0:d}'.format(nbnds))

 aux    = root.findall("./HEADER/UNITS_FOR_ENERGY")[0].attrib['UNITS']
 print('The units for energy are {0:s}'.format(aux))

 Efermi = float(root.findall("./HEADER/FERMI_ENERGY")[0].text.split()[0])*Ry2eV
 print('Fermi energy: {0:f} eV '.format(Efermi))

 nawf   =int(root.findall("./HEADER/NUMBER_OF_ATOMIC_WFC")[0].text.split()[0])
 print('Number of atomic wavefunctions: {0:d}'.format(nawf))

 #Read eigenvalues and projections

 U = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)
 my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
 for ispin in range(nspin):
   for ik in range(nkpnts):
     #Reading eigenvalues
     if nspin==1:
         eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].attrib['type']
     else:
         eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].attrib['type']
     if eigk_type != 'real':
       sys.exit('Reading eigenvalues that are not real numbers')
     if nspin==1:
       eigk_file=np.array([float(i) for i in root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].text.split()])
     else:
       eigk_file=np.array([float(i) for i in root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].text.split().split()])
     my_eigsmat[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef

     #Reading projections
     for iin in range(nawf): #There will be nawf projections. Each projector of size nbnds x 1
       if nspin==1:
         wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].attrib['type']
         aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].text
       else:
         wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,iin+1))[0].attrib['type']
         aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,ispin+1,iin+1))[0].text

       aux = np.array([float(i) for i in re.split(',|\n',aux.strip())])

       if wfc_type=='real':
         wfc = aux.reshape((nbnds,1))#wfc = nbnds x 1
         U[:,iin,ik,ispin] = wfc[:,0]
       elif wfc_type=='complex':
         wfc = aux.reshape((nbnds,2))
         U[:,iin,ik,ispin] = wfc[:,0]+1j*wfc[:,1]
       else:
         sys.exit('neither real nor complex??')

 if read_S:
   Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
   for ik in range(nkpnts):
     #There will be nawf projections. Each projector of size nbnds x 1
     ovlp_type = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].attrib['type']
     aux = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].text
     aux = np.array([float(i) for i in re.split(',|\n',aux.strip())])

     if ovlp_type !='complex':
       sys.exit('the overlaps are assumed to be complex numbers')
     if len(aux) != nawf**2*2:
       sys.exit('wrong number of elements when reading the S matrix')

     aux = aux.reshape((nawf**2,2))
     ovlp_vector = aux[:,0]+1j*aux[:,1]
     Sks[:,:,ik] = ovlp_vector.reshape((nawf,nawf))
   return(U,Sks, my_eigsmat, alat, a_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, nk1,nk2,nk3)
 else:
   return(U, my_eigsmat, alat, a_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, nk1,nk2,nk3)

def plot_compare_TB_DFT_eigs(Hks,Sks,my_eigsmat,read_S):
    import matplotlib.pyplot as plt
    import os

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))

    ispin = 0 #plots only 1 spin channel
    #for ispin in range(nspin):
    for ik in range(nkpnts):
        if read_S:
		eigval,_ = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
	else:	
		eigval,_ = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    fig=plt.figure
    nbnds_dft,_,_=my_eigsmat.shape
    for i in range(nbnds_dft):
        #print("{0:d}".format(i))
        yy = my_eigsmat[i,:,ispin]
        if i==0:
          plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime',label='DFT')
        else:
          plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime')

    for i in range(nbnds_tb):
        yy = E_k[i,:,ispin]
        if i==0:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None',label='TB')
        else:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None')

    plt.xlabel('k-points')
    plt.ylabel('Energy - E$_F$ (eV)')
    plt.legend()
    plt.title('Comparison of TB vs. DFT eigenvalues')
    plt.savefig('comparison.pdf',format='pdf')
    #os.system('open comparison.pdf') #for macs


def plot_TB_eigs(Hks,Sks,kmod,read_S):
    import matplotlib.pyplot as plt
    import os

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))

    ispin = 0 #plots only 1 spin channel
    #for ispin in range(nspin):
    for ik in range(nkpnts):
        if read_S:
		eigval,_ = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
	else:	
		eigval,_ = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    fig=plt.figure

    for i in range(nbnds_tb):
        xx = kmod[i]
        yy = E_k[i,:,ispin]
        if i==0:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None',label='TB')
        else:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None')

    plt.xlabel('k-points')
    plt.ylabel('Energy - E$_F$ (eV)')
    plt.legend()
    plt.title('TB eigenvalues')
    plt.savefig('interpolation.pdf',format='pdf')
    #os.system('open comparison.pdf') #for macs

def plot_grid(x,y,z):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_title('FCC crystal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

        ax.view_init(elev=12, azim=40)              # elevation and angle
        ax.dist=12                                  # distance

        ax.scatter(
        x, y, z,                                   # data
        color='purple',                            # marker colour
        marker='o',                                # marker shape
        s=30                                       # marker size
        )

        plt.show()
