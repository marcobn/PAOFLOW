#
# AflowPI_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
import numpy as np
import xml.etree.cElementTree as ET
import sys
import re

#units
Ry2eV   = 13.60569193

def read_QE_output_xml(fpath,read_S):
 atomic_proj = fpath+'/atomic_proj.xml'
 data_file   = fpath+'/data-file.xml'

 # Reading data-file.xml
 print('...reading data-file.xml')
 tree  = ET.parse(data_file)
 root  = tree.getroot()

 alatunits  = root.findall("./CELL/LATTICE_PARAMETER")[0].attrib['UNITS']
 alat   = float(root.findall("./CELL/LATTICE_PARAMETER")[0].text.split()[0])

 #print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a1")[0].text.split()
 a1=[float(i) for i in aux]

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a2")[0].text.split()
 a2=[float(i) for i in aux]

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a3")[0].text.split()
 a3=[float(i) for i in aux]

 a_vectors = np.array([a1,a2,a3])/alat #in units of alat
# print(a_vectors.shape)
# print(a_vectors)
 aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b1")[0].text.split()
 b1=[float(i) for i in aux]

 aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b2")[0].text.split()
 b2=[float(i) for i in aux]

 aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b3")[0].text.split()
 b3=[float(i) for i in aux]

 b_vectors = np.array([b1,b2,b3]) #in units of 2pi/alat

 # Monkhorst&Pack grid
 nk1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk1'])
 nk2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk2'])
 nk3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk3'])
 k1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k1'])
 k2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k2'])
 k3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k3'])
 print('Monkhorst&Pack grid',nk1,nk2,nk3,k1,k2,k3)

 # Reading atomic_proj.xml
 print('...reading atomic-proj.xml')
 tree  = ET.parse(atomic_proj)
 root  = tree.getroot()

 nkpnts = int(root.findall("./HEADER/NUMBER_OF_K-POINTS")[0].text.strip())
 #print('Number of kpoints: {0:d}'.format(nkpnts))

 nspin  = int(root.findall("./HEADER/NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
 #print('Number of spin components: {0:d}'.format(nspin))

 kunits = root.findall("./HEADER/UNITS_FOR_K-POINTS")[0].attrib['UNITS']
 #print('Units for the kpoints: {0:s}'.format(kunits))

 aux = root.findall("./K-POINTS")[0].text.split()
 kpnts  = np.array([float(i) for i in aux]).reshape((nkpnts,3))
 #print('Read the kpoints')

 aux = root.findall("./WEIGHT_OF_K-POINTS")[0].text.split()
 kpnts_wght  = np.array([float(i) for i in aux])

 if kpnts_wght.shape[0] != nkpnts:
 	sys.exit('Error in size of the kpnts_wght vector')

 nbnds  = int(root.findall("./HEADER/NUMBER_OF_BANDS")[0].text.split()[0])
 print('Number of bands: {0:d}'.format(nbnds))

 aux    = root.findall("./HEADER/UNITS_FOR_ENERGY")[0].attrib['UNITS']
 #print('The units for energy are {0:s}'.format(aux))

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
   return(U,Sks, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, \
		nk1, nk2, nk3)
 else:
   return(U, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, \
		nk1, nk2, nk3)

