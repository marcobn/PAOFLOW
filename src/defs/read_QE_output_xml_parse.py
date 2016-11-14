#
# AFLOWpi_TB
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

def read_QE_output_xml(fpath):
    atomic_proj = fpath+'/atomic_proj.xml'
    data_file   = fpath+'/data-file.xml'
    read_S = False

    # Reading data-file.xml

    for event,elem in ET.iterparse(data_file,events=('start','end')):
        if event == 'end':
            if elem.tag == "CELL":
                alatunits  = elem.findall("LATTICE_PARAMETER")[0].attrib['UNITS']
                alat   = float(elem.findall("LATTICE_PARAMETER")[0].text.split()[0])
                print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

                aux=elem.findall("DIRECT_LATTICE_VECTORS/a1")[0].text.split()
                a1=np.array(aux,dtype="float32")

                aux=elem.findall("DIRECT_LATTICE_VECTORS/a2")[0].text.split()
                a2=np.array(aux,dtype="float32")

                aux=elem.findall("DIRECT_LATTICE_VECTORS/a3")[0].text.split()
                a3=np.array(aux,dtype="float32")

                a_vectors = np.array([a1,a2,a3])/alat #in units of alat
                aux=elem.findall("RECIPROCAL_LATTICE_VECTORS/b1")[0].text.split()
                b1=np.array(aux,dtype='float32')

                aux=elem.findall("RECIPROCAL_LATTICE_VECTORS/b2")[0].text.split()
                b2=np.array(aux,dtype='float32')

                aux=elem.findall("RECIPROCAL_LATTICE_VECTORS/b3")[0].text.split()
                b3=np.array(aux,dtype='float32')

                b_vectors = np.array([b1,b2,b3]) #in units of 2pi/alat

                elem.clear()

            if elem.tag == 'BRILLOUIN_ZONE':
                # Monkhorst&Pack grid
                nk1=int(elem.findall("MONKHORST_PACK_GRID")[0].attrib['nk1'])
                nk2=int(elem.findall("MONKHORST_PACK_GRID")[0].attrib['nk2'])
                nk3=int(elem.findall("MONKHORST_PACK_GRID")[0].attrib['nk3'])
                k1=int(elem.findall("MONKHORST_PACK_OFFSET")[0].attrib['k1'])
                k2=int(elem.findall("MONKHORST_PACK_OFFSET")[0].attrib['k2'])
                k3=int(elem.findall("MONKHORST_PACK_OFFSET")[0].attrib['k3'])
                elem.clear()

                print('Monkhorst&Pack grid',nk1,nk2,nk3,k1,k2,k3)


    # Reading atomic_proj.xml

    group_nesting = 0
    readEigVals = False; readProj = False
    for event,elem in ET.iterparse(atomic_proj,events=('start','end')):
        if event == 'end' and  elem.tag == "HEADER":
            nkpnts = int(elem.findall("NUMBER_OF_K-POINTS")[0].text.strip())
            print('Number of kpoints: {0:d}'.format(nkpnts))

            nspin  = int(elem.findall("NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
            print('Number of spin components: {0:d}'.format(nspin))

            kunits = elem.findall("UNITS_FOR_K-POINTS")[0].attrib['UNITS']

            nbnds  = int(elem.findall("NUMBER_OF_BANDS")[0].text.split()[0])
            print('Number of bands: {0:d}'.format(nbnds))

            aux    = elem.findall("UNITS_FOR_ENERGY")[0].attrib['UNITS']

            Efermi = float(elem.findall("FERMI_ENERGY")[0].text.split()[0])*Ry2eV
            print('Fermi energy: {0:f} eV '.format(Efermi))

            nawf   =int(elem.findall("NUMBER_OF_ATOMIC_WFC")[0].text.split()[0])
            print('Number of atomic wavefunctions: {0:d}'.format(nawf))


            elem.clear()

        if event == 'end' and elem.tag =="K-POINTS":
            kpnts  = np.array(elem.text.split(),dtype="float32").reshape((nkpnts,3))
            elem.clear()

        if event == 'end' and elem.tag =="WEIGHT_OF_K-POINTS":
            kpnts_wght  = np.array(elem.text.split(),dtype='float32')

            if kpnts_wght.shape[0] != nkpnts:
                sys.exit('Error in size of the kpnts_wght vector')
            elem.clear()

        #Read eigenvalues and projections


        if event == 'start':
            if elem.tag == "EIGENVALUES":
                group_nesting += 1
                readEigVals = True
                U = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)
                my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
                elem.clear()
            if elem.tag == "PROJECTIONS":
                group_nesting += 1
                ispin = 0
                readProj = True
                elem.clear()
            if "K-POINT" in elem.tag:
                if readProj :
                    ik = int(float(elem.tag.split('.')[-1]))-1
                    group_nesting += 1
                    elem.clear()
            if 'SPIN' in elem.tag and group_nesting ==2:    #PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}
                ispin = int(float(elem.tag.split('.')[-1]))-1
                group_nesting += 1
                elem.clear()

        if event == 'end':
            #Read eigen values for each k-point
            if "K-POINT" in elem.tag and group_nesting == 1 and readEigVals: # EIGENVALUES/K-POINT.{0:d}
                ik = int(float(elem.tag.split('.')[-1]))-1
                if nspin ==1:
                    ispin = 0
                    eigk_type=elem.findall("EIG")[0].attrib['type']
                    eigk_file=np.array(elem.findall("EIG")[0].text.split(),dtype='float32')
                    my_eigsmat[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef

                else:
                    for ispin in range(nspin):
                        eigk_type=elem.findall("EIG.%d"%(ispin+1))[0].attrib['type']
                        eigk_file=np.array(elem.findall("EIG.%d"%(ispin+1))[0].text.split(),dtype='float32')
                        my_eigsmat[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef
                elem.clear()


            if 'ATMWFC' in elem.tag and readProj : #PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d} || PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}
                if group_nesting ==2 : ispin == 0
                iin = int(float(elem.tag.split('.')[-1]))-1
                wfc_type=elem.attrib['type']
                aux     =elem.text
                aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')

                if wfc_type=='real':
                    wfc = aux.reshape((nbnds,1))#wfc = nbnds x 1
                    U[:,iin,ik,ispin] = wfc[:,0]
                elif wfc_type=='complex':
                    wfc = aux.reshape((nbnds,2))
                    U[:,iin,ik,ispin] = wfc[:,0]+1j*wfc[:,1]
                else:
                    sys.exit('neither real nor complex??')

                elem.clear()
            #Finish reading eigen values
            if elem.tag == "EIGENVALUES":
                if group_nesting == 1:
                    elem.clear()
                    readEigVals = False
                    group_nesting = 0
                    ik = 0
                    ispin = 0
            #Finish reading projections
            if elem.tag == "PROJECTIONS":
                if group_nesting == 2 or group_nesting ==3:
                    elem.clear()
                    readProj = False
                    group_nesting = 0
                    ik = 0
                    ispin = 0


    return(U, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, nk1, nk2, nk3)
