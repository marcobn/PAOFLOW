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

import re
import sys
import numpy as np
from mpi4py import MPI
import xml.etree.cElementTree as ET


def read_QE_output_xml( data_controller ):

    # Units
    Ry2eV = 13.60569193

    rank = MPI.COMM_WORLD.Get_rank()

    data_arrays = data_controller.data_arrays
    data_attributes = data_controller.data_attributes

    fpath = data_attributes['fpath']
    verbose = data_attributes['verbose']

    Sks = None
    non_ortho = True #data_attributes['non_ortho']

    atomic_proj = fpath+'/atomic_proj.xml'
    data_file   = fpath+'/data-file.xml'

    # Reading data-file.xml

    for event,elem in ET.iterparse(data_file,events=('start','end')):
        if event == 'end':
            if elem.tag == "CELL":
                alatunits  = elem.findall("LATTICE_PARAMETER")[0].attrib['UNITS']
                alat   = float(elem.findall("LATTICE_PARAMETER")[0].text.split()[0])
                if rank == 0 and verbose: print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

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
                if rank == 0 and verbose: print('Monkhorst&Pack grid',nk1,nk2,nk3,k1,k2,k3)
                elem.clear()

            # number of atoms
            if elem.tag == 'IONS':
                species = []
                pseudos = []
                nspecs = int(elem.findall("NUMBER_OF_SPECIES")[0].text.split()[0])
                for n in range(nspecs):
                    string = "SPECIE."+str(n+1)
                    species.append(elem.findall(string+"/ATOM_TYPE")[0].text.split()[0])
                    pseudos.append(elem.findall(string+"/PSEUDO")[0].text.split()[0])

                atoms = []
                natoms = int(elem.findall("NUMBER_OF_ATOMS")[0].text.split()[0])
                tau = np.zeros((natoms,3), dtype=float)
                for n in range(natoms):
                    string = "ATOM."+str(n+1)
                    aux = elem.findall(string)[0]
                    atoms.append(aux.attrib['SPECIES'][:-1])
                    tau[n,:] = np.array(aux.attrib['tau'].split(), dtype="float32")

    # Reading atomic_proj.xml

    group_nesting = 0
    readEigVals = False; readProj = False
    for event,elem in ET.iterparse(atomic_proj,events=('start','end')):

        if event == 'end' and  elem.tag == "HEADER":
            nkpnts = int(elem.findall("NUMBER_OF_K-POINTS")[0].text.strip())
            if rank == 0 and verbose: print('Number of kpoints: {0:d}'.format(nkpnts))

            nspin  = int(elem.findall("NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
            dftSO = False
            if nspin == 4:
                nspin = 1
                dftSO = True
            if rank == 0 and verbose: print('Number of spin components: {0:d}'.format(nspin))
            nelec = float(elem.findall("NUMBER_OF_ELECTRONS")[0].text.split()[0])
            nelec = int(nelec)
            if rank == 0 and verbose: print('Number of electrons: {0:d}'.format(nelec))

            kunits = elem.findall("UNITS_FOR_K-POINTS")[0].attrib['UNITS']

            nbnds  = int(elem.findall("NUMBER_OF_BANDS")[0].text.split()[0])
            if rank == 0 and verbose: print('Number of bands: {0:d}'.format(nbnds))

            aux    = elem.findall("UNITS_FOR_ENERGY")[0].attrib['UNITS']

            Efermi = float(elem.findall("FERMI_ENERGY")[0].text.split()[0])*Ry2eV
            if rank == 0 and verbose: print('Fermi energy: {0:f} eV '.format(Efermi))

            nawf   =int(elem.findall("NUMBER_OF_ATOMIC_WFC")[0].text.split()[0])
            if rank == 0 and verbose: print('Number of atomic wavefunctions: {0:d}'.format(nawf))


            elem.clear()

        if event == 'end' and elem.tag =="K-POINTS":
            kpnts  = np.array(elem.text.split(),dtype="float32").reshape((nkpnts,3))
            elem.clear()

        if event == 'end' and elem.tag =="WEIGHT_OF_K-POINTS":
            kpnts_wght  = np.array(elem.text.split(),dtype='float32')

            if kpnts_wght.shape[0] != nkpnts:
                sys.exit('Error in size of the kpnts_wght vector')
            elem.clear()

        #Read eigenvalues, projections ad overlaps (if non_ortho)

        if event == 'start':
            if elem.tag == "EIGENVALUES":
                group_nesting += 1
                readEigVals = True
                U = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)
                my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
            if elem.tag == "PROJECTIONS":
                group_nesting += 1
                ispin = 0
                readProj = True
            if elem.tag == "OVERLAPS" and non_ortho:
                readProj = True
                group_nesting += 1
                Sks = np.zeros((nawf,nawf,nkpnts),dtype=complex)
            if "K-POINT" in elem.tag:
                if readProj :
                    ik = int(float(elem.tag.split('.')[-1]))-1
                    group_nesting = 2
            if "SPIN" in elem.tag:
                if readProj:
                    group_nesting=3

        if event == 'end':
            #Read eigen values for each k-point
            if "K-POINT" in elem.tag and group_nesting == 1 and readEigVals: # EIGENVALUES/K-POINT.{0:d}
                ik = int(float(elem.tag.split('.')[-1]))-1
                if nspin ==1:
                    ispin = 0
                    subelem = elem.findall("EIG")[0]
                    #if verbose:print("Reading eigenvalues of ",  elem.tag)
                    eigk_type=subelem.attrib['type']
                    eigk_file=np.array(subelem.text.split(),dtype='float32')
                    my_eigsmat[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef

                else:
                    for ispin in range(nspin):
                        subelem = elem.findall("EIG.%d"%(ispin+1))[0]
                        #if verbose:print("Reading eigenvalues of ",elem.tag)
                        eigk_type=subelem.attrib['type']
                        eigk_file=np.array(subelem.text.split(),dtype='float32')
                        my_eigsmat[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef


            if 'ATMWFC' in elem.tag and readProj and nspin==1 : #PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}
                ispin == 0
                #if verbose:print("Reading", elem.tag, "of non-spin calc")
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

            if 'SPIN' in elem.tag and readProj and group_nesting ==3:
                ispin = int(float(elem.tag.split('.')[-1]))-1
                for iin in range(nawf):
                    subelem = elem.findall("ATMWFC.%d"%(iin+1))[0]
                    #if verbose:print("Reading ", subelem.tag, elem.tag, "of k-point",ik)
                    wfc_type= subelem.attrib['type']
                    aux     = subelem.text
                    aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')
                    if wfc_type=='real':
                        wfc = aux.reshape((nbnds,1))#wfc = nbnds x 1
                        U[:,iin,ik,ispin] = wfc[:,0]
                    elif wfc_type=='complex':
                        wfc = aux.reshape((nbnds,2))
                        U[:,iin,ik,ispin] = wfc[:,0]+1j*wfc[:,1]
                    else:
                        sys.exit('neither real nor complex??')

            if 'OVERLAP.' in elem.tag and readProj and non_ortho : #OVERLAPS/K-POINT.{0:d}/OVERLAP.1
                iin = int(float(elem.tag.split('.')[-1]))-1
                ovlp_type=elem.attrib['type']
                aux     =elem.text
                aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')

                if ovlp_type !='complex':
                    sys.exit('the overlaps are assumed to be complex numbers')
                if len(aux) != nawf**2*2:
                    sys.exit('wrong number of elements when reading the S matrix')

                aux = aux.reshape((nawf**2,2))
                ovlp_vector = aux[:,0]+1j*aux[:,1]
                Sks[:,:,ik] = ovlp_vector.reshape((nawf,nawf))
            #else:
            #    if 'OVERLAP' in elem.tag: print('OVERLAP')
            #    if readProj: print('readProj')
            #    if non_ortho: print('non_ortho')
            #    sys.exit('no overlaps found')

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
                if group_nesting == 2:
                    elem.clear()
                    readProj = False
                    group_nesting = 0
                    ik = 0
                    ispin = 0
                if group_nesting == 3 and ik == nkpnts:
                    elem.clear()
                    readProj = False
                    group_nesting = 0
                    ik = 0
                    ispin = 0
            #Finish reading OVERLAPS
            if elem.tag == "OVERLAPS" and non_ortho:
                    elem.clear()
                    readProj = False
                    group_nesting = 0
                    ik = 0

    omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))

    data_attributes['nawf'] = nawf
    data_attributes['nk1'] = nk1
    data_attributes['nk2'] = nk2
    data_attributes['nk3'] = nk3
    data_attributes['nkpnts'] = nkpnts
    data_attributes['nspin'] = nspin
    data_attributes['nelec'] = nelec
    data_attributes['natoms'] = natoms
    data_attributes['nbnds'] = nbnds
    data_attributes['alat'] = alat
    data_attributes['omega'] = omega
    data_attributes['Efermi'] = Efermi
    data_attributes['dftSO'] = dftSO
    data_arrays['tau'] = tau
    data_arrays['kpnts'] = kpnts
    data_arrays['atoms'] = atoms
    data_arrays['species'] = zip(species,pseudos)
    data_arrays['kpnts_wght'] = kpnts_wght
    data_arrays['a_vectors'] = a_vectors
    data_arrays['b_vectors'] = b_vectors
    data_arrays['my_eigsmat'] = my_eigsmat
    data_arrays['U'] = U
    if Sks is not None:
      data_arrays['Sks'] = Sks
