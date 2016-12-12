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
from __future__ import print_function
import numpy as np
import xml.etree.cElementTree as ET
import sys,time
import re
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *

#units
Ry2eV   = 13.60569193

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_QE_output_xml(fpath,non_ortho):
    atomic_proj = fpath+'/atomic_proj.xml'
    data_file   = fpath+'/data-file.xml'

    verbose = None
    verbose == False

# Reading data-file.xml
    tree  = ET.parse(data_file)
    root  = tree.getroot()

    alatunits  = root.findall("./CELL/LATTICE_PARAMETER")[0].attrib['UNITS']
    alat   = float(root.findall("./CELL/LATTICE_PARAMETER")[0].text.split()[0])

    if rank == 0 and verbose == True: print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

    aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a1")[0].text.split()
    a1=np.array(aux,dtype="float32")

    aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a2")[0].text.split()
    a2=np.array(aux,dtype="float32")

    aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a3")[0].text.split()
    a3=np.array(aux,dtype="float32")

    a_vectors = np.array([a1,a2,a3])/alat #in units of alat
    aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b1")[0].text.split()
    b1=np.array(aux,dtype='float32')

    aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b2")[0].text.split()
    b2=np.array(aux,dtype='float32')

    aux=root.findall("./CELL/RECIPROCAL_LATTICE_VECTORS/b3")[0].text.split()
    b3=np.array(aux,dtype='float32')

    b_vectors = np.array([b1,b2,b3]) #in units of 2pi/alat

    # numbor of atoms
    natoms=int(float(root.findall("./IONS/NUMBER_OF_ATOMS")       [0].text.split()[0]))

    # Monkhorst&Pack grid
    nk1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk1'])
    nk2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk2'])
    nk3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_GRID")[0].attrib['nk3'])
    k1=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k1'])
    k2=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k2'])
    k3=int(root.findall("./BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET")[0].attrib['k3'])
    if rank == 0 and  verbose == True: print('Monkhorst&Pack grid',nk1,nk2,nk3,k1,k2,k3)

    if rank == 0 and  verbose == True: print('reading data-file.xml in ',time.clock(),' sec')
    reset=time.clock()

    # Reading atomic_proj.xml
    tree  = ET.parse(atomic_proj)
    root  = tree.getroot()

    nkpnts = int(root.findall("./HEADER/NUMBER_OF_K-POINTS")[0].text.strip())
    #if rank == 0: print('Number of kpoints: {0:d}'.format(nkpnts))

    nspin  = int(root.findall("./HEADER/NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
    #if rank == 0: print('Number of spin components: {0:d}'.format(nspin))

    kunits = root.findall("./HEADER/UNITS_FOR_K-POINTS")[0].attrib['UNITS']
    #if rank == 0: print('Units for the kpoints: {0:s}'.format(kunits))

    aux = root.findall("./K-POINTS")[0].text.split()
    kpnts  = np.array(aux,dtype="float32").reshape((nkpnts,3))
    #if rank == 0: print('Read the kpoints')

    aux = root.findall("./WEIGHT_OF_K-POINTS")[0].text.split()
    kpnts_wght  = np.array(aux,dtype='float32')

    if kpnts_wght.shape[0] != nkpnts:
        sys.exit('Error in size of the kpnts_wght vector')


    nbnds  = int(root.findall("./HEADER/NUMBER_OF_BANDS")[0].text.split()[0])
    if rank == 0 and  verbose == True: print('Number of bands: {0:d}'.format(nbnds))

    aux    = root.findall("./HEADER/UNITS_FOR_ENERGY")[0].attrib['UNITS']
    #if rank == 0: print('The units for energy are {0:s}'.format(aux))

    Efermi = float(root.findall("./HEADER/FERMI_ENERGY")[0].text.split()[0])*Ry2eV
    if rank == 0 and  verbose == True: print('Fermi energy: {0:f} eV '.format(Efermi))

    nawf   =int(root.findall("./HEADER/NUMBER_OF_ATOMIC_WFC")[0].text.split()[0])
    if rank == 0 and  verbose == True: print('Number of atomic wavefunctions: {0:d}'.format(nawf))

    #Read eigenvalues and projections

    U = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)  # final data array
    my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
    Uaux = np.zeros((nbnds,nawf,nkpnts,nspin,1),dtype=complex) # read data from task
    my_eigsmataux = np.zeros((nbnds,nkpnts,nspin,1))
    Uaux1 = np.zeros((nbnds,nawf,nkpnts,nspin,1),dtype=complex) # receiving data array
    my_eigsmataux1 = np.zeros((nbnds,nkpnts,nspin,1))

    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nkpnts)

    Uaux[:,:,:,:,0] = read_proj(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi)

    if rank == 0:
        U[:,:,:,:]=Uaux[:,:,:,:,0]
        for i in xrange(1,size):
            comm.Recv(Uaux1,ANY_SOURCE)
            U[:,:,:,:] += Uaux1[:,:,:,:,0]
    else:
        comm.Send(Uaux,0)
    U = comm.bcast(U)

    my_eigsmataux[:,:,:,0] = read_eig(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi)

    if rank == 0:
        my_eigsmat[:,:,:]=my_eigsmataux[:,:,:,0]
        for i in xrange(1,size):
            comm.Recv(my_eigsmataux1,ANY_SOURCE)
            my_eigsmat[:,:,:] += my_eigsmataux1[:,:,:,0]
    else:
        comm.Send(my_eigsmataux,0)
    my_eigsmat = comm.bcast(my_eigsmat)

    if rank == 0 and  verbose == True: print('reading eigenvalues and projections in ',time.clock()-reset,' sec')
    reset=time.clock()

    if non_ortho:
        Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
        for ik in xrange(nkpnts):
            #There will be nawf projections. Each projector of size nbnds x 1
            ovlp_type = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].attrib['type']
            aux = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].text
            #aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')
            aux = np.array([float(i) for i in re.split(',|\n',aux.strip())])

            if ovlp_type !='complex':
                sys.exit('the overlaps are assumed to be complex numbers')
            if len(aux) != nawf**2*2:
                sys.exit('wrong number of elements when reading the S matrix')

            aux = aux.reshape((nawf**2,2))
            ovlp_vector = aux[:,0]+1j*aux[:,1]
            Sks[:,:,ik] = ovlp_vector.reshape((nawf,nawf))

        return(U,Sks, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, \
             nk1, nk2, nk3, natoms)

    else:
        return(U, my_eigsmat, alat, a_vectors, b_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf, \
             nk1, nk2, nk3, natoms)

def read_eig(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi):

    my_eigsmat_p = np.zeros((nbnds,nkpnts,nspin))

    for ik in xrange(ini_ik,end_ik):
        for ispin in xrange(nspin):
        #Reading eigenvalues
            if nspin==1:
                eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].attrib['type']
            else:
                eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].attrib['type']
            if eigk_type != 'real':
                sys.exit('Reading eigenvalues that are not real numbers')
            if nspin==1:
                eigk_file=np.array(root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].text.split(),dtype='float32')
            else:
                eigk_file=np.array(root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].text.split(),dtype='float32')
            my_eigsmat_p[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef

    return(my_eigsmat_p)


def read_proj(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi):

    U_p = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)

    for ik in xrange(ini_ik,end_ik):
        for ispin in xrange(nspin):
            #Reading projections
            for iin in xrange(nawf): #There will be nawf projections. Each projector of size nbnds x 1
                if nspin==1:
                    wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].attrib['type']
                    aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].text
                else:
                    wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,ispin+1,iin+1))[0].attrib['type']
                    aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,ispin+1,iin+1))[0].text

                aux = np.array(re.split(',|\n',aux.strip()),dtype='float32')

                if wfc_type=='real':
                    wfc = aux.reshape((nbnds,1))#wfc = nbnds x 1
                    U_p[:,iin,ik,ispin] = wfc[:,0]
                elif wfc_type=='complex':
                    wfc = aux.reshape((nbnds,2))
                    U_p[:,iin,ik,ispin] = wfc[:,0]+1j*wfc[:,1]
                else:
                    sys.exit('neither real nor complex??')
    return(U_p)

if __name__ == '__main__':
    read_QE_output_xml('./',False)
