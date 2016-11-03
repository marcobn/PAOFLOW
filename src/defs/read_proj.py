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
from __future__ import print_function
import numpy as np
import xml.etree.cElementTree as ET
import sys,time
import re
from mpi4py import MPI

#units
Ry2eV   = 13.60569193

def read_proj(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi):

        U_p = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)
        my_eigsmat_p = np.zeros((nbnds,nkpnts,nspin))

        for ik in range(ini_ik,end_ik):
                for ispin in range(nspin):
                        #Reading projections
                        for iin in range(nawf): #There will be nawf projections. Each projector of size nbnds x 1
                                if nspin==1:
                                        wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].attrib['type']
                                        aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].text
                                else:
                                        wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,iin+1))[0].attrib['type']
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
