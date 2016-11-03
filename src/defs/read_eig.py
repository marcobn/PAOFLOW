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

def read_eig(ini_ik,end_ik,root,nbnds,nawf,nkpnts,nspin,Efermi):

        my_eigsmat_p = np.zeros((nbnds,nkpnts,nspin))

        for ik in range(ini_ik,end_ik):
                for ispin in range(nspin):
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
                                eigk_file=np.array(root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1)) \
                                        [0].text.split().split(),dtype='float32')
                        my_eigsmat_p[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef

	return(my_eigsmat_p)
