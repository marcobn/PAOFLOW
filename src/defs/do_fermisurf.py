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
import cmath
import sys, time

from write2bxsf import *
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
def do_fermisurf(E_k,alat,A,nk1,nk2,nk3,nawf,ispin):
    #maximum number of bands crossing fermi surface
    A1 = A[0]*alat
    A2 = A[1]*alat
    A3 = A[2]*alat

    nbndx_plot = 5
    nktot = nk1*nk2*nk3
    
#    vkpt_int_cry = np.zeros((3,nktot), dtype=float)
    eigband = np.zeros((nk1,nk2,nk3,nbndx_plot),dtype=float)
    ind_plot = np.zeros(nbndx_plot)
    E_K = np.reshape(E_k,(nk1,nk2,nk3,nawf))
    Efermi = 0.0
    
    #collect the interpolated eignvalues
    icount = 0
    for ib in range(nawf):
        if (np.amin(E_k[:,ib]) < Efermi and np.amax(E_k[:,ib] > Efermi)):
            if ( icount > nbndx_plot ): sys.exit("too many bands contributing")
            eigband[:,:,:,icount] = E_K[:,:,:,ib]
            ind_plot[icount] = ib
            icount +=1
    B1= 2*np.pi*np.cross(A2,A3)/(np.dot(A1,np.cross(A2,A3)))
    B2= 2*np.pi*np.cross(A1,A3)/(np.dot(A2,np.cross(A1,A3)))
    B3= 2*np.pi*np.cross(A1,A2)/(np.dot(A3,np.cross(A1,A2)))
    x0 = np.zeros(3,dtype=float)   

    write2bxsf(eigband, nk1, nk2, nk3, icount, ind_plot, Efermi, x0, B1, B2, B3, 'FermiSurf_'+str(ispin)+'.bxsf')   
    
    return()
