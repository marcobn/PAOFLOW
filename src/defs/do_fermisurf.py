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
import numpy as np
import cmath
import sys, time
import os
from write2bxsf import *
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from write3Ddatagrid import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def do_fermisurf(fermi_dw,fermi_up,E_k,alat,b_vectors,nk1,nk2,nk3,nawf,ispin,npool,inputpath):
    #maximum number of bands crossing fermi surface

    E_k_full = gather_full(E_k,npool)

    nbndx_plot = 10
    nktot = nk1*nk2*nk3
    

    if rank==0:
        eigband = np.zeros((nk1,nk2,nk3,nbndx_plot),dtype=float)
        ind_plot = np.zeros(nbndx_plot)

        E_k_rs = np.reshape(E_k_full,(nk1,nk2,nk3,nawf))



        Efermi = 0.0

        #collect the interpolated eignvalues
        icount = 0
        for ib in range(nawf):
            if ((np.amin(E_k_full[:,ib]) < fermi_up and np.amax(E_k_full[:,ib]) > fermi_up) or \
                (np.amin(E_k_full[:,ib]) < fermi_dw and np.amax(E_k_full[:,ib]) > fermi_dw) or \
                (np.amin(E_k_full[:,ib]) > fermi_dw and np.amax(E_k_full[:,ib]) < fermi_up)):
                if ( icount > nbndx_plot ): sys.exit("too many bands contributing")
                eigband[:,:,:,icount] = E_k_rs[:,:,:,ib]
                ind_plot[icount] = ib
                icount +=1
        x0 = np.zeros(3,dtype=float)   

        write2bxsf(fermi_dw,fermi_up,eigband, nk1, nk2, nk3, icount, ind_plot, Efermi, alat,x0, b_vectors, 'FermiSurf_'+str(ispin)+'.bxsf',inputpath)   

        for ib in xrange(icount):
            np.savez(os.path.join(inputpath,'Fermi_surf_band_'+str(ib)), nameband = eigband[:,:,:,ib])


    E_k_full = E_k_rs = None
