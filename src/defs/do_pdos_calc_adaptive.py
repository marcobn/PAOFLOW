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
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from communication import scatter_array
from smearing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_pdos_calc_adaptive(E_k,emin,emax,delta,v_k,nk1,nk2,nk3,nawf,ispin,smearing,inputpath):
    # PDOS calculation with gaussian smearing
    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1000
    ene = np.arange(emin,emax,de,dtype=float)

    E_k = np.real(E_k)

    if rank==0:
        pdos = np.zeros((nawf,ene.size),dtype=float)
    else: pdos= None

    pdosaux = np.zeros((nawf,ene.size),dtype=float)

    v_kaux = np.real(np.abs(v_k)**2)

    taux = np.zeros((delta.shape[0],nawf),dtype=float)

    for e in range (ene.size):
        if smearing == 'gauss':
            taux = gaussian(ene[e],E_k,delta) 
        elif smearing == 'm-p':
            taux = metpax(ene[e],E_k,delta)
        for i in range(nawf):
                # adaptive Gaussian smearing
                pdosaux[i,e] += np.sum(taux*v_kaux[:,i,:])


    comm.Reduce(pdosaux,pdos,op=MPI.SUM)    

    if rank == 0:
        pdos /= float(nk1*nk2*nk3)
        pdos_sum = np.zeros(ene.size,dtype=float)
        for m in range(nawf):
            pdos_sum += pdos[m]
            f=open(os.path.join(inputpath,str(m)+'_pdosdk_'+str(ispin)+'.dat'),'w')
            for ne in range(ene.size):
                f.write('%.5f  %.5f \n' %(ene[ne],pdos[m,ne]))
            f.close()
        f=open(os.path.join(inputpath,'pdosdk_sum_'+str(ispin)+'.dat'),'w')
        for ne in range(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],pdos_sum[ne]))
        f.close()

    comm.Barrier()

    return
