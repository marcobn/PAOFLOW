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
import numpy as np
import cmath
import sys, time
import os
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from  communication import scatter_full

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def do_pdos_calc(E_k,emin,emax,delta,v_k,nk1,nk2,nk3,nawf,ispin,inputpath):

    # PDOS calculation with gaussian smearing
    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1000
    ene = np.arange(emin,emax,de,dtype=float)
    nktot = nk1*nk2*nk3
    # Load balancing

    if rank==0:
        pdos = np.zeros((nawf,ene.size),dtype=float)
    else: pdos= None

    pdosaux = np.zeros((nawf,ene.size),dtype=float)



    v_kaux = np.abs(v_k)**2

    for m in range(nawf):
        for e in range(ene.size):
            pdosaux[m,e] += np.sum(np.exp(-((ene[e]-E_k)/delta)**2)*(v_kaux[:,m,:]))

    comm.Barrier()
    comm.Reduce(pdosaux,pdos,op=MPI.SUM)    

    if rank == 0:
        pdos = pdos/float(nktot)*1.0/np.sqrt(np.pi)/delta
        pdos_sum = np.zeros(ene.size,dtype=float)
        for m in range(nawf):
            pdos_sum += pdos[m]
            f=open(os.path.join(inputpath,str(m)+'_pdos_'+str(ispin)+'.dat'),'w')
            for ne in range(ene.size):
                f.write('%.5f  %.5f \n' %(ene[ne],pdos[m,ne]))
            f.close()
        f=open(os.path.join(inputpath,'pdos_sum_'+str(ispin)+'.dat'),'w')
        for ne in range(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],pdos_sum[ne]))
        f.close()



    return
