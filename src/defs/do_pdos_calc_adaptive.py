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

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from communication import scatter_array
from smearing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_pdos_calc_adaptive(E_k,emin,emax,delta,v_k,nk1,nk2,nk3,nawf,ispin,smearing):
  try:
    # PDOS calculation with gaussian smearing
    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1000
    ene = np.arange(emin,emax,de,dtype=float)
    nktot = nk1*nk2*nk3
    # Load balancing
    ini_ik, end_ik = load_balancing(size,rank,nktot)

    nsize = end_ik-ini_ik
    pdos = np.zeros((nawf,ene.size),dtype=float)
    for m in range(nawf):

        comm.Barrier()
        E_kaux = scatter_array(E_k)
        v_kaux = scatter_array(v_k)
        auxd = scatter_array(delta)

        pdosaux = np.zeros((nawf,ene.size),dtype=float)
        pdossum = np.zeros((nawf,ene.size),dtype=float)
        for n in range (nsize):
            for i in range(nawf):
                if smearing == 'gauss':
                    # adaptive Gaussian smearing
                    pdosaux[i,:] += gaussian(ene,E_kaux[n,m],auxd[n,m])*(np.abs(v_kaux[n,i,m])**2)
                elif smearing == 'm-p':
                    # adaptive Methfessel and Paxton smearing
                    pdosaux[i,:] += metpax(ene,E_kaux[n,m],auxd[n,m])*(np.abs(v_kaux[n,i,m])**2)

        comm.Barrier()
        comm.Reduce(pdosaux,pdossum,op=MPI.SUM)
        pdos = pdos+pdossum

    pdos = pdos/float(nktot)

    if rank == 0:
        pdos_sum = np.zeros(ene.size,dtype=float)
        for m in range(nawf):
            pdos_sum += pdos[m]
            f=open(str(m)+'_pdosdk_'+str(ispin)+'.dat','w')
            for ne in range(ene.size):
                f.write('%.5f  %.5f \n' %(ene[ne],pdos[m,ne]))
            f.close()
        f=open('pdosdk_sum_'+str(ispin)+'.dat','w')
        for ne in range(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],pdos_sum[ne]))
        f.close()

    return
  except Exception as e:
    raise e
