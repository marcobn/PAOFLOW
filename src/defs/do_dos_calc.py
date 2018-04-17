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
from communication import scatter_full

from do_non_ortho import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_dos_calc(eig,emin,emax,delta,netot,nawf,ispin,inputpath,npool):
    # DOS calculation with gaussian smearing

    #emin = np.min(eig)-1.0
    #emax = np.max(eig)-shift/2.0
    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1000
    ene = np.arange(emin,emax,de,dtype=float)

    if rank==0:
        dos = np.zeros((ene.size),dtype=float)
    else: dos = None

    dosaux=np.zeros((ene.size),order="C")

    for ne in range(ene.size):
        dosaux[ne] = np.sum(np.exp(-((ene[ne]-eig)/delta)**2))

    comm.Barrier()
    comm.Reduce(dosaux,dos,op=MPI.SUM)

    dosaux = None

    if rank == 0:
        dos *= float(nawf)/float(netot)*1.0/np.sqrt(np.pi)/delta
        f=open(os.path.join(inputpath,'dos_'+str(ispin)+'.dat'),'w')
        for ne in range(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],dos[ne]))
        f.close()

    return
