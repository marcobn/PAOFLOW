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
import math, cmath
import sys, time


from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from communication import scatter_array
from smearing import *

from do_non_ortho import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_carrier_conc(eig,emin,emax,delta,netot,nawf,ispin,smearing,alat,a_vectors):
#  try:
    # DOS calculation with adaptive smearing

    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1001
    ene = np.arange(emin,emax,de,dtype=float)

    dos = np.zeros((ene.size),dtype=float)

    for ne in xrange(ene.size):

        dossum = np.zeros(1,dtype=float)

        comm.Barrier()
        aux = scatter_array(eig)
        auxd = scatter_array(delta)

        if smearing == 'gauss':
            # adaptive Gaussian smearing
            dosaux = np.sum(gaussian(ene[ne],aux,auxd))
        elif smearing == 'm-p':
            # adaptive Methfessel and Paxton smearing
            dosaux = np.sum(metpax(ene[ne],aux,auxd))

        comm.Barrier()
        comm.Reduce(dosaux,dossum,op=MPI.SUM)
        dos[ne] = dossum*float(nawf)/float(netot)

    if rank == 0:
        f=open('dosdk_'+str(ispin)+'.dat','w')
        for ne in xrange(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],dos[ne]))
        f.close()

    if rank == 0:

      vol_cm3 = (0.529177*alat)**3*np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))*\
                 (1.0e-8)**3

      temp = 300#0.025852

      en_k_sort  = np.sort(eig)

      nstates = int(np.around((eig.shape[0]/nawf)*(nelec/2.0),decimals=0))            
      e_fermi = 0.0
      dk = 1.0/float(eig.shape[0]/nawf)

      #isolate dos above fermi level 
      enec = ene[np.where(ene>e_fermi)]
      dosc = dos[np.where(ene>e_fermi)]

      #find lowest energy above fermi level with zero dos (states)
      top_cb =  np.amin(enec[np.where(dosc<dk)])

      
      eig_cond=eig[np.where(np.logical_and(eig<top_cb,eig>e_fermi))]
      
      conc = 0.0

      k_B = 8.6173303e-5

      conc = np.sum(1.0/(1.0+np.exp(eig_cond/(k_B*temp))))*dk


      print "valence band top (eV)  =",np.around(top_cb,decimals=3)
      print "states per unit cell   =",conc
      #scale to cm^-3
      conc /= vol_cm3
      print "carrrier conc (cm^-3)  =",conc


    return
#  except Exception as e:
#    raise e
