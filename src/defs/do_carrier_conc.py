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

def do_carrier_conc(eig,emin,emax,delta,netot,nawf,ispin,smearing,alat,a_vectors,nelec):
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

#    aux=None
#    auxd=None
#    dosaux=None
    if rank == 0:
        f=open('dosdk_'+str(ispin)+'.dat','w')
        for ne in xrange(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],dos[ne]))
        f.close()

    if rank == 0:

      vol_cm3 = (0.529177*alat)**3*np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))*\
                 (1.0e-8)**3

    

      en_k_sort  = np.sort(eig)

      nstates = int(np.around((eig.shape[0]/nawf)*(nelec/2.0),decimals=0))            
      e_fermi = en_k_sort[nstates-1]


      dk = 1.0/float(eig.shape[0]/nawf)

      #isolate dos above fermi level 

      enec = ene[np.where(ene>e_fermi)]
      dosc = dos[np.where(ene>e_fermi)]

      #find lowest energy above fermi level with zero dos (states)
      top_cb =  np.amin(enec[np.where(dosc<dk)])

      
      eig_cond=eig[np.where(np.logical_and(eig<top_cb,eig>e_fermi))]
      eig_cond_shape = eig_cond.shape[0]

      
      conc = np.zeros((240),order="C")

    else:
        conc=None
        eig_cond=None
        dk=None
        e_fermi=None
        eig_cond_shape=None

    dk = comm.bcast(dk)
    e_fermi = comm.bcast(e_fermi)
    eig_cond_shape = comm.bcast(eig_cond_shape)

    conc_aux = np.zeros((240),order="C")

    k_B = 8.6173303e-5

    temp = np.linspace(5,1200,240,endpoint=True)

    if eig_cond_shape!=0:
        eig_cond = scatter_array(eig_cond)

        for T in xrange(temp.shape[0]):
            conc_aux[T] = np.sum(1.0/(1.0+np.exp((eig_cond-e_fermi)/(k_B*temp[T]))))*dk

        comm.Reduce(conc_aux,conc)

    if rank == 0:
        #scale to cm^-3
        conc /= vol_cm3

        f=open('carrier_conc_'+str(ispin)+'.dat','w')
        for T in xrange(temp.shape[0]):
            f.write('% .5f  %.5e \n' %(temp[T],conc[T]))
        f.close()

      # print "valence band top (eV)  =",np.around(top_cb,decimals=3)
      # print "states per unit cell   =",conc
      # 
      
      # print "carrrier conc (cm^-3)  =",conc


    return
#  except Exception as e:
#    raise e
