#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
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
import numpy.polynomial.hermite as HERMITE
import math, cmath
import sys, time

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *

from do_non_ortho import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_dos_calc_adaptive(eig,emin,emax,delta,netot,nawf,ispin,smearing):
    # DOS calculation with gaussian smearing

    #emin = np.min(eig)-1.0
    #emax = np.max(eig)-shift/2.0
    emin = float(emin)
    emax = float(emax)
    de = (emax-emin)/1000
    ene = np.arange(emin,emax,de,dtype=float)

    # Load balancing
    ini_ie, end_ie = load_balancing(size,rank,netot)

    nsize = end_ie-ini_ie

    dos = np.zeros((ene.size),dtype=float)

    # for Methfessel and Paxton smearing
    nh = 5
    coeff = np.zeros(2*nh)
    coeff[0] = 1.
    for n in xrange(2,2*nh,2):
        m = n/2
        coeff[n] = (-1.)**m/(math.factorial(m)*4.0**m*np.sqrt(np.pi))

    for ne in xrange(ene.size):

        dossum = np.zeros(1,dtype=float)
        aux = np.zeros(nsize,dtype=float)
        auxd = np.zeros(nsize,dtype=float)

        comm.Barrier()
        comm.Scatter(eig,aux,root=0)
        comm.Scatter(delta,auxd,root=0)

        if smearing == 'gauss':
            # simple Gaussian smearing
            dosaux = np.sum(1.0/np.sqrt(np.pi)*np.exp(-((ene[ne]-aux)/auxd)**2)/auxd)
        elif smearing == 'm-p':
            # Methfessel and Paxton smearing
            dosaux = np.sum(HERMITE.hermval((ene[ne]-aux)/auxd,coeff)*np.exp(-((ene[ne]-aux)/auxd)**2)/auxd/np.sqrt(np.pi))

        comm.Barrier()
        comm.Reduce(dosaux,dossum,op=MPI.SUM)
        dos[ne] = dossum*float(nawf)/float(netot)

    if rank == 0:
        f=open('dosdk_'+str(ispin)+'.dat','w')
        for ne in xrange(ene.size):
            f.write('%.5f  %.5f \n' %(ene[ne],dos[ne]))
        f.close()

    return
