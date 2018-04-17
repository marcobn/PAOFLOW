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
from constants import *
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import load_balancing

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def add_ext_field(HRs,tau_wf,R,alat,Efield,Bfield,HubbardU):
    nawf,nawf,nk1,nk2,nk3,nspin = HRs.shape
    HRs = np.reshape(HRs,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')

    tau_wf /= ANGSTROM_AU
    alat /= ANGSTROM_AU

    if Efield.any() != 0.0:
        # Electric field
        for n in range(nawf):
            HRs[n,n,0,:] -= Efield.dot(tau_wf[n,:])

    if Bfield.any() != 0.0:
        if rank == 0: print('calculation in magnetic supercell not implemented')
        pass
    # Magnetic field in units of magnetic flux quantum (hc/e)
    #for i in xrange(nk1*nk2*nk3):
    #    for n in xrange(nawf):
    #        for m in xrange(nawf):
    #            arg = 0.5*np.dot((np.cross(Bfield,alat*R[i,:]+tau_wf[m,:])+np.cross(Bfield,tau_wf[n,:])),(alat*R[i,:]+tau_wf[m,:]-tau_wf[n,:]))
    #            HRs[n,m,i,:] *= np.exp(-np.pi*arg*1.j)

    if HubbardU.any() != 0:
        for n in range(nawf):
            HRs[n,n,0,:] -= HubbardU[n]/2.0

    HRs = np.reshape(HRs,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
    return(HRs)
