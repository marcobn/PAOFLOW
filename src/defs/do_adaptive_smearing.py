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

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from communication import *
import numpy as np
from numpy import linalg as LAN
from load_balancing import *

def do_adaptive_smearing(pksp,nawf,nspin,alat,a_vectors,nk1,nk2,nk3,smearing):

    if smearing == None:
        return

    #----------------------
    # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
    #----------------------


    deltakp = np.zeros((pksp.shape[0],nawf,nspin),dtype=float)

    diag = np.diag_indices(nawf)

    omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
    dk = (8.*np.pi**3/omega/(nk1*nk2*nk3))**(1./3.)

    if smearing == 'gauss':
        afac = 0.7
    elif smearing == 'm-p':
        afac = 1.0        


    pksaux = np.ascontiguousarray(pksp[:,:,diag[0],diag[1]])

    deltakp = np.zeros((pksp.shape[0],nawf,nspin),dtype=float)
    deltakp2 = np.zeros((pksp.shape[0],nawf,nawf,nspin),dtype=float)


    for n in xrange(nawf):
        deltakp[:,n] = LAN.norm(np.real(pksaux[:,:,n]),axis=1)
        for m in xrange(nawf):
            deltakp2[:,n,m,:] = LAN.norm(pksaux[:,:,n,:] - pksaux[:,:,m,:],axis=1)

    pksaux=None
    deltakp*=afac*dk
    deltakp2*=afac*dk

    comm.Barrier()

    return deltakp,deltakp2
