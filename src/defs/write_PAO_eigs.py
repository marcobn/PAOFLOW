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
from scipy import linalg as LA
from numpy import linalg as LAN
import numpy as np
import os

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def write_PAO_eigs(Hks,Sks,read_S,ispin,evecs,inputpath):

    nawf,nawf,nkpnts,nspin = Hks.shape
    E_k = np.zeros((nkpnts,nawf),dtype=float)
    v_k = None
    if evecs:
        v_k = np.zeros((nkpnts,nawf,nawf),dtype=complex)

    for ik in xrange(nkpnts):
        if read_S:
            eigval,eigvec = LA.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik])
        else:
            eigval,eigvec = LAN.eigh(Hks[:,:,ik,ispin],UPLO='U')
        E_k[ik,:] = np.real(eigval)
        if evecs:
            v_k[ik,:,:] = eigvec

    if rank == 0:
        ipad = False
        if ipad:
            f=open(os.path.join(inputpath,'bands_'+str(ispin)+'.dat'),'w')
            for ik in xrange(nkpnts):
                for nb in xrange(nawf):
                    f.write('%3d  %.5f \n' %(ik,E_k[ik,nb]))
            f.close()
        else:
            f=open(os.path.join(inputpath,'bands_'+str(ispin)+'.dat'),'w')
            for ik in xrange(nkpnts):
                s="%d\t"%ik
                for  j in E_k[ik,:]:s += "%3.5f\t"%j
                s+="\n"
                f.write(s)
            f.close()

    return(E_k,v_k)
