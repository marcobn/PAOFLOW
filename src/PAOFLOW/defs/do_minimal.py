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

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def do_minimal(data_controller):
    
    arry,attr = data_controller.data_dicts()
    
    import numpy.random as rd
    from numpy import linalg as npl
    
    bnd = attr['bnd']
    nawf = attr['nawf']
    nspin = attr['nspin']
    nkpnts = attr['nkpnts']
    
    arry['Hks'] = np.reshape(arry['Hks'],(nawf,nawf,nkpnts,nspin))
    
    Hks = np.zeros((bnd,bnd,nkpnts,nspin), dtype=complex)
    for ik in range(nkpnts):
        for ispin in range(nspin):
            Sbd = np.zeros((nawf,nawf),dtype=complex)
            Sbdi = np.zeros((nawf,nawf),dtype=complex)
            S = sv = np.zeros((nawf,nawf),dtype=complex)
            e = se = np.zeros(nawf,dtype=float)
            e,S = npl.eigh(arry['Hks'][:,:,ik,ispin])
            S11 = S[:bnd,:bnd] + 1.0*rd.random(bnd)/10000.
            S21 = S[:bnd,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
            S12 = S21.T
            S22 = S[bnd:,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
            S22 = S22 + S21.T.dot(np.dot(npl.inv(S11),S12.T))
            Sbd[:bnd,:bnd] = 0.5*(S11+np.conj(S11.T))
            Sbd[bnd:,bnd:] = 0.5*(S22+np.conj(S22.T))
            Sbdi = npl.inv(np.dot(Sbd,np.conj(Sbd.T)))
            se,sv = npl.eigh(Sbdi)
            se = np.sqrt(se+0.0j)*np.identity(nawf,dtype=complex)
            Sbdi = sv.dot(se).dot(np.conj(sv).T)
            T = S.dot(np.conj(Sbd.T)).dot(Sbdi)
            Hbd = np.conj(T.T).dot(np.dot(arry['Hks'][:,:,ik,ispin],T))
            Hks[:,:,ik,ispin] = 0.5*(Hbd[:bnd,:bnd]+np.conj(Hbd[:bnd,:bnd].T))
            
    arry['Hks'] = Hks
    attr['nawf'] = bnd
    ashape = (attr['nawf'],attr['nawf'],attr['nk1'],attr['nk2'],attr['nk3'],attr['nspin'])
    arry['Hks'] = np.reshape(arry['Hks'], ashape)
    
def do_minimal2(data_controller):
    
    arry,attr = data_controller.data_dicts()
    
    import numpy.random as rd
    from numpy import linalg as npl
    
    bnd = attr['bnd']
    nawf = attr['nawf']
    nspin = attr['nspin']
    nkpnts = attr['nkpnts']
    
    
    Hks = np.zeros((nkpnts,bnd,bnd,nspin), dtype=complex)
    for ik in range(nkpnts):
        for ispin in range(nspin):
            Sbd = np.zeros((nawf,nawf),dtype=complex)
            Sbdi = np.zeros((nawf,nawf),dtype=complex)
            S = sv = np.zeros((nawf,nawf),dtype=complex)
            e = se = np.zeros(nawf,dtype=float)
            e,S = npl.eigh(arry['Hksp'][ik,:,:,ispin])
            S11 = S[:bnd,:bnd] + 1.0*rd.random(bnd)/10000.
            S21 = S[:bnd,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
            S12 = S21.T
            S22 = S[bnd:,bnd:] + 1.0*rd.random(nawf-bnd)/10000.
            S22 = S22 + S21.T.dot(np.dot(npl.inv(S11),S12.T))
            Sbd[:bnd,:bnd] = 0.5*(S11+np.conj(S11.T))
            Sbd[bnd:,bnd:] = 0.5*(S22+np.conj(S22.T))
            Sbdi = npl.inv(np.dot(Sbd,np.conj(Sbd.T)))
            se,sv = npl.eigh(Sbdi)
            se = np.sqrt(se+0.0j)*np.identity(nawf,dtype=complex)
            Sbdi = sv.dot(se).dot(np.conj(sv).T)
            T = S.dot(np.conj(Sbd.T)).dot(Sbdi)
            Hbd = np.conj(T.T).dot(np.dot(arry['Hksp'][ik,:,:,ispin],T))
            Hks[ik,:,:,ispin] = 0.5*(Hbd[:bnd,:bnd]+np.conj(Hbd[:bnd,:bnd].T))
            
    arry['Hksp'] = Hks
    attr['nawf'] = bnd
