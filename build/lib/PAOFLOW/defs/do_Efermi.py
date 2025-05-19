#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
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
from .smearing import intmetpax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def E_Fermi ( Hksp, data_controller, parallel=False ):
  # Calculate the Fermi energy using a braketing algorithm

  arry,attr = data_controller.data_dicts()

  insulator = attr['insulator']
  nktot,nbnd = attr['nkpnts'],attr['bnd']
  nelec,dftSO = attr['nelec'],attr['dftSO']
  nawf,_,snktot,nspin = Hksp.shape
  eig = np.zeros((nawf,snktot,nspin))
  Hksp = Hksp.reshape((nawf,nawf,snktot,nspin), order='C')

  for ispin in range(nspin):
    for ik in range(snktot):
      eig[:,ik,ispin] = np.linalg.eigvalsh(Hksp[:,:,ik,ispin])

  if insulator:
    Efr = np.amax(eig[(nelec-1 if dftSO else nelec//2-1)])
    if parallel:
      Efm = np.zeros((1), dtype=float) if rank==0 else None
      comm.Reduce(Efr, Efm, op=MPI.MAX)
      return comm.bcast(Efm[0] if rank==0 else None)
    else:
      return Efr

  else:
    Elw = 1.0e+8
    Eup = -1.0e+8
    eps = 1.0e-10
    degauss = 0.01

    nmbnd = nbnd-1 if nbnd == nawf else nbnd
    for ispin in range(nspin):
      for kp in range(snktot):
        Elw = min(Elw,eig[0,kp,ispin])
        Eup = max(Elw,eig[nmbnd,kp,ispin])

    Eup = Eup + 2 * degauss
    Elw = Elw - 2 * degauss

    # bisection method
    fac = 1 if dftSO else 2
    sumkup_aux = fac*np.sum(intmetpax(eig[:nbnd,:,:],Eup,degauss))
    sumklw_aux = fac*np.sum(intmetpax(eig[:nbnd,:,:],Elw,degauss))

    if parallel:
      sumkup = np.zeros((1), dtype=float) if rank==0 else None
      sumklw = np.zeros((1), dtype=float) if rank==0 else None
      comm.Reduce(sumkup_aux, sumkup, op=MPI.SUM)
      comm.Reduce(sumklw_aux, sumklw, op=MPI.SUM)
      sumkup = comm.bcast(sumkup[0]/nktot if rank==0 else None)
      sumklw = comm.bcast(sumklw[0]/nktot if rank==0 else None)
    else:
      sumkup = sumkup_aux/nktot
      sumklw = sumklw_aux/nktot

    if (sumkup - nelec) < -eps or (sumklw - nelec) > eps:
      if rank == 0:
        print('Error: cannot bracket Ef')

    maxiter = 100
    for i in range(maxiter):

      Ef = (Eup + Elw)/2
      sumkmid_aux = fac*np.sum(intmetpax(eig[:,:,:],Ef,degauss))
      if parallel:
        sumkmid = np.zeros((1,), dtype=float) if rank==0 else None
        comm.Reduce(sumkmid_aux, sumkmid, op=MPI.SUM)
        sumkmid = comm.bcast(sumkmid[0]/nktot if rank==0 else None)
      else:
        sumkmid = sumkmid_aux/nktot

      if np.abs( sumkmid-nelec ) < eps:
        break
      elif sumkmid-nelec < -eps:
        Elw = Ef
      else:
        Eup = Ef

    return Ef
