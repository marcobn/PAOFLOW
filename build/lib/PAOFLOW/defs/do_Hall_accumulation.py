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

from re import I
from tkinter import W
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
    
def intra_band ( data_controller, emin, emax, ne, delta, ipol, spol, Op1 ):
  from .perturb_split import perturb_split
  from .smearing import metpax, gaussian
  from .constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL
  
  
  arrays,attributes = data_controller.data_dicts()

  nawf = attributes['nawf']
  nspin = attributes['nspin']
  nktot = arrays['pksp'].shape[0]


  # Hall Magnetization Calculation with Gaussian Smearing
  ene = np.linspace(emin, emax, ne)
  
  nawf = attributes['nawf']

  v_kaux = np.zeros_like(arrays['v_k'])

  O1=np.zeros_like(arrays['pksp'])
  O2=np.zeros_like(arrays['pksp'])      
  
  for ik in range(nktot):
    for ispin in range(nspin):
      O1[ik,spol,:,:,ispin],O2[ik,ipol,:,:,ispin]= perturb_split(Op1[spol,:,:],arrays['dHksp'][ik,ipol,:,:,ispin],arrays['v_k'][ik,:,:,ispin], arrays['degen'][ispin][ik])
      
  for ispin in range(attributes['nspin']):


    E_k = np.real(arrays['E_k'][:,:,ispin])

    accaux = np.zeros((ne), dtype=float)

    for ik in range(nktot):
      
      v_kaux[ik,:,:,ispin]= O1[ik,spol,:,:,ispin]*O2[ik,ipol,:,:,ispin]

    if attributes['smearing'] != None :
      taux = np.zeros((arrays['deltakp'].shape[0],nawf), dtype=float)
      
    for n in range (ne):
      # Adaptive Gaussian Smearing
      if attributes['smearing'] == 'gauss':
        taux = gaussian(ene[n], E_k, arrays['deltakp'][:,:,ispin]) 
      # Adaptive M-P smearing
      elif attributes['smearing'] == 'm-p':
        taux = metpax(ene[n], E_k, arrays['deltakp'][:,:,ispin])      
      elif attributes['smearing'] == None:
        taux = np.exp(-((ene[n]-E_k[:,:])/delta)**2)/np.sqrt(np.pi)

      accaux[n]+=np.sum(np.real(taux*np.diagonal(v_kaux[:,:,:,ispin],axis1=1,axis2=2)))

    acc = (np.zeros((ne), dtype=float) if rank==0 else None)

    comm.Reduce(accaux, acc, op=MPI.SUM)
    accaux = None

    if rank == 0:
      if(attributes['smearing']==None):
        acc /= (float(attributes['nkpnts'])*np.sqrt(np.pi)*delta)
      else:
        acc /= (float(attributes['nkpnts']))

    cart_indices = (str(LL[ipol]),str(LL[spol]),str(ispin))

    facc = 'accumulation_%s%s_%s.dat'%cart_indices
    data_controller.write_file_row_col(facc, ene, acc)

