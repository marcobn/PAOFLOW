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

from ast import operator
from re import I
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def do_conductivity ( data_controller, emin, emax, ne, delta, ipol, jpol ):
  from .perturb_split import perturb_split
  from .smearing import metpax, gaussian
  from .constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL

  arrays,attributes = data_controller.data_dicts()

  nawf = attributes['nawf']
  nspin = attributes['nspin']
  nktot = arrays['pksp'].shape[0] #attributes['nkpnts']
  # conductivity calculation with gaussian smearing

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  # Conductivity Calculation with Gaussian Smearing
  emax = np.amin(np.array([attributes['shift'], emax]))
  ene = np.linspace(emin, emax, ne)

  nawf = attributes['nawf']
  

  O1=np.zeros_like(arrays['pksp'])
  O2=np.zeros_like(arrays['pksp'])  
  #O1=np.zeros_like((nktot,3,nawf,nawf,nspin),dtype=complex)
  #O2=np.zeros_like((nktot,3,nawf,nawf,nspin),dtype=complex)

  v_kaux = np.zeros_like(arrays['v_k'])

  #O1 = velocity operator with polarization ipol
  #O2 = velocity operator with polarization jpol
  for ik in range(nktot):
    for ispin in range(nspin):
      O1[ik,ipol,:,:,ispin],O2[ik,jpol,:,:,ispin]= perturb_split(arrays['dHksp'][ik,ipol,:,:,ispin],arrays['dHksp'][ik,jpol,:,:,ispin], arrays['v_k'][ik,:,:,ispin], arrays['degen'][ispin][ik])

  for ispin in range(attributes['nspin']):
    E_k = np.real(arrays['E_k'][:,:,ispin])
    condaux = np.zeros((ne), dtype=float)

    for ik in range(nktot):
      v_kaux[ik,:,:,ispin]= O1[ik,ipol,:,:,ispin]*O2[ik,jpol,:,:,ispin]
      
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
      
      condaux[n]+=np.sum(np.real(taux*np.diagonal(v_kaux[:,:,:,ispin],axis1=1,axis2=2)))

    cond = (np.zeros((ne), dtype=float) if rank==0 else None)

    comm.Reduce(condaux, cond, op=MPI.SUM)
    condaux = None

    if rank == 0:
      #for i in range(2):
     #   nksum+=arrays['pksp'].shape[0]
      if(attributes['smearing']==None):
        cond /= ((float(attributes['nkpnts']))*np.sqrt(np.pi)*delta)
      else:
        cond /= (float(attributes['nkpnts']))

    cart_indices = (str(LL[ipol]),str(LL[jpol]),str(ispin))

    fcond = 'cond_%s%s_%s.dat'%cart_indices
    data_controller.write_file_row_col(fcond, ene, cond)

