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


def do_dos_calc_adaptive ( data_controller ):
#def do_dos_calc_adaptive(eig,emin,emax,delta,netot,nawf,ispin,smearing,inputpath):
  from smearing import gaussian, metpax
  from mpi4py import MPI
  import numpy as np

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  # DOS calculation with adaptive smearing
  emin = float(attributes['emin'])
  emax = float(attributes['emax'])
  de = (emax-emin)/1000.
  ene = np.arange(emin, emax, de)
  esize = ene.size

  bnd = attributes['bnd']
  netot = attributes['nkpnts']*bnd

  for ispin in range(attributes['nspin']):

    E_k = arrays['E_k'][:,:bnd,ispin].reshape(arrays['E_k'].shape[0]*bnd)
    delta = np.ravel(arrays['deltakp'][:,:bnd,ispin], order='C')

### Parallelization wastes time and memory here!!! 
    dosaux = np.zeros((esize), dtype=float)

    for ne in range(esize):
      if smearing == 'gauss':
        # adaptive Gaussian smearing
        dosaux[ne] = np.sum(gaussian(ene[ne],E_k,delta))
      elif smearing == 'm-p':
        # adaptive Methfessel and Paxton smearing
        dosaux[ne] = np.sum(metpax(ene[ne],E_k,delta))

    dosaux *= float(nawf)/float(netot)

    dos = (np.zeros((esize), dtype=float) if rank==0 else None)

    comm.Reduce(dosaux, dos, op=MPI.SUM)


## Decide how to write...
    if rank == 0:
      import os
      #dos *= float(nawf)/float(netot)
      f = open(os.path.join(attributes['inputpath'],'dosdk_'+str(ispin)+'.dat'), 'w')
      for ne in range(esize):
        f.write('%.5f  %.5f\n' %(ene[ne],dos[ne]))
      f.close()

  comm.Barrier()
