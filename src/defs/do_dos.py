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


##### Not finished
def do_dos ( data_controller, emin=-10., emax=2. ):
#def do_dos_calc(eig,emin,emax,delta,netot,nawf,ispin,inputpath,npool):
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arrays,attributes = data_controller.data_dicts()

    bnd = attributes['bnd']
    netot = attributes['nkpnts']*bnd

    # DOS calculation with gaussian smearing
    de = (emax-emin)/1000
    ene = np.arange(emin, emax, de)
    esize = ene.size

    dos = np.zeros((esize), dtype=float) if rank == 0 else None

    dosaux = np.zeros((esize), order="C")

    for ispin in range(attributes['nspin']):

      for ne in range(esize):
          dosaux[ne] = np.sum(np.exp(-((ene[ne]-arrays['E_k'])/attributes['delta'])**2))

      comm.Reduce(dosaux,dos,op=MPI.SUM)

      dosaux = None

      if rank == 0:
        dos *= float(bnd)/(float(netot)*np.sqrt(np.pi)*attributes['delta'])
      else:
        dos = None

      fdos = 'dos_%s.dat'%str(ispin)
      data_controller.write_file_row_col(fdos, ene, dos)


def do_dos_adaptive ( data_controller ):
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
      if attributes['smearing'] == 'gauss':
        # adaptive Gaussian smearing
        dosaux[ne] = np.sum(gaussian(ene[ne],E_k,delta))

      elif attributes['smearing'] == 'm-p':
        # adaptive Methfessel and Paxton smearing
        dosaux[ne] = np.sum(metpax(ene[ne],E_k,delta))

    dosaux *= float(bnd)/float(netot)

    dos = (np.zeros((esize), dtype=float) if rank==0 else None)

    comm.Reduce(dosaux, dos, op=MPI.SUM)

    fdosdk = 'dosdk_%s.dat'%str(ispin)
    data_controller.write_file_row_col(fdosdk, ene, dos)
