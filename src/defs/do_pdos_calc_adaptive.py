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


def do_pdos_calc_adaptive ( data_controller ):
#def do_pdos_calc_adaptive ( E_k,emin,emax,delta,v_k,nk1,nk2,nk3,nawf,ispin,smearing,inputpath ):
  from mpi4py import MPI
  from smearing import metpax, gaussian

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays = data_controller.data_arrays
  attributes = data_controller.data_attributes

  # PDoS Calculation with Gaussian Smearing
  emin = float(attributes['emin'])
  emax = float(attributes['emax'])
  de = (emax-emin)/1000.
  ene = np.arange(emin, emax, de)
  esize = ene.size

  nawf = attributes['nawf']

  for ispin in range(attributes['nspin']):

    E_k = np.real(arrays['E_k'][:,:,ispin])

    pdosaux = np.zeros((nawf,esize), dtype=float)

    v_kaux = np.real(np.abs(arrays['v_k'][:,:,:,ispin])**2)

    taux = np.zeros((arrays['deltakp'].shape[0],nawf), dtype=float)

### Parallelization wastes time and memory here!!! 
    for e in range (ene.size):
      if smearing == 'gauss':
        taux = gaussian(ene[e], E_k, arrays['deltakp'][:,:,ispin]) 
      elif smearing == 'm-p':
        taux = metpax(ene[e], E_k, arrays['deltakp'][:,:,ispin])
      for i in range(nawf):
          # Adaptive Gaussian Smearing
          pdosaux[i,e] += np.sum(taux*v_kaux[:,i,:])

    pdosaux /= float(attributes['nkpnts'])

    pdos = (np.zeros((nawf,esize), dtype=float) if rank==0 else None)

    comm.Reduce(pdosaux, pdos, op=MPI.SUM)


#### Decide how to write....
    if rank == 0:
      import os
#      pdos /= float(attributes['nkpnts'])
      pdos_sum = np.zeros(esize, dtype=float)
      for m in range(nawf):
        pdos_sum += pdos[m]
        f = open(os.path.join(attributes['inputpath'],str(m)+'_pdosdk_'+str(ispin)+'.dat'), 'w')
        for ne in range(esize):
          f.write('%.5f  %.5f\n' %(ene[ne],pdos[m,ne]))
        f.close()
      f = open(os.path.join(attributes['inputpath'],'pdosdk_sum_'+str(ispin)+'.dat'), 'w')
      for ne in range(esize):
        f.write('%.5f  %.5f\n' %(ene[ne],pdos_sum[ne]))
      f.close()
