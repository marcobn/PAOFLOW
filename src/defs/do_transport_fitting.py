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

def do_transport ( data_controller, temps,emin,emax,ne,ene,velkp,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac,write_to_file):
  import numpy as np
  import scipy.optimize as sp
  import scipy.integrate
  from mpi4py import MPI
  from os.path import join
  from numpy import linalg as npl
  from .do_Boltz_tensors import do_Boltz_tensors_no_smearing

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  arrays,attr = data_controller.data_dicts()
  siemen_conv,temp_conv,omega_conv = 6.9884,11604.52500617,1.481847093e-25

  nspin,t_tensor = attr['nspin'],arrays['t_tensor']
  nelec,omega = attr['nelec'],attr['omega']*omega_conv
  spin_mult = 1. if nspin==2 or attr['dftSO'] else 2.

  if rank == 0:
    arrays['sigma'] = np.empty((3,3), dtype=float)
  for ispin in range(nspin):

    # Quick function opens file in output folder with name 's'
    if write_to_file == True:

      ojf = lambda st,sp : open(join(attr['opath'],'%s_%d.dat'%(st,sp)),'w')
      fsigma = ojf('sigma', ispin)

    for temp in temps:

      itemp = temp/temp_conv

      wtup = lambda fn,tu : fn.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tu)

      # Quick function to get tuple elements to write
      gtup = lambda tu,i : (temp,ene[i],tu[0,0,i],tu[1,1,i],tu[2,2,i],tu[0,1,i],tu[0,2,i],tu[1,2,i])

      if attr['smearing'] != None:
        L0 = do_Boltz_tensors_smearing(data_controller, itemp, ene, velkp, ispin,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac)

        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------
        if rank == 0:
          # convert in units of 10*21 siemens m^-1 s^-1
          L0 *= spin_mult*siemen_conv/attr['omega']

          # convert in units of siemens m^-1 s^-1
          sigma = L0*1.e21

       #   wtup(fsigmadk, gtup(sigma,0))
        comm.Barrier()
      L0,L1,L2 = do_Boltz_tensors_no_smearing(data_controller, itemp, ene, velkp, ispin,tau_dict,ms,a_imp,a_ac,a_pop,a_op,a_iv,a_pac)
      if rank == 0:
        #----------------------
        # Conductivity (in units of /Ohm/m/s)
        #----------------------

        # convert in units of 10*21 siemens m^-1 s^-1
        L0 *= spin_mult*siemen_conv/attr['omega']

        sigma = L0*1.e21 # convert in units of siemens m^-1 s^-1
        arrays['sigma'][:,:] = sigma[:,:,0]
        if write_to_file == True:
          wtup(fsigma, gtup(sigma,0))
      comm.Barrier()

    if write_to_file == True:
      fsigma.close()
    data_controller.broadcast_single_array('sigma', dtype=float)
