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


def do_transport ( data_controller, temps, ene, velkp ):
  import numpy as np
  from mpi4py import MPI
  from os.path import join
  from numpy import linalg as LAN
  from .do_Boltz_tensors import do_Boltz_tensors_smearing
  from .do_Boltz_tensors import do_Boltz_tensors_no_smearing

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  siemen_conv = 6.9884
  temp_conv = 11604.52500617

  t_tensor = arrays['t_tensor']

  esize = ene.size

  spin_mult = (1. if attributes['nspin']==2 or attributes['dftSO'] else 2.)

  nspin = attributes['nspin']

  for ispin in range(nspin):

    op = attributes['opath']

    fnPF = join(op, 'PF_%d.dat'%ispin)
    fnkappa = join(op, 'kappa_%d.dat'%ispin)
    fnsigma = join(op, 'sigma_%d.dat'%ispin)
    fnsigmadk = join(op, 'sigmadk_%d.dat'%ispin)
    fnSeebeck = join(op, 'Seebeck_%d.dat'%ispin)

    fPF = open(fnPF, 'w')
    fkappa = open(fnkappa, 'w')
    fsigma = open(fnsigma, 'w')
    fsigmadk = open(fnsigmadk, 'w')
    fSeebeck = open(fnSeebeck, 'w')

    for temp in temps:

      itemp = temp/temp_conv

      if attributes['smearing'] != None:
        L0 = do_Boltz_tensors_smearing(data_controller, itemp, ene, velkp, ispin)

        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------
        sigma = None
        if rank == 0:
          # convert in units of 10*21 siemens m^-1 s^-1
          L0 *= spin_mult*siemen_conv/attributes['omega']

          # convert in units of siemens m^-1 s^-1
          sigma = L0*1.e21

          for i in range(esize):
            tup = (temp,ene[i],sigma[0,0,i],sigma[1,1,i],sigma[2,2,i],sigma[0,1,i],sigma[0,2,i],sigma[1,2,i])
            fsigmadk.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tup)
          sigma = None
        comm.Barrier()

      L0,L1,L2 = do_Boltz_tensors_no_smearing(data_controller, itemp, ene, velkp, ispin)

      sigma = None
      if rank == 0:
        #----------------------
        # Conductivity (in units of /Ohm/m/s)
        #----------------------

        # convert in units of 10*21 siemens m^-1 s^-1
        L0 *= spin_mult*siemen_conv/attributes['omega']

        sigma = L0*1.e21 # convert in units of siemens m^-1 s^-1

        for i in range(esize):
          tup = (temp,ene[i],sigma[0,0,i],sigma[1,1,i],sigma[2,2,i],sigma[0,1,i],sigma[0,2,i],sigma[1,2,i])
          fsigma.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tup)
        sigma = None
      comm.Barrier()

      S = None
      if rank == 0:
        #----------------------
        # Seebeck (in units of V/K)
        #----------------------

        # convert in units of 10^21 Amperes m^-1 s^-1
        L1 *= spin_mult*siemen_conv/(temp*attributes['omega'])

        S = np.zeros((3,3,esize), dtype=float)

        for n in range(esize):
          try:
            S[:,:,n] = -1.*LAN.inv(L0[:,:,n])*L1[:,:,n]
          except:
            from .report_exception import report_exception
            print('check t_tensor components - matrix cannot be singular')
            report_exception()
            comm.Abort()

        for i in range(esize):
          tup = (temp,ene[i],S[0,0,i],S[1,1,i],S[2,2,i],S[0,1,i],S[0,2,i],S[1,2,i])
          fSeebeck.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tup)
      comm.Barrier()

      PF = None
      kappa = None
      if rank == 0:
        #----------------------
        # Electron thermal conductivity ((in units of W/m/K/s)
        #----------------------

        # convert in units of kg m s^-4
        L2 *= spin_mult*siemen_conv*1.e15/(temp*attributes['omega'])

        kappa = np.zeros((3,3,esize),dtype=float)
        for n in range(esize):
          kappa[:,:,n] = (L2[:,:,n] - temp*L1[:,:,n]*LAN.inv(L0[:,:,n])*L1[:,:,n])*1.e6
        L1 = L2 = None

        for i in range(esize):
          tup = (temp,ene[i],kappa[0,0,i],kappa[1,1,i],kappa[2,2,i],kappa[0,1,i],kappa[0,2,i],kappa[1,2,i])
          fkappa.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tup)
        kappa = None

        PF = np.zeros((3,3,esize), dtype=float)
        for n in range(esize):
          PF[:,:,n] = np.dot(np.dot(S[:,:,n],L0[:,:,n]),S[:,:,n])*1.e21
        S = L0 = None

        for i in range(esize):
          tup = (temp,ene[i],PF[0,0,i],PF[1,1,i],PF[2,2,i],PF[0,1,i],PF[0,2,i],PF[1,2,i])
          fPF.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tup)
        PF = None
      comm.Barrier()

    fPF.close()
    fkappa.close()
    fsigma.close()
    fsigmadk.close()
    fSeebeck.close()
