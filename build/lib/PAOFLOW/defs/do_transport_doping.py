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

def do_transport ( data_controller, temps,emin,emax,ne,ene,velkp,a_imp,a_ac,a_pop,write_to_file ):
  import numpy as np
  import scipy.optimize as sp
  import scipy.integrate
  from mpi4py import MPI
  from os.path import join
  from numpy import linalg as npl
  #from .do_Boltz_tensors import do_Boltz_tensors_smearing
  from .do_Boltz_tensors import do_Boltz_tensors_no_smearing
  from .do_doping import calc_N
  from .do_doping import FD
  from .do_doping import solve_for_mu
  from .do_dos import do_dos
  from .do_dos import do_dos_adaptive

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attr = data_controller.data_dicts()
  siemen_conv,temp_conv,omega_conv = 6.9884,11604.52500617,1.481847093e-25

  nspin,t_tensor = attr['nspin'],arrays['t_tensor']
  nelec,omega = attr['nelec'],attr['omega']*omega_conv
  spin_mult = 1. if nspin==2 or attr['dftSO'] else 2.
  
  attr['delta'] = 0.01
  dos = arrays['dos']
  for ispin in range(nspin):

    # Quick function opens file in output folder with name 's'
    
    if write_to_file == True:
      ojf = lambda st,sp : open(join(attr['opath'],'%s_%d.dat'%(st,sp)),'w')
    
      fPF = ojf('PF', ispin)
      fkappa = ojf('kappa', ispin)
      fsigma = ojf('sigma', ispin)
      fSeebeck = ojf('Seebeck', ispin)
      fsigmadk = ojf('sigmadk', ispin) if attr['smearing']!=None else None
    margin = 9. * temps.max()
    mumin = ene.min() + margin
    mumax = ene.max() - margin
    nT = len(temps)
    doping = attr['doping_conc']
    mur = np.empty(nT)
    msize = mur.size
    Nc = np.empty(nT)
    N = nelec - doping * omega
    for iT,temp in enumerate(temps):

      itemp = temp/temp_conv

      wtup = lambda fn,tu : fn.write('%8.2f % .5f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tu)

      # Quick function to get tuple elements to write
      gtup = lambda tu,i : (temp,mur[iT],Nc[iT],tu[0,0,i],tu[1,1,i],tu[2,2,i],tu[0,1,i],tu[0,2,i],tu[1,2,i])

      if rank == 0:
        #dopingmin = calc_N(data_controller,ene, dos, mumax, temp,dosweight=2.) + nelec
        #dopingmin /= omega
        #dopingmax = calc_N(data_controller,ene, dos, mumin, temp,dosweight=2.) + nelec
        #dopingmax /= omega
        mur[iT] = solve_for_mu(ene,dos,N,temp,refine=False,try_center=False)

        for imu,mu in enumerate(mur):
          Nc[iT] = calc_N(ene, dos, mu, temp) + nelec
      mur[iT] = comm.bcast(mur[iT], root=0)

      if attr['smearing'] != None:
        L0 = do_Boltz_tensors_smearing(data_controller, itemp, mur[iT], velkp, ispin)

        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------
        if rank == 0:
          # convert in units of 10*21 siemens m^-1 s^-1
          L0 *= spin_mult*siemen_conv/attr['omega']

          # convert in units of siemens m^-1 s^-1
          sigma = L0*1.e21

          wtup(fsigmadk, gtup(sigma,0))
        comm.Barrier()
      L0,L1,L2 = do_Boltz_tensors_no_smearing(data_controller, itemp, mur[iT], velkp, ispin,a_imp,a_ac,a_pop)
      if rank == 0:
        #----------------------
        # Conductivity (in units of /Ohm/m/s)
        #----------------------

        # convert in units of 10*21 siemens m^-1 s^-1
        L0 *= spin_mult*siemen_conv/attr['omega']

        sigma = L0*1.e21 # convert in units of siemens m^-1 s^-1
        if write_to_file == True:
          wtup(fsigma, gtup(sigma,0))
      comm.Barrier()

      S = None
      if rank == 0:
        #----------------------
        # Seebeck (in units of V/K)
        #----------------------

        # convert in units of 10^21 Amperes m^-1 s^-1
        L1 *= spin_mult*siemen_conv/(temp*attr['omega'])

        S = np.zeros((3,3,1), dtype=float)

        try:
          S[:,:,0] = -1.*npl.inv(L0[:,:,0])*L1[:,:,0]
        except:
          from .report_exception import report_exception
          print('check t_tensor components - matrix cannot be singular')
          report_exception()
          comm.Abort()

        wtup(fSeebeck, gtup(S,0))
      comm.Barrier()

      PF = None
      kappa = None
      if rank == 0:
        #----------------------
        # Electron thermal conductivity ((in units of W/m/K/s)
        #----------------------

        # convert in units of kg m s^-4
        L2 *= spin_mult*siemen_conv*1.e15/(temp*attr['omega'])

        kappa = np.zeros((3,3,1),dtype=float)
        kappa[:,:,0] = (L2[:,:,0] - temp*L1[:,:,0]*npl.inv(L0[:,:,0])*L1[:,:,0])*1.e6
        L1 = L2 = None

        wtup(fkappa, gtup(kappa,0))
        kappa = None

        PF = np.zeros((3,3,1), dtype=float)
        PF[:,:,0] = np.dot(np.dot(S[:,:,0],L0[:,:,0]),S[:,:,0])*1.e21
        S = L0 = None

        wtup(fPF, gtup(PF,0))
        PF = None
      comm.Barrier()

    fPF.close()
    fkappa.close()
    fsigma.close()
    if attr['smearing'] != None:
      fsigmadk.close()
    fSeebeck.close()

