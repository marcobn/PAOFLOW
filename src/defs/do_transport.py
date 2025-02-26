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


def do_transport ( data_controller, temps, ene, velkp, channels, weights, do_hall, write_to_file, save_tensors ):
  import numpy as np
  from os.path import join
  from numpy import linalg as npl
  from .do_Boltz_tensors import do_Boltz_tensors,do_Boltz_tensors_hall

  comm,rank = data_controller.comm,data_controller.rank
  arrays,attr = data_controller.data_dicts()

  esize = ene.size
  snktot = arrays['E_k'].shape[0]
  siemen_conv,temp_conv,hall_SI = 6.9884,11604.52500617,9.248931724005307e-13
  nspin,t_tensor = attr['nspin'],arrays['t_tensor']
  spin_mult = 1. if nspin==2 or attr['dftSO'] else 2.

  for ispin in range(nspin):
    # Quick function opens file in output folder with name 's'
    if write_to_file:
      ojf = lambda st,sp : open(join(attr['opath'],'%s_%d.dat'%(st,sp)), 'w')

      fsigma = ojf('sigma', ispin)
      fPF = ojf('PF', ispin)
      fkappa = ojf('kappa', ispin)
      fSeebeck = ojf('Seebeck', ispin)
      fsigmadk = ojf('sigmadk', ispin) if attr['smearing']!=None else None
      if do_hall:
        fhall = ojf('hall_trace', ispin)

    for iT,temp in enumerate(temps):

      itemp = temp/temp_conv

      # Quick function to write Transport Formatted line to file
      wtup = lambda fn,tu : fn.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tu)

      # Quick function to get tuple elements to write
      gtup = lambda tu,i : (temp,ene[i],tu[0,0,i],tu[1,1,i],tu[2,2,i],tu[0,1,i],tu[0,2,i],tu[1,2,i])

      if do_hall:
        wtup_hall = lambda fn,tu : fn.write('%8.2f % .5f % 9.5e \n'%tu)
        gtup_hall = lambda tu,i : (temp,ene[i],tu[i])

      if attr['smearing'] is not None:
        L0,_,_ = do_Boltz_tensors(data_controller, attr['smearing'], itemp, ene, velkp, ispin, channels, weights)
        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------
        if rank == 0:
          # convert in units of 10*21 siemens m^-1 s^-1
          L0 *= spin_mult*siemen_conv/attr['omega']
          # convert in units of siemens m^-1 s^-1
          sigma = L0*1.e21

          if write_to_file:   
            for i in range(esize):
              wtup(fsigmadk, gtup(sigma,i))
            sigma = None

        comm.Barrier()

      L0,L1,L2 = do_Boltz_tensors(data_controller, None, itemp, ene, velkp, ispin, channels, weights)

      if do_hall: 
        L0_hall = do_Boltz_tensors_hall(data_controller, None, itemp, ene, velkp, ispin, channels, weights)

      if rank == 0:
        #----------------------
        # Conductivity (in units of /Ohm/m/s)
        # convert in units of 10*21 siemens m^-1 s^-1
        #----------------------
        L0_unconverted = L0*spin_mult/attr['omega']
        L0 *= spin_mult*siemen_conv/attr['omega']
        sigma = L0*1.e21 # convert in units of siemens m^-1 s^-1
        if write_to_file:
          for i in range(esize):
            wtup(fsigma, gtup(sigma,i))
          sigma = None
        if save_tensors:
          arrays['sigma'] = sigma

        if do_hall:
          L0_hall *= spin_mult/(attr['omega'])
          R_hall = np.zeros((3,3,3,esize), dtype=float)
          R_hall_trace = np.zeros((esize), dtype=float)
          for n in range(esize):
            try:
              for r in range(3):
                R_hall[:,:,r,n] = npl.inv(L0_unconverted[:,:,n]) @ L0_hall[:,:,r,n] @ npl.inv(L0_unconverted[:,:,n])
                #----------------------   
                # The equivalent to the trace of the Hall tensor is an average
                # over the even permutations of [0, 1, 2].
                #----------------------  
              R_hall_trace[n] = (R_hall[0,1,2,n]+R_hall[2,0,1,n]+R_hall[1,2,0,n])*hall_SI/3

            except Exception as e:
              from .report_exception import report_exception
              print('check t_tensor components - matrix cannot be singular')
              report_exception()
              raise
          if write_to_file:
            for i in range(esize):
              wtup_hall(fhall, gtup_hall(R_hall_trace,i))
          if save_tensors:
            arrays['R_hall_trace'] = R_hall_trace

        #----------------------
        # Seebeck (in units of V/K)
        # convert in units of 10^21 Amperes m^-1 s^-1
        #----------------------
        L1 *= spin_mult*siemen_conv/(temp*attr['omega'])

        S = np.zeros((3,3,esize), dtype=float)

        for n in range(esize):
          try:
            S[:,:,n] = -1.*npl.inv(L0[:,:,n]) @ L1[:,:,n]
          except Exception as e:
            from .report_exception import report_exception
            print('check t_tensor components - matrix cannot be singular')
            report_exception()
            raise
        if write_to_file:
          for i in range(esize):
            wtup(fSeebeck, gtup(S,i))
        if save_tensors:
          arrays['S'] = S

        #----------------------
        # Electron thermal conductivity ((in units of W/m/K/s)
        # convert in units of kg m s^-4
        #----------------------
        L2 *= spin_mult*siemen_conv*1.e15/(temp*attr['omega'])

        kappa = np.zeros((3,3,esize),dtype=float)
        for n in range(esize):
          kappa[:,:,n] = (L2[:,:,n] - temp*L1[:,:,n] @ npl.inv(L0[:,:,n]) @ L1[:,:,n])*1.e6
        L1 = L2 = None
        if write_to_file:
          for i in range(esize):
            wtup(fkappa, gtup(kappa,i))
          kappa = None
        if save_tensors:
          arrays['kappa'] = kappa

        PF = np.zeros((3,3,esize), dtype=float)
        for n in range(esize):
          PF[:,:,n] = np.dot(np.dot(S[:,:,n],L0[:,:,n]),S[:,:,n])*1.e21
        S = L0 = None
        if write_to_file:
          for i in range(esize):
            wtup(fPF, gtup(PF,i))
          PF = None
      comm.Barrier()

    if write_to_file:
      fsigma.close()
      fPF.close()
      fkappa.close()
      fSeebeck.close()
      if attr['smearing'] is not None:
        fsigmadk.close()
      if do_hall:
        fhall.close()

    if save_tensors:
      data_controller.broadcast_single_array('sigma', dtype=float)
      data_controller.broadcast_single_array('S', dtype=float)
      data_controller.broadcast_single_array('kappa', dtype=float)
      if do_hall:
        data_controller.broadcast_single_array('R_hall_trace', dtype=float)
