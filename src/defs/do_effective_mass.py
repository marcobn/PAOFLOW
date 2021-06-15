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


def do_effective_mass( data_controller,ene ):

  from mpi4py import MPI
  import numpy as np
  from os.path import join
  from numpy import linalg as npl
  from .communication import gather_full
  from .get_K_grid_fft import get_K_grid_fft_crystal
  from os.path import join

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  ary,attr = data_controller.data_dicts()

  omega = attr['omega']
  bnd   = attr['bnd']
  nspin = attr['nspin']
  spin_mult = 1. if nspin==2 or attr['dftSO'] else 2.
  E_k   = ary['E_k']
  d2Ed2k= ary['d2Ed2k']
  kq_wght= ary['kq_wght']
  me = 9.1093837015e-31
  eminBT =np.amin(ene)
  emaxBT =np.amax(ene)

  en_buff=1.0

  #only count the states from emin-en_buff emin+en_buff
  E_k_mask = np.where(np.logical_and(E_k[:,:bnd,:]>=(eminBT-en_buff),E_k[:,:bnd,:]<=(emaxBT+en_buff)))
  E_k_range = np.ascontiguousarray(E_k[E_k_mask[0],E_k_mask[1],E_k_mask[2]])
  d2Ed2k_range = np.ascontiguousarray(d2Ed2k[:,E_k_mask[0],E_k_mask[1],E_k_mask[2]])

  write_masses=True
  if write_masses:
    # writing the effective masses to file
    #scale factor
    SI_conv = 0.036749302892341
    em_flat = d2Ed2k*SI_conv
    em_flat = np.ascontiguousarray(np.transpose(em_flat,axes=(1,2,3,0)))
    em_flat = gather_full(em_flat,attr['npool'])

    if rank==0:
      nk,bnd,nspin,_ = em_flat.shape
      em_flat = np.ascontiguousarray(np.transpose(em_flat,axes=(2,0,1,3)))

      em_tens=np.zeros((nspin,nk,bnd,3,3))
      e_mass =np.zeros((nspin,nk,bnd,11))

      # build the effective mass tensors from the flattened version
      em_tens[...,0,0]=em_flat[...,0]
      em_tens[...,1,1]=em_flat[...,1]
      em_tens[...,2,2]=em_flat[...,2]
      em_tens[...,0,1]=em_flat[...,3]
      em_tens[...,1,0]=em_flat[...,3]
      em_tens[...,0,2]=em_flat[...,4]
      em_tens[...,2,0]=em_flat[...,4]
      em_tens[...,1,2]=em_flat[...,5]
      em_tens[...,2,1]=em_flat[...,5]
      # diagonalize
      for sp in range(nspin):
        for k in range(nk):
          for b in range(bnd):
            effm =  np.linalg.eigvals(np.linalg.inv(em_tens[sp,k,b]))
            e_mass[sp,k,b,[4,5,6]] = 1/(em_flat[sp,k,b,[0,1,2]])

            if np.prod(effm)<0:
              dos_em = -np.prod(np.abs(effm))**(1.0/3.0)
            else:
              dos_em =  np.prod(np.abs(effm))**(1.0/3.0)

            e_mass[sp,k,b,7] = dos_em

    effm=dos_em=em_tens=em_flat=None

    E_k_temp=gather_full(E_k,attr['npool'])
    if rank==0:
      E_k_temp = np.transpose(E_k_temp,axes=(2,0,1))
      e_mass[...,3]  = E_k_temp[:,:,:attr['bnd']]
      e_mass[...,:3] = get_K_grid_fft_crystal(attr['nk1'],attr['nk2'],attr['nk3'])[None,:,None]

      for sp in range(nspin):
        fpath = join(attr['opath'],'effective_masses_%d.dat'%sp)
        with open(fpath,'w') as ofo:
          ofo.write('    k_1     k_2     k_3     E-E_f              m_1              m_2              m_3            m_dos\n')
          ofo.write('-'*101)
          ofo.write('\n')

          for sp in range(nspin):
            for k in range(nk):
              for b in range(bnd):
                ofo.write('% 4.4f % 4.4f % 4.4f % 9.4f % 16.4f % 16.4f % 16.4f % 16.4f\n'%tuple(e_mass[sp,k,b].tolist()))

    E_k_temp=None
