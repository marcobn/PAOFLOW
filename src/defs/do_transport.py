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


def do_transport ( data_controller, temps, ene, velkp, save_L0=False ):
  import numpy as np
  from mpi4py import MPI
  from os.path import join
  from numpy import linalg as npl
  from .do_Boltz_tensors import do_Boltz_tensors

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attr = data_controller.data_dicts()

  esize = ene.size

  siemen_conv,temp_conv = 6.9884,11604.52500617

  nspin,t_tensor = attr['nspin'],arrays['t_tensor']

  spin_mult = 1. if nspin==2 or attr['dftSO'] else 2.

  # if we are doing carrier concentration we need to save
  # the sigma tensors for when we calculate the hall tensor
  if save_L0:
    arrays['boltz_L0']=np.zeros((nspin,temps.size,3,3,esize))

  for ispin in range(nspin):
    counter=0
    # Quick function opens file in output folder with name 's'
    ojf = lambda st,sp : open(join(attr['opath'],'%s_%d.dat'%(st,sp)),'w')

    fPF = ojf('PF', ispin)
    fkappa = ojf('kappa', ispin)
    fsigma = ojf('sigma', ispin)
    fSeebeck = ojf('Seebeck', ispin)
    fsigmadk = ojf('sigmadk', ispin) if attr['smearing']!=None else None

    for temp in temps:

      itemp = temp/temp_conv

      # Quick function to write Transport Formatted line to file
      wtup = lambda fn,tu : fn.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tu)

      # Quick function to get tuple elements to write
      gtup = lambda tu,i : (temp,ene[i],tu[0,0,i],tu[1,1,i],tu[2,2,i],tu[0,1,i],tu[0,2,i],tu[1,2,i])

      if attr['smearing'] is not None:
        L0,_,_ = do_Boltz_tensors(data_controller, attr['smearing'], itemp, ene, velkp, ispin)
        #----------------------
        # Conductivity (in units of 1.e21/Ohm/m/s)
        #----------------------
        if rank == 0:
          # convert in units of 10*21 siemens m^-1 s^-1
          L0 *= spin_mult*siemen_conv/attr['omega']
          # convert in units of siemens m^-1 s^-1
          sigma = L0*1.e21
          
          for i in range(esize):
            wtup(fsigmadk, gtup(sigma,i))
          sigma = None
        comm.Barrier()

      L0,L1,L2 = do_Boltz_tensors(data_controller, None, itemp, ene, velkp, ispin)
      if rank == 0:
        #----------------------
        # Conductivity (in units of /Ohm/m/s)
        # convert in units of 10*21 siemens m^-1 s^-1
        #----------------------
        L0 *= spin_mult*siemen_conv/attr['omega']
        sigma = L0*1.e21 # convert in units of siemens m^-1 s^-1

        for i in range(esize):
          wtup(fsigma, gtup(sigma,i))

        if save_L0:            
          arrays['boltz_L0'][ispin,counter,:]=np.copy(L0)
          counter+=1

        sigma = None

        #----------------------
        # Seebeck (in units of V/K)
        # convert in units of 10^21 Amperes m^-1 s^-1
        #----------------------
        L1 *= spin_mult*siemen_conv/(temp*attr['omega'])

        S = np.zeros((3,3,esize), dtype=float)

        for n in range(esize):
          try:
            S[:,:,n] = -1.*npl.inv(L0[:,:,n])*L1[:,:,n]
          except:
            from .report_exception import report_exception
            print('check t_tensor components - matrix cannot be singular')
            report_exception()
            comm.Abort()

        for i in range(esize):
          wtup(fSeebeck, gtup(S,i))

        #----------------------
        # Electron thermal conductivity ((in units of W/m/K/s)
        # convert in units of kg m s^-4
        #----------------------
        L2 *= spin_mult*siemen_conv*1.e15/(temp*attr['omega'])

        kappa = np.zeros((3,3,esize),dtype=float)
        for n in range(esize):
          kappa[:,:,n] = (L2[:,:,n] - temp*L1[:,:,n]*npl.inv(L0[:,:,n])*L1[:,:,n])*1.e6
        L1 = L2 = None

        for i in range(esize):
          wtup(fkappa, gtup(kappa,i))
        kappa = None

        PF = np.zeros((3,3,esize), dtype=float)
        for n in range(esize):
          PF[:,:,n] = np.dot(np.dot(S[:,:,n],L0[:,:,n]),S[:,:,n])*1.e21
        S = L0 = None

        for i in range(esize):
          wtup(fPF, gtup(PF,i))
        PF = None
      comm.Barrier()

    fPF.close()
    fkappa.close()
    fsigma.close()
    fSeebeck.close()
    if attr['smearing'] is not None:
      fsigmadk.close()


def do_carrier_conc( data_controller,velkp,ene,temps ):

  from mpi4py import MPI
  from .do_Boltz_tensors import do_Hall_tensors
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

  siemen_conv,temp_conv = 6.9884,11604.52500617  
  eminBT =np.amin(ene)
  emaxBT =np.amax(ene)


  en_buff=1.0

  #only count the states from emin-en_buff emin+en_buff
  E_k_mask = np.where(np.logical_and(E_k[:,:bnd,:]>=(eminBT-en_buff),E_k[:,:bnd,:]<=(emaxBT+en_buff)))
  E_k_range = np.ascontiguousarray(E_k[E_k_mask[0],E_k_mask[1],E_k_mask[2]])
  velkp_range = np.swapaxes(velkp,1,0)
  velkp_range = np.ascontiguousarray(velkp_range[:,E_k_mask[0],E_k_mask[1],E_k_mask[2]])
  d2Ed2k_range = np.ascontiguousarray(d2Ed2k[:,E_k_mask[0],E_k_mask[1],E_k_mask[2]])    

  # combine spin channels
  L0_temps = np.sum(ary['boltz_L0'],axis=0)/nspin


  cc_str = ''

  for temp in range(temps.shape[0]):

    itemp = temps[temp]/temp_conv   
    inv_L0=np.zeros_like(L0_temps[0])

    for n in range(ene.size):
      try:
        inv_L0[:,:,n] = npl.inv(L0_temps[temp,:,:,n])
      except:
        inv_L0[:,:,n]= 0.0


    inv_L0*=siemen_conv/attr['omega']

    # get sig_ijk
    sig_ijk = do_Hall_tensors( E_k_range,velkp_range,d2Ed2k_range,
                               kq_wght,itemp,ene)


    # writing the effective masses to file

    #scale factor
    sf = 11.055095423844927*0.003324201
    em_flat = d2Ed2k*sf
    em_flat = np.ascontiguousarray(np.transpose(em_flat,axes=(1,2,3,0)))
    em_flat = gather_full(em_flat,attr['npool'])
    
    if rank==0:
      nk,bnd,nspin,_ = em_flat.shape
      em_flat = np.ascontiguousarray(np.transpose(em_flat,axes=(2,0,1,3)))

      em_tens=np.zeros((nspin,nk,bnd,3,3))
      e_mass =np.zeros((nspin,nk,bnd,8))

      # build the effective mass tensors from the flattened version
      em_tens[...,0,0]=em_flat[...,0]
      em_tens[...,1,1]=em_flat[...,1]
      em_tens[...,2,2]=em_flat[...,2]
      em_tens[...,0,1]=em_flat[...,3]
      em_tens[...,1,0]=em_flat[...,3]
      em_tens[...,1,2]=em_flat[...,4]
      em_tens[...,2,1]=em_flat[...,4]
      em_tens[...,0,2]=em_flat[...,5]
      em_tens[...,2,0]=em_flat[...,5]
      # diagonalize
      for sp in range(nspin):
        for k in range(nk):
          for b in range(bnd):
            effm =  np.linalg.eigvals(np.linalg.inv(em_tens[sp,k,b]))
            e_mass[sp,k,b,[4,5,6]] = effm

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


    # calculate hall conductivity
    if rank==0:
      
        R_ijk = np.zeros_like(sig_ijk)
        
#        #return inverse L0 to base units
#        inv_L0 *= 1.0/omega
#        inv_L0 *= 6.9884 

        #scale by cell size 

        #multiply by spin multiplier
        sig_ijk *= spin_mult

        # loop over 27 components of R_ijk
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for a in range(3):
                        for b in range(3):
                            R_ijk[i,j,k,:] += -inv_L0[a,j,:]*sig_ijk[a,b,k,:]*inv_L0[i,b,:]


        # take average 
        for n in range(ene.size):
            pcp = 3.0/(R_ijk[1,2,0,n]+R_ijk[2,0,1,n]+R_ijk[0,1,2,n])
            pcpm = pcp/(omega*(5.29177249e-9**3))

            cc_str+='%8.2f % .5f % 9.5e % 9.5e \n' \
                %(temps[temp],ene[n],pcp,pcpm)

  if rank==0:
    with open(join(attr['opath'],'carrier_conc.dat'),'w') as ofo:
      ofo.write(cc_str)
        

        
