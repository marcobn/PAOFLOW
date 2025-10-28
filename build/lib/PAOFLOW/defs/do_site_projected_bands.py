#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

def site_projeted_bands(data_controller):

    import cmath
    import numpy as np
    from mpi4py import MPI
    from os.path import join
    from scipy import fftpack as FFT
    from .constants import ANGSTROM_AU
    from scipy.fftpack import fftshift
    from .communication import scatter_full,gather_full
    
    arry,attr = data_controller.data_dicts()

    
    nkpi   = arry['kq'].shape[1]
    mask   = np.zeros_like(arry['v_k'][:,:,:,0])
    cs     = np.zeros_like(arry['v_k'][:,:,:,0])

    s=0
    # Only used if ad-hoc SOC
    if (attr['do_spin_orbit']):
      s=int(attr['nawf']/2)

    nspin=arry['v_k'].shape[3]
   
    for ispin in range(nspin):
      f = open(join(attr['opath'],'site-projected-bands_'+str(ispin)+'.dat'), 'w')

      for i in range(arry['site_proj'].shape[0]):
    
        idx= np.sum(arry['naw'][0:arry['site_proj'][i]])
        fdx= idx + arry['naw'][arry['site_proj'][i]]
    
        mask[:,idx:fdx,:]     = complex(1.0, 1.0)
        mask[:,idx+s:fdx+s,:] = complex(1.0, 1.0) # Only used if ad-hoc SOC
    
        cs[:,:,:] = np.multiply(mask[:,:,:],arry['v_k'][:,:,:,ispin])

    
      for i in range(attr['nawf']):
        for k in range(nkpi):
          f.write(''.join(['%s %s %s\n'%(k,float(arry['E_k'][k,i]),np.sum(np.absolute(np.square((cs[k,:,i])))))]))
      f.close()
