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

# this version works only for non-magnetic or non-collienar calculations
def wave_function_site_projection(data_controller):
    from scipy import fftpack as FFT
    import numpy as np
    import cmath
    import os, sys, time
    from mpi4py import MPI
    from scipy.fftpack import fftshift
    from constants import ANGSTROM_AU
    from communication import scatter_full,gather_full
    
    arrays,attributes = data_controller.data_dicts()

    naw   = arrays['naw']    
    v_k  = arrays['v_k']    
    tau   = arrays['tau'] / ANGSTROM_AU
    bands = arrays['bands_proj']    
    k_index = attributes['k_proj']    
    
    # sites to project the wave function
    site = np.zeros_like(tau[:,0])
    site[:]=tau[:,0]

    do_spin_orbit = attributes['do_spin_orbit']    
    nawf = attributes['nawf']   
    dim = attributes['dimension']

    for idx in xrange(bands.shape[0]):
        bnd_idx=bands[idx]  # index of the band to be projected
        # open file
        f=open(os.path.join(attributes['opath'],'site-projected-wave-function-'+str(bnd_idx)+'.dat'),'w')
        for n in xrange(tau.shape[0]):

            # creating masks to consirer only the n site.
            mask   = np.zeros_like(v_k[k_index:k_index+1,:,:,0])
            cs     = np.zeros_like(v_k[k_index:k_index+1,:,:,0])

            # seting up the nonzero parts of the mask and the wave-function
            idx= np.sum(naw[0:n]) # initial
            fdx= idx + naw[n]     # final

            # Do to the doubling of the Hamiltonian when SOC is included in the PAO Hamiltonian.
            if (do_spin_orbit):
                s=int(nawf/2)
                mask[:,idx:fdx,:]    =  complex(1.0, 1.0)
                mask[:,idx+s:fdx+s,:] = complex(1.0, 1.0)
            else: # no SOC or SOC form QE.
                mask[:,idx:fdx,:]=complex(1.0, 1.0)

            #
            cs[:,:,:] = np.multiply(mask[:,:,:],v_k[k_index:k_index+1,:,:,0])

            total=0
            total+=np.sum(np.absolute(np.square((cs[:,:,bnd_idx]))))

            if(dim==3): # ploting for 3D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write (("%5.4f %5.4f %5.4f %5.4f \n") %(tau[n,0],tau[n,1],tau[n,2],total+0.0001))
            if(dim==2): # ploting for 2D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write (("%5.4f %5.4f %5.4f  \n") %(tau[n,0],tau[n,1],total+0.0001))

        f.close()

    #print("Fuck",site.shape)
    #print("Fuck",naw,e_k.shape)
