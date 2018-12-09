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
    import cmath
    import numpy as np
    from mpi4py import MPI
    from os.path import join
    from scipy import fftpack as FFT
    from constants import ANGSTROM_AU
    from scipy.fftpack import fftshift
    from communication import scatter_full,gather_full
    
    arry,attr = data_controller.data_dicts()

    tau = arry['tau'] / ANGSTROM_AU
    naw,v_k = arry['naw'],array['v_k']   
    bands,k_index = arry['bands_proj'],attr['k_proj']   
    
    # sites to project the wave function
    site = np.copy(tau[:,0])

    do_spin_orbit = attr['do_spin_orbit']    
    nawf,dim = attr['nawf'],attr['dimension']

    for idx in range(bands.shape[0]):
        bnd_idx=bands[idx]  # index of the band to be projected
        # open file
        f=open(join(attr['opath'],'site-projected-wave-function-'+str(bnd_idx)+'.dat'), 'w')
        for n in range(tau.shape[0]):

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
