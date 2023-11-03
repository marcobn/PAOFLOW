#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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

# this version works only for non-magnetic or non-collienar calculations
def wave_function_site_projection(data_controller):
    import cmath
    import numpy as np
    from mpi4py import MPI
    from os.path import join
    from scipy import fftpack as FFT
    from .constants import ANGSTROM_AU
    from scipy.fftpack import fftshift
    from .communication import scatter_full,gather_full
    
    arry,attr = data_controller.data_dicts()

    tau = arry['tau'] / ANGSTROM_AU
    naw,v_k = arry['naw'],arry['v_k']   
    bands,k_index = arry['bands_proj'],attr['k_proj']   
    
    # sites to project the wave function
    site = np.copy(tau[:,0])

    do_spin_orbit = attr['do_spin_orbit']    
    nawf,dim = attr['nawf'],attr['dimension']

    for idb in range(len(bands)):
        bnd_idx = bands[idb]  # index of the band to be projected
        # open file
        f = open(join(attr['opath'],'site-projected-wave-function-'+str(bnd_idx)+'.dat'), 'w')
        for n in range(tau.shape[0]):

            # Do to the doubling of the Hamiltonian when SOC is included in the PAO Hamiltonian.
            if (do_spin_orbit):
                # creating masks to consirer only the n site.
                # seting up the nonzero parts of the mask and the wave-function
                idx= int(np.sum(naw[0:n])) # initial
                fdx= int(idx + naw[n])     # final
                s = int(nawf/2)

                usector_idx = np.arange(idx,fdx,dtype=int)
                dsector_idx = np.arange(idx+s,fdx+s,dtype=int)
                idx_list = list(np.append(usector_idx,dsector_idx))

                total=0
                total+=np.sum(np.absolute(np.square(v_k[k_index:k_index+1,idx_list,bnd_idx,0])))

            else: # no SOC or SOC form QE.
                # creating masks to consirer only the n site.
                # seting up the nonzero parts of the mask and the wave-function
                idx= int(np.sum(naw[0:n])) # initial
                fdx= int(idx + naw[n])     # final

                total=0
                total+=np.sum(np.absolute(np.square(v_k[k_index:k_index+1,idx:fdx,bnd_idx,0])))

            if(dim==3): # ploting for 3D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write (("%5.4f %5.4f %5.4f %5.4f \n") %(tau[n,0],tau[n,1],tau[n,2],total+0.0001))
            if(dim==2): # ploting for 2D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write (("%5.4f %5.4f %5.4f  \n") %(tau[n,0],tau[n,1],total+0.0001))
            if(dim==1): # ploting for 1D system
                # we sum a very small part 0.0001 for ploting purpose.
                f.write (("%5.4f %5.4f  \n") %(tau[n,2],total+0.0001))

        f.close()

