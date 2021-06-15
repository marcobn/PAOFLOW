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
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

def do_band_curvature ( data_controller ):
    '''
    Calculate the Gradient of the k-space Hamiltonian, 'Hksp'
    Requires 'Hksp' and 'pksp'
    Yields 'd2Hksp'

    Arguments:
        None

    Returns:
        None
    '''

    from .do_d2Hd2k import do_d2Hd2k_ij
    import numpy as np
    from .perturb_split import perturb_split

    ary,attr = data_controller.data_dicts()
    bnd = attr['bnd']    
    nawf = attr['nawf']    
    E_k = ary['E_k']    

    # not really the inverse mass tensor..it's actually tksp
    # but we are calling it d2Ed2k for now to save memory.
    d2Ed2k,dvec_list = do_d2Hd2k_ij(ary['Hksp'],ary['Rfft'],attr['alat'],
                                    attr['npool'],ary['v_k'],
                                    bnd,ary['degen'])

    
    # d2Ed2k is only the 6 unique components of the curvature  
    # (inverse effective mass ) tensor. This is one to save memory.
    ij_ind = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]],dtype=int)
    E_temp = np.zeros((bnd,nawf),order="C")

    #----------------------
    # for d2E/d2k_ij
    #----------------------
    for ispin in range(d2Ed2k.shape[3]):
      for ik in range(d2Ed2k.shape[1]):          

        # tksp_ij = <psi|d2Hd2k_ij|psi>

        # ij component of second derivative of the energy is:
        # tksp_ij + sum_i( (pksp_i*pksp_j.T + pksp_j*pksp_i.T)/(E_i-E_j) )
        E_temp = ((E_k[ik,:,ispin]-E_k[ik,:,ispin][:,None])[:,:]).T
        E_temp[np.where(np.abs(E_temp)<1.e-5)]=np.inf

        for ij in range(ij_ind.shape[0]):
            ipol = ij_ind[ij,0]
            jpol = ij_ind[ij,1]

            # to avoid a zero in the denominator when E_i=E_j
            if dvec_list[ij][ispin][ik].size:
                v_k=dvec_list[ij][ispin][ik]
            else:
                v_k=ary['v_k'][ik,:,:,ispin]
            
            pksp_i=np.conj(v_k.T).dot(ary['dHksp'][ik,ipol,:,:,ispin]).dot(v_k)
            pksp_j=np.conj(v_k.T).dot(ary['dHksp'][ik,jpol,:,:,ispin]).dot(v_k)

            # this is where d2Ed2k becomes the actual curvature tensor
            d2Ed2k[ij,ik,:,ispin] += np.sum((((pksp_i*pksp_j.T +\
                                               pksp_j*pksp_i.T) / E_temp).real),axis=1)[:bnd]

    ary['d2Ed2k']=d2Ed2k
