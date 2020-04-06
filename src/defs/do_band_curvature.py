


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
    ij_ind = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]],dtype=int)
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
            #pksp_i=ary['pksp'][ik,ipol,:,:,ispin]
            #pksp_j=ary['pksp'][ik,jpol,:,:,ispin]
            # this is where d2Ed2k becomes the actual curvature tensor
            d2Ed2k[ij,ik,:,ispin] += np.sum((((pksp_i*pksp_j.T +\
                                               pksp_j*pksp_i.T) / E_temp).real),axis=1)[:bnd]


    ary['d2Ed2k']=d2Ed2k
      
              
