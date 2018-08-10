


import numpy as np

def get_degeneracies(E_k,bnd):

    all_degen = []

    E_k_round = np.around(E_k,decimals=6)
    
    for ispin in range(E_k_round.shape[2]):

        by_spin =[]
        for ik in range(E_k_round.shape[0]):
            by_kp = []
            eV=  np.unique(E_k_round[ik,:,ispin][:-1]\
                               [np.isclose(E_k_round[ik,:,ispin][1:],
                                           E_k_round[ik,:,ispin][:-1],atol=1.e-6)])

            for i in range(len(eV)):
                inds= np.where(np.isclose(E_k_round[ik,:,ispin],eV[i],atol=1.e-6))[0]
                if len(inds)>1:# and np.all(inds < bnd):                    
                    by_kp.append(inds)

            by_spin.append(by_kp)


        all_degen.append(by_spin)

    return np.asarray(all_degen)
                    
