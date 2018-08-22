import numpy as np
import scipy.linalg as LAN

def do_perturb_split(op,rot_op,v_k,degen):
    if len(degen)==0:
        return op

    v_k_temp = np.copy(v_k)
    
    for i in range(len(degen)):

        # degenerate subspace indices upper and lower lim
        ll = degen[i][0]
        ul = degen[i][-1]+1

        # diagonalize in degenerate subspace
        _,weight = LAN.eigh(op[ll:ul,ll:ul])

        # linear combination of eigenvectors of H that diagonalize
        v_k_temp[:,ll:ul]=np.dot(v_k_temp[:,ll:ul],weight)

    # return new operator in non degenerate basis
    return np.dot(np.conj(v_k_temp.T),np.dot(rot_op,v_k_temp))




def do_perturb_split_twoop(rot_op1,rot_op2,v_k,degen):
    op1 = np.dot(np.conj(v_k.T),np.dot(rot_op1,v_k))
    if len(degen)==0:
        op2 = np.dot(np.conj(v_k.T),np.dot(rot_op2,v_k))
        return op1,op2

    v_k_temp = np.copy(v_k)
    
    for i in range(len(degen)):

        # degenerate subspace indices upper and lower lim
        ll = degen[i][0]
        ul = degen[i][-1]+1

        # diagonalize in degenerate subspace
        _,weight = LAN.eigh(op1[ll:ul,ll:ul])

        # linear combination of eigenvectors of H that diagonalize
        v_k_temp[:,ll:ul]=np.dot(v_k_temp[:,ll:ul],weight)

    # return new operator in non degenerate basis
    op1 = np.dot(np.conj(v_k_temp.T),np.dot(rot_op1,v_k_temp))
    op2 = np.dot(np.conj(v_k_temp.T),np.dot(rot_op2,v_k_temp))

    return op1,op2
