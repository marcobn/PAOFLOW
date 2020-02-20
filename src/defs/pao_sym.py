
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

import numpy as np
import scipy.sparse as sprs
import scipy.linalg as LA
from scipy.special import factorial as fac
from tempfile import NamedTemporaryFile
import re
from .communication import scatter_full, gather_full,allgather_full,gather_scatter
from scipy.spatial.distance import cdist
from mpi4py import MPI
from .zero_pad import zero_pad
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def check(Hksp_s,si_per_k,new_k_ind,orig_k_ind,phase_shifts,U,a_index,inv_flag,equiv_atom,kp,symop,fg,isl,sym_TR):

    # compares saved full grid hamiltonain to one generated 
    # from wedge... only for debugging purposes
    nawf=Hksp_s.shape[1]
    # load for testing
    Hksp_f = np.load("ONLY_FOR_TESTING_kham.npy")
    Hksp_f = np.reshape(Hksp_f,(nawf,nawf,Hksp_s.shape[0]))
    Hksp_f = np.transpose(Hksp_f,axes=(2,0,1))

    print(np.allclose(Hksp_f,Hksp_s,atol=1.e-4,rtol=1.e-4))

    bad_symop=np.ones(symop.shape[0],dtype=bool)
    good_symop=np.ones(symop.shape[0],dtype=bool)
    good=[]
    bad=[]
    print(a_index)
    st=0
    fn=8
#    for j in range(20):        
    for j in range(Hksp_s.shape[0]):        
        isym = si_per_k[j]
        nki  = new_k_ind[j]
        oki  = orig_k_ind[j]

        HP = Hksp_f[nki]
        THP=Hksp_s[nki]

        U_k     = get_U_k(kp[oki],phase_shifts[isym],a_index,U[isym])

        good.append(isym)

        if np.all(np.isclose(HP,THP,rtol=1.e-4,atol=1.e-4)):
            bad.append(isym)   
        else:
 
            good_symop[isym]=False
            continue
            print(j,isym)
            print('old_k=',kp[oki])
            print('new_k=',fg[nki])
            print(equiv_atom[isym])
            print(symop[isym])
            print()
            print("inv:",inv_flag[isym])
            print()
            print("*"*50)
            print("REAL "*10)
            print("*"*50)
            print()
 #           print(U_k[st:fn,st:fn])
            print()
            print("CORRECT")
            print(HP[st:fn,st:fn].real)
            print("NEW")
            print(THP[st:fn,st:fn].real)
            print("EQUAL?")
            print(np.isclose(HP[st:fn,st:fn].real,THP[st:fn,st:fn].real,
                             rtol=1.e-4,atol=1.e-4))

            print("*"*50)
            print("IMAG "*10)
            print("*"*50)
            print("CORRECT")
            print(HP[st:fn,st:fn].imag)
            print("NEW")
            print(THP[st:fn,st:fn].imag)
            print("EQUAL?")
            print(np.isclose(HP[st:fn,st:fn].imag,THP[st:fn,st:fn].imag,
                             rtol=1.e-4,atol=1.e-4))


            print("*"*100)
            print("*"*100)
            print("*"*100)
#            raise SystemExit
    print(len(good)-kp.shape[0],len(bad)-kp.shape[0])


    print()
    for i in range(symop.shape[0]):
        if i in si_per_k:
            if not  good_symop[i]:
                print("%2d"%(i+1),"BAD ",isl[i],sym_TR[i])
            else:
                print("%2d"%(i+1),"GOOD",isl[i],sym_TR[i])




############################################################################################
############################################################################################
############################################################################################

def LPF(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    '''
    Pad frequency domain with zeroes, such that any relationship between
        aux[k] and aux[N-k] is preserved.

    Arguments:
        aux (ndarray): unpadded frequency domain data
        nk1 (int): current size of aux along axis 0
        nk2 (int): current size of aux along axis 1
        nk3 (int): current size of aux along axis 2
        nfft1 (int): number of zeroes to pad axis 0 by
        nfft1 (int): number of zeroes to pad axis 1 by
        nfft1 (int): number of zeroes to pad axis 2 by

    Returns:
        auxp3 (ndarray): padded frequency domain data
    '''

    # post-padding dimensions
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # halfway points
    sk1 = int((nk1+1)/2)
    sk2 = int((nk2+1)/2)
    sk3 = int((nk3+1)/2)
    # parities (even <-> p==1)
    p1 = (nk1 & 1)^1
    p2 = (nk2 & 1)^1
    p3 = (nk3 & 1)^1

    # accomodate nfft==0
    if nfft1 == 0:  p1 = 0
    if nfft2 == 0:  p2 = 0
    if nfft3 == 0:  p3 = 0

    # first dimension
    auxp1 = np.zeros((nk1,nk2,nk3p),dtype=complex)
    auxp1[:,:,:sk3+p3]=aux[:,:,:sk3+p3]
    auxp1[:,:,nfft3+sk3:]=aux[:,:,sk3:]
    # second dimension
    auxp2 = np.zeros((nk1,nk2p,nk3p),dtype=complex)
    auxp2[:,:sk2+p2,:]=auxp1[:,:sk2+p2,:]
    auxp2[:,nfft2+sk2:,:]=auxp1[:,sk2:,:]
    # third dimension
    auxp3 = np.zeros((nk1p,nk2p,nk3p),dtype=complex)
    auxp3[:sk1+p1,:,:]=auxp2[:sk1+p1,:,:]
    auxp3[nfft1+sk1:,:,:]=auxp2[sk1:,:,:]

    if rank==0:
        print(sk1)
    # halve Nyquist axes
    if p1:
        auxp3[ sk1-1,:,:] = 0.0
        auxp3[-sk1+1,:,:] = 0.0
        auxp3[sk1,:,:]    = 0.0
    if p2:
        auxp3[:, sk2-1,:] = 0.0
        auxp3[:,-sk2+1,:] = 0.0
        auxp3[:,-sk2,:]   = 0.0
    if p3:
        auxp3[:,:, sk3-1] = 0.0
        auxp3[:,:,-sk3+1] = 0.0
        auxp3[:,:,-sk3]   = 0.0
    return(auxp3)

############################################################################################
############################################################################################
############################################################################################

def down_samp(aux,nk1,nk2,nk3,nfft1,nfft2,nfft3):
    '''
    Pad frequency domain with zeroes, such that any relationship between
        aux[k] and aux[N-k] is preserved.

    Arguments:
        aux (ndarray): unpadded frequency domain data
        nk1 (int): current size of aux along axis 0
        nk2 (int): current size of aux along axis 1
        nk3 (int): current size of aux along axis 2
        nfft1 (int): number of zeroes to pad axis 0 by
        nfft1 (int): number of zeroes to pad axis 1 by
        nfft1 (int): number of zeroes to pad axis 2 by

    Returns:
        auxp3 (ndarray): padded frequency domain data
    '''

    nk1+=nfft1
    nk2+=nfft2
    nk3+=nfft3

    nfft1*=-1
    nfft2*=-1
    nfft3*=-1

    # post-padding dimensions
    nk1p = nfft1+nk1
    nk2p = nfft2+nk2
    nk3p = nfft3+nk3
    # halfway points
    sk1 = int((nk1+1)/2)
    sk2 = int((nk2+1)/2)
    sk3 = int((nk3+1)/2)
    # parities (even <-> p==1)
    p1 = (nk1 & 1)^1
    p2 = (nk2 & 1)^1
    p3 = (nk3 & 1)^1

    # accomodate nfft==0
    if nfft1 == 0:  p1 = 0
    if nfft2 == 0:  p2 = 0
    if nfft3 == 0:  p3 = 0

    # first dimension
    auxp1 = np.zeros((nk1p,nk2p,nk3),dtype=complex)
    auxp1[:,:,:sk3+p3] =aux[:,:,:sk3+p3]   
    auxp1[:,:,sk3:]    =aux[:,:,nfft3+sk3:]
    # second dimension
    auxp2 = np.zeros((nk1p,nk2,nk3),dtype=complex)
    auxp2[:,:sk2+p2,:]=auxp1[:,:sk2+p2,:]   
    auxp2[:,sk2:,:]   =auxp1[:,nfft2+sk2:,:]
    # third dimension
    auxp3 = np.zeros((nk1,nk2,nk3),dtype=complex)
    auxp3[:sk1+p1,:,:]=auxp2[:sk1+p1,:,:]   
    auxp3[sk1:,:,:]   =auxp2[nfft1+sk1:,:,:]


    # double Nyquist axes
    if p1:
        auxp3[ sk1,:,:] *= 2
    if p2:
        auxp3[:, sk2,:] *= 2
    if p3:
        auxp3[:,:, sk3] *= 2


    return auxp3

############################################################################################
############################################################################################
############################################################################################

def map_equiv_atoms(a_index,map_ind):
    # generate unitary tranformation for swapping
    # between atoms equivalent by symmetry op

    nawf = a_index.shape[0]
    U_wyc = np.zeros((map_ind.shape[0],nawf,nawf),dtype=complex)

    for isym in range(map_ind.shape[0]):
        map_U=np.zeros((nawf,nawf),dtype=complex)
        for i in range(map_ind[isym].shape[0]):
            fi=np.where(a_index==i)[0]
            si=np.where(a_index==map_ind[isym][i])[0]

            for j in range(si.shape[0]):
                map_U[fi[j],si[j]]=1

        U_wyc[isym]=map_U

    return U_wyc


############################################################################################
############################################################################################
############################################################################################

def get_U_TR(jchia):
    # operator that swaps J for -J

    J=0.5
    TR_J05=np.fliplr(np.eye(int(2*J+1))*((-1)**(J-np.arange(-J,J+1))))
    J=1.5
    TR_J15=np.fliplr(np.eye(int(2*J+1))*((-1)**(J-np.arange(-J,J+1))))
    J=2.5
    TR_J25=np.fliplr(np.eye(int(2*J+1))*((-1)**(J-np.arange(-J,J+1))))
    J=3.5
    TR_J35=np.fliplr(np.eye(int(2*J+1))*((-1)**(J-np.arange(-J,J+1))))


    TR=[TR_J05,TR_J15,TR_J25,TR_J35]


    nawf = int(np.sum(2*jchia+1)*2)

    U_TR=np.zeros((nawf,nawf),dtype=complex)

    blocks=[]

    flr_jchia=np.array([int(np.floor(a)) for a in jchia])

    #block diagonal transformation matrix of T @ H(k) @ T^-1
    for s in flr_jchia:
        blocks.append(TR[s])
    U_TR = LA.block_diag(*blocks)

    return U_TR

############################################################################################
############################################################################################
############################################################################################

def add_U_TR(U,sym_TR,jchia):
    # applies time inversion to symops as needed
    U_TR = get_U_TR(jchia)

    for isym in range(U.shape[0]):
        if sym_TR[isym]:
            U[isym]=  U_TR @ U[isym] 
    
    return U

############################################################################################
############################################################################################
############################################################################################


def get_trans():
    # get transformation from angular momentum number to 
    # chemistry orbital form for the angular momentum    
    rsh=np.sqrt(0.5)
    ish=1.0j*np.sqrt(0.5)

    trans_l0=np.array([[1.0]],dtype=complex)

    # to convert from m=[-1,0,1] to px,py,pz for l=1
    trans_l1=np.array([[ 0, 1, 0],
                       [-1, 0, 1],
                       [ 1, 0, 1],],dtype=complex)

    trans_l1[1]*=rsh
    trans_l1[2]*=ish

    # to convert from m=[-2,-1,0,1,2] to dx2,dzx,dzy,dx2-y2,dxy for l=2
    trans_l2=np.array([[ 0, 0, 1, 0, 0],
                       [ 0,-1, 0, 1, 0],
                       [ 0, 1, 0, 1, 0],
                       [ 1, 0, 0, 0, 1],
                       [-1, 0, 0, 0, 1],],dtype=complex)

    trans_l2[1]*=rsh
    trans_l2[2]*=ish
    trans_l2[3]*=rsh
    trans_l2[4]*=ish

    # to convert from m=[-3,-2,-1,0,1,2,3] to real combinations for l=3
    trans_l3=np.array([[ 0, 0, 0, 1, 0, 0, 0],
                       [ 0, 0,-1, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 1, 0],
                       [ 0,-1, 0, 0, 0, 1, 0],
                       [-1, 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0, 1],],dtype=complex)

    trans_l3[1]*=rsh
    trans_l3[2]*=ish
    trans_l3[3]*=rsh
    trans_l3[4]*=ish
    trans_l3[5]*=rsh
    trans_l3[6]*=ish

    return trans_l0,trans_l1,trans_l2,trans_l3

############################################################################################
############################################################################################
############################################################################################


def d_mat_l(AL,BE,GA,l):
    # gets wigner_d matrix for a given alpha,beta,gamma
    # that rotates basis from m' to m

    # list of m and m'
    ms  = np.arange(-l,l+1)
    mps = ms

    d_mat=np.zeros((ms.shape[0],ms.shape[0]),dtype=complex)

    # find wigner_d matrix elements for each m and m'
    for m_i in range(len(ms)):
        for mp_i in range(len(mps)):
            m=ms[m_i]
            mp=mps[mp_i]
            w_max=int(l+mp+1)
            dmm = 0.0

            out_sum = np.sqrt(fac(int(l+m))*fac(int(l-m))*fac(int(l+mp))*fac(int(l-mp)))

            # loop over w for summation
            for w in range(0,w_max):
                # factorials in denominator must be positive
                df1  = int(l+mp-w)
                df2  = int(l-m-w)
                df3  = int(w)
                df4  = int(w+m-mp)
                if df1<0 or df2<0 or df3<0 or df4<0: continue

                # for each w in summation over w
                tmp  = (-1)**w
                tmp /= fac(df1)*fac(df2)*fac(df3)*fac(df4) 
                tmp *= np.cos(BE/2.0)**(2*l-m+mp-2*w)
                tmp *= np.sin(BE/2.0)**(2*w-mp+m)
                
                dmm+=tmp

            d_mat[mp_i,m_i]=dmm*np.exp(-1.0j*AL*mp)*np.exp(-1.0j*GA*m)*out_sum
        
    return d_mat

############################################################################################
############################################################################################
############################################################################################

def get_wigner(symop):
    # gets wigner_d matrix associated with each symop
    wigner_l0=np.zeros((symop.shape[0],1,1),dtype=complex)
    wigner_l1=np.zeros((symop.shape[0],3,3),dtype=complex)
    wigner_l2=np.zeros((symop.shape[0],5,5),dtype=complex)
    wigner_l3=np.zeros((symop.shape[0],7,7),dtype=complex)
    #inversion flag
    inv_flag =np.zeros((symop.shape[0]),dtype=bool)

    for i in range(symop.shape[0]):
        # get euler angles alpha,beta,gamma from the symop
        AL,BE,GA =  mat2eul(symop[i])
        AL,BE,GA = np.deg2rad(np.around(np.rad2deg([AL,BE,GA]),decimals=0))

        # check if there is an inversion in the symop
        if not np.all(np.isclose(eul2mat(AL,BE,GA),symop[i])):
            inv_flag[i]=True
            AL,BE,GA = mat2eul(-symop[i])
            AL,BE,GA = np.deg2rad(np.around(np.rad2deg([AL,BE,GA]),decimals=0))
            if not np.all(np.isclose(eul2mat(AL,BE,GA),-symop[i])):
                print("ERROR IN MAT2EUL!")
                print(i+1)
                print(symop[i])
                raise SystemExit

        # wigner_d matrix for l=0
        wigner_l0[i]=d_mat_l(AL,BE,GA,0)
        # wigner_d matrix for l=1
        wigner_l1[i]=d_mat_l(AL,BE,GA,1)
        # wigner_d matrix for l=2
        wigner_l2[i]=d_mat_l(AL,BE,GA,2)
        # wigner_d matrix for l=3
        wigner_l3[i]=d_mat_l(AL,BE,GA,3)

    return [wigner_l0,wigner_l1,wigner_l2,wigner_l3,inv_flag],inv_flag

############################################################################################
############################################################################################
############################################################################################

def get_wigner_so(symop):
    # gets wigner_d matrix associated with each symop
    wigner_j05=np.zeros((symop.shape[0],2,2),dtype=complex)
    wigner_j15=np.zeros((symop.shape[0],4,4),dtype=complex)
    wigner_j25=np.zeros((symop.shape[0],6,6),dtype=complex)
    wigner_j35=np.zeros((symop.shape[0],8,8),dtype=complex)
    #inversion flag
    inv_flag =np.zeros((symop.shape[0]),dtype=bool)

    for i in range(symop.shape[0]):
        # get euler angles alpha,beta,gamma from the symop
        AL,BE,GA =  mat2eul(symop[i])
        AL,BE,GA = np.deg2rad(np.around(np.rad2deg([AL,BE,GA]),decimals=0))
        # check if there is an inversion in the symop
        if not np.all(np.isclose(correct_roundoff(eul2mat(AL,BE,GA)),
                                 symop[i],atol=1.e-3,rtol=1.e-2)):
            inv_flag[i]=True

            AL,BE,GA = mat2eul(-symop[i])
            AL,BE,GA = np.deg2rad(np.around(np.rad2deg([AL,BE,GA]),decimals=0))
            if not np.all(np.isclose(correct_roundoff(eul2mat(AL,BE,GA)),
                                     -symop[i],atol=1.e-3,rtol=1.e-2)):
                print("ERROR IN MAT2EUL!")
                print(i+1)
                print('RESULT')
                print(-eul2mat(AL,BE,GA))
                print('CORRECT')
                print(symop[i])
                raise SystemExit

        # wigner_d for l=0
        wigner_j05[i]=d_mat_l(AL,BE,GA,0.5)
        # wigner_d for l=1                                
        wigner_j15[i]=d_mat_l(AL,BE,GA,1.5)
        # wigner_d for l=2                                
        wigner_j25[i]=d_mat_l(AL,BE,GA,2.5)
        # wigner_d for l=3                                
        wigner_j35[i]=d_mat_l(AL,BE,GA,3.5)


    return [wigner_j05,wigner_j15,wigner_j25,wigner_j35,inv_flag],inv_flag

############################################################################################
############################################################################################
############################################################################################

def convert_wigner_d(wigner):
    # get transformation from angular momentum number to 
    # chemistry orbital form for the angular momentum

    trans_l0,trans_l1,trans_l2,trans_l3 = get_trans()

    c_wigner_l0 = np.zeros_like(wigner[0])
    c_wigner_l1 = np.zeros_like(wigner[1])
    c_wigner_l2 = np.zeros_like(wigner[2])
    c_wigner_l3 = np.zeros_like(wigner[3])

    for i in range(wigner[0].shape[0]):
        c_wigner_l0[i] = trans_l0 @ wigner[0][i] @ np.conj(trans_l0.T)
        c_wigner_l1[i] = trans_l1 @ wigner[1][i] @ np.conj(trans_l1.T)
        c_wigner_l2[i] = trans_l2 @ wigner[2][i] @ np.conj(trans_l2.T)
        c_wigner_l3[i] = trans_l3 @ wigner[3][i] @ np.conj(trans_l3.T)

    return [c_wigner_l0,c_wigner_l1,c_wigner_l2,c_wigner_l3]

############################################################################################
############################################################################################
############################################################################################

def eul2mat(ALPHA,BETA,GAMMA):
    # generates rotation matrix from euler angles
    C_1=np.cos(ALPHA)
    C_2=np.cos(BETA)
    C_3=np.cos(GAMMA)
    S_1=np.sin(ALPHA)
    S_2=np.sin(BETA)
    S_3=np.sin(GAMMA)

    euler =  np.array([[ C_1*C_2*C_3 -S_1*S_3, -C_1*C_2*S_3 -C_3*S_1,  C_1*S_2, ],
                       [ C_2*C_3*S_1 +C_1*S_3, -C_2*S_1*S_3 +C_1*C_3,  S_1*S_2, ],
                       [             -C_3*S_2,               S_2*S_3,  C_2    , ]])
    
    return euler

############################################################################################
############################################################################################
############################################################################################

def mat2eul(R):
    # finds euler angles from rotation matrix in ZYZ convention
    if R[2,2] < 1.0:
        if R[2,2] > -1.0:
            ALPHA =  np.arctan2(R[1,2], R[0,2])
            BETA  =  np.arccos( R[2,2])
            GAMMA =  np.arctan2(R[2,1],-R[2,0])
        else:
            ALPHA = -np.arctan2(R[1,0], R[1,1])
            BETA  =  np.pi
            GAMMA =  0.0
    else:
        ALPHA =       np.arctan2(R[1,0], R[1,1])
        BETA  =       0.0
        GAMMA =       0.0

    return np.array([ALPHA,BETA,GAMMA])

############################################################################################
############################################################################################
############################################################################################

def find_equiv_k(kp,symop,full_grid,sym_TR,check=True,include_self=False):
    # find indices and symops that generate full grid H from wedge H
    orig_k_ind = []
    new_k_ind = []
    si_per_k = []
    counter = 0
    kp_track = []
    kp = correct_roundoff(kp)


    full_grid_mask = np.copy(full_grid)

    for isym in range(symop.shape[0]):
        #transform k -> k' with the sym op
        if sym_TR[isym]:
            newk = ((((-symop[isym] @ (kp.T%1.0))%1.0)+0.5)%1.0)-0.5
        else:
            newk = (((( symop[isym] @ (kp.T%1.0))%1.0)+0.5)%1.0)-0.5
        newk = correct_roundoff(newk)
        newk[np.where(np.isclose(newk,0.5))]=-0.5
        newk[np.where(np.isclose(newk,-1.0))]=0.0
        newk[np.where(np.isclose(newk,1.0))]=0.0

        # find index in the full grid where this k -> k' with this sym op
        nw = np.where(np.isclose(cdist(newk.T,full_grid_mask),0.0,atol=1.e-6,rtol=1.e-6,))

        if not include_self:
            full_grid_mask[nw[1]]=np.ma.masked

        new_k_ind.extend(nw[1].tolist())
        si_per_k.extend([isym]*nw[1].shape[0])
        orig_k_ind.extend(nw[0].tolist())

    new_k_ind  = np.array(new_k_ind)
    orig_k_ind = np.array(orig_k_ind)
    si_per_k   = np.array(si_per_k)

    if not include_self:
        inds = np.unique(new_k_ind,return_index=True)

        new_k_ind  = new_k_ind[inds[1]]
        orig_k_ind = orig_k_ind[inds[1]]
        si_per_k   = si_per_k[inds[1]]
        counter=inds[1].shape[0]

    #check to make sure we have all the k points accounted for
    if counter!=full_grid.shape[0] and check:

        print('NOT ALL KPOINTS ACCOUNTED FOR')
        print(counter,full_grid.shape[0])
        print('missing k:')
        print(full_grid[np.setxor1d(new_k_ind,np.arange(full_grid.shape[0]))])
        raise SystemExit

    else:
        pass

    return new_k_ind,orig_k_ind,si_per_k

############################################################################################
############################################################################################
############################################################################################


def build_U_matrix(wigner,shells):
    # builds U from blocks 
    
    nawf = int(np.sum(2*shells+1))

    U=np.zeros((wigner[0].shape[0],nawf,nawf),dtype=complex)
    # for the so case...
    flr_shells=np.array([int(np.floor(a)) for a in shells])

    for i in range(wigner[0].shape[0]):
        blocks=[]
        #block diagonal transformation matrix of H(k)->H(k')
        for s in flr_shells:
            blocks.append(wigner[s][i])
        U[i] = LA.block_diag(*blocks)
    return U

############################################################################################
############################################################################################
############################################################################################

def get_phase_shifts(atom_pos,symop,equiv_atom):
    # calculate phase shifts for U
    phase_shift=np.zeros((symop.shape[0],atom_pos.shape[0],3),dtype=float)
    for isym in range(symop.shape[0]):
        for p in range(atom_pos.shape[0]):
            p1 = equiv_atom[isym,p]            
            phase_shift[isym,p1] =  ( symop[isym].T @ atom_pos[p])-atom_pos[p1]

    return correct_roundoff(np.around(phase_shift,decimals=2))

############################################################################################
############################################################################################
############################################################################################

def get_U_k(k,shift,a_index,U):
    # add phase shift to U
    U_k=U*np.exp(2.0j*np.pi*(shift[a_index] @ k))
    return U_k

############################################################################################
############################################################################################
############################################################################################

def correct_roundoff(arr,incl_hex=False,atol=1.e-8):
    #correct for round off
    arr[np.where(np.isclose(arr, 0.0,atol=atol))] =  0.0
    arr[np.where(np.isclose(arr, 1.0,atol=atol))] =  1.0
    arr[np.where(np.isclose(arr,-1.0,atol=atol))] = -1.0

    if incl_hex:
        sq3o2 = np.sqrt(3)/2.0
        arr[np.where(np.isclose(arr, sq3o2,atol=atol))] = sq3o2
        arr[np.where(np.isclose(arr,-sq3o2,atol=atol))] = -sq3o2
        arr[np.where(np.isclose(arr, 0.5,atol=atol))] =  0.5
        arr[np.where(np.isclose(arr,-0.5,atol=atol))] = -0.5

    return arr

############################################################################################
############################################################################################
############################################################################################

def get_full_grid(nk1,nk2,nk3):
  # generates full k grid in crystal fractional coords
  nktot=nk1*nk2*nk3
  b_vectors=np.eye(3)
  Kint = np.zeros((nktot,3), dtype=float)

  for i in range(nk1):
    for j in range(nk2):
      for k in range(nk3):
        n = k + j*nk3 + i*nk2*nk3
        Rx = float(i)/float(nk1)
        Ry = float(j)/float(nk2)
        Rz = float(k)/float(nk3)
        if Rx >= 0.5: Rx=Rx-1.0
        if Ry >= 0.5: Ry=Ry-1.0
        if Rz >= 0.5: Rz=Rz-1.0
        Rx -= int(Rx)
        Ry -= int(Ry)
        Rz -= int(Rz)

        Kint[n] = Rx*b_vectors[0,:]+Ry*b_vectors[1,:]+Rz*b_vectors[2,:]

  return Kint

############################################################################################
############################################################################################
############################################################################################

def get_inv_op(shells):
    # returns the inversion operator
    orb_index = np.hstack([[l]*(2*l+1) for l in shells])
    sign_inv=(-np.ones(orb_index.shape[0]))**orb_index

    return np.outer(sign_inv,sign_inv)

############################################################################################
############################################################################################
############################################################################################

def read_shell ( workpath,savedir,species,atoms,spin_orb=False):
    # reads in shelks from pseudo files
    from os.path import join,exists
    import numpy as np

    # Get Shells for each species
    sdict = {}
    jchid = {}
    jchia = None

    for s in species:
      sdict[s[0]],jchid[s[0]] = read_pseudopotential(join(workpath,savedir,s[1]))

    #double the l=0 if spin orbit
    if spin_orb:
        for s,p in sdict.items():
            tmp_list=[]
            for o in p:
                tmp_list.append(o)
                # if l=0 include it twice
                if o==0 or len(jchid[s])==0:
                    tmp_list.append(o)
            
            sdict[s] = np.array(tmp_list)

            # when using scalar rel pseido with spin orb..
            if len(jchid[s])==0:
                tmp=[]
                for o in sdict[s][::2]:
                    if o==0:
                        tmp.extend([0.5])
                    if o==1:
                        tmp.extend([0.5,1.5])
                    if o==2:
                        tmp.extend([1.5,2.5])
                    if o==3:
                        tmp.extend([2.5,3.5])
                jchid[s]=np.array(tmp)


        jchia = np.hstack([jchid[a] for a in atoms])


    # index of which orbitals belong to which atom in the basis
    a_index = np.hstack([[a]*np.sum((2*sdict[atoms[a]])+1) for a in range(len(atoms))])

    # value of l
    shell   = np.hstack([sdict[a] for a in atoms])
    return shell,a_index,jchia

############################################################################################
############################################################################################
############################################################################################

def read_pseudopotential ( fpp ):
  '''
  Reads a psuedopotential file to determine the included shells and occupations.

  Arguments:
      fnscf (string) - Filename of the pseudopotential, copied to the .save directory

  Returns:
      sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                       nl is a list of occupations at each site
      sh and nl are representative of one atom only
  '''

  import numpy as np
  import xml.etree.cElementTree as ET
  import re

  sh = []
  # fully rel case
  jchi=[]

  # clean xnl before reading
  with open(fpp) as ifo:
      temp_str=ifo.read()

  temp_str = re.sub('&',' ',temp_str)
  f = NamedTemporaryFile(mode='w',delete=True)
  f.write(temp_str)

  try:
      iterator_obj = ET.iterparse(f.name,events=('start','end'))
      iterator     = iter(iterator_obj)
      event,root   = next(iterator)

      for event,elem in iterator:        
          try:
              for i in elem.findall("PP_PSWFC/"):
                  sh.append(int(i.attrib['l']))
          except Exception as e:
              pass
          for i in elem.findall("PP_SPIN_ORB/"):
              try:
                  jchi.append(float(i.attrib["jchi"]))
              except: pass
              
      jchi = np.array(jchi)
      sh   = np.array(sh)

  except Exception as e:

      with open(fpp) as ifo:
          ifs=ifo.read()
      res=re.findall("(.*)\s*Wavefunction",ifs)[1:]      
      sh=np.array(list(map(int,list([x.split()[1] for x in res]))))


  return sh,jchi

############################################################################################
############################################################################################
############################################################################################

def enforce_t_rev(Hksp_s,nk1,nk2,nk3,spin_orb,U_inv,jchia):
    # enforce time reversal symmetry on H(k)
    nawf=Hksp_s.shape[1]
    
    Hksp_s= np.reshape(Hksp_s,(nk1,nk2,nk3,nawf,nawf),order="C")

    if spin_orb:
        U_TR = get_U_TR(jchia)

    for i in range(int(nk1/2)+1):
        for j in range(int(nk2/2)+1):
            for k in range(int(nk3/2)+1):
                iv= (nk1-i)%nk1
                jv= (nk2-j)%nk2
                kv= (nk3-k)%nk3
                if not spin_orb:
                    temp1 = np.conj(Hksp_s[i,j,k])
                    temp2 = np.conj(Hksp_s[iv,jv,kv])
                    Hksp_s[iv,jv,kv] = (Hksp_s[iv,jv,kv] + temp1)/2.0
                    Hksp_s[i,j,k]    = (Hksp_s[i,j,k]    + temp2)/2.0
                else:
                    temp1=np.conj(U_inv*(U_TR @ Hksp_s[i,j,k] @ np.conj(U_TR.T)))
#                    temp2=np.conj(U_inv*(U_TR @ Hksp_s[iv,jv,kv] @ np.conj(U_TR.T)))
#                    Hksp_s[iv,jv,kv] = (Hksp_s[iv,jv,kv] + temp1)/2.0
#                    Hksp_s[i,j,k]    = (Hksp_s[i,j,k]    + temp2)/2.0
                    Hksp_s[iv,jv,kv] = temp1


    Hksp_s= np.reshape(Hksp_s,(nk1*nk2*nk3,nawf,nawf),order="C")

    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

def apply_t_rev(Hksp,kp,spin_orb,U_inv,jchia):
    # apply time reversal operator to get H(-k) from H(k)

    # gamma only case
    if Hksp.shape[0]==1:
        return Hksp,kp
    new_kp_list=[]
    new_Hk_list=[]

    if spin_orb:
        U_TR = get_U_TR(jchia)

    for i in range(Hksp.shape[0]):
        new_kp= -kp[i]
        if not np.any(np.all(np.isclose(new_kp,kp),axis=1)):
            new_kp_list.append(new_kp)
            if not spin_orb:
                new_Hk_list.append(np.conj(Hksp[i]))
            else:
                new_Hk_list.append(np.conj(U_inv*(U_TR @ Hksp[i] @ np.conj(U_TR.T))))

    if len(new_kp_list)==0:
        return Hksp,kp
        
    kp=np.vstack([kp,np.array(new_kp_list)])
    Hksp=np.vstack([Hksp,np.array(new_Hk_list)])

    return Hksp,kp

############################################################################################
############################################################################################
############################################################################################

def enforce_hermaticity(Hksp):
    # enforce H(k) to be hermitian (it should be already)
    for k in range(Hksp.shape[0]):
        Hksp[k] = (Hksp[k] + np.conj(Hksp[k].T))/2.0

    return Hksp

############################################################################################
############################################################################################
############################################################################################

def add_U_wyc(U,U_wyc):
    # add operator that shifts orbital from one equivalent site to another
    for isym in range(U.shape[0]):
        U[isym]=U_wyc[isym] @ U[isym]
        U[isym]=correct_roundoff(U[isym],incl_hex=True)

    return U

############################################################################################
############################################################################################
############################################################################################

def wedge_to_grid(Hksp,U,a_index,phase_shifts,kp,new_k_ind,orig_k_ind,si_per_k,inv_flag,U_inv,sym_TR,npool):
    # generates full grid from k points in IBZ
    nawf     = Hksp.shape[1]



    fgm        = scatter_full(np.arange(si_per_k.shape[0],dtype=int),npool)
    new_k_ind  = scatter_full(new_k_ind,npool)
    orig_k_ind = scatter_full(orig_k_ind,npool)
    si_per_k   = scatter_full(si_per_k,npool)

    Hksp_s=np.zeros((new_k_ind.shape[0],nawf,nawf),dtype=complex)

    for j in range(new_k_ind.shape[0]):
        isym = si_per_k[j]
        oki  = orig_k_ind[j]
        nki  = np.where(new_k_ind[j]==fgm)[0]
        

        # if symop is identity
        if isym==0:
            Hksp_s[nki]=Hksp[oki]
            continue

        # other cases
        H  = Hksp[oki]

        # get k dependent U
        U_k     = get_U_k(kp[oki],phase_shifts[isym],a_index,U[isym])

        #transformated H(k)            
        THP = U_k @ H @ np.conj(U_k.T)

        # apply inversion operator if needed
        if inv_flag[isym]:
            THP*=U_inv

        # time inversion is anti-unitary
        if sym_TR[isym]:
            THP*= U_inv
            THP = np.conj(THP)

        Hksp_s[nki]=THP

    # make sure of hermiticity of each H(k)
    Hksp_s = enforce_hermaticity(Hksp_s)

    Hksp = None
    Hksp_s = gather_full(Hksp_s,npool)

    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

def open_grid(Hksp,full_grid,kp,symop,symop_cart,atom_pos,shells,a_index,equiv_atom,sym_info,sym_shift,nk1,nk2,nk3,spin_orb,sym_TR,jchia,mag_calc,symm_grid,thresh,max_iter,nelec,verbose):
    # calculates full H(k) grid from wedge
    npool=4


    nawf = Hksp.shape[1]

    # get inversion operator
    U_inv = get_inv_op(shells)

    # apply time reversal symmetry H(k) = H(-k)* where appropriate
    if not (spin_orb and mag_calc):
        Hksp,kp=apply_t_rev(Hksp,kp,spin_orb,U_inv,jchia)

    # get array with wigner_d rotation matrix for each symop
    # for each of the orbital angular momentum l=[0,1,2,3]
    if spin_orb:
        wigner,inv_flag = get_wigner_so(symop_cart)
    else:
        wigner,inv_flag = get_wigner(symop_cart)
        # convert the wigner_d into chemistry form for each symop
        wigner = convert_wigner_d(wigner)

    # get phase shifts from rotation symop
    phase_shifts = get_phase_shifts(atom_pos,symop,equiv_atom)

    # build U and U_inv from blocks
    if spin_orb:
        U = build_U_matrix(wigner,jchia)
    else:
        U = build_U_matrix(wigner,shells)

    # if any symop involve time inversion add the TR to the symop
    if np.any(sym_TR):
        U = add_U_TR(U,sym_TR,jchia)

    # adds transformation to U that maps orbitals from
    # atom A to equivalent atom B atoms after symop
    U_wyc = map_equiv_atoms(a_index,equiv_atom)

    # combine U_wyc and U
    U = add_U_wyc(U,U_wyc)

    # get index of k in wedge, index in full grid, 
    # and index of symop that transforms k to k'        
    new_k_ind,orig_k_ind,si_per_k = find_equiv_k(kp,symop,full_grid,sym_TR,check=True)

    # transform H(k) -> H(k')
    Hksp = wedge_to_grid(Hksp,U,a_index,phase_shifts,kp,
                         new_k_ind,orig_k_ind,si_per_k,inv_flag,U_inv,sym_TR,npool)

    if rank==0:
        # enforce time reversion where appropriate
        if not (spin_orb and mag_calc):
            Hksp = enforce_t_rev(Hksp,nk1,nk2,nk3,spin_orb,U_inv,jchia)        

    else:
        Hksp=np.zeros((full_grid.shape[0],nawf,nawf),dtype=complex)

    comm.Bcast(Hksp)

    if symm_grid:

        symop_inv=np.zeros_like(symop)
        for i in range(symop.shape[0]):
            symop_inv[i]=LA.inv(symop[i])

        nkl=[]
        partial_grid = scatter_full(full_grid,npool)
        for i in range(partial_grid.shape[0]):
            nkl.append(find_equiv_k(partial_grid[i][None],symop_inv,full_grid,sym_TR,check=False,include_self=True))
        nkl_no_interp=np.array(nkl)

        Hksp,tmax = symmetrize_grid(Hksp,U,a_index,phase_shifts,kp,inv_flag,U_inv,sym_TR,
                                    full_grid,symop,jchia,spin_orb,mag_calc,nk1,nk2,nk3,
                                    nkl_no_interp,partial_grid,npool)

        upscale1=int(0.25*nk1)
        upscale2=int(0.25*nk2)
        upscale3=int(0.25*nk3)
        

        nfft1=nk1+upscale1
        nfft2=nk2+upscale2
        nfft3=nk3+upscale3

        full_grid_interp = get_full_grid(nfft1,nfft2,nfft3)
        nkl=[]
        partial_grid_interp = scatter_full(full_grid_interp,npool)
        for i in range(partial_grid_interp.shape[0]):
            nkl.append(find_equiv_k(partial_grid_interp[i][None],symop_inv,full_grid_interp,sym_TR,check=False,include_self=True))
        nkl_interp=np.array(nkl)

        #max difference bewtween H(k) and H(k*)
        tmax=999999

        for i in range(int(max_iter*2)):
            st=time.time()
            add1=upscale1*((-1)**i)
            add2=upscale2*((-1)**i)
            add3=upscale3*((-1)**i)

            nfft1=nk1+add1
            nfft2=nk2+add2
            nfft3=nk3+add3

            if rank==0:
                Hksp = np.reshape(Hksp,(nk1*nk2*nk3,nawf*nawf))
                Hksp = np.ascontiguousarray(Hksp.T)

            Hksp = scatter_full(Hksp,npool)                    

            Hksp = np.reshape(Hksp,(Hksp.shape[0],nk1,nk2,nk3))
            HRs = np.fft.ifftn(Hksp,axes=(1,2,3))

            switch=True
            if switch==True:

                Hksp=None
                Hksp=np.zeros((HRs.shape[0],nfft1,nfft2,nfft3),dtype=complex)

                for m in range(Hksp.shape[0]):
                    
                    Hksp[m,:,:,:]=np.fft.fftn(zero_pad(HRs[m,:,:,:],nk1,nk2,nk3,add1,add2,add3))                            
#                     if not i%2:
#                         Hksp[m,:,:,:]=np.fft.fftn(zero_pad(HRs[m,:,:,:],nk1,nk2,nk3,add1,add2,add3))                        
#                     else:
# #                        Hksp[m,:,:,:]=np.fft.fftn(LPF(HRs[m,:,:,:],nk1,nk2,nk3,add1,add2,add3))                        
#                         Hksp[m,:,:,:]=np.fft.fftn(down_samp(HRs[m,:,:,:],nk1,nk2,nk3,add1,add2,add3))                        

                HRs  = None
                Hksp = np.reshape(Hksp,(Hksp.shape[0],nfft1*nfft2*nfft3))
                Hksp = gather_full(Hksp,npool)

                if rank==0:
                    Hksp = np.ascontiguousarray(Hksp.T)
                    Hksp = np.reshape(Hksp,(nfft1*nfft2*nfft3,nawf,nawf))
                else:
                    Hksp=np.zeros((nfft1*nfft2*nfft3,nawf,nawf),dtype=complex)

                comm.Bcast(Hksp)


                # Hksp=None
                # Hksp=np.zeros((HRs.shape[0],nfft1,nfft2,nfft3),dtype=complex)

                # for m in range(Hksp.shape[0]):
                #     Hksp[m,:,:,:]=np.fft.fftn(zero_pad(HRs[m,:,:,:],nk1,nk2,nk3,add1,add2,add3))

                # HRs  = None
                # Hksp = np.reshape(Hksp,(Hksp.shape[0],nfft1*nfft2*nfft3))
                # Hksp = gather_full(Hksp,npool)
                # if rank==0:
                #     Hksp = np.ascontiguousarray(Hksp.T)
                #     Hksp = np.reshape(Hksp,(nfft1*nfft2*nfft3,nawf,nawf))
                # else:
                #     Hksp=np.zeros((nfft1*nfft2*nfft3,nawf,nawf),dtype=complex)
                # comm.Bcast(Hksp)

            else:
                HRs=np.zeros((Hksp.shape[0],nfft1,nfft2,nfft3),dtype=complex)
                for m in range(Hksp.shape[0]):
                    HRs[m,:,:,:]=zero_pad(Hksp[m,:,:,:],nk1,nk2,nk3,add1,add2,add3)
                Hksp=None
                Hksp=np.fft.fftn(HRs,axes=(1,2,3))
                HRs=None
                Hksp = np.reshape(Hksp,(Hksp.shape[0],nfft1*nfft2*nfft3))
                Hksp = gather_scatter(Hksp,1,npool)
                Hksp = np.ascontiguousarray(Hksp.T)
                Hksp = gather_full(Hksp,npool)
                if rank==0:
                    Hksp = np.reshape(Hksp,(nfft1*nfft2*nfft3,nawf,nawf))            
                    print(Hksp[0,:8,:8].real)
                
                Hksp = np.reshape(Hksp,(nfft1*nfft2*nfft3,nawf,nawf))            

            # if it's the non interpolated grid
            if i%2:
                Hksp,_ = symmetrize_grid(Hksp,U,a_index,phase_shifts,kp,inv_flag,U_inv,
                                         sym_TR,full_grid,symop,jchia,spin_orb,mag_calc,
                                         nfft1,nfft2,nfft3,nkl_no_interp,partial_grid,npool)

            # if it's the interpolated grid
            else:
                Hksp,tmax = symmetrize_grid(Hksp,U,a_index,phase_shifts,kp,inv_flag,U_inv,
                                            sym_TR,full_grid_interp,symop,jchia,spin_orb,
                                            mag_calc,nfft1,nfft2,nfft3,nkl_interp,partial_grid_interp,npool)

            nk1+=add1
            nk2+=add2
            nk3+=add3



            if rank==0 and i%2 and verbose:
                print("Sym iter #%2d: %6.4e"%((i//2)+1,tmax[0]))

            # stop if we hit threshold
            if tmax<thresh:                
                if i%2 and i>=3:                   
                    break

    Hksp = reshift_efermi(Hksp,npool,nelec,spin_orb)                    
    
    # for debugging purposes
    try:
        if rank==0:
            check(Hksp,si_per_k,new_k_ind,orig_k_ind,phase_shifts,U,
                  a_index,inv_flag,equiv_atom,kp,symop,full_grid,sym_info,sym_TR)        
    except: pass

    return Hksp

############################################################################################
############################################################################################
############################################################################################

def open_grid_wrapper(data_controller):
#    np.set_printoptions(precision=3,suppress=True,linewidth=220)



    # wrapper function to unload everything from
    # data controller and do a few conversions

    data_arrays = data_controller.data_arrays
    data_attr   = data_controller.data_attributes
    alat        = data_attr['alat']
    nelec       = data_attr['nelec']
    nk1         = data_attr['nk1']
    nk2         = data_attr['nk2']
    nk3         = data_attr['nk3']
    spin_orb    = data_attr['dftSO']
    mag_calc    = data_attr['dftMAG']
    symm_grid   = data_attr['symmetrize']
    thresh      = data_attr['symm_thresh']
    max_iter    = data_attr['symm_max_iter']
    verbose     = data_attr['verbose']
    Hks         = data_arrays['Hks']
    atom_pos    = data_arrays['tau']/alat
    atom_lab    = data_arrays['atoms']
    equiv_atom  = data_arrays['equiv_atom']
    kp_red      = data_arrays['kpnts']
    b_vectors   = data_arrays['b_vectors']
    a_vectors   = data_arrays['a_vectors']
    sym_info    = data_arrays['sym_info']
    sym_shift   = data_arrays['sym_shift']
    symop       = data_arrays['sym_rot']
    sym_TR      = data_arrays['sym_TR']


    a_vectors   = a_vectors
    b_vectors   = b_vectors
    nspin       = Hks.shape[3]
    nawf        = Hks.shape[0]

    # convert atomic positions to crystal fractional coords
    conv=LA.inv(a_vectors)
    atom_pos = atom_pos @ conv
    atom_pos = correct_roundoff(atom_pos)
    atom_pos=np.around(atom_pos,decimals=6)

    # get symop crystal -> cartesian
    symop = correct_roundoff(symop)
    symop_cart = np.zeros_like(symop)
    inv_a_vectors = LA.inv(a_vectors)
    for isym in range(symop.shape[0]):
        symop_cart[isym] = (inv_a_vectors @ symop[isym] @ a_vectors)

    symop_cart = correct_roundoff(symop_cart,incl_hex=True,atol=1.e-6)

    # convert k points from cartesian to crystal fractional
    conv = LA.inv(b_vectors)
    conv = correct_roundoff(conv)
    kp_red = kp_red @ conv
    kp_red = correct_roundoff(kp_red)

    # get full grid in crystal fractional coords
    full_grid = get_full_grid(nk1,nk2,nk3)

    # get shells and atom indices for blocks of the hamiltonian
    shells,a_index,jchia = read_shell(data_attr['workpath'],data_attr['savedir'],
                                      data_arrays['species'],atom_lab,
                                      spin_orb=spin_orb)


    kp_red = correct_roundoff_kp(kp_red,full_grid)



    # correct small differences due to conversion


    # we wont need this for now
    if rank==0:
        if nspin==2:
            data_arrays['Hks'] = np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
        else:
            data_arrays['Hks'] = None
    else:
        data_arrays['Hks'] = None

    # expand grid from wedge
    for ispin in range(nspin):
        Hksp = np.ascontiguousarray(np.transpose(Hks,axes=(2,0,1,3))[:,:,:,ispin])

        Hksp = open_grid(Hksp,full_grid,kp_red,symop,symop_cart,atom_pos,
                         shells,a_index,equiv_atom,sym_info,sym_shift,
                         nk1,nk2,nk3,spin_orb,sym_TR,jchia,mag_calc,
                         symm_grid,thresh,max_iter,nelec,verbose)


        if rank==0:
            if nspin==2:
                data_arrays['Hks'][:,:,:,ispin]=np.ascontiguousarray(np.transpose(Hksp,axes=(1,2,0)))
            else:
                data_arrays['Hks']=np.ascontiguousarray(np.transpose(Hksp,axes=(1,2,0))[...,None])
            np.save("kham.npy",np.ascontiguousarray(np.transpose(Hksp,axes=(1,2,0))))
        else:
            Hksp=None


############################################################################################
############################################################################################
############################################################################################

def symmetrize(Hksp,U,a_index,phase_shifts,kp,new_k_ind,orig_k_ind,si_per_k,inv_flag,U_inv,sym_TR,full_grid,reverse=False):
    # generates full grid from k points in IBZ
    nawf     = Hksp.shape[1]
    Hksp_s=np.zeros((new_k_ind.shape[0],nawf,nawf),dtype=complex)

    for j in range(new_k_ind.shape[0]):
        isym = si_per_k[j]
        oki  = orig_k_ind[j]
        nki  = new_k_ind[j]

        # if symop is identity
        if isym==0:
            Hksp_s[j]=Hksp[nki]
            continue

        # other cases
        H  = Hksp[nki]

        # get k dependent U
        U_k     = get_U_k(full_grid[nki],phase_shifts[isym],a_index,U[isym])

        #transformated H(k)            
        if not reverse:
            THP = U_k @ H @ np.conj(U_k.T)
        else:
            THP = np.conj(U_k.T) @ H @ U_k

        # apply inversion operator if needed
        if inv_flag[isym]:
            THP*=U_inv

        # time inversion is anti-unitary
        if sym_TR[isym]:
            THP*= U_inv
            THP = np.conj(THP)

        Hksp_s[j]=THP
    
    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

def symmetrize_grid(Hksp,U,a_index,phase_shifts,kp,inv_flag,U_inv,sym_TR,full_grid,symop,jchia,spin_orb,mag_calc,nk1,nk2,nk3,nkl,partial_grid,npool):

    max_iter=1
    tmax=[]
    Hksp_d=np.zeros((partial_grid.shape[0],Hksp.shape[1],Hksp.shape[2]),dtype=complex)

    for i in range(partial_grid.shape[0]):
        new_k_ind,orig_k_ind,si_per_k=nkl[i]

        temp = symmetrize(Hksp,U,a_index,phase_shifts,kp,new_k_ind,
                          orig_k_ind,si_per_k,inv_flag,U_inv,sym_TR,full_grid)

        tmax.append(np.amax(np.abs(temp[0][None]-temp)))
        Hksp_d[i]=np.sum(temp,axis=0)/(temp.shape[0])


    # make sure of hermiticity of each H(k)
    Hksp_d = enforce_hermaticity(Hksp_d)

    if rank==0:
        Hksp=gather_full(Hksp_d,npool)
        if not (spin_orb and mag_calc):
             Hksp = enforce_t_rev(Hksp,nk1,nk2,nk3,spin_orb,U_inv,jchia)        

    else:
        gather_full(Hksp_d,npool)

    tm=np.array([np.amax(tmax)])
    tmax=np.copy(tm)
    comm.Reduce(tm,tmax,op=MPI.MAX)
    comm.Bcast(tmax)


    return Hksp,tmax

############################################################################################
############################################################################################
############################################################################################

def correct_roundoff_kp(kp,full_grid):
    kp_c =np.copy(kp)
    nw = np.where(np.isclose(cdist(kp_c,full_grid),0.0,atol=1.e-5,rtol=1.e-5,))    
    kp_c[nw[0]]=full_grid[nw[1]]
    
    return kp_c

############################################################################################
############################################################################################
############################################################################################

def reshift_efermi(Hksp,npool,nelec,spin_orb):

#    Hksp=scatter_full(Hksp,npool)
    eig = np.zeros((Hksp.shape[0],Hksp.shape[1]))

    dinds = np.diag_indices(Hksp.shape[1])

    for kp in range(Hksp.shape[0]):
        eig[kp] = LA.eigvalsh(Hksp[kp])

#    eigs=gather_full(eig,npool)
    

    nbnd=nelec
    if not spin_orb:
        nbnd=np.floor(nelec/2.0)
    
    if rank==0:
        nk=eig.shape[0]
        eig=np.sort(np.ravel(eig))
        efermi=eig[int(nk*nbnd-1)]
#        print(efermi)
    else:
        efermi=None

    efermi = comm.bcast(efermi)

    Hksp[:,dinds[0],dinds[1]] -= efermi
#    Hksp=gather_full(Hksp,npool)
    
    return Hksp
