import numpy as np
from get_K_grid_fft import *
import scipy.linalg as LA
import read_qe
from scipy.special import factorial as fac


############################################################################################
############################################################################################
############################################################################################

def check(Hksp_s,Hksp_f,si_per_k,new_k_ind,orig_k_ind,phase_shifts,U,a_index,inv_flag,equiv_atom):

    nawf=Hksp_s.shape[1]
    # load for testing
    Hksp_f = np.load("kham_full.npy")
    Hksp_f = np.reshape(Hksp_f,(nawf,nawf,Hksp_s.shape[0]))
    Hksp_f = np.transpose(Hksp_f,axes=(2,0,1))

    bad_symop=np.ones(48,dtype=bool)
    good_symop=np.ones(48,dtype=bool)
    good=[]
    bad=[]

    st=0
    fn=27
    for j in range(Hksp_s.shape[0]):
        
        isym = si_per_k[j]
        nki  = new_k_ind[j]
        oki  = orig_k_ind[j]

        HP = Hksp_f[nki]
        THP=Hksp_s[nki]

        U_k     = get_U_k(kp[oki],phase_shifts[isym],a_index,U[isym])


        good.append(isym)
        if np.all(np.isclose(HP[st:fn,st:fn],THP[st:fn,st:fn],
                             rtol=1.e-4,atol=1.e-4)):
            bad.append(isym)   
        else:
            good_symop[isym]=False
#            continue

            print(j,isym)
            print(equiv_atom[isym])
            print(symop[isym])
            print()
            print("inv:",inv_flag[isym])
            print()
            print("*"*50)
            print("REAL "*10)
            print("*"*50)
            print()
            print(U_k[st:fn,st:fn])
            print()
#            print("ORIG")
#            print(H[st:fn,st:fn].real)
#            print()
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
#            print()
#            print("ORIG")
#            print(H[st:fn,st:fn].imag)
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
            raise SystemExit
    print(len(good)-kp.shape[0],len(bad)-kp.shape[0])

    isl=sym_list()
    eac=np.array([0,1,2,3,4])

    print()
    for i in range(48):
        if i in si_per_k:
            if not  good_symop[i]:
                print("BAD ",isl[i],np.all(equiv_atom[i]==eac))
    #            print(symop[i] @ np.ones(3))


    for i in range(48):            
        if i in si_per_k:
            if  good_symop[i]:
                print("GOOD",isl[i],np.all(equiv_atom[i]==eac))

############################################################################################
############################################################################################
############################################################################################

def sym_list():
    isl=[]
    isl.append("identity")
    isl.append("180 deg rotation - cart. axis [0,0,1]")        
    isl.append("180 deg rotation - cart. axis [0,1,0]")        
    isl.append("180 deg rotation - cart. axis [1,0,0]")        
    isl.append("180 deg rotation - cart. axis [1,1,0]")        
    isl.append("180 deg rotation - cart. axis [1,-1,0]")       
    isl.append(" 90 deg rotation - cart. axis [0,0,-1]")       
    isl.append(" 90 deg rotation - cart. axis [0,0,1]")        
    isl.append("180 deg rotation - cart. axis [1,0,1]")        
    isl.append("180 deg rotation - cart. axis [-1,0,1]")       
    isl.append(" 90 deg rotation - cart. axis [0,1,0]")        
    isl.append(" 90 deg rotation - cart. axis [0,-1,0]")       
    isl.append("180 deg rotation - cart. axis [0,1,1]")        
    isl.append("180 deg rotation - cart. axis [0,1,-1]")       
    isl.append(" 90 deg rotation - cart. axis [-1,0,0]")       
    isl.append(" 90 deg rotation - cart. axis [1,0,0]")        
    isl.append("120 deg rotation - cart. axis [-1,-1,-1]")     
    isl.append("120 deg rotation - cart. axis [-1,1,1]")       
    isl.append("120 deg rotation - cart. axis [1,1,-1]")       
    isl.append("120 deg rotation - cart. axis [1,-1,1]")       
    isl.append("120 deg rotation - cart. axis [1,1,1]")        
    isl.append("120 deg rotation - cart. axis [-1,1,-1]")      
    isl.append("120 deg rotation - cart. axis [1,-1,-1]")      
    isl.append("120 deg rotation - cart. axis [-1,-1,1]")      
    isl.append("inversion")                                    
    isl.append("inv. 180 deg rotation - cart. axis [0,0,1]")   
    isl.append("inv. 180 deg rotation - cart. axis [0,1,0]")   
    isl.append("inv. 180 deg rotation - cart. axis [1,0,0]")   
    isl.append("inv. 180 deg rotation - cart. axis [1,1,0]")   
    isl.append("inv. 180 deg rotation - cart. axis [1,-1,0]")  
    isl.append("inv.  90 deg rotation - cart. axis [0,0,-1]")  
    isl.append("inv.  90 deg rotation - cart. axis [0,0,1]")   
    isl.append("inv. 180 deg rotation - cart. axis [1,0,1]")   
    isl.append("inv. 180 deg rotation - cart. axis [-1,0,1]")  
    isl.append("inv.  90 deg rotation - cart. axis [0,1,0]")   
    isl.append("inv.  90 deg rotation - cart. axis [0,-1,0]")  
    isl.append("inv. 180 deg rotation - cart. axis [0,1,1]")   
    isl.append("inv. 180 deg rotation - cart. axis [0,1,-1]")  
    isl.append("inv.  90 deg rotation - cart. axis [-1,0,0]")  
    isl.append("inv.  90 deg rotation - cart. axis [1,0,0]")   
    isl.append("inv. 120 deg rotation - cart. axis [-1,-1,-1]")
    isl.append("inv. 120 deg rotation - cart. axis [-1,1,1]")  
    isl.append("inv. 120 deg rotation - cart. axis [1,1,-1]")  
    isl.append("inv. 120 deg rotation - cart. axis [1,-1,1]")  
    isl.append("inv. 120 deg rotation - cart. axis [1,1,1]")   
    isl.append("inv. 120 deg rotation - cart. axis [-1,1,-1]") 
    isl.append("inv. 120 deg rotation - cart. axis [1,-1,-1]") 
    isl.append("inv. 120 deg rotation - cart. axis [-1,-1,1]") 

    return isl

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
    d_mat=np.zeros((2*l+1,2*l+1),dtype=complex)
    # list of m and m'
    ms  = list(range(-l,l+1))
    mps = ms

    # find wigner_d matrix elements for each m and m'
    for m_i in range(len(ms)):
        for mp_i in range(len(mps)):
            m=ms[m_i]
            mp=mps[mp_i]
            w_max=l+mp+1
            dmm = 0.0

            out_sum = np.sqrt(fac(l+m)*fac(l-m)*fac(l+mp)*fac(l-mp))

            # loop over w for summation
            for w in range(0,w_max):
                # factorials in denominator must be positive
                df1  = l+mp-w
                df2  = l-m-w
                df3  = w
                df4  = w+m-mp
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

        # check if there is an inversion in the symop
        if not np.all(np.isclose(eul2mat(AL,BE,GA),symop[i])):
            inv_flag[i]=True
            AL,BE,GA = mat2eul(-symop[i])
            if not np.all(np.isclose(eul2mat(AL,BE,GA),-symop[i])):
                print("ERROR IN MAT2EUL!")
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

def convert_wigner_d(wigner):
    # get transformation from angular momentum number to 
    # chemistry orbital form for the angular momentum
    trans_l0,trans_l1,trans_l2,trans_l3 = get_trans()

    c_wigner_l0 = np.zeros_like(wigner[0])
    c_wigner_l1 = np.zeros_like(wigner[1])
    c_wigner_l2 = np.zeros_like(wigner[2])
    c_wigner_l3 = np.zeros_like(wigner[3])

    for i in range(wigner[0].shape[0]):
        c_wigner_l0[i] =  (trans_l0).dot(wigner[0][i]).dot(LA.inv(trans_l0))
        c_wigner_l1[i] =  (trans_l1).dot(wigner[1][i]).dot(LA.inv(trans_l1))
        c_wigner_l2[i] =  (trans_l2).dot(wigner[2][i]).dot(LA.inv(trans_l2))
        c_wigner_l3[i] =  (trans_l3).dot(wigner[3][i]).dot(LA.inv(trans_l3))

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

def find_equiv_k(kp,symop,full_grid):
    # find indices and symops that generate full grid H from wedge H
    orig_k_ind = []
    new_k_ind = []
    si_per_k = []
    counter = 0
    kp_track = []
    for k in range(kp.shape[0]):
        for sym in range(symop.shape[0]):
            #transform k -> k' with the sym op
            newk = ((symop[sym].dot(kp[k])+0.5)%1.0)-0.5

            # find index in the full grid where this k -> k' with this sym op
            nw = np.where(np.all(np.isclose(newk[None],full_grid,
                                            atol=1.e-6,rtol=1.e-6),axis=1))[0]
            if len(nw)!=0:
                if nw[0] in kp_track:
                    continue
                else:
                    kp_track.append(nw[0])
                    counter+=1
                    si_per_k.append(sym)
                    orig_k_ind.append(k)
                    new_k_ind.append(nw[0])

    #check to make sure we have all the k points accounted for
    if counter!=full_grid.shape[0]:
        print('NOT ALL KPOINTS ACCOUNTED FOR')
    else:
        pass

    new_k_ind=np.array(new_k_ind)
    orig_k_ind=np.array(orig_k_ind)
    si_per_k=np.array(si_per_k)

    return new_k_ind,orig_k_ind,si_per_k

############################################################################################
############################################################################################
############################################################################################

def find_equiv_k_mod(kp,symop,full_grid):
    # find indices and symops that generate full grid H from wedge H
    kp_track = []
    orig_k_ind = []
    new_k_ind = []
    si_per_k = []
    counter = 0

    for k in range(kp.shape[0]):
        for sym in range(symop.shape[0]):
            #transform k -> k' with the sym op
            newk = ((symop[sym].dot(kp[k])+0.5)%1.0)-0.5
            new_k= correct_roundoff(newk)

            # Find index in the full grid where this k -> k' with this sym op
            nw = np.where(np.all(np.isclose(newk[None],full_grid,atol=1.e-5,
                                            rtol=1.e-5),axis=1))[0]

            if len(nw)!=0:
                #check if we already have an equivilent k' for this k
                if nw[0] in kp_track:
                    continue
                else:
                    kp_track.append(nw[0])
                    counter+=1
                    si_per_k.append(sym)
                    orig_k_ind.append(k)
                    new_k_ind.append(nw[0])

    new_k_ind=np.array(new_k_ind)
    orig_k_ind=np.array(orig_k_ind)
    si_per_k=np.array(si_per_k)

    return new_k_ind,orig_k_ind,si_per_k

############################################################################################
############################################################################################
############################################################################################

def build_U_matrix(wigner,shells):
    # builds U from blocks 
    nawf = np.sum(2*shells+1)
    U=np.zeros((wigner[0].shape[0],nawf,nawf),dtype=complex)

    for i in range(wigner[0].shape[0]):
        blocks=[]
        #block diagonal transformation matrix of H(k)->H(k')
        for s in shells:
            blocks.append(wigner[s][i])
        U[i] = LA.block_diag(*blocks)
    return U

############################################################################################
############################################################################################
############################################################################################

def get_phase_shifts(atom_pos,symop,inv_flag,equiv_atom):
    # calculate phase shifts for U
    phase_shift=np.zeros((symop.shape[0],atom_pos.shape[0],3),dtype=float)
    for isym in range(symop.shape[0]):
        for p in range(atom_pos.shape[0]):
            p1 = equiv_atom[isym,p]

            if inv_flag[isym]:
                phase_shift[isym,p1] = (-symop[isym].T @ atom_pos[p])-atom_pos[p1]
            else:
                phase_shift[isym,p1] = ( symop[isym].T @ atom_pos[p])-atom_pos[p1]

    return phase_shift

############################################################################################
############################################################################################
############################################################################################

def get_U_k(k,shift,a_index,U):
    # add phase shift to U
    U_k=U*np.exp(2.0j*np.pi*np.dot(shift[a_index],k))
    return U_k

############################################################################################
############################################################################################
############################################################################################

def correct_roundoff(arr):
    #correct for round off
    arr[np.where(np.isclose(arr, 0.0))] =  0.0
    arr[np.where(np.isclose(arr, 1.0))] =  1.0
    arr[np.where(np.isclose(arr,-1.0))] = -1.0

    return arr

############################################################################################
############################################################################################
############################################################################################

def construct_reduced(red_k_ind,full_k_ind,si_per_k,symop,H):
    #check function to check if wedge H can be constructed from full grid H
    red_dim=np.unique(red_k_ind).shape[0]
    H_red = np.zeros((red_dim,H.shape[1],H.shape[2]),dtype=complex)
    for j in range(full_k_ind.shape[0]):
        if np.allclose(symop[si_per_k[j]],np.eye(3)):
            oki  = red_k_ind[j]
            nki  = full_k_ind[j]
            H_red[oki]=H[nki]

    return H_red

############################################################################################
############################################################################################
############################################################################################

def check_reduced(red_k_ind,full_k_ind,si_per_k,symop,H,H_wedge):
    #check function to check if wedge H can be constructed from full grid H
    H_red=construct_reduced(red_k_ind,full_k_ind,si_per_k,symop,H)

    if not np.allclose(H_red,H_wedge):
        print('FULL AND REDUCED KHAM NOT EQUAL!!!')
        raise SystemExit

############################################################################################
############################################################################################
############################################################################################

def check_symop(symop):
    # check function to make sure s.T==inv(s)
    for isym in range(symop.shape[0]):
        if not np.allclose(symop[isym] @ symop[isym].T,np.eye(3)):
            if not np.allclose(symop[isym] @ LA.inv(symop[isym]),np.eye(3)):
                if not np.allclose(symop[isym].T @ LA.inv(symop[isym])):
                    print('BAD SYMOP')
                    print(symop[isym])
                    raise SystemExit

############################################################################################
############################################################################################
############################################################################################

def wedge_to_grid(Hksp,U,a_index,phase_shifts,kp,new_k_ind,orig_k_ind,si_per_k,inv_flag,U_wyc):
    # generates full grid from k points in IBZ
    nawf     = Hksp.shape[1]
    nkp_grid = new_k_ind.shape[0]
    Hksp_s=np.zeros((nkp_grid,nawf,nawf),dtype=complex)

    for j in range(Hksp_s.shape[0]):
        isym = si_per_k[j]
        oki  = orig_k_ind[j]
        nki  = new_k_ind[j]

        H  = Hksp[oki]

        U_k     = get_U_k(kp[oki],phase_shifts[isym],a_index,U[isym])
        U_k_inv = LA.inv(U_k)

        #transformated H(k)
        THP = U_k @ H @ U_k_inv

        # swap equiv atoms if needed
        THP = U_wyc[isym] @ THP @ U_wyc[isym].T

        # invert if the symop has inversion
        if inv_flag[isym]:
            THP=np.conj(THP)

        Hksp_s[nki]=THP


    check(Hksp_s,Hksp_s,si_per_k,new_k_ind,orig_k_ind,
          phase_shifts,U,a_index,inv_flag,equiv_atom)

    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

def open_grid(Hksp,nk1,nk2,nk3,symop,atom_pos,shells,a_index,equiv_atom):

    nawf = Hksp.shape[1]

    # full grid
    full_grid,_,_,_ = get_K_grid_fft(nk1,nk2,nk3,np.eye(3))
    full_grid=full_grid.T

    # get index of k in wedge, index in full grid, 
    # and index of symop that transforms k to k'
    new_k_ind,orig_k_ind,si_per_k = find_equiv_k(kp,symop,full_grid)

    # get array with wigner_d rotation matrix for each symop
    # for each of the orbital angular momentum l=[0,1,2,3]
    wigner,inv_flag = get_wigner(symop)

    # convert the wigner_d into chemistry form for each symop
    wigner = convert_wigner_d(wigner)

    # get phase shifts from rotation symop
    phase_shifts     = get_phase_shifts(atom_pos,symop,inv_flag,equiv_atom)

    # build U and U_inv from blocks
    U = build_U_matrix(wigner    ,shells)

    # adds transformation to U that maps orbitals from
    # atom A to equivalent atom B atoms after symop
    U_wyc = map_equiv_atoms(a_index,equiv_atom)

    # transform H(k) -> H(k')
    Hksp_s = wedge_to_grid(Hksp,U,a_index,phase_shifts,kp,
                           new_k_ind,orig_k_ind,si_per_k,inv_flag,U_wyc)

    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

np.set_printoptions(precision=3,suppress=True,linewidth=160)

# path to data-file.xml
fp='wedge.save/'

# load from QE data file
nk1,nk2,nk3,b_vectors,a_vectors,kp,symop,nawf,equiv_atom = read_qe.read_new_QE_output_xml(fp,False,False)

# load for testing
Hksp = np.load("kham_wedge.npy")
Hksp = np.reshape(Hksp,(nawf,nawf,kp.shape[0]))
Hksp = np.transpose(Hksp,axes=(2,0,1))


atom_pos=np.array([[0.0,0.0,0.0],
                   [0.5,0.5,0.5],
                   [0.5,0.5,0.0],
                   [0.0,0.5,0.5],
                   [0.5,0.0,0.5],])



a_index = np.array([1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,])-1
shells=np.array([0,1,0,0,1,0,2,0,1,0,1,0,1])


Hksp_s = open_grid(Hksp,nk1,nk2,nk3,symop,atom_pos,shells,a_index,equiv_atom)

Hksp_s=np.ravel(np.transpose(Hksp_s,axes=(1,2,0)))
np.save("Hksp_s.npy",Hksp_s)







# for oki in range(Hksp_s.shape[0]):
#     new_k_ind,_,si_per_k=find_equiv_k_mod(full_grid[oki][None],symop,full_grid)
#     num_g=new_k_ind.shape[0]

#     for g in range(num_g):
#         nki  = new_k_ind[g]
#         isym = si_per_k[g]
#         HP = Hksp_f[nki]

#         U_k     = get_U_k(full_grid[oki],phase_shifts[isym],a_index,U[isym])
#         U_k_inv = LA.inv(U_k)

#         #transformated H(k)
#         THP = U_k @ HP @ U_k_inv

#         # invert if the symop has inversion
#         if inv_flag[isym]:
#             THP=np.conj(THP)

#         Hksp_s[oki]+=THP/num_g
                        
#     print(oki)

# Hksp_s=np.ravel(np.transpose(Hksp_s,axes=(1,2,0)))
# np.save("Hksp_s.npy",Hksp_s)

