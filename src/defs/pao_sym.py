import numpy as np
import scipy.linalg as LA
#import read_qe
from scipy.special import factorial as fac
from PAOFLOW.defs.get_K_grid_fft import get_K_grid_fft
import sys




def check(Hksp_s,si_per_k,new_k_ind,orig_k_ind,phase_shifts,U,a_index,inv_flag,equiv_atom,kp,symop,fg,isl):

    nawf=Hksp_s.shape[1]
    # load for testing
    Hksp_f = np.load("kham_full.npy")
    Hksp_f = np.reshape(Hksp_f,(nawf,nawf,Hksp_s.shape[0]))
    Hksp_f = np.transpose(Hksp_f,axes=(2,0,1))
    print(np.allclose(Hksp_f,Hksp_s,atol=1.e-4,rtol=1.e-4))

    bad_symop=np.ones(symop.shape[0],dtype=bool)
    good_symop=np.ones(symop.shape[0],dtype=bool)
    good=[]
    bad=[]
    print(a_index)
    st=0
    fn=10

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
            print(U_k[st:fn,st:fn])
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
                print("BAD ",isl[i])
    #            print(symop[i] @ np.ones(3))


    for i in range(symop.shape[0]):            
        if i in si_per_k:
            if  good_symop[i]:
                print("GOOD",isl[i])


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

def convert_wigner_d(wigner):
    # get transformation from angular momentum number to 
    # chemistry orbital form for the angular momentum
    trans_l0,trans_l1,trans_l2,trans_l3 = get_trans()

    c_wigner_l0 = np.zeros_like(wigner[0])
    c_wigner_l1 = np.zeros_like(wigner[1])
    c_wigner_l2 = np.zeros_like(wigner[2])
    c_wigner_l3 = np.zeros_like(wigner[3])

    inv_trans_l0 = LA.inv(trans_l0)
    inv_trans_l1 = LA.inv(trans_l1)
    inv_trans_l2 = LA.inv(trans_l2)
    inv_trans_l3 = LA.inv(trans_l3)

    for i in range(wigner[0].shape[0]):
        c_wigner_l0[i] = trans_l0 @ wigner[0][i] @ inv_trans_l0
        c_wigner_l1[i] = trans_l1 @ wigner[1][i] @ inv_trans_l1
        c_wigner_l2[i] = trans_l2 @ wigner[2][i] @ inv_trans_l2
        c_wigner_l3[i] = trans_l3 @ wigner[3][i] @ inv_trans_l3

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

def find_equiv_k(kp,symop,full_grid,sym_shift,check=True):
    # find indices and symops that generate full grid H from wedge H
    orig_k_ind = []
    new_k_ind = []
    si_per_k = []
    counter = 0
    kp_track = []
    kp = correct_roundoff(kp)
    for k in range(kp.shape[0]):
        for isym in range(symop.shape[0]):

            #transform k -> k' with the sym op
            
            newk = ((((symop[isym] @ (kp[k]%1.0))%1.0)+0.5)%1.0)-0.5
            newk = correct_roundoff(newk)
            newk[np.where(np.isclose(newk,0.5))]=-0.5
            newk[np.where(np.isclose(newk,-1.0))]=0.0
            newk[np.where(np.isclose(newk,1.0))]=0.0
            
            # find index in the full grid where this k -> k' with this sym op
            nw = np.where(np.all(np.isclose(newk[None],full_grid,
                                            atol=1.e-6,rtol=1.e-6,),axis=1))[0]
                                            
            if len(nw)==1:
                if nw[0] not in new_k_ind:
                    new_k_ind.append(nw[0])
                    si_per_k.append(isym)
                    orig_k_ind.append(k)
                    counter+=1
            else:                
                    print(kp[k],newk)


    new_k_ind=np.array(new_k_ind)
    orig_k_ind=np.array(orig_k_ind)
    si_per_k=np.array(si_per_k)

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

def get_phase_shifts(atom_pos,symop,equiv_atom):
    # calculate phase shifts for U
    phase_shift=np.zeros((symop.shape[0],atom_pos.shape[0],3),dtype=float)
    for isym in range(symop.shape[0]):
        for p in range(atom_pos.shape[0]):
            p1 = equiv_atom[isym,p]
            phase_shift[isym,p1] =   (symop[isym].T @ atom_pos[p])-atom_pos[p1]

    return phase_shift

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

def get_full_grid(nk1,nk2,nk3):
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
    for s in species:
      sdict[s[0]] = np.array(read_pseudopotential(join(workpath,savedir,s[1])))
    
    #double the l=0 if spin orbit
    if spin_orb:
        for s,p in sdict.items():
            tmp_list=[]
            for o in p:
                tmp_list.append(o)
                # if l=0 include it twice
                if o==0:
                    tmp_list.append(o)

            sdict[s] = np.array(tmp_list)

    # index of which orbitals belong to which atom in the basis
    a_index = np.hstack([[a]*np.sum((2*sdict[atoms[a]])+1) for a in range(len(atoms))])

    # value of l
    shell   = np.hstack([sdict[a] for a in atoms])

    return shell,a_index

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

  try:
      iterator_obj = ET.iterparse(fpp,events=('start','end'))
      iterator     = iter(iterator_obj)
      event,root   = next(iterator)

      for event,elem in iterator:        
          try:
              for i in elem.findall("PP_PSWFC/"):
                  sh.append(int(i.attrib['l']))
          except Exception as e:
              print(e)    

      sh=np.array(sh)


  except:
      with open(fpp) as ifo:
          ifs=ifo.read()
      res=re.findall("(.*)\s*Wavefunction",ifs)[1:]

      
      sh=np.array(list(map(int,list([x.split()[1] for x in res]))))

  return sh

############################################################################################
############################################################################################
############################################################################################

def wedge_to_grid(Hksp,U,a_index,phase_shifts,kp,new_k_ind,orig_k_ind,si_per_k,inv_flag,U_inv):
    # generates full grid from k points in IBZ
    nawf     = Hksp.shape[1]

    Hksp_s=np.zeros((new_k_ind.shape[0],nawf,nawf),dtype=complex)


    for j in range(new_k_ind.shape[0]):
        isym = si_per_k[j]
        oki  = orig_k_ind[j]
        nki  = new_k_ind[j]

        # if symop is identity
        if isym==0:
            Hksp_s[nki]=Hksp[oki]
            continue

        # other cases
        H  = Hksp[oki]

        # get k dependent U
        U_k     = get_U_k(kp[oki],phase_shifts[isym],a_index,U[isym])
        U_k_inv = LA.inv(U_k)

        #transformated H(k)
        THP = U_k @ H @ U_k_inv

        # apply inversion operator if needed
        if inv_flag[isym]:
            THP*=U_inv

        Hksp_s[nki]=THP
    
    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

def enforce_t_rev(Hksp_s,nk1,nk2,nk3):
    nawf=Hksp_s.shape[1]
    Hksp_s= np.reshape(Hksp_s,(nk1,nk2,nk3,nawf,nawf))
    Hksp_g=np.copy(Hksp_s)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                iv= (nk1-i)%nk1
                jv= (nk2-j)%nk2
                kv= (nk3-k)%nk3
                Hksp_s[i,j,k] = (Hksp_g[i,j,k]+np.conj(Hksp_g[iv,jv,kv]))/2.0


    Hksp_s= np.reshape(Hksp_s,(nk1*nk2*nk3,nawf,nawf))
    
    return Hksp_s

############################################################################################
############################################################################################
############################################################################################

def apply_t_rev(Hksp,kp):

    new_kp_list=[]
    new_Hk_list=[]
    for i in range(Hksp.shape[0]):
        new_kp= -kp[i]
        if not np.any(np.all(np.isclose(new_kp,kp),axis=1)):
            new_kp_list.append(new_kp)
            new_Hk_list.append(np.conj(Hksp[i]))

    kp=np.vstack([kp,np.array(new_kp_list)])
    Hksp=np.vstack([Hksp,np.array(new_Hk_list)])

    return Hksp,kp

############################################################################################
############################################################################################
############################################################################################

def add_U_wyc(U,U_wyc):
    for isym in range(U.shape[0]):
        U[isym]=U_wyc[isym] @ U[isym]

    return U

############################################################################################
############################################################################################
############################################################################################

def mat2aa(symop):
    # turn improper to proper rotation
    symop*=LA.det(symop)
    TR=np.trace(symop)
    ang=np.arccos(np.around((TR-1.0)/2.0,decimals=5))
    axis=np.zeros(3)
    # if angle is zero the axis is arbitrary
    if np.isclose(ang,0.0):
        axis=np.array([0,0,1])

    # if angle is pi..sign of direction of axis is arbitrary 
    elif np.isclose(ang,np.pi):
        S = symop+symop.T+((1-TR)*np.eye(3))
        N=S/(3.0-TR)
        axis[0]=N[0,0]
        axis[1]=N[1,1]
        axis[2]=N[2,2]
        if axis[0] != 0:
            axis[0]/=axis[0]
        if axis[1] != 0:
            axis[1]/=axis[1]
        if axis[2] != 0:
            axis[2]/=axis[2]
        
    # other cases
    else:
        axis[0] = symop[2,1] - symop[1,2]
        axis[1] = symop[0,2] - symop[2,0]
        axis[2] = symop[1,0] - symop[0,1]
        axis   *= -1.0/np.sqrt((3.0-TR)*(1.0+TR))
    return correct_roundoff(axis),ang

############################################################################################
############################################################################################
############################################################################################

def get_spin_rot(axis,angle):
    sr=np.zeros((2,2),dtype=complex)
    sr[0,0] = np.cos(angle/2.0)-1.0j*axis[2]*np.sin(angle/2.0)
    sr[1,1] = np.cos(angle/2.0)+1.0j*axis[2]*np.sin(angle/2.0)
    sr[0,1] = (-1.0j*axis[0]-axis[1])*np.sin(angle/2.0)
    sr[1,0] = (-1.0j*axis[0]+axis[1])*np.sin(angle/2.0)
                          
    return sr

############################################################################################
############################################################################################
############################################################################################

def get_spin_rot_rep(symop):
    # gets representation of a rotation for a spinor
    sr=np.zeros((symop.shape[0],2,2),dtype=complex)
    for isym in range(symop.shape[0]):
        axis,angle=mat2aa( symop[isym])
        sr[isym] = get_spin_rot(axis,angle)        
        
    return sr

############################################################################################
############################################################################################
############################################################################################

def add_spin_rot_rep(U,symop):
    sr = get_spin_rot_rep(symop)

    U_so = np.zeros_like(U)
    for isym in range(U.shape[0]):
        args=[sr[isym]]*(int(U[isym].shape[1]/2))
        so=LA.block_diag(*args)
        U_so[isym] = so @ U[isym] @ LA.inv(so)

    return U_so


############################################################################################
############################################################################################
############################################################################################

def open_grid(Hksp,full_grid,kp,symop,symop_cart,atom_pos,shells,a_index,equiv_atom,sym_info,sym_shift,nk1,nk2,nk3):

    nawf = Hksp.shape[1]

    # apply time reversal symmetry H(k) = H(-k)*
    Hksp,kp=apply_t_rev(Hksp,kp)


    # get index of k in wedge, index in full grid, 
    # and index of symop that transforms k to k'        
    new_k_ind,orig_k_ind,si_per_k = find_equiv_k(kp,symop,full_grid,sym_shift,check=True)

    # get array with wigner_d rotation matrix for each symop
    # for each of the orbital angular momentum l=[0,1,2,3]
    wigner,inv_flag = get_wigner(symop_cart)

    # convert the wigner_d into chemistry form for each symop
    wigner = convert_wigner_d(wigner)

    # get phase shifts from rotation symop
    phase_shifts     = get_phase_shifts(atom_pos,symop,equiv_atom)

    # build U and U_inv from blocks
    U = build_U_matrix(wigner,shells)

    # adds transformation to U that maps orbitals from
    # atom A to equivalent atom B atoms after symop
    U_wyc = map_equiv_atoms(a_index,equiv_atom)

    # combine U_wyc and U
    U = add_U_wyc(U,U_wyc)

    # get inversion operator
    U_inv = get_inv_op(shells)




#    U = add_spin_rot_rep(U,symop_cart)

#    print(U.shape,U_inv.shape,Hksp[0].shape)
#    raise SystemExit




    # transform H(k) -> H(k')
    Hksp = wedge_to_grid(Hksp,U,a_index,phase_shifts,kp,
                         new_k_ind,orig_k_ind,si_per_k,inv_flag,U_inv)

    Hksp = enforce_t_rev(Hksp,nk1,nk2,nk3)
#    check(Hksp,si_per_k,new_k_ind,orig_k_ind,phase_shifts,U,a_index,inv_flag,equiv_atom,kp,symop,full_grid,sym_info)        
    return Hksp

############################################################################################
############################################################################################
############################################################################################

def open_grid_wrapper(data_controller):
    np.set_printoptions(precision=4,suppress=True,linewidth=160)


    data_arrays = data_controller.data_arrays
    data_attr   = data_controller.data_attributes
    alat        = data_attr['alat']
    nk1         = data_attr['nk1']
    nk2         = data_attr['nk2']
    nk3         = data_attr['nk3']
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
    a_vectors   = correct_roundoff(a_vectors)
    b_vectors   = correct_roundoff(b_vectors)
    nspin       = Hks.shape[3]
    nawf        = Hks.shape[0]


    # convert atomic positions to crystal fractional coords
    conv=LA.inv(a_vectors)
    atom_pos = atom_pos @ conv
    atom_pos = correct_roundoff(atom_pos)

    # get symop crystal -> cartesian
    symop = correct_roundoff(symop)
    symop_cart = np.zeros_like(symop)
    inv_a_vectors = LA.inv(a_vectors)
    for isym in range(symop.shape[0]):
        symop_cart[isym] = (inv_a_vectors @ symop[isym] @ a_vectors)
    symop_cart = correct_roundoff(symop_cart)

    # convert k points from cartesian to crystal fractional
    conv = LA.inv(b_vectors)
    conv = correct_roundoff(conv)
    kp_red =  kp_red @ conv
    kp_red=correct_roundoff(kp_red)

    # get full grid in crystal fractional coords
    full_grid = get_full_grid(nk1,nk2,nk3)



    spin_orb=False
    # get shells and atom indices for blocks of the hamiltonian
    shells,a_index = read_shell(data_attr['workpath'],data_attr['savedir'],
                                data_arrays['species'],atom_lab,
                                spin_orb=spin_orb)


    # expand grid from wedge
    Hksp_temp=np.zeros((nawf,nawf,nk1*nk2*nk3,nspin),dtype=complex)
    for ispin in range(nspin):
        Hksp = np.ascontiguousarray(np.transpose(Hks,axes=(2,0,1,3))[:,:,:,ispin])

        Hksp = open_grid(Hksp,full_grid,kp_red,symop,symop_cart,atom_pos,
                           shells,a_index,equiv_atom,sym_info,sym_shift,nk1,nk2,nk3)

        Hksp_temp[:,:,:,ispin] = np.ascontiguousarray(np.transpose(Hksp,axes=(1,2,0)))


    np.save("kham.npy",Hksp_temp)
    data_arrays['Hks']=Hksp_temp

############################################################################################
############################################################################################
############################################################################################




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


# def find_equiv_k_mod(kp,symop,full_grid):
#     # find indices and symops that generate full grid H from wedge H
#     kp_track = []
#     orig_k_ind = []
#     new_k_ind = []
#     si_per_k = []
#     counter = 0

#     for k in range(kp.shape[0]):
#         for sym in range(symop.shape[0]):
#             #transform k -> k' with the sym op
#             newk = ((symop[sym].dot(kp[k])+0.5)%1.0)-0.5
#             newk= correct_roundoff(newk)


#             # Find index in the full grid where this k -> k' with this sym op
#             nw = np.where(np.all(np.isclose(newk[None],full_grid,atol=1.e-5,
#                                             rtol=1.e-5),axis=1))[0]

#             if len(nw)!=0:
#                 #check if we already have an equivilent k' for this k
#                 if nw[0] in kp_track:
#                     continue
#                 else:
#                     kp_track.append(nw[0])
#                     counter+=1
#                     si_per_k.append(sym)
#                     orig_k_ind.append(k)
#                     new_k_ind.append(nw[0])
#             else:
#                 print(kp[k])
#     new_k_ind=np.array(new_k_ind)
#     orig_k_ind=np.array(orig_k_ind)
#     si_per_k=np.array(si_per_k)

#     return new_k_ind,orig_k_ind,si_per_k

# ############################################################################################
# ############################################################################################
# ############################################################################################


# def symmetrize(Hksp,U,a_index,phase_shifts,kp,new_k_ind,orig_k_ind,si_per_k,inv_flag,U_wyc):
#     # generates full grid from k points in IBZ
#     nawf     = Hksp.shape[1]
#     nkp_grid = new_k_ind.shape[0]
#     Hksp_s=np.zeros((nkp_grid,nawf,nawf),dtype=complex)

#     for j in range(Hksp_s.shape[0]):
#         isym = si_per_k[j]

#         oki  = orig_k_ind[j]
#         nki  = new_k_ind[j]

#         H  = Hksp[j]

#         U_k     = get_U_k(kp[j],phase_shifts[isym],a_index,U[isym])
#         U_k_inv = LA.inv(U_k)

#         #transformated H(k)
#         THP = U_k @ H @ U_k_inv

#         # swap equiv atoms if needed
#         THP = U_wyc[isym] @ THP @ U_wyc[isym].T

#         # invert if the symop has inversion
#         if inv_flag[isym]:
#             THP=np.conj(THP)

#         Hksp_s[j]=THP

#     return np.sum(Hksp_s,axis=0)/Hksp_s.shape[0]

# ############################################################################################
# ############################################################################################
# ############################################################################################

# def symmetrize_grid(Hksp,full_grid,kp,symop,symop_cart,atom_pos,shells,a_index,equiv_atom,sym_info):

#     nawf = Hksp.shape[1]
#     symop_inv=np.zeros_like(symop)

#     for i in range(symop.shape[0]):
#         symop_inv[i]=LA.inv(symop[i])

#     symop_inv_cart=np.zeros_like(symop_cart)
#     for i in range(symop.shape[0]):
#         symop_inv_cart[i]=LA.inv(symop_cart[i])

#     # get array with wigner_d rotation matrix for each symop_inv
#     # for each of the orbital angular momentum l=[0,1,2,3]
#     wigner,inv_flag = get_wigner(symop_cart)

#     # if there is an inversion we need yo remap for that
#     equiv_atom = invert_atom_pos_map(atom_pos,equiv_atom,inv_flag)

#     # convert the wigner_d into chemistry form for each symop_inv
#     wigner = convert_wigner_d(wigner)

#     # get phase shifts from rotation symop_inv
#     phase_shifts     = get_phase_shifts(atom_pos,symop,inv_flag,equiv_atom)

#     # build U and U_inv from blocks
#     U = build_U_matrix(wigner,shells)

#     # adds transformation to U that maps orbitals from
#     # atom A to equivalent atom B atoms after symop_inv
#     U_wyc = map_equiv_atoms(a_index,equiv_atom)

#     # get index of k in wedge, index in full grid, 
#     # and index of symop_inv that transforms k to k'

#     Hksp_d = np.zeros_like(Hksp)
#     for i in range(full_grid.shape[0]):
#         new_k_ind,orig_k_ind,si_per_k = find_equiv_k(full_grid[i][None],symop_inv,
#                                                      full_grid,check=False)
#         print(i)
# #        print5B(orig_k_ind)
# #        continue

#         # transform H(k) -> H(k')
#         temp = symmetrize(Hksp[new_k_ind],U,a_index,phase_shifts,full_grid[new_k_ind],
#                                orig_k_ind,new_k_ind,si_per_k,inv_flag,U_wyc)

#         Hksp[i]=temp
# #        print(Hksp_d[i][:3,:3])
# #        print()
# #        print(Hksp[i][:3,:3])
# #        print("*"*20)

 


#     return Hksp

# ############################################################################################
# ############################################################################################
# ############################################################################################
