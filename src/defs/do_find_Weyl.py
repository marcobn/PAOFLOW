#
# PAOpy
#
# Utility to construct and operate on Hamliltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
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
# Pino D'Amico, Luis gapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 



import z2pack
import tbmodels
import scipy.optimize as so
#import matplotlib.pyplot as plt
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys
import scipy
try:
    import pyfftw
except:
    from scipy import fftpack as FFT
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from kpnts_interpolation_mesh import *
from do_non_ortho import *
import do_momentum
#import do_gradient
import os
#from load_balancing import *
from get_K_grid_fft import *
from constants import *
import time
import scipy.optimize as OP
from numpy import linalg as LAN
from load_balancing import *
from do_double_grid import *
#import do_bandwarping_calc

from clebsch_gordan import *
from get_R_grid_fft import *
#import nxtval
from communication import gather_full
#from Gatherv_Scatterv_wrappers import Gatherv_wrap


# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.set_printoptions(precision=6, threshold=100, edgeitems=50, linewidth=350, suppress=True)
load=False

from numpy import linalg as LAN


en_range=0.50
#en_range=5.50
#first_thresh=1.e-4
first_thresh=0.01


def band_loop_H(ini_ik,end_ik,nspin,nawf,nkpi,HRaux,kq,R):

   nawf,nawf,_,nspin = HRaux.shape
   kdot = np.zeros((1,R.shape[0]),dtype=complex,order="C")
   kdot = np.tensordot(R,2.0j*np.pi*kq[:,ini_ik:end_ik],axes=([1],[0]))
   np.exp(kdot,kdot)

   auxh = np.zeros((nawf,nawf,1,nspin),dtype=complex,order="C")
   for ispin in range(nspin):
       auxh[:,:,ini_ik:end_ik,ispin]=np.tensordot(HRaux[:,:,:,ispin],kdot,axes=([2],[0]))
   kdot  = None
   return(auxh)



def gen_eigs(HRaux,kq,Rfft,band,b_vectors):
    # Load balancing

    kq = np.dot(kq,b_vectors)
    read_S=False
    nawf,nawf,_,nspin = HRaux.shape
    E_kp = np.zeros((1,nawf,nspin),dtype=np.float64)
    kq=np.array([kq,],dtype=complex).T
    nkpi=1

    Hks_int  = np.zeros((nawf,nawf,1,nspin),dtype=complex,order='C') # final data arrays
    Hks_int = band_loop_H(0,1,nspin,nawf,nkpi,HRaux,kq,Rfft)

    v_kp  = np.zeros((1,nawf,nawf,nspin),dtype=np.complex)
    for ispin in range(nspin):
        E_kp[:,:,ispin],v_kp[:,:,:,ispin] =  LAN.eigh(Hks_int[:,:,0,ispin],UPLO='U')

    return E_kp.real,v_kp







def find_egap(HRaux,kq,Rfft,band,b_vectors,ef_index,ispin):

    E_kp,_= gen_eigs(HRaux,kq,Rfft,band,b_vectors)

    egapp = E_kp[0,ef_index,ispin]-E_kp[0,ef_index-1,ispin]

    return egapp




def do_search_grid(nk1,nk2,nk3,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],endpoint=False):

    nk1_arr   = np.linspace(snk1_range[0],snk1_range[1],num=nk1,   endpoint=endpoint)
    nk2_arr   = np.linspace(snk2_range[0],snk2_range[1],num=nk2,   endpoint=endpoint)
    nk3_arr   = np.linspace(snk3_range[0],snk3_range[1],num=nk3,   endpoint=endpoint)



    nk_str = np.zeros((nk1*nk2*nk3,3),order='C')
    nk_str  = np.array(np.meshgrid(nk1_arr,nk2_arr,nk3_arr,indexing='ij')).T.reshape(-1,3)

    return nk_str





def loop_min(ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,a_vectors,b_vectors,v_k,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],npool=1,shift=0.0,nl=None,sh=None):

    ini=[[0.25,0.25,0.25],[0.25,0.25,0.75],[0.25,0.75,0.25],[0.25,0.75,0.75],[0.75,0.25,0.25],[0.75,0.25,0.75],[0.75,0.75,0.25],[0.75,0.75,0.75]]
    CANDIDATES = {}

    candidates = 0

    for initial in ini:

        CANDIDATES = find_min(initial,CANDIDATES,candidates,ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,)
    band =bnd

    WEYL = {}

    model = tbmodels.Model.from_wannier_files(hr_file='z2pack_hamiltonian.dat')
    system = z2pack.tb.System(model,bands=ef_index)

    candidates=0

    for i in CANDIDATES.keys():
        kq=list(map(float, i[1:-1].split( )))

        result_1 = z2pack.surface.run(system=system,surface=z2pack.shape.Sphere(center=tuple(kq), radius=0.005))
        invariant = z2pack.invariant.chern(result_1)

        if invariant != 0:
            new = True
            for t in WEYL.keys():
                if np.linalg.norm(np.asarray(kq)-list(map(float, t[1:-1].split( ))))<0.005:
                    new=False
            if new:
                candidates += 1
                WEYL[str(kq).replace(",", "")]=invariant

    if bool(WEYL):
        j = 1
        for k in WEYL.keys():
            print ('Found Candidate No. {} at {} with Chirality:{}'.format(j,k,WEYL[k]))
            j = j + 1

    else:
        print("No Candidate found.")


def sym_test(Candidates,NewCandidates,candidates,HRaux,Rfft,band,b_vectors,ef_index,ispin):
    print ('starting sym test')
    for i in NewCandidates.keys():
        p=list(map(float, i[1:-1].split( )))
        p=np.dot(p,b_vectors)
        for x in [-1,1]:
            for y in [-1,1]:
                for z in [-1,1]:
                    kq = np.dot([p[0]*x,p[1]*y,p[2]*z],np.linalg.inv(b_vectors))
                    egap = find_egap(HRaux,kq,Rfft,band,b_vectors,ef_index,ispin)
                    if egap<0.00001:
                        new = True
                        for t in Candidates.keys():
                            
                            if np.linalg.norm(np.asarray(kq)-list(map(float, t[1:-1].split( ))))<0.0001:
                                new = False

                        if new:
                            candidates += 1
                            print ('Sym Candidate No.{} found at {} with gap:{} eV'.format(candidates,str(kq).replace(",", ""),egap))
                            Candidates[str(kq).replace(",", "")] = [egap]
    return Candidates,candidates





def find_min(initial,Candidates,candidates,ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,a_vectors,b_vectors,v_k):

    snk1_range=[-0.5,0.5]
    snk2_range=[-0.5,0.5]
    snk3_range=[-0.5,0.5]
    npool=1
    shift=0.0

    band = bnd
    np.set_printoptions(precision=6, threshold=100, edgeitems=50, linewidth=350, suppress=True)
    comm.Barrier()

    nk1+=1
    nk2+=1
    nk3+=1

    snk1=snk2=snk3=10


    #do the bounds for each search subsection of FBZ
    bounds_K  = np.zeros((search_grid.shape[0],3,2))
    guess_K   = np.zeros((search_grid.shape[0],3))
    #full grid

    #get the search grid off possible HSP

    #search in boxes
    bounds_K[:,:,0] = search_grid
    bounds_K[:,:,1] = search_grid+1.0*np.array([1.02/(snk1),1.02/(snk2),1.02/(snk3)])

    #initial guess is in the middle of each box
    guess_K = search_grid+np.array([initial[0]/snk1,initial[1]/snk2,initial[2]/snk3])

    #partial grid

    #swap bounds axes to put in the root finder
    bounds_K=np.swapaxes(bounds_K,1,2)

    #add two extra columns for info on which band and what spin for the extrema
    #so we don't lose informatioen we reshape to reduce the duplicate entries

    all_extrema_shape = [snk1*snk2*snk3,bnd,nspin,6]
    all_extrema_shape[0] =snk1*snk2*snk3

    vk_mesh = None
    velkp   = None
    velkp1  = None
    velkp1p = None
    velkp1=None
    comm.Barrier()

    sst=0

    if rank==0:
        nawf,nawf,nr1,nr2,nr3,nspin = HRaux.shape
        HRaux = HRaux.reshape((nawf,nawf,nr1*nr2*nr3,nspin))
    else:
        nspin=nawf=nr1=nr2=nr3=None

    nspin = comm.bcast(nspin)
    nawf  = comm.bcast(nawf)
    nr1   = comm.bcast(nr1)
    nr2   = comm.bcast(nr2)
    nr3   = comm.bcast(nr3)


    if rank!=0:
        HRaux=np.zeros((nawf,nawf,nr1*nr2*nr3,nspin),dtype=complex,order='C')

    comm.Barrier()
    comm.Bcast(HRaux)
    comm.Barrier()

    NewCandidates = {}
    R,_,_,_,_ = get_R_grid_fft(nr1,nr2,nr3,)
    Rfft=R

    ll=30
    st=0:

    sg = np.array(np.meshgrid(range(guess_K.shape[0]),list(range(1)),range(nspin),indexing='ij')).T.reshape(-1,3)

    comm.Barrier()

    fold_coord = lambda x: ((x+0.5)%1.0-0.5)

    extrema=[]
    timer_avg = 0.0

    fp=0
    sp=0
    ep=0
    
    c = -1
    for i in range(sg.shape[0]//size+1):

        c+=1

        if c>=sg.shape[0]:
            continue

        i     = sg[c][0]
        b     = sg[c][1]
        ispin = sg[c][2]


        lam_XiP  = lambda K: find_egap(HRaux,K,Rfft,band,b_vectors,ef_index,ispin)
        #loop over the search grid 


        startt = time.time()
        #fist pass
        x0 = np.asarray(guess_K[i]).ravel()
        n, = x0.shape

        current_bounds=[(bounds_K[i,0,0],bounds_K[i,1,0]),
                        (bounds_K[i,0,1],bounds_K[i,1,1]),
                        (bounds_K[i,0,2],bounds_K[i,1,2])]

        solx = OP.fmin_l_bfgs_b(lam_XiP,guess_K[i],bounds=current_bounds,
                                approx_grad=True,maxiter=3000)

        if np.abs(solx[1]<0.00001):

            if len(Candidates.keys()) == 0:
                Candidates[str(solx[0])] = [solx[1]]
                NewCandidates[str(solx[0])] = [solx[1]]
                candidates += 1
            else:
                real = True
                for i in Candidates.keys():
                    if np.linalg.norm(solx[0]-list(map(float, i[1:-1].split( ))))<0.0001:
                        real = False
                if real:
                    candidates += 1

                    Candidates[str(solx[0])] = [solx[1]]
                    NewCandidates[str(solx[0])] = [solx[1]]

        else:
            fp+=1
            continue



    comm.Barrier()

    return (Candidates,Rfft)

