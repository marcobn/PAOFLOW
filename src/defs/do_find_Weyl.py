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
from scipy import fftpack as FFT
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import os
import scipy.optimize as OP
from numpy import linalg as LAN
from .load_balancing import *
#import do_bandwarping_calc
from .communication import gather_full, scatter_full
from numpy import linalg as LAN

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.set_printoptions(precision=6, threshold=100, edgeitems=50, linewidth=350, suppress=True)



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



def gen_eigs(HRaux,kq,R):
    # Load balancing

    nawf,nawf,_,nspin = HRaux.shape
    E_kp = np.zeros((1,nawf,nspin),dtype=np.float64)
    kq=kq[:,None]

    nkpi=1

    Hks_int  = np.zeros((nawf,nawf,1,nspin),dtype=complex,order='C') # final data arrays
    Hks_int = band_loop_H(0,1,nspin,nawf,nkpi,HRaux,kq,R)


    for ispin in range(nspin):
        E_kp[:,:,ispin] =  LAN.eigvalsh(Hks_int[:,:,0,ispin],UPLO='U')

    return E_kp







def get_gap(HR,kq,R,nelec):
    E_kp = gen_eigs(HR,kq,R)
    egapp = E_kp[0,nelec,0]-E_kp[0,nelec-1,0]


    return egapp


def get_R_grid_fft ( nr1, nr2, nr3):

  R=np.zeros((nr1*nr2*nr3,3))

  for i in range(nr1):
    for j in range(nr2):
      for k in range(nr3):
        n = k + j*nr3 + i*nr2*nr3
        Rx = float(i)/float(nr1)
        Ry = float(j)/float(nr2)
        Rz = float(k)/float(nr3)
        if Rx >= 0.5: Rx=Rx-1.0
        if Ry >= 0.5: Ry=Ry-1.0
        if Rz >= 0.5: Rz=Rz-1.0
        Rx -= int(Rx)
        Ry -= int(Ry)
        Rz -= int(Rz)

        R[n,0] = Rx*nr1 
        R[n,1] = Ry*nr2 
        R[n,2] = Rz*nr3

  return R  



def do_search_grid(nk1,nk2,nk3,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],endpoint=False):

    nk1_arr   = np.linspace(snk1_range[0],snk1_range[1],num=nk1,   endpoint=endpoint)
    nk2_arr   = np.linspace(snk2_range[0],snk2_range[1],num=nk2,   endpoint=endpoint)
    nk3_arr   = np.linspace(snk3_range[0],snk3_range[1],num=nk3,   endpoint=endpoint)

    nk_str = np.zeros((nk1*nk2*nk3,3),order='C')
    nk_str  = np.array(np.meshgrid(nk1_arr,nk2_arr,nk3_arr,indexing='ij')).T.reshape(-1,3)

    return nk_str





def find_weyl(HRs,nelec,nk1,nk2,nk3):


#    HRs,nk1,nk2,nk3 = interp_odd(HRs,nk1,nk2,nk3)
    nawf,nawf,nk1,nk2,nk3,nspin = HRs.shape
    R= get_R_grid_fft ( nk1, nk2, nk3)

    HRs=np.reshape(HRs,(nawf,nawf,nk1*nk2*nk3,nspin))



    CANDIDATES = find_min(HRs,nelec,R)
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


def find_min(HRs,nelec,R):

    snk1=snk2=snk3=10
    search_grid = do_search_grid(snk1,snk2,snk3)
    
    #do the bounds for each search subsection of FBZ
    bounds_K  = np.zeros((search_grid.shape[0],3,2))
    guess_K   = np.zeros((search_grid.shape[0],3))
    #full grid

    #get the search grid off possible HSP
    
    start1=-0.5
    start2=-0.5
    start3=-0.5
    end1  = 0.5
    end2  = 0.5
    end3  = 0.5

    #search in boxes
    bounds_K[:,:,0] = search_grid
    bounds_K[:,:,1] = search_grid+1.0*np.array([(end1-start1)/(snk1),(end2-start2)/(snk2),(end3-start3)/(snk3)])
    #initial guess is in the middle of each box
    guess_K = search_grid+0.5*np.array([(end1-start1)/(snk1),(end2-start2)/(snk2),(end3-start3)/(snk3)])

    #swap bounds axes to put in the root finder
    bounds_K=np.swapaxes(bounds_K,1,2)
#    print(R)

    lam_XiP = lambda K: get_gap(HRs,K,R,nelec)

    sgi = np.arange(bounds_K.shape[0],dtype=int)

    sgi=scatter_full(sgi,1)

#    print(sgi)


    candidates=[]

    for i in sgi:


#        solx = OP.fmin_l_bfgs_b(lam_XiP,guess_K[i],bounds=bounds_K[i].T,
#                                pgtol=1.e-10,approx_grad=True)
        # if np.abs(solx[1]<0.00001):
        #     candidates.append(solx[0])
        solx = OP.shgo(lam_XiP,bounds=bounds_K[i].T,n=250,iters=4,options={"f_tol":1.e-10})
        if np.abs(solx.fun<0.00001):
            print(solx.fun,solx.x)
            candidates.append(solx.x)
            

    
    if len(candidates)!=0:
       candidates = np.array(candidates)
    else:
       candidates=np.array([])

    comm.Barrier()
    if rank==0:
       print("done")
    candidates = gather_full(candidates,1)
    if rank==0:
       print(candidates)
    return (candidates)


def zp_HR ( HRs,nk1,nk2,nk3 ):

  nfft1=nk1%2
  nfft2=nk2%2
  nfft3=nk3%2

  HRs_pad=np.zeros((int(nawf**2),nk1,nk2,nk3,nspin),dtype=complex)

  for ispin in range(nspin):
    for n in range(HRs.shape[0]):
        HRs_pad[n,:,:,:,ispin] = zero_pad(HRs[n,:,:,:,ispin],nk1,nk2,nk3,nfft1,nfft2,nfft3)
