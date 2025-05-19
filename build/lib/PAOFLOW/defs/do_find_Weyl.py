#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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
from .communication import gather_full, scatter_full
from numpy import linalg as LAN
from .constants import BOHR_RADIUS_ANGS

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#np.set_printoptions(precision=8, threshold=100, edgeitems=50, linewidth=350, suppress=True)

def band_loop_H ( ini_ik, end_ik, HRaux, kq, R ):
  nawf,_,_,nspin = HRaux.shape
  kdot = np.zeros((1,R.shape[0]), dtype=complex, order='C')
  kdot = np.tensordot(R, 2.0j*np.pi*kq[:,ini_ik:end_ik], axes=([1],[0]))
  np.exp(kdot, kdot)

  auxh = np.zeros((nawf,nawf,1,nspin), dtype=complex, order='C')
  for ispin in range(nspin):
    auxh[:,:,ini_ik:end_ik,ispin] = np.tensordot(HRaux[:,:,:,ispin], kdot, axes=([2],[0]))
  return auxh

def gen_eigs ( HRaux, kq, R ):
  # Load balancing

  nawf,_,_,nspin = HRaux.shape
  E_kp = np.zeros((1,nawf,nspin), dtype=np.float64)

  kq=kq[:,None]

#  Hks_int  = np.zeros((nawf,nawf,1,nspin),dtype=complex,order='C') # final data arrays
  Hks_int = band_loop_H(0,1,HRaux,kq,R)

  for ispin in range(nspin):
    E_kp[:,:,ispin] =  LAN.eigvalsh(Hks_int[:,:,0,ispin],UPLO='U')

  return E_kp


def get_gap ( HR, kq, R, nelec ):
  E_kp = gen_eigs(HR, kq, R)
  return E_kp[0,nelec,0] - E_kp[0,nelec-1,0]


def get_R_grid_fft ( nr1, nr2, nr3 ):

  R = np.zeros((nr1*nr2*nr3,3))

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


def get_search_grid ( nk1, nk2, nk3, snk1_range=[-0.5,0.5], snk2_range=[-0.5,0.5], snk3_range=[-0.5,0.5], endpoint=False ):

  nk1_arr = np.linspace(snk1_range[0], snk1_range[1], num=nk1, endpoint=endpoint)
  nk2_arr = np.linspace(snk2_range[0], snk2_range[1], num=nk2, endpoint=endpoint)
  nk3_arr = np.linspace(snk3_range[0], snk3_range[1], num=nk3, endpoint=endpoint)

  #nk_str = np.zeros((nk1*nk2*nk3,3), order='C')
  return np.array(np.meshgrid(nk1_arr,nk2_arr,nk3_arr,indexing='ij')).T.reshape(-1,3)

def find_weyl ( data_controller, test_rad, search_grid ):
  import os

  arry,attr = data_controller.data_dicts()

  symf = attr['symmetrize']
  nelec,verbose = attr['nelec'],attr['verbose']
  HRs,symops,b_vectors,TR_flag = arry['HRs'],arry['sym_rot'],arry['b_vectors'],arry['sym_TR']

  nawf,_,nk1,nk2,nk3,nspin = HRs.shape
  R = get_R_grid_fft(nk1, nk2, nk3)

  HRs = np.reshape(HRs, (nawf,nawf,nk1*nk2*nk3,nspin))

  mag_soc = np.logical_and(attr["dftMAG"], attr["dftSO"])

  CAND, ene = find_min(HRs, nelec, R, b_vectors, symf, verbose, search_grid)

  WEYL = {}
  if rank == 0:
    if symf:
      # get all equiv k
      CAND = get_equiv_k(CAND, symops, TR_flag, mag_soc)

    if verbose:
      print()
      for i in range(CAND.shape[0]):
        E_kp = gen_eigs(HRs, CAND[i], R)
        ene = E_kp[0,nelec,0]
        gap = E_kp[0,nelec,0] - E_kp[0,nelec-1,0]
        tup = (i+1,) + tuple(CAND[i][j] for j in range(3)) + (ene,gap)
        print('Weyl point candidate #%d crystal coord: [ %6.4f %6.4f %6.4f ] ene=%.4f gap=%.6e' % tup)

        in_cart = b_vectors.T.dot(CAND[i])
        tup = (i+1,) + tuple(in_cart[j] for j in range(3)) + (ene,gap)
        print('Weyl point candidate #%d 2pi/alat     : [ %6.4f %6.4f %6.4f ] ene=%.4f gap=%.6e' % tup)
        print()

    try:
      import z2pack
      import tbmodels      
      model = tbmodels.Model.from_wannier_files(hr_file=os.path.join(attr['opath'],'z2pack_hamiltonian.dat'))
      system = z2pack.tb.System(model,bands=nelec)

      candidates=0

      if not verbose:
        import logging
        logging.getLogger('z2pack').setLevel(logging.WARNING)

      for kq in CAND:
        # if distance between two candidates is very small
        k_rad = np.amin(np.sqrt(np.sum((kq-CAND)**2,axis=1)))*0.5
        if k_rad > test_rad:
          k_rad = test_rad
        k_rad = 0.005

        surface = z2pack.shape.Sphere(center=tuple(kq), radius=k_rad)
        result_1 = z2pack.surface.run(system=system, surface=surface)
        invariant = z2pack.invariant.chern(result_1)

        if invariant != 0:
          candidates += 1
          WEYL[str(kq).replace(',','')] = invariant

    except:
      print('Could not load z2pack to verify chirality of weyl points')
      for kq in CAND:
        WEYL[str(kq).replace(',', '')] = '?'

    if verbose:
      print()

    wcs = '{0:>3} {1:>10} {2:>10} {3:>10} {4:>2} {5:>7}\n'.format('#','2pi/alat','2pi/alat','2pi/alat','C', 'ene')
    wcs += '  #   alat = {0} angstrom\n'.format(attr['alat']*BOHR_RADIUS_ANGS)
    wcs += '-'*72 + '\n'

    for j,k in enumerate(WEYL.keys()):
      fm = np.array(list(map(float,k[1:-1].split())))
      E_kp = gen_eigs(HRs, fm, R)
      ene = E_kp[0,nelec,0]
      fm = b_vectors.T.dot(fm[:,None])[:,0]
      try:
        if verbose:
          fstring = 'Found Candidate No. {0} at [{1:>7.4f} {2:>7.4f} {3:>7.4f}] with Chirality:{4:>2} ene={5:>7.4f}'
          print(fstring.format(j+1,fm[0],fm[1],fm[2],int(WEYL[k]),ene))
        wcs += '{0:>3d} {1:>10.4f} {2:>10.4f} {3:>10.4f} {4:>2} {5:>7.4f}\n'.format(j+1,fm[0],fm[1],fm[2],int(WEYL[k]),ene)
      except:
        wcs += '{0:>3d} {1:>10.4f} {2:>10.4f} {3:>10.4f} {4:>2} {5:>7.4f}\n'.format(j+1,fm[0],fm[1],fm[2],WEYL[k],ene)

    with open(os.path.join(attr['opath'],'weyl_points.dat'), 'w') as ofo:
      ofo.write(wcs)

def find_min ( HRs, nelec, R, a_vectors, symf, verbose, search_grid=[8,8,8] ):

  snk = tuple(search_grid[i] for i in range(3))
  #snk2 = search_grid[1]
  #snk3 = search_grid[2]
  #search_grid = do_search_grid(snk1,snk2,snk3)
  search_grid = get_search_grid(*snk)
  
  #get the search grid off possible HSP
  end = np.array([.5]*3)
  start = np.array([-.5]*3)
  #end1 = end2 = end3 = 0.5
  #start1 = start2 = start3 = -0.5
  bounds_K  = np.zeros((search_grid.shape[0],3,2))
  #guess_K   = np.zeros((search_grid.shape[0],3))

  #do the bounds for each search subsection of FBZ
  #search in boxes
  bounds_K[:,:,0] = search_grid
  bbox = np.array([(end[i]-start[i])/snk[i] for i in range(3)])
  bounds_K[:,:,1] = search_grid + bbox
  #initial guess is in the middle of each box
  guess_K = search_grid + 0.5*bbox

  sgi = np.arange(bounds_K.shape[0], dtype=int)
  sgi = scatter_full(sgi, 1)
  print('finding Weyl points... rank={0} npoints={1}'.format(rank, sgi.shape[0]))

  candidates = np.zeros((sgi.shape[0],3))

  lam_XiP = lambda K: get_gap(HRs, K, R, nelec)
  for i in range(sgi.shape[0]):
    solx = OP.minimize(lam_XiP, guess_K[sgi[i]], bounds=bounds_K[sgi[i]], method="L-BFGS-B", options={"ftol":1.e-14,"gtol":1.e-12})

    if np.abs(solx.fun) < 1.e-5:
      candidates[i] = solx.x
      
  candidates = gather_full(candidates,1)

  ene = None
  if rank == 0:  
    # filter out non hits
    candidates = candidates[np.where(np.sum(candidates,axis=1)!=0.0)]
    # calculate energy at each candidate to reduce equiv ones
    ene = np.zeros((candidates.shape[0]))
    gaps = np.zeros((candidates.shape[0]))
    for i in range(ene.shape[0]):
      eigs = gen_eigs(HRs,candidates[i],R)[0,:,0]
      ene[i] = eigs[nelec]
      gaps[i] = eigs[nelec]-eigs[nelec-1]

    if symf:
      # sort by gap size (should be nearly zero)
      idx = np.argsort(gaps)
      ene = ene[idx]
      candidates = candidates[idx]     
      # filter out duplicates by degeneracy
      _,idx=np.unique(np.around(ene,decimals=4),return_index=True)
      ene = ene[idx]
      candidates = candidates[idx]
      if verbose:
       print('\nfound %s non-equivilent candidates'%candidates.shape[0])
       for i in range(ene.shape[0]):
         tup = tuple(candidates[i][j] for j in range(3)) + (ene[i],)
         print("[ % 7.4f % 7.4f % 7.4f ] % 4.4f"%tup)
       print()

  comm.Barrier()
  return (candidates, ene)


def get_equiv_k(kp,symop,sym_TR,mag_soc):
  from .pao_sym import correct_roundoff

  # if we have time inversion sym
  if not mag_soc:
     kp=np.vstack([kp,-kp])
  

  kp = correct_roundoff(kp)
  newk_tot = np.copy(kp)

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

    newk_tot=np.vstack([newk_tot,newk.T])

  # filter duplicates
  newk_r=np.around(newk_tot,decimals=4)
  newk_r[np.where(np.abs(newk_r)<1.e-4)]=0

  b_pos = np.ascontiguousarray(newk_r).view(np.dtype((np.void,newk_r.dtype.itemsize *newk_r.shape[1])))
  _, idx = np.unique(b_pos, return_index=True)
  newk_tot=newk_tot[idx]

  return newk_tot
