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

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.special
import scipy.interpolate
from scipy.io import FortranFile
import scipy.fft as FFT

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from .read_upf import UPF

# Unitility functions to build atomic wfc from the pseudopotential and build the projections U (and overlaps if needed)

def read_pswfc_from_upf(data_controller, atom):
  arry, attr = data_controller.data_dicts()
  
  # loop over species
  for (at, pseudo) in arry['species']:
    if atom == at:
      upf = UPF(os.path.join(attr['fpath'], pseudo))
      attr['jchia'][at] = upf.jchia
      return upf.r, upf.pswfc, pseudo
    
  raise RuntimeError('atom not found')
  
def radialfft_simpson(r, f, l, qmesh, volume):
  fq = np.zeros_like(qmesh)
  fact = 4.0*np.pi / np.sqrt(volume)
  
  f[r > 10.0] = 0.0
  for iq in range(len(qmesh)):
    q = qmesh[iq]
    bess = scipy.special.spherical_jn(l, r*q)
    aux = f * bess * r
    fq[iq] = scipy.integrate.simps(aux, r)*fact
  return fq



def assign_jm(basis):
    ib = 0
    while ib < len(basis):
        b = basis[ib]
        l, m = b['l'], b['m']
        
        if l == 0:
            for c,jm in enumerate((0,1)):
                basis[ib+c]['jm'] = jm
            ib += 2

        elif l == 1:
            for c,jm in enumerate((2,3,4,5,6,7,)):
                basis[ib+c]['jm'] = jm
            ib += 6
            
        elif l == 2:
            for c,jm in enumerate((8,9,10,11,12,13,14,15,16,17)):
                basis[ib+c]['jm'] = jm
            ib += 10
            
        elif l == 3:
            for c,jm in enumerate((18,19,20,21,22,23,24,25,26,27,28,29,30,31)):
                basis[ib+c]['jm'] = jm
            ib += 14
        
        else:
            raise NotImplemented('l > 3 not implemented')


def build_pswfc_basis_all(data_controller):
  arry, attr = data_controller.data_dicts()
  verbose = attr['verbose']
  
  # build the mesh in q space
  ecutrho = attr['ecutrho']
  dq = 0.01
  qmesh = np.arange(0, np.sqrt(ecutrho) + 4, dq)
  volume = attr['omega']
  
  # loop over atoms
  basis,shells = [],{}
  attr['jchia'] = {}
  for na in range(len(arry['atoms'])):
    atom = arry['atoms'][na]
    tau = arry['tau'][na]
    r, pswfc, pseudo = read_pswfc_from_upf(data_controller, atom)
    if verbose and rank == 0:
      print('atom: {0:2s}  pseudo: {1:30s}  tau: {2}'.format(atom, pseudo, tau))
      
    # loop over pswfc'c
    a_shells = []
    jchia = []

    s = 0.5
    for pao in pswfc:
      l = 'SPDF'.find(pao['label'][1].upper())
      assert l != -1

      a_shells.append(l)
      if l == 0:
         jchia.append(0.5)
         s = 0.5
      else:
         jchia.append(l-s)
         s = -s

      wfc_g = radialfft_simpson(r, pao['wfc'], l, qmesh, volume)
      
      for m in range(1, 2*l+2):
        basis.append({'atom': atom, 'tau': tau, 'l': l, 'm': m, 'label': pao['label'],
          'r': r, 'wfc': pao['wfc'].copy(), 'qmesh': qmesh, 'wfc_g': wfc_g})
        if verbose and rank == 0:
          print('      atwfc: {0:3d}  {3}  l={1:d}, m={2:-d}'.format(len(basis), l, m, pao['label']))

      if attr['dftSO']:
        if l == 0:
            basis.append({'atom': atom, 'tau': tau, 'l': l, 'm': m, 'label': pao['label'],
              'r': r, 'wfc': pao['wfc'].copy(), 'qmesh': qmesh, 'wfc_g': wfc_g})
            if verbose and rank == 0:
              print('      atwfc: {0:3d}  {3}  l={1:d}, m={2:-d}'.format(len(basis), l, m, pao['label']))

    if atom not in shells:
      shells[atom] = a_shells
      attr['jchia'][atom] = jchia
    
  # in case of SO: assign the jm index
  if attr['dftSO']:
    assign_jm(basis)
    #for b in basis:
    #    l,m,jm = b['l'], b['m'], b['jm']
    #    print(l,m,jm)

  return basis,shells



def build_aewfc_basis(data_controller):
  arry, attr = data_controller.data_dicts()
  verbose = attr['verbose']
  
  # read the atomic bases
  aebasis = []
  for na in range(len(arry['atoms'])):
    elem = arry['atoms'][na]
    aefiles = glob.glob(attr['basispath']+str(elem)+'/*.dat')
    label = []
    for entry in aefiles:
      label.append(entry.split('/')[-1].split('.')[0])
    aebasis.append(dict(zip(label,aefiles)))
  
  # build the mesh in q space
  ecutrho = attr['ecutrho']
  dq = 0.01
  qmesh = np.arange(0, np.sqrt(ecutrho) + 4, dq)
  volume = attr['omega']
  
  # loop over atoms
  basis,shells = [],{}
  attr['jchia'] = {}
  for na in range(len(arry['atoms'])):
    atom = arry['atoms'][na]
    tau = arry['tau'][na]
    aewfc = []
    for shell in arry['shells'][atom]:
      data = np.loadtxt(aebasis[na][shell])    
      aewfc.append({shell : data[:,1], 'r' : data[:,0]})
      
      if verbose and rank == 0:
        print('atom: {0:2s}  AEWFC: {1:30s}  tau: {2}'.format(atom, aebasis[na][shell], tau))

    ash = []
    jchia = []
    for n in range(len(aewfc)):
      l = 'SPDF'.find(list(aewfc[n].items())[0][0][1].upper())
      assert l != -1
      ash.append(l)

      if attr['dftSO']:
         if l == 0:
             jchia.append(0.5)
         else:
             ash.append(l)
             jchia.append(l-0.5)
             jchia.append(l+0.5)

      wfc_g = radialfft_simpson(aewfc[n]['r'], aewfc[n][list(aewfc[n].items())[0][0]], l, qmesh, volume)

      twice = 1
      if attr['dftSO']: twice = 2

      for _ in range(twice):
        for m in range(1, 2*l+2):
          basis.append({'atom': atom, 'tau': tau, 'l': l, 'm': m, 'label': list(aewfc[n].items())[0][0],
                        'r': aewfc[n]['r'], 'wfc': aewfc[n][list(aewfc[n].items())[0][0]].copy(), 
                        'qmesh': qmesh, 'wfc_g': wfc_g})
          if verbose and rank == 0:
              print('      atwfc: {0:3d}  {3}  l={1:d}, m={2:-d}'.format(len(basis), l, m, list(aewfc[n].items())[0][0]))

    if atom not in shells:
      shells[atom] = ash
      attr['jchia'][atom] = jchia
    
  # in case of SO: assign the jm index
  if attr['dftSO']:
    assign_jm(basis)
    #for b in basis:
    #    l,m,jm = b['l'], b['m'], b['jm']
    #    print(l,m,jm)

  return basis,shells



def fft_wfc_G2R_old(wfc, igwx, gamma_only, mill, nr1, nr2, nr3, omega):
  wfcg = np.zeros((nr1,nr2,nr3), dtype=complex)
  
  for ig in range(igwx):
    wfcg[mill[0,ig],mill[1,ig],mill[2,ig]] = wfc[ig]
    if gamma_only:
      wfcg[-mill[0,ig],-mill[1,ig],-mill[2,ig]] = np.conj(wfc[ig])
      
  wfcr = FFT.ifftn(wfcg) * nr1 * nr2 * nr3 / np.sqrt(omega)
  return wfcr

def fft_wfc_G2R(wfc, gkspace, nr1, nr2, nr3, omega):
  igwx = gkspace['igwx']
  gamma_only = gkspace['gamma_only']
  mill = gkspace['mill']
  wfcg = np.zeros((nr1,nr2,nr3), dtype=complex)
  
  for ig in range(igwx):
    wfcg[mill[0,ig],mill[1,ig],mill[2,ig]] = wfc[ig]
    if gamma_only:
      wfcg[-mill[0,ig],-mill[1,ig],-mill[2,ig]] = np.conj(wfc[ig])
      
  wfcr = FFT.ifftn(wfcg) * nr1 * nr2 * nr3 / np.sqrt(omega)
  return wfcr

def fft_allwfc_G2R(wfc, gkspace, nr1, nr2, nr3, omega):
  igwx = gkspace['igwx']
  gamma_only = gkspace['gamma_only']
  mill = gkspace['mill']
  try: 
    nox = wfc.shape[0]
    nwx = wfc.shape[1]
    wfcg = np.zeros((nwx,nr1,nr2,nr3), dtype=complex)
  except:
    nox = 0
    wfcg = np.zeros((nr1,nr2,nr3), dtype=complex)
  if nox == 0:
    for ig in range(igwx):
      wfcg[:,mill[0,ig],mill[1,ig],mill[2,ig]] = wfc[:,ig]
      if gamma_only:
        wfcg[:,-mill[0,ig],-mill[1,ig],-mill[2,ig]] = np.conj(wfc[:,ig])
    wfcr = FFT.ifftn(wfcg) * nr1 * nr2 * nr3 / np.sqrt(omega)
  else:
    wfcr = np.zeros((nox,nr1,nr2,nr3), dtype=complex)
    for no in range(nox):
      for ig in range(igwx):
        wfcg[no,mill[0,ig],mill[1,ig],mill[2,ig]] = wfc[no,ig]
        if gamma_only:
          wfcg[no,-mill[0,ig],-mill[1,ig],-mill[2,ig]] = np.conj(wfc[no,ig])
      wfcr[no] = FFT.ifftn(wfcg[no]) * nr1 * nr2 * nr3 / np.sqrt(omega)
  return wfcr
  
def fft_wfc_R2G(wfc, igwx, mill, omega):
  tmp = FFT.fftn(wfc) / np.sqrt(omega)
  
  wfcg = np.zeros((igwx,), dtype=complex)
  for ig in range(igwx):
    wfgc[tmp] = tmp[mill[0,ig],mill[1,ig],mill[2,ig]]
    
  return wfcg

def read_QE_wfc(data_controller, ik, ispin):
  arry, attr = data_controller.data_dicts()
  
  if attr['nspin'] == 1 or attr['nspin'] == 4:
    wfcfile = 'wfc{0}.dat'.format(ik+1)
  elif attr['nspin'] == 2 and ispin == 0:
    wfcfile = 'wfcdw{0}.dat'.format(ik+1)
  elif attr['nspin'] == 2 and ispin == 1:
    wfcfile = 'wfcup{0}.dat'.format(ik+1)
  else:
    print('no wfc file found')
    
  with FortranFile(os.path.join(attr['fpath'], wfcfile), 'r') as f:
    record = f.read_ints(np.int32)
    assert len(record) == 11, 'something wrong reading fortran binary file'
    
    ik_ = record[0]
    assert ik+1 == ik_, 'wrong k-point in wfc file???'
    
    xk = np.frombuffer(record[1:7], np.float64)
    ispin = record[7]
    gamma_only = (record[8] != 0)
    scalef = np.frombuffer(record[9:], np.float64)[0]
    #print('ik =', ik, '  ispin =', ispin, '  gamma_only =', gamma_only, '  scalef =', scalef)
    #print('xk =', xk)
    
    ngw, igwx, npol, nbnd = f.read_ints(np.int32)
    #print('ngw =', ngw, '  igwx = ', igwx, '  npol = ', npol, '  nbnd = ', nbnd)
    bg = f.read_reals(np.float64).reshape(3,3,order='F')
    #print('bg = ', bg)
    mill = f.read_ints(np.int32).reshape(3,igwx,order='F')
    #print('mill.shape = ', mill.shape)
    
    wfc = []
    for i in range(nbnd):
      wfc.append(f.read_reals(np.complex128))
  
  # compute overlap
  ovp = np.zeros((nbnd,nbnd),dtype=complex)
  for n in range(nbnd):
    for m in range(nbnd):
      ovp[n,m] = np.sum(np.conj(wfc[n]).dot(wfc[m]))
  eigs, eigv = np.linalg.eigh(ovp)
  assert (np.all(eigs>=0))

  X = scipy.linalg.sqrtm(ovp)
  owfc = np.linalg.solve(X.T, wfc) 
  
  wfc = np.array(wfc) * scalef
  gkspace = { 'xk': xk, 'igwx': igwx, 'mill': mill, 'bg': bg, 'gamma_only': gamma_only }
  return gkspace, { 'wfc': owfc, 'npol': npol, 'nbnd': nbnd, 'ispin': ispin }


def calc_ylmg(k_plus_G, q):
    # cubic harmonics: build the angular part, no spin orbit
    kGx = np.zeros_like(q)
    kGy = np.zeros_like(q)
    kGz = np.zeros_like(q)
    for ig in range(len(q)):
      if abs(q[ig]) > 1e-6:
        kGx[ig] = k_plus_G[ig,0] / q[ig]
        kGy[ig] = k_plus_G[ig,1] / q[ig]
        kGz[ig] = k_plus_G[ig,2] / q[ig]
    
    lmax = 3
    ylmg = np.zeros((len(q), (lmax+1)*(lmax+1)))

    # normalization factors for l=0,1,2,3
    n0 = np.sqrt(1.0/(4.0*np.pi))
    n1 = np.sqrt(3.0/(4.0*np.pi))
    n2 = np.sqrt(15.0/(4.0*np.pi))
    n3 = np.sqrt(105.0/(4.0*np.pi))

    # l = 0
    ylmg[:,0] = n0 * np.ones_like(q)

    # l = 1
    ylmg[:,1] = n1 * kGz
    ylmg[:,2] = -n1 * kGx         # note the '-' sign, due to the (lack of) Condon–Shortley phase
    ylmg[:,3] = -n1 * kGy         # the same in the l=2 and l=3 blocks, when |m| is odd

    # l = 2
    ylmg[:,4] = n2 * (3*kGz*kGz - 1)/(2*np.sqrt(3))
    ylmg[:,5] = -n2 * kGz*kGx
    ylmg[:,6] = -n2 * kGy*kGz
    ylmg[:,7] = n2 * (kGx*kGx - kGy*kGy) / 2.0
    ylmg[:,8] = n2 * kGx * kGy

    # l = 3
    ylmg[:,9] = n3 * kGz * (2*kGz*kGz - 3*kGx*kGx - 3*kGy*kGy) / (2.0*np.sqrt(15.0))
    ylmg[:,10]= -n3 * kGx * (4*kGz*kGz - kGx*kGx - kGy*kGy) / (2.0*np.sqrt(10.0))
    ylmg[:,11]= -n3 * kGy * (4*kGz*kGz - kGx*kGx - kGy*kGy) / (2.0*np.sqrt(10.0))
    ylmg[:,12]= n3 * kGz * (kGx*kGx - kGy*kGy) / 2.0
    ylmg[:,13]= n3 * kGx * kGy * kGz
    ylmg[:,14]= -n3 * kGx * (kGx*kGx - 3*kGy*kGy) / (2.0*np.sqrt(6.0))
    ylmg[:,15]= -n3 * kGy * (3*kGx*kGx - kGy*kGy) / (2.0*np.sqrt(6.0))

    return ylmg


def calc_ylmg_complex_0(ylmg):
    # complex spherical harmonics
    ylmgc = np.zeros_like(ylmg, np.complex)

    sqrt2 = np.sqrt(2.0)

    # l = 0
    ylmgc[:,0] = ylmg[:,0]

    # l = 1
    ylmgc[:,1] = ylmg[:,1]
    #ylmgc[:,2] = -(ylmg[:,2] + 1j*ylmg[:,3])/sqrt2   # m=1
    #ylmgc[:,3] =  (ylmg[:,2] - 1j*ylmg[:,3])/sqrt2   # m=-1
    ylmgc[:,2] =  (ylmg[:,2] + 1j*ylmg[:,3])/sqrt2   # m=1     # because QE cub. harm. have opposite sign
    ylmgc[:,3] = -(ylmg[:,2] - 1j*ylmg[:,3])/sqrt2   # m=-1

    # l = 2
    ylmgc[:,4] = ylmg[:,4]
    #ylmgc[:,5] = -(ylmg[:,5] + 1j*ylmg[:,6])/sqrt2   # m=1
    #ylmgc[:,6] =  (ylmg[:,5] - 1j*ylmg[:,6])/sqrt2   # m=-1
    ylmgc[:,5] =  (ylmg[:,5] + 1j*ylmg[:,6])/sqrt2   # m=1     # because QE cub. harm. have opposite sign
    ylmgc[:,6] = -(ylmg[:,5] - 1j*ylmg[:,6])/sqrt2   # m=-1
    ylmgc[:,7] = +(ylmg[:,7] + 1j*ylmg[:,8])/sqrt2   # m=2
    ylmgc[:,8] =  (ylmg[:,7] - 1j*ylmg[:,8])/sqrt2   # m=-2
    
    # l = 3
    ylmgc[:,9] = ylmg[:,9]
    #ylmgc[:,10] = -(ylmg[:,10] + 1j*ylmg[:,11])/sqrt2   # m=1
    #ylmgc[:,10] =  (ylmg[:,10] - 1j*ylmg[:,11])/sqrt2   # m=-1
    ylmgc[:,10] =  (ylmg[:,10] + 1j*ylmg[:,11])/sqrt2   # m=1     # because QE cub. harm. have opposite sign
    ylmgc[:,10] = -(ylmg[:,10] - 1j*ylmg[:,11])/sqrt2   # m=-1
    ylmgc[:,12] = +(ylmg[:,12] + 1j*ylmg[:,13])/sqrt2   # m=2
    ylmgc[:,13] =  (ylmg[:,12] - 1j*ylmg[:,13])/sqrt2   # m=-2
    #ylmgc[:,14] = -(ylmg[:,14] + 1j*ylmg[:,15])/sqrt2   # m=3
    #ylmgc[:,15] =  (ylmg[:,14] - 1j*ylmg[:,15])/sqrt2   # m=-3
    ylmgc[:,14] =  (ylmg[:,14] + 1j*ylmg[:,15])/sqrt2   # m=3     # because QE cub. harm. have opposite sign
    ylmgc[:,15] = -(ylmg[:,14] - 1j*ylmg[:,15])/sqrt2   # m=-3

    return ylmgc


def calc_ylmg_so(ylmgc):
    # spinor spherical harmonics
    npw = ylmgc.shape[0]
    nylm = ylmgc.shape[1]
    ylmgso = np.zeros((2*npw,2*nylm), np.complex)
    sqrt = np.sqrt

    # generated automatically by cb.py
    #l=0, j=0.5 m_j=-0.5 upper=sqrt(0)*Y(0,-1) 	 lower=sqrt(1)*Y(0,0)
    ylmgso[:npw,0]=0.0; ylmgso[npw:,0]=sqrt(1)*ylmgc[:npw,0]; 

    #l=0, j=0.5 m_j= 0.5 upper=sqrt(1)*Y(0,0) 	 lower=sqrt(0)*Y(0,1)
    ylmgso[:npw,1]=sqrt(1)*ylmgc[:npw,0]; ylmgso[npw:,1]=0.0; 

    #l=1, j=0.5 m_j=-0.5 upper=sqrt(2/3)*Y(1,-1) 	 lower=-sqrt(1/3)*Y(1,0))
    ylmgso[:npw,2]=sqrt(2/3)*ylmgc[:npw,3]; ylmgso[npw:,2]=-sqrt(1/3)*ylmgc[:npw,1]; 

    #l=1, j=0.5 m_j= 0.5 upper=sqrt(1/3)*Y(1,0) 	 lower=-sqrt(2/3)*Y(1,1))
    ylmgso[:npw,3]=sqrt(1/3)*ylmgc[:npw,1]; ylmgso[npw:,3]=-sqrt(2/3)*ylmgc[:npw,2]; 

    #l=1, j=1.5 m_j=-1.5 upper=sqrt(0)*Y(1,-2) 	 lower=sqrt(1)*Y(1,-1)
    ylmgso[:npw,4]=0.0; ylmgso[npw:,4]=sqrt(1)*ylmgc[:npw,3]; 

    #l=1, j=1.5 m_j=-0.5 upper=sqrt(1/3)*Y(1,-1) 	 lower=sqrt(2/3)*Y(1,0)
    ylmgso[:npw,5]=sqrt(1/3)*ylmgc[:npw,3]; ylmgso[npw:,5]=sqrt(2/3)*ylmgc[:npw,1]; 

    #l=1, j=1.5 m_j= 0.5 upper=sqrt(2/3)*Y(1,0) 	 lower=sqrt(1/3)*Y(1,1)
    ylmgso[:npw,6]=sqrt(2/3)*ylmgc[:npw,1]; ylmgso[npw:,6]=sqrt(1/3)*ylmgc[:npw,2]; 

    #l=1, j=1.5 m_j= 1.5 upper=sqrt(1)*Y(1,1) 	 lower=sqrt(0)*Y(1,2)
    ylmgso[:npw,7]=sqrt(1)*ylmgc[:npw,2]; ylmgso[npw:,7]=0.0; 

    #l=2, j=1.5 m_j=-1.5 upper=sqrt(4/5)*Y(2,-2) 	 lower=-sqrt(1/5)*Y(2,-1))
    ylmgso[:npw,8]=sqrt(4/5)*ylmgc[:npw,8]; ylmgso[npw:,8]=-sqrt(1/5)*ylmgc[:npw,6]; 

    #l=2, j=1.5 m_j=-0.5 upper=sqrt(3/5)*Y(2,-1) 	 lower=-sqrt(2/5)*Y(2,0))
    ylmgso[:npw,9]=sqrt(3/5)*ylmgc[:npw,6]; ylmgso[npw:,9]=-sqrt(2/5)*ylmgc[:npw,4]; 

    #l=2, j=1.5 m_j= 0.5 upper=sqrt(2/5)*Y(2,0) 	 lower=-sqrt(3/5)*Y(2,1))
    ylmgso[:npw,10]=sqrt(2/5)*ylmgc[:npw,4]; ylmgso[npw:,10]=-sqrt(3/5)*ylmgc[:npw,5]; 

    #l=2, j=1.5 m_j= 1.5 upper=sqrt(1/5)*Y(2,1) 	 lower=-sqrt(4/5)*Y(2,2))
    ylmgso[:npw,11]=sqrt(1/5)*ylmgc[:npw,5]; ylmgso[npw:,11]=-sqrt(4/5)*ylmgc[:npw,7]; 

    #l=2, j=2.5 m_j=-2.5 upper=sqrt(0)*Y(2,-3) 	 lower=sqrt(1)*Y(2,-2)
    ylmgso[:npw,12]=0.0; ylmgso[npw:,12]=sqrt(1)*ylmgc[:npw,8]; 

    #l=2, j=2.5 m_j=-1.5 upper=sqrt(1/5)*Y(2,-2) 	 lower=sqrt(4/5)*Y(2,-1)
    ylmgso[:npw,13]=sqrt(1/5)*ylmgc[:npw,8]; ylmgso[npw:,13]=sqrt(4/5)*ylmgc[:npw,6]; 

    #l=2, j=2.5 m_j=-0.5 upper=sqrt(2/5)*Y(2,-1) 	 lower=sqrt(3/5)*Y(2,0)
    ylmgso[:npw,14]=sqrt(2/5)*ylmgc[:npw,6]; ylmgso[npw:,14]=sqrt(3/5)*ylmgc[:npw,4]; 

    #l=2, j=2.5 m_j= 0.5 upper=sqrt(3/5)*Y(2,0) 	 lower=sqrt(2/5)*Y(2,1)
    ylmgso[:npw,15]=sqrt(3/5)*ylmgc[:npw,4]; ylmgso[npw:,15]=sqrt(2/5)*ylmgc[:npw,5]; 

    #l=2, j=2.5 m_j= 1.5 upper=sqrt(4/5)*Y(2,1) 	 lower=sqrt(1/5)*Y(2,2)
    ylmgso[:npw,16]=sqrt(4/5)*ylmgc[:npw,5]; ylmgso[npw:,16]=sqrt(1/5)*ylmgc[:npw,7]; 

    #l=2, j=2.5 m_j= 2.5 upper=sqrt(1)*Y(2,2) 	 lower=sqrt(0)*Y(2,3)
    ylmgso[:npw,17]=sqrt(1)*ylmgc[:npw,7]; ylmgso[npw:,17]=0.0; 

    #l=3, j=2.5 m_j=-2.5 upper=sqrt(6/7)*Y(3,-3) 	 lower=-sqrt(1/7)*Y(3,-2))
    ylmgso[:npw,18]=sqrt(6/7)*ylmgc[:npw,15]; ylmgso[npw:,18]=-sqrt(1/7)*ylmgc[:npw,13]; 

    #l=3, j=2.5 m_j=-1.5 upper=sqrt(5/7)*Y(3,-2) 	 lower=-sqrt(2/7)*Y(3,-1))
    ylmgso[:npw,19]=sqrt(5/7)*ylmgc[:npw,13]; ylmgso[npw:,19]=-sqrt(2/7)*ylmgc[:npw,11]; 

    #l=3, j=2.5 m_j=-0.5 upper=sqrt(4/7)*Y(3,-1) 	 lower=-sqrt(3/7)*Y(3,0))
    ylmgso[:npw,20]=sqrt(4/7)*ylmgc[:npw,11]; ylmgso[npw:,20]=-sqrt(3/7)*ylmgc[:npw,9]; 

    #l=3, j=2.5 m_j= 0.5 upper=sqrt(3/7)*Y(3,0) 	 lower=-sqrt(4/7)*Y(3,1))
    ylmgso[:npw,21]=sqrt(3/7)*ylmgc[:npw,9]; ylmgso[npw:,21]=-sqrt(4/7)*ylmgc[:npw,10]; 

    #l=3, j=2.5 m_j= 1.5 upper=sqrt(2/7)*Y(3,1) 	 lower=-sqrt(5/7)*Y(3,2))
    ylmgso[:npw,22]=sqrt(2/7)*ylmgc[:npw,10]; ylmgso[npw:,22]=-sqrt(5/7)*ylmgc[:npw,12]; 

    #l=3, j=2.5 m_j= 2.5 upper=sqrt(1/7)*Y(3,2) 	 lower=-sqrt(6/7)*Y(3,3))
    ylmgso[:npw,23]=sqrt(1/7)*ylmgc[:npw,12]; ylmgso[npw:,23]=-sqrt(6/7)*ylmgc[:npw,14]; 

    #l=3, j=3.5 m_j=-3.5 upper=sqrt(0)*Y(3,-4) 	 lower=sqrt(1)*Y(3,-3)
    ylmgso[:npw,24]=0.0; ylmgso[npw:,24]=sqrt(1)*ylmgc[:npw,15]; 

    #l=3, j=3.5 m_j=-2.5 upper=sqrt(1/7)*Y(3,-3) 	 lower=sqrt(6/7)*Y(3,-2)
    ylmgso[:npw,25]=sqrt(1/7)*ylmgc[:npw,15]; ylmgso[npw:,25]=sqrt(6/7)*ylmgc[:npw,13]; 

    #l=3, j=3.5 m_j=-1.5 upper=sqrt(2/7)*Y(3,-2) 	 lower=sqrt(5/7)*Y(3,-1)
    ylmgso[:npw,26]=sqrt(2/7)*ylmgc[:npw,13]; ylmgso[npw:,26]=sqrt(5/7)*ylmgc[:npw,11]; 

    #l=3, j=3.5 m_j=-0.5 upper=sqrt(3/7)*Y(3,-1) 	 lower=sqrt(4/7)*Y(3,0)
    ylmgso[:npw,27]=sqrt(3/7)*ylmgc[:npw,11]; ylmgso[npw:,27]=sqrt(4/7)*ylmgc[:npw,9]; 

    #l=3, j=3.5 m_j= 0.5 upper=sqrt(4/7)*Y(3,0) 	 lower=sqrt(3/7)*Y(3,1)
    ylmgso[:npw,28]=sqrt(4/7)*ylmgc[:npw,9]; ylmgso[npw:,28]=sqrt(3/7)*ylmgc[:npw,10]; 

    #l=3, j=3.5 m_j= 1.5 upper=sqrt(5/7)*Y(3,1) 	 lower=sqrt(2/7)*Y(3,2)
    ylmgso[:npw,29]=sqrt(5/7)*ylmgc[:npw,10]; ylmgso[npw:,29]=sqrt(2/7)*ylmgc[:npw,12]; 

    #l=3, j=3.5 m_j= 2.5 upper=sqrt(6/7)*Y(3,2) 	 lower=sqrt(1/7)*Y(3,3)
    ylmgso[:npw,30]=sqrt(6/7)*ylmgc[:npw,12]; ylmgso[npw:,30]=sqrt(1/7)*ylmgc[:npw,14]; 

    #l=3, j=3.5 m_j= 3.5 upper=sqrt(1)*Y(3,3) 	 lower=sqrt(0)*Y(3,4)
    ylmgso[:npw,31]=sqrt(1)*ylmgc[:npw,14]; ylmgso[npw:,31]=0.0; 

    return ylmgso    



def calc_atwfc_k(basis, gkspace, dftSO=False):
  # construct atomic wfc at k
  atwfc_k = []
  natwfc = len(basis)
  
  xk, igwx, mill, bg, gamma_only = [gkspace[s] for s in ('xk', 'igwx', 'mill', 'bg', 'gamma_only')]
  #print('xk=', xk)
  
  # build k+G vectors
  hkl = mill.T
  k_plus_G = np.dot(hkl, bg.T)
  for ig in range(igwx):
      k_plus_G[ig,:] += xk

  # pre-calculate spherical harmonics
  q = np.linalg.norm(k_plus_G, axis=1)
  ylmg = calc_ylmg(k_plus_G, q)
  if dftSO:
      ylmgc = calc_ylmg_complex_0(ylmg)
      ylmgso = calc_ylmg_so(ylmgc)
 
  # loop over atoms 
  for i in range(natwfc):
    # 1. build the structure factor
    strf = np.zeros((igwx,), dtype=complex)
    tau = basis[i]['tau']
    k_plus_G_dot_tau = np.dot(k_plus_G, tau)
    strf = np.exp(-1j*k_plus_G_dot_tau)
    
    # 2. build the form factor
    qmesh, wfc_g = basis[i]['qmesh'], basis[i]['wfc_g']
    #fact = InterpolatedUnivariateSpline(qmesh, wfc_g)(q)
    fact = scipy.interpolate.interp1d(qmesh, wfc_g, kind='linear')(q)
    
    # 3. build the angular part
    l, m = basis[i]['l'], basis[i]['m']
    if l > 3: raise NotImplementedError('l>3 not implemented yet')
    lm = l*l + (m-1)
    if dftSO:
        jm = basis[i]['jm']

    # 4. final
    if not dftSO:
        atwfc = strf * fact * ylmg[:,lm] * (1.0j)**l
    else:
        atwfc = np.hstack((strf,strf)) * np.hstack((fact,fact)) * ylmgso[:,jm] * (1.0j)**l
 
    atwfc_k.append(atwfc)
    
  return np.array(atwfc_k)



def ortho_atwfc_k(atwfc_k):
  # orthonormalize atwfcs
  natwfc = atwfc_k.shape[0]
  ovp = np.zeros((natwfc, natwfc), dtype=np.complex)
  
  for i in range(natwfc):
    for j in range(natwfc):
      ovp[i,j] = np.dot(np.conj(atwfc_k[i]), atwfc_k[j])
      
      # check that eigenvalues are positive
  eigs, eigv = np.linalg.eigh(ovp)
  assert (np.all(eigs>=0))
  
  # orthogonalize
  if True:
    X = scipy.linalg.sqrtm(ovp)
    oatwfc_k = np.linalg.solve(X.T, atwfc_k)
  else:
    eigs = 1.0/np.sqrt(eigs)
    X = np.dot(np.conj(eigv), np.dot(np.diag(eigs), eigv.T))
    oatwfc_k = np.dot(X, atwfc_k)
    
    # check ortonormalization
  oovp = np.zeros((natwfc, natwfc), dtype=np.complex)
  for i in range(natwfc):
    for j in range(natwfc):
      oovp[i,j] = np.dot(np.conj(oatwfc_k[i]), oatwfc_k[j])
      
  diff = np.linalg.norm(oovp - np.eye(natwfc))
  if np.abs(diff) > 1e-4:
    raise RuntimeError('ortogonalization failed')
    
  return oatwfc_k


def calc_proj_k(data_controller, basis, ik, ispin):
  arry, attr = data_controller.data_dicts()
  gkspace, wfc = read_QE_wfc(data_controller, ik, ispin)
  atwfc_k = calc_atwfc_k(basis, gkspace, attr['dftSO'])
  oatwfc_k = ortho_atwfc_k(atwfc_k)
  proj_k = np.dot(np.conj(oatwfc_k), wfc['wfc'].T)
  return (proj_k.T)


def calc_gkspace(data_controller,ik,gamma_only=False):
  arry, attr = data_controller.data_dicts()
  # calculate sphere of Miller indeces for k + G
  gcutm = attr['ecutrho'] / (2*np.pi/attr['alat'])**2
  at = arry['a_vectors']
  nx = 2*int(np.sqrt(gcutm)*np.sqrt(at[0,0]**2 + at[1,0]**2 + at[2,0]**2)) + 1
  ny = 2*int(np.sqrt(gcutm)*np.sqrt(at[0,1]**2 + at[1,1]**2 + at[2,1]**2)) + 1
  nz = 2*int(np.sqrt(gcutm)*np.sqrt(at[0,2]**2 + at[1,2]**2 + at[2,2]**2)) + 1
  nx = int((nx+1)/2)
  ny = int((ny+1)/2)
  nz = int((nz+1)/2)
  # sphere of Miller indeces 
  mill_g = []
  for i in range(-nx, nx+1):
    for j in range(-ny, ny+1):
      for k in range(-nz, nz+1):
        k_plus_G = np.array([i,j,k]) @ arry['b_vectors']
        if np.linalg.norm(k_plus_G)**2 <= attr['ecutrho']/(2*np.pi/attr['alat'])**2:
          mill_g.append(np.array([i,j,k]))
  mill_g = np.swapaxes(np.array(mill_g),1,0)
  igwx_g = mill_g.shape[1]
  
  mill = []
  for ig in range(igwx_g):
    k_plus_G = mill_g[:,ig]@arry['b_vectors'] + arry['kgrid'][:,ik]
    if np.linalg.norm(k_plus_G)**2 <= attr['ecutwfc']/(2*np.pi/attr['alat'])**2:
      mill.append(mill_g[:,ig])
  mill = np.swapaxes(np.array(mill),1,0)
  igwx = mill.shape[1]
  
  xk = arry['kgrid'][:,ik] * 2*np.pi/attr['alat']
  
  bg = arry['b_vectors'].T*2*np.pi/attr['alat']
  
  names = ['xk','igwx','mill','bg','gamma_only']
  arrays = [xk,igwx,mill,bg,gamma_only]
  gkspace = dict(zip(names,arrays))
  arry['gkspace'] = gkspace
  
  return(gkspace)
