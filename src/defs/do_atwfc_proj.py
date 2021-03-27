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

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.special
import scipy.interpolate
from scipy.io import FortranFile
import scipy.fft as FFT

from .read_upf import UPF

# Unitility functions to build atomic wfc from the pseudopotential and build the projections U (and overlaps if needed)

def read_pswfc_from_upf(data_controller, atom):
  arry, attr = data_controller.data_dicts()
  
  # loop over species
  for (at, pseudo) in arry['species']:
    if atom == at:
      upf = UPF(os.path.join(attr['fpath'], pseudo))
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

def build_pswfc_basis_all(data_controller, verbose=False):
  arry, attr = data_controller.data_dicts()
  basis = []
  
  # build the mesh in q space
  ecutrho = attr['ecutrho']
  dq = 0.01
  qmesh = np.arange(0, np.sqrt(ecutrho) + 4, dq)
  volume = attr['omega']
  
  # loop over atoms
  for na in range(len(arry['atoms'])):
    atom = arry['atoms'][na]
    tau = arry['tau'][na]
    r, pswfc, pseudo = read_pswfc_from_upf(data_controller, atom)
    if verbose:
      print('atom: {0:2s}  pseudo: {1:30s}  tau: {2}'.format(atom, pseudo, tau))
      
      # loop over pswfc'c
    for pao in pswfc:
      l = 'SPDF'.find(pao['label'][1])
      #### CHECK IF THERE IS A BETTER WAY ####
      if l == -1:
        l = 'spdf'.find(pao['label'][1])
        
      wfc_g = radialfft_simpson(r, pao['wfc'], l, qmesh, volume)
      
      for m in range(1, 2*l+2):
        basis.append({'atom': atom, 'tau': tau, 'l': l, 'm': m, 'label': pao['label'],
          'r': r, 'wfc': pao['wfc'].copy(), 'qmesh': qmesh, 'wfc_g': wfc_g})
        if verbose:
          print('      atwfc: {0:3d}  {3}  l={1:d}, m={2:-d}'.format(len(basis), l, m, pao['label']))
          
  return basis

def fft_wfc_G2R(wfc, igwx, gamma_only, mill, nr1, nr2, nr3, omega):
  wfcg = np.zeros((nr1,nr2,nr3), dtype=complex)
  
  for ig in range(igwx):
    wfcg[mill[0,ig],mill[1,ig],mill[2,ig]] = wfc[ig]
    if gamma_only:
      wfcg[-mill[0,ig],-mill[1,ig],-mill[2,ig]] = np.conj(wfc[ig])
      
  wfcr = FFT.ifftn(wfcg) * nr1 * nr2 * nr3 / np.sqrt(omega)
  return wfcr

def fft_wfc_R2G(wfc, igwx, mill, omega):
  tmp = FFT.fftn(wfc) / np.sqrt(omega)
  
  wfcg = np.zeros((igwx,), dtype=complex)
  for ig in range(igwx):
    wfgc[tmp] = tmp[mill[0,ig],mill[1,ig],mill[2,ig]]
    
  return wfcg

def read_QE_wfc(data_controller, ik):
  arry, attr = data_controller.data_dicts()
  
  wfcfile = 'wfc{0}.dat'.format(ik+1)
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
      
  wfc = np.array(wfc) * scalef
  gkspace = { 'xk': xk, 'igwx': igwx, 'mill': mill, 'bg': bg, 'gamma_only': gamma_only }
  return gkspace, { 'wfc': wfc, 'npol': npol, 'nbnd': nbnd, 'ispin': ispin }

def calc_atwfc_k(basis, gkspace):
  # construct atomic wfc at k
  atwfc_k = []
  natwfc = len(basis)
  
  xk, igwx, mill, bg, gamma_only = [gkspace[s] for s in ('xk', 'igwx', 'mill', 'bg', 'gamma_only')]
  #print('xk=', xk)
  
  for i in range(natwfc):
    
    # 1. build the structure factor
    strf = np.zeros((igwx,), dtype=complex)
    
#    hkl = np.array([mill[0,:],mill[1,:],mill[2,:]]).T
    hkl = mill.T
    k_plus_G = np.dot(hkl, bg.T)
    for ig in range(igwx):
      k_plus_G[ig,:] += xk
      
    tau = basis[i]['tau']
    k_plus_G_dot_tau = np.dot(k_plus_G, tau)
    strf = np.exp(-1j*k_plus_G_dot_tau)
    
    # 2. build the form factor
    l, m = basis[i]['l'], basis[i]['m']
    qmesh, wfc_g = basis[i]['qmesh'], basis[i]['wfc_g']
    q = np.linalg.norm(k_plus_G, axis=1)
    #fact = InterpolatedUnivariateSpline(qmesh, wfc_g)(q)
    fact = scipy.interpolate.interp1d(qmesh, wfc_g, kind='linear')(q)
    
    # 3. build the angular part
    kGx = np.zeros_like(q)
    kGy = np.zeros_like(q)
    kGz = np.zeros_like(q)
    for ig in range(igwx):
      if abs(q[ig]) > 1e-6:
        kGx[ig] = k_plus_G[ig,0] / q[ig]
        kGy[ig] = k_plus_G[ig,1] / q[ig]
        kGz[ig] = k_plus_G[ig,2] / q[ig]
        
    if l == 0:
      n0 = np.sqrt(1.0/(4.0*np.pi))
      ylmg = n0 * np.ones_like(q)
    elif l == 1:
      n1 = np.sqrt(3.0/(4.0*np.pi))
      if m == 1:
        ylmg = n1 * kGz
      elif m == 2:
        ylmg = -n1 * kGx
      elif m == 3:
        ylmg = -n1 * kGy
    elif l == 2:
      n2 = np.sqrt(15.0/(4.0*np.pi))
      if m == 1:
        ylmg = n2 * (3*kGz*kGz - 1)/(2*np.sqrt(3))
      elif m == 2:
        ylmg = -n2 * kGz*kGx
      elif m == 3:
        ylmg = -n2 * kGy*kGz
      elif m == 4:
        ylmg = n2 * (kGx*kGx - kGy*kGy) / 2.0
      elif m == 5:
        ylmg = n2 * kGx * kGy
    elif l == 3:
      n3 = np.sqrt(105.0/(4.0*np.pi))
      if m == 1:
        ylmg = n3 * kGz * (2*kGz*kGz - 3*kGx*kGx - 3*kGy*kGy) / (2.0*np.sqrt(15.0))
      elif m == 2:
        ylmg = -n3 * kGx * (4*kGz*kGz - kGx*kGx - kGy*kGy) / (2.0*np.sqrt(10.0))
      elif m == 3:
        ylmg = -n3 * kGy * (4*kGz*kGz - kGx*kGx - kGy*kGy) / (2.0*np.sqrt(10.0))
      elif m == 4:
        ylmg = n3 * kGz * (kGx*kGx - kGy*kGy) / 2.0
      elif m == 5:
        ylmg = n3 * kGx * kGy * kGz
      elif m == 6:
        ylmg = -n3 * kGx * (kGx*kGx - 3*kGy*kGy) / (2.0*np.sqrt(6.0))
      elif m == 7:
        ylmg = -n3 * kGy * (3*kGx*kGx - kGy*kGy) / (2.0*np.sqrt(6.0))
    else:
      raise NotImplementedError('l>2 not implemented yet')
      
    atwfc = strf * fact * ylmg * (1.0j)**l
    
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

def calc_proj_k(data_controller, basis, ik):
  gkspace, wfc = read_QE_wfc(data_controller, ik)
  atwfc_k = calc_atwfc_k(basis, gkspace)
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
        k_plus_G = np.array([i,j,k])@arry['b_vectors']
        if np.linalg.norm(k_plus_G)**2 <= attr['ecutrho']/(2*np.pi/attr['alat'])**2:
          mill_g.append(np.array([i,j,k]))
  mill_g = np.swapaxes(np.array(mill_g),1,0)
  igwx_g = mill_g.shape[1]
  
  mill = []
  for ig in range(igwx_g):
    k_plus_G = mill_g[:,ig]@arry['b_vectors'] + arry['kpnts'][ik]
    if np.linalg.norm(k_plus_G)**2 <= attr['ecutwfc']/(2*np.pi/attr['alat'])**2:
      mill.append(mill_g[:,ig])
  mill = np.swapaxes(np.array(mill),1,0)
  igwx = mill.shape[1]
  
  xk = arry['kpnts'][ik] * 2*np.pi/attr['alat']
  
  bg = arry['b_vectors'].T*2*np.pi/attr['alat']
  
  names = ['xk','igwx','mill','bg','gamma_only']
  arrays = [xk,igwx,mill,bg,gamma_only]
  gkspace = dict(zip(names,arrays))
  
  return(gkspace)