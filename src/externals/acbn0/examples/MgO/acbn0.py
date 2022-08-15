'''
 Thu Jul 25 12:00:46 CDT 2013
 By Luis Agapito @ Marco Buongiorno Nardelli at UNT

 Revised Jul 5 2022
  Frank T. Cerasoli

 Based on:
 Ints.py Basic routines for integrals in the PyQuante framework
 This program is part of the PyQuante quantum chemistry program suite.
 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 
 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
'''
from pyints import contr_coulomb
from pyints import CGBF
from mpi4py import MPI
from sys import argv
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def read_basis_files ( species ):
  basis = {}
  for s in species:
    for k,v in __import__(f'{s}_basis').basis_data.items():
      basis[s] = v
  return basis


def getbasis ( basis_functions, species, lattice, coords ):

  basis_functions = []
  for a,pos in zip(species, coords):
    for shell in basis[a]:
      for subshell in shell:

        bf = CGBF(pos*1.88973, a)
        for lx,ly,lz,coeff,zeta in subshell:
          bf.pnorms.append(1.)
          bf.pexps.append(zeta)
          bf.pcoefs.append(coeff)
          bf.powers.append((lx,ly,lz))

        basis_functions.append(bf)

  return basis_functions


def Dk ( basis_dm, basis_2e, Hks, Sks ):
  from scipy.linalg import eigh

  nbasis,_,nkpnts = Hks.shape
  size_dm,size_2e = basis_dm.shape[0],basis_2e.shape[0]
  D_k = np.zeros((nbasis,nbasis,nkpnts), dtype=complex)
  nlm_k = np.zeros((size_2e,nbasis,nkpnts), dtype=complex)

  # Find the density matrix for each k
  for ik in range(nkpnts):

    eig,vec = eigh(Hks[:,:,ik], Sks[:,:,ik])

    occ_ind = np.where(eig <= 0.)[0]
    nocc = len(occ_ind)

    lm_dm = np.zeros((size_dm,nocc), dtype=complex)
    lm_2e = np.zeros((size_2e,nocc), dtype=complex)

    sk_dm = Sks[basis_dm, :, ik]
    sk_2e = Sks[basis_2e, :, ik]
    for im in range(nocc):
      vim = vec[:, im]
      lm_dm[:,im] = np.conj(vim[basis_dm]) * sk_dm.dot(vim)
      lm_2e[:,im] = np.conj(vim[basis_2e]) * sk_2e.dot(vim)

    evec2 = vec[:, occ_ind]
    nlm_k[:,:nocc,ik] = lm_2e
    lm_dm = np.sum(lm_dm, axis=0)

    D_k[:,:,ik] = np.tensordot(np.conj(evec2*lm_dm), evec2, axes=([1],[1]))

  nlm_k[:,:,0] = 0
  return D_k, nlm_k


def Nmm ( nlm, Hks, kwght ):

  lm_size,nbasis,nkp = nlm.shape
  nlm_aux = np.zeros((lm_size,nbasis), dtype=complex)
  for ik,wght in enumerate(kwght):
    nlm_aux += wght * nlm[:,:,ik]

  return np.sum(nlm_aux/np.sum(kwght), axis=1)


def DR ( Dk, kwght ):

  nawf = Dk.shape[0]
  nkpnts = kwght.shape[0]

  D = np.zeros((nawf,nawf), dtype=complex)
  for ik,wght in enumerate(kwght):
    D += wght * Dk[:,:,ik]

  return D.real/np.sum(kwght)


def hartree_energy ( DR_up, DR_dn, basis, basis_2e ):
  import itertools

  tmp_U,tmp_J = 0,0
  ind = list(itertools.product(basis_2e,repeat=4))
  ind = np.array_split(ind, size)
  ind = comm.scatter(ind)

  for k,l,m,n in ind:
    int_U = coulomb(basis[m], basis[n], basis[k], basis[l])
    int_J = coulomb(basis[m], basis[k], basis[n], basis[l])

    a_b = DR_up[m,n]*DR_up[k,l] + DR_dn[m,n]*DR_dn[k,l]
    ab_ba = DR_dn[m,n]*DR_up[k,l] + DR_up[m,n]*DR_dn[k,l]

    tmp_U += int_U * (a_b + ab_ba)
    tmp_J += int_J * a_b

  tmp_U = comm.reduce(tmp_U)
  tmp_J = comm.reduce(tmp_J)

  return tmp_U, tmp_J


def read_gridfile ( fname ):
  return np.load(fname)


def read_ham_data ( fpath, nspin ):
  from os.path import join

  kpnts = np.loadtxt(open(join(fpath,'k.txt'),'r'))
  kwght = np.loadtxt(open(join(fpath,'wk.txt'),'r'))

  if len(kpnts.shape) == 1:
    kpnts = np.array([kpnts])
    kwght = np.array([kwght])
  nkpnts = kpnts.shape[0]

  kovp = read_gridfile(join(fpath,'kovp.npy'))
  nbasis = int(np.sqrt(kovp.shape[0]/nkpnts))
  kovp = kovp.reshape((nbasis,nbasis,nkpnts))

  kham_up = kham_dn = None
  for ispin in range(nspin):

    fname = 'kham'
    if nspin == 2:
      fname += '_up' if ispin==0 else '_dn'
    fname += '.npy'

    kham = read_gridfile(join(fpath,fname))
    kham = kham.reshape((nbasis,nbasis,nkpnts))

    if ispin == 0:
      kham_up = kham
    elif ispin == 1:
      kham_dn = kham

  return kpnts, kwght, kovp, kham_up, kham_dn


def coulomb ( a, b, c, d ):
  ' Coulomb interaction between four contracted Gaussians '
  return contr_coulomb(a.pexps,a.pcoefs,a.pnorms,a.origin,a.powers,
                       b.pexps,b.pcoefs,b.pnorms,b.origin,b.powers,
                       c.pexps,c.pcoefs,c.pnorms,c.origin,c.powers,
                       d.pexps,d.pcoefs,d.pnorms,d.origin,d.powers)


argc = len(argv)
if len(argv) == 1:
  if rank == 0:
    print('Usage:\n  python acbn0.py {prefix}')
  quit()

data = {}
with open(argv[1], 'r') as f:
  for l in f.readlines():
    if l.startswith('#'):
      continue
    k,v = (v.strip() for v in l.split('='))
    data[k] = v

ang_to_bohr = 1.88973
sym = data['symbol']
nspin = int(data['nspin'])
species = data['atlabels'].split(',')

lattice = np.array([float(v) for v in data['latvects'].split(',')])
lattice = lattice.reshape((3,3)) / ang_to_bohr

coords = np.array([float(v) for v in data['coords'].split(',')])
coords = coords.reshape((coords.shape[0]//3,3)) / ang_to_bohr

split_bstr = lambda bstr : np.array([int(v) for v in bstr.split(',')])
basis_dm = split_bstr(data['reduced_basis_dm'])
basis_2e = split_bstr(data['reduced_basis_2e'])

basis = read_basis_files(species)
gauss_basis = getbasis(basis, species, lattice, coords)
nbasis = len(gauss_basis)

fpath = './output'
kpnts,kwght,Sks,Hks_up,Hks_dw = read_ham_data(fpath, nspin)

dk,nlm = Dk(basis_dm, basis_2e, Hks_up, Sks)
nlm = Nmm(nlm, Hks_up, kwght)
nnlm = nlm.shape[0]

dk_dn = None
den_U,den_J = 0,0
if nspin == 1:
  Naa,Nab = 0., 0.
  for i1,m1 in enumerate(nlm):
    for i2,m2 in enumerate(nlm):
      nlm12 = m1 * m2
      Nab += nlm12
      if i1 != i2:
        Naa += nlm12
  den_U = 2 * (Naa.real + Nab.real)
  den_J = 2 * Naa.real

else:
  dk_dn,nlmd = Dk(basis_dm, basis_2e, Hks_dw, Sks)
  nlmd = Nmm(nlmd, Hks_dw, kwght)

  Naa,Nbb,Nab = 0., 0., 0.
  for i1 in range(nnlm):
    for i2 in range(nnlm):
      Nab += nlm[i1] * nlmd[i2]
      if i1 != i2:
        Naa += nlm[i1] * nlm[i2]
        Nbb += nlmd[i1] * nlmd[i2]
  den_U = Naa.real + Nbb.real + 2*Nab.real
  den_J = Naa.real + Nbb.real

DR_0 = DR(dk, kwght)

DR_0_dn = DR_0
if nspin == 2:
  DR_0_dn = DR(dk_dn, kwght)

num_U,num_J = hartree_energy(DR_0, DR_0_dn, gauss_basis, basis_2e)

hartree_to_eV = 27.211396132
U = U_eff = hartree_to_eV * num_U / den_U
if den_J == 0:
  J = 'Inf'
else:
  J = hartree_to_eV * num_J / den_J
  U_eff -= J

with open(f'{sym}_UJ.txt', 'w') as f:
  f.write(f'U : {U}\nJ : {J}\nU_eff : {U_eff}\n')

