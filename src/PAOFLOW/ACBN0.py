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
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo, R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np

class ACBN0:

  def __init__ ( self, prefix, pthr=0.95, workdir='./', mpi_qe='', nproc=1, qe_path='', qe_options='', mpi_python='', python_path='',
                projection='atomic' ):
    from .defs.file_io import struct_from_inputfile_QE
    from .defs.upf_gaussfit import gaussian_fit
    from .defs.header import header
    from os.path import join
    from os import chdir

    header()
    print('\nPerforming ACBN0 self-consistent determination of Hubbard U corrections.\n')

    self.prefix = prefix
    self.pthr = pthr
    self.workdir = workdir
    self.mpi_qe = mpi_qe
    self.nproc = nproc
    self.qpath = qe_path
    self.mpi_python = mpi_python
    self.ppath = python_path
    self.qoption = qe_options
    self.projection = projection

    self.uVals = {}
    self.occ_states = {}
    self.occ_values = {}
    self.hubbard_occ = {}
    self.hubbard_tag = 'HUBBARD ('+self.projection+')'

    chdir(self.workdir)

    # Get structure information
    self.blocks,self.cards = struct_from_inputfile_QE(f'{self.prefix}.scf.in')
    self.nspin = int(struct['nspin']) if 'nspin' in self.blocks['system'] else 1

    # Generate gaussian fits
    print('Generating gaussian fits for pseudopotential basis states.\n')
    self.basis = {}
    self.uspecies = []
    for s in self.cards['ATOMIC_SPECIES'][1:]:
      ele,_,pp = s.split()
      self.uspecies.append(ele)
      atno,basis = gaussian_fit(pp)
      self.basis[ele] = basis

    # Store U values from input template
    if 'HUBBARD' in self.cards:
      import re

      self.hubbard_tag = self.cards['HUBBARD'][0]
      for h in self.cards['HUBBARD'][1:]:

        _,sym,uval = h.split()
        self.uVals[sym] = float(uval)

        ele,occ = sym.split('-')
        if ele not in self.occ_states:
          self.occ_states[ele] = []
        self.occ_states[ele].append(occ)

      # Store occupations from input template
      nat = len(self.uspecies)
      for s in self.blocks['system']:
        if 'hubbard_occ' in s:

          i,j = map(int, re.findall('\(([^\)]+),([^\)]+)\)', s)[0])
          if i > nat:
            msg = f'hubbard_occ index 1 (value:{i}) out of range for {nat} species.'
            raise ValueError(msg)

          spec = self.uspecies[i-1]
          nstates = len(self.occ_states[spec])
          if j > nstates:
            msg = f'hubbard_occ index 2 (value:{j}) out of range, {nstates} states listed for {spec}.'
            raise ValueError(msg)

          state = self.occ_states[spec][j-1]
          self.occ_values[f'{spec}-{state}'] = float(self.blocks['system'][s])
          self.hubbard_occ[s] = self.blocks['system'][s]


  def set_hubbard_parameters ( self, hubbard ):

    htype = type(hubbard)
    if htype in [list, tuple]:

      for h in hubbard:
        ele,occ = h.split('-')
        self.uVals[h] = 0.01
        if ele not in self.occ_states:
          self.occ_states[ele] = []
        self.occ_states[ele].append(occ)

    elif htype is dict:

      for k,v in hubbard.items():

        self.uVals[k] = 0.01
        if v is None:
          continue

        try:

          vtype = type(v)
          if vtype not in [list, tuple]:
            self.uVals[k] = float(v)

          else:
            if len(v) >= 1 and v[0] is not None:
              self.uVals[k] = v[0]

            if len(v) >= 2 and v[1] is not None:

              ele,occ = k.split('-')
              if ele not in self.occ_states:
                self.occ_states[ele] = [occ]

              elif occ not in self.occ_states[ele]:
                self.occ_states[ele].append(occ)

              i = 1 + self.uspecies.index(ele)
              j = 1 + self.occ_states[ele].index(occ)

              key = f'hubbard_occ({i},{j})'
              self.hubbard_occ[key] = float(v[1])

        except Exception as e:
          print('Dictionary values should either be the initial U value, or a list/tuple containing (initial U, hubbard occupation).')
          raise e

    else:
      msg = 'Input type should be either list or dict.'
      raise TypeError(msg)


  def optimize_hubbard_U ( self, convergence_threshold=0.01 ):

    print('\nBeginning self-consistent loop.\n')
    itr = 0
    converged = False
    while not converged:

      itr += 1
      print(f'Iteration #{itr}\n')

      self.run_dft(self.prefix, self.uspecies, self.uVals)

      save_prefix = self.blocks['control']['prefix'].strip('"').strip("'")
      self.run_paoflow(self.prefix, save_prefix)

      new_U = self.run_acbn0(self.prefix)

      converged = True
      print('\nNew U values:')
      for k,v in new_U.items():
        print(f'  {k} : {v}')
        if converged and np.abs(self.uVals[k]-v) > convergence_threshold:
          converged = False
      print('', flush=True)

      self.uVals = new_U


  def acbn0 ( self, fname ):

    data = {}
    with open(fname, 'r') as f:
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

    gauss_basis = self.getbasis(self.basis, species, lattice, coords)
    nbasis = len(gauss_basis)

    fpath = './output'
    kpnts,kwght,Sks,Hks_up,Hks_dw = self.read_ham_data(fpath, nspin)

    dk,nlm = self.Dk(basis_dm, basis_2e, Hks_up, Sks)
    nlm = self.Nmm(nlm, Hks_up, kwght)
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
      dk_dn,nlmd = self.Dk(basis_dm, basis_2e, Hks_dw, Sks)
      nlmd = self.Nmm(nlmd, Hks_dw, kwght)

      Naa,Nbb,Nab = 0., 0., 0.
      for i1 in range(nnlm):
        for i2 in range(nnlm):
          Nab += nlm[i1] * nlmd[i2]
          if i1 != i2:
            Naa += nlm[i1] * nlm[i2]
            Nbb += nlmd[i1] * nlmd[i2]
      den_U = Naa.real + Nbb.real + 2*Nab.real
      den_J = Naa.real + Nbb.real

    DR_0 = self.DR(dk, kwght)

    DR_0_dn = DR_0
    if nspin == 2:
      DR_0_dn = self.DR(dk_dn, kwght)

    num_U,num_J = self.hartree_energy(DR_0, DR_0_dn, gauss_basis, basis_2e)

    hartree_to_eV = 27.211396132
    U = U_eff = hartree_to_eV * num_U / den_U
    if den_J == 0:
      J = 'Inf'
    else:
      J = hartree_to_eV * num_J / den_J
      U_eff -= J

    with open(f'{sym}_UJ.txt', 'w') as f:
      f.write(f'U : {U}\nJ : {J}\nU_eff : {U_eff}\n')


  def exec_command ( self, command ):
    from os import system

    print(f'Starting Process: {command}', flush=True)
    return system(command)


  def exec_QE ( self, executable, fname ):
    from os.path import join

    exe = join(self.qpath, executable)
    fout = fname.replace('in', 'out')
    
    command = f'{self.mpi_qe} {exe} {self.qoption} < {fname} > {fout}'
    return self.exec_command(command)


  def exec_PAOFLOW ( self ):
    from os.path import join

    prefix = join(self.ppath, 'python')
    command = f'{self.mpi_python} {prefix} acbn0.py > paoflow.out'
    return self.exec_command(command)


  def hubbard_card ( self ):

    if len(self.uVals) == 0:
      msg = 'No U found. Add U to the template inputfiles or with set_hubbard_parameters.'
      raise ValueError(msg)

    card = [self.hubbard_tag]
    for k,v in self.uVals.items():
      card.append(' U {} {}'.format(k, v))

    return card


  def run_dft ( self, prefix, species, uVals ):
    from .defs.file_io import create_atomic_inputfile
    from .defs.file_io import struct_from_inputfile_QE

    blocks,cards = struct_from_inputfile_QE(f'{prefix}.scf.in')
    cards['HUBBARD'] = self.hubbard_card()
    for k,v in self.hubbard_occ.items():
      blocks['system'][k] = v
    create_atomic_inputfile('scf', blocks, cards)

    blocks,cards = struct_from_inputfile_QE(f'{prefix}.nscf.in')
    cards['HUBBARD'] = self.hubbard_card()
    for k,v in self.hubbard_occ.items():
      blocks['system'][k] = v
    create_atomic_inputfile('nscf', blocks, cards)

    blocks,cards = struct_from_inputfile_QE(f'{prefix}.projwfc.in')
    create_atomic_inputfile('projwfc', blocks, cards)

    executables = {'scf':'pw.x', 'nscf':'pw.x', 'projwfc':'projwfc.x'}
    for c in ['scf', 'nscf', 'projwfc']:
      ecode = self.exec_QE(executables[c], f'{c}.in')


  def run_paoflow ( self, prefix, save_prefix ):
    from .defs.file_io import create_acbn0_inputfile
    from os import system

    fstr = f'{prefix}_PAO_bands' + '{}.in'
    calcs = []
    if self.nspin == 1:
      calcs.append(fstr.format('')) 
    else:
      calcs.append(fstr.format('_up'))
      calcs.append(fstr.format('_down'))

    create_acbn0_inputfile(save_prefix, self.pthr)
    ecode = self.exec_PAOFLOW()


  def read_cell_atoms ( self, fname ):

    lines = None
    with open(fname, 'r') as f:
      lines = f.readlines()

    il = 0
    while 'lattice parameter (alat)' not in lines[il]:
      il += 1
    alat = float(lines[il].split()[4])

    while 'number of atoms/cell' not in lines[il]:
      il += 1
    nat = int(lines[il].split()[4])

    while 'crystal axes:' not in lines[il]:
      il += 1
    il += 1
    lattice = np.array([[float(v) for v in lines[il+i].split()[3:6]] for i in range(3)])

    while 'site n.' not in lines[il]:
      il += 1
    il += 1
    species = []
    positions = np.empty((nat,3), dtype=float)
    for i in range(nat):
      ls = lines[il+i].split()
      species.append(ls[1])
      positions[i,:] = np.array([float(v) for v in ls[6:9]])

    lattice *= alat
    positions *= alat
    return lattice,species,positions


  def hubbard_orbital ( self, ele ):

    orb = ele[-1]
    orbitals = {'s':0, 'p':1, 'd':2}

    if orb in orbitals:
      return orbitals[orb]

    else:
      raise Exception(f'Element {ele} has no defined Hubbard orbital')


  def run_acbn0 ( self, prefix ):
    import re

    lattice,species,positions = self.read_cell_atoms('scf.out')
    lattice_string = ','.join([str(v) for a in lattice for v in a])
    position_string = ','.join([str(v) for a in positions for v in a])
    species_string = ','.join(species)

    sind = 0
    state_lines = open('projwfc.out', 'r').readlines()
    while 'state #' not in state_lines[sind]:
      sind += 1
    send = sind
    while 'state #' in state_lines[send]:
      send += 1
    state_lines = state_lines[sind:send]

    uVals = {}
    #for s in list(set(species)):
    for k,v in self.uVals.items():

      ostates = []
      ustates = []
      s = k.split('-')[0]
      horb = self.hubbard_orbital(k)
      for n,sl in enumerate(state_lines):
        stateN = re.findall('\(([^\)]+)\)', sl)
        oele = stateN[0].strip()
        oL = int(re.split('=| ', stateN[1])[1])
        if s in oele and oL == horb:
          ostates.append(n)
          if s == oele:
            ustates.append(n)
      sstates = [ustates[0]]
      for i,us in enumerate(ustates[1:]):
        if us == 1 + sstates[i]:
          sstates.append(us)
        else:
          break

      basis_dm = ','.join(list(map(str,ostates)))
      basis_2e = ','.join(list(map(str,sstates)))

      fname = f'{prefix}_acbn0_infile_{k}.txt'
      with open(fname, 'w') as f:
        nspin = self.nspin
        f.write(f'symbol = {k}\n')
        f.write(f'latvects = {lattice_string}\n')
        f.write(f'coords = {position_string}\n')
        f.write(f'atlabels = {species_string}\n')
        f.write(f'nspin = {nspin}\n')
        f.write(f'fpath = ./\n')
        f.write(f'outfile = {prefix}_acbn0_outfile_{k}.txt\n')
        f.write(f'reduced_basis_dm = {basis_dm}\n')
        f.write(f'reduced_basis_2e = {basis_2e}\n')

      self.acbn0(fname)
      with open(f'{k}_UJ.txt', 'r') as f:
        lines = f.readlines()
        uVals[k] = float(lines[2].split(':')[1])

    return uVals


  def getbasis ( self, basis, species, lattice, coords ):
    from .defs.pyints import CGBF
 
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


  def Dk ( self, basis_dm, basis_2e, Hks, Sks ):
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


  def Nmm ( self, nlm, Hks, kwght ):

    lm_size,nbasis,nkp = nlm.shape
    nlm_aux = np.zeros((lm_size,nbasis), dtype=complex)
    for ik,wght in enumerate(kwght):
      nlm_aux += wght * nlm[:,:,ik]

    return np.sum(nlm_aux/np.sum(kwght), axis=1)


  def DR ( self, Dk, kwght ):

    nawf = Dk.shape[0]
    nkpnts = kwght.shape[0]

    D = np.zeros((nawf,nawf), dtype=complex)
    for ik,wght in enumerate(kwght):
      D += wght * Dk[:,:,ik]

    return D.real/np.sum(kwght)


  def hartree_energy ( self, DR_up, DR_dn, basis, basis_2e ):
    import itertools

    tmp_U,tmp_J = 0,0
    ind = list(itertools.product(basis_2e,repeat=4))

    for k,l,m,n in ind:
      int_U = self.coulomb(basis[m], basis[n], basis[k], basis[l])
      int_J = self.coulomb(basis[m], basis[k], basis[n], basis[l])

      a_b = DR_up[m,n]*DR_up[k,l] + DR_dn[m,n]*DR_dn[k,l]
      ab_ba = DR_dn[m,n]*DR_up[k,l] + DR_up[m,n]*DR_dn[k,l]

      tmp_U += int_U * (a_b + ab_ba)
      tmp_J += int_J * a_b

    return tmp_U, tmp_J

  def read_ham_data ( self, fpath, nspin ):
    from os.path import join

    kpnts = np.loadtxt(open(join(fpath,'k.txt'),'r'))
    kwght = np.loadtxt(open(join(fpath,'wk.txt'),'r'))

    if len(kpnts.shape) == 1:
      kpnts = np.array([kpnts])
      kwght = np.array([kwght])
    nkpnts = kpnts.shape[0]

    kovp = np.load(join(fpath,'kovp.npy'))
    nbasis = int(np.sqrt(kovp.shape[0]/nkpnts))
    kovp = kovp.reshape((nbasis,nbasis,nkpnts))

    kham_up = kham_dn = None
    for ispin in range(nspin):

      fname = 'kham'
      if nspin == 2:
        fname += '_up' if ispin==0 else '_dn'
      fname += '.npy'

      kham = np.load(join(fpath,fname))
      kham = kham.reshape((nbasis,nbasis,nkpnts))

      if ispin == 0:
        kham_up = kham
      elif ispin == 1:
        kham_dn = kham

    return kpnts, kwght, kovp, kham_up, kham_dn


  def coulomb ( self, a, b, c, d ):
    from .defs.pyints import contr_coulomb
    ' Coulomb interaction between four contracted Gaussians '
    return contr_coulomb(a.pexps,a.pcoefs,a.pnorms,a.origin,a.powers,
                         b.pexps,b.pcoefs,b.pnorms,b.origin,b.powers,
                         c.pexps,c.pcoefs,c.pnorms,c.origin,c.powers,
                         d.pexps,d.pcoefs,d.pnorms,d.origin,d.powers)
