#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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

import xml.etree.cElementTree as ET
from mpi4py import MPI
import numpy as np

def parse_qe_data_file_schema ( data_controller, fname ):
  '''
  Parse the data_file_schema.xml file produced by Quantum Espresso.
  Populated the DataController object with all necessay information.

  Arugments:
    data_controller (DataController): Data controller to populate
    fname (str): Path and name of the xml file.
  '''
  import re

  arry,attr = data_controller.data_dicts()
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  verbose = attr['verbose']

  Hart2eV = 2 * 13.60569193

  tree = ET.parse(fname)
  root = tree.getroot()

  qe_version = float(re.findall('\d+\.\d+', root.find('general_info/creator').attrib['VERSION'])[0])

  dftSO = True if root.find('input/spin/spinorbit').text=='true' else False

  elem = root.find('output')

  species,pseudos = [],[]
  lspecies = elem.findall('atomic_species/species')
  for s in lspecies:
    species.append(s.attrib['name'])
    pseudos.append(s.find('pseudo_file').text)

  astruct = elem.find('atomic_structure')
  alat = float(astruct.attrib['alat'])
  natoms = int(astruct.attrib['nat'])

  basis_elem = elem.find('basis_set')
  ecutwfc = float(basis_elem.find('ecutwfc').text)*2
  ecutrho = float(basis_elem.find('ecutrho').text)*2 # it is read in Ha and must be converted to Ry

  a_vectors,b_vectors = [],[]
  vec_array = lambda n,e,s : np.array(e.find('%s%d'%(s,n)).text.split(), dtype=float)
  for i in range(1, 4):
    a_vectors.append(vec_array(i,astruct,'cell/a'))
    b_vectors.append(vec_array(i,basis_elem,'reciprocal_lattice/b'))
  a_vectors = np.array(a_vectors)/alat
  b_vectors = np.array(b_vectors)

  atoms = []
  tau = np.empty((natoms,3), dtype=float)
  latoms = elem.findall('atomic_structure/atomic_positions/atom')
  for i in range(natoms):
    atoms.append(latoms[i].attrib['name'].strip())
    tau[i,:] = np.array(latoms[i].text.split(), dtype=float)

  insulator = True
  smearing = elem.find('band_structure/smearing')
  if smearing is not None:
    insulator = False

  dftMag = False
  mag_elem = elem.find('magnetization/do_magnetization')
  if mag_elem is not None:
    if mag_elem.text == 'true':
      dftMag = True

  mpg = elem.find('band_structure/starting_k_points/monkhorst_pack').attrib
  k1,k2,k3 = int(mpg['k1']),int(mpg['k2']),int(mpg['k3'])
  nk1,nk2,nk3 = int(mpg['nk1']),int(mpg['nk2']),int(mpg['nk3'])
  if verbose: print('Monkhorst and Pack grid:',nk1,nk2,nk3,k1,k2,k3)

  bs = elem.find('band_structure')
  nkpnts = int(bs.find('nks').text)
  nelec = int(float(bs.find('nelec').text))
  nawf = int(bs.find('num_of_atomic_wfc').text)
  lsda = True if bs.find('lsda').text=='true' else False
  nspin = 2 if lsda else 1

  if nspin == 1:
    nbnds = int(bs.find('nbnd').text)
  else:
    nb_up = int(bs.find('nbnd_up').text)
    nb_dw = int(bs.find('nbnd_dw').text)
    if nb_up != nb_dw and verbose: print('nbnd_up (%d) != nbnd_dw (%d)'%(nbnd_up,nbnd_dw))
    nbnds = np.max([nb_up, nb_dw])

  if rank == 0 and verbose:
    print('Number of kpoints: {0:d}'.format(nkpnts))
    print('Number of electrons: {0:d}'.format(nelec))
    print('Number of bands: {0:d}'.format(nbnds))

  ef_text = None
  try:
    ef_text = bs.find('fermi_energy').text
  except:
    try:
      ef_text = bs.find('highestOccupiedLevel').text
    except Exception as e:
      print('Fermi energy not located in QE data file.')
      raise e
  Efermi = float(ef_text) * Hart2eV

  kpnts = np.empty((nkpnts,3), dtype=float)
  kpnt_weights = np.empty((nkpnts), dtype=float)
  eigs = np.empty((nbnds,nkpnts,nspin), dtype=float)

  kse = bs.findall('ks_energies')
  for i,k in enumerate(kse):
    kp = k.find('k_point')
    kpnts[i,:] = np.array(kp.text.split(), dtype=float)
    kpnt_weights[i] = float(kp.attrib['weight'])
    eigs[:,i,:] = np.array(k.find('eigenvalues').text.split(), dtype=float).reshape((nspin,nbnds)).T
  eigs = eigs * Hart2eV - Efermi

  sym_rot,shifts = [],[]
  eq_atoms,sym_info,time_rev = [],[],[]

  lsyms = elem.findall('symmetries/symmetry')
  for i,sym in enumerate(lsyms):
    info = sym.find('info')
    if info.text == 'crystal_symmetry':
      sym_info.append(info.attrib['name'])

      shift_text = sym.find('fractional_translation').text
      shifts.append([float(v) for v in shift_text.split()])

      eq_atom_text = sym.find('equivalent_atoms').text
      eq_atoms.append([int(v) for v in eq_atom_text.split()])

      if 'time_reversal' in info.attrib:
        time_rev.append(info.attrib['time_reversal'])

      rots = []
      for rot in sym.find('rotation').text.split('\n'):
        line = rot.split()
        if len(line) != 0:
          rots.append([float(v) for v in rot.split()])
      sym_rot.append(rots)

  shifts = np.array(shifts)
  eq_atoms = np.array(eq_atoms) - 1
  sym_info = np.array(sym_info)

  sym_rot = np.transpose(np.array(sym_rot), axes=(0,2,1))

  if len(time_rev) == 0:
    time_rev = np.zeros(sym_info.shape[0], dtype=bool)
  else:
    time_rev = np.array([True if v=='true' else False for v in time_rev])

  omega = alat**3 * a_vectors[0,:].dot(np.cross(a_vectors[1,:],a_vectors[2,:]))

  attrs = [('qe_version',qe_version),('alat',alat),('nk1',nk1),('nk2',nk2),('nk3',nk3),('ok1',k1),('ok2',k2),('ok3',k3),('natoms',natoms),('ecutrho',ecutrho),('ecutwfc',ecutwfc),('nawf',nawf),('nbnds',nbnds),('nspin',nspin),('nkpnts',nkpnts),('nelec',nelec),('Efermi',Efermi),('omega',omega),('dftSO',dftSO),('dftMAG',dftMag),('insulator',insulator)]
  for s,v in attrs:
    attr[s] = v

  spec = [(species[i],pseudos[i]) for i in range(len(species))]
  arrys = [('tau',tau),('atoms',atoms),('species',spec),('a_vectors',a_vectors),('b_vectors',b_vectors),('equiv_atom',eq_atoms),('kpnts',kpnts),('kpnts_wght',kpnt_weights),('my_eigsmat',eigs),('sym_rot',sym_rot),('sym_shift',shifts),('sym_info',sym_info),('sym_TR',time_rev)]
  for s,v in arrys:
    arry[s] = v


def parse_qe_data_file ( data_controller, fpath, fname ):
  '''
  Parse the data_file.xml file produced by earlier versions of Quantum Espresso.
  Populated the DataController object with all necessay information.

  Arugments:
    data_controller (DataController): Data controller to populate
    fname (str): Path and name of the xml file.
  '''
  import re
  from os.path import join

  arry,attr = data_controller.data_dicts()
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  verbose = attr['verbose']

  Hart2eV = 2 * 13.60569193

  tree = ET.parse(join(fpath,fname))
  root = tree.getroot()

  qe_version = float(re.findall('\d+\.\d+', root.find('HEADER/CREATOR').attrib['VERSION'])[0])

  dftSO = root.find('SPIN/SPIN-ORBIT_CALCULATION').text.replace('\n','').strip()
  dftSO = True if dftSO=='T' else False

  species,pseudos = [],[]
  alat = float(root.find('CELL/LATTICE_PARAMETER').text)
  nspec = int(root.find('IONS/NUMBER_OF_SPECIES').text)
  natoms = int(root.find('IONS/NUMBER_OF_ATOMS').text)

  for i in range(1,nspec+1):
    spec_elem = root.find('IONS/SPECIE.%d'%i)
    species.append(spec_elem.find('ATOM_TYPE').text.replace('\n','').strip())
    pseudos.append(spec_elem.find('PSEUDO').text.replace('\n','').split()[0])

  mpg = root.find('BRILLOUIN_ZONE/MONKHORST_PACK_OFFSET').attrib
  k1,k2,k3 = int(mpg['k1']),int(mpg['k2']),int(mpg['k3'])

  mpg = root.find('BRILLOUIN_ZONE/MONKHORST_PACK_GRID').attrib
  nk1,nk2,nk3 = int(mpg['nk1']),int(mpg['nk2']),int(mpg['nk3'])

  if verbose: print('Monkhorst and Pack grid:',nk1,nk2,nk3,k1,k2,k3)

  ecutrho = float(root.find('PLANE_WAVES/RHO_CUTOFF').text.strip())

  a_vectors,b_vectors = [],[]
  a_elem = root.find('CELL/DIRECT_LATTICE_VECTORS')
  b_elem = root.find('CELL/RECIPROCAL_LATTICE_VECTORS')
  vec_array = lambda n,e,s : np.array(e.find('%s%d'%(s,n)).text.split(), dtype=float)
  for i in range(1, 4):
    a_vectors.append(vec_array(i,a_elem,'a'))
    b_vectors.append(vec_array(i,b_elem,'b'))
  a_vectors = np.array(a_vectors)/alat
  b_vectors = np.array(b_vectors)

  atoms = []
  tau = np.empty((natoms,3), dtype=float)
  for i in range(natoms):
    latom = root.find('IONS/ATOM.%d'%(i+1))
    atoms.append(latom.attrib['SPECIES'].strip())
    tau[i,:] = np.array(latom.attrib['tau'].split(), dtype=float)

  insulator = False if root.find('OCCUPATIONS/SMEARING_METHOD').text.strip()=='T' else True

  if root.find('MAGNETIZATION_INIT').text.strip() == 'T':
    raise NotImplementedError('Two Fermi energies unhandled')

  bs = root.find('BAND_STRUCTURE_INFO')
  nkpnts = int(bs.find('NUMBER_OF_K-POINTS').text.strip())
  nawf = int(bs.find('NUMBER_OF_ATOMIC_WFC').text.strip())
  nbnds = int(bs.find('NUMBER_OF_BANDS').text.strip())
  nelec = int(float(bs.find('NUMBER_OF_ELECTRONS').text.strip()))
  Efermi = float(bs.find('FERMI_ENERGY').text.strip()) * Hart2eV
  lsda = True if root.find('SPIN/LSDA').text.strip()=='T' else False
  nspin = 2 if lsda else 1

  if rank == 0 and verbose:
    print('Number of kpoints: {0:d}'.format(nkpnts))
    print('Number of electrons: {0:d}'.format(nelec))
    print('Number of bands: {0:d}'.format(nbnds))

  kpnts = np.empty((nkpnts,3), dtype=float)
  kpnt_weights = np.empty((nkpnts), dtype=float)
  eigs = np.empty((nbnds,nkpnts,nspin), dtype=float)

  ev = root.find('EIGENVALUES')
  data_tags = ['DATAFILE'] if nspin==1 else ['DATAFILE.1', 'DATAFILE.2']
  for i in range(nkpnts):
    k = ev.find('K-POINT.%d'%(i+1))
    kpnts[i,:] = np.array(k.find('K-POINT_COORDS').text.split(), dtype=float)
    kpnt_weights[i] = float(k.find('WEIGHT').text)
    for ispin,ftag in enumerate(data_tags):
      efname = k.find(ftag).attrib['iotk_link']
      eroot = ET.parse(join(fpath,efname)).getroot()
      eigs[:,i,ispin] = np.array(eroot.find('EIGENVALUES').text.split(), dtype=float)
  eigs = eigs * Hart2eV - Efermi

  dftMag = False
  if root.find('SPIN/SPIN-ORBIT_DOMAG').text.replace('\n','').strip() == 'T':
    dftMag = True

  sym_rot,shifts = [],[]
  eq_atoms,sym_info,time_rev = [],[],[]

  lsyms = root.find('SYMMETRIES')
  nsyms = int(lsyms.find('NUMBER_OF_SYMMETRIES').text)
  for i in range(1,nsyms+1):
    sym = lsyms.find('SYMM.%d'%i)

    shift_text = sym.find('FRACTIONAL_TRANSLATION').text
    shifts.append([float(v) for v in shift_text.split()])

    eq_atom_text = sym.find('EQUIVALENT_IONS').text
    eq_atoms.append([int(v) for v in eq_atom_text.split()])

    info = sym.find('INFO')
    sym_info.append(info.attrib['NAME'])
    time_rev.append(int(info.attrib['T_REV']))

    rots = []
    for rot in sym.find('ROTATION').text.split('\n'):
      line = rot.split()
      if len(line) != 0:
        rots.append([float(v) for v in line])
    sym_rot.append(rots)

  shifts = np.array(shifts)
  eq_atoms = np.array(eq_atoms) - 1

  sym_rot = np.transpose(np.array(sym_rot), axes=(0,2,1))

  sym_info = np.array(sym_info)
  time_rev = np.array(time_rev, dtype=bool)

  # Compute the cell volume
  omega = alat**3 * a_vectors[0,:].dot(np.cross(a_vectors[1,:],a_vectors[2,:]))

  # Add the attributes and arrays to the data controller
  attrs = [('qe_version',qe_version),('alat',alat),('nk1',nk1),('nk2',nk2),('nk3',nk3),('ok1',k1),('ok2',k2),('ok3',k3),('natoms',natoms),('ecutrho',ecutrho),('nawf',nawf),('nbnds',nbnds),('nspin',nspin),('nkpnts',nkpnts),('nelec',nelec),('Efermi',Efermi),('omega',omega),('dftSO',dftSO),('dftMAG',dftMag),('insulator',insulator)]
  for s,v in attrs:
    attr[s] = v

  spec = [(species[i],pseudos[i]) for i in range(len(species))]
  arrys = [('tau',tau),('atoms',atoms),('species',spec),('a_vectors',a_vectors),('b_vectors',b_vectors),('equiv_atom',eq_atoms),('kpnts',kpnts),('kpnts_wght',kpnt_weights),('my_eigsmat',eigs),('sym_rot',sym_rot),('sym_shift',shifts),('sym_info',sym_info),('sym_TR',time_rev)]
  for s,v in arrys:
    arry[s] = v


def parse_qe_atomic_proj ( data_controller, fname ):
  '''
  Parse the atomic_proj.xml file produced by Quantum Espresso.
  Populated the DataController object with all necessay information.

  Arugments:
    data_controller (DataController): Data controller to populate
    fname (str): Path and name of the xml file.
  '''

  arry,attr = data_controller.data_dicts()
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  verbose = attr['verbose']
  acbn0 = attr['acbn0']

  tree = ET.parse(fname)
  root = tree.getroot()
  qe_version = attr['qe_version']

  Ry2eV = 13.60569193
  Efermi = attr['Efermi']

  elem = root.find('HEADER')
  if qe_version > 6.5:
    header = elem.attrib
    nkpnts = int(header['NUMBER_OF_K-POINTS'])
    nspin = int(header['NUMBER_OF_SPIN_COMPONENTS'])
    nbnds = int(header['NUMBER_OF_BANDS'])
    nawf = int(header['NUMBER_OF_ATOMIC_WFC'])
  else:
    nbnds = int(elem.find('NUMBER_OF_BANDS').text)
    nkpnts = int(elem.find('NUMBER_OF_K-POINTS').text)
    nspin = int(elem.find('NUMBER_OF_SPIN_COMPONENTS').text)
    nawf = int(elem.find('NUMBER_OF_ATOMIC_WFC').text)

  if nspin == 4:
    nspin = 1

  wavefunctions = np.empty((nbnds,nawf,nkpnts,nspin), dtype=complex)
  overlaps = np.empty((nawf,nbnds,nkpnts), dtype=complex) if acbn0 else None

  if qe_version > 6.5:

    elem = root.find('EIGENSTATES')
    ewfc = elem.findall('PROJS')
    for ispin in range(nspin):
      for i in range(nkpnts):
        ik = ispin*nkpnts + i
        for j,proj in enumerate(ewfc[ik].findall('ATOMIC_WFC')):
          ind = int(proj.attrib['index'])-1
          spin = int(proj.attrib['spin'])-1
          text = [float(v) for v in proj.text.split()]
          for k in range(len(text)//2):
            k2 = 2*k
            wavefunctions[k,ind,i,ispin] = complex(text[k2],text[k2+1])

    if acbn0:
      elem = root.find('OVERLAPS')
      for i,ovp in enumerate(elem.findall('OVPS')):
        dim = int(ovp.attrib['dim'])
        ispin = int(ovp.attrib['spin'])-1
        text = ovp.text.split()
        for j in range(len(text)//2):
          j2 = 2*j
          i1 = j//dim
          i2 = j%dim
          v1,v2 = float(text[j2]),float(text[j2+1])
          overlaps[i1,i2,i] = complex(v1,v2)

  else:

    def read_wf ( kpnt, i, ispin ):
      for j,wf in enumerate(kpnt):
        text = wf.text.replace(',', ' ').split()
        wft = np.array(text, dtype=float).reshape((nbnds,2))
        wavefunctions[:,j,i,ispin] = wft[:,0] + 1j*wft[:,1]

    elem = root.find('PROJECTIONS')
    for i,kpnt in enumerate(elem):
      if nspin == 1:
        read_wf(kpnt, i , 0)
      else:
        for ispin in range(nspin):
          nele = kpnt.find('SPIN.%d'%(ispin+1))
          read_wf(nele, i, ispin)

    if acbn0:
      elem = root.find('OVERLAPS')
      for i,kpnt in enumerate(elem):
        for j,ovp in enumerate(kpnt):
          ispin = int(ovp.tag.split('.')[1]) - 1
          text = ovp.text.replace(',', ' ').split()
          for k in range(len(text)//2):
            k0,k1 = k//nbnds,k%nbnds
            k2 = 2*k
            v1,v2 = float(text[k2]),float(text[k2+1])
            overlaps[k0,k1,i] = complex(v1,v2)

  arrys = [('U',wavefunctions)]
  if acbn0:
    arrys += [('Sks',overlaps)]
  for s,v in arrys:
    arry[s] = v
