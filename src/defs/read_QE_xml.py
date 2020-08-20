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

  Ry2eV = 13.60569193

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

  mpg = elem.find('band_structure/starting_k_points/monkhorst_pack').attrib
  k1,k2,k3 = int(mpg['k1']),int(mpg['k2']),int(mpg['k3'])
  nk1,nk2,nk3 = int(mpg['nk1']),int(mpg['nk2']),int(mpg['nk3'])
  if verbose: print('Monkhorst and Pack grid:',nk1,nk2,nk3,k1,k2,k3)

  a_vectors,b_vectors = [],[]
  basis_elem = elem.find('basis_set')
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

  ef_text = None
  try:
    ef_text = elem.find('band_structure/fermi_energy').text
  except:
    try:
      ef_text = elem.find('band_structure/highestOccupiedLevel').text
    except Exception as e:
      print('Fermi energy not located in QE data file.')
      raise e
  Efermi = float(ef_text)*Ry2eV

  dftMag = False
  if elem.find('magnetization/do_magnetization').text == 'true':
    dftMag = True

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

  attrs = [('qe_version',qe_version),('alat',alat),('nk1',nk1),('nk2',nk2),('nk3',nk3),('natoms',natoms),('Efermi',Efermi),('omega',omega),('dftSO',dftSO),('dftMAG',dftMag)]
  for s,v in attrs:
    attr[s] = v

  spec = [(species[i],pseudos[i]) for i in range(len(species))]
  arrys = [('tau',tau),('atoms',atoms),('species',spec),('a_vectors',a_vectors),('b_vectors',b_vectors),('equiv_atom',eq_atoms),('sym_rot',sym_rot),('sym_shift',shifts),('sym_info',sym_info),('sym_TR',time_rev)]
  for s,v in arrys:
    arry[s] = v


def parse_qe_data_file ( data_controller, fname ):
  '''
  Parse the data_filw.xml file produced by earlier versions of Quantum Espresso.
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

  Ry2eV = 13.60569193

  tree = ET.parse(fname)
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

  if root.find('MAGNETIZATION_INIT').text.replace('\n','') == 'T':
    raise NotImplementedError('Two Fermi energies unhandled')
  Efermi = float(root.find('BAND_STRUCTURE_INFO/FERMI_ENERGY').text)*Ry2eV

  dftMag = False
  if root.find('SPIN/SPIN-ORBIT_DOMAG').text.replace('\n','') == 'T':
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
  attrs = [('qe_version',qe_version),('alat',alat),('nk1',nk1),('nk2',nk2),('nk3',nk3),('natoms',natoms),('Efermi',Efermi),('omega',omega),('dftSO',dftSO),('dftMAG',dftMag)]
  for s,v in attrs:
    attr[s] = v

  spec = [(species[i],pseudos[i]) for i in range(len(species))]
  arrys = [('tau',tau),('species',spec),('a_vectors',a_vectors),('b_vectors',b_vectors),('atoms',atoms),('equiv_atom',eq_atoms),('sym_rot',sym_rot),('sym_shift',shifts),('sym_info',sym_info),('sym_TR',time_rev)]

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
  non_ortho = attr['non_ortho']

  tree = ET.parse(fname)
  root = tree.getroot()
  qe_version = attr['qe_version']

  Ry2eV = 13.60569193
  Efermi = attr['Efermi']

  nawf = None
  nspin = None
  nelec = None
  nbnds = None
  nkpnts = None

  eigs = None
  kpnts = None
  overlaps = None
  kpnt_weights = None
  wavefunctions = None

  elem = root.find('HEADER')
  if qe_version > 6.5:
    header = elem.attrib
    nkpnts = int(header['NUMBER_OF_K-POINTS'])
    nspin = int(header['NUMBER_OF_SPIN_COMPONENTS'])
    nbnds = int(header['NUMBER_OF_BANDS'])
    nawf = int(header['NUMBER_OF_ATOMIC_WFC'])
    nelec = int(float(header['NUMBER_OF_ELECTRONS']))
  else:
    nbnds = int(elem.find('NUMBER_OF_BANDS').text)
    nkpnts = int(elem.find('NUMBER_OF_K-POINTS').text)
    nspin = int(elem.find('NUMBER_OF_SPIN_COMPONENTS').text)
    nawf = int(elem.find('NUMBER_OF_ATOMIC_WFC').text)
    nelec = int(float(elem.find('NUMBER_OF_ELECTRONS').text))

  if nspin == 4:
    nspin = 1

  kpnts = np.empty((nkpnts,3), dtype=float)
  eigs = np.empty((nbnds,nkpnts,nspin), dtype=float)
  kpnt_weights = np.empty((nkpnts), dtype=float)
  wavefunctions = np.empty((nbnds,nawf,nkpnts,nspin), dtype=complex)
  if non_ortho:
    oshape = (nawf,nbnds,nkpnts)
    if nspin > 1:
      oshape = oshape + (nspin,)
    overlaps = np.empty(oshape, dtype=complex)

  def add_overlap ( i0, i1, i2, iS, val ):
    if nspin > 1:
      overlaps[i1,i2,i0,iS] = val
    else:
      overlaps[i1,i2,i0] = val

  if rank == 0 and verbose:
    print('Number of kpoints: {0:d}'.format(nkpnts))
    print('Number of spin components: {0:d}'.format(nspin))
    print('Number of electrons: {0:d}'.format(nelec))
    print('Number of bands: {0:d}'.format(nbnds))
    print('Number of atomic wavefunctions: {0:d}'.format(nawf))

  if qe_version > 6.5:
    elem = root.find('EIGENSTATES')
    ekpnts = elem.findall('K-POINT')
    eE = elem.findall('E')
    ewfc = elem.findall('PROJS')
    for i in range(nkpnts):
      kpnt_weights[i] = float(ekpnts[i].attrib['Weight'])
      kpnts[i,:] = np.array(ekpnts[i].text.split())
      for j,v in enumerate(eE[i].text.split()):
        eigs[j,i,:] = float(v)
      for j,proj in enumerate(ewfc[i].findall('ATOMIC_WFC')):
        ind = int(proj.attrib['index'])-1
        spin = int(proj.attrib['spin'])-1
        text = [float(v) for v in proj.text.split()]
        for k in range(len(text)//2):
          k2 = 2*k
          wavefunctions[k,ind,i,spin] = complex(text[k2],text[k2+1])
    eigs = eigs * Ry2eV - Efermi

    if non_ortho:
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
          add_overlap(i, i1, i2, ispin, complex(v1,v2))

  else:
    elem = root.find('K-POINTS')
    text = elem.text.split()
    for i in range(len(text)):
      i1,i2 = i//3,i%3
      kpnts[i1,i2] = float(text[i])

    elem = root.find('WEIGHT_OF_K-POINTS')
    kpnt_weights[:] = np.array(elem.text.split(), dtype=float)

    elem = root.find('EIGENVALUES')
    for i,k in enumerate(elem):
      for ispin in range(nspin):
        key = 'EIG' if nspin==1 else 'EIG.%d'%(ispin+1)
        text = k.find(key).text.split()
        eigs[:,i,ispin] = np.array(text, dtype=float)
    eigs = eigs * Ry2eV - Efermi


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

    if non_ortho:
      elem = root.find('OVERLAPS')
      for i,kpnt in enumerate(elem):
        for j,ovp in enumerate(kpnt):
          ispin = int(ovp.tag.split('.')[1]) - 1
          text = ovp.text.replace(',', ' ').split()
          for k in range(len(text)//2):
            k0,k1 = k//nbnds,k%nbnds
            k2 = 2*k
            v1,v2 = float(text[k2]),float(text[k2+1])
            add_overlap(i, k0, k1, ispin, complex(v1,v2))

  attrs = [('nawf',nawf),('nspin',nspin),('nelec',nelec),('nbnds',nbnds),('nkpnts',nkpnts)]
  for s,v in attrs:
    attr[s] = v

  arrys = [('kpnts',kpnts),('kpnts_wght',kpnt_weights),('my_eigsmat',eigs),('U',wavefunctions)]
  if non_ortho:
    arrys += [('Sks',overlaps)]
  for s,v in arrys:
    arry[s] = v
