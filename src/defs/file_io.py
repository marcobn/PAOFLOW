from numpy import ndarray


def struct_from_outputfile_QE ( fname:str ):
  '''
  '''
  from os.path import isfile,join
  import numpy as np

  if not isfile(fname):
    msg = 'File {} does not exist.'.format(join(getcwd(),fname))
    raise FileNotFoundError(msg)

  struct = {'lunit':'bohr', 'aunit':'alat'}
  with open(fname, 'r') as f:
    lines = f.readlines()

    eL = 0
    nL = len(lines)

    try:
      struct['species'] = []
      celldm = np.empty(6, dtype=float)
      while 'bravais-lattice' not in lines[eL]:
        eL += 1
      ibrav = int(lines[eL].split()[3])

      while  'celldm' not in lines[eL]:
        eL += 1
      celldm[:3] = [float(v) for i,v in enumerate(lines[eL].split()) if i%2==1]
      celldm[3:] = [float(v) for i,v in enumerate(lines[eL+1].split()) if i%2==1]

      if ibrav != 0:
        from .lattice_format import lattice_format_QE
        struct['lattice'] = lattice_format_QE(ibrav, celldm)
      else:
        while 'crystal axes' not in lines[eL]:
          eL += 1
        coord = []
        for l in lines[eL+1:eL+4]:
          coord.append([celldm[0]*float(v) for v in l.split()[3:6]])
        struct['lattice'] = np.array(coord)

      while 'site n.' not in lines[eL]:
        eL += 1
      eL += 1
      apos = []
      while 'End' not in lines[eL] and lines[eL] != '\n':
        line = lines[eL].split()
        struct['species'].append(line[1])
        apos.append([float(v) for v in line[6:9]])
        eL += 1
      apos = celldm[0] * np.array(apos)
      struct['abc'] = apos @ np.linalg.inv(struct['lattice'])

    except Exception as e:
      print('ERROR: Could not read the QE output.')
      raise e

  return struct


def read_relaxed_coordinates_QE ( fname:str ):
  '''
    Reads relaxed atomic positions from a QE .out file. If vcrelax is set True, the crystal coordinates are also read.

    Arguments:
      fname (str): File name (including path) for the .out file
      vcrelax (bool): True reads crystal coordinates in addition to atomic positions
      read_all (bool): True forces all relax steps to be read. If EoF is encountered before 'final coordinates' the last coordinates to appear in the file are retunred. If no coordinates are found, an empty dictionary is returned.

    Returns:
      (dict): Dictionary with one or two entries - 'apos' for atomic positions and 'coord' for crystal coordinates.
  '''
  from os.path import isfile,join
  from os import getcwd
  import numpy as np
  import re

  abc = []
  cell_params = []
  struct = struct_from_outputfile_QE(fname)

  with open(fname, 'r') as f:
    lines = f.readlines()

    eL = 0
    nL = len(lines)

    try:
      def read_apos ( sind ):
        apos = []
        while lines[sind] != '\n' and not 'End final coordinates' in lines[sind]:
          apos.append([float(v) for v in lines[sind].split()[1:4]])
          sind += 1
        return sind, apos

      while eL < nL:
        while eL < nL and 'CELL_PARAMETERS' not in lines[eL] and 'ATOMIC_POSITIONS' not in lines[eL]:
          eL += 1
        if eL >= nL:
          break
        if 'ATOMIC_POSITIONS' in lines[eL]:
          unit = lines[eL].split()[1].strip('(){{}}')
          if len(unit) > 1:
            struct['aunit'] = unit
          eL,apos = read_apos(eL+1)
          abc.append(apos)
        elif 'CELL_PARAMETERS' in lines[eL]:
          coord = []
          unit = lines[eL].split()[1].strip('(){{}}')

          alat = 1
          if 'alat' in unit or len(unit) == 0:
            struct['lunit'] = 'alat'
            if 'alat' in unit:
              cpattern = re.search('\(([^\)]+)\)', lines[eL])
              if cpattern is not None:
                alat = float(cpattern.group(0)[1:-1].split('=')[1])
          else:
            struct['lunit'] = unit
          for l in lines[eL+1:eL+4]:
            coord.append(alat*np.array([float(v) for v in l.split()]))
          cell_params.append(coord)
          eL += 4

          while 'ATOMIC_POSITIONS' not in lines[eL]:
            eL += 1
          eL,apos = read_apos(eL+1)
          abc.append(apos)

    except Exception as e:
      print('WARNING: No atomic positions or cell coordinates were found.', flush=True)
      raise e

  struct['lattice'] = np.array([struct['lattice']] + cell_params)
  struct['abc'] = np.array([struct['abc']] + abc)

  return struct


def struct_from_inputfile_QE ( fname:str ) -> dict:
  '''
    Generate a dictionary containing all atomic information from a QE inputfile
    WARNING: Currently only the control blocks are read. Atomic cards are not...

    Arguments:
      fname (str): Name (including path) of the inputfile

    Returns:
      (dict): Structure dictionary
  '''
  from os.path import isfile
  import numpy as np
  import re

  if not isfile(fname):
    raise FileNotFoundError('File {} does not exist.'.format(fname))

  fstr = None
  with open(fname, 'r') as f:
    fstr = f.read()

  # Datatype format helpers for QE input
  nocomma = lambda s : s.replace(',', '')
  qebool = lambda s : True if s.split('.')[1][0].lower() == 't' else False
  qenum = lambda s : s.split('=')[1].replace('d', 'e')
  qeint = lambda s : int(qenum(s))
  qefloat = lambda s : float(qenum(s))
  def inquote ( s ):
    v = '"' if '"' in s else "'"
    return s.split(v)[1]

  # Process blocks
  cards = {}
  blocks = {}
  natom = ntype = 0
  pattern = re.compile('&(.*?)/@')
  celldm = np.zeros(6, dtype=float)
  comment = lambda v : v != '' and v[0] != '!'
  matches = pattern.findall(fstr.replace(' ', '').replace('\n', '@ '))
  for m in matches:
    m = [s.replace(' ', '').split('!')[0] for s in re.split(', |@', m)]
    mf = []
    for v in m:
      mf += v.split(',')
    block = mf.pop(0).lower()
    mf = filter(comment, mf)
    blocks[block] = {}
    for s in mf:
      k,v = s.split('=')
      blocks[block][k] = v
      if k == 'ntyp':
        ntype = int(v)
      elif k == 'nat':
        natom = int(v)

  # Process CARDS
  fstr = list(filter(comment, fstr.split('\n')))
  def scan_blank_lines ( nl ):
    nl += 1
    while fstr[nl] == '':
      nl += 1
    return nl

  il = 0
  nf = len(fstr)
  while il < nf and 'ATOMIC_POSITIONS' not in fstr[il]:
    il += 1
  if il < nf:
    cards['ATOMIC_POSITIONS'] = [fstr[il]]
    il = scan_blank_lines(il)
    for i in range(natom):
      cards['ATOMIC_POSITIONS'].append(fstr[il+i])

  sl = 0
  while sl < nf and 'ATOMIC_SPECIES' not in fstr[sl]:
    sl += 1
  if sl < nf:
    cards['ATOMIC_SPECIES'] = [fstr[sl]]
    sl = scan_blank_lines(sl)
    for i in range(ntype):
      cards['ATOMIC_SPECIES'].append(fstr[sl+i])

  kl = 0
  while kl < nf and 'K_POINTS' not in fstr[kl]:
    kl += 1
  if kl < nf:
    cards['K_POINTS'] = [fstr[kl]]
    if 'gamma' in fstr[kl].lower():
      pass
    else:
      cards['K_POINTS'].append(fstr[kl+1])
      if 'automatic' not in fstr[kl]:
        nk = int(fstr[kl+1])
        kl += 2
        for i in range(nk):
          cards['K_POINTS'].append(fstr[kl+i])

  cl = 0
  while cl < nf and 'CELL_PARAM' not in fstr[cl]:
    cl += 1
  if cl < nf:
    cards['CELL_PARAMETERS'] = []
    for i in range(4):
      cards['CELL_PARAMETERS'].append(fstr[cl+i])

  return blocks, cards


def create_atomic_inputfile ( calculation, blocks, cards ):

  with open(f'{calculation}.in', 'w') as f:
    f.write('\n')
    for kb,vb in blocks.items():
      f.write(f' &{kb}\n')
      for ks,vs in vb.items():
        f.write(f'  {ks} = {vs}\n')
      f.write(' /\n\n')
    if 'ATOMIC_SPECIES' in cards:
      for s in cards['ATOMIC_SPECIES']:
        f.write(s + '\n')
      f.write('\n')
      del cards['ATOMIC_SPECIES']
    for kc,vc in cards.items():
      for s in vc:
        f.write(s + '\n')
      f.write('\n')
