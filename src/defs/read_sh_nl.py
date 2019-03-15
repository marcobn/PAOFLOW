
def read_sh_nl ( data_controller, fnscf ):
  '''
  Determines the shell configuration for based on the QE nscf inputfile, 'fnscf', located in either 'workdir' or the .save directory

  Arguments:
     fnscf (string) - Filename of the nscf inputfile, copied to the workdir or .save directory

  Returns:
      sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                       nl is a list of occupations at each site
      sh and nl are representative of the entire system
  '''

  from os.path import join,exists

  arry,attr = data_controller.data_dicts()
  wpath = attr['workpath']

  fpath = None
  p1 = join(wpath,fnscf)
  p2 = join(join(wpath,attr['savedir']),fnscf)
  if exists(p1):
    fpath = p1
  elif exists(p2):
    fpath = p2
  else:
    raise Exception('ERROR: Unable to find nscf inputfile')
  species,atoms = read_nscf(fnscf)

  # Get Shells for each species
  sdict = {}
  for s in species:
    sdict[s[0]] = read_pseudopotential(s[1])

  # Concatenate shells for each atom
  sh = []
  nl = []
  for a in atoms:
    sh += sdict[a][0]
    nl += sdict[a][1]

  return(sh, nl)


def read_nscf ( fnscf ):
  '''
  Reads the nscf file to determine the atmoic species and associated pseudopotential files

  Arguments:
     fnscf (string) - Filename of the nscf inputfile

  Returns:
     [(species,pseudos), atoms]  (list) - 1st element is a tuple with 2 elements (atomic species and corresponding pseudopotential files)
            2nd element is a list of each unique atom in the system, ordered appropriately to the DFT system
  '''

  import re

  atoms = []
  species = []
  pseudos = []

  with open(fnscf, 'r') as f:
    ln = 0
    nat = ntyp = None
    lines = f.readlines()
    rel = lambda s : int(re.findall('\d+',s)[0])

    # Walk to Atomic Species, collecting the number of species and number of atoms
    while 'ATOMIC_SPECIES' not in lines[ln]:
      if ('nat' or 'ntyp') in lines[ln]:
        ls = lines[ln].split(',')
        for s in ls:
          if 'nat' in s:
            nat = rel(s)
          if 'ntyp' in s:
            ntyp = rel(s)
      ln += 1

    # Collect species and pseudopotential filenames
    nf = 0
    while nf < ntyp:
      ls = lines[ln].split()
      if len(ls) == 3:
        species.append(ls[0])
        pseudos.append(ls[2])
        nf += 1
      ln += 1

   # Walk to Atomic Positions
    while 'ATOMIC_POSITIONS' not in lines[ln]:
      ln += 1

    # Collect each atom in the appropriate order
    nf = 0
    while nf < nat:
      ls = lines[ln].split()
      if len(ls) == 4:
        atoms.append(ls[0])
        nf += 1
      ln += 1

  return(zip(species,pseudos), atoms)


def read_pseudopotential ( fpp ):
  '''
  Reads a psuedopotential file to determine the included shells and occupations.

  Arguments:
      fnscf (string) - Filename of the pseudopotential, copied to the .save directory

  Returns:
      sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                       nl is a list of occupations at each site
      sh and nl are representative of one atom only
  '''

  sh = []
  nl = []

  with open(fpp,'r') as f:

    ln = 0
    lines = f.readlines()

    # Walk to Valence Configuration
    while 'Valence configuration:' not in lines[ln]:
      ln += 1

    # Reference Line is the first line of valence data
    ln = ln+2

    get_ql = lambda l : int(lines[l].split()[2])

    prev = None
    while 'Generation' not in lines[ln]:
      ql = get_ql(ln)
      if ql != prev:
        prev = ql
        sh.append(ql)
        nl.append(1)
      ln += 1

  return(sh, nl)
