
######### USER PRESETS #########
exec_prefix_QE = 'mpirun -np 4'
exec_postfix_QE = '-npool 8'
path_QE = '/home/ftc/Software/qe-6.8/bin'

pthr_PAO = 0.95
exec_prefix_PAO = 'mpirun -np 4'
exec_postfix_PAO = ''

path_python = '/home/ftc/Software/anaconda3/bin'


def exec_QE ( executable, fname ):
  from os.path import join

  exe = join(path_QE, executable)
  fout = fname.replace('in', 'out')
  return f'{exec_prefix_QE} {exe} < {fname} > {fout}'


def exec_PAOFLOW ( ):
  from os.path import join

  prefix = join(path_python, 'python')
  return f'{exec_prefix_PAO} python main.py > paoflow.out'


def assign_Us ( struct, species, uVals ):
  struct['lda_plus_u'] = '.true.'
  for i,s in enumerate(species):
    struct['Hubbard_U({})'.format(i+1)] = uVals[s]
  return struct


def run_dft ( prefix, species, uVals ):
  from PAOFLOW.defs.file_io import create_atomic_inputfile
  from PAOFLOW.defs.file_io import struct_from_inputfile_QE
  from os import system

  blocks,cards = struct_from_inputfile_QE(f'{prefix}.scf.in')
  blocks['system'] = assign_Us(blocks['system'], species, uVals)
  create_atomic_inputfile('scf', blocks, cards)

  blocks,cards = struct_from_inputfile_QE(f'{prefix}.nscf.in')
  blocks['system'] = assign_Us(blocks['system'], species, uVals)
  create_atomic_inputfile('nscf', blocks, cards)

  blocks,cards = struct_from_inputfile_QE(f'{prefix}.projwfc.in')
  create_atomic_inputfile('projwfc', blocks, cards)

  executables = {'scf':'pw.x', 'nscf':'pw.x', 'projwfc':'projwfc.x'}
  for c in ['scf', 'nscf', 'projwfc']:
    command = exec_QE(executables[c], f'{c}.in')
    print(f'Starting Process: {command}')
    #ecode = system(command)


def run_paoflow ( prefix, save_prefix, nspin ):
  from PAOFLOW.defs.file_io import create_acbn0_inputfile
  from os import system

  fstr = f'{prefix}_PAO_bands' + '{}.in'
  calcs = []
  if nspin == 1:
    calcs.append(fstr.format('')) 
  else:
    calcs.append(fstr.format('_up'))
    calcs.append(fstr.format('_down'))

  create_acbn0_inputfile(save_prefix, pthr_PAO)
  command = exec_PAOFLOW()
  #ecode = system(command)


def read_cell_atoms ( fname ):
  import numpy as np

  lines = None
  with open(fname, 'r') as f:
    lines = f.readlines()

  il = 0
  while 'lattice parameter' not in lines[il]:
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


def hubbard_orbital ( ele ):
  #d elements
  if ele in {'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co',
             'Ni', 'Cu', 'Zn', 'Zr', 'Nb', 'Mo',
             'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
             'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir',
             'Pt', 'Au', 'Hg','Sc', 'Ga', 'In', 'Y'}:
    return 2

  #p elements
  elif ele in {'C', 'N', 'O', 'Se', 'S', 'Te','Sn',
               'B','F','Al','Si','P','Cl', 'Ge','As',
               'Br','Sb','I','Tl','Pb','Bi','Po','At'}:
    return 1

  #s elements
  elif ele in {'H', 'Sr','Mg', 'Ba','Li','Be','Na','K','Ca','Rb','Cs'}:
    return 0

  else:
    raise Exception(f'Element {ele} has no defined Hubbard orbital')


def run_acbn0 ( prefix, nspin ):

  lattice,species,positions = read_cell_atoms('scf.out')
  lattice_string = ','.join([str(v) for a in lattice for v in a])
  position_string = ','.join([str(v) for a in positions for v in a])
  species_string = ','.join(species)

  for s in species:
    print(hubbard_orbital(s))


if __name__ == '__main__':
  from PAOFLOW.defs.file_io import struct_from_inputfile_QE
  from sys import argv

  argc = len(argv)
  if argc < 2:
    print('Usage:\n  python scfuj.py <prefix>')
    print('\nFiles prefix.scf.in, prefix.nscf.in, and prefix.proj.in must exist in current directory.')
    quit()

  # Get structure information
  prefix = argv[1]
  blocks,cards = struct_from_inputfile_QE(f'{prefix}.scf.in')
  nspin = int(struct['nspin']) if 'nspin' in blocks['system'] else 1
  uspecies = []
  for s in cards['ATOMIC_SPECIES'][1:]:
    uspecies.append(s.split()[0])

  # Set initial UJ
  uVals = {}
  threshold_U = 0.01
  for i,s in enumerate(uspecies):
    uVals[s] = threshold_U
	  
  run_dft(prefix, uspecies, uVals)

  save_prefix = blocks['control']['prefix'].strip('"').strip('"')
  run_paoflow(prefix, save_prefix, nspin)

  run_acbn0(prefix, nspin)
  # While not converged
  ### Determine LSD
  ###   single scf step?
  ### Run SCF, NSCF, PROJ, PAOFLOW
  ### ACBN0
  ##### generate inputfile
  ##### Run ACBN0
  ### Get new UJ
  ### Check for convergence
