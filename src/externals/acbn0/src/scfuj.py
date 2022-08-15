
############ USER PRESETS ############
exec_prefix_QE = 'mpirun -np 8'
exec_postfix_QE = '-npool 4 -northo 1'
path_QE = '/home/ftc/Software/qe-6.8/bin'

pthr_PAO = 0.95
exec_prefix_PAO = 'mpirun -np 4'
exec_postfix_PAO = ''

path_python = '/home/ftc/Software/anaconda3/bin'
########## END USER PRESETS ##########

def exec_command ( command ):
  from os import system

  print(f'Starting Process: {command}')
  return system(command)


def exec_QE ( executable, fname ):
  from os.path import join

  exe = join(path_QE, executable)
  fout = fname.replace('in', 'out')
  command = f'{exec_prefix_QE} {exe} {exec_postfix_QE} < {fname} > {fout}'
  return exec_command(command)


def exec_PAOFLOW ( ):
  from os.path import join

  prefix = join(path_python, 'python')
  command = f'{exec_prefix_PAO} {prefix} {exec_postfix_PAO} main.py > paoflow.out'
  return exec_command(command)


def exec_ACBN0 ( fname ):
  from os.path import join

  prefix = join(path_python, 'python')
  command = f'{prefix} acbn0.py {fname} > /dev/null'
  return exec_command(command)


def assign_Us ( struct, species, uVals ):
  struct['lda_plus_u'] = '.true.'
  for i,s in enumerate(species):
    struct['Hubbard_U({})'.format(i+1)] = uVals[s]
  return struct


def run_dft ( prefix, species, uVals ):
  from PAOFLOW.defs.file_io import create_atomic_inputfile
  from PAOFLOW.defs.file_io import struct_from_inputfile_QE

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
    ecode = exec_QE(executables[c], f'{c}.in')


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
  ecode = exec_PAOFLOW()


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
  import re

  lattice,species,positions = read_cell_atoms('scf.out')
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
  for s in list(set(species)):
    ostates = []
    ustates = []
    horb = hubbard_orbital(s)
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

    fname = f'{prefix}_acbn0_infile_{s}.txt'
    with open(fname, 'w') as f:
      f.write(f'symbol = {s}\n')
      f.write(f'latvects = {lattice_string}\n')
      f.write(f'coords = {position_string}\n')
      f.write(f'atlabels = {species_string}\n')
      f.write(f'nspin = {nspin}\n')
      f.write(f'fpath = ./\n')
      f.write(f'outfile = {prefix}_acbn0_outfile_{s}.txt\n')
      f.write(f'reduced_basis_dm = {basis_dm}\n')
      f.write(f'reduced_basis_2e = {basis_2e}\n')

    exec_ACBN0(fname)
    with open(f'{s}_UJ.txt', 'r') as f:
      lines = f.readlines()
      uVals[s] = float(lines[2].split(':')[1])

  return uVals


if __name__ == '__main__':
  from PAOFLOW.defs.file_io import struct_from_inputfile_QE
  from upf_gaussfit import gaussian_fit
  from os.path import join
  from os import getcwd
  from sys import argv
  import numpy as np

  argc = len(argv)
  if argc < 2:
    print('Usage:\n  python scfuj.py <prefix>')
    print('\nFiles prefix.scf.in, prefix.nscf.in, prefix.proj.in, and all relevant pseudoptnetials must exist in current directory.')
    quit()

  # Get structure information
  cwd = getcwd()
  prefix = argv[1]
  blocks,cards = struct_from_inputfile_QE(f'{prefix}.scf.in')
  nspin = int(struct['nspin']) if 'nspin' in blocks['system'] else 1

  # Generate gaussian fits
  uspecies = []
  for s in cards['ATOMIC_SPECIES'][1:]:
    ele,_,pp = s.split()
    uspecies.append(ele)
    gaussian_fit(join(cwd,pp))

  # Set initial UJ
  uVals = {}
  threshold_U = 0.01
  blocks['lda_plus_u'] = '.true.'
  for i,s in enumerate(uspecies):
    uVals[s] = threshold_U
  
  # Perform self consistent calculation of Hubbard parameters
  converged = False
  while not converged:

    # Update U values provided to inputfiles
    for i,s in enumerate(uspecies):
      blocks['Hubbard_U({})'.format(i+1)] = str(uVals[s])

    run_dft(prefix, uspecies, uVals)

    save_prefix = blocks['control']['prefix'].strip('"').strip('"')
    run_paoflow(prefix, save_prefix, nspin)

    new_U = run_acbn0(prefix, nspin)
    converged = True
    print('New U values:')
    for k,v in new_U.items():
      print(f'  {k} : {v}')
      if converged and np.abs(uVals[k]-v) > threshold_U:
        converged = False

    uVals = new_U
