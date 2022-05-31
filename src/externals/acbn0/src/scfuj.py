
######### USER PRESETS #########
exec_prefix_QE = 'mpirun -np 4'
exec_postfix_QE = '-npool 8'
path_QE = '/home/ftc/Programs/qe-6.8/bin'

pthr_PAO = 0.95
exec_prefix_PAO = 'mpirun -np 4'
exec_postfix_PAO = ''

path_python = '/home/ftc/Programs/anaconda3/bin'


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
  ecode = system(command)


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
  print(blocks)
  print(cards)
  uspecies = []
  for s in cards['ATOMIC_SPECIES'][1:]:
    uspecies.append(s.split()[0])
  print(uspecies)

  # Set initial UJ
  uVals = {}
  threshold_U = 0.01
  for i,s in enumerate(uspecies):
    uVals[s] = threshold_U
  
  run_dft(prefix, uspecies, uVals)

  save_prefix = blocks['control']['prefix'].strip('"').strip('"')
  run_paoflow(prefix, save_prefix, nspin)
  # While not converged
  ### Determine LSD
  ###   single scf step?
  ### Run SCF, NSCF, PROJ, PAOFLOW
  ### ACBN0
  ##### generate inputfile
  ##### Run ACBN0
  ### Get new UJ
  ### Check for convergence
