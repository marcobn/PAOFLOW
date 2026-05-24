

from PAOFLOW.ACBN0 import ACBN0

prefix = 'ZnO'
acbn0 = ACBN0(prefix,
              workdir='./',
              mpi_qe='/opt/homebrew/bin/mpirun -np 8',
              qe_options='-npool 4',
              qe_path='/Users/marco/Local/Programs/qe-7.0/bin',
              mpi_python='mpirun -np 4',
              python_path='/Users/marco/anaconda3/envs/Work/bin/')

# Here, the Hubbard modifications are presented in three equivalent ways

#  1) Simply specify the species and state on which to apply Hubbard corrections.
#      U values default to 0.01 eV
hubbard = ['Zn-3d', 'O-2p']

#  2) Specify the species and state, with custom initial U values
# hubbard = { 'Zn-3d' : 6.0,
#             'O-2p'  : 1.0 }

#  3) Specify custom hubbard occupation for Oxygen (initial_U, occupation)
#hubbard = { 'Zn-3d' : 6.0,
#            'O-2p'  : (1.0, 4.0)}

acbn0.set_hubbard_parameters(hubbard)

acbn0.optimize_hubbard_U(convergence_threshold=0.01)

print('\nFinal U values:')
for k,v in acbn0.uVals.items():
  print(f'{k}: {v}')
