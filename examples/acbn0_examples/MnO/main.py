

from PAOFLOW.ACBN0 import ACBN0

prefix = 'MnO'
acbn0 = ACBN0(prefix,
              workdir='./',
              mpi_qe='srun',
              mpi_python='srun',
              qe_options='-npool 4',
              qe_path='',
              python_path='/usr/bin/',
              outputdir='./tmp')


# Here, the Hubbard modifications are presented in three equivalent ways

#  1) Simply specify the species and state on which to apply Hubbard corrections.
#      U values default to 0.01 eV
hubbard = ['MnA-3d', 'MnB-3d', 'O-2p']

#  2) Specify the species and state, with custom initial U values
# hubbard = { 'MnA-3d' : 5.0,
#             'MnB-3d' : 5.0,
#             'O-2p'  : 2.0 }

#  3) Specify custom hubbard occupation for Oxygen (initial_U, occupation)
# hubbard = { 'MnA-3d' : 5.0,
#             'MnB-3d' : 5.0,
#             'O-2p'  : (2.0, 4.0)}

acbn0.set_hubbard_parameters(hubbard)

acbn0.optimize_hubbard_U(convergence_threshold=0.01)

print('\nFinal U values:')
for k,v in acbn0.uVals.items():
  print(f'  {k}: {v}')
