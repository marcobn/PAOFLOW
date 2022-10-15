from PAOFLOW.ACBN0 import ACBN0

prefix = 'MgO'
acbn0 = ACBN0(prefix,
              workdir='./',
              mpi_qe='/usr/bin/mpirun -np 8',
              qe_options='-npool 4',
              qe_path='/home/ftc/Software/qe-6.8/bin',
              mpi_python='mpirun -np 4',
              python_path='/home/ftc/Software/anaconda3/bin')

print('\nFinal U values:')
for k,v in acbn0.uVals.items():
  print(f'{k}: {v}')
