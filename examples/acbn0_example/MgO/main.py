# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2024      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
#
#  This file is part of AFLOW software.
#
#  AFLOW is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************************

from PAOFLOW.ACBN0 import ACBN0

prefix = 'MgO'
acbn0 = ACBN0(prefix,
              workdir='./',
              mpi_qe='/usr/bin/mpirun -np 8',
              mpi_python='/home/ftc/Software/anaconda3/bin/mpirun -np 4',
              qe_options='-npool 4',
              qe_path='/home/ftc/Software/qe-7.3/bin',
              python_path='/home/ftc/Software/anaconda3/bin')


# Here, the Hubbard modifications are presented in three equivalent ways

#  1) Simply specify the species and state on which to apply Hubbard corrections.
#      U values default to 0.01 eV
hubbard = ['Mg-3s', 'O-2p']

#  2) Specify the species and state, with custom initial U values
#hubbard = { 'Mg-3s' : 1.0,
#            'O-2p'  : 8.0 }

#  3) Specify custom hubbard occupation for Oxygen (initial_U, occupation)
#hubbard = { 'Mg-3s' : 1.0,
#            'O-2p'  : (8.0, 4.0)}

acbn0.set_hubbard_parameters(hubbard)

acbn0.optimize_hubbard_U(convergence_threshold=0.01)

print('\nFinal U values:')
for k,v in acbn0.uVals.items():
  print(f'  {k}: {v}')

