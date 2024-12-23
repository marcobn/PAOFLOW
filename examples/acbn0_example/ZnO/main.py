# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2018      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
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

prefix = 'ZnO'
acbn0 = ACBN0(prefix,
              workdir='./',
              mpi_qe='/opt/homebrew/bin/mpirun -np 8',
              qe_options='-npool 4',
              qe_path='/Users/marco/Local/Programs/qe-7.0/bin',
              mpi_python='mpirun -np 4',
              python_path='/Users/marco/anaconda3/envs/Work/bin/')

print('\nFinal U values:')
for k,v in acbn0.uVals.items():
  print(f'{k}: {v}')
