# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2018      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2018 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
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

from PAOFLOW import PAOFLOW

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='silicon.save', outputdir='output', smearing='gauss', npool=1, verbose=True)

  # The orbitals are saved to the data arrays under the key 'atorb'
  #   (rank 0 has the array, other ranks have None)

  # Real space grid dimensions and number of cells in the supercells per direction
  #   are saved with keys 'nr1', 'nr2', 'nr3', and 'nsc'

  # To supress the atomic orbital output in xsf format set fprefix=None

  paoflow.atomic_orbitals(nr1=20, nr2=20, nr3=20, fprefix='Si_orbitals')

  paoflow.finish_execution()

if __name__== '__main__':
  main()

