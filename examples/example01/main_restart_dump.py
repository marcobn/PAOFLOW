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

  label = 'sym'
  outdir = 'output_%s'%label
  savedir = 'silicon_%s.save'%label

  # Initialize PAOFLOW, indicating the name of the QE save directory.
  paoflow = PAOFLOW.PAOFLOW(savedir=savedir, outputdir=outdir, smearing='gauss', npool=1, verbose=True)
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  # Calculate eigenvalues on the default ibrav=2 path
  paoflow.bands(ibrav=2, nk=2000)

  # Dimension of the grid is doubled by default
  #  e.g. 12x12x12 -> 24x24x24
  paoflow.interpolated_hamiltonian()

  # Calculate eigenvalues on the entire BZ grid
  paoflow.pao_eigh()

  paoflow.gradient_and_momenta()

  # Dump PAOFLOW data for future runs.
  paoflow.restart_dump(fname_prefix='paoflow_dump')

  paoflow.finish_execution()

if __name__== '__main__':
  main()

