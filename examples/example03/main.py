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

from PAOFLOW_class import PAOFLOW

def main():

  paoflow = PAOFLOW(savedir='pt.save', smearing='m-p')
  paoflow.calc_projectability()
  paoflow.calc_pao_hamiltonian()
  paoflow.calc_interpolated_hamiltonian()
  paoflow.calc_pao_eigh()
  paoflow.calc_gradient_and_momenta()
  paoflow.calc_adaptive_smearing()
  paoflow.calc_dos_adaptive(emin=-8., emax=4., delta=.2)
  paoflow.calc_transport(t_tensor=[[0,0]])
  paoflow.calc_dielectric_tensor(metal=True, kramerskronig=False, emin=.5, emax=10., d_tensor=[[0,0]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()

