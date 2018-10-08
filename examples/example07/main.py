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

  paoflow = PAOFLOW(savedir='al.save', npool=8)
  paoflow.projectability(pthr=.97)
  paoflow.pao_hamiltonian(non_ortho=True)
  paoflow.add_external_fields()
  paoflow.bands(ibrav=2)
  paoflow.interpolated_hamiltonian()
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(do_pdos=False, emin=-12., emax=3.)
  paoflow.transport(emin=-2., emax=2., t_tensor=[[0,0]])
  paoflow.dielectric_tensor(metal=True, emin=.05, emax=6., d_tensor=[[0,0]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()

