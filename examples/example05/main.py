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

  paoflow = PAOFLOW.PAOFLOW(savedir='pt.save')
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.bands(ibrav=2, nk=2000)
  paoflow.spin_operator(sh=[0,1,2], nl=[1,1,1])
  paoflow.topology(Berry=True, eff_mass=True, spin_Hall=True, spol=2, ipol=0, jpol=1)
  paoflow.interpolated_hamiltonian()
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(do_pdos=False, emin=-8., emax=4.)
  paoflow.spin_Hall(emin=-8., emax=4., s_tensor=[[0,1,2]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()

