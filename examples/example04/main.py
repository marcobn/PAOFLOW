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

import numpy as np
from PAOFLOW_class import PAOFLOW

def main():
  paoflow = PAOFLOW(savedir='./fe.save')
  paoflow.calc_projectability(pthr=0.95)
  paoflow.calc_pao_hamiltonian()
  paoflow.calc_bands(ibrav=3)
  paoflow.calc_topology(spol=2, ipol=1, jpol=2, eff_mass=True, Berry=True)
  paoflow.calc_interpolated_hamiltonian()
  paoflow.calc_pao_eigh()
  paoflow.calc_gradient_and_momenta()
  paoflow.calc_adaptive_smearing(smearing='gauss')
  paoflow.calc_dos_adaptive(do_pdos=False)
  paoflow.calc_anomalous_Hall(a_tensor=np.array([[0,1]]))
  paoflow.finish_execution()

if __name__== '__main__':
  main()
