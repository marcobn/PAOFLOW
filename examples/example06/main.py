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

  paoflow = PAOFLOW(savedir='alp.save')

  paoflow.calc_projectability()
  paoflow.calc_pao_hamiltonian()

  correction_Hubbard = np.zeros(32, dtype=float)
  correction_Hubbard[1:4] = .1
  correction_Hubbard[17:20] = 2.31
  paoflow.add_external_fields(HubbardU=correction_Hubbard)

  paoflow.calc_bands(ibrav=2)
  paoflow.calc_interpolated_hamiltonian(nfft1=24, nfft2=24, nfft3=24)
  paoflow.calc_pao_eigh()
  paoflow.calc_gradient_and_momenta()
  paoflow.calc_adaptive_smearing()
  paoflow.calc_dos_adaptive(do_pdos=False, emin=-8., emax=5., delta=.1)
  paoflow.calc_transport(emin=-8., emax=5., t_tensor=[[0,0]])
  paoflow.calc_dielectric_tensor(emax=6., d_tensor=[[0,0]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()

