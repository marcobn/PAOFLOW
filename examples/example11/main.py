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

from PAOFLOW import PAOFLOW
import numpy as np

def main():
  
  # Start PAOFLOW, interpolate Hamiltonian, spin operator, spin texture, compute gradient and momenta
  paoflow = PAOFLOW.PAOFLOW(savedir='./Te-L.save')
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.interpolated_hamiltonian(nfft1=60, nfft2=60, nfft3=40)
  paoflow.pao_eigh()
  paoflow.spin_operator()
  paoflow.spin_texture(fermi_up=0.0, fermi_dw=-0.5)
  paoflow.gradient_and_momenta()
  
  # Compute adaptive smearing and calculate the Rashba-Edelstein tensor elements
  paoflow.adaptive_smearing() 
  paoflow.rashba_edelstein(emin=-0.5, emax=0.0, ne=501)

  paoflow.finish_execution()

if __name__== '__main__':
  main()


