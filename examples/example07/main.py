# *************************************************************************************
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
import numpy as np

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='./pt.save')
  arry,attr = paoflow.data_controller.data_dicts()

  paoflow.projectability()
  paoflow.pao_hamiltonian()

  paoflow.adhoc_spin_orbit(phi=0.0,theta=0.0,
                          naw       = np.array([9]),                   # number of orbitals for each atom
                          lambda_p  = np.array([0.0]),                 # p orbitals SOC strengh for each atom
                          lambda_d  = np.array([0.5534]),              # d orbitals SOC strengh for each atom
                          orb_pseudo = ['spd'])                        # type of pseudo potential for each atom

  path = 'gG-X-W-K-gG-L-U-W-L-K|U-X'
  special_points = {'gG'   : (0.0, 0.0, 0.0),
              'K'  : (0.375, 0.375, 0.750),
              'L'  : (0.5, 0.5, 0.5),
              'U'  : (0.625, 0.250, 0.625),
              'W'  : (0.5, 0.25, 0.75),
              'X'  : (0.5, 0.0, 0.5)}

  paoflow.bands(ibrav=2, nk=1000, band_path=path, high_sym_points=special_points)


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

