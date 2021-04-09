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

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='SnTe.save', npool=4, verbose=1)

  paoflow.projectability(pthr=0.90)
  paoflow.pao_hamiltonian()

  # Define high symmetry points and the path comprised of such points
  # '-' defines a continuous transition to the next high symmetry point
  # '|' defines a discontinuous break in from one high symmetry point to the next

  path = 'G-X-S-Y-G'
  special_points = {'G':[0.0, 0.0, 0.0],'S':[0.5, 0.5, 0.0],'X':[0.5, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  paoflow.bands(ibrav=8, nk=1000, band_path=path, high_sym_points=special_points)

  paoflow.interpolated_hamiltonian(nfft1=140, nfft2=140, nfft3=1)  
  paoflow.pao_eigh()
  paoflow.spin_operator()
  paoflow.fermi_surface(fermi_up=0.0, fermi_dw=-1. )
  paoflow.spin_texture(fermi_up=0.0, fermi_dw=-1. )
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.spin_Hall(twoD=True, emin=-3.5, emax=1.5, s_tensor=[[0,1,2],[1,0,2]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()
