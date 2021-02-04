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

  # Fake call to initiallize PAOFLOW
  paoflow = PAOFLOW.PAOFLOW(workpath='./',outputdir='./',savedir='./co.save',verbose=True)
  arry,attr = paoflow.data_controller.data_dicts()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  # TB model for a Graphene nearest naeighbors only with pz orbitals
  paoflow.TBmodels(model="Graphene",t=2.7)

  path = 'G-M-K-G'
  special_points = {'G':[0.0, 0.0, 0.0],'K':[2.0/3.0, 1.0/3.0, 0.0],'M':[1.0/2.0, 0.0/2.0, 0.0]}
  paoflow.bands(ibrav=4, nk=100, band_path=path, high_sym_points=special_points)

  paoflow.finish_execution()

if __name__== '__main__':
  main()

