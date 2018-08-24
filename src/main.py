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

# future imports


import sys
import time
import numpy as np
from PAOFLOW_class import PAOFLOW
from mpi4py import MPI

def main():

  rank = MPI.COMM_WORLD.Get_rank()

  arg1 = './'
  arg2 = 'inputfile.xml'
  try:
    arg1 = os.path.abspath(sys.argv[1])
    if os.path.isfile(arg1):
      arg2 = os.path.basename(arg1)
      arg1 = os.path.dirname(arg1)
  except:
    pass

  paoflow = PAOFLOW(inputpath=arg1,inputfile=arg2,verbose=False)

  arrays, attributes = paoflow.data_controller.data_dicts()


  paoflow.calc_projectability(pthr=None)

  paoflow.calc_pao_hamiltonian()

  if attributes['non_ortho']:
    paoflow.orthogonalize_hamiltonian()

  paoflow.add_external_fields()

  if attributes['do_bands']:
    paoflow.calc_bands()

  ## MUST KNOW DOUBLE_GRID IN ADVANCE
  if attributes['double_grid']:
    paoflow.calc_double_grid()

  paoflow.calc_pao_eigh()

  paoflow.calc_dos(do_dos=attributes['do_dos'], do_pdos=attributes['do_pdos'], emin=attributes['emin'], emax=attributes['emax'])

  if attributes['fermisurf']:
    paoflow.calc_fermi_surface()

  if attributes['spintexture']:
    paoflow.calc_spin_texture()

  paoflow.calc_gradient_and_momenta()

  paoflow.calc_adaptive_smearing(smearing=attributes['smearing'])

  paoflow.calc_dos_adaptive(do_dos=attributes['do_dos'], do_pdos=attributes['do_pdos'], emin=attributes['emin'], emax=attributes['emax'])

  paoflow.calc_spin_Hall(do_shc=attributes['spin_Hall'], do_ac=attributes['ac_cond_spin'])

  quit()

  # PAOFLOW may be called with one argument specifying the directory containing 'inputfile.xml'.
  outDict = paoflow(inputpath=arg1,inputfile=arg2)

  # Check for output dictionary and print the keys.
  if outDict is not None:
    print('\noutDict keys:')
    for k in list(outDict.keys()):
      print('\''+k+'\'')

if __name__== "__main__":
  main()
