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
from sys import argv

## Usage:
##    python main.py
##    python main.py <work_directory>
##    python main.py <work_directory> <inputfile_name>
##
## Defaults:
##    <work_directory> - './'
##    <inputfile_name> - 'inputfile.xml'

def main():

  argc = len(argv)
  arg1 = ('./' if argc<2 else argv[1])
  arg2 = ('inputfile.xml' if argc<3 else argv[2])

  # Start PAOFLOW with an inputfile in the current directory
  #
  # PAOFLOW will us data attributes read from 
  #   inputfile.xml for the following calculations 
  paoflow = PAOFLOW.PAOFLOW(workpath=arg1, inputfile=arg2, verbose=True,outputdir="")

  # Get dictionary containers with the
  #   attributes and arrays read from inputfiles
  arry,attr = paoflow.data_controller.data_dicts()

  paoflow.projectability()

  paoflow.pao_hamiltonian(non_ortho=attr['non_ortho'],expand_wedge=attr['expand_wedge'],thresh=attr['symm_thresh'],
                          symmetrize=attr['symmetrize'],max_iter=attr['symm_max_iter'])

  if attr['write2file']:
    paoflow.write2file()

  paoflow.add_external_fields()

  if attr['writez2pack']:
    paoflow.z2_pack(fname='z2pack_hamiltonian.dat')

  if attr['do_bands'] or attr['band_topology']:

    paoflow.bands(ibrav=int(attr["ibrav"]))

  if attr['spintexture'] or attr['spin_Hall']:
    paoflow.spin_operator(spin_orbit=attr['do_spin_orbit'])

  if attr['band_topology'] and not attr['onedim']:
    paoflow.topology()
  elif attr['onedim']:
    print('1D Band topology not supported with the PAOFLOW class')

  if attr['double_grid']:
    paoflow.interpolated_hamiltonian(nfft1=attr['nfft1'], nfft2=attr['nfft2'], nfft3=attr['nfft3'])

  paoflow.pao_eigh()

  if attr['fermisurf']:
    paoflow.fermi_surface()

  if attr['spintexture']:
    paoflow.spin_texture()

  paoflow.gradient_and_momenta()

  if attr['smearing'] is not None:
    paoflow.adaptive_smearing()

  if attr['do_dos'] or attr['do_pdos']:
    paoflow.dos(do_dos=attr['do_dos'], do_pdos=attr['do_pdos'], emin=attr['emin'], emax=attr['emax'])

  if attr['spin_Hall']:
    paoflow.spin_Hall(do_ac=attr['ac_cond_spin'], emin=attr['eminSH'], emax=attr['emaxSH'],s_tensor=arry['s_tensor'])

  if attr['Berry']:
    paoflow.anomalous_Hall(do_ac=attr['ac_cond_Berry'], emin=attr['eminAH'], emax=attr['emaxSH'],a_tensor=arry['a_tensor'])

  if attr['Boltzmann']:
    paoflow.transport(tmin=attr['tmin'], tmax=attr['tmax'], tstep=attr['tstep'], emin=attr['emin'], emax=attr['emax'], ne=attr['ne'],
                      t_tensor=arry['t_tensor'])

  if attr['epsilon']:
    paoflow.dielectric_tensor(metal=attr['metal'], emin=attr['epsmin'], emax=attr['epsmax'], ne=attr['ne'],d_tensor=arry['d_tensor'])

  # Print the total execution time and request
  #   desired quantites for further processing
  paoflow.finish_execution()


if __name__== '__main__':
  main()
