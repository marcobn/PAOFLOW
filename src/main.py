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

  arg1 = './'
  arg2 = 'inputfile.xml'

  paoflow = PAOFLOW(workpath=arg1, inputfile=arg2, verbose=False)

  arry,attr = paoflow.data_controller.data_dicts()

  paoflow.calc_projectability()

  paoflow.calc_pao_hamiltonian()

  if attr['non_ortho']:
    paoflow.orthogonalize_hamiltonian()

  paoflow.add_external_fields()

  if attr['do_bands']:
    paoflow.calc_bands()

  if attr['band_topology'] and not attr['onedim']:
    paoflow.calc_topology()

  if attr['double_grid']:
    paoflow.calc_interpolated_hamiltonian()

  paoflow.calc_pao_eigh()

  if attr['smearing'] is None:
    paoflow.calc_dos(do_dos=attr['do_dos'], do_pdos=attr['do_pdos'], emin=attr['emin'], emax=attr['emax'])

  if attr['fermisurf']:
    paoflow.calc_fermi_surface()

  if attr['spintexture'] or attr['spin_Hall']:
    paoflow.calc_spin_operator(spin_orbit=attr['do_spin_orbit'])

  if attr['spintexture']:
    paoflow.calc_spin_texture()

  paoflow.calc_gradient_and_momenta()

  if attr['smearing'] is not None:
    paoflow.calc_adaptive_smearing()

    paoflow.calc_dos_adaptive(do_dos=attr['do_dos'], do_pdos=attr['do_pdos'], emin=attr['emin'], emax=attr['emax'])

  if attr['spin_Hall']:
    paoflow.calc_spin_Hall(do_ac=attr['ac_cond_spin'], emin=attr['eminSH'], emax=attr['emaxSH'])

  if attr['Berry']:
    paoflow.calc_anomalous_Hall(do_ac=attr['ac_cond_Berry'], emin=attr['eminAH'], emax=attr['emaxSH'])

  if attr['Boltzmann']:
    paoflow.calc_transport(tmin=attr['tmin'], tmax=attr['tmax'], tstep=attr['tstep'], emin=attr['emin'], emax=attr['emax'], ne=attr['ne'])

  if attr['epsilon']:
    paoflow.calc_dielectric_tensor(metal=attr['metal'], kramerskronig=attr['kramerskronig'], emin=attr['epsmin'], emax=attr['epsmax'], ne=attr['ne'])

  paoflow.finish_execution()

if __name__== '__main__':
  main()

