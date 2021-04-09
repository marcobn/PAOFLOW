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
from PAOFLOW import PAOFLOW
from PAOFLOW.defs.TauModel import TauModel

def main():

  # Start PAOFLOW, interpolate Hamiltonian, compute gradient an momenta
  paoflow = PAOFLOW.PAOFLOW(savedir='GaAs.save', smearing=None, npool=1, verbose=True)
  arrays,attr = paoflow.data_controller.data_dicts()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.interpolated_hamiltonian(nfft1=40, nfft2=40, nfft3=40)
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()

  # Compute the chemical potential at specified doping concentration for various temperatures
  doping = -3.5e17
  paoflow.doping(tmin=380, tmax=812, nt=28, emin=-36, emax=2, ne=5000, doping_conc=doping)

  # Define the functional form for our 'custom' TauModel
  me = 9.10938e-31 # Electron Mass
  ev2j = 1.60217662e-19 # Electron Charge
  def acoustic_model ( temp, eigs, params ):
    # Formula from Fiorentini paper on Mg3Sb2, DOI: 10.1088/1361-648X/aaf364
    from scipy.constants import hbar
    temp *= ev2j
    E = eigs * ev2j # Eigenvalues in J
    v = 5.2e3 # Velocity in m/s
    rho = 5.31e3 # Mass density kg/m^3
    ms = .7 * me #effective mass tensor in kg 
    D_ac = 7 * ev2j # Acoustic deformation potential in J
    return (2*ms)**1.5*(D_ac**2)*np.sqrt(E)*temp/(2*np.pi*rho*(hbar**2*v)**2) 

  # Create the TauModel object
  acoustic_tau = TauModel(function=acoustic_model)

  # Load the temperatures and corresponding chemical potentials
  fname = 'doping_n%s.dat'%np.abs(doping)
  temp = np.loadtxt('output/%s'%fname, usecols=(0,))
  mu = np.loadtxt('output/%s'%fname, usecols=(1,))

  # Define the desired scattering channels, 1 user-defined and 3 built-in.
  channels = [acoustic_tau, 'polar_optical', 'impurity', 'polar_acoustic']

  # Define quantities required for each built-in TauModel
  tau_params = {'doping_conc':-3.5e17, 'D_ac':7., 'rho':5.31e3,
                  'a':5.653e-10, 'nI':3.5e17, 'eps_inf':11.6, 'eps_0':13.5,
                  'v':5.2e3, 'Zi':1, 'hwlo':[0.03536], 'D_op':3e10, 'Zf':6,
                  'piezo':0.16, 'ms':0.7}

  # Compute the transport properties for each temperature and chemical potential
  rho = []
  for t,m in zip(temp,mu):
    if paoflow.rank == 0:
      print('\nTemp, Mu: %f, %f'%(t,m))

    # Update the Fermi energy in the parameters for scattering models
    tau_params['Ef'] = abs(m)

    paoflow.transport(tmin=t, tmax=t, nt=1, emin=m, emax=m, ne=1, scattering_channels=channels,
                      tau_dict=tau_params, save_tensors=True, write_to_file=False)

    # Average the diagonal componenets of sigma
    sigma = np.sum([sig for sig in np.diag(arrays['sigma'][:,:,0])])/3
    rho.append(1e2/sigma)

  # Write the sigmas
  if paoflow.rank == 0:
    with open('output/rho_rta_n3.5e17.dat' ,'w') as rho_file:
      for i,t in enumerate(temp):
        rho_file.write('%8.2f %9.5e\n'%(t,rho[i]))
 
  paoflow.finish_execution()

if __name__== '__main__':
  main()

