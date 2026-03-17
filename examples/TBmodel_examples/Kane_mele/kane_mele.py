# *************************************************************************************
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2024      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
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
from PAOFLOW import GPAO
import matplotlib.pyplot as plt

def main():


  pplt = GPAO.GPAO()
  model = {'label':'Kane_Mele', 't':1.0, 'r_par' : 0.0 ,'v_par' : 0.0 , 'soc_par':0.1, 'alat':1.0}
  paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./kane_mele', verbose=True)

  path = 'G-M-K-G'
  special_points = {'G':[0.0, 0.0, 0.0],'K':[2.0/3.0, 1.0/3.0, 0.0],'M':[1.0/2.0, 0.0/2.0, 0.0]}
  paoflow.bands(ibrav=4, nk=100, band_path=path, high_sym_points=special_points)

  # Bandstructure plot
  outputdir='./kane_mele/' 
  f_band = outputdir + 'bands_0.dat'
  f_symp = outputdir + 'kpath_points.txt'

  pplt.plot_bands( f_band, f_symp, None, None, y_lim = (-4,4))


  paoflow.interpolated_hamiltonian(nfft1=60,nfft2=60,nfft3=1)
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.spin_Hall(twoD=True,emin=-4., emax=4., s_tensor=[[0,1,2]])


  # Spin Hall conductivity plot
  shc    = np.loadtxt('./kane_mele/shcEf_z_xy.dat')

  fig = plt.figure(figsize=(8,5))
  plt.plot(shc[:,0],shc[:,1]*2*np.pi    ,color='green',linewidth=3)
  plt.xlim(-4,4)
  plt.xlabel(r" E- E$_f$ (eV)")
  plt.ylabel(r'$\sigma^{z}_{xy}\;(e/2\pi)$')
  plt.title(r'SHC Kane-Mele Models')
  plt.savefig('kane_mele/shc_z_xy.png',bbox_inches='tight')
 

  paoflow.finish_execution()

if __name__== '__main__':
  main()

