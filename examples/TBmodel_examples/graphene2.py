# *************************************************************************************
# *                                                                                   *
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
import matplotlib.pyplot as plt

def main():

  model = {'label':'graphene2', 't': -1.0, 'delta': -0.2}
  paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./graphene2', verbose=True)

  arry, attr = paoflow.data_controller.data_dicts()

  path = 'G-M-K-G'
  special_points = {'G':[0.0, 0.0, 0.0],'K':[2.0/3.0, 1.0/3.0, 0.0],'M':[1.0/2.0, 0.0/2.0, 0.0]}
  paoflow.bands(ibrav=0, nk=401, band_path=path, high_sym_points=special_points)

  kcenter = np.array([1.0/3.0,2.0/3.0,0.0])
  kradius = 0.05

  paoflow.berry_phase(kspace_method='circle', nk1=51, kradius=kradius, kcenter=kcenter, sub=[0], closed=True, method='berry',fname='circle_berry_phase_0')
  circle_phase_0 = attr['berry_phase']
  paoflow.berry_phase(kspace_method='circle', nk1=51, kradius=kradius, kcenter=kcenter, sub=[1], closed=True, method='berry',fname='circle_berry_phase_1')
  circle_phase_1 = attr['berry_phase']
  paoflow.berry_phase(kspace_method='circle', nk1=51, kradius=kradius, kcenter=kcenter, sub=[0,1], closed=True, method='berry',fname='circle_berry_phase_01')
  circle_phase_01 = attr['berry_phase']

  print()
  print("Berry phase along circle with radius: ",kradius)
  print("  centered at k-point: ",kcenter)
  print("  for band 0 equals    : ", circle_phase_0)
  print("  for band 1 equals    : ", circle_phase_1)
  print("  for both bands equals: ",   circle_phase_01)
  print()

  klength = 0.1

  paoflow.berry_phase(kspace_method='square', nk1=51, nk2=51, kxlim=[kcenter[0]-klength/2,kcenter[0]+klength/2], kylim=[kcenter[1]-klength/2,kcenter[1]+klength/2], sub=[0], method='berry',fname='square_berry_phase_0')
  square_flux_0 = attr['berry_flux']
  paoflow.berry_phase(kspace_method='square', nk1=51, nk2=51, kxlim=[kcenter[0]-klength/2,kcenter[0]+klength/2], kylim=[kcenter[1]-klength/2,kcenter[1]+klength/2], sub=[1], method='berry',fname='square_berry_phase_1')
  square_flux_1 = attr['berry_flux']
  paoflow.berry_phase(kspace_method='square', nk1=51, nk2=51, kxlim=[kcenter[0]-klength/2,kcenter[0]+klength/2], kylim=[kcenter[1]-klength/2,kcenter[1]+klength/2], sub=[0,1], method='berry',fname='square_berry_phase_01')
  square_flux_01 = attr['berry_flux']

  print()
  print("Berry flux on square patch with length: ",klength)
  print("  centered at k-point: ",kcenter)
  print("  for band 0 equals    : ", square_flux_0)
  print("  for band 1 equals    : ", square_flux_1)
  print("  for both bands equals: ", square_flux_01)
  print()

  nk1 = 51
  nk2 = 51

  berry = np.genfromtxt('graphene2/square_berry_phase_0.dat')
  berry = berry.reshape((nk1,nk2,3),order='F')

  kgrid = np.genfromtxt('graphene2/square_berry_phase_0_kgrid_corners.dat')
  kgrid = kgrid.reshape((nk1+1,nk2+1,2),order='F')

  fig, ax = plt.subplots(figsize=(5,5))
  ax.pcolormesh(kgrid[:,:,0],kgrid[:,:,1],berry[:,:,2],shading='flat',rasterized=True,cmap='viridis_r')
  ax.set_xlabel(r'$k_x$',fontsize=18)
  ax.set_ylabel(r'$k_y$',fontsize=18)
  fig.tight_layout()
  fig.savefig('graphene2/berry_phase_k.pdf',bbox_inches='tight')

  paoflow.finish_execution()

if __name__== '__main__':
  main()

