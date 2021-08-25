from PAOFLOW import PAOFLOW
import numpy as np

def main ():

  params = { 'label':'magnetic_bilayer' }

  paoflow = PAOFLOW.PAOFLOW(model=params, verbose=True)

  arry,attr = paoflow.data_controller.data_dicts()

  hamiltonian = arry['HRs']

  path = 'X-G-M-X'
  special_points = {'G':[0.0, 0.0, 0.0],'M':[.5, .5, 0.0],'X':[.5, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  #special_points = {'G':[0.0, 0.0, 0.0],'M':[np.pi/2, np.pi/2, 0.0],'X':[np.pi/2, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  paoflow.bands(ibrav=1, nk=200, band_path=path, high_sym_points=special_points)

if __name__ == '__main__':
  main()
