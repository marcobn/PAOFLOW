from matplotlib import pyplot as plt
from PAOFLOW import PAOFLOW
import numpy as np

def main ():

  params = { 'label':'magnetic_bilayer' }

  paoflow = PAOFLOW.PAOFLOW(model=params, verbose=True)

  arry,attr = paoflow.data_controller.data_dicts()

  bands = []
  kq = [[i,0] for i in range(50,0,-1)] + [[i,i] for i in range(51)] + [[50,i] for i in range(49,0,-1)]
  for k in kq:
    ham = arry['Hks'][:,:,k[0],k[1],0,0]
    E,v_k = np.linalg.eigh(ham)
    bands.append(E)

  paoflow.pao_eigh()

  paoflow.gradient_and_momenta()

  paoflow.adaptive_smearing()

  # * #
  paoflow.spin_operator(spin_orbit=True)
  # * #

  paoflow.spin_Hall(twoD=True, emin=-1., emax=1., fermi_up=1., fermi_dw=-1., s_tensor=[[0,1,2]])

  plt.plot(np.array(bands))
  plt.show()

  '''
  quit()
  path = 'X-G-M-X'
  special_points = {'G':[0.0, 0.0, 0.0],'M':[.5, .5, 0.0],'X':[.5, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  #special_points = {'G':[0.0, 0.0, 0.0],'M':[np.pi/2, np.pi/2, 0.0],'X':[np.pi/2, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  paoflow.bands(ibrav=1, nk=200, band_path=path, high_sym_points=special_points)
  '''
if __name__ == '__main__':
  main()
