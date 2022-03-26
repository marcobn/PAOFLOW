
if __name__ == '__main__':
  from PAOFLOW import GPAO

  f_sigma = './data_files/Al.sigmadk_0.dat'

  pplt = GPAO.GPAO()

  # Argument t_ele default is [[0,0], [1,1], [2,2]]
  #   which plots the 3 diagonal elements
  pplt.plot_electrical_conductivity(f_sigma, title='Ba8Cu16As30 Conductivity')

