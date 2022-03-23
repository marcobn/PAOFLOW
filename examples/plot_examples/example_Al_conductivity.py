
if __name__ == '__main__':
  from XtraCrysPy import PAO_Plot,QE_Plot

  f_sigma = './data_files/Al.sigmadk_0.dat'

  pplt = PAO_Plot.PAO_Plot()

  # Argument t_ele default is [[0,0], [1,1], [2,2]]
  #   which plots the 3 diagonal elements
  pplt.plot_electrical_conductivity_PAO(f_sigma, title='Ba8Cu16As30 Conductivity')

