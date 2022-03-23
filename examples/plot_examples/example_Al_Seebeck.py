
if __name__ == '__main__':
  from XtraCrysPy import PAO_Plot,QE_Plot

  f_seebeck = './data_files/AlP.Seebeck_0.dat'

  pplt = PAO_Plot.PAO_Plot()

  # Setting t_ele to [] causes the diagonal elements to be averaged.
  pplt.plot_seebeck_PAO(f_seebeck, t_ele=[], x_lim=(-.5,3.5), y_lim=(-2e4,2e4), col='black')

