
if __name__ == '__main__':
  from PAOFLOW import GPAO

  f_seebeck = './data_files/AlP.Seebeck_0.dat'

  pplt = GPAO.GPAO()

  # Setting t_ele to [] causes the diagonal elements to be averaged.
  pplt.plot_seebeck(f_seebeck, t_ele=[], x_lim=(-.5,3.5), y_lim=(-2e4,2e4), col='black')

