
if __name__ == '__main__':
  from PAOFLOW import GPAO

  f_symp = './data_files/Fe.kpath_points.txt'
  f_band = './data_files/Fe.bands_0.dat'
  f_berry = './data_files/Fe.Omega_z_xy.dat'

  pplt = GPAO.GPAO()

  # Functions arguments (tiltle y_lim, etc) can be used in any of the plot functions
  pplt.plot_berry(f_berry, sym_points=f_symp)
  pplt.plot_bands(f_band, sym_points=f_symp, y_lim=(-4,4))
  berry_label = '$\Omega^{z}$($\\bf{k}$)'
  pplt.plot_berry_under_bands(f_berry, f_band, sym_points=f_symp, y_lim=(-4,4), dos_ticks=True, berry_label=berry_label)
