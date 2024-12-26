
if __name__ == '__main__':
  from PAOFLOW import GPAO

  f_dos = './data_files/Si.dosdk_0.dat'
  f_band = './data_files/Si.bands_0.dat'
  f_symp = './data_files/Si.kpath_points.txt'

  pplt = GPAO.GPAO()

  # Functions arguments (tiltle y_lim, etc) can be used in any of the plot functions
  pplt.plot_dos(f_dos, title='Si2 FCC DoS', vertical=False)
  pplt.plot_bands(f_band, sym_points=f_symp)
  pplt.plot_dos_beside_bands(f_dos, f_band, sym_points=f_symp, y_lim=(-11.5,2), dos_ticks=True)
