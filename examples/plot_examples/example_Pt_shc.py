
if __name__ == '__main__':
  from PAOFLOW import GPAO
  from glob import glob

  pplt = GPAO.GPAO()

  # Plot a single shc file
  f_shc = './data_files/Pt.shcEf_z_xy.dat'
  pplt.plot_shc(f_shc, title='Pt SHC', legend=False, cols='black')

  # Plot a list of shc files
  #   cols argument can also be provided as a list of strings or 3-colors
  files = glob('./data_files/Pt.shc*.dat')
  pplt.plot_shc(files, title='Pt SHC')

