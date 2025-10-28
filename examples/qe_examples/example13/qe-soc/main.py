from PAOFLOW import PAOFLOW
from PAOFLOW import GPAO
import z2pack
import tbmodels
import scipy.optimize as so
import matplotlib.pyplot as plt
import   numpy as np

def main():

  pplt = GPAO.GPAO()
  paoflow = PAOFLOW.PAOFLOW(savedir='./Bi.save')

  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  paoflow.write_Hamiltonian(fname='Bi_bilayer_HRs.dat')
  path = 'M-G-K-M'

  special_points = {'M': [0.0, 0.5, 0.0],'G':[0.0, 0.0, 0.0],'K':[1.0/3.0,1.0/3.0,0.0]}
  paoflow.bands(ibrav=0, nk=200, band_path=path, high_sym_points=special_points)

  # Bandstructure plot
  outputdir='./output/' 
  f_band = outputdir + 'bands_0.dat'
  f_symp = outputdir + 'kpath_points.txt'

  pplt.plot_bands( f_band, f_symp, None, None, y_lim = (-3,1))

  print("#######################################################")
  print("                     Z2PACK                            ")
  print("#######################################################")

  model = tbmodels.Model.from_wannier_files(hr_file='./output/Bi_bilayer_HRs.dat')
  system = z2pack.tb.System(model, bands=30)

  result = z2pack.surface.run(
    system=system,
    surface=lambda t1, t2: [t1 / 2, t2, 0],
    load=False
  )

  print('Z2 topological invariant :   {0}'.format(z2pack.invariant.z2(result)))

  # Combining the two plots
  fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9,5))
  z2pack.plot.wcc(result, axis=ax[0])
  z2pack.plot.wcc(result, axis=ax[1])
  plt.savefig('plot-kz.pdf', bbox_inches='tight')


  paoflow.finish_execution()

if __name__== '__main__':
  main()

