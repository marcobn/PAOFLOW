from PAOFLOW import PAOFLOW
from PAOFLOW import GPAO
import   numpy as np

def main():

  pplt = GPAO.GPAO()
  paoflow = PAOFLOW.PAOFLOW(savedir='./Bi.save')

  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  paoflow.adhoc_spin_orbit(phi=0.0,theta=0.0,soc_strengh={ 'Bi': [1.5,0.0] })

  path = 'xX-G-X'
  special_points = {'G':[0.0, 0.0, 0.0],'X':[0.5,0.0,0.0],'xX':[-0.5,0.0,0.0]}
  paoflow.bands(ibrav=0, nk=100, band_path=path, high_sym_points=special_points)

  # Projection on the outmost sites of the nanoribbon
  # index of the sites to obtain the projection. 
  paoflow.site_projected_bands(site_proj=np.array([0,1,2,3,4,5,18,19,20,21,22,23]))
 
  # Ploting Site Projection

  outputdir='./output/' 
  f_band = outputdir + 'site-projected-bands_0.dat'
  f_symp = outputdir + 'kpath_points.txt'

  label       = '$\epsilon-\epsilon_{F}$ (eV)'
  cbar_label  = 'Edge Sites Projection'
  filename    = 'edge_states_projection.png'

  pplt.plot_weighted_bands( outputdir, f_band, f_symp, None, cbar_label,  
                           label, filename, y_lim = (-1,1))

  paoflow.finish_execution()

if __name__== '__main__':
  main()

