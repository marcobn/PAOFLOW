from PAOFLOW import PAOFLOW
import numpy as np

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='./CrI3.save',outputdir='./output-per-species',verbose=True)
  arry,attr = paoflow.data_controller.data_dicts()

  paoflow.read_atomic_proj_QE()
  paoflow.projectability(pthr = 0.90)
  paoflow.pao_hamiltonian()
  # SOC per Species
  # soc_strengh={ 'Cr': [p-orbitals,d-orbitals],'I': [p-orbitals,d-orbitals] 
  paoflow.adhoc_spin_orbit(phi=0.0,theta=0.0,soc_strengh={ 'Cr': [0.0,0.037],'I': [0.829,0.0] })


  path = 'G-M-K-G'
  special_points = {'G':[0.0, 0.0, 0.0],'M':[0.0, 0.5, 0.0],'K':[0.333333333,0.33333333,0.0]}
  paoflow.bands(ibrav=4, nk=100, band_path=path, high_sym_points=special_points)

  paoflow.finish_execution()

if __name__== '__main__':
  main()
