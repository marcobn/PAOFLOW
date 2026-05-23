from PAOFLOW import PAOFLOW
import numpy as np

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='./pt.save')

  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  paoflow.adhoc_spin_orbit(phi=0.0,theta=0.0,soc_strengh={ 'Pt': [0.0,0.553] })

  path = 'gG-X-W-K-gG-L-U-W-L-K|U-X'
  special_points = {'gG'   : (0.0, 0.0, 0.0),
              'K'  : (0.375, 0.375, 0.750),
              'L'  : (0.5, 0.5, 0.5),
              'U'  : (0.625, 0.250, 0.625),
              'W'  : (0.5, 0.25, 0.75),
              'X'  : (0.5, 0.0, 0.5)}

  paoflow.bands(ibrav=2, nk=100, band_path=path, high_sym_points=special_points)


  paoflow.topology(Berry=True, eff_mass=True, spin_Hall=True, spol=2, ipol=0, jpol=1)
  paoflow.interpolated_hamiltonian()
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(do_pdos=False, emin=-8., emax=4., ne=100)
  paoflow.spin_Hall(emin=-8., emax=4., s_tensor=[[0,1,2]], ne=100)


  paoflow.finish_execution()

if __name__== '__main__':
  main()
