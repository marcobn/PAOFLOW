from PAOFLOW import PAOFLOW

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='pt.save')
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.bands(ibrav=2, nk=2000)
  paoflow.topology(Berry=True, eff_mass=True, spin_Hall=True, spol=2, ipol=0, jpol=1)
  paoflow.interpolated_hamiltonian()
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(do_pdos=False, emin=-8., emax=4.)
  paoflow.spin_Hall(emin=-8., emax=4., s_tensor=[[0,1,2]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()
