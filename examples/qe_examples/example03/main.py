from PAOFLOW import PAOFLOW

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='pt.save', smearing='m-p')
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.bands(ibrav=2)
  paoflow.interpolated_hamiltonian()
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(emin=-8., emax=4., delta=.2)
  paoflow.transport(emin=-8., emax=4.)
  paoflow.finish_execution()

if __name__== '__main__':
  main()
