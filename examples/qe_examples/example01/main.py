from PAOFLOW import PAOFLOW

def main():

  # Initialize PAOFLOW, indicating the name of the QE save directory.
  #   outputdir is named 'output' by default
  #   smearing is 'gauss' by default
  paoflow = PAOFLOW.PAOFLOW(savedir='silicon.save', outputdir='output', smearing='gauss', npool=1, verbose=True)
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

  # Calculate eigenvalues on the default ibrav=2 path
  paoflow.bands(ibrav=2, nk=2000)

  # Dimension of the grid is doubled by default
  #  e.g. 12x12x12 -> 24x24x24
  paoflow.interpolated_hamiltonian()

  # Calculate eigenvalues on the entire BZ grid
  paoflow.pao_eigh()

  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(emin=-12., emax=2.2, ne=1000)
  paoflow.transport(emin=-12., emax=2.2)
  paoflow.finish_execution()

if __name__== '__main__':
  main()
