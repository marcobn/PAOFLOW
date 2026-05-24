from PAOFLOW import PAOFLOW

def main():

  label = 'sym'
  outdir = 'output_%s'%label
  savedir = 'silicon_%s.save'%label

  # Initialize PAOFLOW, indicating the name of the QE save directory.
  paoflow = PAOFLOW.PAOFLOW(savedir=savedir, outputdir=outdir, smearing='gauss', npool=1, verbose=True)
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

  # Dump PAOFLOW data for future runs.
  paoflow.restart_dump(fname_prefix='paoflow_dump')

  paoflow.finish_execution()

if __name__== '__main__':
  main()
