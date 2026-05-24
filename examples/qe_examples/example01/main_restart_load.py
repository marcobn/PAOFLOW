from PAOFLOW import PAOFLOW

def main():

  # Initialize PAOFLOW as a 'restart' run
  paoflow = PAOFLOW.PAOFLOW(restart=True)

  # Load the dumped data using the same file prefix
  paoflow.restart_load(fname_prefix='paoflow_dump')

  # Continue calculations
  paoflow.adaptive_smearing()
  paoflow.dos(emin=-12., emax=2.2, ne=1000)
  paoflow.transport(emin=-12., emax=2.2, t_tensor=[[0,0]])
  paoflow.dielectric_tensor(emax=6., d_tensor=[[0,0]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()
