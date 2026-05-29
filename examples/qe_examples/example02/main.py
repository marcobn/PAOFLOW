from PAOFLOW import PAOFLOW

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='al.save',verbose=True)
  paoflow.read_atomic_proj_QE()
  paoflow.projectability(pthr=.97)
  paoflow.pao_hamiltonian()
  paoflow.interpolated_hamiltonian()
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.dos(do_pdos=False, delta=.1, emin=-12., emax=3.)
  paoflow.transport(emin=-12., emax=3.)
  paoflow.finish_execution()

if __name__== '__main__':
  main()
