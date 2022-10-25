from PAOFLOW import PAOFLOW
import numpy as np

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='./pds.save', outputdir = 'Ekai_xx_tst', npool = 8, verbose = 1)
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.interpolated_hamiltonian(nfft1=16, nfft2=16, nfft3=14)
  paoflow.pao_eigh()
  paoflow.spin_operator()
  paoflow.spin_texture(fermi_up= 0.2, fermi_dw = -0.2)
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing() 
  paoflow.rashba_edelstein(emin = -0.2, emax = 0.2, ne = 501)

  paoflow.finish_execution()

if __name__== '__main__':
  main()


