from PAOFLOW import PAOFLOW
import numpy as np

def main():

  # Start PAOFLOW, interpolate Hamiltonian, spin operator, spin texture, compute gradient and momenta
  paoflow = PAOFLOW.PAOFLOW(savedir='./Te-L.save')
  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  paoflow.interpolated_hamiltonian(nfft1=20, nfft2=20, nfft3=16)
  paoflow.pao_eigh()
  paoflow.spin_operator()
  paoflow.spin_texture(fermi_up=0.0, fermi_dw=-0.5)
  paoflow.gradient_and_momenta()

  # Compute adaptive smearing and calculate the Rashba-Edelstein tensor elements
  paoflow.adaptive_smearing()
  paoflow.rashba_edelstein(emin=-0.5, emax=0.0, ne=501)

  paoflow.finish_execution()

if __name__== '__main__':
  main()
