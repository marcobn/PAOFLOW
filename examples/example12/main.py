from PAOFLOW import PAOFLOW
import numpy as np
def main():
  paoflow = PAOFLOW.PAOFLOW(outputdir='./output',savedir='./MoS2.save',verbose=True,npool=2)
  arry,attr = paoflow.data_controller.data_dicts()

  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()
  
  paoflow.interpolated_hamiltonian(50,50,1)
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()

  paoflow.orbital_Hall(twoD=True,emin=-5.0,emax=1.5,o_tensor=[[0,1,2]])
  
  paoflow.finish_execution()

if __name__== '__main__':
  main()
