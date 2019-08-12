from PAOFLOW import PAOFLOW

def main():

  paoflow = PAOFLOW.PAOFLOW(savedir='SnTe.save', npool=4, verbose=1)

  paoflow.projectability(pthr=0.90)
  paoflow.pao_hamiltonian()

  # Define high symmetry points and the path comprised of such points
  # '-' defines a continuous transition to the next high symmetry point
  # '|' defines a discontinuous break in from one high symmetry point to the next

  path = 'G-X-S-Y-G'
  special_points = {'G':[0.0, 0.0, 0.0],'S':[0.5, 0.5, 0.0],'X':[0.5, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  paoflow.bands(ibrav=8, nk=1000, band_path=path, high_sym_points=special_points)

  paoflow.interpolated_hamiltonian(nfft1=140, nfft2=140, nfft3=1)  
  paoflow.pao_eigh()
  paoflow.spin_operator()
  paoflow.fermi_surface(fermi_up=0.0, fermi_dw=-1. )
  paoflow.spin_texture(fermi_up=0.0, fermi_dw=-1. )
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
  paoflow.spin_Hall(twoD=True, emin=-3.5, emax=1.5, s_tensor=[[0,1,2],[1,0,2]])
  paoflow.finish_execution()

if __name__== '__main__':
  main()
