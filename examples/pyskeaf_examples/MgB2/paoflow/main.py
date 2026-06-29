from PAOFLOW import PAOFLOW

def main():

  # Initialize PAOFLOW, indicating the name of the QE save directory.
  paoflow = PAOFLOW.PAOFLOW(savedir='MgB2.save', outputdir='output_paoflow', smearing='gauss', npool=4, verbose=True, restart=False)
  # paoflow.restart_load(fname_prefix='paoflow_dump')     # Load the saved data using the same file prefix
  data_controller = paoflow.data_controller
  arry, attr = data_controller.data_dicts()
  paoflow.read_atomic_proj_QE()
  paoflow.projectability(pthr=0.80)
  paoflow.pao_hamiltonian()

  # Calculate eigenvalues on the default ibrav=2 path
  # paoflow.bands(ibrav=4, nk=2000)


  paoflow.interpolated_hamiltonian()    # Dimension of the grid is doubled by default. e.g. 12x12x12 -> 24x24x24

  paoflow.pao_eigh()    # Calculate eigenvalues on the entire BZ grid (i.e. output in k-space)
  # paoflow.gradient_and_momenta()
  # paoflow.adaptive_smearing()
  # paoflow.spin_operator() # Only if for not the band
  # paoflow.transport(emin=-12., emax=2.2)
  paoflow.fermi_surface(fermi_dw=-0.2, fermi_up=0.2)
  # paoflow.spin_texture(fermi_dw=-0.2, fermi_up=0.2) # Same as for fermi surface
  # paoflow.dos(do_dos=True, do_pdos=True, delta=0.01, emin=-0.1, emax=0.1, ne=1000)
  # paoflow.anomalous_Hall(do_ac=False, emin=-0.2, emax=0.2, fermi_dw=-0.2, fermi_up=0.2, a_tensor=None )
  # paoflow.spin_Hall(twoD=False, do_ac=False,fermi_dw=-0.2, fermi_up=0.2, s_tensor=None ) # Change values fermi_dw=-0.2, fermi_up=0.2,

  
  # paoflow.restart_dump(fname_prefix='paoflow_dump')     # Save PAOFLOW data for future runs.


  paoflow.finish_execution()

if __name__== '__main__':
  main()
