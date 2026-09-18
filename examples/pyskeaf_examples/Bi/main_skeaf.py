from PAOFLOW import PAOFLOW

def main():

  paoflow = PAOFLOW.PAOFLOW(outputdir='output', verbose=True)
  # paoflow.restart_load(fname_prefix='paoflow_dump')     # Load the saved data using the same file prefix
  # data_controller = paoflow.data_controller
  # arry, attr = data_controller.data_dicts()
  # paoflow.read_atomic_proj_QE()
  # paoflow.projectability(pthr=0.95)
  # paoflow.pao_hamiltonian()

  # # Calculate eigenvalues on the default ibrav=2 path
  # # paoflow.bands(ibrav=2, nk=2000)


  # paoflow.interpolated_hamiltonian()    # Dimension of the grid is doubled by default. e.g. 12x12x12 -> 24x24x24

  # paoflow.pao_eigh()    # Calculate eigenvalues on the entire BZ grid (i.e. output in k-space)
  # # paoflow.gradient_and_momenta()
  # # paoflow.adaptive_smearing()
  # # paoflow.spin_operator() # Only if for not the band
  # # paoflow.transport(emin=-12., emax=2.2)
  # paoflow.fermi_surface(fermi_dw=-0.2, fermi_up=0.2)
  paoflow.pyskeaf(
      fermi_energy=0.0,
      num_interpolation=60,
      azimuthal=(0.0, 0.0),
      polar=(0.0, 90.0),
      num_angles=37,
      frequency_tolerance=0.01,
      orbit_tolerance=0.05,
      allow_wall_orbits=True,
      verbose=False,
  )

  paoflow.finish_execution()

if __name__== '__main__':
  main()
