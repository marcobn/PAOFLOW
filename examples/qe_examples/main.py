from PAOFLOW import PAOFLOW
from sys import argv

## Usage:
##    python main.py
##    python main.py <work_directory>
##    python main.py <work_directory> <inputfile_name>
##
## Defaults:
##    <work_directory> - './'
##    <inputfile_name> - 'inputfile.xml'

def main():

  argc = len(argv)
  arg1 = ('./' if argc<2 else argv[1])
  arg2 = ('inputfile.xml' if argc<3 else argv[2])

  # Start PAOFLOW with an inputfile in the current directory
  #
  # PAOFLOW will us data attributes read from
  #   inputfile.xml for the following calculations
  paoflow = PAOFLOW.PAOFLOW(workpath=arg1, inputfile=arg2, outputdir="")

  # Get dictionary containers with the
  #   attributes and arrays read from inputfiles
  arry,attr = paoflow.data_controller.data_dicts()

  paoflow.projectability()

  paoflow.pao_hamiltonian(expand_wedge=attr['expand_wedge'],thresh=attr['symm_thresh'],
                          symmetrize=attr['symmetrize'],max_iter=attr['symm_max_iter'])

  if attr['write2file']:
    paoflow.write2file()

  paoflow.add_external_fields()

  if attr['writez2pack']:
    paoflow.write_Hamiltonian(fname='hamiltonian.dat')

  if attr['do_bands'] or attr['band_topology']:

    paoflow.bands(ibrav=int(attr["ibrav"]))

  if attr['spintexture'] or attr['spin_Hall']:
    paoflow.spin_operator(spin_orbit=attr['do_spin_orbit'])

  if attr['band_topology'] and not attr['onedim']:
    paoflow.topology()
  elif attr['onedim']:
    print('1D Band topology not supported with the PAOFLOW class')

  if attr['double_grid']:
    paoflow.interpolated_hamiltonian(nfft1=attr['nfft1'], nfft2=attr['nfft2'], nfft3=attr['nfft3'])

  paoflow.pao_eigh()

  if attr['fermisurf']:
    paoflow.fermi_surface()

  if attr['spintexture']:
    paoflow.spin_texture()

  paoflow.gradient_and_momenta()

  if attr['smearing'] is not None:
    paoflow.adaptive_smearing()

  if attr['do_dos'] or attr['do_pdos']:
    paoflow.dos(do_dos=attr['do_dos'], do_pdos=attr['do_pdos'], emin=attr['emin'], emax=attr['emax'])

  if attr['spin_Hall']:
    paoflow.spin_Hall(do_ac=attr['ac_cond_spin'], emin=attr['eminSH'], emax=attr['emaxSH'],s_tensor=arry['s_tensor'])

  if attr['Berry']:
    paoflow.anomalous_Hall(do_ac=attr['ac_cond_Berry'], emin=attr['eminAH'], emax=attr['emaxSH'],a_tensor=arry['a_tensor'])

  if attr['Boltzmann']:
    nt = (attr['tmax'] - attr['tmin'])/attr['tstep']
    if nt==0: nt = 1
    paoflow.transport(tmin=attr['tmin'], tmax=attr['tmax'], nt=nt, emin=attr['emin'], emax=attr['emax'], ne=attr['ne'], write_to_file=True)

  # Print the total execution time and request
  #   desired quantites for further processing
  paoflow.finish_execution()


if __name__== '__main__':
  main()
