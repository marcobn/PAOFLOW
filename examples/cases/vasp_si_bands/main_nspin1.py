import os

from PAOFLOW import PAOFLOW

paoflow = PAOFLOW.PAOFLOW(savedir='./nscf_nspin1/',  
                  outputdir='./output_nspin1/', 
                  verbose=True,
                  dft="VASP")


# all-electron radial tables: BASIS/ of the PAOFLOW source checkout
basis_path = os.path.join(os.path.dirname(PAOFLOW.__file__), '..', '..', 'BASIS') + os.sep
basis_config = {'Si':['3S','3P','3D','4S','4P','4D']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability()
paoflow.pao_hamiltonian()
paoflow.bands(ibrav=2, nk=500)


import matplotlib.pyplot as plt

data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()
eband = arry['E_k']

fig = plt.figure(figsize=(6,4))
plt.plot(eband[:,0], color='black')
for ib in range(1, eband.shape[1]):
    plt.plot(eband[:,ib], color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.savefig('Si_nspin1_VASP.png',bbox_inches='tight')  
