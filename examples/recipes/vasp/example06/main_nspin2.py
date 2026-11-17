from PAOFLOW import PAOFLOW

paoflow = PAOFLOW(savedir='./nscf_nspin2',  
                  outputdir='./output_nspin2', 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = {'Fe':['3D','4S','4P','4D','5S','5P'],
                'Rh':['4D','4F','5S','5P','5D']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability(pthr=0.90)
paoflow.pao_hamiltonian()
paoflow.bands(ibrav=1, nk=500)

import matplotlib.pyplot as plt

data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()
eband = arry['E_k']

# Plot Bands
fig = plt.figure(figsize=(6,4))
plt.plot(eband[:,0,0], color='red', alpha=0.6, label="up")
plt.plot(eband[:,0,1], color='blue', alpha=0.6, label="down")
for ib in range(1, eband.shape[1]):
    plt.plot(eband[:,ib,0], color='blue', alpha=0.6)
    plt.plot(eband[:,ib,1], color='red', alpha=0.6)
plt.legend()
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.savefig('FeRh_nspin2_VASP.png',bbox_inches='tight')  
