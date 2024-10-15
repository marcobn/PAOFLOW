from PAOFLOW import PAOFLOW

paoflow = PAOFLOW(savedir='./nscf/',  
                  outputdir='./output/', 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = {'Mn':['3P','3D','4S','4P','4D','4F'],
                'F':['2P','3S','3P','3D','4S']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability(pthr=0.90)
paoflow.pao_hamiltonian()
paoflow.bands(ibrav=6, nk=500)

import matplotlib.pyplot as plt

data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()
eband = arry['E_k']

# Plot Bands (all bands and bands near Fermi energy)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
ax[0].plot(eband[:,0,0], color='red', alpha=0.6, label="up")
ax[0].plot(eband[:,0,1], color='blue', alpha=0.6, label="down")
for ib in range(1, eband.shape[1]):
    ax[0].plot(eband[:,ib,0], color='red', alpha=0.6)
    ax[0].plot(eband[:,ib,1], color='blue', alpha=0.6)
    ax[1].plot(eband[:,ib,0], color='red', alpha=0.6)
    ax[1].plot(eband[:,ib,1], color='blue', alpha=0.6)
ax[1].set_ylim([-1,0.2])
ax[0].set_xlim([0,eband.shape[0]-1])
ax[1].set_xlim([0,eband.shape[0]-1])
ax[0].set_ylabel("E (eV)")
ax[0].legend()
plt.savefig('MnF2_VASP.png',bbox_inches='tight')

