from PAOFLOW import PAOFLOW

paoflow = PAOFLOW(savedir='./nscf_nspin2/',  
                  outputdir='./output_nspin2/', 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = {'Cr':['3D','4S','4P','4D','5S','5P','5D'],
                'I':['5S','5P','5D','6S','6P','4F']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability(pthr=0.85)
paoflow.pao_hamiltonian()
path = 'G-M-K-G'
sym_points = {'G':[0.0, 0.0, 0.0],
            'M':[0.5, 0.0, 0.0],
            'K':[1/3, 1/3, 0.0]}
paoflow.bands(ibrav=0, nk=500, band_path = path, high_sym_points = sym_points)

import matplotlib.pyplot as plt

data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()
eband = arry['E_k']

# Plot Bands
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ax[0].plot(eband[:,0,0], color='red', alpha=0.5, label="up")
ax[0].plot(eband[:,0,1], color='blue', alpha=0.5, label="down")
ax[1].plot(eband[:,0,0], color='red', alpha=0.5, label="up")
ax[1].plot(eband[:,0,1], color='blue', alpha=0.5, label="down")
for ib in range(1,eband.shape[1]):
    ax[0].plot(eband[:,ib,0], color='red', alpha=0.5)
    ax[0].plot(eband[:,ib,1], color='blue', alpha=0.5)
    ax[1].plot(eband[:,ib,0], color='red', alpha=0.5)
    ax[1].plot(eband[:,ib,1], color='blue', alpha=0.5)
    
ax[0].legend(loc=[0.05,0.2])
ax[1].legend(loc=[0.05,0.5])
ax[0].set_xlim([0,eband.shape[0]-1])
ax[1].set_xlim([0,eband.shape[0]-1])
ax[1].set_ylim([-1.5,1.5])
ax[0].set_ylabel("E (eV)")
plt.savefig('CrI3_nspin2_VASP.png',bbox_inches='tight') 

