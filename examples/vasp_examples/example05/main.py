from PAOFLOW import PAOFLOW

paoflow = PAOFLOW(savedir='./nscf/',  
                  outputdir='./output/', 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = { 'Ir': ['5P','5D','6S','6P','7S'],
                 'Mn': ['3P','3D','4S','4P','4D'] }

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
for ib in range(eband.shape[1]):
    plt.plot(eband[:,ib], color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.savefig('Mn3Ir_VASP.png',bbox_inches='tight')

