from src.PAOFLOW import PAOFLOW
import numpy as np

outdir = './output'
paoflow = PAOFLOW(savedir='./nscf',  
                  outputdir=outdir, 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = {'Pt':['5D','6S','6P','7S','7P']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability()
paoflow.pao_hamiltonian(expand_wedge=True)
paoflow.bands(ibrav=2, nk=500)

paoflow.interpolated_hamiltonian()
paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing(smearing='m-p')
paoflow.spin_Hall(emin=-8., emax=4., s_tensor=[[0,1,2]])


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
plt.show()

# Plot SHC
shc = np.loadtxt(f'{outdir}/shcEf_z_xy.dat') 
fig = plt.figure(figsize=(4,3))
plt.plot(shc[:,0], shc[:,1], color='black')
plt.xlabel("E (eV)")
plt.ylabel("SHC_xy")
plt.show()
