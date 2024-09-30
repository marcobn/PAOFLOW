from src.PAOFLOW import PAOFLOW
import numpy as np

outdir = './output'
paoflow = PAOFLOW(savedir='./nscf',  
                  outputdir=outdir, 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = {'Fe':['3P','3D','4S','4P','4D']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability(pthr=0.85)
paoflow.pao_hamiltonian()
paoflow.bands(ibrav=3, nk=500)

paoflow.interpolated_hamiltonian()
paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing(smearing='m-p')
paoflow.anomalous_Hall(do_ac=True, emin=-6., emax=4., a_tensor=np.array([[0,1]]))


import matplotlib.pyplot as plt

data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()
eband = arry['E_k']

# Plot Bands
fig = plt.figure(figsize=(6,4))
plt.plot(eband[:,0], color='black', label="k = 12*12*12")
for ib in range(1, eband.shape[1]):
    plt.plot(eband[:,ib], color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()

# Plot AHC
ahc = np.loadtxt(outdir+'ahcEf_xy.dat')
fig = plt.figure(figsize=(4,3))
plt.xlabel("E (eV)")
plt.ylabel("AHC_xy")
plt.plot(ahc[:,0], ahc[:,1], color='black')
plt.show()

