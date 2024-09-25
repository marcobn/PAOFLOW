#!/usr/bin/env python
# coding: utf-8

# # Important Notes
# (1) When using VASP, PAOFLOW requires spglib. All examples show below are run with ISYM = 2 in VASP. ISYM = -1 and 0 also work (For LSORBIT=T and Magnetic system, ISYM = -1 and 0 are equivalent, i.e. full grid).  Make sure the PAOFLOW printed space group is correct.   
# (2) Either run VASP SCF with LWAVE=T and a fine K-mesh, or run VASP SCF (LCHARG=T) first with a coarse K-mesh, then non-SCF (ICHARG=11 and LWAVE=T) with finer K-mesh. The later one saves some computational cost.  
# (3) If magnetic, MAGMOM in INCAR is necessary. This is because PAOFLOW reads the MAGMOM tag in vasprun.xml to determine the symmetry, which comes from INCAR.  
# (4) PAOFLOW reads "vasprun.xml" and "WAVECAR", make sure they are in the directory.   
# (5) If using VASP, when calling pao_hamiltonian(), DO NOT set "open_wedge=False" even if symmetry is turned off (ISYM = -1). This is because VASP uses a different set of k-mesh and a mapping between the k-points is required.

# In[1]:


# Change PAOFLOW export directory if necessary
from src.PAOFLOW import PAOFLOW
import numpy as np
import matplotlib.pyplot as plt


# # Example01: Si

# In[2]:


paoflow = PAOFLOW(savedir='./examples_vasp/Si/nscf_nspin1',  
                  outputdir='./examples_vasp/Si/output_nspin1', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[3]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Si':['3S','3P','3D','4S','4P','4F']}
paoflow.projections(internal=True)  # "internal=True" is optional, always use internal basis when dft == 'VASP'


# In[4]:


paoflow.projectability()


# In[5]:


paoflow.pao_hamiltonian()


# In[6]:


paoflow.bands(ibrav=2,nk=500)


# In[7]:


fig = plt.figure(figsize=(6,4))
eband = arry['E_k']
plt.plot(eband[:,0],color='black')
for ib in range(1,eband.shape[1]):
    plt.plot(eband[:,ib],color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
# plt.savefig('Si_VASP.png',bbox_inches='tight')  


# # Example 01: Si (nspin = 2)

# In[8]:


paoflow = PAOFLOW(savedir='./examples_vasp/Si/nscf_nspin2',  
                  outputdir='./examples_vasp/Si/output_nspin2', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[9]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Si':['3S','3P','3D','4S','4P','4F']}
paoflow.projections(internal=True)  # "internal=True" is optional, always use internal basis when dft == 'VASP'


# In[10]:


paoflow.projectability()


# In[11]:


paoflow.pao_hamiltonian()


# In[12]:


paoflow.bands(ibrav=2,nk=500)


# In[13]:


fig = plt.figure(figsize=(6,4))
eband = arry['E_k']
plt.plot(eband[:,0,0],color='blue',alpha=0.6,label="up")
plt.plot(eband[:,0,1],color='red',alpha=0.6,label="down")
for ib in range(1,eband.shape[1]):
    plt.plot(eband[:,ib,0],color='blue',alpha=0.6)
    plt.plot(eband[:,ib,1],color='red',alpha=0.6)
plt.legend()
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
# plt.savefig('Si_VASP_nspin2.png',bbox_inches='tight')  


# # Example02: Pt (with SOC)

# In[14]:


outdir = './examples_vasp/Pt/output/'
paoflow = PAOFLOW(savedir='./examples_vasp/Pt/nscf/',
                  outputdir=outdir, 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[15]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Pt':['5D','6S','6P','7S','7P']}
paoflow.projections(internal=True)  # "internal=True" is optional, always use internal basis when dft == 'VASP'


# In[16]:


paoflow.projectability()


# In[17]:


paoflow.pao_hamiltonian(expand_wedge=True)


# In[18]:


paoflow.bands(ibrav=2,nk=500)


# In[19]:


fig = plt.figure(figsize=(6,4))
eband = arry['E_k']
for ib in range(eband.shape[1]):
    plt.plot(eband[:,ib],color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()


# In[20]:


paoflow.interpolated_hamiltonian()
paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing(smearing='m-p')


# In[21]:


paoflow.spin_Hall(emin=-8., emax=4., s_tensor=[[0,1,2]])


# In[22]:


shc = np.loadtxt(outdir+'shcEf_z_xy.dat') 
fig = plt.figure(figsize=(4,3))
plt.plot(shc[:,0],shc[:,1],color='black')
plt.xlabel("E (eV)")
plt.ylabel("SHC_xy")
plt.show()


# # Example03: Fe (with SOC, FM)

# In[23]:


outdir = './examples_vasp/Fe/output/'
paoflow = PAOFLOW(savedir='./examples_vasp/Fe/nscf/',
                  outputdir=outdir, 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[24]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Fe':['3P','3D','4S','4P','4D']}
paoflow.projections(internal=True)


# In[25]:


paoflow.projectability(pthr=0.85)


# In[26]:


paoflow.pao_hamiltonian()


# In[27]:


paoflow.bands(ibrav=3,nk=500)


# In[28]:


fig = plt.figure(figsize=(6,4))
eband = arry['E_k']
plt.plot(eband[:,0],color='black',label="k = 12*12*12")
for ib in range(1,eband.shape[1]):
    plt.plot(eband[:,ib],color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
# plt.savefig('Fe_VASP.png',bbox_inches='tight')


# In[29]:


paoflow.interpolated_hamiltonian()
paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing(smearing='m-p')


# In[30]:


paoflow.anomalous_Hall(do_ac=True, emin=-6., emax=4., a_tensor=np.array([[0,1]]))


# In[31]:


ahc = np.loadtxt(outdir+'ahcEf_xy.dat')
fig = plt.figure(figsize=(4,3))
plt.xlabel("E (eV)")
plt.ylabel("AHC_xy")
plt.plot(ahc[:,0],ahc[:,1],color='black')
plt.show()


# # Example04: MnF2 (nspin=2, collinear AFM)

# In[32]:


paoflow = PAOFLOW(savedir='./examples_vasp/MnF2/nscf',
                  outputdir='./examples_vasp/MnF2/output', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[33]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Mn':['3P','3D','4S','4P','4D'],
                        'F':['3S','3P','3D','4F']}
paoflow.projections(internal=True) 


# In[34]:


paoflow.projectability(pthr=0.85)


# In[35]:


paoflow.pao_hamiltonian()


# In[36]:


paoflow.bands(ibrav=6,nk=500)


# In[37]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
# plot the paoflow bands (all bands and bands near Fermi energy)
eband = arry['E_k']
ax[0].plot(eband[:,0,0],color='red',alpha=0.6,label="up")
ax[0].plot(eband[:,0,1],color='blue',alpha=0.6,label="down")
for ib in range(1,eband.shape[1]):
    ax[0].plot(eband[:,ib,0],color='red',alpha=0.6)
    ax[0].plot(eband[:,ib,1],color='blue',alpha=0.6)
    ax[1].plot(eband[:,ib,0],color='red',alpha=0.6)
    ax[1].plot(eband[:,ib,1],color='blue',alpha=0.6)
ax[1].set_ylim([-1,0.2])
ax[0].set_xlim([0,eband.shape[0]-1])
ax[1].set_xlim([0,eband.shape[0]-1])
ax[0].set_ylabel("E (eV)")
ax[0].legend()
plt.show()
# plt.savefig('MnF2_VASP.png',bbox_inches='tight')


# # Example05: Mn3Ir (with SOC, noncollinear 120$^\circ$ AFM)

# In[38]:


paoflow = PAOFLOW(savedir='./examples_vasp/Mn3Ir/nscf',
                  outputdir='./examples_vasp/Mn3Ir/output', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[39]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Ir':['5P','5D','6S','6P','7S'],
                        'Mn':['3P','3D','4S','4P','4D']}
paoflow.projections(internal=True) 


# In[40]:


paoflow.projectability(pthr=0.9)


# In[41]:


paoflow.pao_hamiltonian()


# In[42]:


paoflow.bands(ibrav=1,nk=500)


# In[43]:


fig = plt.figure(figsize=(6,4))
# plot the paoflow bands
eband = arry['E_k']
for ib in range(eband.shape[1]):
    plt.plot(eband[:,ib],color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
# plt.savefig('Mn3Ir_VASP.png',bbox_inches='tight')


# # Example06: FeRh (with SOC, FM)

# In[44]:


paoflow = PAOFLOW(savedir='./examples_vasp/FeRh/nscf_soc/',
                  outputdir='./examples_vasp/FeRh/output_soc/', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[45]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Fe':['3P','3D','4S','4P','4D'],
                        'Rh':['4P','4D','5S','5P']}
paoflow.projections(internal=True)


# In[46]:


paoflow.projectability(pthr=0.84)


# In[47]:


paoflow.pao_hamiltonian()


# In[48]:


paoflow.bands(ibrav=1,nk=500)


# In[49]:


fig = plt.figure(figsize=(6,4))
eband = arry['E_k']
for ib in range(eband.shape[1]):
    plt.plot(eband[:,ib],color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
# plt.savefig('FeRh_VASP.png',bbox_inches='tight')  


#  # Example06: FeRh (nspin = 2, FM)

# In[50]:


paoflow = PAOFLOW(savedir='./examples_VASP/FeRh/nscf_nspin2/',
                  outputdir='./examples_VASP/FeRh/output_nspin2/', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[51]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Fe':['3P','3D','4S','4P','4D'],
                        'Rh':['4P','4D','5S','5P']}
paoflow.projections(internal=True)


# In[52]:


paoflow.projectability(pthr=0.85)


# In[53]:


paoflow.pao_hamiltonian()


# In[54]:


paoflow.bands(ibrav=1,nk=500)


# In[55]:


fig = plt.figure(figsize=(6,4))
eband = arry['E_k']
plt.plot(eband[:,0,0],color='red',alpha=0.6,label="up")
plt.plot(eband[:,0,1],color='blue',alpha=0.6,label="down")
for ib in range(1,eband.shape[1]):
    plt.plot(eband[:,ib,0],color='blue',alpha=0.6)
    plt.plot(eband[:,ib,1],color='red',alpha=0.6)
plt.legend()
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
# plt.savefig('FeRh_VASP_nspin2.png',bbox_inches='tight')  


# # Example 07: CrI3 monolayer (nspin = 2, FM)

# In[56]:


paoflow = PAOFLOW(savedir='./examples_vasp/CrI3_monolayer/nscf_nspin2/',  
                  outputdir='./examples_vasp/CrI3_monolayer/output_nspin2/', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()


# In[57]:


attr['basispath'] = './BASIS/'
arry['configuration'] = {'Cr':['3D','4S','4P','4D'],
                        'I':['5S','5P','5D','4F']}
paoflow.projections(internal=True) 


# In[58]:


paoflow.projectability(pthr=0.75)


# In[59]:


paoflow.pao_hamiltonian()


# In[60]:


path = 'G-M-K-G'
sym_points = {'G':[0.0, 0.0, 0.0],
            'M':[0.5, 0.0, 0.0],
            'K':[1/3, 1/3, 0.0]}
paoflow.bands(ibrav=0, nk=500, band_path = path, high_sym_points = sym_points)


# In[61]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
eband = arry['E_k']
ax[0].plot(eband[:,0,0],color='red',alpha=0.5,label="up")
ax[0].plot(eband[:,0,1],color='blue',alpha=0.5,label="down")
ax[1].plot(eband[:,0,0],color='red',alpha=0.5,label="up")
ax[1].plot(eband[:,0,1],color='blue',alpha=0.5,label="down")
for ib in range(1,eband.shape[1]):
    ax[0].plot(eband[:,ib,0],color='red',alpha=0.5)
    ax[0].plot(eband[:,ib,1],color='blue',alpha=0.5)
    ax[1].plot(eband[:,ib,0],color='red',alpha=0.5)
    ax[1].plot(eband[:,ib,1],color='blue',alpha=0.5)
    
ax[0].legend(loc=[0.05,0.2])
ax[1].legend(loc=[0.05,0.5])
ax[0].set_xlim([0,502])
ax[1].set_xlim([0,502])
ax[1].set_ylim([-1.5,1.5])
ax[0].set_ylabel("E (eV)")
# ax[0].set_xticks(ticks=k_label,labels=["GM","M","K","GM"])
# ax[1].set_xticks(ticks=k_label,labels=["GM","M","K","GM"])
# for _,x in enumerate(k_label):
#     ax[0].axvline(x,color='k',linewidth=0.5)
#     ax[1].axvline(x,color='k',linewidth=0.5)
plt.show()


# # Example 07: CrI3 monolayer (with SOC, FM)

# In[62]:


paoflow_soc = PAOFLOW(savedir='./examples_vasp/CrI3_monolayer/nscf_soc/',  
                  outputdir='./examples_vasp/CrI3_monolayer/output_soc/', 
                  verbose=True,
                  dft="VASP")
data_controller = paoflow_soc.data_controller
arry_soc,attr_soc = paoflow_soc.data_controller.data_dicts()


# In[63]:


attr_soc['basispath'] = './BASIS/'
arry_soc['configuration'] = {'Cr':['3D','4S','4P'],
                        'I':['5S','5P','5D']}
paoflow_soc.projections(internal=True) 


# In[64]:


paoflow_soc.projectability(pthr=0.7)


# In[65]:


paoflow_soc.pao_hamiltonian()


# In[66]:


path = 'G-M-K-G'
sym_points = {'G':[0.0, 0.0, 0.0],
            'M':[0.5, 0.0, 0.0],
            'K':[1/3, 1/3, 0.0]}
paoflow_soc.bands(ibrav=0, nk=500, band_path = path, high_sym_points = sym_points)


# In[67]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
eband = arry_soc['E_k']
for ib in range(eband.shape[1]):
    ax[0].plot(eband[:,ib],color='k',alpha=0.5)
    ax[1].plot(eband[:,ib],color='k',alpha=0.5)
ax[0].set_xlim([0,502])
ax[1].set_xlim([0,502])
ax[1].set_ylim([-1.5,1.5])
ax[0].set_ylabel("E (eV)")
# ax[0].set_xticks(ticks=k_label,labels=["GM","M","K","GM"])
# ax[1].set_xticks(ticks=k_label,labels=["GM","M","K","GM"])
# for _,x in enumerate(k_label):
#     ax[0].axvline(x,color='k',linewidth=0.5)
#     ax[1].axvline(x,color='k',linewidth=0.5)
plt.show()


# In[ ]:




