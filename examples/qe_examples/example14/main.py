from PAOFLOW import PAOFLOW
from PAOFLOW import GPAO
import z2pack
import tbmodels
import scipy.optimize as so
import matplotlib.pyplot as plt
import   numpy as np

def main():

  pplt = GPAO.GPAO()
  paoflow = PAOFLOW.PAOFLOW(savedir='./pt.save')

  paoflow.read_atomic_proj_QE()
  paoflow.projectability()
  paoflow.pao_hamiltonian()

# Inplane band-structure. Notice the oscilations around -10eV. This is due to few k-points. K-mesh 5x5x3 is only for testing. 
  path = 'G-X-Y-G'
  special_points = {'G':[0.0, 0.0, 0.0],'X':[0.5, 0.0, 0.0],'Y':[0.0, 0.5, 0.0]}
  paoflow.bands(ibrav=0, nk=100, band_path=path, high_sym_points=special_points)

  # Bandstructure plot
  outputdir='./output/' 
  f_band = outputdir + 'bands_0.dat'
  f_symp = outputdir + 'kpath_points.txt'
  pplt.plot_bands( f_band, f_symp, None, None, y_lim = (-10,5))

# Interpolate Bands. K-mesh 10x10x6 is only for testing. 
  paoflow.interpolated_hamiltonian(nfft1=10,nfft2=10,nfft3=6)
  paoflow.pao_eigh()
  paoflow.gradient_and_momenta()
  paoflow.adaptive_smearing()
# Calculating total SHC
  print("               Calculating total SHC")
  paoflow.spin_Hall(emin=-1.0, emax=1.0, ne=1001, s_tensor=[[0,1,2]])

  shc_total    = np.loadtxt('./output/shcEf_z_xy.dat')
  shc_total_ef = shc_total[500,1] # SHC at Fermi Level

# SHC contribution from the first layer (atomic site = 0)
# shc_proj is an array with the sites to project indices. Here we are projeting on site zero. First layer.

  print("               Calculating First Layer SHC")
  paoflow.spin_Hall(twoD=False,emin=-1.0, emax=1.0, ne=1001, s_tensor=[[0,1,2]],shc_proj=[0])
  
  shc_0    = np.loadtxt('./output/shcEf_z_xy.dat')
  shc_0_ef = shc_0[500,1] # SHC at Fermi Level

# SHC Total x First Layer. Notice, since our calculation has 4 layers. We further normalized the Total SHC by the number of layers (4).
  fig = plt.figure(figsize=(8,5))
  plt.plot(shc_total[:,0],shc_total[:,1]/4,label='Total/4'    ,color='red'  ,linewidth=3)
  plt.plot(shc_0[:,0]    ,shc_0[:,1]    ,label='1st Layer',color='green',linewidth=3)
  plt.xlim(-1,1)
  plt.xlabel(r" E- E$_f$ (eV)")
  plt.ylabel(r'$\sigma^{z}_{xy}\;[\,(\hbar/2e)\,\Omega^{-1}\,\mathrm{m^{-1}}\,]$')
  plt.title(r'SHC Total x First Layer')
  plt.legend()
  plt.show()

# Calculating SHC for each Layer
  print("               Calculating SHC for each Layer")
  layers=4 # Number of layers

  shc_layer = np.zeros((layers,1001,2),dtype=float)
  shc_ef    = np.zeros(layers,dtype=float)

  for i in range(layers):

      paoflow.spin_Hall(emin=-1, emax=1, ne=1001, s_tensor=[[0,1,2]],shc_proj=[i])

      shc_layer[i] = np.loadtxt('./output/shcEf_z_xy.dat')
      shc_ef[i]    = shc_layer[i,500,1] # SHC at Fermi Level

# Plot of total SHC x Layer resolved. 

  fig = plt.figure(figsize=(8,5))
  plt.title(r'SHC : Total x Layers')

  plt.plot(shc_total[:,0],shc_total[:,1]/4      ,label='Total/4'    ,color='red'    ,linewidth=3)
  plt.plot(shc_total[:,0],shc_layer[0,:,1]    ,label='1st Layer',color='green'  ,linewidth=2)
  plt.plot(shc_total[:,0],shc_layer[1,:,1]    ,label='2nd Layer',color='blue'   ,linewidth=2)
  plt.plot(shc_total[:,0],shc_layer[2,:,1]    ,label='3rd Layer',color='gray'   ,linewidth=2)
  plt.plot(shc_total[:,0],shc_layer[3,:,1]    ,label='4th Layer',color='magenta',linewidth=2)
  plt.xlim(-1,1)
  plt.xlabel(r" E- E$_f$ (eV)")
  plt.ylabel(r'$\sigma^{z}_{xy}\;[\,(\hbar/2e)\,\Omega^{-1}\,\mathrm{m^{-1}}\,]$')
  plt.legend()
  plt.show()

# SHC: Total X Layers Sum. Here we are summing all layer contribution. No need to normalize for the layer number.

  shc_layer_sum = shc_layer[0] + shc_layer[1]  + shc_layer[2]  + shc_layer[3] 

  fig = plt.figure(figsize=(8,5))
  plt.title(r'SHC : Total x Layers sum')

  plt.plot(shc_total[:,0],shc_total[:,1]            ,label='Total'     ,color='red'    ,linewidth=3)
  plt.plot(shc_total[:,0],shc_layer_sum[:,1]    ,':',label='Layers Sum',color='green'  ,linewidth=3)


  plt.xlim(-1,1)
  plt.xlabel(r" E- E$_f$ (eV)")
  plt.ylabel(r'$\sigma^{z}_{xy}\;[\,(\hbar/2e)\,\Omega^{-1}\,\mathrm{m^{-1}}\,]$')

  plt.legend()
  plt.show()



  paoflow.finish_execution()

if __name__== '__main__':
  main()

