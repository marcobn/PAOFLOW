import numpy as np
import matplotlib.pyplot as plt
import glob
import re

colors = ['blue', 'green', 'yellow', 'red', 'black', 'orange', 'purple', 'brown']

files_nz = glob.glob('./output_pyskeaf/results_freqvsangle_*.out')
files_nz.sort()
print(files_nz)

def plot_freq(file, col):

  freq = np.loadtxt(file, delimiter=',', skiprows=1)
  y = freq[:,2]
  x = freq[:,1]

  plt.axvline(x=0, color='k', lw=1.00)
  plt.axhline(y=0, color='k', lw=1.00)

  plt.plot(x, y, color = col, linestyle='None', marker = 'o', markersize = 3.5)
  #plt.plot(x, y, color = col)
  
  plt.xlabel("angle ($\u00B0$)",fontsize=20)
  plt.ylabel(r"$\rm{B_F}$ ($10^3$ T)",fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  # plt.xlim(0.0, 90)
  plt.ylim(0, 10)
  # plt.yscale('log')

fig = plt.figure()

for i in range(len(files_nz)):
  plot_freq(files_nz[i], colors[i])
  print(' ')
  print(files_nz[i], 'CORRECT ! !')
  print(' ')

plt.tight_layout()
plt.savefig('plot_frequencies.png',dpi=300)
# plt.show()

