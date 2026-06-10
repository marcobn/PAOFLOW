import numpy as np
import matplotlib.pyplot as plt
import glob

colors = ['red', 'green', 'black', 'blue', 'yellow']

files_nz = glob.glob('results_freqvsangle*.out')
files_nz.sort()
print(files_nz)

def plot_shc(file, col):

  shc = np.loadtxt(file, delimiter=',', skiprows=1)
  y = shc[:,2]
  x = shc[:,1]

  plt.axvline(x=0, color='k', lw=1.00)
  plt.axhline(y=0, color='k', lw=1.00)

  plt.plot(x, y, color = col, linestyle='None', marker = 'o', markersize = 3.5)
  #plt.plot(x, y, color = col)

  plt.xlim(0.0, 90.0)
  plt.ylim(0, 2)

 # plt.yscale('log')

fig = plt.figure(figsize=(4, 7))

for i in range(len(files_nz)):

  plot_shc(files_nz[i], colors[i])


#plt.savefig('plot_frequencies.eps',format='eps',dpi=300)
plt.show()

