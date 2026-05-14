import numpy as np
import matplotlib.pyplot as plt
import glob

colors = ['blue', 'green', 'yellow', 'red', 'black']

files_nz = glob.glob('results_freqvsangle.out')
files_nz.sort()

def plot_freq(file, col):

  freq = np.loadtxt(file, delimiter=',', skiprows=1)
  y = freq[:,2]
  x = freq[:,1]

  plt.axvline(x=0, color='k', lw=1.00)
  plt.axhline(y=0, color='k', lw=1.00)

  plt.plot(x, y, color = col, linestyle='None', marker = 'o')

  plt.xlim(0.0, 90.0)
  plt.ylim(0, 30)

fig = plt.figure()

for i in range(len(files_nz)):

  plot_freq(files_nz[i], colors[i])


plt.show()

#plt.savefig('plot-shc.out.eps',format='eps',dpi=300)
