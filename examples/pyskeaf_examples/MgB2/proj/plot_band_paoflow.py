from matplotlib import pyplot as plt
from itertools import islice
import numpy as np
'''THIS SCRIPT TAKES bands_0.dat FILE FROM PAOFLOW AND PLOTS THE BAND SCRIPT'''

##with soc ta
bands = np.loadtxt('./output_paoflow/bands_0.dat')
fig = plt.figure()
for i in range(0, len(bands[0,:])):
  plt.plot(bands[:,0], bands[:,i], color='black', linestyle='-', linewidth=0.9)

plt.axhline(y=0., color='b', linestyle = '--', lw=.50)
# plt.axvline(x=71, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=188, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=259, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=376, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=439, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=510, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=627, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=698, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=815, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=878, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=941, color='k', linestyle = '--', lw=.50)
# plt.axvline(x=1004, color='k', linestyle = '--', lw=.50)


x = [0, 259, 409, 708, 904, 1163, 1313, 1612, 1612, 1808, 1808, 2004]


labels = ['\u0393', 'M', 'K', '\u0393', 'A', 'L', 'H', 'A', 'L', 'M', 'K', 'H']

plt.xticks(x, labels)
plt.ylabel('Energy (eV)', size=20)
plt.ylim(-4.0, 4.0)
plt.xlim(0,len(bands[:,0]))

plt.savefig('./plot_bands_paoflow.png',dpi=600)
# plt.show()
