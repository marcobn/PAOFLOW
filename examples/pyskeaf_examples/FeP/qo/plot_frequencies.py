import numpy as np
import matplotlib.pyplot as plt
import glob

colors = ['red','orange', 'green', 'black', 'blue', 'yellow', 'purple', 'gray']

files_nz = glob.glob('results_freqvsangle_*.out')
files_nz.sort()
print(files_nz)

def plot_shc(file, col):
    shc = np.loadtxt(file, delimiter=',', comments='Theta(deg)', skiprows=0)
    x = shc[:, 1]   # phi
    y = shc[:, 2]

    plt.axvline(x=0, color='k', linestyle = '--', lw=.7)
    plt.axhline(y=0, color='k', linestyle = '--', lw=.7)

    plt.plot(x, y, color = col, linestyle='None', marker = 'o', markersize = 3.5)

    plt.xlabel("$\phi (^\circ)$",fontsize=20)
    plt.ylabel("$\omega (kT)$",fontsize=20)
    plt.xlim(0.0, 90.0)
    # plt.ylim(0, 5)

 # plt.yscale('log')

# fig = plt.figure(figsize=(4, 7))
fig = plt.figure()

for i in range(len(files_nz)):
    plot_shc(files_nz[i], colors[i])
    print(' ')
    print(files_nz[i], 'CORRECT ! !')
    print(' ')

plt.savefig('plot_frequencies.png',format='png',dpi=600)
plt.show()
