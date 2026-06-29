import numpy as np
import matplotlib.pyplot as plt
import glob

#colors = ['red','orange', 'green', 'black', 'blue', 'yellow']
colors = ['red', 'blue', 'green']

files_nz = glob.glob('results_freqvsangle_*.out')
files_nz.sort()
print(files_nz)

def plot_freq(file, col):    
    freq = np.loadtxt(file, delimiter=',', comments='Theta(deg)', skiprows=0)
    # x = freq[:, 0]   # theta (deg)
    x = freq[:, 1]-90.0000000   # phi (deg)
    y = freq[:, 2]*1000   # frequencies (Tesla)

    # plt.axhline(y=min(freq[:, 2]), color='b', linestyle = '--', lw=.7)
    # plt.text(min(freq[:, 1])+20, min(freq[:, 2]), str(min(freq[:, 2])), color='b')
    
    # plt.axhline(y=max(freq[:, 2]), color='r', linestyle = '--', lw=.7)
    # plt.text(min(freq[:, 1])+20, max(freq[:, 2]), str(max(freq[:, 2])), color='r')
    
    plt.text(min(x)+15, max(y)*(3.8/4.), "$\phi = 0^\circ$, $0 \leq \u03B8 \leq 90^\circ$", color='k',fontsize=25)
    
    plt.plot(x, y, color = col, linestyle='None', marker = 'o', markersize = 3.5)
    
    plt.xlabel("$\u03B8 (^\circ)$",fontsize=25)
    # plt.xlabel("$\phi \; (^\circ)$",fontsize=25)
    plt.ylabel("$B_F \; (T)$",fontsize=25)
    plt.xlim(-1, 91)
    plt.xticks(np.arange(0, 91, 10))
    # plt.ylim(0, 5)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

# fig = plt.figure(figsize=(4, 7))
fig = plt.figure(figsize=(6, 6))

for i in range(len(files_nz)):
    plot_freq(files_nz[i], colors[i])
    print(' ')
    print(files_nz[i], 'CORRECT ! !')
    print(' ')

plt.savefig('plot_frequencies.png',format='png',dpi=600)
# plt.show()

