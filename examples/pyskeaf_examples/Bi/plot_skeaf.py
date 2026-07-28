import glob
import os

import matplotlib.pyplot as plt
import numpy as np

colors = ['blue', 'green', 'yellow', 'red', 'black', 'magenta', 'orange', 'purple', 'brown']

files_nz = glob.glob('./output_pyskeaf/results_freqvsangle_*.out')
files_nz.sort()
print(files_nz)


def plot_freq(file, col):
    if os.path.getsize(file) == 0:
        print(f'{file}: skipped empty file')
        return False
    with open(file, encoding='utf-8') as handle:
        next(handle, None)
        if not any(line.strip() for line in handle):
            print(f'{file}: skipped file with no data rows')
            return False

    freq = np.loadtxt(file, delimiter=',', skiprows=1)
    freq = np.atleast_2d(freq)
    if freq.shape[1] < 3:
        print(f'{file}: skipped malformed file with {freq.shape[1]} columns')
        return False

    y = freq[:, 2]
    x = freq[:, 1]

    plt.axvline(x=0, color='k', lw=1.00)
    plt.axhline(y=0, color='k', lw=1.00)

    plt.plot(x, y, color=col, linestyle='None', marker='o', markersize=3.5)
    # plt.plot(x, y, color = col)

    plt.xlabel('angle ($\u00b0$)', fontsize=20)
    plt.ylabel(r'$\rm{B_F}$ ($10^3$ T)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.xlim(0.0, 90)
    # plt.ylim(0, 30)
    # plt.yscale('log')
    return True


fig = plt.figure()

for i, file in enumerate(files_nz):
    if i >= len(colors):
        print(f'{file}: skipped, no color configured')
        continue
    if not plot_freq(file, colors[i]):
        continue
    print(' ')
    print(file, 'CORRECT ! !')
    print(' ')

plt.tight_layout()
plt.savefig('./output_pyskeaf/plot_frequencies.png', dpi=300)
# plt.show()
