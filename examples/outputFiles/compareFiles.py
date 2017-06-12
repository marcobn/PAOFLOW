import numpy as np
import glob


# This tool calculates the speedup of cpu vs gpu fft evaluation.
# Place this file in a directory with all of the inputfile.out
#   files you want to test the speedup of.
# The files must be named such that 'cpu' and 'gpu' are
#   the first distinct characters in the filenames.
# E.g.  'cpu_1.out', 'cpu_2.out'...'gpu_1.out', 'gpu_2.out'
def main():

    fileNames = glob.glob('*.out')
    fileNames.sort()

    timeDiffs = []
    gradTimes = []
    totalTimes = []
 
    for fn in fileNames:
        f = open(fn, 'r')
        for line in f:
            strings = line.split()
            if len(strings) > 0:
                if strings[0] == 'gradient':
                    gradTimes.append(np.float(strings[len(strings)-2]))
#                    print('\tGradient Time: '+strings[len(strings)-2])
                if  strings[0] == 'Total':
                    totalTimes.append(np.float(strings[len(strings)-2]))
#                    print('\tTotal Time: '+strings[len(strings)-2])
        f.close()

    cpu_grad_times = np.array(gradTimes[:len(gradTimes)/2])
    gpu_grad_times = np.array(gradTimes[len(gradTimes)/2:])

    print("Speedup: %s" % (cpu_grad_times/gpu_grad_times))

if __name__ == "__main__":
    main()
