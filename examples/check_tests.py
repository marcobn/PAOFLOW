#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import os, sys
import glob
import numpy as np
        
def verifyData ( subdir ):

    ########## User Defined Variables ##########
    showFileResult = True  # Show PASS or FAIL for each file
    showAbsoluteErrors = True  # Flag to print out error values
    tolerance = 0.01  # Percentage that error can deviate from average to pass tests
    ######### End User Defined Variables ########

    # Get new data files and existing reference data files
    datFiles = glob.glob('*.dat')
    refFiles = glob.glob('./Reference/*.dat')

    # Sort the lists of files
    datFiles.sort()
    refFiles.sort()

    # Ensure that the lists are identical
    if datFiles != [r.replace('./Reference/', '') for r in refFiles]:
        print('\tList of calculated .dat files does not match reference files.')
        return False

    # Compare data files
    maxError = -1.  # Will store maximum error value
    maxErrorIndex = -1  # Will store file index of maximum error value
    allDataResult = 'PASS'  # Stores status of the entire example calculation
    for i in range(len(datFiles)):
        
        # Gather data from files
        df = open(datFiles[i], 'r')
        rf = open(refFiles[i], 'r')
        dl = np.array([[np.float(s) for s in l.split()] for l in df.readlines()]).transpose()
        rl = np.array([[np.float(s) for s in l.split()] for l in rf.readlines()]).transpose()
        df.close()
        rf.close()

        nCol = len(dl)
        nRow = len(dl[0])

        # Compute absolute error and averages excluding the first column
        absoluteError = np.sum(np.abs(np.abs(dl[1:nCol, :]) - np.abs(rl[1:nCol, :])), axis=1) / nRow
        average = np.sum(dl[1:nCol, :], axis=1) / nRow
        errorPercentage = []

        # Compare computed error against data average
        validData = True
        for j in xrange(nCol-1):

            if absoluteError[j] > maxError:
                maxError = absoluteError[j]
                maxErrorIndex = i

            if average[j] == 0.:
                errorPercentage.append(0.)
            else:
                perc = np.abs(absoluteError[j]/average[j])
                errorPercentage.append(perc)
                if perc > tolerance:
                    validData = False

        if validData:
            result = 'PASS'
        else:
            allDataResult = result = 'FAIL'

        if showAbsoluteErrors:
            print('\t%s:\n\t\tMean Absolute Errors: %s\n\t\tAverage Values: %s\n\t\tError Percentages of Averages: %s\n' % (datFiles[i], absoluteError, average, errorPercentage))
        if showFileResult:
            print('\t%s ---------- [%s]\n' % (datFiles[i], result))

    if showAbsoluteErrors:
        print('The maximum absolute error in %s was %E in %s\n' % (subdir, maxError, datFiles[maxErrorIndex]))
    print('%s ---------- [%s]\n' % (subdir, allDataResult))


def main():
 
    alldir = glob.glob('example*')
    for n in xrange(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]

        print('Verifying .dat files for %s' % subdir)
        verifyData(subdir)

        os.chdir('../')


if __name__ == "__main__":
    main()
