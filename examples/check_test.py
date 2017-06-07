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
    showFileResult = False  # Show PASS or FAIL for each file
    showErrors = False  # Flag to print out error values
    tolerance = 0.005  # Percentage that error can deviate from average to pass tests
    ######### End User Defined Variables ########

    print('Verifying .dat files for %s' % subdir)

    # Get new data files and existing reference data files
    datFiles = glob.glob('*.dat')
    refFiles = glob.glob('./Reference/*.dat')

    # Sort the lists of files
    datFiles.sort()
    refFiles.sort()

    # Ensure that the lists are identical
    if datFiles != [r.replace('./Reference/', '') for r in refFiles]:
        print('\tList of calculated .dat files does not match reference files.')
        return

    # Compare data files
    maxError = -1.  # Will store maximum error value
    maxErrorIndex = -1  # Will store file index of maximum error value
    maxRelError = -1.  # Store maximum relative error
    maxRelErrorIndex = -1  # Store file index of maximum relative error value
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

        # Compute absolute error and data range excluding the first column
        absoluteError = np.sum(abs(abs(dl[1:nCol, :]) - abs(rl[1:nCol, :])), axis=1) / nRow
        dataRange = np.amax(np.amax(dl[1:nCol, :], axis=1), axis=0) - np.amin(np.amin(dl[1:nCol, :], axis=1), axis=0)
        relativeErrors = []

        # Compare computed error against data average
        validData = True
        for j in xrange(nCol-1):

            # Store maximum absolute error
            if absoluteError[j] > maxError:
                maxError = absoluteError[j]
                maxErrorIndex = i

            # Compute relative error
            relError = absoluteError[j]/dataRange
            relativeErrors.append(relError)

            # Store maximum relative error
            if relError > maxRelError:
                maxRelError = relError
                maxRelErrorIndex = i

            # Ensure that relative error is less than user defined tolerance
            if relError > tolerance:
                validData = False

        if validData:
            result = 'PASS'
        else:
            allDataResult = result = 'FAIL'

        if showErrors:
            print('\t%s:\n\t\tMean Absolute Errors: %s\n\t\tRelative Errors: %s' % (datFiles[i], absoluteError, relativeErrors))
        if showFileResult:
            print('\t%s ---------- [%s]\n' % (datFiles[i], result))

    if showErrors:
        print('The maximum absolute error in %s was %E in %s' % (subdir, maxError, datFiles[maxErrorIndex]))
        print('The maximum relative error in %s was %E in %s' % (subdir, maxRelError, datFiles[maxRelErrorIndex]))

    print('%s ---------- [%s]\n' % (subdir, allDataResult))


def main():
    alldir = glob.glob('example*')
    for n in xrange(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]
        verifyData(subdir)
        os.chdir('../')


if __name__ == "__main__":
    main()
