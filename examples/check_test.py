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


############# Verifies the output of PAOFLOW #############
## Usage:
##  "python check_test.py [test_directory_pattern] [output_directory] [reference_directory]"
##
## Default:
##  "python check_test.py example* output Reference"
##
##########################################################

def verifyData ( subdir, datPattern, refPattern ):

    ########## User Defined Variables ##########
    showFileResult = False  # Show PASS or FAIL for each file
    showErrors = False  # Flag to print out error values
    tolerance = 0.01  # Percentage that error can deviate from average to pass tests
    ######### End User Defined Variables ########

    print(('Verifying .dat files for %s' % subdir))

    # Get new data files and existing reference data files
    datFiles = glob.glob(datPattern+'/*.dat')
    refFiles = glob.glob(refPattern+'/*.dat')

    # Verify that .dat files exist in reference directory
    if len(refFiles) == 0:
        print('\tReference directory is empty or does not exist.')
        return

    # Sort the lists of files
    datFiles.sort()
    refFiles.sort()

    # Quick function to replace directory path
    rp = lambda f, p : [r.replace(p,'') for r in f]

    if len(datFiles) == 0:
      print('\tNo output files found')
      return

    # Ensure that the lists are identical
    if rp(datFiles, datFiles[0].split('/')[0]) != rp(refFiles, refFiles[0].split('/')[0]):
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
        relativeError = []

        # Compare computed error against data average
        validData = True
        for j in range(nCol-1):

            # Store maximum absolute error
            if absoluteError[j] > maxError:
                maxError = absoluteError[j]
                maxErrorIndex = i

            # Compute relative error
            relError = absoluteError[j]/dataRange
            relativeError.append(relError)

            # Store maximum relative error
            if relError > maxRelError:
                maxRelError = relError
                maxRelErrorIndex = i

            # Ensure that relative error is less than user defined tolerance
            if relError > tolerance:
                validData = False

        if np.isnan(absoluteError).any() or np.isnan(relativeError).any():
            validData = False

        if validData:
            result = 'PASS'
        else:
            allDataResult = result = 'FAIL'

        if showErrors:
            print(('\t%s:\n\t\tMean Absolute Errors: %s\n\t\tRelative Errors: %s' % (datFiles[i], absoluteError, relativeError)))
        if showFileResult:
            print(('\t%s ---------- [%s]\n' % (datFiles[i], result)))

    if showErrors:
        print(('The maximum absolute error in %s was %E in %s' % (subdir, maxError, datFiles[maxErrorIndex])))
        print(('The maximum relative error in %s was %E in %s' % (subdir, maxRelError, datFiles[maxRelErrorIndex])))

    print(('%s ---------- [%s]\n' % (subdir, allDataResult)))


def main():

    # Look for test directory pattern argument
    if len(sys.argv) > 1:
        alldir = sorted(glob.glob(sys.argv[1]))
    else:
        alldir = sorted(glob.glob('example*'))

    # Assign default reference directory pattern, then look for argument
    datPattern = 'output'
    if len(sys.argv) > 2:
        datPattern = sys.argv[2]

    # Assign default reference directory pattern, then look for argument
    refPattern = 'Reference'
    if len(sys.argv) > 3:
        refPattern = sys.argv[3]

    # Verify data for each test matching the input or default pattern
    for n in range(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]
        verifyData(subdir, datPattern, refPattern)
        os.chdir('../')


if __name__ == "__main__":
    main()
