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
##  "python check_test.py [test_directory_pattern] [reference_directory_pattern]"
##
## Default:
##  "python check_test.py example* ./Reference/"
##
##########################################################

def verifyData ( subdir, refPattern ):

    ########## User Defined Variables ##########
    showFileResult = False  # Show PASS or FAIL for each file
    showErrors = False  # Flag to print out error values
    tolerance = 0.005  # Percentage that error can deviate from average to pass tests
    ######### End User Defined Variables ########

    RED   = "\x1B[31m"
    GREEN = "\x1B[32m"
    RESET = "\x1B[0m"
    test_set_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(subdir))))


    # Get new data files and existing reference data files
    datFiles  = glob.glob('*.dat')
#    datFiles += glob.glob('*.bxsf')
    refFiles  = glob.glob(refPattern+'*.dat')
#    refFiles += glob.glob(refPattern+'*.bxsf')


    # Verify that .dat files exist in reference directory
    if len(refFiles) == 0:
        print('\tReference directory is empty or does not exist.')
        return

    # Sort the lists of files
    datFiles.sort()
    refFiles.sort()

    # Ensure that the lists are identical
    if datFiles != [r.replace(refPattern, '') for r in refFiles]:
        print('\tList of calculated .dat files does not match reference files.')
        return

    # Compare data files
    maxError = -1.  # Will store maximum error value
    maxErrorIndex = -1  # Will store file index of maximum error value
    maxRelError = -1.  # Store maximum relative error
    maxRelErrorIndex = -1  # Store file index of maximum relative error value
    allDataResult = GREEN+'PASS'+RESET  # Stores status of the entire example calculation
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
            result = GREEN+'PASS'+RESET
        else:
            allDataResult = result = RED+'FAIL'+RESET

        if showErrors:
            print('\t%s:\n\t\tMean Absolute Errors: %s\n\t\tRelative Errors: %s' % (datFiles[i], absoluteError, relativeError))
        showFileResult_tmp = showFileResult
        if result == RED+'FAIL'+RESET:
           showFileResult_tmp = True 
        if showFileResult_tmp:
            print('\t[%s] ---------- %s' % (result,datFiles[i]))
        showFileResult_tmp = showFileResult

    if showErrors:
        print('The maximum absolute error in %s was %E in %s' % (test_set_dir+'/'+subdir, maxError, datFiles[maxErrorIndex]))
        print('The maximum relative error in %s was %E in %s' % (test_set_dir+'/'+subdir, maxRelError, datFiles[maxRelErrorIndex]))

    print('[%s] ---------- %s' % ( allDataResult,test_set_dir+'/'+subdir))


def main():

    # Look for test directory pattern argument
    if len(sys.argv) > 1:
        alldir = glob.glob(sys.argv[1])
    else:
        alldir = sorted(glob.glob('./*/*example*/'))

    # Assign default reference directory pattern, then look for argument
    refPattern = './Reference/'
    if len(sys.argv) > 2:
        refPattern = sys.argv[2]
        if refPattern[0] != '.' and refPattern[0] != '/' and refPattern != '~':
            refPattern = './'+refPattern
        if refPattern[len(refPattern)-1] != '/':
            refPattern += '/'

    # Verify data for each test matching the input or default pattern
    for n in range(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]
        verifyData(subdir, refPattern)
        os.chdir('../../')


if __name__ == "__main__":
    main()
