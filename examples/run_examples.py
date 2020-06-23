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

import os,sys,time
import re,glob
import numpy
import subprocess
from check_test import verifyData

def get_exeCmd ( engine, calcType, inputFile ):

#################### User defined parameters ####################
    prefix_QE = '/Users/marco/Local/Programs/openmpi/bin/mpirun -np 16'
    prefix_Python = '/Users/marco/Local/Programs/anaconda3/bin/mpirun -np 8'

    path_QE = '/Users/marco/Local/Programs/qe-6.2.1/bin/'
    path_Python = '/Users/marco/Local/Programs/anaconda3/bin/'

    if engine=='qe':
        execDict={'scf':'pw.x -npool 2','nscf':'pw.x -npool 2','proj':'projwfc.x -npool 2'}
        exeDir = path_QE
################ end of user defined parameters #################
################ DO NOT MODIFY BELOW THIS POINT #################

    if engine=='PAO':
        execDict={'PAO':'main.py'}
        exeDir = path_Python

    executable = execDict[calcType]
    outputFile = inputFile.split('.')[0] + '.out'


    if engine == 'qe':
        command  = '%s %s < %s > %s'%(prefix_QE, os.path.join(exeDir,executable), inputFile, outputFile)
    elif engine == 'PAO':
        command  = '%s %s %s > %s'%(prefix_Python, os.path.join(exeDir,'python'), executable, outputFile)
    else:
      raise ValueError('No engine type: %s'%engine)

    return command

def oneRun(subdir):

    calcList = []
    fileList = []
    if len(glob.glob('*.save')) == 0:
        calcList = ['scf','nscf','proj']
        fileList = ['scf.in','nscf.in','proj.in']
        if 'example01' in subdir:
          calcList = 2*calcList
          fileList += ['scf_nosym.in', 'nscf_nosym.in', 'proj_nosym.in']
    calcList += ['PAO']
    fileList += ['inputfile.xml']

    engine = {'scf':'qe',
              'nscf':'qe',
              'proj':'qe',
              'PAO':'PAO',}

    n = 0
    for calc in calcList:

        command = get_exeCmd(engine[calc.split('_')[0]],calc.split('_')[0],fileList[n])
        n += 1
        try:
            print('%s in %s'%(command, subdir))
            subprocess.check_output([command],shell=True)
        except subprocess.CalledProcessError as e:
            print('######### SEQUENCE ######### \n FAILED %s in %s\n %s\n'%(command, subdir,e))
            raise SystemExit
    return

def main():
    
    start = reset = time.time()
    if len(sys.argv) > 1:
        alldir = glob.glob(sys.argv[1])
    else:
        alldir = sorted(glob.glob('example*'))

    datPattern = 'output'
    if len(sys.argv) > 2:
        datPattern = sys.argv[2]

    refPattern = 'Reference'
    if len(sys.argv) > 3:
        refPattern = sys.argv[3]

    for n in range(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]
        try:
            oneRun(subdir)
        except Exception as e:
            print(('Exception in %s'%subdir))
            raise e
        try:
            verifyData(subdir, datPattern, refPattern)
        except Exception as e:
            print(('Exception in %s'%subdir))
            raise e
        os.chdir('../')
        print(('Test run in %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10)))
        reset=time.time()

    print(('All test runs in %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10)))

if __name__ == '__main__':
    main()
