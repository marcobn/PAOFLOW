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

def get_exeCmd(engine, calcType,inputFile):

#################### User defined parameters ####################
    execPrefix = "mpirun -np 64"
    execPostfix = " "
    QE_path = "/home/marco/Programs/qe-6.0/bin/"



    if engine=='qe':
        execDict={'scf':'pw.x -npool 8 ','nscf':'pw.x -npool 8 ','proj':'projwfc.x -npool 8 -northo 1 '}
        exeDir = QE_path
################ end of user defined parameters #################
################ DO NOT MODIFY BELOW THIS POINT #################


    PAO_path = "python ../../../src/"

    if engine=='PAO':
        execDict={'PAO':'main.py ./'}
        exeDir = PAO_path

    executable = execDict[calcType]
    outputFile = inputFile.split('.')[0] + '.out'


    if engine=='qe':
        command  = '%s %s < %s %s >  %s' % ( execPrefix, os.path.join(exeDir, executable),inputFile, execPostfix, outputFile )
    else:
        command  = '%s %s %s >  %s' % (execPrefix, os.path.join(exeDir, executable),execPostfix, outputFile )

    return command

def run_pw(subdir):

    if len(glob.glob('*.save')) == 0:
        calcList = ['scf','nscf','proj']
        fileList = ['scf.in','nscf.in','proj.in']
    else: 
        calcList = []
        fileList = []

    engine = {'scf':'qe',
              'nscf':'qe',
              'proj':'qe',
              'PAO':'PAO',}

    n = 0
    for calc in calcList:

        command = get_exeCmd(engine[calc.split("_")[0]],calc.split("_")[0],fileList[n])
        n += 1
        try:
            print("%s in %s"%(command, subdir))
            subprocess.check_output([command],shell=True)
        except subprocess.CalledProcessError as e:
            print("######### SEQUENCE ######### \n FAILED %s in %s\n %s\n"%(command, subdir,e))
            raise SystemExit
    return


def run_pao(subdir):


    calcList = ['PAO']
    fileList = ['inputfile.xml']

    engine = {'scf':'qe',
              'nscf':'qe',
              'proj':'qe',
              'PAO':'PAO',}

    n = 0
    for calc in calcList:

        command = get_exeCmd(engine[calc.split("_")[0]],calc.split("_")[0],fileList[n])
        n += 1
        try:

            subprocess.check_output([command],shell=True)
        except subprocess.CalledProcessError as e:
            print("######### SEQUENCE ######### \n FAILED %s in %s\n %s\n"%(command, subdir,e))
            raise SystemExit
    return


def main():
    
    start = reset = time.time()

    pwdir  =  sorted(glob.glob('./pw_data/*/'))

    alldir = sorted(glob.glob('./*/example*/'))

    refPattern = './Reference/'
    if len(sys.argv) > 2:
        refPattern = sys.argv[2]
        if refPattern[0] != '.' and refPattern[0] != '/' and refPattern[0] != '~':
            refPattern = './'+refPattern
        if refPattern[len(refPattern)-1] != '/':
            refPattern += '/'

    for n in range(len(pwdir)):
        os.chdir(pwdir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]

        try:
            run_pw(subdir)
        except:
            print(('Exception in %s'%subdir))
            quit()
        os.chdir('../../')

    for n in range(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]

        try:
            run_pao(subdir)
        except:
            print(('Exception in %s'%subdir))
            quit()
        verifyData(subdir, refPattern)
        os.chdir('../../')

        reset=time.time()

    print(('all test runs in %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10)))

if __name__ == "__main__":
    main()
