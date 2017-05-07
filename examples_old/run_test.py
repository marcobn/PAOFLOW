#
# PAOpy
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

def get_exeCmd(engine, calcType,inputFile):

#################### User defined parameters ####################
    execPrefix = "mpirun -np 32"
    execPostfix = " "
    QE_path = "/home/marco/Programs/qe-6.0/bin/"
    PAO_path = "python /home/marco/Programs/PAOpy/src/"

    if engine=='qe':
        execDict={'scf':'pw.x -npool 8 ','nscf':'pw.x -npool 8 ','proj':'projwfc.x -npool 8 '}
        exeDir = QE_path
################ end of user defined parameters #################
################ DO NOT MODIFY BELOW THIS POINT #################

    if engine=='PAO':
        execDict={'PAO':'PAOpy.py'}
        exeDir = PAO_path

    executable = execDict[calcType]
    outputFile = inputFile.split('.')[0] + '.out'


    if engine=='qe':
        command  = '%s %s < %s %s >  %s' % ( execPrefix, os.path.join(exeDir, executable),inputFile, execPostfix, outputFile )
    else:
        command  = '%s %s %s >  %s' % (execPrefix, os.path.join(exeDir, executable),execPostfix, outputFile )

    return command

def oneRun(subdir):

    calcList = ['scf','nscf','proj','PAO']
    fileList = ['scf.in','nscf.in','proj.in','inputfile.py']

    engine = {'scf':'qe',
              'nscf':'qe',
              'proj':'qe',
              'PAO':'PAO',}

    n = 0
    for calc in calcList:

        command = get_exeCmd(engine[calc.split("_")[0]],calc.split("_")[0],fileList[n])
        n += 1
        try:
            print "%s in %s"%(command, subdir)
            subprocess.check_output([command],shell=True)
        except subprocess.CalledProcessError as e:
            print "######### SEQUENCE ######### \n FAILED %s in %s\n %s\n"%(command, subdir,e)
            raise SystemExit

    return

def main():

    start = reset = time.time()
    alldir = glob.glob('example*')
    for n in xrange(len(alldir)):
        os.chdir(alldir[n])
        subdir = str(os.getcwd()).split('/')[len(str(os.getcwd()).split('/'))-1]
        oneRun(subdir)
        os.chdir('../')
        print('test run in %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
        reset=time.time()

    print('all test runs in %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))

if __name__ == "__main__":
    main()
