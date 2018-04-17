# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2018      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2018 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
#
#  This file is part of AFLOW software.
#
#  AFLOW is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************************

# future imports


import sys
import numpy as np
from PAOFLOW import *

def main():

    arg1 = './'
    arg2 = "inputfile.xml"
    try:
        arg1 = os.path.abspath(sys.argv[1])
        if os.path.isfile(arg1):
            arg2 = os.path.basename(arg1)
            arg1 = os.path.dirname(arg1)
    except: pass




    # PAOFLOW may be called with one argument specifying the directory containing 'inputfile.xml'.
    outDict = paoflow(inputpath=arg1,inputfile=arg2)

    # Check for output dictionary and print the keys.
    if outDict is not None:
        print('\noutDict keys:')
        for k in list(outDict.keys()):
            print('\''+k+'\'')

if __name__== "__main__":
    main()

