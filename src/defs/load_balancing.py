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

import numpy as np

def load_balancing(size,rank,n):

    # Load balancing
    ini = np.zeros((size),dtype=int)
    end = np.zeros((size),dtype=int)
    splitsize = 1.0/size*n
    for i in xrange(size):
        ini[i] = int(round(i*splitsize))
        end[i] = int(round((i+1)*splitsize))
    start = ini[rank]
    stop = end[rank]

    return(start,stop)
