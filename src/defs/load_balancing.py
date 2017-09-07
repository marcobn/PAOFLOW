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
    splitsize = 1.0/size*n
    start = int(round(rank*splitsize))
    stop = int(round((rank+1)*splitsize))
    return(start,stop)

# For each processor calculate 3 values:
# 0 - Total number of items to be scattered/gathered on this processor
# 1 - Index in complete array where the subarray begins
# 2 - Dimension of the subarray on this processor
def load_sizes(size,n,dim):
    sizes = np.empty((size,3),dtype=int)
    splitsize = 1.0/size*n
    for i in xrange(size):
        start = int(round(i*splitsize))
        stop = int(round((i+1)*splitsize))
        sizes[i][0] = dim*(stop-start)
        sizes[i][1] = dim*start
        sizes[i][2] = stop-start
    return sizes
