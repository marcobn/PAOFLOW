#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

import numpy as np
import sys, time
from scipy import fftpack as FFT
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import *
from get_R_grid_fft import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def Scatterv_wrap_size(array):

    if rank==0:
        #get the indices of the flattened array that will be split and sent to the procs
        split_size=np.zeros(size,dtype=np.int64,order='C')
        for loop_rank in xrange(size):
            split_size[loop_rank] = np.abs(np.diff(np.array(load_balancing(size,loop_rank,array.shape[0]))))[0]

        #get number of them for load balancing
        num_entries = array.shape[0]
        #starting indice in flattened array to go to each proc
        start_ele = np.zeros(size,dtype=np.int64,order='C')
        #number of elements for each proc
        ele       = np.zeros(size,dtype=np.int64,order='C')                                                   
        for loop_rank in xrange(size):
            #the number of elements for the H(k)
            ele[loop_rank] = split_size[loop_rank]*np.prod(array.shape[1:])
            if loop_rank==0:
                #starts at zero if rank 0
                start_ele[loop_rank]=0
            else:
                #starts on previous starting point+elements of previous starting point
                start_ele[loop_rank] = start_ele[loop_rank-1]+ele[loop_rank-1]    
 
    if rank!=0:        
        #empty arrays for bcast
        ele=np.zeros(size,dtype=np.int64)
        start_ele=np.zeros(size,dtype=np.int64)
        split_size=np.zeros(size,dtype=np.int64)

    #broadcast arrays for displacements and num elements for Scatterv
    comm.Bcast(ele)
    comm.Bcast(start_ele)
    comm.Bcast(split_size)
        
    
    #broadcast size of split array
    if rank==0:
        index={"array_shape":list(array.shape)}
    else:
        index=None
    index=comm.bcast(index)
    array_shape=index["array_shape"]

    #tuples for Scatterv
    start_ele_tup = tuple(start_ele.tolist())
    ele_tup       = tuple(ele.tolist())

    #put num elements in split array for each proc
    array_shape[0] = ele[rank]/np.prod(array_shape[1:])
    array_shape = tuple(array_shape)

    return ele_tup,start_ele_tup,array_shape


def Scatterv_wrap(array):
    if rank==0:
        #make sure array is c contiguous
        if not array.flags['C_CONTIGUOUS']:
            array=np.ascontiguousarray(array)

    #for large arrays 
    if rank == 0:
        #size of flattened array in units units of int max
        nchunks = (np.prod(array.shape)/2147483647)+1
        for chunk in xrange(nchunks):
            chunk_start,chunk_end = load_balancing(nchunks,chunk,np.prod(array.shape[1:]))

    #get info for Scatterv
    ele_tup,start_ele_tup,split_array_shape = Scatterv_wrap_size(array)        

    index = None
    if rank == 0:
        array_shape = np.asarray(array.shape,dtype=np.int64,order='C')
        mpi_dtype = get_MPI_dtype(array)
        index={'mpi_dtype':mpi_dtype,'array_shape':array_shape,'np_dtype':array.dtype.str,
               'ele_tup':ele_tup,'start_ele_tup':start_ele_tup,'split_array_shape':split_array_shape}

    #broadcast info about input array to all proc
    index = comm.bcast(index,root=0)
    array_shape       = index['array_shape']
    np_dtype          = index['np_dtype']
    mpi_dtype         = index['mpi_dtype']
    
    #recieving array
    array_aux = np.zeros(np.prod(split_array_shape),dtype=np_dtype,order='C')
    comm.Barrier()
    if mpi_dtype=='C_DOUBLE_COMPLEX':
        #scatter the array
        if rank==0:
            #specify as C double precision complex 
            comm.Scatterv([np.ravel(array,order='C'),ele_tup,start_ele_tup,MPI.C_DOUBLE_COMPLEX],
                           [array_aux,MPI.C_DOUBLE_COMPLEX])                 
        else: comm.Scatterv(None,[array_aux,MPI.C_DOUBLE_COMPLEX])
    if mpi_dtype=='C_DOUBLE_PRECISION':
        #scatter the Hk
        if rank==0:
            #specify as C double precision complex 
            comm.Scatterv([np.ravel(array,order='C'),ele_tup,start_ele_tup,MPI.DOUBLE_PRECISION],
                           [array_aux,MPI.DOUBLE_PRECISION])                 
        else: comm.Scatterv(None,[array_aux,MPI.DOUBLE_PRECISION])            
    comm.Barrier()

    #return as reshaped and contiguous
    return np.ascontiguousarray(np.reshape(array_aux,split_array_shape,order='C'))
    
def Gatherv_wrap_size(array):

    one_size = np.zeros(size,dtype=np.int64)

    #num elements in each procs array
    one_size[rank]=np.prod(array.shape)
    ele = np.zeros(size,dtype=np.int64)
    comm.Allreduce(one_size,ele)

    #get offsets in gathered array
    start_ele = np.cumsum(ele)-ele

    #get shape of gathered array
    first_ind = np.int64(np.sum(ele)/np.prod(array.shape[1:]))
    array_shape = list(array.shape)
    array_shape[0] = first_ind

    #return info to all proc
    return tuple(ele.tolist()),tuple(start_ele.tolist()),tuple(array_shape)

    
def Gatherv_wrap(array):
    #make sure arrays are c contiguous
    if not array.flags['C_CONTIGUOUS']:
        array=np.ascontiguousarray(array)

    #size of flattened array in units units of int max
    chunks = (np.prod(array.shape)/2147483647)+1

    #get info for Gatherv
    ele_tup,start_ele_tup,split_array_shape = Gatherv_wrap_size(array)
    
    np_dtype          = array.dtype.str
    mpi_dtype         = get_MPI_dtype(array)


    comm.Barrier()
    if mpi_dtype=='C_DOUBLE_COMPLEX':
        #scatter the Hk
        if rank==0:
            array_aux = np.zeros(np.prod(split_array_shape),dtype=np_dtype,order='C')

            comm.Gatherv([np.ravel(array,order='C'),MPI.C_DOUBLE_COMPLEX],
                         [array_aux,ele_tup,start_ele_tup,MPI.C_DOUBLE_COMPLEX])
        else:
            comm.Gatherv([np.ravel(array,order='C'),MPI.C_DOUBLE_COMPLEX],None)
        #wait for the processes to send to root before moving on

    if mpi_dtype=='C_DOUBLE_PRECISION':
        #scatter the Hk
        if rank==0:
            array_aux = np.zeros(np.prod(split_array_shape),dtype=np_dtype,order='C')

            comm.Gatherv([np.ravel(array,order='C'),MPI.DOUBLE_PRECISION],
                         [array_aux,ele_tup,start_ele_tup,MPI.DOUBLE_PRECISION])
        else:
            comm.Gatherv([np.ravel(array,order='C'),MPI.DOUBLE_PRECISION],None)
        #wait for the processes to send to root before moving on
    comm.Barrier()

        
    if rank==0:
        #return reshaped, contiguous gathered array to root proc
        return np.ascontiguousarray(np.reshape(array_aux,split_array_shape,order="C"))
    

    



def get_MPI_dtype(array):

    np_dtype = array.dtype.num
    c_contig=False
    f_contig=False
    if array.flags['C_CONTIGUOUS']:
        c_contig=True
    elif array.flags['F_CONTIGUOUS']:
        f_contig=True
    else:
        print 'array is not contiguous. exiting'
        raise SystemExit

    if np_dtype == 15:
        if c_contig:
            return 'C_DOUBLE_COMPLEX'
        if f_contig:
            return 'DOUBLE_COMPLEX'

    if np_dtype == 12:
        if c_contig:
            return 'C_DOUBLE_PRECISION'
        if f_contig:
            return 'DOUBLE_PRECISION'



