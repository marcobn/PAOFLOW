import os
import time
import numpy as np
from mpi4py import MPI
from communication import *
from read_inputfile_xml_parse import *
from read_QE_output_xml_parse import *
from read_new_QE_output_xml_parse import *


class DataController:

    comm = rank = size = None

    data_arrays = data_attributes = None

    def __init__ ( self, inputpath, inputfile ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if rank == 0:
            self.data_arrays = {}
            self.data_attributes = {}

            self.data_attributes['inputpath'] = inputpath
            self.data_attributes['inputfile'] = inputfile


    def clean_data ( self ):
        print(self.data_arrays.keys())


    def read_external_files ( self ):
        # Read Input Data
        self.read_pao_inputfile()
        self.read_qe_output()

        # Broadcast Data
        self.broadcast_data_arrays()
        self.broadcast_data_attributes()

    def read_pao_inputfile ( self ):
        if self.rank == 0:
            read_inputfile_xml(self.data_attributes['inputpath'], self.data_attributes['inputfile'], self)

    def read_qe_output ( self ):
        if self.rank == 0:
            fpath = self.data_attributes['fpath']
            if os.path.exists(fpath+'/data-file.xml'):
                read_QE_output_xml(self)
            elif os.path.exists(fpath+'/data-file-schema.xml'):
                read_new_QE_output_xml(self)
            else:
                raise Exception('data-file.xml or data-file-schema.xml were not found.\n')


    def write_file_row_col ( self, filename, col1, col2 ):
        if self.rank == 0:
            if len(col1) != len(col2):
                print('Data does not have the same shape')
                self.comm.Abort()

            with open(filename, 'w') as f:
                for i in range(len(col1)):
                    f.write('%.5f %.5e\n'%(col1[i],col2[i]))
        self.comm.Barrier()


    def broadcast_single_attribute ( self, key ):
        if rank == 0:
            for i in xrange(1,self.size):
                self.comm.send(self.data_attributes[key], dest=i)
        else:
            self.data_attributes[key] = self.comm.recv(source=0)


    def broadcast_data_attributes ( self ):
        if rank == 0:
            for i in xrange(1,self.size):
                self.comm.send(self.data_attributes, dest=i)
        else:
            self.data_attributes = self.comm.recv(source=0)


    def broadcast_data_arrays ( self ):
        if rank == 0:
            for i in xrange(1,self.size):
                self.comm.send(self.data_arrays, dest=i)
        else:
            self.data_arrays = self.comm.recv(source=0)


    def scatter_data_array ( self, key ):
        self.data_arrays[key] = scatter_array(self.data_arrays[key])


    def gather_data_array ( self, key ):
        arr = self.data_arrays[key]
        aux = None
        size1 = arr.shape[0]

        if rank == 0:
            for i in xrange(1,self.size):
                size1 += comm.recv(source=i)
        else:
            comm.send(arr.shape[0], dest=0)

        if rank == 0:
            if len(arr.shape) > 1:
                aux = np.ndarray(shape=((size1,)+arr.shape[1:]), dtype=arr.dtype)
            else:
                aux = np.ndarray(shape=(size1), dtype=arr.dtype)

        gather_array(aux, arr)

        if self.rank == 0:
            self.data_arrays[key] = aux
        else:
            self.data_arrays[key] = None
