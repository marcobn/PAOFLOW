#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

class DataController:

  comm = rank = size = None

  data_arrays = data_attributes = None

  error_handler = report_exception = None

  def __init__ ( self, workpath, outputdir, inputfile, savedir, smearing, npool, verbose ):
    '''
    Initialize the DataController

    Arguments:
        workpath (str): Path to the working directory
        outputdir (str): Name of the output directory (created in the working directory path)
        inputfile (str): (optional) Name of the xml inputfile
        savedir (str): QE .save directory
        smearing (str): Smearing type (None, m-p, gauss)
        verbose (bool): False supresses debugging output

    Returns:
        None
    '''
    from os import mkdir
    from mpi4py import MPI
    from os.path import join,exists
    from .ErrorHandler import ErrorHandler

    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

    if inputfile is None and savedir is None:
      if self.rank == 0:
        print('\nERROR: Must specify \'.save\' directory path, either in PAOFLOW constructor or in an inputfile.')
      quit()

    self.error_handler = ErrorHandler()
    self.report_exception = self.error_handler.report_exception

    if self.rank == 0:
      self.data_arrays = arrays = {}
      self.data_attributes = attr = {}

      # Set or update attributes
      attr['savedir'] = savedir
      attr['verbose'] = verbose
      attr['workpath'] = workpath
      attr['inputfile'],attr['outputdir'] = inputfile,outputdir
      attr['fpath'] = join(workpath, (savedir if inputfile==None else inputfile))
      attr['opath'] = join(workpath, outputdir)

      if inputfile == None:
        attr['temp'] = .025852
        attr['npool'],attr['smearing'] = npool,smearing

      # Create the output directory
      if not exists(attr['opath']):
        mkdir(attr['opath'])

      self.add_default_arrays()

      attr['abort_on_exception'] = True

      # Read inputfile, if it exsts
      try:
        if inputfile != None:
          if not exists(attr['fpath']):
            raise Exception('ERROR: Inputfile does not exist\n%s'%attr['fpath'])
          self.read_pao_inputfile()
        else:
          attr['do_spin_orbit'] = False
        self.read_qe_output()
      except:
        print('\nERROR: Could not read QE xml data file')
        self.report_exception('Data Controller Initialization')
        self.comm.Abort()

      # Ensure that the number of k-points from QE matches the grid size
      nkpnts = attr['nk1']*attr['nk2']*attr['nk3']
#      if nkpnts != attr['nkpnts']:
#        print('\nERROR: Number of QE k-points does not match the MP grid size. Calculate nscf with nosym=.true. & noinv=.true.\n')
#        self.comm.Abort()

    self.comm.Barrier()

    # Broadcast Data
    try:
      self.data_arrays = self.comm.bcast(self.data_arrays, root=0)
      self.data_attributes = self.comm.bcast(self.data_attributes, root=0)
    except:
      print('ERROR: MPI was unable to broadcast')
      self.report_exception('Initialization Broadcast')
      self.comm.Abort()


  def data_dicts ( self ):
    '''
    Get the data dictionaries

    Arguments:
        None

    Returns:
       (arrays, attributes): Tuple of two dictionaries, data_arrays and data_attributes 
    '''
    return(self.data_arrays, self.data_attributes)


  def print_data ( self ):
    '''
    Print the data dictionary keys

    Arguments:
        None

    Returns:
        None
    '''
    if self.rank == 0:
      print('\nData Attributes:')
      print(self.data_attributes.keys())
      print('\nData Arrays:')
      print(self.data_arrays.keys())
      print('\n')


  def add_default_arrays ( self ):
    import numpy as np

    # Band Path
    self.data_attributes['band_path'] = None
    self.data_arrays['high_sym_points'] = {}

    # Electric Field
    self.data_arrays['Efield'] = np.zeros(3, dtype=float)
    # Magnetic Field
    self.data_arrays['Bfield'] = np.zeros(3, dtype=float)
    # Hubbard Parameters
    self.data_arrays['HubbardU'] = np.zeros(1, dtype=float)

    # Tensor components
    # Dielectric function
    self.data_arrays['d_tensor'] = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    # Boltzmann transport
    self.data_arrays['t_tensor'] = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    # Berry curvature
    self.data_arrays['a_tensor'] = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    # Spin Berry curvature
    self.data_arrays['s_tensor'] = np.array([[0,0,0],[0,1,0],[0,2,0],[1,0,0],[1,1,0],[1,2,0],[2,0,0],[2,1,0],[2,2,0], \
                                             [0,0,1],[0,1,1],[0,2,1],[1,0,1],[1,1,1],[1,2,1],[2,0,1],[2,1,1],[2,2,1], \
                                             [0,0,2],[0,1,2],[0,2,2],[1,0,2],[1,1,2],[1,2,2],[2,0,2],[2,1,2],[2,2,2]])


  def read_pao_inputfile ( self ):
    if self.rank == 0:
      from .defs.read_inputfile_xml_parse import read_inputfile_xml
      read_inputfile_xml(self.data_attributes['workpath'], self.data_attributes['inputfile'], self)

  def read_qe_output ( self ):
    if self.rank == 0:
      from os.path import exists
      fpath = self.data_attributes['fpath']
      if exists(fpath+'/data-file.xml'):
        from .defs.read_QE_output_xml_parse import read_QE_output_xml
        read_QE_output_xml(self)
      elif exists(fpath+'/data-file-schema.xml'):
        from .defs.read_new_QE_output_xml_parse import read_new_QE_output_xml
        read_new_QE_output_xml(self)
      else:
        raise Exception('data-file.xml or data-file-schema.xml were not found.\n')


  def write_file_row_col ( self, fname, col1, col2 ):
    '''
    Write a file with 2 columns

    Arguments:
        fname (str): Name of the file (written to outputdir)
        col1 (ndarray): 1D array of values for the rightmost column
        col2 (ndarray): 1D array of values for the leftmost column

    Returns:
        None
    '''
    if self.rank == 0:
      from os.path import join
      if len(col1) != len(col2):
        print('ERROR: Cannot write file: %s'%fname)
        print('Data does not have the same shape')
        self.comm.Abort()

      attr = self.data_attributes

      with open(join(attr['opath'],fname), 'w') as f:
        for i in range(len(col1)):
          f.write('%.5f %.5e\n'%(col1[i],col2[i]))
    self.comm.Barrier()


  def write_bxsf ( self, fname, bands, nbnd, indices=None ):
    '''
    Write a file in the bxsf format

    Arguments:
        fname (str): Name of the file (written to outputdir)
        bands (ndarray): The data array of values to be written
        nbnd (int): Number of values from array to write

    Returns:
        None
    '''
    attr = self.data_attributes

    if self.rank == 0:
      from .defs.write2bxsf import write2bxsf

      write2bxsf(self, fname, bands, nbnd, indices, attr['fermi_up'], attr['fermi_dw'])


  def write_bands ( self, fname, bands ):
    '''
    Write electronic band structure file

    Arguments:
        fname (str): Name of the file (written to outputdir)
        bands (ndarray): Band data array (shape: (nk,nbnd,nspin))

    Returns:
        None
    '''
    if self.rank == 0:
      from os.path import join

      arry,attr = self.data_dicts()
      nspin,nkpi = attr['nspin'],arry['kq'].shape[1]

      for ispin in range(nspin):
        with open(join(attr['opath'],fname+'_'+str(ispin)+'.dat'), 'w') as f:
          for ik in range(nkpi):
            f.write(' '.join(['%6d'%ik]+['% 14.8f'%j for j in bands[ik,:,ispin]])+'\n') 
    self.comm.Barrier()


  def write_kpnts_path ( self, fname, path, kpnts, b_vectors ):
    '''
    Write the k-path through the BZ

    Arguments:
        fname (str): Name of the file (written to outputdir)
        kpnts (ndarray): k-point path (shape: (3,nk))

    Returns:
        None
    '''
    if self.rank == 0:
      from os.path import join

      kpnts= b_vectors.T.dot(kpnts)
      with open(join(self.data_attributes['opath'],fname), 'w') as f:
        f.write(path)
        f.write(''.join(['%s %s %s\n'%(kpnts[0,i],kpnts[1,i],kpnts[2,i]) for i in range(kpnts.shape[1])]))
    self.comm.Barrier()



  def write_Hk_acbn0 ( self ):
    #----------------------
    # write to file Hks,Sks,kpnts,kpnts_wght
    #----------------------                                                                                                                           
    import numpy as np
    import os,sys

    arry,attr    = self.data_dicts()
    nspin        = attr['nspin']
    inputpath    = attr['opath']
    write_binary = attr['write_binary']
    nawf         = attr['nawf']
    non_ortho    = attr['non_ortho']
    kpnts_wght   = arry['kpnts_wght']
    kpnts        = arry['kpnts']
    Hks          = arry['Hks']
    Sks          = arry['Sks']
    nkpnts       = kpnts.shape[0]
    


    if write_binary:# or whatever you want to call it            
        if nspin==1:#postfix .npy just to make it clear what they are
            np.save('kham.npy',np.ravel(Hks[...,0],order='C'))
        if nspin==2:
            np.save('kham_up.npy',np.ravel(Hks[...,0],order='C'))
            np.save('kham_dn.npy',np.ravel(Hks[...,1],order='C'))
        if non_ortho:
            np.save('kovp.npy',np.ravel(Sks,order='C'))
    else:
        if nspin == 1:
            f=open(os.path.join(inputpath,'kham.txt'),'w')
            for ik in range(nkpnts):
                for i in range(nawf):
                    for j in range(nawf):
                        f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
            f.close()
        elif nspin == 2:
            f=open(os.path.join(inputpath,'kham_up.txt'),'w')
            for ik in range(nkpnts):
                for i in range(nawf):
                    for j in range(nawf):
                        f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
            f.close()
            f=open(os.path.join(inputpath,'kham_down.txt'),'w')
            for ik in range(nkpnts):
                for i in range(nawf):
                    for j in range(nawf):
                        f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,1]),np.imag(Hks[i,j,ik,1])))
            f.close()
        if non_ortho:
            f=open(os.path.join(inputpath,'kovp.txt'),'w')
            for ik in range(nkpnts):
                for i in range(nawf):
                    for j in range(nawf):
                        f.write('%20.13f %20.13f \n' %(np.real(Sks[i,j,ik]),np.imag(Sks[i,j,ik])))
            f.close()
    f=open(os.path.join(inputpath,'k.txt'),'w')
    for ik in range(nkpnts):
        f.write('%20.13f %20.13f %20.13f \n' %(kpnts[ik,0],kpnts[ik,1],kpnts[ik,2]))
    f.close()
    f=open(os.path.join(inputpath,'wk.txt'),'w')
    for ik in range(nkpnts):
        f.write('%20.13f \n' %(kpnts_wght[ik]))
    f.close()

    print('H(k),S(k),k,wk written to file')
    sys.exit()


  def write_z2pack ( self, fname ):
    '''
    Write HRs in the Z2 Pack format

    Arguments:
        fname (str): Name of the file (written to outputdir)

    Returns:
        None
    '''
    try:
      if self.rank == 0:
        import numpy as np
        from os.path import join
  
        arry,attr = self.data_dicts()
  
        with open(join(attr['opath'],fname), 'w') as f:#'z2pack_hamiltonian.dat','w')
  
          nawf,nkpts = attr['nawf'],attr['nkpnts']
          nk1,nk2,nk3 = attr['nk1'],attr['nk2'],attr['nk3']
  
          f.write("PAOFLOW Generated \n")
          f.write('%5d \n'%nawf)
  
          f.write('%5d \n'%nkpts)
  
          nl = 15 # z2pack read the weights in lines of 15 items
  
          nlines = nkpts//nl # number of lines
          nlast = nkpts%nl   # number of items of laste line if needed
  
          kq_wght_int = np.zeros(arry['kq_wght'].shape,dtype=int)
          for kn in range(arry['kq_wght'].shape[0]):
              kq_wght_int[kn] = int(round(arry['kq_wght'][kn]*attr['nkpnts']))

          # print each cell weight
          for i in range(nlines):
            jp = i * nl
            f.write('   '.join('{:d} '.format(j) for j in kq_wght_int[jp:jp+nl]) + '\n')
  
          # Last line if needed
          if nlast != 0:
            f.write('   '.join('{:d} '.format(j) for j in kq_wght_int[nlines*nl:nkpts]) + '\n')
  
  #### Can be condensed
          for i in range(nk1):
            for j in range(nk2):
              for k in range(nk3):
                n = k + j*nk3 + i*nk2*nk3
                Rx = float(i)/float(nk1)
                Ry = float(j)/float(nk2)
                Rz = float(k)/float(nk3)
                if Rx >= 0.5: Rx=Rx-1.0
                if Ry >= 0.5: Ry=Ry-1.0
                if Rz >= 0.5: Rz=Rz-1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                # the minus sign in Rx*nk1 is due to the Fourier transformation (Ri-Rj)
                ix=-round(Rx*nk1,0)
                iy=-round(Ry*nk2,0)
                iz=-round(Rz*nk3,0)
                for m in range(nawf):
                  for l in range(nawf):
                    # l+1,m+1 just to start from 1 not zero
                    f.write('%3d %3d %3d %5d %5d %14f %14f\n'%(ix,iy,iz,l+1,m+1,arry['HRs'][l,m,i,j,k,0].real,arry['HRs'][l,m,i,j,k,0].imag))
    except:
      self.report_exception('z2_pack')
      if self.data_attributes['abort_on_exception']:
        self.comm.Abort()

    self.comm.Barrier()



  def broadcast_single_array ( self, key, dtype=complex, root=0 ):
    '''
    Broadcast array from 'data_arrays' with 'key' from 'root' to all other ranks

    Arguments:
        key (str): The key for the array to broadcast (key must exist in dictionary 'data_arrays')
        dtype (dtype): The data type of the array to broadcast
        root (int): The rank which is the source of the broadcasted array

    Returns:
        None
    '''
    import numpy as np
    ashape = self.comm.bcast((None if self.rank!=root else self.data_arrays[key].shape), root=root)
    if self.rank != root:
      self.data_arrays[key] = np.zeros(ashape, dtype=dtype, order='C')
    self.comm.Bcast(np.ascontiguousarray(self.data_arrays[key]), root=root)

  def broadcast_attribute ( self, key, root=0 ):
    '''
    Broadcast attribute from 'data_attributes' with 'key' from 'root' to all other ranks

    Arguments:
        key (str): The key for the attribute to broadcast (key must exist in dictionary 'data_attributes')
        root (int): The rank which is the source of the broadcasted attribut

    Returns:
        None
    '''
    attr = self.data_attributes[key] if self.rank==root else None
    self.data_attributes[key] = self.comm.bcast(attr, root=root)

### This section is under construction
### Only 'broadcast_single_array' should be used
  def broadcast_data_attributes ( self ):
    if self.rank == 0:
      for i in range(1,self.size):
        self.comm.send(self.data_attributes, dest=i)
    else:
      self.data_attributes = self.comm.recv(source=0)

  def broadcast_data_arrays ( self ):
    if self.rank == 0:
      for i in range(1,self.size):
        self.comm.send(self.data_arrays, dest=i)
    else:
      self.data_arrays = self.comm.recv(source=0)

  def scatter_data_array ( self, key ):
    from .defs.communication import scatter_array
    self.data_arrays[key] = scatter_array(self.data_arrays[key])

  def gather_data_array ( self, key ):
    import numpy as np
    from .defs.communication import gather_array
    arr = self.data_arrays[key]
    aux = None
    size1 = arr.shape[0]

    if self.rank == 0:
      for i in range(1,self.size):
        size1 += comm.recv(source=i)
    else:
      comm.send(arr.shape[0], dest=0)

    if self.rank == 0:
      if len(arr.shape) > 1:
        aux = np.ndarray(shape=((size1,)+arr.shape[1:]), dtype=arr.dtype)
      else:
        aux = np.ndarray(shape=(size1), dtype=arr.dtype)

    gather_array(aux, arr)

    if self.rank == 0:
      self.data_arrays[key] = aux
    else:
      self.data_arrays[key] = None
