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

def write2bxsf ( data_controller, fname, bands, nbnd, indices, fermi_up, fermi_dw ):
  '''
  Writes a bxsf File to 'opath' (outputdir)

  Arguments:
      data_controller (DataController): The current PAOFLOW DataController
      fname (str): File name
      bands (ndarray): Array with data to write
      nbnd (int): Number of columns of band-like data values to write
      indices (list): A list of included band indices, in ascending order
      fermi_up (float): Highest acceptable energy
      fermi_dw (float): Lowest acceptable energy

  Returns:
      None
  '''
  import numpy as np
  from os.path import join

  pi = np.pi
  arrays,attributes = data_controller.data_dicts()

  x0 = np.zeros(3, dtype=float)
  if indices is None:
    indices = np.zeros(nbnd, dtype=float)

  Efermi = attributes['Efermi']
  alat,b_vectors = attributes['alat'],arrays['b_vectors']
  nx,ny,nz = attributes['nk1'],attributes['nk2'],attributes['nk3']

  with open (join(attributes['workpath'],attributes['outputdir'],'{0}'.format(fname)),'w') as f:
    f.write('\nBEGIN_INFO\n  Fermi Energy: {:15.9f}\n  Shift Range: {:12.9f}eV to{:12.9f}eV\nEND_INFO\n'.format(Efermi,fermi_dw,fermi_up))
    # BXSF scalar-field header
    f.write('\nBEGIN_BLOCK_BANDGRID_3D\nband_energies\nBANDGRID_3D_BANDS\n')  
    # number of points in each direction
    f.write('{:12d}\n'.format(nbnd))
    f.write('{:12d}{:12d}{:12d}\n'.format(nx+1,ny+1,nz+1))
    # origin (should be zero, if I understan correctly)
    f.write('  {}\n'.format(''.join('%10.6f'%F for F in x0)))
    # 1st spanning (=lattice) vector
    f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[0]*2*pi/alat)))
    # 2nd spanning (=lattice) vector
    f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[1]*2*pi/alat)))
    # 3rd spanning (=lattice) vector
    f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[2]*2*pi/alat)))

    for ib in range(nbnd):
      f.write('  BAND: {:5d}\n'.format(int(indices[ib])+1))
      for ix in range(nx):
        for iy in range(ny):
          f.write('    {}'.format(''.join('%15.9f'%F for F in bands[ix,iy,:,ib])))
          f.write('{:15.9f}\n'.format(bands[ix,iy,0,ib]))
        f.write('    {}'.format(''.join('%15.9f'%F for F in bands[ix,0,:,ib])))
        f.write('{:15.9f}\n'.format(bands[ix,0,0,ib]))
      for iy in range(ny):
        f.write('    {}'.format(''.join('%15.9f'%F for F in bands[0,iy,:,ib])))
        f.write('{:15.9f}\n'.format(bands[0,iy,0,ib]))
      f.write('    {}'.format(''.join('%15.9f'%F for F in bands[0,0,:,ib])))
      f.write('{:15.9f}\n'.format(bands[0,0,0,ib]))        
    f.write('END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3d\n')
