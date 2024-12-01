#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

def write2bxsf4skeaf ( data_controller, fname, bands, nbnd, indices, fermi_up, fermi_dw ):
  '''
  Writes a bxsf File compatible with skeaf to 'opath' (outputdir)

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

  Efermi = 0.0
  Ryd_conv = 0.0734986176
  Ryd_bands = np.zeros_like(bands)
  Ryd_bands[:,:,:,:]=Ryd_conv*bands[:,:,:,:] 
  alat,b_vectors = attributes['alat'],arrays['b_vectors']
  nx,ny,nz = attributes['nk1'],attributes['nk2'],attributes['nk3']

  for ib in range(nbnd):
    with open (join(attributes['workpath'],attributes['outputdir'],'Fermi_surf_band_%d.bxsf'%(ib+1)),'w') as f:
      f.write('\nBEGIN_INFO\n  Fermi Energy: {:15.9f}\nEND_INFO\n'.format(Efermi))
      # BXSF scalar-field header
      f.write('\nBEGIN_BLOCK_BANDGRID_3D\nband_energies\nBANDGRID_3D_BANDS\n')  
      # number of points in each direction
      f.write('{:12d}\n'.format(1))
      f.write('{:12d}{:12d}{:12d}\n'.format(nx+1,ny+1,nz+1))
      # origin (should be zero, if I understan correctly)
      f.write('  {}\n'.format(''.join('%10.6f'%F for F in x0)))
      # 1st spanning (=lattice) vector
      f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[0]/alat)))
      # 2nd spanning (=lattice) vector
      f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[1]/alat)))
      # 3rd spanning (=lattice) vector
      f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[2]/alat)))

      f.write('  BAND: {:5d}\n'.format(int(indices[ib])+1))
      combined_band = []
      for ix in range(nx):
        for iy in range(ny):
          for F in Ryd_bands[ix,iy,:,ib]:
            combined_band.append(F)
          combined_band.append((Ryd_bands[ix,iy,0,ib]))
        for F in Ryd_bands[ix,0,:,ib]:
          combined_band.append(F)
        combined_band.append(Ryd_bands[ix,0,0,ib])
      for iy in range(ny):
        for F in Ryd_bands[0,iy,:,ib]:
          combined_band.append(F)
        combined_band.append((Ryd_bands[0,iy,0,ib]))
      for F in Ryd_bands[0,0,:,ib]:
        combined_band.append(F)
      combined_band.append((Ryd_bands[0,0,0,ib]))
      for i in range(len(combined_band)):
        if (i+1)%6 == 0:
          f.write('{:15.9f}\n'.format(combined_band[i]))
        else:
          f.write('{:15.9f}'.format(combined_band[i]))
      f.write('\nEND_BANDGRID_3D\nEND_BLOCK_BANDGRID_3d\n')
      f.close()
