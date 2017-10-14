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
import cmath
import sys, time
import os

def write2bxsf(fermi_dw,fermi_up,bands, nx, ny, nz, nbnd, ind_plot, Efermi, alat,x0, b_vectors, filename, inputpath):
    with open (os.path.join(inputpath,'{0}'.format(filename)),'w') as f:
        f.write('\nBEGIN_INFO\n  Fermi Energy: {:15.9f}\n  Shift Range: {:12.9f}eV to{:12.9f}eV\nEND_INFO\n'.format(Efermi,fermi_dw,fermi_up))
        # BXSF scalar-field header
        f.write('\nBEGIN_BLOCK_BANDGRID_3D\nband_energies\nBANDGRID_3D_BANDS\n')  
        # number of points in each direction
        f.write('{:12d}\n'.format(nbnd))
        f.write('{:12d}{:12d}{:12d}\n'.format(nx+1,ny+1,nz+1))
        # origin (should be zero, if I understan correctly)
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in x0 )))
        # 1st spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[0]*2*np.pi/alat )))
        # 2nd spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[1]*2*np.pi/alat )))
        # 3rd spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in b_vectors[2]*2*np.pi/alat )))
        
        for ib in range(nbnd):
            f.write('  BAND: {:5d}\n'.format(int(ind_plot[ib])+1))
            for ix in range(nx):
                for iy in range(ny):
			f.write('    {}'.format(''.join('%15.9f'%F for F in bands[ix,iy,:,ib] )))
			f.write('{:15.9f}\n'.format(bands[ix,iy,0,ib]))
	        f.write('    {}'.format(''.join('%15.9f'%F for F in bands[ix,0,:,ib] )))
		f.write('{:15.9f}\n'.format(bands[ix,0,0,ib]))
	    for iy in range(ny):
	        f.write('    {}'.format(''.join('%15.9f'%F for F in bands[0,iy,:,ib] )))
                f.write('{:15.9f}\n'.format(bands[0,iy,0,ib]))
            f.write('    {}'.format(''.join('%15.9f'%F for F in bands[0,0,:,ib] )))
            f.write('{:15.9f}\n'.format(bands[0,0,0,ib]))        
        f.write('END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3d\n')
            
    return()

