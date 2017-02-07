#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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




import numpy as np
import cmath
import sys, time

def write3D( data,nx, ny, nz, alat,x0,B, filename):
    with open ('{0}'.format(filename),'w') as f:
        # XSF scalar-field header
        f.write('\nBEGIN_BLOCK_DATAGRID_3D\n3D_PAOPI\nDATAGRID_3D_UNKNOWN\n')
        # number of points in each direction
        f.write('{:12d}{:12d}{:12d}\n'.format(nx,ny,nz))
        # origin (should be zero, if I understan correctly)
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in x0 )))
        # 1st spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in B[0]*2*np.pi/alat )))
        # 2nd spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in B[1]*2*np.pi/alat )))
        # 3rd spanning (=lattice) vector
        f.write('  {}\n'.format(''.join('%10.6f'%F for F in B[2]*2*np.pi/alat )))

        for ix in range(nx):
            for iy in range(ny):
                f.write('    {}\n'.format(''.join('%15.9f'%F for F in data[ix,iy,:] )))
        f.write('END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n')
    return()

