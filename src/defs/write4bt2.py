#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
#F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli, Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np
from .get_K_grid_fft import get_K_grid_fft_crystal

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def write4bt2(data_controller):
	
	'''
		Write data in the GENE format suitable to be read by BoltzTrap2
		Returns also band derivatives (momentum matix) if available
		It shuld be called after:
		paoflow.pao_hamiltonian()
		paoflow.pao_eigh()
		paoflow.gradient_and_momenta()
		optional: call paoflow.interpolated_hamiltonian() to improve on the initial meshh of k-points
		
	'''
	
	if rank == 0:
		
		Ry2eV = 13.60569193
		arry,attr = data_controller.data_dicts()
		
		arry['kgrid'] = get_K_grid_fft_crystal(attr['nk1'],attr['nk2'],attr['nk3'])
		
		# write data in the GENE format for BoltzTrap2
		prefix = attr['savedir'].split('.')[0]
		fname_energy = prefix + '.energy'
		fname_struct = prefix + '.structure'
		
		f_energy = prefix + '\n'
		f_energy += str(attr['nkpnts']) + ' ' + str(int(attr['nspin'])) + ' ' + str(0) + '\n'
		for ik in range(attr['nkpnts']):
			f_energy += str(arry['kgrid'][ik][0]) + ' ' +\
						str(arry['kgrid'][ik][1]) + ' ' +\
						str(arry['kgrid'][ik][2]) +' ' + str(attr['nbnds']) + '\n'
			for ib in range(attr['nbnds']):
				f_energy += str(arry['E_k'][ik,ib,0]/Ry2eV) + '\n'
		
		f = open(fname_energy, 'w')
		f.write(f_energy)
		f.close()
		
		f_struct = prefix + '\n'
		for i in range(3):
			f_struct += str(arry['a_vectors'][i][0]*attr['alat']) + ' ' +\
						str(arry['a_vectors'][i][1]*attr['alat']) + ' ' +\
						str(arry['a_vectors'][i][2]*attr['alat']) + '\n'
		f_struct += str(attr['natoms']) + '\n'
		for ia in range(attr['natoms']):
			f_struct += str(arry['atoms'][ia]) + ' '
			f_struct += str(arry['tau'][ia].dot(arry['b_vectors'].T)[0]/attr['alat']) + ' ' +\
						str(arry['tau'][ia].dot(arry['b_vectors'].T)[1]/attr['alat']) + ' ' +\
						str(arry['tau'][ia].dot(arry['b_vectors'].T)[2]/attr['alat']) + '\n'
		
		f = open(fname_struct, 'w')
		f.write(f_struct)
		f.close()
		
		try:
			mommat = np.zeros((attr['nkpnts'],attr['nbnds'],3),dtype=float)
			for ib in range(attr['nbnds']):
				mommat[:,ib,:] = -np.real(arry['pksp'][:,:,ib,ib,0])/(2*Ry2eV)
			return(mommat)
		except:
			print('momentum matrix not available')
			return(None)
