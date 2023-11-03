#
# PAOFLOW
#
# Copyright 2016-2022 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
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

from mendeleev import element
import numpy as np

def write2xsf(data_controller,filename,data=None):
	
	arry,attr = data_controller.data_dicts()
	
	fileobj = open(filename,'w')
	atoms = arry['atoms']
	
	fileobj.write('CRYSTAL\n')
	
	fileobj.write('PRIMVEC\n')
	for i in range(3):
		fileobj.write(' %.14f %.14f %.14f\n' % tuple(arry['a_vectors'][i]*attr['alat']))
		
	fileobj.write('PRIMCOORD\n')
	fileobj.write(str(len(atoms))+' 1\n')         
	for na in range(len(atoms)):
		atom = element(arry['atoms'][na])
		fileobj.write(' %2d' % atom.atomic_number)
		fileobj.write(' %20.14f %20.14f %20.14f' % tuple(arry['tau'][na]))
		fileobj.write('\n')
		
	if data is None:
		fileobj.close()
		return
	
	fileobj.write('BEGIN_BLOCK_DATAGRID_3D\n')
	fileobj.write(' data\n')
	fileobj.write(' BEGIN_DATAGRID_3Dgrid#1\n')
	
	data = np.asarray(data)
	if data.dtype == np.complex128:
		data = np.abs(data)**2
		
	shape = data.shape
	fileobj.write('  %d %d %d\n' % (shape[0]+1, shape[1]+1, shape[2]+1))
	
	origin = np.zeros(3)
	fileobj.write('  %f %f %f\n' % tuple(origin))
	# These are the actual coordinates for the data
	for i in range(3):
		fileobj.write('  %f %f %f\n' %
			tuple(np.array(arry['a_vectors'][i])*attr['alat']))
		
	for k in range(shape[2]+1):
		for j in range(shape[1]+1):
			fileobj.write('   ')
			for i in range(shape[0]+1):
				fileobj.write('%12.8e ' % (data[i%shape[0], j%shape[1], k%shape[2]]))
			fileobj.write('\n')
			
	fileobj.write(' END_DATAGRID_3D\n')
	fileobj.write('END_BLOCK_DATAGRID_3D\n')
	fileobj.close()
