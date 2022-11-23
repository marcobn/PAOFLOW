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

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def do_rashba_edelstein (data_controller, ene, temperature, regularization, twoD_structure, lattice_height, structure_thickness, write_to_file):
  import numpy as np
  from os.path import join
  from .smearing import gaussian
  from .constants import ELECTRONVOLT_SI,BOHR_RADIUS_CM,HBAR,LL

  comm,rank = data_controller.comm,data_controller.rank
  arrays,attr = data_controller.data_dicts()

  snktot = arrays['v_k'].shape[0]
  ind_plot = arrays['ind_plot']
  nstates = len(ind_plot)
  nktot = attr['nkpnts']
  tau_const = 1.2e-9 #spin relaxation time (this value is for WTe2)
  esize = ene.size

  pksp = np.take(np.diagonal(np.real(arrays['pksp'][:,:,:,:,0]),axis1=2,axis2=3), ind_plot, axis=2)

  deltakp = np.take(arrays['deltakp'], ind_plot, axis=1)[:,:,0]
  E_k = np.take(arrays['E_k'], ind_plot, axis=1)[:,:,0]
  St = np.real(arrays['sktxt'])

  kai_aux = np.zeros((snktot,3,3,nstates), dtype=float)
  j_aux = np.zeros((snktot,3,3,nstates), dtype=float)
  for l in range(3):
    for m in range(3):
      kai_aux[:,l,m,:] = tau_const*St[:,l,:]*pksp[:,m,:]
      j_aux[:,l,l,:] = tau_const*pksp[:,l,:]*pksp[:,l,:]   

  kai_eaux = np.zeros((snktot,3,3,esize), dtype=float)
  j_eaux = np.zeros((snktot,3,3,esize), dtype=float)

  def dfermi(E,ene,temp):
    return -1/(4*temp*(np.cosh((E-ene)/(2*temp))**2))

  for i in range(esize):
    gaussian_smear = None
    if attr['smearing'] == 'gauss':
      if temperature == 0:
        gaussian_smear = gaussian(E_k, ene[i], deltakp)
      else:
        gaussian_smear = dfermi(E_k, ene[i], temperature)
    else:
      raise ValueError('Routine requires \'gauss\' smearing')
    for l in range(3):
      for m in range(3):
        kai_eaux[:,l,m,i] = np.sum(kai_aux[:,l,m,:]*gaussian_smear, axis=1)
        j_eaux[:,l,l,i] = np.sum(j_aux[:,l,l,:]*gaussian_smear, axis=1)
  kai_aux = None
  j_aux = None

  kai = (np.zeros((3,3,esize),dtype=float) if rank==0 else None)
  jc = (np.zeros((3,3,esize),dtype=float) if rank==0 else None)

  kai_eaux = np.ascontiguousarray(np.sum(kai_eaux,axis=0))
  j_eaux = np.ascontiguousarray(np.sum(j_eaux,axis=0))

  comm.Reduce(kai_eaux, kai, op=MPI.SUM)
  comm.Reduce(j_eaux, jc, op=MPI.SUM)

  if rank == 0:
    if not twoD_structure:
      Ekai_xx = ((-1*kai[0,0]*HBAR) / ((jc[0,0]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_xy = ((-1*kai[0,1]*HBAR) / ((jc[1,1]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_xz = ((-1*kai[0,2]*HBAR) / ((jc[2,2]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_yx = ((-1*kai[1,0]*HBAR) / ((jc[0,0]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_yy = ((-1*kai[1,1]*HBAR) / ((jc[1,1]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_yz = ((-1*kai[1,2]*HBAR) / ((jc[2,2]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_zx = ((-1*kai[2,0]*HBAR) / ((jc[0,0]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_zy = ((-1*kai[2,1]*HBAR) / ((jc[1,1]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
      Ekai_zz = ((-1*kai[2,2]*HBAR) / ((jc[2,2]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))
    else:
      Ekai_xx = ((-1*kai[0,0]*HBAR) / ((jc[0,0]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_xy = ((-1*kai[0,1]*HBAR) / ((jc[1,1]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_xz = ((-1*kai[0,2]*HBAR) / ((jc[2,2]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_yx = ((-1*kai[1,0]*HBAR) / ((jc[0,0]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_yy = ((-1*kai[1,1]*HBAR) / ((jc[1,1]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_yz = ((-1*kai[1,2]*HBAR) / ((jc[2,2]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_zx = ((-1*kai[2,0]*HBAR) / ((jc[0,0]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_zy = ((-1*kai[2,1]*HBAR) / ((jc[1,1]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)
      Ekai_zz = ((-1*kai[2,2]*HBAR) / ((jc[2,2]*ELECTRONVOLT_SI*BOHR_RADIUS_CM)+regularization))*(lattice_height / structure_thickness)

    wtup_kai = lambda fn,tu : fn.write('% .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tu)
    gtup_kai = lambda tu,i : (ene[i],tu[0,0,i],tu[0,1,i],tu[0,2,i],tu[1,0,i],tu[1,1,i],tu[1,2,i],tu[2,0,i],tu[2,1,i],tu[2,2,i])
    wtup_current = lambda fn,tu : fn.write('% .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tu)
    gtup_current = lambda tu,i : (ene[i],tu[0,0,i],tu[0,1,i],tu[0,2,i],tu[1,0,i],tu[1,1,i],tu[1,2,i],tu[2,0,i],tu[2,1,i],tu[2,2,i])

    wtup_Ekai_xx = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_xy = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_xz = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_yx = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_yy = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_yz = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_zx = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_zy = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    wtup_Ekai_zz = lambda fn,tu : fn.write('% .5f % 9.5e\n'%tu)
    gtup_Ekai_xx = lambda tu,i : (ene[i],tu[i]) 
    gtup_Ekai_xy = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_xz = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_yx = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_yy = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_yz = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_zx = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_zy = lambda tu,i : (ene[i],tu[i])
    gtup_Ekai_zz = lambda tu,i : (ene[i],tu[i])

    if write_to_file:
      fkai = open(join(attr['opath'],'kai.dat'), 'w')
      fcurrent = open(join(attr['opath'],'current.dat'), 'w')

      fEkai_xx = open(join(attr['opath'],'Ekai_xx.dat'), 'w')
      fEkai_xy = open(join(attr['opath'],'Ekai_xy.dat'), 'w')
      fEkai_xz = open(join(attr['opath'],'Ekai_xz.dat'), 'w')
      fEkai_yx = open(join(attr['opath'],'Ekai_yx.dat'), 'w')
      fEkai_yy = open(join(attr['opath'],'Ekai_yy.dat'), 'w')
      fEkai_yz = open(join(attr['opath'],'Ekai_yz.dat'), 'w')
      fEkai_zx = open(join(attr['opath'],'Ekai_zx.dat'), 'w')
      fEkai_zy = open(join(attr['opath'],'Ekai_zy.dat'), 'w')
      fEkai_zz = open(join(attr['opath'],'Ekai_zz.dat'), 'w')

      for i in range(esize):
        wtup_kai(fkai,gtup_kai(kai,i))
        wtup_current(fcurrent,gtup_current(jc,i))

        wtup_Ekai_xx(fEkai_xx, gtup_Ekai_xx(Ekai_xx,i))
        wtup_Ekai_xy(fEkai_xy, gtup_Ekai_xy(Ekai_xy,i))
        wtup_Ekai_xz(fEkai_xz, gtup_Ekai_xz(Ekai_xz,i))
        wtup_Ekai_yx(fEkai_yx, gtup_Ekai_yx(Ekai_yx,i))
        wtup_Ekai_yy(fEkai_yy, gtup_Ekai_yy(Ekai_yy,i))
        wtup_Ekai_yz(fEkai_yz, gtup_Ekai_yz(Ekai_yz,i))
        wtup_Ekai_zx(fEkai_zx, gtup_Ekai_zx(Ekai_zx,i))
        wtup_Ekai_zy(fEkai_zy, gtup_Ekai_zy(Ekai_zy,i))
        wtup_Ekai_zz(fEkai_zz, gtup_Ekai_zz(Ekai_zz,i))
        
      fkai.close()
      fcurrent.close()

      fEkai_xx.close()
      fEkai_xy.close()
      fEkai_xz.close()
      fEkai_yx.close()
      fEkai_yy.close()
      fEkai_yz.close()
      fEkai_zx.close()
      fEkai_zy.close()
      fEkai_zz.close()

