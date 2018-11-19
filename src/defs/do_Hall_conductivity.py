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


def do_spin_Hall_conductivity ( data_controller, jksp, pksp, ipol, jpol ):
  import numpy as np
  from mpi4py import MPI
  from communication import gather_full
  from smearing import intgaussian, intmetpax

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  snktot = jksp.shape[0]
  nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']

  # Compute the optical conductivity tensor sigma_xy(ene)

  ispin = 0

  emin = 0.0
  emax = attributes['shift']
  de = (emax-emin)/500.
  ene = np.arange(emin, emax, de)
  esize = ene.size

  sigxy_aux = smear_sigma_loop(data_controller, ene, jksp, pksp, ispin, ipol, jpol)

  sigxy = (np.zeros((esize),dtype=complex) if rank==0 else None)

  comm.Reduce(sigxy_aux, sigxy, op=MPI.SUM)
  sigxy_aux = None

  if rank == 0:
    sigxy /= float(attributes['nkpnts'])
    return(ene, sigxy)
  else:
    return(None, None)


def do_Berry_conductivity ( data_controller, pksp_i, pksp_j, ipol, jpol ):
  import numpy as np
  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  arrays,attributes = data_controller.data_dicts()

  snktot = pksp_j.shape[0]
  bnd = attributes['bnd']

  # Compute the optical conductivity tensor sigma_xy(ene)

  ispin = 0

  emin = 0.0
  emax = attributes['shift']
  de = (emax-emin)/500.
  ene = np.arange(emin, emax, de)
  esize = ene.size

  sigxy_aux = np.zeros((esize),dtype=complex)

  sigxy_aux = smear_sigma_loop(data_controller, ene, pksp_i, pksp_j, ispin, ipol, jpol)

  sigxy = (np.zeros((esize),dtype=complex) if rank==0 else None)

  comm.Reduce(sigxy_aux, sigxy, op=MPI.SUM)
  sigxy_aux = None

  if rank == 0:
    sigxy /= float(attributes['nkpnts'])
    return(ene, sigxy)
  else:
    return(None, None)


def smear_sigma_loop ( data_controller, ene, pksp_i, pksp_j, ispin, ipol, jpol ):
  import numpy as np
  from smearing import intgaussian,intmetpax

  arrays,attributes = data_controller.data_dicts()

  esize = ene.size
  sigxy = np.zeros((esize), dtype=complex)

  snktot,nawf,_,nspin = pksp_j.shape
  f_nm = np.zeros((snktot,nawf,nawf), dtype=float)
  E_diff_nm = np.zeros((snktot,nawf,nawf), dtype=float)

  print(attributes['temp'], ispin, nawf)

  Ef = 0.0
  eps = 1.0e-16
  delta = 0.05

  if attributes['smearing'] == None:
    fn = 1.0/(np.exp(arrays['E_k'][:,:,ispin]/attributes['temp'])+1)
  elif attributes['smearing'] == 'gauss':
    fn = intgaussian(arrays['E_k'][:,:,ispin], Ef, arrays['deltakp'][:,:,ispin])
  elif smearing == 'm-p':
    fn = intmetpax(arrays['E_k'][:,:,ispin], Ef, arrays['deltakp'][:,:,ispin]) 

  # Collapsing the sum over k points
  for n in range(nawf):
    for m in range(nawf):
      if m != n:
        E_diff_nm[:,n,m] = (arrays['E_k'][:,n,ispin]-arrays['E_k'][:,m,ispin])**2
        f_nm[:,n,m] = (fn[:,n] - fn[:,m])*np.imag(pksp_i[:,n,m,ispin]*pksp_j[:,m,n,ispin])

  fn = None

  for e in range(esize):
    if attributes['smearing'] != None:
      sigxy[e] = np.sum(f_nm[:,:,:]/(E_diff_nm[:,:,:]-(ene[e]+1.j*arrays['deltakp2'][:,:nawf,:nawf,ispin])**2+eps))
    else:
      sigxy[e] = np.sum(f_nm[:,:,:]/(E_diff_nm[:,:,:]-(ene[e]+1.j*arrays['delta'])**2+eps))

  F_nm = None
  E_diff_nm = None

  return np.nan_to_num(sigxy)
