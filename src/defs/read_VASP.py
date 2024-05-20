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

import xml.etree.cElementTree as ET
from mpi4py import MPI
import numpy as np
import spglib

def parse_vasprun_data ( data_controller, fname ):
  '''
  Parse the data_file_schema.xml file produced by Quantum Espresso.
  Populated the DataController object with all necessary information.

  Arguments:
    data_controller (DataController): Data controller to populate
    fname (str): Path and name of the xml file.
  '''
  import re

  arry,attr = data_controller.data_dicts()
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  verbose = attr['verbose']
  Ryd2eV = 13.605826  # Note this number is slightly different between VASP and QE
  AUTOA = 0.529177249  # see vasp/src/constant.inc

  tree = ET.parse(fname)
  root = tree.getroot()
  
  version = root.find("./generator/i[@name='version']").text
  if verbose: print('VASP version: {0:s}'.format(version))
  qe_version = list(map(int, version.strip().split(".")))

  dftSO = True if root.find(".//i[@name='LSORBIT']").text.strip() == 'T' else False
  ecutwfc = float(root.find(".//i[@name='ENCUT']").text.strip())/Ryd2eV # convert to Ryd
  ecutrho = ecutwfc*5
  nelec = float(root.find(".//i[@name='NELECT']").text.strip())
  nbnds = int(float(root.find(".//i[@name='NBANDS']").text.strip()))
  nawf = nbnds 
  nspin = int(root.find(".//i[@name='ISPIN']").text.strip())
  Efermi = float(root.find(".//i[@name='efermi']").text.strip()) # in eV


  # tag = atominfo  
  atom_elem = root.find('atominfo')
  species,pseudos = [],[]
  specs = atom_elem.find("./array[@name='atomtypes']/set")
  for i,s in enumerate(specs.findall(".//c")):
    if i%5 == 1:
      species.append(s.text.strip())
    elif i%5 == 4:
      pseudos.append(s.text.strip())  # Reference only, pseudos not used

  alat = 1.0
  natoms = int(atom_elem.find("atoms").text.strip())
  atms = atom_elem.find("./array[@name='atoms']/set")
  atoms = []
  for i,a in enumerate(atms.findall(".//c")):
    if i%2 == 0:
      atoms.append(a.text.strip())

  # tag = structure
  struct_elem = root.find("./structure[@name='initialpos']")
  basis = struct_elem.find("./crystal/varray[@name='basis']")
  rec_basis = struct_elem.find("./crystal/varray[@name='rec_basis']")
  a_Angstrom, b_Angstrom = [], []
  for a_vec in basis.findall("./v"):
    a_Angstrom.append([float(a) for a in a_vec.text.strip().split()])
  for b_vec in rec_basis.findall("./v"):
    b_Angstrom.append([float(b) for b in b_vec.text.strip().split()])
  # The definitions here are different from read_QE_xml.py
  # We fix alat = 1.0 Bohr for the VASP.
  a_vectors = np.array(a_Angstrom)/AUTOA # Convert Angstrom to Bohr
  b_vectors = np.array(b_Angstrom)*AUTOA
  omega = alat ** 3 * a_vectors[0, :].dot(np.cross(a_vectors[1, :], a_vectors[2, :]))

  pos_list = []
  positions = struct_elem.find("./varray[@name='positions']")
  for x in positions.findall("./v"):
    pos_list.append([float(t) for t in x.text.strip().split()])
  pos_arry = np.array(pos_list)
  tau = pos_arry @ a_vectors  # Atomic coord in cartesian, Bohr

  # tag = kpoints
  k_elem = root.find("./kpoints")
  kmesh = k_elem.find(".//v[@name='divisions']").text.strip().split()
  nk1, nk2, nk3 = int(kmesh[0]), int(kmesh[1]), int(kmesh[2])
  # if Monkhorst-Pack grid with even nk, shift of 0.5
  # otherwise, no shift
  mpgrid = k_elem.find("./generation").attrib['param']
  if mpgrid[0] == "M":
    k1, k2, k3 = int(nk1%2==0), int(nk2%2==0), int(nk3%2==0)
  else:
    k1, k2, k3 = 0, 0, 0
  if verbose: print('Monkhorst and Pack grid:', nk1, nk2, nk3, k1, k2, k3)

  kpnts_temp, kpnt_weights_temp = [], []
  for ks in k_elem.findall("./varray[@name='kpointlist']/v"):
    kpnts_temp.append([float(k) for k in ks.text.strip().split()])
  kpnts = np.array(kpnts_temp)@b_vectors
  nkpnts = int(kpnts.shape[0])

  for k in k_elem.findall("./varray[@name='weights']/v"):
    kpnt_weights_temp.append(float(k.text.strip()))
  kpnt_weights = np.array(kpnt_weights_temp)

  eigs = np.empty((nbnds,nkpnts,nspin), dtype=float)
  E_elem = root.find("./calculation/eigenvalues")
  for i, kpt in enumerate(E_elem.findall(".//set[@comment='spin 1']/set")):
    energy_at_k = []
    for e in kpt.findall("./r"):
      energy_at_k.append(float(e.text.strip().split()[0]))
    eigs[:,i,0] = energy_at_k
  if nspin == 2:
    for i, kpt in enumerate(E_elem.findall(".//set[@comment='spin 2']/set")):
      energy_at_k = []
      for e in kpt.findall("./r"):
        energy_at_k.append(float(e.text.strip().split()[0]))
      eigs[:,i,1] = energy_at_k
  eigs -= Efermi

  # Check metal or insulator
  # Metal if CBM and VBM are less than 0.05 eV apart
  cbm = np.min(eigs[eigs > 0])
  vbm = np.max(eigs[eigs < 0])
  insulator = (cbm-vbm) > 0.05

  # read MAGMOM
  # Note: the magnetic moments in vasprun.xml are the values defined in INCAR
  # The default is ferromagnetic
  magnetization = root.find(".//v[@name='MAGMOM']").text.strip().split()
  magmom = np.asarray([float(m) for m in magnetization])
  if dftSO:
    magmom = magmom.reshape((-1,3))
  finite_magmom = np.any(np.abs(magmom)>1e-3)
  dftMag = (nspin == 1 and dftSO and finite_magmom) or (nspin ==2 and finite_magmom)

  if nkpnts == nk1 * nk2 * nk3:
    # Check whether VASP calculation uses symmetry (ISYM = -1 or 2)
    ID = np.identity(3,dtype=int)
    sym_rot_transpose = ID[np.newaxis, :]
    shifts = np.zeros((1, 3))
    sym_info = np.asarray([None])
    time_rev = np.asarray([False])
    eq_atoms = np.arange(natoms)
    eq_atoms = eq_atoms[np.newaxis,:]

  else:
    #  get symmetry from spglib
    _, atom_numbers = np.unique(atoms, return_inverse=True)
    if dftMag:
      cell = (a_vectors, pos_arry, atom_numbers, magmom)
      spglib_sym = spglib.get_magnetic_symmetry(cell)
      sym_rot = spglib_sym['rotations']
      shifts = spglib_sym['translations']
      time_rev = spglib_sym['time_reversals']
      if dftSO:
        sym_rot = sym_rot[~time_rev,:,:]
        shifts = shifts[~time_rev,:]
        time_rev = np.asarray([False] * sym_rot.shape[0])
    else:
      cell = (a_vectors, pos_arry, atom_numbers)
      spglib_sym = spglib.get_symmetry(cell)
      sym_rot = spglib_sym['rotations']
      shifts = spglib_sym['translations']
      time_rev = np.asarray([False] * sym_rot.shape[0])


    # Find equivalent atoms
    # vasp/src/symlib.F SUBROUTINE POSMAP
    tol = 1e-5
    nops = sym_rot.shape[0]
    eq_atoms = np.zeros((nops, natoms), dtype=int)
    sym_rot_transpose = np.zeros((nops,3,3),dtype=int)
    for i_ops in range(nops):
      ops = sym_rot[i_ops,:,:]
      sym_rot_transpose[i_ops,:,:] = ops.T
      new_pos_arry = pos_arry @ ops.T + shifts[i_ops,:]
      for i_atm in range(natoms):
        for j_atm in range(natoms):
          diff_pos = new_pos_arry[j_atm,:] - pos_arry[i_atm, :]
          if np.linalg.norm(np.mod(diff_pos + 0.5, 1) - 0.5) < tol:
            eq_atoms[i_ops, j_atm] = i_atm

    if not np.all(np.sum(eq_atoms,axis=1) == natoms*(natoms-1)/2):
      raise Exception("Fail to find equivalent atoms")

    sym_info = np.asarray([None]*nops)

  if rank == 0 and verbose:
    print('Number of kpoints: {0:d}'.format(nkpnts))
    print('Number of electrons: {0:f}'.format(nelec))
    print('Number of bands: {0:d}'.format(nbnds))
    print('Insulator: {0}'.format(insulator))
    print('Magnetic: {0}'.format(dftMag))

  attrs = [('qe_version',qe_version),('alat',alat),('nk1',nk1),('nk2',nk2),('nk3',nk3),('ok1',k1),('ok2',k2),('ok3',k3),('natoms',natoms),('ecutrho',ecutrho),('ecutwfc',ecutwfc),('nawf',nawf),('nbnds',nbnds),('nspin',nspin),('nkpnts',nkpnts),('nelec',nelec),('Efermi',Efermi),('omega',omega),('dftSO',dftSO),('dftMAG',dftMag),('insulator',insulator)]
  for s,v in attrs:
    attr[s] = v

  spec = [(species[i],pseudos[i]) for i in range(len(species))]
  arrys = [('tau',tau),('atoms',atoms),('species',spec),('a_vectors',a_vectors),('b_vectors',b_vectors),('equiv_atom',eq_atoms),('kpnts',kpnts),('kpnts_wght',kpnt_weights),('my_eigsmat',eigs),('sym_rot',sym_rot_transpose),('sym_shift',shifts),('sym_info',sym_info),('sym_TR',time_rev)]
  for s,v in arrys:
    arry[s] = v


