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

from __future__ import print_function
import os, sys, traceback
import xml.etree.cElementTree as ET
import numpy as np
import re

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def read_attribute ( aroot, default_value, attr, atype, alen=1 ):
    txt = aroot.findall(attr)
    read_value = None
    if len(txt) == alen:
        if atype == 'logical':
            read_value = (str(txt[0].text) == 'T')
        elif atype == 'integer':
            read_value = int(txt[0].text)
        elif atype == 'decimal':
            read_value = float(txt[0].text)
        elif atype == 'array':
            ntxt = aroot.findall(attr+'/a')
            m = len(ntxt)
            if m > 0:
                n = len(ntxt[0].text.split())
                read_value = np.zeros((m,n), dtype=float)
                for i in range(m):
                    line = ntxt[i].text.split()
                    for j in range(n):
                        read_value[i,j] = line[j]

        elif atype == 'string_array':
            tmp_list = []
            ntxt = aroot.findall(attr+'/a')
            m = len(ntxt)
            if m > 0:
                n = len(ntxt[0].text)
                for i in range(m):
                    line = ntxt[i].text.split()
                    tmp_list.append(line)
            read_value = np.asarray(tmp_list,dtype=str)
            tmp_list = None

    if atype == 'string':
      if len(txt) == 1:
        read_value = str(txt[0].text)
      elif len(txt) > 0:
        ov = []
        read_value = [ov.append(str(t.text)) for t in txt]
    if read_value is not None:
        return read_value
    else:
        return default_value

def read_inputfile_xml ( fpath, inputfile ):

    fname = inputfile
    inputfile = os.path.join(fpath,fname)

    # Control
    fpath = None
    restart = verbose = non_ortho = write2file = write_binary = writedata = use_cuda = writez2pack = False
    shift_type = 1
    shift = 'auto'
    pthr = 0.95
    npool = 1

    # Return Request Definitions
    out_vals = ''

    # Calculation

    # Compare PAO bands with original DFT bands on the original MP mesh
    do_comparison = False

    # Dimensions of the atomic basis for each atom (order must be the same as in the output of projwfc.x)
    naw=np.array([[0,0]]) # naw.shape[0] = natom

    # Shell order and degeneracy for SO (order must be the same as in the output of projwfc.x)
    sh = np.array([[0,1,2,0,1,2]])    # order of shells with l angular momentum
    nl = np.array([[2,1,1,1,1,1]])    # multiplicity of each l shell

    # External fields
    Efield = np.array([[0,0,0]]) # static electric field (eV)
    Bfield = np.array([[0,0,0]]) # static magnetic firld placeholder: magnetic supercell not implemented!
    HubbardU = np.zeros(32,dtype=float) # non scf ACBN0 calculation
    HubbardU[1:4] = 0.0
    bval = 0 # top valence band number (nelec/2) to correctly shift eigenvalues

    # Bands interpolation along a path from a 1D string of k points
    onedim = False
    # Bands interpolation on a path from the original MP mesh 
    do_bands = False
    ibrav = 0
    # string of the band path
    band_path = ''
    # high symmetry point coordinates and label
    high_sym_points = np.array([[]])
    dkres = 0.1
    nk    = 2000
    # Band topology analysis
    band_topology = False
    eff_mass = False
    spol = 0  # spin
    ipol = 0
    jpol = 0

    # Construct PAO spin-orbit Hamiltonian
    do_spin_orbit = False
    theta = 0.0
    phi = 0.0
    lambda_p = 0.0
    lambda_d = 0.0

    # Hamiltonian interpolation on finer MP mesh
    double_grid = False
    nfft1 = 0
    nfft2 = 0
    nfft3 = 0

    # DOS(PDOS) calculation
    do_dos = False
    do_pdos = False
    emin = -10.
    emax = 2
    delta = 0.01

    # Adaptive smearing
    smearing = 'gauss' # other available values are None or 'm-p'

    # Plot Fermi Surface (spin texture)
    fermisurf = False
    fermi_up = 0.1
    fermi_dw = -0.1
    spintexture = False

    # Tensor components
    # Dielectric function
    d_tensor = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    # Boltzmann transport
    t_tensor = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    # Berry curvature
    a_tensor = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    # Spin Berry curvature
    s_tensor = np.array([[0,0,0],[0,1,0],[0,2,0],[1,0,0],[1,1,0],[1,2,0],[2,0,0],[2,1,0],[2,2,0], \
                  [0,0,1],[0,1,1],[0,2,1],[1,0,1],[1,1,1],[1,2,1],[2,0,1],[2,1,1],[2,2,1], \
                  [0,0,2],[0,1,2],[0,2,2],[1,0,2],[1,1,2],[1,2,2],[2,0,2],[2,1,2],[2,2,2]])

    # Set temperature in eV
    temp = 0.025852  # room temperature

    # Boltzmann transport calculation
    Boltzmann = False

    # Dielectric function calculation
    epsilon = False
    metal = False
    kramerskronig = True
    epsmin=0.0
    epsmax=10.0
    ne = 500

    # Critical points
    critical_points = False

    # Berry curvature and AHC
    Berry = False
    eminAH = -1.0
    emaxAH = 1.0
    ac_cond_Berry = False

    # Spin Berry curvature and SHC
    spin_Hall = False
    eminSH = -1.0
    emaxSH = 1.0
    ac_cond_spin = False


    tree = ET.parse(inputfile)
    aroot = tree.getroot()

    # Read String Input Values
    fpath = read_attribute(aroot, fpath, 'fpath', 'string')
    shift = read_attribute(aroot, shift, 'shift', 'string')
    try:
        shift = read_attribute(aroot, shift, 'shift', 'decimal')
    except:
        pass
    out_vals = read_attribute(aroot, out_vals, 'out_vals', 'string')
    smearing = read_attribute(aroot, smearing, 'smearing', 'string')
    if smearing == 'None': smearing = None
    band_path = read_attribute(aroot, band_path, 'band_path', 'string')


    # Read Logical Input Values
    restart = read_attribute(aroot, restart, 'restart', 'logical')
    writez2pack = read_attribute(aroot, writez2pack, 'writez2pack', 'logical')
    verbose = read_attribute(aroot, verbose, 'verbose', 'logical')
    non_ortho = read_attribute(aroot, non_ortho, 'non_ortho', 'logical')
    write2file = read_attribute(aroot, write2file, 'write2file', 'logical')
    write_binary = read_attribute(aroot, write_binary, 'write_binary', 'logical')
    writedata = read_attribute(aroot, writedata, 'writedata', 'logical')
    use_cuda = read_attribute(aroot, use_cuda, 'use_cuda', 'logical')
    do_comparison = read_attribute(aroot, do_comparison, 'do_comparison', 'logical')
    onedim = read_attribute(aroot, onedim, 'onedim', 'logical')
    do_bands = read_attribute(aroot, do_bands, 'do_bands', 'logical')
    band_topology = read_attribute(aroot, band_topology, 'band_topology', 'logical')
    eff_mass = read_attribute(aroot, eff_mass, 'eff_mass_topology', 'logical')
    do_spin_orbit = read_attribute(aroot, do_spin_orbit, 'do_spin_orbit', 'logical')
    double_grid = read_attribute(aroot, double_grid, 'double_grid', 'logical')
    do_dos = read_attribute(aroot, do_dos, 'do_dos', 'logical')
    do_pdos = read_attribute(aroot, do_pdos, 'do_pdos', 'logical')
    fermisurf = read_attribute(aroot, fermisurf, 'fermisurf', 'logical')
    spintexture = read_attribute(aroot, spintexture, 'spintexture', 'logical')
    Boltzmann = read_attribute(aroot, Boltzmann, 'Boltzmann', 'logical')
    epsilon = read_attribute(aroot, epsilon, 'epsilon', 'logical')
    metal = read_attribute(aroot, metal, 'metal', 'logical')
    kramerskronig = read_attribute(aroot, kramerskronig, 'kramerskronig', 'logical')
    critical_points = read_attribute(aroot, critical_points, 'critical_points', 'logical')
    Berry = read_attribute(aroot, Berry, 'Berry', 'logical')
    ac_cond_Berry = read_attribute(aroot, ac_cond_Berry, 'ac_cond_Berry', 'logical')
    spin_Hall = read_attribute(aroot, spin_Hall, 'spin_Hall', 'logical')
    ac_cond_spin = read_attribute(aroot, ac_cond_spin, 'ac_cond_spin', 'logical')


    # Read Integer Input Values
    shift_type = read_attribute(aroot, shift_type, 'shift_type', 'integer')
    npool = read_attribute(aroot, npool, 'npool', 'integer')
    bval = read_attribute(aroot, bval, 'bval', 'integer')
    ibrav = read_attribute(aroot, ibrav, 'ibrav', 'integer')
    nk = read_attribute(aroot, nk, 'nk', 'integer')
    spol = read_attribute(aroot, spol, 'spol', 'integer')
    ipol = read_attribute(aroot, ipol, 'ipol', 'integer')
    jpol = read_attribute(aroot, jpol, 'jpol', 'integer')
    nfft1 = read_attribute(aroot, nfft1, 'nfft1', 'integer')
    nfft2 = read_attribute(aroot, nfft2, 'nfft2', 'integer')
    nfft3 = read_attribute(aroot, nfft3, 'nfft3', 'integer')
    ne = read_attribute(aroot, ne, 'ne', 'integer')

    # Read Decimal Input Values
    pthr = read_attribute(aroot, pthr, 'pthr', 'decimal')
    dkres = read_attribute(aroot, dkres, 'dkres', 'decimal')
    theta = read_attribute(aroot, theta, 'theta', 'decimal')
    phi = read_attribute(aroot, phi, 'phi', 'decimal')
    lambda_p = read_attribute(aroot, lambda_p, 'lambda_p', 'decimal')
    lambda_d = read_attribute(aroot, lambda_d, 'lambda_d', 'decimal')
    emin = read_attribute(aroot, emin, 'emin', 'decimal')
    emax = read_attribute(aroot, emax, 'emax', 'decimal')
    delta = read_attribute(aroot, delta, 'delta', 'decimal')
    fermi_up = read_attribute(aroot, fermi_up, 'fermi_up', 'decimal')
    fermi_dw = read_attribute(aroot, fermi_dw, 'fermi_dw', 'decimal')
    temp = read_attribute(aroot, temp, 'temp', 'decimal')
    epsmin = read_attribute(aroot, epsmin, 'epsmin', 'decimal')
    epsmax = read_attribute(aroot, epsmax, 'epsmax', 'decimal')
    eminAH = read_attribute(aroot, eminAH, 'eminAH', 'decimal')
    emaxAH = read_attribute(aroot, emaxAH, 'emaxAH', 'decimal')
    eminSH = read_attribute(aroot, eminSH, 'eminSH', 'decimal')
    emaxSH = read_attribute(aroot, emaxSH, 'emaxSH', 'decimal')

    # Read Array Input Values
    naw = read_attribute(aroot, naw, 'naw', 'array')[0].astype(int)
    sh = [int(i) for i in read_attribute(aroot, sh, 'sh', 'array')[0]]
    nl = [int(i) for i in read_attribute(aroot, nl, 'nl', 'array')[0]]
    Efield = read_attribute(aroot, Efield, 'Efield', 'array')[0]
    Bfield = read_attribute(aroot, Bfield, 'Bfield', 'array')[0]
    HubbardU = read_attribute(aroot, HubbardU, 'HubbardU', 'array')[0]
    d_tensor = read_attribute(aroot, d_tensor, 'd_tensor', 'array').astype(int)
    t_tensor = read_attribute(aroot, t_tensor, 't_tensor', 'array').astype(int)
    a_tensor = read_attribute(aroot, a_tensor, 'a_tensor', 'array').astype(int)
    s_tensor = read_attribute(aroot, s_tensor, 's_tensor', 'array').astype(int)
    high_sym_points = read_attribute(aroot, high_sym_points, 'high_sym_points', 'string_array')


    return fpath,restart,verbose,non_ortho,write2file,write_binary,writedata,writez2pack,use_cuda,shift_type, \
        shift,pthr,npool,do_comparison,naw,sh,nl,Efield,Bfield,HubbardU,bval,onedim,do_bands, \
        ibrav,dkres,nk,band_topology,spol,ipol,jpol,do_spin_orbit,theta,phi,lambda_p,lambda_d, \
        double_grid,nfft1,nfft2,nfft3,do_dos,do_pdos,emin,emax,delta,smearing,fermisurf, \
        fermi_up,fermi_dw,spintexture,d_tensor,t_tensor,a_tensor,s_tensor,temp,Boltzmann, \
        epsilon,metal,kramerskronig,epsmin,epsmax,ne,critical_points,Berry,eminAH,emaxAH, \
        ac_cond_Berry,spin_Hall,eminSH,emaxSH,ac_cond_spin,eff_mass,out_vals.split(),band_path, \
        high_sym_points
