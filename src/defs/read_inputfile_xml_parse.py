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

import os
import re
import sys
import traceback
import numpy as np
import xml.etree.cElementTree as ET

def read_attribute ( aroot, default_value, attr, atype, alen=1 ):
    txt = aroot.findall(attr)
    read_value = None
    try:
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
                tmp_dict = {}
                ntxt = aroot.findall(attr+'/a')
                m = len(ntxt)
                if m > 0:
                  for i in range(m):
                    line = ntxt[i].text.split()
                    tmp_dict[line[0]] = np.asarray(line[1:], dtype=float)
                read_value = tmp_dict

        if atype == 'string':
          if len(txt) == 1:
            read_value = str(txt[0].text)
          elif len(txt) > 0:
            ov = []
            read_value = [ov.append(str(t.text)) for t in txt]
    except: pass
    if read_value is not None:
        return read_value
    else:
        return default_value

def read_inputfile_xml ( fpath, inputfile, data_controller ):

    fname = inputfile
    inputfile = os.path.join(fpath,fname)
    data_arrays = data_controller.data_arrays
    data_attributes = data_controller.data_attributes

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
    naw = np.array([[0,0]])

    # Shell order and degeneracy for SO (order must be the same as in the output of projwfc.x)
    sh = np.array([[]])    # order of shells with l angular momentum
    nl = np.array([[]])    # multiplicity of each l shell

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
    dkres = 0.1
    nk    = 2000
    band_path = None
    high_sym_points = np.array([[]])

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
    lambda_p = np.array([[0.0]])
    lambda_d = np.array([[0.0]])

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

    tmin = 300        # initial temperature in Kelvin
    tmax = 300      # highest temperature in Kelvin
    tstep= 1        # temperature step in Kelvin

    # Boltzmann transport calculation
    Boltzmann = False
    eminBT = -1.
    emaxBT = 1.

    # Evaluate the Carrier Concentration
    carrier_conc = False

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

    # symmetrization and wedge -> grid
    expand_wedge  = True
    symmetrize    = False
    symm_thresh   = 1.e-6
    symm_max_iter = 16

    tree = ET.parse(inputfile)
    aroot = tree.getroot()

    # Read String Input Values
    data_attributes['fpath'] = read_attribute(aroot, fpath, 'fpath', 'string')
    data_attributes['savedir'] = data_attributes['fpath']
    data_attributes['shift'] = read_attribute(aroot, shift, 'shift', 'string')
    try:
        data_attributes['shift'] = read_attribute(aroot, shift, 'shift', 'decimal')
    except:
        pass
    data_attributes['out_vals'] = read_attribute(aroot, out_vals, 'out_vals', 'string')
    smearing = read_attribute(aroot, smearing, 'smearing', 'string')
    if smearing == 'None':
        smearing = None
    data_attributes['smearing'] = smearing

    # Read Logical Input Values
    data_attributes['restart'] = read_attribute(aroot, restart, 'restart', 'logical')
    data_attributes['writez2pack'] = read_attribute(aroot, writez2pack, 'writez2pack', 'logical')
    data_attributes['verbose'] = read_attribute(aroot, verbose, 'verbose', 'logical')
    data_attributes['non_ortho'] = read_attribute(aroot, non_ortho, 'non_ortho', 'logical')
    data_attributes['write2file'] = read_attribute(aroot, write2file, 'write2file', 'logical')
    data_attributes['write_binary'] = read_attribute(aroot, write_binary, 'write_binary', 'logical')
    data_attributes['writedata'] = read_attribute(aroot, writedata, 'writedata', 'logical')
    data_attributes['use_cuda'] = read_attribute(aroot, use_cuda, 'use_cuda', 'logical')
    data_attributes['do_comparison'] = read_attribute(aroot, do_comparison, 'do_comparison', 'logical')
    data_attributes['onedim'] = read_attribute(aroot, onedim, 'onedim', 'logical')
    data_attributes['do_bands'] = read_attribute(aroot, do_bands, 'do_bands', 'logical')
    data_attributes['band_topology'] = read_attribute(aroot, band_topology, 'band_topology', 'logical')
    data_attributes['eff_mass'] = read_attribute(aroot, eff_mass, 'eff_mass_topology', 'logical')
    data_attributes['do_spin_orbit'] = read_attribute(aroot, do_spin_orbit, 'do_spin_orbit', 'logical')
    data_attributes['double_grid'] = read_attribute(aroot, double_grid, 'double_grid', 'logical')
    data_attributes['do_dos'] = read_attribute(aroot, do_dos, 'do_dos', 'logical')
    data_attributes['do_pdos'] = read_attribute(aroot, do_pdos, 'do_pdos', 'logical')
    data_attributes['fermisurf'] = read_attribute(aroot, fermisurf, 'fermisurf', 'logical')
    data_attributes['spintexture'] = read_attribute(aroot, spintexture, 'spintexture', 'logical')
    data_attributes['Boltzmann'] = read_attribute(aroot, Boltzmann, 'Boltzmann', 'logical')
    data_attributes['carrier_conc'] = read_attribute(aroot, carrier_conc, 'carrier_conc', 'logical')
    data_attributes['epsilon'] = read_attribute(aroot, epsilon, 'epsilon', 'logical')
    data_attributes['metal'] = read_attribute(aroot, metal, 'metal', 'logical')
    data_attributes['kramerskronig'] = read_attribute(aroot, kramerskronig, 'kramerskronig', 'logical')
    data_attributes['critical_points'] = read_attribute(aroot, critical_points, 'critical_points', 'logical')
    data_attributes['Berry'] = read_attribute(aroot, Berry, 'Berry', 'logical')
    data_attributes['ac_cond_Berry'] = read_attribute(aroot, ac_cond_Berry, 'ac_cond_Berry', 'logical')
    data_attributes['spin_Hall'] = read_attribute(aroot, spin_Hall, 'spin_Hall', 'logical')
    data_attributes['ac_cond_spin'] = read_attribute(aroot, ac_cond_spin, 'ac_cond_spin', 'logical')


    # Read Integer Input Values
    data_attributes['shift_type'] = read_attribute(aroot, shift_type, 'shift_type', 'integer')
    data_attributes['npool'] = read_attribute(aroot, npool, 'npool', 'integer')
    data_attributes['bval'] = read_attribute(aroot, bval, 'bval', 'integer')
    data_attributes['ibrav'] = read_attribute(aroot, ibrav, 'ibrav', 'integer')
    data_attributes['band_path'] = read_attribute(aroot, band_path, 'band_path', 'string')
    data_attributes['ne'] = read_attribute(aroot, ne, 'ne', 'integer')
    data_attributes['nk'] = read_attribute(aroot, nk, 'nk', 'integer')
    data_attributes['spol'] = read_attribute(aroot, spol, 'spol', 'integer')
    data_attributes['ipol'] = read_attribute(aroot, ipol, 'ipol', 'integer')
    data_attributes['jpol'] = read_attribute(aroot, jpol, 'jpol', 'integer')
    data_attributes['nfft1'] = read_attribute(aroot, nfft1, 'nfft1', 'integer')
    data_attributes['nfft2'] = read_attribute(aroot, nfft2, 'nfft2', 'integer')
    data_attributes['nfft3'] = read_attribute(aroot, nfft3, 'nfft3', 'integer')

    # Read Decimal Input Values
    data_attributes['pthr'] = read_attribute(aroot, pthr, 'pthr', 'decimal')
    data_attributes['dkres'] = read_attribute(aroot, dkres, 'dkres', 'decimal')
    data_attributes['theta'] = read_attribute(aroot, theta, 'theta', 'decimal')
    data_attributes['phi'] = read_attribute(aroot, phi, 'phi', 'decimal')
    data_attributes['emin'] = read_attribute(aroot, emin, 'emin', 'decimal')
    data_attributes['emax'] = read_attribute(aroot, emax, 'emax', 'decimal')
    data_attributes['delta'] = read_attribute(aroot, delta, 'delta', 'decimal')
    data_attributes['fermi_up'] = read_attribute(aroot, fermi_up, 'fermi_up', 'decimal')
    data_attributes['fermi_dw'] = read_attribute(aroot, fermi_dw, 'fermi_dw', 'decimal')
    data_attributes['temp'] = read_attribute(aroot, temp, 'temp', 'decimal')
    data_attributes['tmin'] = read_attribute(aroot, tmin, 'tmin', 'decimal')
    data_attributes['tmax'] = read_attribute(aroot, tmax, 'tmax', 'decimal')
    data_attributes['tstep'] = read_attribute(aroot, tstep, 'tstep', 'decimal')
    data_attributes['epsmin'] = read_attribute(aroot, epsmin, 'epsmin', 'decimal')
    data_attributes['epsmax'] = read_attribute(aroot, epsmax, 'epsmax', 'decimal')
    data_attributes['eminAH'] = read_attribute(aroot, eminAH, 'eminAH', 'decimal')
    data_attributes['emaxAH'] = read_attribute(aroot, emaxAH, 'emaxAH', 'decimal')
    data_attributes['eminSH'] = read_attribute(aroot, eminSH, 'eminSH', 'decimal')
    data_attributes['emaxSH'] = read_attribute(aroot, emaxSH, 'emaxSH', 'decimal')
    data_attributes['eminBT'] = read_attribute(aroot, eminBT, 'eminBT', 'decimal')
    data_attributes['emaxBT'] = read_attribute(aroot, emaxBT, 'emaxBT', 'decimal')

    # Read Array Input Values
    data_arrays['naw'] = read_attribute(aroot, naw, 'naw', 'array')[0].astype(int)
    data_arrays['sh'] = [int(i) for i in read_attribute(aroot, sh, 'sh', 'array')[0]]
    data_arrays['nl'] = [int(i) for i in read_attribute(aroot, nl, 'nl', 'array')[0]]
    if len(data_arrays['sh']) == 0 or len(data_arrays['sh']) != len(data_arrays['nl']):
      del data_arrays['sh']
      del data_arrays['nl']
    data_arrays['Efield'] = read_attribute(aroot, Efield, 'Efield', 'array')[0]
    data_arrays['Bfield'] = read_attribute(aroot, Bfield, 'Bfield', 'array')[0]
    data_arrays['HubbardU'] = read_attribute(aroot, HubbardU, 'HubbardU', 'array')[0]
    data_arrays['lambda_p'] = read_attribute(aroot, lambda_p, 'lambda_p', 'array')[0]
    data_arrays['lambda_d'] = read_attribute(aroot, lambda_d, 'lambda_d', 'array')[0]
    data_arrays['d_tensor'] = read_attribute(aroot, d_tensor, 'd_tensor', 'array').astype(int)
    data_arrays['t_tensor'] = read_attribute(aroot, t_tensor, 't_tensor', 'array').astype(int)
    data_arrays['a_tensor'] = read_attribute(aroot, a_tensor, 'a_tensor', 'array').astype(int)
    data_arrays['s_tensor'] = read_attribute(aroot, s_tensor, 's_tensor', 'array').astype(int)
    data_arrays['high_sym_points'] = read_attribute(aroot, high_sym_points, 'high_sym_points', 'string_array')

    # symmetrization and wedge -> grid
    data_attributes['expand_wedge'] = read_attribute(aroot, expand_wedge, 'expand_wedge', 'logical')
    data_attributes['symmetrize'] = read_attribute(aroot, symmetrize, 'symmetrize', 'logical')
    data_attributes['symm_thresh'] = read_attribute(aroot,symm_thresh, 'symm_thresh', 'decimal')
    data_attributes['symm_max_iter'] = read_attribute(aroot, symm_max_iter, 'symm_max_iter', 'integer')

