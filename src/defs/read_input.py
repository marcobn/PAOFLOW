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
#
import numpy as np
import sys
import re

def read_input(input_file):

    non_ortho  = False
    shift_type = 2
    shift      = 20
    pthr       = 0.9
    do_comparison = False
    double_grid = False
    do_bands = False
    onedim = False
    do_dos = False
    do_spin_orbit = False
    nfft1 = 0
    nfft2 = 0
    nfft3 = 0
    ibrav = 0
    dkres = 0.1
    Boltzmann = False
    epsilon = False
    eff_mass = False

    f = open(input_file)
    lines=f.readlines()
    f.close
    for line in lines:
        line = line.strip()
        if re.search('fpath',line):
            p = line.split()
            fpath = p[1]
        if re.search('non_ortho',line):
            p = line.split()
            non_ortho = p[1]
            if non_ortho == 'False':
                non_ortho = (1 == 2)
            else:
                non_ortho = (1 == 1)
        if re.search('do_comparison',line):
            p = line.split()
            do_comparison = p[1]
            if do_comparison == 'False':
                do_comparison = (1 == 2)
            else:
                do_comparison = (1 == 1)
        if re.search('double_grid',line):
            p = line.split()
            double_grid = p[1]
            if double_grid == 'False':
                double_grid = (1 == 2)
            else:
                double_grid = (1 == 1)
        if re.search('do_bands',line):
            p = line.split()
            do_bands = p[1]
            if do_bands == 'False':
                do_bands = (1 == 2)
            else:
                do_bands = (1 == 1)
        if re.search('onedim',line):
            p = line.split()
            onedim = p[1]
            if onedim == 'False':
                onedim = (1 == 2)
            else:
                onedim = (1 == 1)
        if re.search('do_dos',line):
            p = line.split()
            do_dos = p[1]
            if do_dos == 'False':
                do_dos = (1 == 2)
            else:
                do_dos = (1 == 1)
        if re.search('delta',line):
            p = line.split()
            delta = float(p[1])
        if re.search('do_spin_orbit',line):
            p = line.split()
            do_spin_orbit = p[1]
            if do_spin_orbit == 'False':
                do_spin_orbit = (1 == 2)
            else:
                do_spin_orbit = (1 == 1)
        if re.search('shift_type',line):
            p = line.split()
            shift_type = int(p[1])
        if re.search('shift',line):
            p = line.split()
            shift = float(p[1])
        if re.search('pthr',line):
            p = line.split()
            pthr = float(p[1])
        if re.search('nfft123',line):
            p = line.split()
            nfft1 = int(p[1])
            nfft2 = int(p[2])
            nfft3 = int(p[3])
        if re.search('ibrav',line):
            p = line.split()
            ibrav = int(p[1])
        if re.search('dkres',line):
            p = line.split()
            dkres = float(p[1])
        if re.search('Boltzmann',line):
            p = line.split()
            Boltzmann = p[1]
            if Boltzmann == 'False':
                Boltzmann = (1 == 2)
            else:
                Boltzmann = (1 == 1)
        if re.search('epsilon',line):
            p = line.split()
            epsilon = p[1]
            if epsilon == 'False':
                epsilon = (1 == 2)
            else:
                epsilon = (1 == 1)
        if re.search('eff_mass',line):
            p = line.split()
            eff_mass = p[1]
            if eff_mass == 'False':
                eff_mass = (1 == 2)
            else:
                eff_mass = (1 == 1)
    if fpath == '':
        sys.exit('missing path to _.save')

    return(non_ortho, shift_type, fpath, shift, pthr, do_comparison, double_grid, \
            do_bands, onedim, do_dos, delta, do_spin_orbit, nfft1, nfft2, nfft3, \
            ibrav, dkres, Boltzmann, epsilon, eff_mass)
