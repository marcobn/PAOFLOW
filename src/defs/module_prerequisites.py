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

report_pre_reqs = 'SUGGESTION: %s must be called before %s'

module_pre_reqs = { 'add_external_fields'      : ['projectability'],\
                    'pao_hamiltonian'          : ['projectability'],\
                    'z2_pack'                  : ['pao_hamiltonian'],\
                    'atomic_orbitals'           : ['pao_hamiltonian'], \
                    'doubling_Hamiltonian'     : ['pao_hamiltonian'],\
                    'cutting_Hamiltonian'      : ['pao_hamiltonian'],\
                    'wave_function_projection' : ['pao_hamiltonian'],\
                    'bands'                    : ['pao_hamiltonian'],\
                    'topology'                 : ['bands', 'spin_operator'],\
                    'interpolated_hamiltonian' : ['pao_hamiltonian'],\
                    'pao_eigh'                 : ['pao_hamiltonian'],\
                    'dos'                      : ['pao_eigh'],\
                    'fermi_surface'            : ['pao_eigh'],\
                    'spin_texture'             : ['pao_eigh'],
                    'gradient_and_momenta'     : ['pao_eigh'],\
                    'ipr'                      : ['pao_eigh'],\
                    'berry_phase'              : ['pao_hamiltonian'],\
                    'real_space_wfc'           : ['gradient_and_momenta', 'atomic_orbitals'], \
                    'adaptive_smearing'        : ['gradient_and_momenta'],\
                    'anomalous_Hall'           : ['gradient_and_momenta'],\
                    'spin_Hall'                : ['gradient_and_momenta', 'spin_operator'],\
                    'transport'                : ['gradient_and_momenta'],\
                    'effective_mass'           : ['gradient_and_momenta'],\
                    'dielectric_tensor'        : ['gradient_and_momenta'] }


key_error_strings = { 'U' : 'Projections must be computed before projectability. Call \'read_atomic_proj_QE\' to read projections from the output of projwfc.x',\
                      'HRs'  : 'HRs must first be calculated with \'build_pao_hamiltonian\'',\
                      'd2Ed2k'  : 'd2Ed2k must first be calculated using band_curvature set to True in \'gradient_and_momenta\'',\
                      'Hks'  : 'Hks must first be calculated with \'build_pao_hamiltonian\''}
