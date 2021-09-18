 PAOFLOW

 Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)

 Copyright 2016-2020 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)

 PAOFLOW is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

 You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


 PAOFLOW's capabilities:

 - Construction of PAO Hamiltonians from the DFT wavefunctions onto pseudo atomic orbitals
 - Construction of PAO Hamiltonians from analytical tight binding models
 - Hamiltonian data for further processing (ACBN0, PAOtransport, etc.)
 - External fields and non scf ACBN0 correction
 - Spin orbit correction of non SO calculations
 - Bands along standard paths in the BZ
 - Real space electronic charge density
 - Interpolation of Hamiltonians on arbitrary Monkhorst and Pack k-meshes
 - Adaptive smearing for BZ and Fermi surface integration
 - Density of states (and projected DOS)
 - Fermi surfaces and spin textures
 - Boltzmann transport (conductivity, Seebeck coefficient, electronic contribution to thermal conductivity
 - dielectric function (absorption coefficients and EELS)
 - Berry curvature and anomalous Hall conductivity (including magnetic circular dichroism spectra)
 - spin Berry curvature and spin Hall conductivity (including spin circular dichroism spectra) 
 - Band topology (Z2 invariants, Berry and spin Berry curvature along standard paths in BZ, critical points


Example code for PAOFLOW is available on GitHub:
https://github.com/marcobn/PAOFLOW/examples/

For installation instructions, see the INSTALL file.

 Use of PAOFLOW should reference:

 M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
 PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
 Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).


 PAOFLOW is integrated in AFLOW𝛑:

 A.R. Supka, T.E. Lyons, L. Liyanage, P. D'Amico, R. Al Rahal Al Orabi, S. Mahatara, P. Gopal, C. Toher, 
 D. Ceresoli, A. Calzolari, S. Curtarolo, M. Buongiorno Nardelli, and M. Fornari,
 AFLOW𝛑: A minimalist approach to high-throughput ab initio calculations including the generation 
 of tight-binding hamiltonians, Computational Materials Science, 136 (2017) 76-84. doi:10.1016/j.commatsci.2017.03.055
 also at www.aflow.org/src/aflowpi


Contributions to PAOFLOW were made by the following developers: Frank Cerasoli, Andrew Supka, Marcio Costa, Laalitha Liyanage, Haihang Wang, Anooja Jayaraj, Jagoda Slawinska, Priya Gopal, Ilaria Siloi


 Other references:

 Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
 Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).

 Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
 Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).

 Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
 Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).

 Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
 and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
 optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
 

