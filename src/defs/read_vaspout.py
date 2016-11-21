#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed wder the terms of the
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
from __future__ import print_function
import numpy as np
import xml.etree.cElementTree as ET
import os,sys,time
import re

#units
Ry2eV   = 13.60569193
ang2bohr = 1.8897261

def read_VASP_output(fpath):
    from pymatgen.io.vasp.outputs import Vasprun
    from pymatgen.electronic_structure.core import Spin


    #Read vasprun.xml
    vasprun_file = os.path.join(fpath,"vasprun.xml")
    data = Vasprun(vasprun_file)
    
    a_vectors  =  data.lattice.matrix*ang2bohr
    b_vectors = data.lattice_rec.matrix/ang2bohr
    kpts_grid = data.kpoints.kpts[0]
    kpts_shift = data.kpoints.kpts_shift
    nspin =1
    if data.is_spin:
        print("Spin polarized calculation")
        nspin=2 #Check if spin-polarized calculation
    kpnts = np.array(data.actual_kpoints)
    nkpnts = len(kpnts)
    kpnts = kpnts.reshape((nkpnts,3))
    kpnts_wght = np.array(data.actual_kpoints_weights)
    eigenVals = data.eigenvalues
    
     
    #Format kpoints list to match QE ordering
    kpnts = np.c_[kpnts,np.arange(len(kpnts))] # Preserve initial index by adding as 4th column
    dt = [('col1', kpnts.dtype),('col2', kpnts.dtype),('col3', kpnts.dtype),('col4', kpnts.dtype)]
    assert kpnts.flags['C_CONTIGUOUS']
    kpnts=kpnts.ravel().view(dt)
    #Order kpoints by columns 1 and then by 2 and finally by 3
    kpnts.sort(order=['col1','col2','col3'])
    kpnts_qe=np.asarray([list(x) for x in kpnts])
    ik_vasp2qe = list(kpnts_qe[:,3])
    kpnts_qe = kpnts_qe[:,[0,2]]
    kpnts_wght_qe = np.zeros(nkpnts)
    for ik in range(nkpnts):
        ikqe = ik_vasp2qe.index(ik)
        kpnts_wght_qe[ik]=kpnts_wght[ikqe]
    
    print('reading vasprun.xml for kpoints and eigen values in ',time.clock(),' sec')
    reset=time.clock()

    if kpnts_wght.shape[0] != nkpnts:
        sys.exit('Error in size of the kpnts_wght vector') 

    #Format eigenvalues 
    nbnds = eigenVals[Spin.up].shape[1]
    my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
    for ik in range(nkpnts):
        ikqe = ik_vasp2qe.index(ik)
        my_eigsmat[:,ikqe,0]=eigenVals[Spin.up][ik][:,0]
        if nspin==2:my_eigsmat[:,ikqe,1]=eigenVals[Spin.down][ik][:,0]

    #Read OUTCAR to decide no.of orbitals
    outcar_lines=open(os.path.join(fpath,"OUTCAR"),'r').read()
    orbs = re.findall(".*VRHFIN\s*=\w+:\s*(\w+)",outcar_lines)
    nlm = 9
    for i in orbs:
        if "f" in i:nlm = 12 
    print("no. of atomic wfc", nlm)

    U = np.zeros((nbnds,nlm,nkpnts,nspin),dtype=complex)

    #Read PROOUT files -- PROOUT.1 and PROOUT.2 -- one for each spin channel
    for ispin in range(nspin):
        proout_file = os.path.join(fpath,"PROOUT.%d"%(ispin+1))
        if not os.path.exists(proout_file):
            print("%s FILE DOES NOT EXIST"%proout_file)
            raise SystemExit

        infile = file(proout_file,'r')
        infile.readline() #Skip 1st header line
        line = infile.readline().split()
        proout_nkpnts = int(float(line[3])); 
        if proout_nkpnts != nkpnts:
            print("K-POINT counts in PROOUT.%d and VASPRUN.xml do not match"%ispin)
            raise SystemExit
        nbnds = int(float(line[7])); 
        nions = int(float(line[-1]));
        line = infile.readline().split()
        ntypes = int(float(line[0]))
        nion_per_type=[int(float(x)) for x in line[2:]]

        #Skip k-point weigts -- already read from vasprun.xml
        for ib in range(nbnds):
            for ik in range(proout_nkpnts):
                infile.readline()

        proj_data = []
        for it in range(ntypes):
            for ik in range(proout_nkpnts):
                for iat in range(nion_per_type[it]):
                    for ib in range(nbnds):
                        proj_data += infile.readline().split()
        
        for it in range(ntypes):
            for ik in range(proout_nkpnts):
                for iat in range(nion_per_type[it]):
                    for ib in range(nbnds):
                        for ilm in range(nlm):
                            if len(proj_data) > 0:
                                real_val = float(proj_data.pop(0))
                                img_val = float(proj_data.pop(0))
                                #Use QE kpoint indexing 
                                ikqe = ik_vasp2qe.index(ik)
                                U[ib,ilm,ikqe,ispin] = real_val + 1j*img_val
        infile.close()
        print('reading projections from %s in '%proout_file, time.clock()-reset,' sec')
        reset=time.clock()
        

            
            

        
    
if __name__ == '__main__':
    read_VASP_output(sys.argv[1])

