#
# ACBN0.py
#
# Utility to construct the ACBN0 effective U values from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2014 Luis A. Agapito and Marco Buongiorno Nardelli, 2016,2017 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Stefano Curtarolo, and Marco Buongiorno Nardelli, Reformulation of DFT+U as a 
# Pseudohybrid Hubbard Density Functional for Accelerated Materials Discovery, Phys. Rev. X 5, 011006 (2015).
# 

import os
import sys
import scipy.io as sio
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla
from Molecule import Molecule
import logging
import integs
import time


try:
    from cints import contr_coulomb_v3 as ccc
except Exception,e:
    logging.warning('cints did not properly import. Switching to pyints.') 
    logging.warning(e)
    print 'cints did not properly import. Switching to pyints.'
    print e
    from pyints import contr_coulomb_v2 as ccc


def get_Nmm_spin(fpath,spin_label,Hks,Sks,kpnts_wght):
    f       = open(fpath+'/Nlm_k_file_'+spin_label,"rb")
    Nlm_k   = np.load(f)
    f.close()

    #short Fourier transform
    lm_size = Nlm_k.shape[0]
    nbasis  = Nlm_k.shape[1]
    nkpnts  = Nlm_k.shape[2]

    Nlm_aux = np.zeros((lm_size,nbasis),dtype=np.complex128)
    for nk in range(nkpnts):
        #kpnts are in units of 2*pi/alat. alat in Bohrs
        Nlm_aux = Nlm_aux + kpnts_wght[nk]*Nlm_k[:,:,nk]
    Nlm_aux = Nlm_aux/float(np.sum(kpnts_wght))
    Nlm_0 = np.sum(Nlm_aux,axis=1)
    print "get_Nmm_spin: Nlm_0 for spin = %s -->"%spin_label, Nlm_0.real
    return Nlm_0

def get_hartree_energy_spin(DR_0_up,DR_0_down,bfs,reduced_basis_2e,fpath):
    import sys
    import integs
    import numpy as np
    
    etemp_U = 0
    etemp_J = 0

    for mu in reduced_basis_2e:
        for nu in reduced_basis_2e:
            for kappa in reduced_basis_2e:
                for lamda in reduced_basis_2e:
                    #Da_01 = P^{alpha}_{mm'}
                    #Da_23 = P^{alpha}_{m''m'''}
                    Da_01 = DR_0_up[mu,nu]
                    Da_23 = DR_0_up[kappa,lamda]
                    Db_01 = DR_0_down[mu,nu]
                    Db_23 = DR_0_down[kappa,lamda]
                    myint_U = integs.coulomb(bfs[mu],bfs[nu],bfs[kappa],bfs[lamda],ccc) 
                    myint_J = integs.coulomb(bfs[mu],bfs[kappa],bfs[nu],bfs[lamda],ccc) #Pisani

                    etemp_U = etemp_U + (Da_01*Da_23 + Da_01*Db_23 + Db_01*Da_23 + Db_01*Db_23)*myint_U
                    etemp_J = etemp_J + (Da_01*Da_23 + Db_01*Db_23)*myint_J
    return etemp_U,etemp_J

def read_basis_unitcell(fpath,latvects,coords,atlabels):
    #nx range of cells in x axis. example nx=range(-2,3)

    import sys
    import integs

    #convert from numpy arrays to PyQuante list of tuples
    myatomlist = []
    for i,atomcoords in enumerate(coords):
        myatomlist.append( (atlabels[i].strip(),(atomcoords[0],atomcoords[1],atomcoords[2])) )
        
    print myatomlist
    atoms=Molecule('unitcell',atomlist = myatomlist,units = 'Angstrom')
    
    #inttol = 1e-6 # Tolerance to which integrals must be equal
    
    basis_file_path = fpath
    bfs = integs.my_getbasis(atoms,basis_file_path)
    print "Done generating bfs"
    return bfs

def write_reduced_Dk_spin_v2(fpath,reduced_basis_dm,reduced_basis_2e,spin_label,Hks,Sks):
    #v2 outputs the number of nocc mos
    #spin_label = "up","down","nospin"
    #Similar to write_reduced_Dk_in_k, but the DM is not reduced
    import scipy.io as sio
    from scipy import linalg as sla
    from numpy import linalg as la
    import numpy as np
    #print "write_reduced_Dk_spin_v2: Writing reduced DM(k) for spin=%s"%(spin_label)
    
    #The size is the full size of the basis
    nbasis  = Hks.shape[0]
    nkpnts  = Hks.shape[2]
    Dk      = np.zeros((nbasis,nbasis,nkpnts),dtype=np.complex128)
    lm_size_2e = reduced_basis_2e.shape[0]
    lm_size_dm = reduced_basis_dm.shape[0]
    Nlm_k   = np.zeros((lm_size_2e,nbasis,nkpnts),dtype=np.complex128)

    #Finding the density matrix at k
    for ik in range(nkpnts):
        #Mind that Hk has to be in nonorthogonal basis
        Hk = Hks[:,:,ik]
        Sk = Sks[:,:,ik]
        #ss = la.inv(sla.sqrtm(Sk)) #S^{-1/2}
        #Hk = Hk.T
        #Sk = Sk.T
        w,v =sla.eigh(Hk,Sk) #working with the transposes
        #arranging the eigs
        evals     =np.sort(w)
        evecs     =v[:,w.argsort()]
        
        smearing = 0.0; #change it to +/- 0.0001, or so, if you need .
        indexes  = np.where(evals <=0+smearing)[0] 
        nocc_mo  = indexes.shape[0]
        if ik==0:
           nocc_mo_at_gamma = indexes.shape[0]
           #print "write_reduced_Dk_spin_v2: nocc orbs at Gamma= %d. Abort if not right."%nocc_mo
        #else:
        #   if nocc_mo_at_gamma != nocc_mo:
        #      print "write_reduced_Dk_spin_v2: Number of occ. orbs changed at kpoint %d w.r.t Gamma, to %d. "%(ik,nocc_mo)
        occ_indexes = indexes[:nocc_mo]
        
        #Computing the density matrix.
        #n belong to indexes, indexes the occupied MOs
        #D_uv = sum_n c_{un}^{*} . c_vn   
       
        #the basis lm is determined by the input reduced_basis 
        #nocc_mo = indexes.shape[0] #number of occupied MOs

        #lm charge decomposition of each occupied band
        n_lm_dm = np.zeros((lm_size_dm,nocc_mo),dtype=np.complex128) 
        n_lm_2e = np.zeros((lm_size_2e,nocc_mo),dtype=np.complex128) 

        for i_mo in range(nocc_mo):
            cv = evecs[:,i_mo]  #the occupied MOs are in ascending eig order
            #n_L[i_mo] = np.vdot(cv[reduced_basis],Sk[reduced_basis,:].dot(cv))
            n_lm_dm[:,i_mo] = np.conj(cv[reduced_basis_dm]) * (Sk[reduced_basis_dm,:].dot(cv))
            n_lm_2e[:,i_mo] = np.conj(cv[reduced_basis_2e]) * (Sk[reduced_basis_2e,:].dot(cv))
        Nlm_k[:,:nocc_mo,ik]=n_lm_2e

        for uu in range(nbasis):
            for vv in range(nbasis):
                Dk[uu,vv,ik] = np.vdot(evecs[uu,occ_indexes],np.sum(n_lm_dm,0)*evecs[vv,occ_indexes]) 


    #print "write_reduced_Dk_spin_v2: number of occ. orbitals at Gamma for spin=%s is %d\n"%(spin_label,nocc_mo_at_gamma)

    f = open(fpath+'/Dk_reduced_file_'+spin_label,"wb")
    np.save(f,Dk)
    f.close()
     
    #this is not a reduced quantity
    f = open(fpath+'/Nlm_k_file_'+spin_label,"wb")
    np.save(f,Nlm_k)
    f.close()
    return nocc_mo_at_gamma

def read_txtdata(fpath,nspin):
    #nspin = 1; non-spin-polarized case
    #nspin = 2; spin-polarized case
    import numpy as np
    fin   = open(fpath+'/'+'wk.txt',"r")
    kpnts_wght = np.loadtxt(fin)
    fin.close

    fin   = open(fpath+'/'+'k.txt',"r")
    kpnts = np.loadtxt(fin)
    fin.close
    nkpnts  = kpnts.shape[0]
    print "read_txt_data: number of kpoints = %d"%nkpnts

    fin   = open(fpath+'/'+'kovp.txt',"r")
    kovp_0 = np.loadtxt(fin)
    fin.close
    nbasis  = int(np.sqrt(kovp_0.shape[0]/float(nkpnts)))
    print "read_txt_data: nbasis = %s"%nbasis
    kovp_1  = kovp_0[:,0]+1j*kovp_0[:,1]
    kovp    = np.reshape(kovp_1,(nbasis,nbasis,nkpnts),order='F')

    for ispin in range(nspin):
        if ispin==0 and nspin==2 : 
           fname = 'kham_up.txt'
        elif ispin==1 and nspin==2 :
           fname = 'kham_down.txt'
        elif ispin==0 and nspin==1 :
           fname = 'kham.txt'
        else :
           print 'wrong case 1'
        fin    = open(fpath+'/'+fname,"r")
        kham_0 = np.loadtxt(fin)
        fin.close
        kham_1 = kham_0[:,0]+1j*kham_0[:,1]
        kham   = np.reshape(kham_1,(nbasis,nbasis,nkpnts),order='F')
        if ispin==0 and nspin==2 : 
           kham_up   = kham
        elif ispin==1 and nspin==2 :
           kham_down = kham
        elif ispin==0 and nspin==1 :
           kham_nospin = kham
        else :
           print 'wrong case 2'
    
    if nspin == 1: 
       f = open(fpath+'/Hk_nospin',"wb")
       np.save(f,kham_nospin)
       f.close()
       f = open(fpath+'/Sk',"wb")
       np.save(f,kovp)
       f.close()
       return nkpnts,kpnts,kpnts_wght,kovp,kham_nospin
    elif nspin == 2: 
       f = open(fpath+'/Hk_up',"wb")
       np.save(f,kham_up)
       f.close()
       f = open(fpath+'/Hk_down',"wb")
       np.save(f,kham_down)
       f.close()
       f = open(fpath+'/Sk',"wb")
       np.save(f,kovp)
       f.close()
       return nkpnts,kpnts,kpnts_wght,kovp,kham_up,kham_down
    else:
       print "wrong case 3"

def get_DR_0_spin(fpath,spin_label,kpnts_wght):
    import numpy as np
    import scipy.io as sio 
    
    #Creating storage for all the infinite H(R)
    #nneighs=length(nx)*length(ny)*length(nz);
    f = open(fpath +'/Dk_reduced_file_'+spin_label,"rb")
    Dk      = np.load(f);
    f.close()
    print "get_DR_0_spin: Dk reduced spin %s found, shaped %d x %d x %d"%(spin_label,Dk.shape[0],Dk.shape[1],Dk.shape[2])
    
    nkpnts     =kpnts_wght.shape[0]
    print "get_DR_0_spin: number of kpoints %d"%nkpnts

    #Overwrite nawf, in case masking of awfc was use
    nawf = Dk.shape[0]
    print "get_DR_0_spin: number of basis %d"%nawf
    print "get_DR_0_spin: total kpoints weight %f"%np.sum(kpnts_wght)
    
    D = np.zeros((nawf,nawf),dtype=np.complex128)
    for nk in range(nkpnts):
        D = D + kpnts_wght[nk]*Dk[:,:,nk]
    D = D/float(np.sum(kpnts_wght))
    
    f = open(fpath+'/DR_0_reduced_file_'+spin_label,"wb")
    np.save(f,D.real)
    f.close()
    return D.real

def test(fpath,reduced_basis_dm,reduced_basis_2e,latvects,coords,atlabels,outfile):
    import os
    import integs
    fout = open(fpath+"/"+outfile, "w")
    fout.close()
    fout = open(fpath+"/"+outfile, "r+")
    fout.write("**********************************************************************\n")
    fout.write("* test_dm_solids_spin.py                                             *\n") 
    fout.write("* Computes on-site HF Coulomb + Exchange parameters                  *\n")
    fout.write("* Luis Agapito and Marco Buongiorno-Nardelli, UNT Physics            *\n")
    fout.write("* January 2014                                                       *\n")
    fout.write("**********************************************************************\n")
    fout.write("fpath:        %s\n"%fpath)
    fout.write("outfile:      %s\n"%outfile)
    fout.write("reduced_basis_dm:%s\n"%reduced_basis_dm)
    fout.write("reduced_basis_de:%s\n"%reduced_basis_2e)
    fout.write("latvects:     %s\n"%str(latvects))
    fout.write("coords:       %s\n"%str(coords))
    fout.write("atlabels:     %s\n"%str(atlabels))

    Ha2eV     = 27.211396132 
    Bohr2Angs =  0.529177249

    #%%
    print "Generate PyQuante instance of the BFS class"
    bfs     = read_basis_unitcell(fpath,latvects,coords,atlabels)
    nbasis  = len(bfs)
    fout.write("PyQuante: Number of basis per prim cell is %d\n"%nbasis)

    ######################################################################
    fout.write('Reading the PAO data\n')
    print('Reading the PAO data')
    if nspin == 1: 
       nkpnts,kpnts,kpnts_wght,Sks,Hks_nospin = read_txtdata(fpath,nspin)
    elif nspin == 2: 
       nkpnts,kpnts,kpnts_wght,Sks,Hks_up,Hks_down = read_txtdata(fpath,nspin)
    else:
       print 'wrong case 1'
    

    fout.write('Calculating Nlm_k and reduced D_k''s\n')
    print('Calculating Nlm_k and reduced D_k''s')
    if nspin == 1: 
       nocc_mo_gamma = write_reduced_Dk_spin_v2(fpath,reduced_basis_dm,reduced_basis_2e,'nospin',Hks_nospin,Sks)
    elif nspin == 2: 
       nocc_mo_gamma = write_reduced_Dk_spin_v2(fpath,reduced_basis_dm,reduced_basis_2e,'up',Hks_up,Sks)
       nocc_mo_gamma = write_reduced_Dk_spin_v2(fpath,reduced_basis_dm,reduced_basis_2e,'down',Hks_down,Sks)
    else:
       print 'wrong case 2'

    fout.write('Calculating Nlm_0\n')
    print('Calculating Nlm_0')
    if nspin == 1: 
       Nlm_0_nospin = get_Nmm_spin(fpath,'nospin',Hks_nospin,Sks,kpnts_wght)
    elif nspin == 2: 
       Nlm_0_up     = get_Nmm_spin(fpath,'up'    ,Hks_up    ,Sks,kpnts_wght)
       Nlm_0_down   = get_Nmm_spin(fpath,'down'  ,Hks_down  ,Sks,kpnts_wght)
    else:
       print 'wrong case 3'
    
    if nspin == 1: 
       Naa=0.0
       lm_size = Nlm_0_nospin.shape[0]
       for m in range(lm_size):
           for mp in range(lm_size):
               if mp == m:
                  continue
               else:
                  Naa = Naa + Nlm_0_nospin[m]*Nlm_0_nospin[mp]
       Nab=0.0
       for m in range(lm_size):
           for mp in range(lm_size):
               Nab = Nab + Nlm_0_nospin[m]*Nlm_0_nospin[mp]
    elif nspin == 2 :
        Naa=0.0
        lm_size = Nlm_0_up.shape[0]
        for m in range(lm_size):
            for mp in range(lm_size):
                if mp == m:
                   continue
                else:
                   Naa = Naa + Nlm_0_up[m]*Nlm_0_up[mp]
        Nbb=0.0
        lm_size = Nlm_0_down.shape[0]
        for m in range(lm_size):
            for mp in range(lm_size):
                if mp == m:
                   continue
                else:
                   Nbb = Nbb + Nlm_0_down[m]*Nlm_0_down[mp]
        Nab=0.0
        for m in range(lm_size):
            for mp in range(lm_size):
                Nab = Nab + Nlm_0_up[m]*Nlm_0_down[mp]
    else:
       print 'wrong case 4'

    if nspin == 1: 
       print "NaNa + NaNb + NbNa + Nbb = %f"%(2*Nab.real+2*Naa.real)
       denominator_U = 2*Nab.real+2*Naa.real
       denominator_J = 2*Naa.real
    elif nspin == 2: 
       print "NaNa + NaNb + NbNa + Nbb = %f"%(2*Nab.real+Naa.real+Nbb.real)
       denominator_U = 2*Nab.real+Naa.real+Nbb.real
       denominator_J = Naa.real+Nbb.real
    else:
       print 'wrong case'
    fout.write("denominator_U = %f\ndenominator_J = %f\n"%(denominator_U,denominator_J))
    print("denominator_U = %f\ndenominator_J = %f"%(denominator_U,denominator_J))


    print "Finding the Coulomb and exchange energies"
    fout.write("Started finding the Coulomb and exchange energies at %s\n"%(time.ctime()))

    ta = time.time()
    if  nspin == 1:
        DR_0_up   = get_DR_0_spin(fpath,'nospin',kpnts_wght)
        DR_0_down = DR_0_up 
    if  nspin == 2:
        DR_0_up   = get_DR_0_spin(fpath,'up',kpnts_wght)
        DR_0_down = get_DR_0_spin(fpath,'down',kpnts_wght)

    
    fout.flush()

    t0   = time.time() 
    U_energy,J_energy = get_hartree_energy_spin(DR_0_up,DR_0_down,bfs,reduced_basis_2e,fpath)
    t1   = time.time() 
    print("Energy Uaa=%+14.10f Ha; Energy Jaa=%+14.10f Ha; %7.3f s"%(U_energy,J_energy,t1-t0))
    fout.write("Energy Uaa=%+14.10f Ha; Energy Jaa=%+14.10f Ha; %7.3f s\n"%(U_energy,J_energy,t1-t0))

    SI = 0

    U = (U_energy -2*SI)/denominator_U
    J = (J_energy -2*SI)/denominator_J

    print("Parameter U=%f eV"%(U*Ha2eV))
    fout.write("Parameter U=%f eV\n"%(U*Ha2eV))
    print("Parameter J=%f eV"%(J*Ha2eV))
    fout.write("Parameter J=%f eV\n"%(J*Ha2eV))
    
    if J*Ha2eV == float('Inf'):
        print("Parameter U_eff = %f eV"%(U*Ha2eV))
        fout.write("Parameter U_eff = %f eV\n"%(U*Ha2eV))
    else:
        print("Parameter U_eff = %f eV"%((U-J)*Ha2eV))
        fout.write("Parameter U_eff = %f eV\n"%((U-J)*Ha2eV))


    tb = time.time()

    fout.write("Finished finding the Coulomb energy at %s, elapsed %f s\n"%(time.ctime(),tb-ta))
    fout.close() 

if __name__ == '__main__':
    Bohr2Angs =  0.529177249
    inputfile = sys.argv[1]
    input_data = {}
    f = open(inputfile)
    data = f.readlines()
    for line in data:
        line = line.strip()
        if line and not line.startswith("#"):
           line = line.strip()
           # parse input, assign val=es to variables
           #print line.split("=")
           key, value = line.split("=")
           input_data[key.strip()] = value.strip()
    f.close()
    
    
    fpath         = input_data['fpath']
    outfile       = input_data['outfile']
    nspin         = int(input_data['nspin'])
    reduced_basis_dm = np.fromstring(input_data['reduced_basis_dm'], dtype=int, sep=',' ) 
    reduced_basis_2e = np.fromstring(input_data['reduced_basis_2e'], dtype=int, sep=',' ) 
    latvects      = np.fromstring(input_data['latvects'], dtype=float, sep=',' ) 
    latvects      = np.reshape(latvects,(3,3))*Bohr2Angs
    atlabels      = input_data['atlabels']
    atlabels      = atlabels.strip(",").split(",")
    coords        = np.fromstring(input_data['coords'], dtype=float, sep=',' ) 
    coords        = np.reshape(coords,(-1,3))*Bohr2Angs

    test(fpath,reduced_basis_dm,reduced_basis_2e,latvects,coords,atlabels,outfile)

#input file
#   cat << EOF > input_auto.txt
#   ##lattice vectors in Bohrs. Use "\" and ","
#   #latvects =    \
#   #3.023110815998280E+000, -5.236012660047771E+000, 0.000000000000000E+000,\
#   #3.023110815998280E+000,  5.236012660047771E+000, 0.000000000000000E+000, \
#   #0.000000000000000E+000,  0.000000000000000E+000, 9.699892749717790E+000 
#   #
#   ##coordinates in Bohrs 
#   #coords =    \
#   #3.023110815998281E+000,  1.745305002814972E+000,  3.701706649346625E+000,\
#   #3.023110815998281E+000, -1.745305002814972E+000,  8.551653024205519E+000,\
#   #3.023110815998280E+000,  1.745304597721543E+000,  8.638125918931237E-003,\
#   #3.023110815998280E+000, -1.745304597721543E+000,  4.858584500777826E+000
#   #
#   ##atomic labels 
#   #atlabels = O , O , Zn, Zn 
#
#   latvects = \
#   3.965000000000000E+000,  3.965000000000000E+000,  7.930000000000000E+000,\
#   3.965000000000000E+000,  7.930000000000000E+000,  3.965000000000000E+000,\
#   7.930000000000000E+000,  3.965000000000000E+000,  3.965000000000000E+000
#    
#   coords = \
#   3.965000000000000E+000, 3.965000000000000E+000, 3.965000000000000E+000,\
#   1.189500000000000E+001, 1.189500000000000E+001, 1.189500000000000E+001,\
#   0.000000000000000E+000, 0.000000000000000E+000, 0.000000000000000E+000,\
#   7.930000000000000E+000, 7.930000000000000E+000, 7.930000000000000E+000
#   
#   atlabels = O, O, Ni, Ni
#
#   #non-spin-polarized DFT => 1
#   #spin-polarized DFT => 2
#   nspin         = 2 
#   fpath = /Users/believe/unt/tests/nio_project/nio/nscf_9x9x9/
#   outfile = outfile_Ni
#   #outfile = outfile_O
#   
#   #coordinates of the basis in R=[0 0 0]
#
#   #Ni1 d
#   reduced_basis = 9,10,11,12,13
#   
#   #O1 p
#   #reduced_basis = 1,2,3 
#   EOF
#
#   ipython test_dm_solids_spin.py input_auto.txt
