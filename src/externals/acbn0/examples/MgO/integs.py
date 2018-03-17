"""\

 Thu Jul 25 12:00:46 CDT 2013
 By Luis Agapito @ Marco Buongiorno Nardelli at UNT

 Based on:
 Ints.py Basic routines for integrals in the PyQuante framework

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""

'''
    #compounded basis for Si
    #basis_file 
    atno      =14
    #atlabel   ='Si'
    basis_file=('Si_s.py','Si_p.py')
    basis_file_path = '/Users/believe/Google Drive/unt2/xintegrals'
    basis = []
    for ibf in basis_file:
        execfile(basis_file_path+'/'+ibf)
        basis.append([ival for ikey,ival in cgto.iteritems()])
    mydict = {atno : basis}
    f = open(basis_file_path+'/'+'Si_basis.py','w+')
    f.write('basis_data' + '=' + repr(mydict) + '\n')
    f.close()

'''
from Molecule import Molecule
from pyints import ijkl2intindex

from pyints import contr_coulomb_v3 as pycc
#from cints import contr_coulomb_v3 as ccc
from time import time

##################################################
class myCGBF:
    "Class for a contracted Gaussian basis function"
    def __init__(self,origin,atid=0):
        self.origin = tuple([float(i) for i in origin])
        self.powers = [] 
        self.pnorms = [] #will always be one
        self.prims = []
        self.pnorms = []
        self.pexps = []
        self.pcoefs = []
        #self.ang_mom = sum(powers)
        self.atid = atid

##################################################
##################################################
def my_getbasis(atoms,basis_files_path):
    """\
    bfs = getbasis(atoms,basis_data=None)
    
    Given a Molecule object and a basis library, form a basis set
    constructed as a list of CGBF basis functions objects.
    """
    import os
    import sys
    import numpy as np
    import math  as m
    bfs = []
    for atom in atoms:
        print 'Building basis for atom ', atom.atid, atom.symbol()
        basis_data_file = basis_files_path+'/'+atom.symbol()+'_basis'
        if os.path.exists(basis_data_file+'.py'):
           print '\tFile '+basis_data_file+' found'
	else:
           print '\tFile '+basis_data_file+' not found. Exiting ...'
           sys.exit()
        exec('execfile("'+basis_data_file+'.py")')
        #print basis_data
        bs = basis_data[atom.atno]
        for shell in bs:
	    print '\t One shell (L) found'
            for subshell in shell:
                bf = myCGBF(atom.pos(),atom.atid)
                pgto_counter = 0
                for lx,ly,lz,coeff,zeta in subshell:
                    bf.powers.append((lx,ly,lz))
                    bf.pnorms.append(1.0)  #should always be 1. Kept for compatibility
                    ##gnorm only for testing with Gaussian09 basis, which adds it internally
                    #gnorm = (2*zeta/np.pi)**(3.0/4)*np.sqrt( 
                    #        (8*zeta)**(lx+ly+lz)*m.factorial(lx)*m.factorial(ly)*m.factorial(lz)/
                    #        m.factorial(2*lx)/m.factorial(2*ly)/m.factorial(2*lz) )
                    #bf.pcoefs.append(coeff*gnorm)
                    bf.pcoefs.append(coeff)
                    bf.pexps.append(zeta)
                    pgto_counter += 1
                print '\t\t One subshell (M) spanned with %d PGTOs found' % pgto_counter
                bfs.append(bf)
    return bfs

##################################################
def get2ints(bfs,coul_func):
    """Store integrals in a long array in the form (ij|kl) (chemists
    notation. We only need i>=j, k>=l, and ij <= kl"""
    from array import array
    print 'Calculationg 2e integrals using %s'%coul_func.__name__
    nbf = len(bfs)
    totlen = nbf*(nbf+1)*(nbf*nbf+nbf+2)/8
    Ints = array('d',[0]*totlen)
    for i in range(nbf):
        for j in range(i+1):
            ij = i*(i+1)/2+j
            for k in range(nbf):
                for l in range(k+1):
                    kl = k*(k+1)/2+l
                    if ij >= kl:
                        ijkl = ijkl2intindex(i,j,k,l)
                        Ints[ijkl] = coulomb(bfs[i],bfs[j],bfs[k],bfs[l],
                                             coul_func)
    return Ints

##################################################

def coulomb(a,b,c,d,coul_func):
    "Coulomb interaction between 4 contracted Gaussians"
    #print a.pexps, a.pcoefs,a.pnorms,a.origin,a.powers
    if coul_func.__name__ == 'contr_coulomb_v3':
       al=[];am=[];an=[]
       for x in a.powers:
           al.append(x[0]) 
           am.append(x[1]) 
           an.append(x[2]) 
       bl=[];bm=[];bn=[]
       for x in b.powers:
           bl.append(x[0]) 
           bm.append(x[1]) 
           bn.append(x[2]) 
       cl=[];cm=[];cn=[]
       for x in c.powers:
           cl.append(x[0]) 
           cm.append(x[1]) 
           cn.append(x[2]) 
       dl=[];dm=[];dn=[]
       for x in d.powers:
           dl.append(x[0]) 
           dm.append(x[1]) 
           dn.append(x[2]) 
       al=map(float,al);am=map(float,am);an=map(float,an)
       bl=map(float,bl);bm=map(float,bm);bn=map(float,bn)
       cl=map(float,cl);cm=map(float,cm);cn=map(float,cn)
       dl=map(float,dl);dm=map(float,dm);dn=map(float,dn)

       Jij = coul_func(a.pexps,a.pcoefs,a.pnorms,a.origin,al,am,an,
                       b.pexps,b.pcoefs,b.pnorms,b.origin,bl,bm,bn,
                       c.pexps,c.pcoefs,c.pnorms,c.origin,cl,cm,cn,
                       d.pexps,d.pcoefs,d.pnorms,d.origin,dl,dm,dn)
    else:
       Jij = coul_func(a.pexps,a.pcoefs,a.pnorms,a.origin,a.powers,
                       b.pexps,b.pcoefs,b.pnorms,b.origin,b.powers,
                       c.pexps,c.pcoefs,c.pnorms,c.origin,c.powers,
                       d.pexps,d.pcoefs,d.pnorms,d.origin,d.powers)
    return Jij

##################################################

##################################################
def test():
    #from PyQuante.Basis.sto3g import basis_data
    #from PyQuante.Basis.p631ss import basis_data
    #r = 1/0.52918
    #http://cccbdb.nist.gov/exp2.asp?casno=12597352
    #atoms=Molecule('si_dimer',atomlist = [('Si',(0,0,0)),('Si',(0,0,2.246))],units = 'Angstrom')
    #atoms=Molecule('si_atom',atomlist = [('Si',(0,0,0))],units = 'Angstrom')
    atoms=Molecule('li_atom',atomlist = [('Li',(0,0,0))],units = 'Angstrom')

    inttol = 1e-6 # Tolerance to which integrals must be equal
    
    basis_files_path = '/Users/believe/Google Drive/unt2/xintegrals'
    bfs = my_getbasis(atoms,basis_files_path)
    
    t0 = time()
    int0 = get2ints(bfs,pycc)
    t1 = time()
    print "time:    ",t1-t0, ' s'
    print 'Calculationg 2e integrals'
    mesg= """Store integrals in a long array in the form (ij|kl) (chemists
    notation. We only need i>=j, k>=l, and ij <= kl"""
    print mesg
    from array import array
    nbf = len(bfs)
    totlen = nbf*(nbf+1)*(nbf*nbf+nbf+2)/8
    Ints = array('d',[0]*totlen)
    for i in range(nbf):
        for j in range(i+1):
            ij = i*(i+1)/2+j
            for k in range(nbf):
                for l in range(k+1):
                    kl = k*(k+1)/2+l
                    if ij >= kl:
                        intval = int0[ijkl2intindex(i,j,k,l)]
                        if intval >= 1E-6:
                         print 'I= %d  J= %d  K= %d  L= %d  Int= %f' %(i+1,j+1,k+1,l+1,intval)
    return int0,bfs

#if __name__ == '__main__': test()

