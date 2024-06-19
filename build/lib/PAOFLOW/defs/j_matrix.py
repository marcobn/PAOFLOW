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
# Contructs the Total Angular Momentum Matrices
def j_matrix ( data_controller, spol):
  import numpy as np
  from re import findall
  from .read_sh_nl import read_pseudopotential 
  from os.path import join,exists


  arry = data_controller.data_arrays
  attr = data_controller.data_attributes

  atom =  np.zeros(attr['nawf'],dtype=int)
  wfc  =  np.zeros(attr['nawf'],dtype=int)
  l    =  np.zeros(attr['nawf'],dtype=int)
  j    =  np.zeros(attr['nawf'],dtype=float)
  mj   =  np.zeros(attr['nawf'],dtype=float)


  natom=0
  nwfc=0
  nnwfc=0

  for a in arry['atoms']:
      for p in range (len(arry['jchia'][a])):
        mj_max=float(arry['jchia'][a][p])
        mj_aux = 0.0 ; n=0
        while (mj_aux < mj_max):

            wfc[nwfc]=nnwfc

            mj_aux=-arry['jchia'][a][p]+n
            mj[nwfc]=mj_aux

            n+=1
            atom[nwfc]=natom
            l[nwfc]=arry['lchia'][a][p]
            j[nwfc]=arry['jchia'][a][p]
            nwfc+=1

        nnwfc+=1
      natom+=1

 
  Tj = np.zeros((attr['nawf'],attr['nawf']),dtype=complex)


  # Building Jx matrix
  if(spol==0):
    for i in range(attr['nawf']):
      for k in range(attr['nawf']):

        if(atom[i]==atom[k]): # Delta on atoms
          if(wfc[i]==wfc[k]): # Delta on orbitails
            if(l[i]==l[k]):   # Delta on l-l'
              if(j[i]==j[k]): # Delta on j-j'

                if(mj[i]==(mj[k]-1)): # Delta mj'-(mj-1)
                  Tj[i,k]+=0.5*np.sqrt(j[k]*(j[k]+1) - mj[k]*(mj[k]-1))

                if(mj[i]==(mj[k]+1)): # Delta  mj'-(mj+1)
                  Tj[i,k]+=0.5*np.sqrt(j[k]*(j[k]+1) - mj[k]*(mj[k]+1))


  # Building Jy matrix
  if(spol==1):
    for i in range(attr['nawf']):
      for k in range(attr['nawf']):

        if(atom[i]==atom[k]): # Delta on atoms
          if(wfc[i]==wfc[k]): # Delta on orbitails
            if(l[i]==l[k]):   # Delta on l-l'
              if(j[i]==j[k]): # Delta on j-j'

                if(mj[i]==(mj[k]-1)): # Delta mj'-(mj-1)
                  Tj[i,k]+=0.5j*np.sqrt(j[i]*(j[i]+1) - mj[k]*(mj[k]-1))

                if(mj[i]==(mj[k]+1)): # Delta  mj'-(mj+1)
                  Tj[i,k]-=0.5j*np.sqrt(j[i]*(j[i]+1) - mj[k]*(mj[k]+1))

  #Building Jz matrix
  if(spol==2):
    for i in range(attr['nawf']):
      Tj[i,i]=mj[i]

  return Tj 
