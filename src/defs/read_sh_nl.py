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

def read_sh_nl ( data_controller ):
  '''
  Determines the shell configuration for based on the referenced QE .save directory

  Arguments:
      data_controller (DataController) - The DataController being initialized to the referenced .save directory

  Returns:
      sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                       nl is a list of occupations at each site
      sh and nl are representative of the entire system
  '''
  from os.path import join,exists




  arry,attr = data_controller.data_dicts()

  # Get Shells for each species
  sdict = {}
  for s in arry['species']:
    sdict[s[0]] = read_pseudopotential(join(attr['workpath'],attr['savedir'],s[1]))

  # Concatenate shells for each atom
  sh = []
  nl = []
  for a in arry['atoms']:
    sh += sdict[a][0]
    nl += sdict[a][1]


  return(sh, nl)


def read_pseudopotential ( fpp ):
  '''
  Reads a psuedopotential file to determine the included shells and occupations.

  Arguments:
      fnscf (string) - Filename of the pseudopotential, copied to the .save directory

  Returns:
      sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                       nl is a list of occupations at each site
      sh and nl are representative of one atom only
  '''

  import numpy as np
  import xml.etree.cElementTree as ET

  sh = []
  nl = []

  

  iterator_obj = ET.iterparse(fpp,events=('start','end'))
  iterator     = iter(iterator_obj)
  event,root   = next(iterator)

  for event,elem in iterator:        
      try:
          for i in elem.findall("PP_PSWFC/"):
              sh.append(int(i.attrib['l']))
      except Exception as e: print(e)

#      for i in elem.findall("PP_SPIN_ORB/"):
#            print(i)


  

  nl=list(range(len(sh)))
  return sh,nl

