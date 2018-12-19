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

def plot_compare_PAO_DFT_eigs(Hks,Sks,my_eigsmat,read_S):
  import numpy as np
  from scipy import linalg as spl
  from numpy import linalg as npl
  from matplotlib import pyplot as plt

  nawf,nawf,nkpnts,nspin = Hks.shape
  E_k = np.zeros((nawf,nkpnts,nspin))

  ispin = 0 #plots only 1 spin channel
  #for ispin in xrange(nspin):
  for ik in range(nkpnts):
      if read_S:
          eigval,_ = spl.eigh(Hks[:,:,ik,ispin],Sks[:,:,ik],lower=False)
      else:
          eigval,_ = npl.eigh(Hks[:,:,ik,ispin],UPLO='U')
      E_k[:,ik,ispin] = np.sort(np.real(eigval))

  fig=plt.figure
  nbnds_dft,_,_=my_eigsmat.shape
  for i in range(nbnds_dft):
      #print("{0:d}".format(i))
      yy = my_eigsmat[i,:,ispin]
      if i==0:
          plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime',label='DFT')
      else:
          plt.plot(yy,'ok',markersize=3,markeredgecolor='lime',markerfacecolor='lime')

  for i in range(nawf):
      yy = E_k[i,:,ispin]
      if i==0:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None',label='PAO')
      else:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None')

  plt.xlabel('k-points')
  plt.ylabel('Energy - E$_F$ (eV)')
  plt.legend()
  plt.title('Comparison of PAO vs. DFT eigenvalues')
  plt.savefig('comparison.pdf',format='pdf')
  return()
