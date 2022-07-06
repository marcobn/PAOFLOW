'''
 Thu Jul 25 12:00:46 CDT 2013
 By Luis Agapito @ Marco Buongiorno Nardelli at UNT

 Revised Jul 5 2022
  Frank T. Cerasoli

 Based on:
 Ints.py Basic routines for integrals in the PyQuante framework
 This program is part of the PyQuante quantum chemistry program suite.
 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 
 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
'''


def coulomb ( a, b, c, d, func ):
  ' Coulomb interaction between four contracted Gaussians '
  import numpy as np

  if func.__name__ == 'contr_coulomb_v3':
    al,am,an = np.array(a.powers).T
    bl,bm,bn = np.array(b.powers).T
    cl,cm,cn = np.array(c.powers).T
    dl,dm,dn = np.array(d.powers).T
    return func(a.pexps,a.pcoefs,a.pnorms,a.origin,al,am,an,
                b.pexps,b.pcoefs,b.pnorms,b.origin,bl,bm,bn,
                c.pexps,c.pcoefs,c.pnorms,c.origin,cl,cm,cn,
                d.pexps,d.pcoefs,d.pnorms,d.origin,dl,dm,dn)
  else:
    return func(a.pexps,a.pcoefs,a.pnorms,a.origin,a.powers,
                b.pexps,b.pcoefs,b.pnorms,b.origin,b.powers,
                c.pexps,c.pcoefs,c.pnorms,c.origin,c.powers,
                d.pexps,d.pcoefs,d.pnorms,d.origin,d.powers)


def get2ints ( bfs, coul_func ):
  '''
    Store integrals in a long array in the form (ij|kl) (chemists
     notation. We only need i>=j, k>=l, and ij <= kl
  '''
  from pyints import ijkl2intindex

  nbf = len(bfs)
  totlen = nbf * (nbf+1) * (2+nbf+nbf**2)/8
  integrals = np.zeros(totlen, dtype=float)
  for i in range(nbf):
    for j in range(i+1):
      ij = i*(i+1)/2 + j
      for k in range(nbf):
        for l in range(k+1):
          kl = k*(k+1)/2 + l
          if ij >= kl:
            ijkl = ijkl2intindex(i, j, k, l)
            integrals = coulomb(bfs[i], bfs[j], bfs[k], bfs[l], coul_func)

