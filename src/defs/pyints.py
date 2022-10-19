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

import numpy as np

class CGBF:
  '''Class for a contracted Gaussian basis function'''
  def __init__ ( self, origin, atid=0 ):
    self.origin = tuple(float(i) for i in origin)
    self.powers = [] 
    self.pnorms = []
    self.prims = []
    self.pnorms = []
    self.pexps = []
    self.pcoefs = []
    self.atid = atid


def gammln ( x ):
  "Numerical recipes, section 6.1"
  cof = [76.18009172947146, -86.50532032941677,
         24.01409824083091, -1.231739572450155,
         0.1208650973866179e-2, -0.5395239384953e-5]
  xt = x
  tmp = x + 5.5
  tmp = tmp - (x+0.5)*np.log(tmp)
  ser = 1.000000000190015
  for c in cof:
    xt += 1
    ser += c / xt
  return -tmp + np.log(2.5066282746310005*ser/x);


def _gser ( a, x ):
  "Series representation of Gamma. NumRec sect 6.1."
  ITMAX = 100
  EPS = 3.e-7

  gln = gammln(a)
  if x < 0:
    raise ValueError(f'Bounds: x >= 0, x={x}')

  if x == 0:
    return 0, gln

  ap = a
  delt = tsum = 1/a
  for i in range(ITMAX):
    ap = 1 + ap
    delt *= x / ap
    tsum += delt
    if abs(delt) < abs(tsum)*EPS:
      break
  else:
    print('a too large, ITMAX too small in gser')

  gamser = tsum * np.exp(-x+a*np.log(x)-gln)
  return gamser, gln


def _gcf ( a, x ):
  "Continued fraction representation of Gamma. NumRec sect 6.1"
  ITMAX = 100
  EPS = 3.e-7
  FPMIN = 1.e-30

  gln = gammln(a)
  b = 1 + x - a
  c = 1 / FPMIN
  d = 1 / b
  h = d
  for i in range(1, ITMAX+1):
    an = -i*(i-a)
    b = b + 2
    d = an*d + b
    if abs(d) < FPMIN:
      d = FPMIN
    c = b + an/c
    if abs(c) < FPMIN:
      c = FPMIN
    d = 1 / d
    delt = d * c
    h *= delt
    if abs(delt-1) < EPS:
      break
  else:
    print('a too large, ITMAX too small in gcf')

  gammcf = h * np.exp(-x+a*np.log(x)-gln)
  return gammcf, gln


def dist ( A, B ):
  return np.sum([(A[i]-B[i])**2 for i in range(3)])


def gaussian_product_center ( alpha1, A, alpha2, B ):
  gamma = alpha1 + alpha2
  return [(alpha1*A[i]+alpha2*B[i])/gamma for i in range(3)]


def Fgamma ( m, x ):
  "Incomplete gamma function"
  mp5 = m + 0.5
  x = max(abs(x), 1e-8)
  return gamm_inc(mp5, x) / (2*x**mp5)


def gammp ( a, x ):
  "Returns the incomplete gamma function P(a;x). NumRec sect 6.2."
  if x <= 0 or a < 0:
    raise ValueError(f'Bounds: (x > 0) and (a >= 0), x={x}, a={a}')

  if x < a + 1: 
    return _gser(a, x)

  gammcf,gln = _gcf(a, x)
  return 1-gammcf, gln 


def gamm_inc ( a, x ):
  "Incomple gamma function; computed from NumRec routine gammp."
  gammap,gln = gammp(a, x)
  return gammap * np.exp(gln)


def fact ( i ):
  "Normal factorial"
  val = 1
  while i > 1:
    val *= i
    i -= 1
  return val


def fact_ratio ( a, b ):
  return fact(a) / fact(b) / fact(a-2*b)


def binomial ( a, b ):
  "Binomial coefficient"
  return fact(a) / fact(b) / fact(a-b)


def binomial_prefactor ( s, ia, ib, xpa, xpb ):
  "From Augspurger and Dykstra"
  bsum = 0
  for t in range(s+1):
    if s-ia <= t <= ib:
      sgn = (xpa)**(ia-s+t) * (xpb)**(ib-t)
      bsum += sgn * binomial(ia, s-t) * binomial(ib, t)
  return bsum


def B0 ( i, r, g ):
  return fact_ratio(i,r) * (4*g)**(r-i)


def fB ( i, l1, l2, P, A, B, r, g ):
  return binomial_prefactor(i,l1,l2,P-A,P-B) * B0(i,r,g)


def B_array ( l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta ):
  ind_max = l1 + l2 + l3 + l4 + 1
  B = np.zeros(ind_max, dtype=float)
  for i1 in range(l1+l2+1):
    for i2 in range(l3+l4+1):
      for r1 in range(i1//2+1):
        for r2 in range(i2//2+1):
          for u in range((i1+i2)//2-r1-r2+1):
            "THO eq. 2.22"
            dm = i1 + i2 - 2*(r1+r2)
            ind = dm - u
            fr = fact_ratio(dm, u)
            pqp = (q-p)**(dm-2*u)
            pdelta = delta**ind
            sign = (-1)**i2 * (-1)**u
            fb1 = fB(i1, l1, l2, p, a, b, r1, g1)
            fb2 = fB(i2, l3, l4, q, c, d, r2, g2)
            B[ind] += sign * fb1 * fb2 * fr * pqp / pdelta
  return B


def coulomb_repulsion(axyz, anorm, almn, aalpha,
                      bxyz, bnorm, blmn, balpha,
                      cxyz, cnorm, clmn, calpha,
                      dxyz, dnorm, dlmn, dalpha ):

  rab = dist(axyz, bxyz)
  rcd = dist(cxyz, dxyz)
  pxyz = gaussian_product_center(aalpha, axyz, balpha, bxyz)
  qxyz = gaussian_product_center(calpha, cxyz, dalpha, dxyz)
  rpq = dist(pxyz, qxyz)
  gamma1 = aalpha + balpha
  gamma2 = calpha + dalpha
  delta = (1/gamma1 + 1/gamma2) / 4

  Bx,By,Bz = (B_array(almn[i], blmn[i], clmn[i], dlmn[i],
               pxyz[i], axyz[i], bxyz[i], qxyz[i], cxyz[i], dxyz[i],
               gamma1, gamma2, delta) for i in range(3))

  i = 0
  bsum = 0
  nl = almn[0]
  for i in range(almn[0]+blmn[0]+clmn[0]+dlmn[0]+1):
    for j in range(almn[1]+blmn[1]+clmn[1]+dlmn[1]+1):
      for k in range(almn[2]+blmn[2]+clmn[2]+dlmn[2]+1):
        bsum += Bx[i]*By[j]*Bz[k]*Fgamma(i+j+k, rpq/delta/4)

  norm = anorm * bnorm * cnorm * dnorm
  g2 = gamma1 * gamma2 * np.sqrt(gamma1+gamma2)
  e1 = np.exp(-aalpha*balpha*rab/gamma1)
  e2 = np.exp(-calpha*dalpha*rcd/gamma2)
  return 2 * np.pi**2.5 * e1 * e2 * bsum * norm / g2


def contr_coulomb ( aexps, acoefs, anorms, xyza, powa,
                    bexps, bcoefs, bnorms, xyzb, powb,
                    cexps, ccoefs, cnorms, xyzc, powc,
                    dexps, dcoefs, dnorms, xyzd, powd ):

  Jij = 0.
  for i in range(len(aexps)):
    for j in range(len(bexps)):
      for k in range(len(cexps)):
        for l in range(len(dexps)):
          incr = coulomb_repulsion(xyza, anorms[i], powa[i], aexps[i],
                                   xyzb, bnorms[j], powb[j], bexps[j],
                                   xyzc, cnorms[k], powc[k], cexps[k],
                                   xyzd, dnorms[l], powd[l], dexps[l])
          Jij += incr * acoefs[i] * bcoefs[j] * ccoefs[k] * dcoefs[l]
  return Jij
