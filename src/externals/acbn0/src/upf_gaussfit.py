#!/usr/bin/env python
#######################################################################
# Fit UPF radial pseudowavefunctions with gaussian orbitals
# Davide Ceresoli - May 2016
# Frank Cerasoli - June 2022
#
# Notes:
# - UPFv1 files must be embedded in <UPF version="1.0">...</UPF> element
# - contraction coefficients for d and f orbitals correspond to the
#   cubic harmonics
#######################################################################
import numpy as np

spn_map = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,
           'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,
           'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,
           'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
           'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,
           'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,
           'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,
           'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,
           'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,
           'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,
           'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,
           'Hg':80,'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,
           'Fr':87,'Ra':88,'Ac':89,'Th':90,'Pa':91,'U':92,'Np':93,
           'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,'Es':99,'Fm':100,
           'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,'Sg':106,
           'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112}

def get_atom_no ( n ):
  if n not in spn_map:
    raise Exception(f'Invalid atomic number: {n}')
  return spn_map[n]

# Double factorial (n!!)
def fact2 ( n ):
  if n <= 1:
    return 1
  return n * fact2(n-2)


#======================================================================
# GTO orbital
#======================================================================
def gto ( r, l, params ):

  alpha,beta = params[:2]
  coeffs = params[2:]

  gto = np.zeros_like(r)
  for j,c in enumerate(coeffs):
    zeta = alpha / beta**j
    i = np.where(zeta*r**2 > -12)
    gto[i] += c * r[i]**l * np.exp(-zeta*r[i]**2)

  return gto


#======================================================================
# Target function whose least square has to be minimized
#======================================================================
def target ( params, r, rab, wfc, l ):
  return wfc - r*gto(r, l, params)

def target_squared ( params, r, rab, wfc, l ):
  return np.sum(target(params,r,rab,wfc,l)**2)


#======================================================================
# Fit radial wfc with gaussians
#======================================================================
def fit ( nzeta, label, l, r, rab, wfc, threshold, least_squares=True ):

  if len(wfc) != len(r):
    raise Exception('wfc and r have different dimensions.')

  wfc,r = np.array(wfc), np.array(r)

  # Initial alpha and beta
  params0 = np.array([4.,4.] + [1.]*nzeta)

  # Least squares
  if least_squares:
    from scipy.optimize import leastsq

    params,fc,info,msg,ier = leastsq(target, params0, full_output=1,
                                     args=(r,rab,wfc,l), maxfev=5e4,
                                     ftol=1e-10, xtol=1e-10)
    if ier > 0:
      print(f'ERROR: ier={ier}\nmesg={msg}')
      print('ERROR: info[nfev]={}'.format(info['nfev']))
      print('ERROR: info[fvec]='.format(np.sum(info['fvec']**2)))

  # Minimize
  else:
    from scipy.optimize import minimize

    opt = minimize(target_squared, params0, args=(r,rab,wfc,l),
                   method='CG', tol=1e-10)
    params = opt.x
    if not opt.success:
      print('ERROR: opt.status={}'.format(opt.status))
      print('ERROR: opt.message={}'.format(opt.message))
      print('ERROR: opt.nfev={}'.format(opt.nfev))
      print('ERROR: opt.fun={}'.format(opt.fun))

  alpha,beta = params[:2]
  coeffs = np.sqrt(fact2(2*l+1)/(4*np.pi)) * params[2:]
  expon = []
  print(f'alpha = {alpha}, beta = {beta}')
  for j,c in enumerate(coeffs):
    zeta = alpha / beta**j
    expon.append(zeta)
    print(f'coeff = {c}, zeta = {zeta}')

  res = target_squared(params, r, rab, wfc, l)
  print('Fit result: {res}')

  if np.abs(res) > threshold:
    raise Exception('Fit did not meet the threshold resolution.')

  return coeffs, expon


#======================================================================
# Construct basis string for orbitals
#======================================================================
def basis_string ( label, l, coeffs, expon ):

  nzeta = len(coeffs)
  rstr = f'# label= {label} l= {l}\n[[\n'

  fcon = '], [\n'
  fbuf = '   {},\n'
  fpat = '({},{},{},{:20.10f},{:20.10f})'
  fline = lambda *arg : fbuf.format(fpat.format(*arg))

  if l == 0:
    for i,c in enumerate(coeffs):
      rstr += fline(0, 0, 0, c, expon[i])

  elif l == 1:
    for n in range(3):
      lind = [0]*3
      lind[2-n] = 1
      for i,c in enumerate(coeffs):
        rstr += fline(*lind, c, expon[i])
      if n < 2:
        rstr += fcon

  elif l == 2:

    # 1/(2*sqrt(3))*(2*z2 - x2 - y2)
    for n in range(3):
      lind = [0]*3
      lind[2-n] = 2
      fact = (1 if n==0 else -0.5) / np.sqrt(3)
      for i,c in enumerate(coeffs):
        rstr += fline(*lind, fact*c, expon[i])
    rstr += fcon

    # xz
    for i,c in enumerate(coeffs):
      rstr += fline(1, 0, 1, c, expon[i])
    rstr += fcon

    # yz
    for i,c in enumerate(coeffs):
      rstr += fline(0, 1, 1, c, expon[i])
    rstr += fcon

    # 1/2 * (x2 - y2)
    for i,c in enumerate(coeffs):
      rstr += fline(2, 0, 0, 0.5*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(0, 2, 0, -0.5*c, expon[i])
    rstr += fcon

    # xy
    for i,c in enumerate(coeffs):
      rstr += fline(1, 1, 0, c, expon[i])

  elif l == 3:
    # fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x3-3y2), fy(3x2-y2)

    # 1/(2*sqrt(15)) * z*(2*z2 - 3*x2 - 3*y2)
    fact = 0.5 / np.sqrt(15)
    for i,c in enumerate(coeffs):
      rstr += fline(0, 0, 3, 2*fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(2, 0, 1, -3*fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(0, 2, 1, -3*fact*c, expon[i])
    rstr += fcon

    # 1/(2*sqrt(10)) * x*(4*z2 - x2 - y2)
    fact = 0.5 / np.sqrt(10)
    for i,c in enumerate(coeffs):
      rstr += fline(1, 0, 2, 4*fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(0, 0, 3, -fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(1, 2, 0, -fact*c, expon[i])
    rstr += fcon

    # 1/(2*sqrt(10)) * y*(4*z2 - x2 - y2)
    fact = 0.5 / np.sqrt(10)
    for i,c in enumerate(coeffs):
      rstr += fline(0, 1, 2, 4*fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(2, 1, 0, -fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(0, 3, 0, -fact*c, expon[i])
    rstr += fcon

    # 1/2 * z*(x2 - y2)
    fact = 0.5
    for i,c in enumerate(coeffs):
      rstr += fline(2, 0, 1, fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(0, 2, 1, -fact*c, expon[i])
    rstr += fcon

    # x*y*z
    for i,c in enumerate(coeffs):
      rstr += fline(1, 1, 1, c, expon[i])
    rstr += fcon

    # 1/(2*sqrt(6)) * x*(x2 - 3*y2)
    fact = 0.5 / np.sqrt(6)
    for i,c in enumerate(coeffs):
      rstr += fline(3, 0, 0, fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(1, 2, 0, -3*fact*c, expon[i])
    rstr += fcon

    # 1/(2*sqrt(6)) * y*(3*x2 - y2)
    fact = 0.5 / np.sqrt(6)
    for i,c in enumerate(coeffs):
      rstr += fline(2, 1, 0, 3*fact*c, expon[i])
    for i,c in enumerate(coeffs):
      rstr += fline(0, 3, 0, -fact*c, expon[i])
    rstr += fcon

  return rstr + ']],\n'
