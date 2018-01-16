#!/usr/bin/env python
#######################################################################
# Fit UPF radial pseudowavefunctions with gaussian orbitals
# Davide Ceresoli - May 2016
#
# Notes:
# - UPFv1 files must be embedded in <UPF version="1.0">...</UPF> element
# - contraction coefficients for d and f orbitals correspond to the
#   cubic harmonics
#######################################################################
import sys
import numpy as np
import argparse
import StringIO
from math import *
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from xml.etree import ElementTree as ET
from elements import ELEMENTS
from matplotlib import pylab

# double factorial (n!!)
def fact2(n):
    if n <= 1: return 1
    return n*fact2(n-2)


#======================================================================
# GTO orbital
#======================================================================
def gto(r, l, params):
    alpha, beta = params[0:2]
    coeffs = params[2:]

    gto = np.zeros_like(r)
    for (j,coeff) in enumerate(coeffs):
        #assert alpha > 0 and beta > 0
        zeta = alpha/beta**j
        i = np.where(zeta*r*r > -12.0)
        gto[i] += r[i]**l * coeff*np.exp(-zeta*r[i]*r[i])

    return gto


#======================================================================
# Target function whose least square has to be minimized
#======================================================================
def target(params, r, rab, wfc, l):
    return wfc - r*gto(r, l, params)

def target_squared(params, r, rab, wfc, l):
    diff = target(params, r, rab, wfc, l)
    return np.dot(diff, diff)

#======================================================================
# Fit radial wfc with gaussians
#======================================================================
def fit(nzeta, label, l, r, rab, wfc):
    assert len(wfc) == len(r)

    wfc = np.array(wfc)
    r = np.array(r)

    params0 = np.array([4.0, 4.0]) # initial alpha and beta
    params0 = np.append(params0, np.ones((nzeta,)))

    # least squares
    if True:
        params, fit_cov, info, mesg, ier = \
            leastsq(target, params0, args=(r, rab, wfc, l), full_output=1, \
            maxfev=10000, ftol=1e-10, xtol=1e-10)
        if ier > 0:
            print "ERROR: ier=", ier, "mesg=", mesg
            print "ERROR: info[nfev]=", info["nfev"]
            print "ERROR: info[fvec]=", sum(info["fvec"]**2.0)
    else: # minimize
        opt = minimize(target_squared, params0, args=(r, rab, wfc, l), \
                       method='CG', tol=1e-10)
        #opt = basinhopping(target_squared, params0, minimizer_kwargs={'args':(r, rab, wfc, l)})
        params = opt.x
        if not opt.success:
           print "ERROR: opt.status=", opt.status
           print "ERROR: opt.message=", opt.message
           print "ERROR: opt.nfev=", opt.nfev
           print "ERROR: opt.fun=", opt.fun


    alpha, beta = params[0:2]
    n = sqrt(fact2(2*l+1)/(4.0*pi))
    coeffs = params[2:] * n
    expon = []
    print "alpha = %f, beta = %f" % (alpha, beta)
    for (j,coeff) in enumerate(coeffs):
        zeta = alpha/beta**j
        expon.append(zeta)
        print "coeff = %f,  zeta = %f" % (coeff, zeta)

    with open("wfc"+label+".dat", "wt") as f:
        gto_r = gto(r, l, params)
        for i in range(len(wfc)):
            f.write("%f %f %f\n" % (r[i], wfc[i], r[i]*gto_r[i]))
    pylab.plot(r, wfc, '.', label=label+"_orig")
    pylab.plot(r, r*gto(r, l, params), label=label+"_fit")
    print "INFO: fit result:", target_squared(params, r, rab, wfc, l)

    return coeffs, expon



#======================================================================
# Print python block for orbitals
#======================================================================
def print_python_block(bfile, label, l, coeffs, expon):
    nzeta = len(coeffs)
    print >>bfile, "# label=", label, "l=", l

    if l == 0:
        print >>bfile, "[["
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,0,0,coeffs[n],expon[n])
        print >>bfile, "]],"

    elif l == 1:
        print >>bfile, "[["
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,0,1,coeffs[n],expon[n])
        print >>bfile, "], ["
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,1,0,coeffs[n],expon[n])
        print >>bfile, "], ["
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (1,0,0,coeffs[n],expon[n])
        print >>bfile, "]],"

    elif l == 2:
        print >>bfile, "[["

        fact = 0.5/sqrt(3.0)
        for n in range(nzeta):  # 1/(2.0*sqrt(3))*(2*z2 - x2 - y2)
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,0,2,2.0*fact*coeffs[n],expon[n])
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,2,0,-fact*coeffs[n],expon[n])
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (2,0,0,-fact*coeffs[n],expon[n])
        print >>bfile, "], ["

        for n in range(nzeta): # xz
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (1,0,1,coeffs[n],expon[n])
        print >>bfile, "], ["

        for n in range(nzeta): # yz
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,1,1,coeffs[n],expon[n])
        print >>bfile, "], ["

        fact = 0.5
        for n in range(nzeta): # 1/2 * (x2 - y2)
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (0,2,0,fact*coeffs[n],expon[n])
        for n in range(nzeta):
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (2,0,0,-fact*coeffs[n],expon[n])
        print >>bfile, "], ["

        for n in range(nzeta): # xy
            print >>bfile, "   (%i,%i,%i,%20.10f,%20.10f)," % (1,1,0,coeffs[n],expon[n])
        print >>bfile, "]],"

    elif l == 3:
        print "l=3 not implemented yet!"

    return



#### parse command line ####
parser = argparse.ArgumentParser(description="Fit UPF radial wavefunctions")
parser.add_argument("--nzeta", dest="nzeta", type=int, default=3,
                    help="number of primitive gaussians")
parser.add_argument("--exclude", dest="exclude", type=str, default="",
                    help="orbitals to exclude from fitting")
parser.add_argument("filename", metavar="filename", help="UPF or UPF.xml file")
args = parser.parse_args()
nzeta = args.nzeta
xml_file = args.filename
exclude = args.exclude


#### open the UPF file ####
try:
    with open(xml_file) as f:
        xml_file_content = f.read()
    xml_file_content = xml_file_content.replace('&', '&amp;')
    root = ET.fromstring(xml_file_content)
    version = root.attrib["version"]
    upfver = int(version.split(".")[0])
    print "INFO: fitting file", xml_file, "with", nzeta, "gaussians"
    print "INFO: UPF version", upfver, "detected"
except Exception, inst:
    print "Unexpected error opening %s: %s" % (xml_file, inst)
    sys.exit(1)


#### get element name ####
if upfver == 1:
    text = root.find('PP_HEADER').text.split()
    for i in range(len(text)):
        if text[i] == 'Element':
            element = text[i-1].strip()
            break
else:
    element = root.find('PP_HEADER').attrib["element"].strip()

atno = ELEMENTS[element].number
print "INFO: element=", element, "atomic number=", atno
print


#### open basis file ####
basisfile = open(xml_file.replace(".xml","")+"_basis.py", "wt")
basisfile.write("basis_data = { %i : [\n" % (atno))
pylab.title(xml_file)


#### read the radial grid ####
text = root.find('PP_MESH/PP_R').text
r = np.array( [float(x) for x in text.split()] )
text = root.find('PP_MESH/PP_RAB').text
rab = np.array( [float(x) for x in text.split()] )


#### read and fit radial wavefunctions ####
if upfver == 1:
    pot = root.find('PP_LOCAL')
    if pot is None: quit()
    v = [float(x) for x in pot.text.split()]
    f = open('vlocal.dat', 'w')
    for i in range(len(v)):
        f.write("%f %f\n" % (r[i], v[i]))
    f.close()
   
    chis = root.find('PP_PSWFC')
    if chis is None:
         print "ERROR: cannot find PP_PSWFC tag"
         sys.exit(1)
    data = StringIO.StringIO(chis.text)
    nlines = len(r)/4
    if len(r) % 4 != 0: nlines += 1

    while True:
        line = data.readline()
        if line == "\n": continue
        if line == "": break
        label, l, occ, dummy = line.split()
        l = int(l)
        occ = float(occ)
        wfc = []

        for i in range(nlines):
            wfc.extend(map(float, data.readline().split()))
        wfc = np.array(wfc)

        if exclude.find(label) >= 0:
            print "INFO: skipping", label
            continue

        norm = sum(wfc*wfc*rab)
        print "INFO: fitting pswfc", label, "l=", l, "norm=", norm
        #wfc *= 1.0/sqrt(norm)
        coeffs, expon = fit(nzeta, label, l, r, rab, wfc)
        print_python_block(basisfile, label, l, coeffs, expon)
        print
 
    betas = root.find('PP_NONLOCAL/PP_BETA')
    if betas is None:
         print "ERROR: cannot find PP_BETA tag"
         sys.exit(1)
    data = StringIO.StringIO(betas.text)

    while True:
        line = data.readline()
        if line == "\n": continue
        if line.strip() == "": break
        ibeta, l, dummy, dummy = line.split()
        ibeta = int(ibeta)
        l = int(l)

        npoints = int(data.readline())
        nlines = npoints/4
        if npoints % 4 != 0: nlines += 1
        beta = []

        for i in range(nlines):
            beta.extend(map(float, data.readline().split()))
        print beta
        beta = np.array(beta)
        f = open("beta_%i_%i.dat" % (ibeta, l), 'w')
        for i in range(len(beta)):
            f.write("%f %f\n" % (r[i], beta[i]))
        f.close()
        line = data.readline()
        print line
        line = data.readline()
        print line
 
else:
    pot = root.find('PP_LOCAL')
    if pot is None: quit()
    v = [float(x) for x in pot.text.split()]
    f = open('vlocal.dat', 'w')
    for i in range(len(v)):
        f.write("%f %f\n" % (r[i], v[i]))
    f.close()

    pot = root.find('PP_GIPAW/PP_GIPAW_VLOCAL/PP_GIPAW_VLOCAL_AE')
    if pot is None: quit()
    v = [float(x) for x in pot.text.split()]
    f = open('vlocal_ae.dat', 'w')
    for i in range(len(v)):
        f.write("%f %f\n" % (r[i], v[i]))
    f.close()

    i = 0
    while True:
        i += 1
        chi = root.find('PP_PSWFC/PP_CHI.%i' % (i))
        if chi is None: break

        label = chi.attrib["label"]
        if exclude.find(label) >= 0:
            print "INFO: skipping", label
            continue
        l = int(chi.attrib["l"])
        wfc = [float(x) for x in chi.text.split()]
        assert len(wfc) == len(r)
        wfc = np.array(wfc)
       
        norm = sum(wfc*wfc*rab)
        print "INFO: fitting pswfc", label, "l=", l, "norm=", norm
        #wfc *= 1.0/sqrt(norm)
        coeffs, expon = fit(nzeta, label, l, r, rab, wfc)
        print_python_block(basisfile, label, l, coeffs, expon)
        print

basisfile.write("]}\n")
basisfile.close()
pylab.legend(loc='lower right')
pylab.grid()
pylab.xlim(0, 50.0)
pylab.xlabel('r (bohrradius)')
pylab.ylabel('radial wfc')
pylab.show()
print "INFO: file", basisfile.name, "created!"


