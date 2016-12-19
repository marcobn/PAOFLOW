#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
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

# This module from AFLOWpi

import numpy as np
import sys


def free2abc(cellparamatrix,cosine=True,degrees=True,string=True,bohr=False):
    '''
    Convert lattice vectors to a,b,c,alpha,beta,gamma of the primitive lattice

    Arguments:
          cellparamatrix (np.matrix): matrix of cell vectors

    Keyword Arguments:
          cosine (bool): If True alpha,beta,gamma are cos(alpha),cos(beta),cos(gamma),
          degrees (bool): If True return alpha,beta,gamma in degrees; radians if False
          string (bool): If True return a,b,c,alpha,beta,gamma as a string; if False return as a list
          bohr (bool): If True return a,b,c in bohr radii; if False return in angstrom

    Returns:
         paramArray (list): a list of the parameters a,b,c,alpha,beta,gamma generated from the input matrix

    '''

    ibrav = getIbravFromVectors(cellparamatrix)
    try:
        cellparamatrix=cellparamatrix.getA()
    except :
        pass
    try:
        a = np.sqrt(cellparamatrix[0].dot(cellparamatrix[0].T))
        b = np.sqrt(cellparamatrix[1].dot(cellparamatrix[1].T))
        c = np.sqrt(cellparamatrix[2].dot(cellparamatrix[2].T))
    except:
        cellparamatrix = np.array(cellparamatrix)
        a = np.sqrt(cellparamatrix[0].dot(cellparamatrix[0]))
        b = np.sqrt(cellparamatrix[1].dot(cellparamatrix[1]))
        c = np.sqrt(cellparamatrix[2].dot(cellparamatrix[2]))

    degree2radian = np.pi/180
    alpha,beta,gamma=(0.0,0.0,0.0)


    alpha = np.arccos(cellparamatrix[1].dot(cellparamatrix[2].T)/(b*c))
    beta  = np.arccos(cellparamatrix[0].dot(cellparamatrix[2].T)/(a*c))
    gamma = np.arccos(cellparamatrix[0].dot(cellparamatrix[1].T)/(a*b))

    if np.abs(alpha)<0.000001:
        alpha=0.0
    if np.abs(beta)<0.000001:
        beta=0.0
    if np.abs(gamma)<0.000001:
        gamma=0.0


    AngstromToBohr = 1.88971616463207
    BohrToAngstrom = 1/AngstromToBohr
    if bohr==False:
        a*=BohrToAngstrom
        b*=BohrToAngstrom
        c*=BohrToAngstrom

        a=float('%10.5e'%np.around(a,decimals=5))
        b=float('%10.5e'%np.around(b,decimals=5))
        c=float('%10.5e'%np.around(c,decimals=5))

    if cosine==True:
        cosBC=np.cos(alpha)
        cosAC=np.cos(beta)
        cosAB=np.cos(gamma)
        paramArray = [a,b,c,cosBC,cosAC,cosAB]

        param_list=[]
        for v in xrange(len(paramArray)):
            param_list.append(float('%10.5e'%np.around(paramArray[v],decimals=5)))
        paramArray=tuple(param_list)

        returnString = 'a=%s,b=%s,c=%s,cos(alpha)=%s,cos(beta)=%s,cos(gamma)=%s' % tuple(paramArray)

    if degrees==True:
        alpha/=degree2radian
        beta/= degree2radian
        gamma/=degree2radian

    if cosine!=True:
        paramArray = (a,b,c,alpha,beta,gamma)

        param_list=[]
        for v in xrange(len(paramArray)):
            param_list.append(float('%10.5e'%np.around(paramArray[v],decimals=5)))
        paramArray=tuple(param_list)

        returnString = 'A=%s,B=%s,C=%s,alpha=%s,beta=%s,gamma=%s' % tuple(paramArray)

    if string==True:
        return returnString
    else:

        return paramArray



def _getHighSymPoints(ibrav,alat,cellOld):
    '''
    Searching for the ibrav number in the input file for the calculation
    to determine the path for the band structure calculation

    Arguments:
          oneCalc (dict): a dictionary containing properties about the AFLOWpi calculation

    Keyword Arguments:
          ID (str): ID string for the particular calculation and step

    Returns:
          special_points (list): list of the HSP names
          band_path (str): path in string form

    '''

    def CUB(cellOld):
        special_points = {
          'gG'   : (0.0, 0.0, 0.0),
          'M'   : (0.5, 0.5, 0.0),
          'R'   : (0.5, 0.5, 0.5),
          'X'   : (0.0, 0.5, 0.0)
        }

        default_band_path = 'gG-X-M-gG-R-X|M-R'
        band_path = default_band_path
        return special_points, band_path
    def FCC(cellOld):
        special_points = {
          'gG'   : (0.0, 0.0, 0.0),
          'K'    : (0.375, 0.375, 0.750),
          'L'    : (0.5, 0.5, 0.5),
          'U'    : (0.625, 0.250, 0.625),
          'W'    : (0.5, 0.25, 0.75),
          'X'    : (0.5, 0.0, 0.5)
        }

        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[-1.0, 0.0, 1.0,],
                                    [ 0.0, 1.0, 1.0],
                                    [-1.0, 1.0, 0.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())



        default_band_path = 'gG-X-W-K-gG-L-U-W-L-K|U-X'
        band_path = default_band_path
        return special_points, band_path
    def BCC(cellOld):

        special_points = {
            'gG'  : [0, 0, 0],
            'H'   : [0.5, -0.5, 0.5],
            'Hp'   : [0.5, 0.5, -0.5],
            'P'   : [0.25, 0.25, 0.25],
            'Np'   : [0.5, 0.0, 0.0],
            'N'   : [0.0, 0.0, 0.5]
            }

        #convert HSP from aflow to qe convention brillouin zones
        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 1.0,],
                                    [-1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-H-N-gG-P-H|P-N'
        #default_band_path = 'gG-H-P-N-gG-Hp-Np-gG'
        band_path = default_band_path
        return special_points, band_path


    def HEX(cellOld):
        special_points = {
          'gG'    : (0, 0, 0),
          'A'    : (0.0, 0.0, 0.5),
          'H'    : (1.0/3.0, 1.0/3.0, 0.5),
          'K'    : (1.0/3.0, 1.0/3.0, 0.0),
          'L'    : (0.5, 0.0, 0.5),
          'M'    : (0.5, 0.0, 0.0)
        }

        default_band_path = 'gG-M-K-gG-A-L-H-A|L-M|K-H'
        band_path = default_band_path
        return special_points, band_path


    def RHL1(cellOld):

        tx = cellOld[0][0]
        c = (((tx**2)*2)-1.0)
        c = -c
        alpha = np.arccos(c)
        eta1 = 1.0 + 4.0*np.cos(alpha)
        eta2 = 2.0 + 4.0*np.cos(alpha)
        eta = eta1/eta2
        nu = 0.75 - eta/2.0
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'B'    : (eta, 0.5, 1.0-eta),
            'B1'    : (0.5, 1.0-eta, eta-1.0),
            'F'    : (0.5, 0.5, 0.0),
            'L'    : (0.5, 0.0, 0.0),
            'L1'    : (0.0, 0.0, -0.5),
            'P'    : (eta, nu, nu),
            'P1'    : (1.0-nu, 1.0-nu, 1.0-eta),
            'P2'    : (nu, nu, eta-1.0),
            'Q'    : (1.0-nu, nu, 0.0),
            'X'    : (nu, 0.0, -nu),
            'Z'    : (0.5, 0.5, 0.5)
          }
        default_band_path = 'gG-L-B1|B-Z-gG-X|Q-F-P1-Z|L-P'
        band_path = default_band_path
        return special_points, band_path

    def RHL2(cellOld):

        tx = cellOld[0][0]
        c = (((tx**2)*2)-1.0)
        c = -c
        alpha = np.arccos(c)
        eta = 1.0/(2*np.tan(alpha/2.0)**2)
        nu = 0.75 - eta/2.0

        special_points = {
           'gG'    : (0.0, 0.0, 0.0),
           'F'    : (0.5, -0.5, 0.0),
           'L'    : (0.5, 0.0, 0.0),
           'P'    : (1.0-nu, -nu, 1.0-nu),
           'P1'    : (nu, nu-1.0, nu-1.0),
           'Q'    : (eta, eta, eta),
           'Q1'    : (1.0-eta, -eta, -eta),
           'Z'    : (0.5, -0.5, 0.5)
         }
        default_band_path = 'gG-P-Z-Q-gG-F-P1-Q1-L-Z'
        band_path = default_band_path
        return special_points, band_path



    def TET(cellOld):

        special_points = {
          'gG'    : (0.0, 0.0, 0.0),
          'A'    : (0.5, 0.5, 0.5),
          'M'    : (0.5, 0.5, 0.0),
          'R'    : (0.0, 0.5, 0.5),
          'X'    : (0.0, 0.5, 0.0),
          'Z'    : (0.0, 0.0, 0.5)
        }
        default_band_path = 'gG-X-M-gG-Z-R-A-Z|X-R|M-A'
        band_path = default_band_path
        return special_points, band_path

    def BCT1(cellOld):
        a = cellOld[1][0]*2
        c = cellOld[1][2]*2

        if a==c:
            return BCC(cellOld)

        eta = (1 + c**2/a**2)/4

        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'M'    : (-0.5, 0.5, 0.5),
            'N'    : (0.0, 0.5, 0.0),
            'P'    : (0.25, 0.25, 0.25),
            'X'    : (0.0, 0.0, 0.5),
            'Z'    : (eta, eta, -eta),
            'Z1'    : (-eta, 1.0-eta, eta)
          }

        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-X-M-gG-Z-P-N-Z1-M|X-P'
        band_path = default_band_path
        return special_points, band_path


    def BCT2(cellOld):

        a = cellOld[1][0]*2
        c = cellOld[1][2]*2
        if a==c:
            return BCC(cellOld)

        eta = (1 + a**2/c**2)/4.0
        zeta = a**2/(2.0*c**2)
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'N'    : (0.0, 0.5, 0.0),
            'P'    : (0.25, 0.25, 0.25),
            'gS'    : (-eta, eta, eta),
            'gS1'    : (eta, 1-eta, -eta),
            'X'    : (0.0, 0.0, 0.5),
            'Y'    : (-zeta, zeta, 0.5),
            'Y1'   : (0.5, 0.5, -zeta),
            'Z'    : (0.5, 0.5, -0.5)
          }

        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-X-Y-gS-gG-Z-gS1-N-P-Y1-Z|X-P'
        band_path = default_band_path
        return special_points, band_path



    def ORC(cellOld):
        special_points = {
          'gG'    : (0.0, 0.0, 0.0),
          'R'    : (0.5, 0.5, 0.5),
          'S'    : (0.5, 0.5, 0.0),
          'T'    : (0.0, 0.5, 0.5),
          'U'    : (0.5, 0.0, 0.5),
          'X'    : (0.5, 0.0, 0.0),
          'Y'    : (0.0, 0.5, 0.0),
          'Z'    : (0.0, 0.0, 0.5)
        }

        default_band_path = 'gG-X-S-Y-gG-Z-U-R-T-Z|Y-T|U-X|S-R'
        band_path = default_band_path
        return special_points, band_path

    def ORCF1(cellOld):

        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        eta = (1 + a**2/b**2 + a**2/c**2)/4
        zeta = (1 + a**2/b**2 - a**2/c**2)/4
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'A'    : (0.5, 0.5 + zeta, zeta),
            'A1'    : (0.5, 0.5-zeta, 1.0-zeta),
            'L'    : (0.5, 0.5, 0.5),
            'T'    : (1.0, 0.5, 0.5),
            'X'    : (0.0, eta, eta),
            'X1'    : (1.0, 1.0-eta, 1.0-eta),
            'Y'    : (0.5, 0.0, 0.5),
            'Z'    : (0.5, 0.5, 0.0)
          }



        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0],
                                    [ 0.0, 1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-Y-T-Z-gG-X-A1-Y|T-X1|X-A-Z|L-gG'
        band_path = default_band_path
        return special_points, band_path

    def ORCF2(cellOld):

        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        eta = (1 + a**2/b**2 - a**2/c**2)/4
        phi = (1 + c**2/b**2 - c**2/a**2)/4
        delta = (1 + b**2/a**2 - b**2/c**2)/4
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'C'    : (0.5, 0.5-eta, 1.0-eta),
            'C1'    : (0.5, 0.5+eta, eta),
            'D'    : (0.5-delta, 0.5, 1.0-delta),
            'D1'    : (0.5+delta, 0.5, delta),
            'L'    : (0.5, 0.5, 0.5),
            'H'    : (1.0-phi, 0.5-phi, 0.5),
            'H1'    : (phi, 0.5+phi, 0.5),
            'X'    : (0.0, 0.5, 0.5),
            'Y'    : (0.5, 0.0, 0.5),
            'Z'    : (0.5, 0.5, 0.0),
            }

        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0],
                                    [ 0.0, 1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())
#           print( k,special_points[k])
        default_band_path = 'gG-Y-C-D-X-gG-Z-D1-H-C|C1-Z|X-H1|H-Y|L-gG'
        band_path = default_band_path
        return special_points, band_path


    def ORCF3(cellOld):

        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        eta = (1 + a**2/b**2 + a**2/c**2)/4
        zeta = (1 + a**2/b**2 - a**2/c**2)/4
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'A'    : (0.5, 0.5 + zeta, zeta),
            'A1'    : (0.5, 0.5-zeta, 1.0-zeta),
            'L'    : (0.5, 0.5, 0.5),
            'T'    : (1.0, 0.5, 0.5),
            'X'    : (0.0, eta, eta),
            'X1'    : (1.0, 1.0-eta, 1.0-eta),
            'Y'    : (0.5, 0.0, 0.5),
            'Z'    : (0.5, 0.5, 0.0)
            }

        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0,],
                                    [ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0,],
                                    [ 1.0, 1.0, 0.0],
                                    [ 0.0, 1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-Y-T-Z-gG-X-A1-Y|X-A-Z|L-R'
        band_path = default_band_path
        return special_points, band_path


    def ORCC(cellOld):

        try:
            cellOld=cellOld.getA()
        except:
            pass
        a = cellOld[0][0]*2
        b = cellOld[1][1]*2
        c = cellOld[2][2]
        myList = [a, b,c]
        c = max(myList)
        a = min(myList)
        myList.remove(a)
        myList.remove(c)
        b = myList[0]

        zeta = (1 + a**2/b**2)/4.0
        special_points = {
            'gG'    : (0.0, 0.0, 0.0),
            'A'    : (zeta, zeta, 0.5),
            'A1'    : (-zeta, 1.0-zeta, 0.5),
            'R'    : (0.0, 0.5, 0.5),
            'S'    : (0.0, 0.5, 0.0),
            'T'    : (-0.5, 0.5, 0.5),
            'X'    : (zeta, zeta, 0.0),
            'X1'    : (-zeta, 1.0-zeta, 0.0),
            'Y'    : (-0.5, 0.5, 0.0),
            'Z'    : (0.0, 0.0, 0.5)
          }

        aflow_conv = np.asarray([[ 1.0,-1.0, 0.0,],
                                    [ 1.0, 1.0, 0.0,],
                                    [ 0.0, 0.0, 2.0,],])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 0.0,],
                                    [-1.0, 1.0, 0.0],
                                    [ 0.0, 0.0, 2.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())


        default_band_path = 'gG-X-S-R-A-Z-gG-Y-X1-A1-T-Y|Z-T'
        band_path = default_band_path
        return special_points, band_path

    def ORCI(cellOld):

        try:
            cellOld=cellOld.getA()
        except:
            pass
        a1 = cellOld[0][0]*2
        b1 = cellOld[1][1]*2
        c1 = cellOld[2][2]*2
        myList = [a1, b1, c1]
        c = max(myList)
        a = min(myList)

        myList.remove(a)
        myList.remove(c)
        b = myList[0]
        print(abs(a-b),abs(a-c),abs(c-b))
        if abs(1.0-a/b)<0.01:
            if abs(1.0-a/c)<0.01:

                return BCC(cellOld)
            if abs(1.0-c/b)<0.01:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)
            else:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)
        if abs(1.0-a/c)<0.01:
            if abs(1.0-a/b)<0.01:
                return BCC(cellOld)
            if abs(1.0-c/b)<0.01:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)


            else:
                if c<a:
                    return BCT1(cellOld)
                else:
                    return BCT2(cellOld)


        chi = (1.0 + (a/c)**2)/4.0
        eta = (1.0 + (b/c)**2)/4.0
        delta = (b*b - a*a)/(4*c*c)
        mu = (b*b + a*a)/(4*c*c)
        special_points = {
          'gG'    : (0, 0, 0),
          'L'    : (-mu, mu, 0.5-delta),
          'L1'   : (mu, -mu, 0.5+delta),
          'L2'   : (0.5-delta, 0.5+delta, -mu),
          'R'    : (0.0, 0.5, 0.0),
          'S'    : (0.5, 0.0, 0.0),
          'T'    : (0.0, 0.0, 0.5),
          'W'    : (0.25,0.25,0.25),
          'X'    : (-chi, chi, chi),
          'X1'   : (chi, 1.0-chi, -chi),
          'Y'    : (eta, -eta, eta),
          'Y1'   : (1.0-eta, eta, -eta),
          'Z'    : (0.5, 0.5, -0.5)
        }

        aflow_conv = np.asarray([[-1.0, 1.0, 1.0,],
                                    [ 1.0,-1.0, 1.0],
                                    [ 1.0, 1.0,-1.0],])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 1.0,],
                                    [-1.0, 1.0, 1.0],
                                    [-1.0,-1.0, 1.0],])/2.0

        for k,v in special_points.iteritems():
            first  = np.array(v).dot(np.linalg.inv(aflow_conv))
            second = qe_conv.dot(first)
            special_points[k]=tuple(second.tolist())

        default_band_path = 'gG-X-L-T-W-R-X1-Z-gG-Y-S-W|L1-Y|Y1-Z'
        band_path = default_band_path
        return special_points, band_path


    def MCLC(cellOld):
        try:
            cellOld=cellOld.getA()
        except:
            pass

        a,b,c,cos_alpha,cos_beta_cos_gamma = free2abc(cellparamatrix,cosine=True,degrees=False,string=False,bohr=False)

        sin_gamma = np.sin(np.arccos(cos_gamma))

        mu        = (1+(b/a)**2.0)/4.0
        delta     = b*c*cos_gamma/(2.0*a**2.0)
        xi        = mu -0.25*+(1.0 - b*cos_gamma/c)*(4.0*(sin_gamma**2.0))
        eta       = 0.5 + 2.0*xi*c*cos_gamma/b
        phi       = 1.0 + xi - 2.0*mu
        psi       = eta - 2.0*delta

    def MCLC12(cellOld):
        pass

    def MCLC5(cellOld):
        pass

    '''
    def MCL(cellOld):
       a1 = cellOld[0][0]
       b1 = (cellOld[1][0]**2 + cellOld[1][1]**2)**(0.5)
       c1 = cellOld[2][2]
       gamma = np.arctan(cellOld[1][1]/cellOld[1][0])
       myList = [a1, b1, c1]
       c = max(myList)
       a = min(myList)
       myList.remove(c)
       myList.remove(a)
       b = myList[0]
       alpha = gamma

       eta = (1 - b*np.cos(alpha)/c)/(2*np.sin(alpha)**2)
       nu = 0.5 - eta*c*np.cos(alpha)/b
       special_points = {
           'gG'    : (0.0, 0.0, 0.0),
           'A'    : (0.5, 0.5, 0.0),
           'C'    : (0.0, 0.5, 0.5),
           'D'    : (0.5, 0.0, 0.5),
           'D1'    : (0.5, 0.0, -0.5),
           'E'    : (0.5, 0.5, 0.5),
           'H'    : (0.0, eta, 1.0-nu),
           'H1'    : (0.0, 1.0-eta, nu),
           'H2'    : (0.0, eta, -nu),
           'M'    : (0.5, eta, 1.0-nu),
           'M1'    : (0.5, 1.0-eta, nu),
           'M2'    : (0.5, eta, -nu),
           'X'    : (0.0, 0.5, 0.0),
           'Y'    : (0.0, 0.0, 0.5),
           'Y1'    : (0.0, 0.0, -0.5),
           'Z'    : (0.5, 0.0, 0.0)
         }

       default_band_path = 'gG-Y-H-C-E-M1-A-X-H1|M-D-Z|Y-D'
       band_path = default_band_path
       return special_points, band_path
    '''


    def TRI1A(cellOld):
        special_points = {
           'gG'   : (0.0,0.0,0.0),
           'L'    : (0.5,0.5,0.0),
           'M'    : (0.0,0.5,0.5),
           'N'    : (0.5,0.0,0.5),
           'R'    : (0.5,0.5,0.5),
           'X'    : (0.5,0.0,0.0),
           'Y'    : (0.0,0.5,0.0),
           'Z'    : (0.0,0.0,0.5),
           }

        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'
        return special_points, band_path
    def TRI2A(cellOld):
        special_points = {
           'gG'   : (0.0,0.0,0.0),
           'L'    : (0.5,0.5,0.0),
           'M'    : (0.0,0.5,0.5),
           'N'    : (0.5,0.0,0.5),
           'R'    : (0.5,0.5,0.5),
           'X'    : (0.5,0.0,0.0),
           'Y'    : (0.0,0.5,0.0),
           'Z'    : (0.0,0.0,0.5),
           }

        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'

        return special_points, band_path

    def TRI1B(cellOld):
        special_points = {
           'gG'   : ( 0.0, 0.0,0.0),
           'L'    : ( 0.5,-0.5,0.0),
           'M'    : ( 0.0, 0.0,0.5),
           'N'    : (-0.5,-0.5,0.5),
           'R'    : ( 0.0,-0.5,0.5),
           'X'    : ( 0.0,-0.5,0.0),
           'Y'    : ( 0.5, 0.0,0.0),
           'Z'    : (-0.5, 0.0,0.5),
           }


        band_path = str("X-gG-Y|L-gG-Z|N-gG-M|R-gG")
        return special_points, band_path

    def TRI2B(cellOld):
        special_points = {
           'gG'   : ( 0.0, 0.0,0.0),
           'L'    : ( 0.5,-0.5,0.0),
           'M'    : ( 0.0, 0.0,0.5),
           'N'    : (-0.5,-0.5,0.5),
           'R'    : ( 0.0,-0.5,0.5),
           'X'    : ( 0.0,-0.5,0.0),
           'Y'    : ( 0.5, 0.0,0.0),
           'Z'    : (-0.5, 0.0,0.5),
           }

        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'
        return special_points, band_path

#############################################################################################################

    def choose_lattice(ibrav, cellOld):
        if(int(ibrav)==1):
            var1, var2 = CUB(cellOld)
            return var1, var2

        elif(int(ibrav)==2):
            var1, var2 = FCC(cellOld)
            return var1, var2

        elif(int(ibrav)==3):
            var1, var2 = BCC(cellOld)
            return var1, var2

        elif(int(ibrav)==4):
            var1, var2 = HEX(cellOld)
            return var1, var2

        elif(int(ibrav)==5):
            try:
                cellOld = cellOld.getA()
            except:
                pass


            tx = cellOld[0][0]

            c = (((tx**2)*2)-1.0)
            c = -c



            alpha=np.arccos(c)


            if alpha < np.pi/2.0:
                var1, var2 = RHL1(cellOld)
                return var1, var2

            elif alpha > np.pi/2.0:
                var1, var2 = RHL2(cellOld)
                return var1, var2


        elif(int(ibrav)==6):
            var1, var2 = TET(cellOld)
            return var1, var2

        elif(int(ibrav)==7):
            try:
                cellOld = cellOld.getA()
            except:
                pass
            a = cellOld[1][0]*2
            c = cellOld[1][2]*2

            if(c < a):
                var1, var2 = BCT1(cellOld)
                return var1, var2


            elif(c > a):
                var1, var2 = BCT2(cellOld)
                return var1, var2

            else:
                var1, var2 = BCC(cellOld)
                return var1, var2

        elif(int(ibrav)==8):
            var1, var2 = ORC(cellOld)
            return var1, var2

        elif(int(ibrav)==9):
            var1, var2 = ORCC(cellOld)
            return var1, var2

        elif(int(ibrav)==10):
            try:
                cellOld = cellOld.getA()
            except:
                pass
            a1 = cellOld[0][0]*2
            b1 = cellOld[1][1]*2
            c1 = cellOld[2][2]*2
            myList = [a1, b1, c1]
            c = max(myList)
            a = min(myList)
            myList.remove(a)
            myList.remove(c)
            b = myList[0]

            if(1.0/a**2 > 1.0/b**2 + 1.0/c**2):
                var1, var2 = ORCF1(cellOld)
                return var1, var2

            elif(1.0/a**2 < 1.0/b**2 + 1.0/c**2):
                var1, var2 = ORCF2(cellOld)
                return var1, var2

            elif(1.0/a**2 == 1.0/b**2 + 1.0/c**2):
                '''
                var1, var2 = ORCF3(cellOld)
                return var1, var2
                '''
                raise Exception
        elif(int(ibrav)==11):

            var1, var2 = ORCI(cellOld)
            return var1, var2
        elif(int(ibrav)==14):

            a,b,c,alpha,beta,gamma =  free2abc(cellOld,cosine=False,bohr=False,string=False)

            minAngle = min([float(alpha),float(beta),float(gamma)])
            maxAngle = max([float(alpha),float(beta),float(gamma)])
            if alpha==90.0 or beta==90.0 or gamma==90.0:
                if alpha>=90.0 or beta>=90.0 or gamma>=90.0:
                    var1,var2 = TRI2A(cellOld)
                    return var1,var2
                if alpha<=90.0 or beta<=90.0 or gamma<=90.0:
                    var1,var2 = TRI2B(cellOld)
                    return var1,var2
            elif minAngle>90.0:
                var1,var2 = TRI1A(cellOld)
                return var1,var2
            elif maxAngle<90:
                var1,var2 = TRI1B(cellOld)
                return var1,var2

        '''
        elif(int(ibrav)==12):
            var1, var2 = MCL(cellOld)
            return var1, var2
        '''
#######################################################################
    if ibrav==0:
        sys.exit('IBRAV = 0 not permitted')
    if ibrav < 0:
        print('Lattice type %s is not implemented' % ibrav)
        logging.error('The ibrav value from expresso has not yet been implemented to the framework')
        raise Exception

    if ibrav==5:
        cellOld/=alat

    special_points, band_path = choose_lattice(ibrav, cellOld)

    return special_points, band_path



def kpnts_interpolation_mesh(ibrav,alat,cell,dk):
    '''
    Get path between HSP

    Arguments:
          dk (float): distance between points

    Returns:
          kpoints : array of arrays kx,ky,kz
          numK    : Total no. of k-points

    '''

    def kdistance(hs, p1, p2):
        g = np.dot(hs.T, hs)
        p1, p2 = np.array(p1), np.array(p2)
        d = p1 - p2
        dist2 = np.dot(d.T, np.dot(g, d).T)
        return np.sqrt(dist2)

    def getSegments(path):
        segments = path.split('|')
        return segments

    def getPoints(pathSegment):
        pointsList = pathSegment.split('-')
        return pointsList
    def getNumPoints(path):
        list1 = getSegments(path)
        numPts = 0
        for index in (list1):
            numPts += len(getPoints(index))
        return numPts

    if ibrav==0:
        sys.exit('IBRAV = 0 not permitted')
    if ibrav<0:
        print('Lattice type %s is not implemented') % ibrav
        logging.error('The ibrav value from QE has not yet been implemented')
        raise Exception

    totalK=0
    special_points, band_path = _getHighSymPoints(ibrav,alat,cell)
    hs = 2*np.pi*np.linalg.inv(cell)  # reciprocal lattice
    segs = getSegments(band_path)

    kx = np.array([])
    ky = np.array([])
    kz = np.array([])

    for index in segs:

        a = getPoints(index) #gets the points in each segment of path separated by |
        point1 = None
        point2 = None

        for index2 in xrange(len(a)-1):
            try:
                point1 = a[index2]
                point2 = a[index2+1]
                p1 = special_points[point1]
                p2 = special_points[point2]

                newDK = (2*np.pi/alat)*dk
                numK = int(np.ceil((kdistance(hs, p1, p2)/newDK)))
                totalK+=numK

                numK = str(numK)

                a0 = np.linspace(p1[0],p2[0],numK).astype(np.float16)
                a1 = np.linspace(p1[1],p2[1],numK).astype(np.float16)
                a2 = np.linspace(p1[2],p2[2],numK).astype(np.float16)

                kx = np.concatenate((kx,a0))
                ky = np.concatenate((ky,a1))
                kz = np.concatenate((kz,a2))

            except Exception as e:
                print(e)

        kpoints = np.array([kx,ky,kz])

    return (kpoints)
