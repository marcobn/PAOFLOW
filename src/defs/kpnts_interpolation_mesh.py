#
# PAOFLOW
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
import sys,os
import copy


from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()



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


###############################################################################
###############################################################################

    #get a,b,c of QE convention conventional cell from primitive lattice vecs
    if ibrav == 1:
        a=np.abs(cellOld[0][0])*2.0
    if ibrav == 2:
        a=np.abs(cellOld[0][0])*2.0
    if ibrav == 3:
        a=np.abs(cellOld[0][0])*2.0
    if ibrav == 4:
        a=np.abs(cellOld[0][0])
        b=a
        c=np.abs(cellOld[2][2]*a)
    if ibrav == 5:
        a=np.sqrt(cellOld[0].dot(cellOld[0].T))
        alpha=np.arccos(cellOld[0].dot(cellOld[1].T)/(a**2))
        alpha_deg = alpha*180.0/np.pi
    if ibrav == 6:
        a=np.abs(cellOld[0][0])
        c=np.abs(cellOld[2][2]*a)
    if ibrav == 7:
        a=np.abs(cellOld[0][0])*2.0
        c=np.abs(cellOld[2][2]*a)*2.0
    if ibrav == 9:
        a=np.abs(cellOld[0][0])*2.0
        b=np.abs(cellOld[1][1])*2.0*a
        c=np.abs(cellOld[2][2])*a
    if ibrav == 10:
        a=np.abs(cellOld[0][0])*2.0
        b=np.abs(cellOld[1][1])*2.0*a
        c=np.abs(cellOld[2][2])*2.0*a
    if ibrav == 11:
        a=np.abs(cellOld[0][0])*2.0
        b=np.abs(cellOld[1][1])*2.0*a
        c=np.abs(cellOld[2][2])*2.0*a
    if ibrav == 12:
        '''swapped compared to AFLOW..ABC->CAB'''
        a=np.sqrt(cellOld[0].dot(cellOld[0].T))
        b=np.sqrt(cellOld[1].dot(cellOld[1].T))
        c=np.sqrt(cellOld[2].dot(cellOld[2].T))
        alpha=np.arccos(cellOld[1].dot(cellOld[2].T)/(c*b))
        beta =np.arccos(cellOld[0].dot(cellOld[2].T)/(a*c))
        gamma=np.arccos(cellOld[0].dot(cellOld[1].T)/(a*b))


    if ibrav==14:
        a=np.sqrt(cellOld[0].dot(cellOld[0].T))
        b=np.sqrt(cellOld[1].dot(cellOld[1].T))
        c=np.sqrt(cellOld[2].dot(cellOld[2].T))
        alpha=np.arccos(cellOld[1].dot(cellOld[2].T)/(c*b))
        beta =np.arccos(cellOld[0].dot(cellOld[2].T)/(a*c))
        gamma=np.arccos(cellOld[0].dot(cellOld[1].T)/(a*b))


    if   ibrav==1:  ibrav_var =  'CUB'
    elif ibrav==2:  ibrav_var =  'FCC'
    elif ibrav==3:  ibrav_var =  'BCC'
    elif ibrav==4:  ibrav_var =  'HEX'
    elif ibrav==6:  ibrav_var =  'TET'
    elif ibrav==8:  ibrav_var =  'ORC'
    elif ibrav==9:  ibrav_var =  'ORCC'
    elif ibrav==11: ibrav_var =  'ORCI'
    elif ibrav==12: ibrav_var =  'MCL'

    elif ibrav==5:
        if alpha_deg   < 90.0: ibrav_var = 'RHL1'
        elif alpha_deg > 90.0: ibrav_var = 'RHL2'
    elif ibrav==7:
        if(c < a):   ibrav_var =  'BCT1'
        elif(c > a): ibrav_var =  'BCT2'
        else:        ibrav_var =  'BCC'
    elif ibrav==10:

        if    (1.0/a**2 > 1.0/b**2+1.0/c**2): ibrav_var =  'ORCF1'
        elif  np.isclose(1.0/a**2, 1.0/b**2+1.0/c**2,1.e-2): ibrav_var =  'ORCF3'
        elif  (1.0/a**2 < 1.0/b**2+1.0/c**2): ibrav_var =  'ORCF2'

    elif(int(ibrav)==14):
        minAngle = np.amin([alpha,beta,gamma])
        maxAngle = np.amax([alpha,beta,gamma])
        if alpha==90.0 or beta==90.0 or gamma==90.0:
            if alpha>=90.0 or beta>=90.0 or gamma>=90.0: ibrav_var =  'TRI2A'
            if alpha<=90.0 or beta<=90.0 or gamma<=90.0: ibrav_var =  'TRI2B'
        elif minAngle>90.0:                              ibrav_var =  'TRI1A'
        elif maxAngle<90:                                ibrav_var =  'TRI1B'
###############################################################################
###############################################################################
    if ibrav_var=='CUB':
        band_path = 'gG-X-M-gG-R-X|M-R'
        special_points = {'gG'   : (0.0, 0.0, 0.0),
                           'M'   : (0.5, 0.5, 0.0),
                           'R'   : (0.5, 0.5, 0.5),
                           'X'   : (0.0, 0.5, 0.0)}
                           
    if ibrav_var=='FCC':
        band_path = 'gG-X-W-K-gG-L-U-W-L-K|U-X'
        special_points = {'gG'   : (0.0, 0.0, 0.0),
                          'K'    : (0.375, 0.375, 0.750),
                          'L'    : (0.5, 0.5, 0.5),
                          'U'    : (0.625, 0.250, 0.625),
                          'W'    : (0.5, 0.25, 0.75),
                          'X'    : (0.5, 0.0, 0.5)}
                          
    if ibrav_var=='BCC':
        band_path = 'gG-H-N-gG-P-H|P-N'
        special_points = {'gG'   : (0, 0, 0),
                          'H'    : (0.5, -0.5, 0.5),
                          'P'    : (0.25, 0.25, 0.25,), 
                          'N'    : (0.0, 0.0, 0.5)}
            
    if ibrav_var=='HEX':
        band_path = 'gG-M-K-gG-A-L-H-A|L-M|K-H'
        special_points = {'gG'   : (0, 0, 0),
                          'A'    : (0.0, 0.0, 0.5),
                          'H'    : (1.0/3.0, 1.0/3.0, 0.5),
                          'K'    : (1.0/3.0, 1.0/3.0, 0.0),
                          'L'    : (0.5, 0.0, 0.5),
                          'M'    : (0.5, 0.0, 0.0)}
        
    if ibrav_var=='RHL1':
        eta = (1.0 + 4.0*np.cos(alpha))/(2.0 + 4.0*np.cos(alpha))
        

        nu =0.75-eta/2.0
        band_path = 'gG-L-B1|B-Z-gG-X|Q-F-P1-Z|L-P'
        special_points = {'gG'   : (0.0, 0.0, 0.0),
                          'B'    : (eta, 0.5, 1.0-eta),
                          'B1'   : (0.5, 1.0-eta, eta-1.0),
                          'F'    : (0.5, 0.5, 0.0),
                          'L'    : (0.5, 0.0, 0.0),
                          'L1'   : (0.0, 0.0, -0.5),
                          'P'    : (eta, nu, nu),
                          'P1'   : (1.0-nu, 1.0-nu, 1.0-eta),
                          'P2'   : (nu, nu, eta-1.0),
                          'Q'    : (1.0-nu, nu, 0.0),
                          'X'    : (nu, 0.0, -nu),
                          'Z'    : (0.5, 0.5, 0.5)}
         
    if ibrav_var=='RHL2':
        eta=1.0/(2.0*np.tan(alpha/2.0)**2)
        nu =0.75-eta/2.0
        band_path = 'gG-P-Z-Q-gG-F-P1-Q1-L-Z'
        special_points = {'gG'   : (0.0, 0.0, 0.0),
                          'F'    : (0.5, -0.5, 0.0),
                          'L'    : (0.5, 0.0, 0.0),
                          'P'    : (1.0-nu, -nu, 1.0-nu),
                          'P1'   : (nu, nu-1.0, nu-1.0),
                          'Q'    : (eta, eta, eta),
                          'Q1'   : (1.0-eta, -eta, -eta),
                          'Z'    : (0.5, -0.5, 0.5)} 

    if ibrav_var=='TET':
        band_path = 'gG-X-M-gG-Z-R-A-Z|X-R|M-A'
        special_points = {'gG'   : (0.0, 0.0, 0.0),
                          'A'    : (0.5, 0.5, 0.5),
                          'M'    : (0.5, 0.5, 0.0),
                          'R'    : (0.0, 0.5, 0.5),
                          'X'    : (0.0, 0.5, 0.0),
                          'Z'    : (0.0, 0.0, 0.5)}

    if ibrav_var=='BCT1':
       eta = (1.0+(c/a)**2)/4.0
       band_path = 'gG-X-M-gG-Z-P-N-Z1-M|X-P'
       special_points = {'gG'    : (0.0, 0.0, 0.0),
                         'M'     : (-0.5, 0.5, 0.5),
                         'N'     : (0.0, 0.5, 0.0),
                         'P'     : (0.25, 0.25, 0.25),
                         'X'     : (0.0, 0.0, 0.5),
                         'Z'     : (eta, eta, -eta),
                         'Z1'    : (-eta, 1.0-eta, eta)}
         
    if ibrav_var=='BCT2':
       band_path = 'gG-X-Y-gS-gG-Z-gS1-N-P-Y1-Z|X-P'
       eta = (1.0+(a/c)**2)/4.0
       zeta = 0.5*(a/c)**2

       special_points = {'gG'    : (0.0, 0.0, 0.0),
                         'N'     : (0.0, 0.5, 0.0),
                         'P'     : (0.25, 0.25, 0.25),
                         'gS'    : (-eta, eta, eta),
                         'gS1'   : (eta, 1-eta, -eta),
                         'X'     : (0.0, 0.0, 0.5),
                         'Y'     : (-zeta, zeta, 0.5),
                         'Y1'    : (0.5, 0.5, -zeta),
                         'Z'     : (0.5, 0.5, -0.5)}
         
    if ibrav_var=='ORC':
         band_path = 'gG-X-S-Y-gG-Z-U-R-T-Z|Y-T|U-X|S-R'
         special_points = {'gG'  : (0.0, 0.0, 0.0),
                           'R'   : (0.5, 0.5, 0.5),
                           'S'   : (0.5, 0.5, 0.0),
                           'T'   : (0.0, 0.5, 0.5),
                           'U'   : (0.5, 0.0, 0.5),
                           'X'   : (0.5, 0.0, 0.0),
                           'Y'   : (0.0, 0.5, 0.0),
                           'Z'   : (0.0, 0.0, 0.5)}

    if ibrav_var=='ORCF1':
       band_path = 'gG-Y-T-Z-gG-X-A1-Y|T-X1|X-A-Z|L-gG'
       sorted_lat=np.sort([a,b,c])
       a=sorted_lat[0]
       b=sorted_lat[1]
       c=sorted_lat[2]
       eta =(1.0+(a/b)**2+(a/c)**2)/4.0
       zeta=(1.0+(a/b)**2-(a/c)**2)/4.0
       special_points = {'gG'    : (0.0, 0.0, 0.0),
                         'A'     : (0.5, 0.5 + zeta, zeta),
                         'A1'    : (0.5, 0.5-zeta, 1.0-zeta),
                         'L'     : (0.5, 0.5, 0.5),
                         'T'     : (1.0, 0.5, 0.5),
                         'X'     : (0.0, eta, eta),
                         'X1'    : (1.0, 1.0-eta, 1.0-eta),
                         'Y'     : (0.5, 0.0, 0.5),
                         'Z'     : (0.5, 0.5, 0.0)}

    if ibrav_var=='ORCF2':
       band_path = 'gG-Y-C-D-X-gG-Z-D1-H-C|C1-Z|X-H1|H-Y|L-gG'
       # sorted_lat=np.sort([a,b,c])
       # a=sorted_lat[0]
       # b=sorted_lat[1]
       # c=sorted_lat[2]
       eta =(1.0+(a/b)**2-(a/c)**2)/4.0
       phi =(1.0+(c/b)**2-(c/a)**2)/4.0
       delta =(1.0+(b/a)**2-(b/c)**2)/4.0

       special_points = {'gG'    : (0.0, 0.0, 0.0),
                         'C'     : (0.5, 0.5-eta, 1.0-eta),
                         'C1'    : (0.5, 0.5+eta, eta),
                         'D'     : (0.5-delta, 0.5, 1.0-delta),
                         'D1'    : (0.5+delta, 0.5, delta),
                         'L'     : (0.5, 0.5, 0.5),
                         'H'     : (1.0-phi, 0.5-phi, 0.5),
                         'H1'    : (phi, 0.5+phi, 0.5),
                         'X'     : (0.0, 0.5, 0.5),
                         'Y'     : (0.5, 0.0, 0.5),
                         'Z'     : (0.5, 0.5, 0.0),}

    if ibrav_var=='ORCF3':
       band_path = 'gG-Y-T-Z-gG-X-A1-Y|X-A-Z|L-R'
       sorted_lat=np.sort([a,b,c])
       a=sorted_lat[0]
       b=sorted_lat[1]
       c=sorted_lat[2]
       eta =(1.0+(a/b)**2+(a/c)**2)/4.0
       zeta=(1.0+(a/b)**2+(a/c)**2)/4.0
       special_points = {'gG'    : (0.0, 0.0, 0.0),
                         'A'     : (0.5, 0.5 + zeta, zeta),
                         'A1'    : (0.5, 0.5-zeta, 1.0-zeta),
                         'L'     : (0.5, 0.5, 0.5),
                         'T'     : (1.0, 0.5, 0.5),
                         'X'     : (0.0, eta, eta),
                         'X1'    : (1.0, 1.0-eta, 1.0-eta),
                         'Y'     : (0.5, 0.0, 0.5),
                         'Z'     : (0.5, 0.5, 0.0)}

    if ibrav_var=='ORCC':
       band_path = 'gG-X-S-R-A-Z-gG-Y-X1-A1-T-Y|Z-T'

       zeta=(1.0+((a/b)**2))/4.0       

       special_points = {'gG'    : (  0.0, 0.0     , 0.0),
                         'A'     : ( zeta, zeta    , 0.5),
                         'A1'    : (-zeta, 1.0-zeta, 0.5),
                         'R'     : (  0.0, 0.5     , 0.5),
                         'S'     : (  0.0, 0.5     , 0.0),
                         'T'     : ( -0.5, 0.5     , 0.5),
                         'X'     : ( zeta, zeta    , 0.0),
                         'X1'    : (-zeta, 1.0-zeta, 0.0),
                         'Y'     : ( -0.5, 0.5     , 0.0),
                         'Z'     : (  0.0, 0.0     , 0.5)}
         
    if ibrav_var=='ORCI':
         band_path = 'gG-X-L-T-W-R-X1-Z-gG-Y-S-W|L1-Y|Y1-Z'
         chi   = (1.0  + (a/c)**2)/(4.0)
         eta   = (1.0  + (b/c)**2)/(4.0)
         delta = (b**2 - a**2    )/(4.0*c**2)
         mu    = (b**2 + a**2    )/(4.0*c**2)
         special_points = {'gG'   : (0, 0, 0),
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
                           'Z'    : (0.5, 0.5, -0.5)}

    if ibrav_var=='MCL':
         #abc->cab
         eta =  (1.0 - (b/a)*np.cos(np.pi-gamma))/(2.0*np.sin(np.pi-gamma)**2)
         nu =   0.5  - eta*(a/b)*np.cos(np.pi-gamma)
         band_path = 'gG-Y-H-C-E-M1-A-X-gG-Z-D-M|Z-A|D-Y|X-H1'
         special_points = {
                           'gG'    : (0.0, 0.0    , 0.0    ),
                           'A'     : (0.5, 0.5    , 0.0    ),
                           'C'     : (0.0, 0.5    , 0.5    ),
                           'D'     : (0.5, 0.0    , 0.5    ),
                           'D1'    : (0.5, 0.0    ,-0.5    ),
                           'E'     : (0.5, 0.5    , 0.5    ),
                           'H'     : (0.0, eta    , 1.0-nu ),
                           'H1'    : (0.0, 1.0-eta, nu     ),
                           'H2'    : (0.0, eta    ,-nu     ),
                           'M'     : (0.5, eta    , 1.0-nu ),
                           'M1'    : (0.5, 1.0-eta, nu     ),
                           'M2'    : (0.5, eta    ,-nu     ),
                           'X'     : (0.0, 0.5    , 0.0    ),
                           'Y'     : (0.0, 0.0    , 0.5    ),
                           'Y1'    : (0.0, 0.0    ,-0.5    ),
                           'Z'     : (0.5, 0.0    , 0.0    )}

         
   
    if ibrav_var=='TRI1A':         
        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG' 
        special_points = {'gG'    : (0.0,0.0,0.0),
                          'L'     : (0.5,0.5,0.0),
                          'M'     : (0.0,0.5,0.5),
                          'N'     : (0.5,0.0,0.5),
                          'R'     : (0.5,0.5,0.5),
                          'X'     : (0.5,0.0,0.0),
                          'Y'     : (0.0,0.5,0.0),
                          'Z'     : (0.0,0.0,0.5),}
        
    if ibrav_var=='TRI2A':        
        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'
        special_points = {'gG'    : (0.0,0.0,0.0),
                          'L'     : (0.5,0.5,0.0),
                          'M'     : (0.0,0.5,0.5),
                          'N'     : (0.5,0.0,0.5),
                          'R'     : (0.5,0.5,0.5),
                          'X'     : (0.5,0.0,0.0),
                          'Y'     : (0.0,0.5,0.0),
                          'Z'     : (0.0,0.0,0.5),}
 
    if ibrav_var=='TRI1B':        
        band_path = "X-gG-Y|L-gG-Z|N-gG-M|R-gG"
        special_points = {'gG'    : ( 0.0, 0.0,0.0),
                          'L'     : ( 0.5,-0.5,0.0),
                          'M'     : ( 0.0, 0.0,0.5),
                          'N'     : (-0.5,-0.5,0.5),
                          'R'     : ( 0.0,-0.5,0.5),
                          'X'     : ( 0.0,-0.5,0.0),
                          'Y'     : ( 0.5, 0.0,0.0),
                          'Z'     : (-0.5, 0.0,0.5),}

    if ibrav_var=='TRI2B':        
        band_path = 'X-gG-Y|L-gG-Z|N-gG-M|R-gG'
        special_points = {'gG'    : ( 0.0, 0.0,0.0),
                          'L'     : ( 0.5,-0.5,0.0),
                          'M'     : ( 0.0, 0.0,0.5),
                          'N'     : (-0.5,-0.5,0.5),
                          'R'     : ( 0.0,-0.5,0.5),

                          'X'     : ( 0.0,-0.5,0.0),
                          'Y'     : ( 0.5, 0.0,0.0),
                          'Z'     : (-0.5, 0.0,0.5),}


    aflow_conv = np.identity(3)
    qe_conv    = np.identity(3)

    if ibrav==2:
        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0],[ 1.0, 0.0, 1.0],[ 1.0, 1.0, 0.0]])/2.0     
        qe_conv    = np.asarray([[-1.0, 0.0, 1.0],[ 0.0, 1.0, 1.0],[-1.0, 1.0, 0.0]])/2.0
    if ibrav==3:
        aflow_conv = np.asarray([[-1.0, 1.0, 1.0],[ 1.0,-1.0, 1.0],[ 1.0, 1.0,-1.0]])/2.0     
        qe_conv    = np.asarray([[ 1.0, 1.0, 1.0],[-1.0, 1.0, 1.0],[-1.0,-1.0, 1.0]])/2.0     
    if ibrav==7:
        aflow_conv = np.asarray([[-1.0, 1.0, 1.0],[ 1.0,-1.0, 1.0],[ 1.0, 1.0,-1.0]])/2.0
        qe_conv    = np.asarray([[ 1.0,-1.0, 1.0],[ 1.0, 1.0, 1.0],[-1.0,-1.0, 1.0]])/2.0
    if ibrav==9:
        aflow_conv = np.asarray([[ 1.0,-1.0, 0.0],[ 1.0, 1.0, 0.0],[ 0.0, 0.0, 2.0]])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 0.0],[-1.0, 1.0, 0.0],[ 0.0, 0.0, 2.0]])/2.0
    if ibrav==10:
        aflow_conv = np.asarray([[ 0.0, 1.0, 1.0],[ 1.0, 0.0, 1.0],[ 1.0, 1.0, 0.0]])/2.0
        qe_conv    = np.asarray([[ 1.0, 0.0, 1.0],[ 1.0, 1.0, 0.0],[ 0.0, 1.0, 1.0]])/2.0  
    if ibrav==11:
        aflow_conv = np.asarray([[-1.0, 1.0, 1.0],[ 1.0,-1.0, 1.0],[ 1.0, 1.0,-1.0]])/2.0
        qe_conv    = np.asarray([[ 1.0, 1.0, 1.0],[-1.0, 1.0, 1.0],[-1.0,-1.0, 1.0]])/2.0
    if ibrav==12:
        aflow_conv = np.asarray([[ 0.0, 0.0, 1.0],[ 0.0, 1.0, 0.0],[ 1.0, 0.0, 0.0]])
        qe_conv    = np.asarray([[ 1.0, 0.0, 0.0],[ 0.0, 1.0, 0.0],[ 0.0, 0.0, 1.0]])
                                   

    for k,v in special_points.iteritems():
        first  = np.array(v).dot(np.linalg.inv(aflow_conv))
        if ibrav==9:
            second = qe_conv.T.dot(first)
        else:
            second = qe_conv.dot(first)
        special_points[k]=tuple((second).tolist())




    return special_points, band_path



def kpnts_interpolation_mesh(ibrav,alat,cell,b_vectors,nk,inputpath):
    '''
    Get path between HSP
    Arguments:
          nk (int): total number of points in path

    Returns:
          kpoints : array of arrays kx,ky,kz
          numK    : Total no. of k-points
    '''
    dk       = 0.00001
    points,_ = get_path(ibrav,alat,cell,dk)

    scaled_dk = dk*(points.shape[1]/nk)
    points    = None
    points,path_file = get_path(ibrav,alat,cell,scaled_dk)


    kq=np.copy(points)
    for n in xrange(kq.shape[1]):
        kq[:,n]=np.dot(kq[:,n],b_vectors)
    for i in xrange(kq.shape[1]):
        path_file+="%s %s %s\n"%(kq[0,i],kq[1,i],kq[2,i])

    if rank==0:
        with  open(os.path.join(inputpath,"kpath_points.txt"),"w") as pfo:
            pfo.write(path_file)

    return points

def get_path(ibrav,alat,cell,dk):

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

    hs = np.linalg.inv(cell)  # reciprocal lattice
    #hs = 2*np.pi*bcell
    segs = getSegments(band_path)

    kx = np.array([])
    ky = np.array([])
    kz = np.array([])

    path_file = ""

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

                newDK = (2.0*np.pi/alat)*dk
                numK = int(np.ceil((kdistance(hs, p1, p2)/newDK)))
                totalK+=numK

                path_file+="%s %s\n"%(point1,numK)
                
                numK = str(numK)

                a0 = np.linspace(p1[0],p2[0],numK).astype(np.float16)
                a1 = np.linspace(p1[1],p2[1],numK).astype(np.float16)
                a2 = np.linspace(p1[2],p2[2],numK).astype(np.float16)

                kx = np.concatenate((kx,a0))
                ky = np.concatenate((ky,a1))
                kz = np.concatenate((kz,a2))

            except Exception as e:
                print(e)


        path_file+="%s %s\n"%(a[-1],0)

    path_file+="\n"
    kpoints = np.array([kx,ky,kz])
        
    return kpoints,path_file
