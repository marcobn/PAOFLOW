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
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

from __future__ import print_function
import numpy as np
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank = comm.Get_rank()

def clebsch_gordan():
    U0 = np.array([[0,1],
                   [1,0]])

    Sz0 = 0.5*np.array([[1, 0],
                    [0,-1]])

    sq13 = np.sqrt(1./3.)
    sq23 = np.sqrt(2./3.)
    U1 = np.array([[0,0   ,    0,sq13,-sq23,0],
                   [0,sq23,-sq13,   0,    0,0],
                   [0,   0,    0,   0,    0,1],
                   [0,   0,    0,sq23, sq13,0],
                   [0,sq13, sq23,   0,    0,0],
                   [1,   0,    0,   0,    0,0]])
    Sz1 = 0.5*np.array([[1, 0,0, 0,0, 0],
                        [0,-1,0, 0,0, 0],
                        [0, 0,1, 0,0, 0],
                        [0, 0,0,-1,0, 0],
                        [0, 0,0, 0,1, 0],
                        [0, 0,0, 0,0,-1]])

    sq15 = np.sqrt(1./5.)
    sq25 = np.sqrt(2./5.)
    sq35 = np.sqrt(3./5.)
    sq45 = np.sqrt(4./5.)

    U2 = np.array([[0,   0,    0,   0,    0,   0,    0,sq15,-sq45,0],
                   [0,   0,    0,   0,    0,sq25,-sq35,   0,    0,0],
                   [0,   0,    0,sq35,-sq25,   0,    0,   0,    0,0],
                   [0,sq45,-sq15,   0,    0,   0,    0,   0,    0,0],
                   [0,   0,    0,   0,    0,   0,    0,   0,    0,1],
                   [0,   0,    0,   0,    0,   0,    0,sq45, sq15,0],
                   [0,   0,    0,   0,    0,sq35, sq25,   0,    0,0],
                   [0,   0,    0,sq25, sq35,   0,    0,   0,    0,0],
                   [0,sq15, sq45,   0,    0,   0,    0,   0,    0,0],
                   [1,   0,    0,   0,    0,   0,    0,   0,    0,0]])

    Sz2 = 0.5*np.array([[1, 0,0, 0,0, 0,0, 0,0, 0],
                        [0,-1,0, 0,0, 0,0, 0,0, 0],
                        [0, 0,1, 0,0, 0,0, 0,0, 0],
                        [0, 0,0,-1,0, 0,0, 0,0, 0],
                        [0, 0,0, 0,1, 0,0, 0,0, 0],
                        [0, 0,0, 0,0,-1,0, 0,0, 0],
                        [0, 0,0, 0,0, 0,1, 0,0, 0],
                        [0, 0,0, 0,0, 0,0,-1,0, 0],
                        [0, 0,0, 0,0, 0,0, 0,1, 0],
                        [0, 0,0, 0,0, 0,0, 0,0,-1]])

    Sz = np.zeros((18,18),dtype=float)
    U = np.zeros((18,18),dtype=float)

    Sz[:2,:2] = Sz0
    Sz[2:8,2:8] = Sz1
    Sz[8:18,8:18] = Sz2

    U[:2,:2] = U0
    U[2:8,2:8] = U1
    U[8:18,8:18] = U2

    SzU = np.dot(U,Sz).dot(U.T)

    return(SzU)
