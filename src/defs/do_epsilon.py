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

import sys

from load_balancing import *

from constants import *
from smearing import *


def do_dielectric_tensor ( data_controller, ene, metal, kramerskronig ):
    import numpy as np
    from mpi4py import MPI
    from constants import LL

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arrays,attributes = data_controller.data_dicts()

    smearing = attributes['smearing']
    if smearing != None and smearing != 'gauss' and smearing != 'm-p':
        if rank == 0:
            print('%s Smearing Not Implemented.'%smearing)
        quit()

    if smearing == None and rank == 0:
        print('Smearing is None\nOnly Re{epsilon} is being calculated')

    d_tensor = arrays['d_tensor']

    for ispin in range(attributes['nspin']):
      for n in range(d_tensor.shape[0]):
        ipol = d_tensor[n][0]
        jpol = d_tensor[n][1]

        epsi, epsr, jdos = do_epsilon(data_controller, ene, metal, kramerskronig, ispin, ipol, jpol)

        indices = (LL[ipol], LL[jpol], ispin)

        fepsi = 'epsi_%s%s_%d.dat'%indices
        data_controller.write_file_row_col(fepsi, ene, epsi)

        fjdos = 'jdos_%s%s_%d.dat'%indices
        data_controller.write_file_row_col(fjdos, ene, jdos)

#        fepsr = 'epsr_%s%s_%d.dat'%indices
#        data_controller.write_file_row_col(fepsr, ene, epsr)


def do_epsilon ( data_controller, ene, metal, kramerskronig, ispin, ipol, jpol ):
#(E_k,pksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,emin,emax,deltak,deltak2,smearing,kramerskronig):
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Compute the dielectric tensor

    arrays,attributes = data_controller.data_dicts()

    esize = ene.size
    if ene[0] == 0.:
      ene[0] = .00001

    #=======================
    # Im
    #=======================

    epsi_aux,jdos_aux = epsi_loop(data_controller, ene, metal, ispin, ipol, jpol)

    epsi = np.zeros(esize, dtype=float)
    comm.Allreduce(epsi_aux, epsi, op=MPI.SUM)
    epsi_aux = None

    jdos = np.zeros(esize, dtype=float)
    comm.Allreduce(jdos_aux, jdos, op=MPI.SUM)
    jods_aux = None

    #=======================
    # Re
    #=======================
    epsr = None
    if kramerskronig:
        if rank == 0:
          print('Kramers Kronig not implemented in PAOFLOW_CLASS')
        quit()
        # Load balancing
        #ini_ie, end_ie = load_balancing(size, rank, esize)

        #epsr_aux = epsr_kramkron(ini_ie,end_ie,ene,epsi,shift,ipol,jpol)

    else:
        if rank == 0 and metal:
          print('CAUTION: direct calculation of epsr in metals is not working!!!!!')

        if attributes['smearing'] == None:
            if rank == 0:
                print('Fixed smearing not implemented\nReal part of epsilon will not be calculated')

        if rank == 0:
            print('Real Smearing not done')
        #epsr_aux = smear_epsr_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,\
        #                      ispin,metal,deltak,deltak2,smearing)

    #epsr = np.zeros((3,3,esize), dtype=float)
    #comm.Allreduce(epsr_aux, epsr, op=MPI.SUM)
    #epsr_aux = None

    #epsr += 1.0

    return(epsi, epsr, jdos)



def epsi_loop ( data_controller, ene, metal, ispin, ipol, jpol):
    import numpy as np
    from mpi4py import MPI
    from constants import EPS0, EVTORY
    from smearing import intgaussian,gaussian,intmetpax,metpax

    rank = MPI.COMM_WORLD.Get_rank()

### What is this?
    orig_over_err = np.geterr()['over']
    np.seterr(over='raise')

    arrays,attributes = data_controller.data_dicts()

    jdos = np.zeros(esize, dtype=float)
    epsi = np.zeros(esize, dtype=float)

    esize = ene.size
    bnd = attributes['bnd']
    temp = attributes['temp']
    delta = attributes['delta']
    snktot = arrays['pksp'].shape[0]
    smearing = attributes['smearing']

    Ef = 0.
    eps=1.e-8
    kq_wght = 1./attributes['nkpnts']

    fn = None
    if smearing == None:
        fn = 1./(1.+np.exp(arrays['E_k'][:,:bnd,ispin]/temp))
    elif smearing == 'gauss':
        fn = intgaussian(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])
    elif smearing == 'm-p':
        fn = intmetpax(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])

    '''upper triangle indices'''
    uind = np.triu_indices(bnd, k=1)
    ni = len(uind[0])

    # E_kn-Ek_m for every k-point and every band combination (m,n)
    E_diff_nm = (np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,1,bnd))-np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,bnd,1)))[:,uind[0],uind[1]]

    # fn_n-fn_m for every k-point and every band combination (m,n)
    f_nm = (np.reshape(fn,(snktot,bnd,1))-np.reshape(fn,(snktot,1,bnd)))[:,uind[0],uind[1]]
    fn = None

    # <p_n|p_m> for every k-point and every band combination (m,n)
    pksp2 = arrays['pksp'][:,ipol,uind[0],uind[1],ispin]*arrays['pksp'][:,jpol,uind[1],uind[0],ispin]
    pksp2 = (abs(pksp2) if smearing is None else np.real(pksp2))

    sq2_dk2 = (1.0/(np.sqrt(np.pi)*arrays['deltakp2'][:,uind[0],uind[1],ispin]) if smearing=='gauss' else None)

    for i,e in enumerate(ene):
        if smearing is None:
            dfunc = np.exp(-((e-E_diff_nm)/delta)**2)/(delta*np.sqrt(np.pi))
        elif smearing == 'gauss':
            dfunc = np.exp(-((e-E_diff_nm)/arrays['deltakp2'][:,uind[0],uind[1],ispin])**2)*sq2_dk2
        elif smearing == 'm-p':
            dfunc = metpax(E_diff_nm, e, arrays['deltakp2'][:,uind[0],uind[1],ispin])
        epsi[i] = np.sum(dfunc*f_nm*pksp2/(e**2+delta**2))
        jdos[i] = np.sum(dfunc*f_nm)

    f_nm = dfunc = uind = pksp2 = sq2_dk2 = E_diff_nm = None

    if metal:
        fnF = None
        if smearing is None:
            fnF = np.empty((snktot,bnd), dtype=float)
            for n in range(bnd):
                for i in range(snktot):
                    try:
                        fnF[i,n] = .5/(1.+np.cosh(arrays['E_k'][i,n,ispin]/temp))
                    except:
                        fnF[i,n] = 1e8
            fnF /= temp
        elif smearing == 'gauss':
### Why .03* here?
            fnF = gaussian(arrays['E_k'][:,:bnd,ispin], Ef, .03*arrays['deltakp'][:,:bnd,ispin])
        elif smearing == 'm-p':
            fnF = metpax(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])

        diag_ind = np.diag_indices(bnd)

        pksp2 = arrays['pksp'][:,ipol,diag_ind[0],diag_ind[1],ispin]*arrays['pksp'][:,jpol,diag_ind[0],diag_ind[1],ispin]

        fnF *= (abs(pksp2) if smearing is None else np.real(pksp2))

        sq2_dk1 = (1./(np.sqrt(np.pi)*arrays['deltakp'][:,:bnd,ispin]) if smearing=='gauss' else None)

        pksp2 = None

        for i,e in enumerate(ene):
            if smearing is None:
                E_diff_nn = (np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,1,bnd))-np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,bnd,1)))[:,diag_ind[0],diag_ind[1]]
                dfunc = np.exp(-((e-E_diff_nn)/delta)**2)/(delta*np.sqrt(np.pi))
            elif smearing == 'gauss':
                dfunc = np.exp(-(e/arrays['deltakp'][:,:bnd,ispin])**2)*sq2_dk1
            elif smearing == 'm-p':
                dfunc = metpax(0., e, arrays['deltakp'][:,:bnd,ispin])
            epsi[i] += np.sum(dfunc*fnF/e)

        fnF = sq2_dk1 = diag_ind = None

    epsi *= 4.0*np.pi*kq_wght/(EPS0 * EVTORY * attributes['omega'])
    jdos *= kq_wght

    np.seterr(over=orig_over_err)
    return(epsi, jdos)


def smear_epsr_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):
    import numpy as np
    from smearing import intgaussian,intmetpax,gaussian,metpax

    epsr = np.zeros((3,3,ene.size),dtype=float)

    dfunc = np.zeros((pksp.shape[0],ene.size),dtype=float)
    effterm = np.zeros((pksp.shape[0],nawf),dtype=complex)
    Ef = 0.0

    for n in range(nawf):
        if smearing == 'gauss':
            fn = intgaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            fnF = gaussian(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        elif smearing == 'm-p':
            fn = intmetpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
            fnF = metpax(E_k[:,n,ispin],Ef,deltak[:,n,ispin])
        else:
            sys.exit('smearing not implemented')
        for m in range(nawf):
            if smearing == 'gauss':
                fm = intgaussian(E_k[:,m,ispin],Ef,deltak[:,m,ispin])
            elif smearing == 'm-p':
                fm = intmetpax(E_k[:,m,ispin],Ef,deltak[:,m,ispin])
            else:
                sys.exit('smearing not implemented')
            eig = ((E_k[:,m,ispin]-E_k[:,n,ispin])*np.ones((pksp.shape[0],ene.size),dtype=float).T).T
            om = ((ene*np.ones((pksp.shape[0],ene.size),dtype=float)).T).T
            del2 = (deltak2[:,n,m,ispin]*np.ones((pksp.shape[0],ene.size),dtype=float).T).T
            dfunc = 1.0/(eig - om + 1.0j*del2)
            if n != m:
                epsr[ipol,jpol,:] += np.real(np.sum((1.0/(eig**2+1.0j*delta**2) * \
                               kq_wght[0] * dfunc * ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T * \
                               (pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin]),axis=1))

    epsr *= 4.0/(EPS0 * EVTORY * omega)

    return epsr


def epsr_kramkron(ini_ie,end_ie,ene,epsi,shift,i,j):
    from scipy.integrate import simps

    epsr = np.zeros((3,3,ene.size),dtype=float)
    de = ene[1]-ene[0]

    if end_ie == ini_ie: return
    if ini_ie < 3: ini_ie = 3
    if end_ie == ene.size: end_ie = ene.size-1
    f_ene = intmetpax(ene,shift,1.0)
    for ie in range(ini_ie,end_ie):
        #epsr[i,j,ie] = 2.0/np.pi * ( np.sum(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
        #               np.sum(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )
        epsr[i,j,ie] = 2.0/np.pi * ( simps(ene[1:(ie-1)]*de*epsi[i,j,1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2)) + \
                       simps(ene[(ie+1):ene.size]*de*epsi[i,j,(ie+1):ene.size]*f_ene[(ie+1):ene.size]/(ene[(ie+1):ene.size]**2-ene[ie]**2)) )

    return epsr
