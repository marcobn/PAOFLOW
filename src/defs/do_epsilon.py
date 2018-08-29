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

    d_tensor = arrays['d_tensor']

    for ispin in range(attributes['nspin']):
      for n in range(d_tensor.shape[0]):
        ipol = d_tensor[n][0]
        jpol = d_tensor[n][1]

        epsi, epsr, jdos = do_epsilon(data_controller, ene, metal, kramerskronig, ispin, ipol, jpol)

        indices = (LL[ipol], LL[jpol], ispin)

        fepsi = 'epsi_%s%s_%d.dat'%indices
        data_controller.write_file_row_col(fepsi, ene, epsi)

#        fepsr = 'epsr_%s%s_%d.dat'%indices
#        data_controller.write_file_row_col(fepsr, ene, epsr)

        fjdos = 'jdos_%s%s_%d.dat'%indices
        data_controller.write_file_row_col(fjdos, ene, jdos)


def do_epsilon ( data_controller, ene, metal, kramerskronig, ispin, ipol, jpol ):
#(E_k,pksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,emin,emax,deltak,deltak2,smearing,kramerskronig):
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Compute the dielectric tensor

    arrays,attributes = data_controller.data_dicts()

    esize = ene.size
    if ene[0]==0.0:
      ene[0]=0.00001

    #=======================
    # Im
    #=======================

    #if attributes['smearing'] == None:
    epsi_aux,jdos_aux = epsi_loop(data_controller, ene, metal, ispin, ipol, jpol)
    #else:
    #    print('Smearing not done in PAOFLOW_CLASS')
    #    quit()
        #epsi_aux,jdos_aux = smear_epsi_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,\
        #                  ispin,metal,deltak,deltak2,smearing)

#    epsi = np.zeros((3,3,esize), dtype=float)
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

    orig_over_err = np.geterr()['over']
    np.seterr(over='raise')

    arrays,attributes = data_controller.data_dicts()

    esize = ene.size

    jdos = np.zeros(esize, dtype=float)
    epsi = np.zeros(esize, dtype=float)
    #epsi = np.zeros((3,3,esize), dtype=float)

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

    #E_diff_nm = np.zeros((snktot, ni), order='C')
    #E_nm_pksp2 = np.zeros((snktot, ni),order='C')
    #f_nm = np.zeros((snktot, ni), order='C')

    E_diff_nm = np.ascontiguousarray((np.reshape(arrays['E_k'][:,:bnd,ispin], (snktot,1,bnd))\
                     -np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,bnd,1)))[:,uind[0],uind[1]])

    f_nm = np.ascontiguousarray((np.reshape(fn,(snktot,bnd,1))-np.reshape(fn,(snktot,1,bnd)))[:,uind[0],uind[1]])
    fn = None

#    f_nm_pksp2=f_nm*np.real(pksp[:,ipol,[:,uind[0],uind[1]],ispin]*\
#                                np.transpose(pksp[:,jpol,:,:,ispin],(0,2,1))[:,uind[0],uind[1]])
    f_nm_pksp2 = np.ascontiguousarray(f_nm*np.real(arrays['pksp'][:,ipol,uind[0],uind[1],ispin]*\
                                arrays['pksp'][:,jpol,uind[1],uind[0],ispin]))

#    dk2_nm = np.ascontiguousarray(deltak2[:,uind[0],uind[1],ispin])
#    dfunc=np.zeros_like(f_nm_pksp2)
#    sq2_dk2 = 1.0/(np.sqrt(np.pi)*deltak2[:,uind[0],uind[1],ispin])
    dk2_nm = arrays['deltakp2'][:,uind[0],uind[1],ispin]
    sq2_dk2 = 1.0/(np.sqrt(np.pi)*arrays['deltakp2'][:,uind[0],uind[1],ispin])

    if rank == 0:
        print(uind)
        print(attributes['nawf'])
        print(attributes['bnd'])
        print(arrays['E_k'].shape)
        print(arrays['deltakp'].shape)
        print(arrays['deltakp2'].shape)
        print(E_diff_nm.shape)
        print(f_nm.shape)
        print(dk2_nm.shape)
        print(sq2_dk2.shape)

    delta = attributes['delta']
    for i,e in enumerate(ene):
        if smearing == None:
            pass
        elif smearing == 'gauss':
            dfunc = np.exp(-((e-E_diff_nm)/dk2_nm)**2)*sq2_dk2
        elif smearing == 'm-p':
            dfunc = metpax(E_diff_nm, e, arrays['deltakp2'][:,uind[0],uind[1],ispin])
        epsi[i] = np.sum(dfunc*f_nm_pksp2/(e**2+delta**2))
        jdos[i] = np.sum(dfunc*f_nm)

    if metal:
        if smearing == None:
            pass
        elif smearing == 'gauss':
            fnF = gaussian(arrays['E_k'][:,:bnd,ispin], Ef, .03*arrays['deltakp'][:,:bnd,ispin])
        elif smearing == 'm-p':
            fnF = metpax(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])

        diag_ind = np.diag_indices(bnd)
        fnF *= np.real(arrays['pksp'][:,ipol,diag_ind[0],diag_ind[1],ispin]*arrays['pksp'][:,jpol,diag_ind[0],diag_ind[1],ispin])
        sq2_dk1 = (1./(np.sqrt(np.pi)*arrays['deltakp'][:,:bnd,ispin]) if smearing=='gauss' else None)
        for i,e in enumerate(ene):
            if smearing == None:
                pass
            elif smearing == 'gauss':
                dfunc = np.exp(-(e/arrays['deltakp'][:,:bnd,ispin])**2)*sq2_dk1
            elif smearing == 'm-p':
                dfunc = metpax(0., e, arrays['deltakp'][:,:bnd,ispin])
            epsi[i] = np.sum(dfunc*fnF/e)

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * attributes['omega'])*kq_wght
    jdos *= kq_wght

    np.seterr(over=orig_over_err)
    return(epsi, jdos)

def epsi_poop ( data_controller, ene, metal, ispin, ipol, jpol):
    import numpy as np
    from constants import EPS0, EVTORY

    orig_over_err = np.geterr()['over']
    np.seterr(over='raise')

    arrays,attributes = data_controller.data_dicts()

    esize = ene.size

    jdos = np.zeros(esize, dtype=float)
    epsi = np.zeros(esize, dtype=float)
    #epsi = np.zeros((3,3,esize), dtype=float)

    bnd = attributes['bnd']
    temp = attributes['temp']
    delta = attributes['delta']
    snktot = arrays['pksp'].shape[0]
    smearing = attributes['smearing']

    Ef = 0.
    eps=1.e-8
    kq_wght = 1./attributes['nkpnts']

    for n in range(bnd):
        fn = None
        if smearing == None:
            fn = 1./(1.+np.exp(arrays['E_k'][:,n,ispin]/temp))
        elif smearing == 'gauss':
            fn = intgaussian(arrays['E_k'][:,n,ispin], Ef, arrays['deltakp'][:,n,ispin])
        elif smearing == 'm-p':
            fn = intmetpax(arrays['E_k'][:,n,ispin], Ef, arrays['deltakp'][:,n,ispin])
        for m in range(bnd):
            fm = None
            if smearing == None:
                fm = 1./(1.+np.exp(arrays['E_k'][:,m,ispin]/temp))
            if smearing == 'gauss':
                fm = intgaussian(arrays['E_k'][:,m,ispin], Ef, arrays['deltak'][:,m,ispin])
            elif smearing == 'm-p':
                fm = intmetpax(arrays['E_k'][:,m,ispin], Ef, arrays['deltakp'][:,m,ispin])


            ediff = ene + np.reshape(np.repeat((arrays['E_k'][:,n,ispin] - arrays['E_k'][:,m,ispin]),esize), (snktot, esize))
            dfunc = np.exp(-(ediff/delta)**2)/np.sqrt(np.pi)
            fdiff = np.reshape(np.repeat((fn - fm), esize), (snktot, esize))
            pkipkj = abs(arrays['pksp'][:,ipol,n,m,ispin]*arrays['pksp'][:,ipol,n,m,ispin])

            jdt = kq_wght*dfunc/delta
            jdos[:] += np.sum((jdt*fdiff).T, axis=1)
            epsi[:] += np.sum(pkipkj*(jdt*fdiff/(ene**2+delta**2)).T, axis=1)
            #epsi[ipol,jpol,:] += np.sum(pkipkj*(jdt*fdiff/(ene**2+delta**2)).T, axis=1)

            fn = fm = ediff = fdiff = None
####
##            epsi[ipol,jpol,:] += np.sum(((1.0/(ene**2+delta**2) * \
###                          kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
##                           abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
##            jdos[:] += np.sum((( \
##                           kq_wght[0] /delta * dfunc * ((fn - fm)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
##                           1.0),axis=1)
            if metal and n == m:
                fnF = np.empty(snktot, dtype=float)
                for ik in range(snktot):
                    try:
                        fnF[ik] = .5/(1.+np.cosh(arrays['E_k'][ik,n,ispin]/temp))
                    except:
                        fnF[ik] = 1.0e8
                epsi[:] += np.sum(pkipkj*(jdt*np.reshape(np.repeat(fnF/temp,esize),(snktot,esize))/ene).T, axis=1)
                #epsi[ipol,jpol,:] += np.sum(pkipkj*(jdt*np.reshape(np.repeat(fnF/temp,esize),(snktot,esize))/ene).T, axis=1)
##                epsi[ipol,jpol,:] += np.sum(((1.0/ene * \
##                               kq_wght[0] /delta * dfunc * ((fnF/temp)*np.ones((pksp.shape[0],ene.size),dtype=float).T).T).T* \
##                               abs(pksp[:,ipol,n,m,ispin] * pksp[:,jpol,m,n,ispin])),axis=1)
            jdt = dfunc = pkipkj = None

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * attributes['omega'])


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

    return(epsr)


def smear_epsi_loop(ipol,jpol,ene,E_k,pksp,kq_wght,nawf,omega,delta,temp,ispin,metal,deltak,deltak2,smearing):

    epsi = np.zeros((3,3,ene.size),dtype=float)
    jdos = np.zeros((ene.size),dtype=float)
    Ef = 0.0
    deltat = 0.1

    if smearing == 'gauss':
        fn = intgaussian(E_k[:,:,ispin],Ef,deltak[:,:,ispin])
    elif smearing == 'm-p':
        fn = intmetpax(E_k[:,:,ispin],Ef,deltak[:,:,ispin])
    else:
        sys.exit('smearing not implemented')

    '''upper triangle indices'''
    uind = np.triu_indices(nawf,k=1)
    nk=pksp.shape[0]

    E_diff_nm=np.zeros((nk,len(uind[0])),order='C')
    E_nm_pksp2=np.zeros((nk,len(uind[0])),order='C')
    f_nm=np.zeros((nk,len(uind[0])),order='C')

    E_diff_nm = np.ascontiguousarray((np.reshape(E_k[:,:,ispin],(nk,1,nawf))\
                     -np.reshape(E_k[:,:,ispin],(nk,nawf,1)))[:,uind[0],uind[1]])

    f_nm=np.ascontiguousarray((np.reshape(fn,(nk,nawf,1))-np.reshape(fn,(nk,1,nawf)))[:,uind[0],uind[1]])
#    f_nm_pksp2=f_nm*np.real(pksp[:,ipol,[:,uind[0],uind[1]],ispin]*\
#                                np.transpose(pksp[:,jpol,:,:,ispin],(0,2,1))[:,uind[0],uind[1]])
    f_nm_pksp2=np.ascontiguousarray(f_nm*np.real(pksp[:,ipol,uind[0],uind[1],ispin]*\
                                pksp[:,jpol,uind[1],uind[0],ispin]))




    fn = None

    dk2_nm = np.ascontiguousarray(deltak2[:,uind[0],uind[1],ispin])
    dfunc=np.zeros_like(f_nm_pksp2)
    sq2_dk2 = 1.0/(np.sqrt(np.pi)*deltak2[:,uind[0],uind[1],ispin])




    # gaussian smearing


    for e in range(ene.size):
        if smearing=='gauss':
            np.exp(-((ene[e]-E_diff_nm)/dk2_nm)**2,out=dfunc)
            dfunc *= sq2_dk2
        if smearing=='m-p':
            dfunc = metpax(E_diff_nm,ene[e],deltak2[:,uind[0],uind[1],ispin])

        epsi[ipol,jpol,e] = np.sum(1.0/(ene[e]**2+delta**2)*dfunc*f_nm_pksp2)
        jdos[e] = np.sum(f_nm*dfunc)

    sq2_dk2    = None
    dfunc      = None
    f_nm       = None
    f_nm_pksp2 = None
    E_diff_nm  = None
    dk2_nm     = None

    if metal:
        dk_cont = np.ascontiguousarray(deltak[:,:,ispin])
        if smearing == 'gauss':
            fnF = gaussian(E_k[:,:,ispin],Ef,0.03*deltak[:,:,ispin])
        elif smearing == 'm-p':
            fnF = metpax(E_k[:,:,ispin],Ef,deltak[:,:,ispin])

        dfunc=np.zeros_like(fnF)
        diag_ind = np.diag_indices(nawf)
        fnF *= np.ascontiguousarray(np.real(pksp[:,ipol,diag_ind[0],diag_ind[1],ispin]*pksp[:,jpol,diag_ind[0],diag_ind[1],ispin]))
        sq2_dk1 = 1.0/(np.sqrt(np.pi)*deltak[:,:,ispin])



        for e in range(ene.size):
            if smearing=='gauss':
                np.exp(-((ene[e])/dk_cont)**2,out=dfunc)
                dfunc *= sq2_dk1
            elif smearing=='m-p':
                dfunc = metpax(0.0,ene[e],deltak[:,:,ispin])
            epsi[ipol,jpol,e] += np.sum(1.0/(ene[e])*dfunc*fnF)

        sq2_dk1 = None
        dfunc   = None
        fnF     = None
        d2_cont = None

    epsi *= 4.0*np.pi/(EPS0 * EVTORY * omega)*kq_wght[0]
    jdos *= kq_wght[0]

    return(epsi,jdos)



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

    return(epsr)

