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

        if epsr is not None:
          fepsr = 'epsr_%s%s_%d.dat'%indices
          data_controller.write_file_row_col(fepsr, ene, epsr)


def do_epsilon ( data_controller, ene, metal, kramerskronig, ispin, ipol, jpol ):
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
        epsr_aux = epsr_kramerskronig(data_controller, ene, epsi)

    elif attributes['smearing'] is None:
        if rank == 0:
            print('Fixed smearing not implemented\nReal part of epsilon will not be calculated')
        return(epsi, None, jdos)

    else:
        if metal and rank == 0 and ispin == 0:
          print('CAUTION: direct calculation of epsr in metals is not working!!!!!')
        epsr_aux = smear_epsr_loop(data_controller, ene, ispin, ipol, jpol)

    epsr = np.zeros(esize, dtype=float)
    comm.Allreduce(epsr_aux, epsr, op=MPI.SUM)
    epsr_aux = None

    epsr += 1.0

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

    esize = ene.size
    bnd = attributes['bnd']
    temp = attributes['temp']
    delta = attributes['delta']
    snktot = arrays['pksp'].shape[0]
    smearing = attributes['smearing']

    Ef = 0.
    eps=1.e-8
    kq_wght = 1./attributes['nkpnts']

    jdos = np.zeros(esize, dtype=float)
    epsi = np.zeros(esize, dtype=float)

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


def smear_epsr_loop ( data_controller, ene, ispin, ipol, jpol ):
    import numpy as np
    from constants import EPS0, EVTORY
    from smearing import intgaussian,intmetpax

    arrays,attributes = data_controller.data_dicts()

    esize = ene.size
    bnd = attributes['bnd']
    delta = attributes['delta']
    snktot = arrays['pksp'].shape[0]
    smearing = attributes['smearing']

    Ef = 0.0
    kq_wght = 1./attributes['nkpnts']

    epsr = np.zeros(esize, dtype=float)

    for n in range(bnd):
        if smearing == 'gauss':
            fn = intgaussian(arrays['E_k'][:,n,ispin], Ef, arrays['deltakp'][:,n,ispin])
        elif smearing == 'm-p':
            fn = intmetpax(arrays['E_k'][:,n,ispin], Ef, arrays['deltak'][:,n,ispin])
        for m in range(bnd):
            if smearing == 'gauss':
                fm = intgaussian(arrays['E_k'][:,m,ispin], Ef, arrays['deltakp'][:,m,ispin])
            elif smearing == 'm-p':
                fm = intmetpax(arrays['E_k'][:,m,ispin], Ef, arrays['deltakp'][:,m,ispin])

            f_nm = np.reshape(np.repeat(fn-fm,esize), (snktot,esize))
            fm = None

            eig = np.reshape(np.repeat(arrays['E_k'][:,m,ispin]-arrays['E_k'][:,n,ispin],esize), (snktot,esize))
            del2 = np.reshape(np.repeat(arrays['deltakp2'][:,n,m,ispin],esize), (snktot,esize))
            dfunc = 1.0/(eig - ene + 1.0j*del2)
            del2 = None

            if n != m:
                pksp2 = arrays['pksp'][:,ipol,n,m,ispin] * arrays['pksp'][:,jpol,m,n,ispin]
                epsr += np.real(np.sum(pksp2*(dfunc*f_nm/(eig**2+1.0j*delta**2)).T, axis=1))

    epsr *= 4./(EPS0*EVTORY*attributes['omega']*attributes['nkpnts'])

    return epsr

#### Try to flatten one loop of epsr_loop
#   fn = None
#   if smearing == None:
#       fn = 1./(1.+np.exp(arrays['E_k'][:,:bnd,ispin]/temp))
#   elif smearing == 'gauss':
#       fn = intgaussian(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])
#   elif smearing == 'm-p':
#       fn = intmetpax(arrays['E_k'][:,:bnd,ispin], Ef, arrays['deltakp'][:,:bnd,ispin])

#   '''upper triangle indices'''
#   uind = np.triu_indices(bnd, k=1)
#   ni = len(uind[0])

#   # E_kn-Ek_m for every k-point and every band combination (m,n)
#   E_diff_nm = (np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,1,bnd))-np.reshape(arrays['E_k'][:,:bnd,ispin],(snktot,bnd,1)))[:,uind[0],uind[1]]

#   # fn_n-fn_m for every k-point and every band combination (m,n)
#   f_nm = (np.reshape(fn,(snktot,bnd,1))-np.reshape(fn,(snktot,1,bnd)))[:,uind[0],uind[1]]
#   fn = None

#   # <p_n|p_m> for every k-point and every band combination (m,n)
#   pksp2 = arrays['pksp'][:,ipol,uind[0],uind[1],ispin]*arrays['pksp'][:,jpol,uind[1],uind[0],ispin]
#   pksp2 = (abs(pksp2) if smearing is None else np.real(pksp2))

#    sq2_dk2 = (1.0/(np.sqrt(np.pi)*arrays['deltakp2'][:,uind[0],uind[1],ispin]) if smearing=='gauss' else None)

#   for i,e in enumerate(ene):
#       if smearing is None:
#           dfunc = np.exp(-((e-E_diff_nm)/delta)**2)/(delta*np.sqrt(np.pi))
#       else:
#           dfunc = 1./(E_diff_nm-e+1.j*arrays['deltakp2'][:,uind[0],uind[1],ispin])
#        elif smearing == 'gauss':
#            dfunc = np.exp(-((e-E_diff_nm)/arrays['deltakp2'][:,uind[0],uind[1],ispin])**2)*sq2_dk2
#        elif smearing == 'm-p':
#            dfunc = metpax(E_diff_nm, e, arrays['deltakp2'][:,uind[0],uind[1],ispin])
#       epsr[i] = np.sum((kq_wght/(E_diff_nm**2+1.j*delta**2))*dfunc*f_nm*pksp2/(e**2+delta**2))

#   f_nm = dfunc = uind = pksp2 = sq2_dk2 = E_diff_nm = None


def epsr_kramerskronig ( data_controller, ene, epsi ):
    from mpi4py import MPI
    from scipy.integrate import simps
    from load_balancing import load_balancing

    comm = MPI.COMM_WORLD

    arrays,attributes = data_controller.data_dicts()

    esize = ene.size
    de = ene[1] - ene[0]

    epsr = np.zeros(esize, dtype=float)

#### PARALLELIZATION
#### Use scatter
    ini_ie,end_ie = load_balancing(comm.Get_size(), comm.Get_rank(), esize)

    # Range checks for Simpson Integrals
    if end_ie == ini_ie:
        return
    if ini_ie < 3:
        ini_ie = 3
    if end_ie == esize:
        end_ie = esize-1

    f_ene = intmetpax(ene, attributes['shift'], 1.)
    for ie in range(ini_ie, end_ie):
        I1 = simps(ene[1:(ie-1)]*de*epsi[1:(ie-1)]*f_ene[1:(ie-1)]/(ene[1:(ie-1)]**2-ene[ie]**2))
        I2 = simps(ene[(ie+1):esize]*de*epsi[(ie+1):esize]*f_ene[(ie+1):esize]/(ene[(ie+1):esize]**2-ene[ie]**2))
        epsr[ie] = 2.*(I1+I2)/np.pi

    return epsr
