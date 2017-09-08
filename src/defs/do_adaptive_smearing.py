

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from communication import *
import numpy as np
from numpy import linalg as LAN
from load_balancing import *

def do_adaptive_smearing(pksp,nawf,nspin,alat,a_vectors,nk1,nk2,nk3,smearing,npool):

    if smearing != None:
        #----------------------
        # adaptive smearing as in Yates et al. Phys. Rev. B 75, 195121 (2007).
        #----------------------

        if rank==0:
            deltakp = np.zeros((nk1*nk2*nk3,nawf,nspin),dtype=float)
            deltakp2 = np.zeros((nk1*nk2*nk3,nawf,nawf,nspin),dtype=float)
        else:
            deltakp=None
            deltakp2=None

        omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
        dk = (8.*np.pi**3/omega/(nk1*nk2*nk3))**(1./3.)

        for pool in xrange(npool):

            ini_ik, end_ik = load_balancing(npool,pool,nk1*nk2*nk3)

            if rank==0:
                pksaux=scatter_array(pksp[ini_ik:end_ik])

            else:
                pksaux=scatter_array(None) 

            deltakpaux = np.zeros((pksaux.shape[0],nawf,nspin),dtype=float)
            deltakp2aux = np.zeros((pksaux.shape[0],nawf,nawf,nspin),dtype=float)

            if smearing == 'gauss':
                afac = 0.7
            elif smearing == 'm-p':
                afac = 1.0

            for n in xrange(nawf):
                deltakpaux[:,n,:] = afac*LAN.norm(np.real(pksaux[:,:,n,n,:]),axis=1)*dk
                for m in xrange(nawf):
                    if smearing == 'gauss':
                        afac = 0.7
                    elif smearing == 'm-p':
                        afac = 1.0
                    deltakp2aux[:,n,m,:] = afac*LAN.norm(np.real(np.absolute(pksaux[:,:,n,n,:]-pksaux[:,:,m,m,:])),
                                                         axis=1)*dk
            if rank==0:
                gather_array(deltakp[ini_ik:end_ik],deltakpaux)
            else:
                gather_array(None,deltakpaux)
            comm.Barrier()

            if rank==0:
                gather_array(deltakp2[ini_ik:end_ik],deltakp2aux)
            else:
                gather_array(None,deltakp2aux)
            comm.Barrier()

            deltakpaux=None
            deltakp2aux=None        

        return deltakp,deltakp2
