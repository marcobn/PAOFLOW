import pickle
from mpi4py import MPI


class ACBN0_Hartree:
    def __init__(self, datafile):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank == 0:
            with open(datafile, 'rb') as f:
                data = pickle.load(f)
        else:
            data = None
        self.data = self.comm.bcast(data, root=0)

    def coulomb(self, a, b, c, d):
        from .defs.pyints import contr_coulomb

        ' Coulomb interaction between four contracted Gaussians '
        return contr_coulomb(
            a.pexps,
            a.pcoefs,
            a.pnorms,
            a.origin,
            a.powers,
            b.pexps,
            b.pcoefs,
            b.pnorms,
            b.origin,
            b.powers,
            c.pexps,
            c.pcoefs,
            c.pnorms,
            c.origin,
            c.powers,
            d.pexps,
            d.pcoefs,
            d.pnorms,
            d.origin,
            d.powers,
        )

    def hartree_energy(self, outputdir):
        import numpy as np
        import itertools
        from os.path import join

        DR_up = self.data['DR_up']
        DR_dn = self.data['DR_dn']
        basis = self.data['basis']
        basis_2e = self.data['basis_2e']

        tmp_U, tmp_J = 0.0, 0.0

        if self.rank == 0:
            ind_all = np.array(list(itertools.product(basis_2e, repeat=4)))
            ind = np.array_split(ind_all, self.size, 0)
        else:
            ind = None

        ind = self.comm.scatter(ind, root=0)

        for k, l, m, n in ind:
            int_U = self.coulomb(basis[m], basis[n], basis[k], basis[l])
            int_J = self.coulomb(basis[m], basis[k], basis[n], basis[l])

            a_b = DR_up[m, n] * DR_up[k, l] + DR_dn[m, n] * DR_dn[k, l]
            ab_ba = DR_dn[m, n] * DR_up[k, l] + DR_up[m, n] * DR_dn[k, l]

            tmp_U += int_U * (a_b + ab_ba)
            tmp_J += int_J * a_b

        tmp_U = self.comm.reduce(tmp_U, op=MPI.SUM, root=0)
        tmp_J = self.comm.reduce(tmp_J, op=MPI.SUM, root=0)

        if self.rank == 0:
            uj = {'U': tmp_U, 'J': tmp_J}
            with open(join(outputdir, 'tmp_uj.pkl'), 'wb') as f:
                pickle.dump(uj, f)
