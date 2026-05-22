"""MPI worker for the ACBN0 on-site Hartree / exchange integrals.

This module is invoked by :class:`PAOFLOW.ACBN0.ACBN0` as a standalone
MPI Python program (typically through the auto-generated
``compute_hartree.py`` launcher) to evaluate the four-centre Coulomb
sums that make up the ACBN0 numerator for a single Hubbard site.

Given the renormalized real-space density matrix ``D(R=0)`` per spin
channel (built by :meth:`PAOFLOW.ACBN0.ACBN0.DR` from the PAOFLOW dump)
and the contracted-Gaussian basis fitted to the pseudopotential atomic
wavefunctions, the kernel accumulates

.. math::

    U_{\\text{num}} = \\sum_{klmn} (mn|kl)\\,
        \\bigl[ D^{\\uparrow}_{mn} D^{\\uparrow}_{kl}
              + D^{\\downarrow}_{mn} D^{\\downarrow}_{kl}
              + D^{\\downarrow}_{mn} D^{\\uparrow}_{kl}
              + D^{\\uparrow}_{mn} D^{\\downarrow}_{kl} \\bigr]

    J_{\\text{num}} = \\sum_{\\substack{klmn \\\\ (m,n)\\neq(k,l)}}
        (mk|nl)\\,
        \\bigl[ D^{\\uparrow}_{mn} D^{\\uparrow}_{kl}
              + D^{\\downarrow}_{mn} D^{\\downarrow}_{kl} \\bigr]

where the four indices run over the indices in ``basis_2e`` (the
Hubbard-active subshell), the two-electron integrals
``(mn|kl) = ∫∫ φ_m(r₁) φ_n(r₁) (1/r₁₂) φ_k(r₂) φ_l(r₂) dr₁ dr₂`` are
evaluated analytically by :func:`PAOFLOW.defs.pyints.contr_coulomb`
over contracted Cartesian Gaussians, and the spin pre-factors follow
the (αβ + βα) decomposition used by the ACBN0 formula
(Agapito *et al.*, Phys. Rev. X **5**, 011006 (2015)).  The
``(m,n) = (k,l)`` term is excluded from the J sum because it
coincides with the direct Coulomb integral entering U and would
otherwise inflate J (hence depress ``U_eff = U − J``); see Eq. (11) of
the same reference.

Parallelization
---------------
The Cartesian product of four indices over ``basis_2e`` is built on
rank 0, split into ``self.size`` roughly equal chunks with
``numpy.array_split`` and scattered with ``comm.scatter``.  Each rank
sums its local contribution to ``tmp_U`` / ``tmp_J``; results are
reduced to rank 0 with ``MPI.SUM`` and pickled to
``<outputdir>/tmp_uj.pkl``.

Cost scales as ``len(basis_2e) ** 4`` times the (primitive count) ** 4
inside :func:`contr_coulomb`.  For light p-shells (e.g. O-2p,
``len(basis_2e) = 3``) the kernel finishes in milliseconds even at
``-np 1``; for heavy d-shells (e.g. Mn-3d, ``len(basis_2e) = 5`` with
up to 15 primitives per basis function) the cost runs into tens of
millions of Boys-function evaluations and benefits substantially from
running under ``mpirun -np N`` with ``N`` matching the number of
physical cores — hence the dedicated ``mpi_hartree`` knob in
:class:`PAOFLOW.ACBN0.ACBN0`.

Input format
------------
The driver writes a pickle file at ``datafile`` (broadcast to all
ranks in :meth:`__init__`) containing:

- ``'DR_up'``, ``'DR_dn'`` — ``(nbasis, nbasis)`` complex arrays, the
  renormalized real-space density matrices (R=0) per spin channel.
- ``'basis'`` — list of :class:`PAOFLOW.defs.pyints.CGBF` contracted
  Gaussian basis functions covering the entire site.
- ``'basis_2e'`` — list of integer indices into ``basis`` selecting
  the Hubbard-active subshell over which the four-index sum is
  taken (e.g. the 5 d-orbitals of a transition-metal site).

Output
------
``<outputdir>/tmp_uj.pkl`` (rank 0 only): a dictionary
``{'U': complex, 'J': complex}`` holding the unnormalized numerator
sums.  The driver divides these by the denominator
(:meth:`PAOFLOW.ACBN0.ACBN0.Nmm`) to obtain the final U and J in eV.

Notes
-----
The module imports :mod:`mpi4py` at top level, so simply importing it
in a non-MPI Python process is harmless (it initialises a singleton
communicator with size 1).  In that mode the kernel runs serially and
takes the full ``basis_2e ** 4`` cost on a single core.
"""

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
        import itertools
        from os.path import join

        import numpy as np

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

            a_b = DR_up[m, n] * DR_up[k, l] + DR_dn[m, n] * DR_dn[k, l]
            ab_ba = DR_dn[m, n] * DR_up[k, l] + DR_up[m, n] * DR_dn[k, l]

            tmp_U += int_U * (a_b + ab_ba)

            # Hund's J counts only parallel-spin exchange between
            # *distinct* orbitals (Agapito PRX 5, 011006 (2015), Eq. 11);
            # the self-exchange (m,n)==(k,l) is the direct Coulomb and
            # must not enter J, otherwise J is inflated and U_eff = U-J
            # is systematically too small.
            if not (m == k and n == l):
                int_J = self.coulomb(basis[m], basis[k], basis[n], basis[l])
                tmp_J += int_J * a_b

        tmp_U = self.comm.reduce(tmp_U, op=MPI.SUM, root=0)
        tmp_J = self.comm.reduce(tmp_J, op=MPI.SUM, root=0)

        if self.rank == 0:
            uj = {'U': tmp_U, 'J': tmp_J}
            with open(join(outputdir, 'tmp_uj.pkl'), 'wb') as f:
                pickle.dump(uj, f)
