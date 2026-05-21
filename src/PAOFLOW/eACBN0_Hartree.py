"""MPI-parallel evaluator of the intersite Hubbard V numerator.

Companion module to :class:`PAOFLOW.eACBN0.eACBN0`.  Implements the
spin-summed Coulomb sum that appears in the numerator of Eq. (8) of

    S.-H. Lee and Y.-W. Son, *First-principles approach with a
    pseudo-hybrid density functional for extended Hubbard
    interactions*, Phys. Rev. Research **2**, 043410 (2020),

.. math::

    V^{IJ}(R) = \\tfrac{1}{2}\\,\\frac{\\text{num}}{\\text{den}}

with the numerator

.. math::

    \\text{num} = \\sum_{\\sigma\\sigma'}\\sum_{ikjl}
        \\Bigl[
            P^{II\\sigma}_{ik}\\, P^{JJ\\sigma'}_{jl}
            - \\delta_{\\sigma\\sigma'}\\,
              P^{IJ\\sigma}_{il}\\, P^{JI\\sigma}_{jk}
        \\Bigr]\\,(ik|jl).

The renormalized pair density matrices ``P^{II}``, ``P^{JJ}``,
``P^{IJ}(R)``, ``P^{JI}(-R)`` are built by
:meth:`PAOFLOW.eACBN0.eACBN0._pair_density_matrices` from the PAOFLOW
Hamiltonian/overlap dump (Eqs. (4)–(5) of the reference); the
two-electron integral ``(ik|jl) = ∫∫ φ_i^I(r₁) φ_k^I(r₁) (1/r₁₂)
φ_j^J(r₂) φ_l^J(r₂) dr₁ dr₂`` is evaluated analytically over
contracted Cartesian Gaussians by
:func:`PAOFLOW.defs.pyints.contr_coulomb`.

The denominator of Eq. (8) is built directly on the driver side in
:meth:`PAOFLOW.eACBN0.eACBN0.run_eacbn0_V` from the bare occupation
matrices ``n^{II}``, ``n^{JJ}``, ``n^{IJ}``, ``n^{JI}`` and combined
with the numerator returned by this kernel to obtain V in eV.

Geometry convention
-------------------
The basis functions ``gauss_I`` are centred at the home-cell position
of atom I, while ``gauss_J`` are centred at ``r_J + R*``, where ``R*``
is the *minimum-image* lattice translation selected by
:meth:`PAOFLOW.eACBN0.eACBN0.run_eacbn0_V`.  As a result, indices
``i, k`` run over the Gaussians on atom I and ``j, l`` over the
translated Gaussians on atom J; no further phase factors are required
inside the kernel.

Parallelization
---------------
The Cartesian product over ``(i, k, j, l) ∈ range(n_I)² × range(n_J)²``
is enumerated on rank 0, split into ``self.size`` roughly equal chunks
with :func:`numpy.array_split` and scattered with ``comm.scatter``.
Each rank accumulates its local contribution to ``tmp`` (complex);
results are reduced to rank 0 with ``MPI.SUM`` and pickled to
``<outputdir>/tmp_v.pkl``.

Cost scales as ``n_I² × n_J² × (primitive count)⁴`` inside
:func:`contr_coulomb`.  For two p-shells (``n_I = n_J = 3``) the
kernel finishes in a few seconds even at ``-np 1``; for a d–p pair
(e.g. Mn-3d to O-2p, ``n_I = 5``, ``n_J = 3``) with the full
multi-zeta Gaussian fits the cost runs into tens of millions of
Boys-function evaluations and benefits substantially from running
under ``mpirun -np N`` with ``N`` matching the number of physical
cores — hence the dedicated ``mpi_hartree`` knob in
:class:`PAOFLOW.ACBN0.ACBN0`, which is reused here.

Input format
------------
The driver writes a pickle file at ``datafile`` (broadcast to all
ranks in :meth:`__init__`) containing one entry per pair:

- ``'gauss_I'``, ``'gauss_J'`` — lists of
  :class:`PAOFLOW.defs.pyints.CGBF` contracted Gaussian basis
  functions covering the Hubbard-active shells on atoms I and J,
  with origins in Bohr (J already translated by ``R*``).
- ``'P_II_up'``, ``'P_II_dn'`` — ``(n_I, n_I)`` complex on-site
  renormalized DMs on atom I, per spin channel.
- ``'P_JJ_up'``, ``'P_JJ_dn'`` — ``(n_J, n_J)`` complex on-site
  renormalized DMs on atom J, per spin channel.
- ``'P_IJ_up'``, ``'P_IJ_dn'`` — ``(n_I, n_J)`` complex intersite
  renormalized DMs at displacement ``+R*``.
- ``'P_JI_up'``, ``'P_JI_dn'`` — ``(n_J, n_I)`` complex intersite
  renormalized DMs at displacement ``-R*``.

For spin-restricted runs (``nspin == 1``) the driver passes identical
up/down halves so that the spin pre-factors below reduce to the
spin-restricted form automatically.

Output
------
``<outputdir>/tmp_v.pkl`` (rank 0 only): a dictionary
``{'num': complex}`` holding the unnormalized numerator sum (no
1/2 prefactor — that factor is applied on the driver side together
with the denominator).

Spin structure
--------------
The bracket ``[direct − exchange]`` inside the four-index sum
factorises into two pieces:

- **Direct (Hartree)** — full ``σ × σ'`` double sum, equal to
  ``(P^{II}_up + P^{II}_dn)_{ik} × (P^{JJ}_up + P^{JJ}_dn)_{jl}``.
  Pre-computed once outside the inner loop as ``PII_sum`` /
  ``PJJ_sum``.
- **Exchange** — same-spin only (``δ_{σσ'}``), summing
  ``P^{IJ}_{σ}_{il} × P^{JI}_{σ}_{jk}`` over ``σ ∈ {up, dn}``.

Notes
-----
The module imports :mod:`mpi4py` at top level, so simply importing it
in a non-MPI Python process is harmless (it initialises a singleton
communicator with size 1).  In that mode the kernel runs serially and
takes the full ``n_I² × n_J²`` cost on a single core.
"""

import pickle
import itertools

from mpi4py import MPI


class eACBN0_Hartree:
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
        """Coulomb integral ``(ab|cd)`` (chemist's notation) between
        four contracted Gaussian basis functions."""
        from .defs.pyints import contr_coulomb

        return contr_coulomb(
            a.pexps, a.pcoefs, a.pnorms, a.origin, a.powers,
            b.pexps, b.pcoefs, b.pnorms, b.origin, b.powers,
            c.pexps, c.pcoefs, c.pnorms, c.origin, c.powers,
            d.pexps, d.pcoefs, d.pnorms, d.origin, d.powers,
        )

    def intersite_energy(self, outputdir):
        """Evaluate the numerator of Eq. (8) and pickle the result to
        ``<outputdir>/tmp_v.pkl``."""
        import numpy as np
        from os.path import join

        gauss_I = self.data['gauss_I']
        gauss_J = self.data['gauss_J']

        P_II_up = self.data['P_II_up']; P_II_dn = self.data['P_II_dn']
        P_JJ_up = self.data['P_JJ_up']; P_JJ_dn = self.data['P_JJ_dn']
        P_IJ_up = self.data['P_IJ_up']; P_IJ_dn = self.data['P_IJ_dn']
        P_JI_up = self.data['P_JI_up']; P_JI_dn = self.data['P_JI_dn']

        # Spin-summed prefactors.
        PII_sum = P_II_up + P_II_dn        # (n_I, n_I)
        PJJ_sum = P_JJ_up + P_JJ_dn        # (n_J, n_J)

        n_I = len(gauss_I)
        n_J = len(gauss_J)

        if self.rank == 0:
            ind_all = np.array(
                list(itertools.product(range(n_I), range(n_I),
                                       range(n_J), range(n_J)))
            )
            ind = np.array_split(ind_all, self.size, 0)
        else:
            ind = None
        ind = self.comm.scatter(ind, root=0)

        tmp = 0.0 + 0.0j
        for i, k, j, l in ind:
            integ = self.coulomb(gauss_I[i], gauss_I[k],
                                 gauss_J[j], gauss_J[l])

            # Direct (Hartree) term: full σ,σ' double sum.
            direct = PII_sum[i, k] * PJJ_sum[j, l]
            # Exchange term: same-spin only (δ_{σσ'}).
            exchange = (P_IJ_up[i, l] * P_JI_up[j, k]
                        + P_IJ_dn[i, l] * P_JI_dn[j, k])

            tmp += integ * (direct - exchange)

        tmp = self.comm.reduce(tmp, op=MPI.SUM, root=0)

        if self.rank == 0:
            with open(join(outputdir, 'tmp_v.pkl'), 'wb') as f:
                pickle.dump({'num': tmp}, f)
