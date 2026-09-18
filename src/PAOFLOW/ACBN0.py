"""ACBN0 self-consistent Hubbard U driver.

This module implements the ACBN0 pseudo-hybrid self-consistent
determination of on-site Hubbard U corrections from first principles,
following

    L. A. Agapito, S. Curtarolo and M. Buongiorno Nardelli,
    *Reformulation of DFT+U as a pseudo-hybrid Hubbard density
    functional for accelerated materials discovery*,
    Phys. Rev. X **5**, 011006 (2015).

The driver is exposed through the :class:`ACBN0` class.  An instance
orchestrates the full self-consistent cycle:

1. Parse a user-supplied Quantum ESPRESSO ``<prefix>.scf.in`` template
   (and the matching ``<prefix>.nscf.in`` / ``<prefix>.projwfc.in``).
2. Fit each pseudopotential's atomic wavefunctions to a contracted
   Gaussian basis (:mod:`PAOFLOW.projection.upf_gaussfit`) so the Coulomb
   integrals required by the ACBN0 formula can be evaluated
   analytically.
3. On every outer iteration, inject the current ``HUBBARD`` card into
   the SCF/NSCF templates and run ``pw.x`` (scf), ``pw.x`` (nscf),
   ``projwfc.x``, PAOFLOW and finally :class:`ACBN0_Hartree`
   (under MPI).
4. Recompute U for every Hubbard-active orbital from the ACBN0
   numerator (renormalized two-electron integrals) and denominator
   (occupation-matrix invariants).
5. Iterate until ``max |Δ U| < convergence_threshold``.

The class is also the base of :class:`eACBN0`, which
adds intersite Hubbard V to the same workflow.

Typical usage
-------------
::

    from PAOFLOW.ACBN0 import ACBN0

    a = ACBN0('MgO', workdir='./',
              mpi_qe='mpirun -np 8',
              mpi_python='mpirun -np 1',
              mpi_hartree='mpirun -np 8',
              qe_path='/path/to/qe/bin',
              python_path='/path/to/python/bin',
              outputdir='./tmp/')

    # Equivalent ways to declare Hubbard-active orbitals:
    a.set_hubbard_parameters(['Mg-3s', 'O-2p'])             # default U
    a.set_hubbard_parameters({'Mg-3s': 1.0, 'O-2p': 8.0})   # custom U
    a.set_hubbard_parameters({'O-2p': (8.0, 4.0)})          # +occupation

    a.optimize_hubbard_U(convergence_threshold=0.01)

    print(a.uVals)   # {'Mg-3s': ..., 'O-2p': ...}

Public API
----------
- :meth:`ACBN0.set_hubbard_parameters` — declare Hubbard-active
  orbitals.  Accepts a list of ``'species-orbital'`` labels (default
  seed U), a dict mapping labels to seed U (eV), or a dict with
  ``(U, occupation)`` tuples to also fix ``hubbard_occ`` in the
  ``&system`` namelist.
- :meth:`ACBN0.set_intersite_V_parameters` — seed entries in
  :attr:`vVals` so the ``HUBBARD`` card emits ``V`` lines; used by
  :class:`eACBN0` and for fixed-V runs.
- :meth:`ACBN0.optimize_hubbard_U` — outer self-consistent loop.
- :meth:`ACBN0.run_dft` — one shot of pw.x (scf) + pw.x (nscf) +
  projwfc.x with the current ``HUBBARD`` card injected.
- :meth:`ACBN0.run_paoflow` — invoke PAOFLOW on the QE save folder to
  produce the PAO Hamiltonian/overlap and k-point dumps consumed by
  the ACBN0 numerator/denominator.
- :meth:`ACBN0.run_acbn0` — single ACBN0 evaluation pass on the
  current PAOFLOW dump.
- :meth:`ACBN0.hubbard_card` — render the current
  ``self.uVals`` / ``self.vVals`` into a QE ``HUBBARD`` card.
- :meth:`ACBN0.read_cell_atoms`, :meth:`ACBN0.read_ham_data` —
  helpers for parsing QE ``scf.out`` and the PAOFLOW dumps.

Internal helpers (prefixed with an underscore convention via their
naming) handle Gaussian basis assembly per atom
(:meth:`ACBN0.getbasis`), construction of the k-resolved density
matrix (:meth:`ACBN0.Dk`), the back-Fourier transform to real space
(:meth:`ACBN0.DR`) and the on-site occupation invariants used in the
ACBN0 denominator (:meth:`ACBN0.Nmm`).

External tools
--------------
The driver shells out to several commands.  The launchers are
configured at construction time:

- ``pw.x`` and ``projwfc.x`` from ``qe_path``, optionally prefixed by
  ``mpi_qe`` (e.g. ``'mpirun -np 8'``) with extra options in
  ``qe_options`` (e.g. ``'-npool 4'``).
- PAOFLOW is launched as a separate Python process under
  ``mpi_python`` (typically ``'mpirun -np 1'`` because PAOFLOW's MPI
  scaling is workflow-dependent and the call here is short).
- The Hartree integrals (:class:`ACBN0_Hartree`) are launched
  as a separate Python process under ``mpi_hartree``, which defaults
  to ``mpi_python`` but can be overridden — typically with a *larger*
  ``-np`` because the pure-Python four-centre Coulomb integrals
  dominate the wall time for heavy d/f shells.

Conventions
-----------
- Energies are in eV; the input ``celldm(1)`` / lattice quantities are
  read in their native QE units (Bohr / Å) and converted internally
  where required.
- Atom indices follow the QE convention: 1-based, ordered as they
  appear in the ``ATOMIC_POSITIONS`` card.
- Orbital labels are written as ``'<element>-<n><l>'``, e.g.
  ``'Mn-3d'``, ``'O-2p'``.  In magnetic AFM cells with distinct
  sublattices, the element symbol in the input may be tagged
  (``'MnA'``, ``'MnB'``) and the same tag must be used in the U keys.
- The class respects an existing ``HUBBARD`` card in the template:
  any U/V entries already present are loaded into ``self.uVals`` and
  ``self.vVals`` at construction time.

Attributes set at construction
------------------------------
- :attr:`uVals` (``dict[str, float]``) — current U values, keyed by
  ``'species-orbital'``.
- :attr:`vVals` (``dict[tuple, float]``) — current intersite V values
  (populated by :class:`eACBN0`).
- :attr:`uspecies` (``list[str]``) — atomic species discovered in the
  ``ATOMIC_SPECIES`` card.
- :attr:`basis` (``dict[str, list]``) — per-species fitted Gaussian
  basis from :func:`PAOFLOW.projection.upf_gaussfit.gaussian_fit`.
- :attr:`blocks` / :attr:`cards` — parsed namelists / cards of the
  SCF input template, as returned by
  :func:`PAOFLOW.inputs.file_io.struct_from_inputfile_QE`.
- :attr:`nspin` — read from the ``&system`` block (defaults to 1).
- :attr:`hubbard_tag` — header line of the ``HUBBARD`` card
  (e.g. ``'HUBBARD (atomic)'``).
- :attr:`hubbard_occ` — fixed ``hubbard_occ(i,j)`` entries to be
  written back into the ``&system`` namelist.

Notes
-----
- The first thing :meth:`__init__` does after parsing the template is
  to write ``compute_hartree.py`` into the current directory; this
  small launcher is what gets executed under ``mpi_hartree`` for the
  Coulomb integrals.
- The PAOFLOW step launched by :meth:`run_paoflow` calls
  ``pao_hamiltonian(write_binary=True, expand_wedge=False)``: ``Hks``
  is kept on whichever k-grid QE produced (IBZ when symmetries are on,
  full BZ when ``nosym = noinv = .true.``) and is written to disk
  without any FFT to real space.  The reshape to
  ``(nawf, nawf, nk1, nk2, nk3, nspin)`` in
  :func:`PAOFLOW.hamiltonian.do_build_pao_hamiltonian.do_build_pao_hamiltonian`
  is therefore skipped for ACBN0; both IBZ and full-BZ k-grids are
  accepted.
- :func:`PAOFLOW.hamiltonian.do_build_pao_hamiltonian.build_Hks` now applies a
  *per-k* projectability filter (``pthr_local``, defaulting to
  ``0.5 * pthr``) in addition to the global ``pthr``.  Bands whose
  local PAO projection at a given k vanishes (typically one of a
  degenerate set at high-symmetry points such as Γ, where QE's gauge
  choice can leave it with near-zero atomic-orbital content) are
  dropped from the construction at that k-point only.  Without this
  guard the previous code renormalised the vanishing projection vector
  to unit norm, corrupting ``H(k)`` at high-symmetry points and, in
  turn, the occupations entering the ACBN0 numerator/denominator.
"""

import itertools
import pickle
import re
import subprocess
from os.path import expanduser, isfile, join

import numpy as np
from mpi4py import MPI

from . import acbn0_native as _acbn0_native

BOHR_RADIUS_ANGS = 0.529177210903
ANGS_TO_BOHR = 1.0 / BOHR_RADIUS_ANGS
HARTREE_TO_EV = 27.211396132


class _HartreeKernel:
    """Common MPI scaffolding shared by :class:`ACBN0_Hartree` and
    :class:`eACBN0_Hartree`.

    Provides:

    - ``__init__(datafile)`` — initialises ``self.comm`` / ``self.rank`` /
      ``self.size``, unpickles ``datafile`` on rank 0, and broadcasts the
      dictionary to all ranks (stored as ``self.data``).
    - ``coulomb(a, b, c, d)`` — chemist's-notation four-centre Coulomb
      integral ``(ab|cd)`` between contracted Cartesian Gaussians,
      delegating to :func:`PAOFLOW.utils.pyints.contr_coulomb`.

    Subclasses implement the kernel-specific accumulation
    (:meth:`ACBN0_Hartree.hartree_energy`,
    :meth:`eACBN0_Hartree.intersite_energy`).
    """

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
        from .utils.pyints import contr_coulomb

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

    def _scatter_indices(self, *iterables):
        """Build the Cartesian product of ``iterables`` on rank 0, split
        into ``self.size`` chunks with :func:`numpy.array_split`, and
        scatter the per-rank chunk to every rank.

        Returns the local ``(n_local, len(iterables))`` integer array of
        index tuples on every rank.
        """
        if self.rank == 0:
            ind_all = np.array(list(itertools.product(*iterables)))
            ind = np.array_split(ind_all, self.size, 0)
        else:
            ind = None
        return self.comm.scatter(ind, root=0)

    def _reduce_and_dump(self, payload, outputdir, filename):
        """Reduce ``payload`` (dict of complex scalars) with ``MPI.SUM``
        and pickle the reduced dict to ``<outputdir>/<filename>`` on
        rank 0 only.
        """
        reduced = {k: self.comm.reduce(v, op=MPI.SUM, root=0) for k, v in payload.items()}
        if self.rank == 0:
            with open(join(outputdir, filename), 'wb') as f:
                pickle.dump(reduced, f)


class ACBN0_Hartree(_HartreeKernel):
    """MPI worker for the ACBN0 on-site Hartree / exchange integrals.

    Invoked by :class:`ACBN0` as a standalone MPI Python program
    (typically through the auto-generated ``compute_hartree.py`` launcher)
    to evaluate the four-centre Coulomb sums that make up the ACBN0
    numerator for a single Hubbard site.

    Given the renormalized real-space density matrix ``D(R=0)`` per spin
    channel (built by :meth:`ACBN0.DR` from the PAOFLOW dump)
    and the contracted-Gaussian basis fitted to the pseudopotential atomic
    wavefunctions, the kernel accumulates

    .. math::

        U_{\\text{num}} = \\sum_{abcd} (ab|cd)\\,
            D^{\\text{tot}}_{ab}\\, D^{\\text{tot}}_{cd},
            \\qquad D^{\\text{tot}} = D^{\\uparrow} + D^{\\downarrow}

        J_{\\text{num}} = \\sum_{abcd} (ac|bd)\\,
            \\bigl[ D^{\\uparrow}_{ab}\\, D^{\\uparrow}_{cd}
                  + D^{\\downarrow}_{ab}\\, D^{\\downarrow}_{cd} \\bigr]

    where the four indices run over ``basis_2e`` (the Hubbard-active
    subshell), the two-electron integrals
    ``(ab|cd) = ∫∫ φ_a(r₁) φ_b(r₁) (1/r₁₂) φ_c(r₂) φ_d(r₂) dr₁ dr₂`` are
    evaluated analytically by :func:`PAOFLOW.utils.pyints.contr_coulomb`
    over contracted Cartesian Gaussians, and the spin pre-factors follow
    the ACBN0 decomposition
    (Agapito *et al.*, Phys. Rev. X **5**, 011006 (2015)).  The U sum
    factorises through ``D^{tot} = D↑ + D↓`` (Hartree-like, full σσ'
    sum); the J sum carries only the like-spin pieces under the
    ``(ac|bd)`` index permutation.

    No self-exchange exclusion is applied to the J sum: empirically the
    no-exclusion form reproduces the published ACBN0 / Lee – Son
    benchmark U values (e.g. ZnO Zn-3d ≈ 15.47 eV, O-2p ≈ 7.33 eV with
    ortho-atomic projection), whereas the strict Eq. (11) form with the
    ``(m,n) = (k,l)`` term removed does not.

    Parallelization
    ---------------
    Only the unique 4-tuples ``(a, b, c, d)`` under the 8-fold permutation
    symmetry ``(ab|cd) = (ba|cd) = (ab|dc) = (cd|ab)`` of real Cartesian
    Gaussians are enumerated (canonical form: ``a ≤ b``, ``c ≤ d``,
    ``(a, b) ≤ (c, d)``).  These are split into ``self.size`` chunks with
    :func:`numpy.array_split`, scattered with ``comm.scatter``, evaluated
    locally and ``allgather``-ed so every rank holds the full ``eri_dict``.
    Each rank unfolds the dict into the full ``(n_2e)^4`` ``ERI`` tensor
    and contracts it with the density-matrix sub-blocks via
    :func:`numpy.einsum`.  The result is replicated on every rank, so
    **no ``MPI.SUM`` reduce is performed** — the pickle
    ``<outputdir>/tmp_uj.pkl`` is written from rank 0 only.

    Cost scales as ``len(basis_2e) ** 4`` times the (primitive count) ** 4
    inside :func:`contr_coulomb`.  For light p-shells (e.g. O-2p,
    ``len(basis_2e) = 3``) the kernel finishes in milliseconds even at
    ``-np 1``; for heavy d-shells (e.g. Mn-3d, ``len(basis_2e) = 5`` with
    up to 15 primitives per basis function) the cost runs into tens of
    millions of Boys-function evaluations and benefits substantially from
    running under ``mpirun -np N`` with ``N`` matching the number of
    physical cores — hence the dedicated ``mpi_hartree`` knob in
    :class:`ACBN0`.

    Input format
    ------------
    The driver writes a pickle file at ``datafile`` (broadcast to all
    ranks in :meth:`__init__`) containing:

    - ``'DR_up'``, ``'DR_dn'`` — ``(nbasis, nbasis)`` complex arrays, the
      renormalized real-space density matrices (R=0) per spin channel.
    - ``'basis'`` — list of :class:`PAOFLOW.utils.pyints.CGBF` contracted
      Gaussian basis functions covering the entire site.
    - ``'basis_2e'`` — list of integer indices into ``basis`` selecting
      the Hubbard-active subshell over which the four-index sum is
      taken (e.g. the 5 d-orbitals of a transition-metal site).

    Output
    ------
    ``<outputdir>/tmp_uj.pkl`` (rank 0 only): a dictionary
    ``{'U': float, 'J': float}`` holding the unnormalized numerator
    sums as real Python floats.  The driver divides these by the
    denominator (:meth:`ACBN0.Nmm`) to obtain the final U and J in eV.

    The real-valued type is load-bearing: QE's ``card_hubbard`` parser
    rejects complex-formatted Hubbard values, so the ERI tensor is
    allocated ``dtype=float``, the density-matrix sub-blocks are
    ``.real``-coerced, and the einsum results are wrapped in ``float(…)``
    before pickling.

    Notes
    -----
    The class imports :mod:`mpi4py` at module level, so simply importing it
    in a non-MPI Python process is harmless (it initialises a singleton
    communicator with size 1).  In that mode the kernel runs serially and
    takes the full ``basis_2e ** 4`` cost on a single core.
    """

    def hartree_energy(self, outputdir):
        DR_up = self.data['DR_up']
        DR_dn = self.data['DR_dn']
        basis = self.data['basis']
        basis_2e = self.data['basis_2e']

        # ------------------------------------------------------------------
        # P6: exploit 8-fold permutation symmetry of (ab|cd) for real
        # Gaussians, evaluate each unique ERI once, then carry out the U
        # and J sums as pure ``np.einsum`` contractions.
        #
        # ``basis_2e`` is an array of indices into ``basis``; we work in
        # local indices ``a,b,c,d ∈ [0, n_2e)`` for the ERI tensor and
        # density-matrix sub-blocks, mapping back to ``basis`` only when
        # calling :meth:`coulomb`.
        # ------------------------------------------------------------------
        n_2e = int(np.asarray(basis_2e).size)
        idx = np.asarray(basis_2e)

        # Enumerate unique 4-tuples under (ab|cd) = (ba|cd) = (ab|dc)
        # = (cd|ab) and combinations thereof.  Canonical form:
        #   a <= b,  c <= d,  (a, b) <= (c, d)  (lexicographic).
        unique_keys = []
        for a in range(n_2e):
            for b in range(a, n_2e):
                for c in range(n_2e):
                    for d in range(c, n_2e):
                        if (a, b) <= (c, d):
                            unique_keys.append((a, b, c, d))

        # MPI-scatter the unique tuples across ranks; each rank evaluates
        # only its local chunk.  Use ``allgather`` so every rank ends up
        # with the full ERI dict (and can do the cheap contraction
        # locally, no extra communication for the final reduce).
        if self.rank == 0:
            chunks = np.array_split(np.asarray(unique_keys, dtype=int), self.size, 0)
        else:
            chunks = None
        my_keys = self.comm.scatter(chunks, root=0)

        # ``contr_coulomb`` of real Cartesian Gaussians returns a real
        # scalar; coerce to ``float`` so the gathered dict and downstream
        # ERI tensor stay real.
        if _acbn0_native.available():
            # Batched Rust path: evaluate the whole rank-local chunk in one
            # FFI call over the Hubbard-active sub-basis (local indices).
            active = [basis[i] for i in idx]
            vals = _acbn0_native.eri_batch(active, my_keys)
            local_vals = {
                (int(a), int(b), int(c), int(d)): float(v) for (a, b, c, d), v in zip(my_keys, vals)
            }
        else:
            local_vals = {
                (int(a), int(b), int(c), int(d)): float(
                    self.coulomb(basis[idx[a]], basis[idx[b]], basis[idx[c]], basis[idx[d]])
                )
                for a, b, c, d in my_keys
            }

        all_local = self.comm.allgather(local_vals)
        eri_dict = {}
        for d_ in all_local:
            eri_dict.update(d_)

        # Unfold the unique values into the full (n_2e)^4 tensor via the
        # 8-fold symmetry.  Cost is n_2e^4 dict lookups — negligible next
        # to a single contr_coulomb call.  ``contr_coulomb`` of real
        # Cartesian Gaussians returns a real scalar, so the ERI tensor
        # is purely real; keep it as ``float`` so the contractions below
        # do not spuriously promote ``tmp_U`` / ``tmp_J`` to complex (the
        # downstream HUBBARD-card writer expects real Python floats).
        ERI = np.empty((n_2e, n_2e, n_2e, n_2e), dtype=float)
        for a in range(n_2e):
            for b in range(n_2e):
                a1, b1 = (a, b) if a <= b else (b, a)
                for c in range(n_2e):
                    for d in range(n_2e):
                        c1, d1 = (c, d) if c <= d else (d, c)
                        if (a1, b1) <= (c1, d1):
                            ERI[a, b, c, d] = eri_dict[(a1, b1, c1, d1)]
                        else:
                            ERI[a, b, c, d] = eri_dict[(c1, d1, a1, b1)]

        # Sub-block the density matrices to the Hubbard-active shell and
        # exploit the factorisation of the spin sum for U:
        #   U_num = Σ_{abcd} (ab|cd) D^tot_{ab} D^tot_{cd},   D^tot = D↑ + D↓
        # For J we need (ac|bd) = ERI.transpose(0, 2, 1, 3); only the
        # like-spin terms contribute:
        #   J_num = Σ_{abcd} (ac|bd) [D↑_{ab} D↑_{cd} + D↓_{ab} D↓_{cd}].
        # ``DR()`` already takes ``.real`` of the k-summed density, but be
        # explicit here so the sub-block is unambiguously real-valued and
        # the contractions stay in float-precision arithmetic.
        D_up = np.asarray(DR_up[np.ix_(idx, idx)]).real
        D_dn = np.asarray(DR_dn[np.ix_(idx, idx)]).real
        D_tot = D_up + D_dn

        ERI_J = ERI.transpose(0, 2, 1, 3)  # (ac|bd) layout
        tmp_U = float(np.einsum('abcd,ab,cd->', ERI, D_tot, D_tot, optimize=True))
        tmp_J = float(
            np.einsum('abcd,ab,cd->', ERI_J, D_up, D_up, optimize=True)
            + np.einsum('abcd,ab,cd->', ERI_J, D_dn, D_dn, optimize=True)
        )

        # Contraction already replicated on every rank — write from rank 0
        # without an MPI.SUM reduce (would over-count by ``self.size``).
        # Store as real Python floats: the downstream HUBBARD-card writer
        # interpolates them into ``scf.in`` with ``'{}'.format(v)`` and a
        # complex value of the form ``(10.55+0j)`` makes QE's
        # ``card_hubbard`` parser bail out.
        if self.rank == 0:
            with open(join(outputdir, 'tmp_uj.pkl'), 'wb') as f:
                pickle.dump({'U': tmp_U, 'J': tmp_J}, f)


class eACBN0_Hartree(_HartreeKernel):
    """MPI-parallel evaluator of the intersite Hubbard V numerator.

    Companion class to :class:`eACBN0`.  Implements the spin-summed
    Coulomb sum that appears in the numerator of Eq. (8) of

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
    :meth:`eACBN0._pair_density_matrices` from the PAOFLOW
    Hamiltonian/overlap dump (Eqs. (4)–(5) of the reference); the
    two-electron integral ``(ik|jl) = ∫∫ φ_i^I(r₁) φ_k^I(r₁) (1/r₁₂)
    φ_j^J(r₂) φ_l^J(r₂) dr₁ dr₂`` is evaluated analytically over
    contracted Cartesian Gaussians by
    :func:`PAOFLOW.utils.pyints.contr_coulomb`.

    The denominator of Eq. (8) is built directly on the driver side in
    :meth:`eACBN0.run_eacbn0_V` from the bare occupation
    matrices ``n^{II}``, ``n^{JJ}``, ``n^{IJ}``, ``n^{JI}`` and combined
    with the numerator returned by this kernel to obtain V in eV.

    Geometry convention
    -------------------
    The basis functions ``gauss_I`` are centred at the home-cell position
    of atom I, while ``gauss_J`` are centred at ``r_J + R*``, where ``R*``
    is the *minimum-image* lattice translation selected by
    :meth:`eACBN0.run_eacbn0_V`.  As a result, indices
    ``i, k`` run over the Gaussians on atom I and ``j, l`` over the
    translated Gaussians on atom J; no further phase factors are required
    inside the kernel.

    Parallelization
    ---------------
    Only the unique 4-tuples ``(i, k, j, l)`` under the 4-fold permutation
    symmetry ``(ik|jl) = (ki|jl) = (ik|lj)`` are enumerated (canonical
    form: ``i ≤ k``, ``j ≤ l``).  Because atoms I and J are distinct,
    the electron-swap symmetry ``(ik|jl) = (jl|ik)`` does *not* apply
    — only 4-fold (not 8-fold) reduction is available.  The unique keys
    are split into ``self.size`` chunks with :func:`numpy.array_split`,
    scattered with ``comm.scatter``, evaluated locally and
    ``allgather``-ed so every rank holds the full ``eri_dict``.  Each
    rank unfolds it into the ``(n_I, n_I, n_J, n_J)`` ``ERI`` tensor and
    contracts via :func:`numpy.einsum` (direct via ``PII_sum``/``PJJ_sum``;
    same-spin exchange via ``P_IJ``/``P_JI``).  The result is replicated
    on every rank, so **no ``MPI.SUM`` reduce is performed** — the pickle
    ``<outputdir>/tmp_v.pkl`` is written from rank 0 only.

    Cost scales as ``n_I² × n_J² × (primitive count)⁴`` inside
    :func:`contr_coulomb`.  For two p-shells (``n_I = n_J = 3``) the
    kernel finishes in a few seconds even at ``-np 1``; for a d–p pair
    (e.g. Mn-3d to O-2p, ``n_I = 5``, ``n_J = 3``) with the full
    multi-zeta Gaussian fits the cost runs into tens of millions of
    Boys-function evaluations and benefits substantially from running
    under ``mpirun -np N`` with ``N`` matching the number of physical
    cores — hence the dedicated ``mpi_hartree`` knob in :class:`ACBN0`,
    which is reused here.

    Input format
    ------------
    The driver writes a pickle file at ``datafile`` (broadcast to all
    ranks in :meth:`__init__`) containing one entry per pair:

    - ``'gauss_I'``, ``'gauss_J'`` — lists of
      :class:`PAOFLOW.utils.pyints.CGBF` contracted Gaussian basis
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
    The class imports :mod:`mpi4py` at module level, so simply importing it
    in a non-MPI Python process is harmless (it initialises a singleton
    communicator with size 1).  In that mode the kernel runs serially and
    takes the full ``n_I² × n_J²`` cost on a single core.
    """

    def intersite_energy(self, outputdir):
        """Evaluate the numerator of Eq. (8) and pickle the result to
        ``<outputdir>/tmp_v.pkl``."""
        gauss_I = self.data['gauss_I']
        gauss_J = self.data['gauss_J']

        P_II_up = self.data['P_II_up']
        P_II_dn = self.data['P_II_dn']
        P_JJ_up = self.data['P_JJ_up']
        P_JJ_dn = self.data['P_JJ_dn']
        P_IJ_up = self.data['P_IJ_up']
        P_IJ_dn = self.data['P_IJ_dn']
        P_JI_up = self.data['P_JI_up']
        P_JI_dn = self.data['P_JI_dn']

        # Spin-summed prefactors.
        PII_sum = P_II_up + P_II_dn  # (n_I, n_I)
        PJJ_sum = P_JJ_up + P_JJ_dn  # (n_J, n_J)

        n_I = len(gauss_I)
        n_J = len(gauss_J)

        # ----------------------------------------------------------------- #
        # P5: exploit the 4-fold permutation symmetry of (ik|jl) for real   #
        # Cartesian Gaussians.  Because atoms I and J are distinct (and    #
        # the gauss_I / gauss_J basis lists are independent) the           #
        # electron-swap (ik|jl) = (jl|ik) does NOT apply, but               #
        # (ik|jl) = (ki|jl) = (ik|lj) does — giving 4× fewer unique         #
        # integrals.  Canonical form: i <= k, j <= l.                       #
        # ----------------------------------------------------------------- #
        unique_keys = [
            (i, k, j, l)
            for i in range(n_I)
            for k in range(i, n_I)
            for j in range(n_J)
            for l in range(j, n_J)
        ]

        if self.rank == 0:
            chunks = np.array_split(np.asarray(unique_keys, dtype=int), self.size, 0)
        else:
            chunks = None
        my_keys = self.comm.scatter(chunks, root=0)

        # ``contr_coulomb`` of real Cartesian Gaussians returns a real
        # scalar; coerce to ``float`` so the gathered dict and ERI
        # tensor stay real (densities are complex, so contractions will
        # naturally promote to complex without spurious imaginary noise
        # from the integral side).
        if _acbn0_native.available():
            # Batched Rust path: i,k index gauss_I and j,l index gauss_J.
            vals = _acbn0_native.eri_batch_2c(gauss_I, gauss_J, my_keys)
            local_vals = {
                (int(i), int(k), int(j), int(l)): float(v) for (i, k, j, l), v in zip(my_keys, vals)
            }
        else:
            local_vals = {
                (int(i), int(k), int(j), int(l)): float(
                    self.coulomb(gauss_I[i], gauss_I[k], gauss_J[j], gauss_J[l])
                )
                for i, k, j, l in my_keys
            }

        all_local = self.comm.allgather(local_vals)
        eri_dict = {}
        for d_ in all_local:
            eri_dict.update(d_)

        # Unfold to full (n_I, n_I, n_J, n_J) ERI tensor via the 4-fold
        # symmetry.  Cost is n_I^2 * n_J^2 dict lookups — negligible.
        ERI = np.empty((n_I, n_I, n_J, n_J), dtype=float)
        for i in range(n_I):
            for k in range(n_I):
                i1, k1 = (i, k) if i <= k else (k, i)
                for j in range(n_J):
                    for l in range(n_J):
                        j1, l1 = (j, l) if j <= l else (l, j)
                        ERI[i, k, j, l] = eri_dict[(i1, k1, j1, l1)]

        # Direct (Hartree) term: full σ,σ' double sum.
        #   Σ (ik|jl) PII_sum[i,k] PJJ_sum[j,l]
        direct = np.einsum('ikjl,ik,jl->', ERI, PII_sum, PJJ_sum, optimize=True)
        # Exchange term: same-spin only (δ_{σσ'}).
        #   Σ (ik|jl) [P_IJ_up[i,l] P_JI_up[j,k] + (down term)]
        exchange = np.einsum('ikjl,il,jk->', ERI, P_IJ_up, P_JI_up, optimize=True) + np.einsum(
            'ikjl,il,jk->', ERI, P_IJ_dn, P_JI_dn, optimize=True
        )

        tmp = complex(direct - exchange)

        # Contraction already replicated on every rank — write from rank 0
        # without an MPI.SUM reduce (which would over-count by ``self.size``).
        if self.rank == 0:
            with open(join(outputdir, 'tmp_v.pkl'), 'wb') as f:
                pickle.dump({'num': tmp}, f)


class ACBN0:
    def __init__(
        self,
        prefix,
        pthr=0.95,
        workdir='./',
        mpi_qe='',
        nproc=1,
        qe_path='',
        qe_options='',
        mpi_python='',
        python_path='',
        outputdir='./output/',
        projection='ortho-atomic',
        mpi_hartree=None,
        use_local_basis=False,
        basispath=None,
        configuration='standard',
        gaussian_threshold=0.01,
    ):
        """Initialize the ACBN0 self-consistent U driver.

        Parameters
        ----------
        use_local_basis : bool, optional
            When ``True`` the Hubbard occupations are obtained from
            PAOFLOW's internal projection onto the local atomic basis
            (:meth:`PAOFLOW.PAOFLOW.PAOFLOW.projections`) instead of from
            ``projwfc.x``.  ``projwfc.x`` is then never run and no
            ``<prefix>.projwfc.in`` template is required.

            With this option PAOFLOW automatically computes and stores the
            true non-orthogonal atomic-orbital overlap
            :math:`S_{\\mu\\nu}(\\mathbf{k})` so that the non-orthogonal
            correction :math:`H \\to S^{1/2} H S^{1/2}` is applied
            correctly and the genuine :math:`S(k)` is written to
            ``kovp.npy``.  Both the on-site ACBN0 U and the intersite
            eACBN0 V therefore work correctly with the internal basis.
            The recommended ``configuration`` for ACBN0/eACBN0 is
            ``'minimal'`` (UPF pseudo-atomic wavefunctions), which matches
            the ``projwfc.x`` projectors most closely and gives the best
            agreement with the reference values.  ``'standard'`` and
            ``'extended'`` are also supported but may yield slightly
            different U/V due to the larger Löwdin-orthogonalized basis.
        basispath : str, optional
            Directory containing the per-element ``<elem>/*.dat`` all-electron
            radial basis files.  Required when ``use_local_basis`` is
            ``True`` and ``configuration`` is ``'standard'`` or
            ``'extended'``.
        configuration : {'minimal', 'standard', 'extended'} or dict, optional
            Projection-basis preset forwarded to
            :meth:`PAOFLOW.PAOFLOW.PAOFLOW.projections` when
            ``use_local_basis`` is ``True``.  ``'minimal'`` uses the UPF
            pseudo-atomic wavefunctions (the closest drop-in replacement
            for ``projwfc.x``); ``'standard'`` / ``'extended'`` augment the
            valence shells with polarization channels built from
            ``basispath``.  A custom ``{element: [shell_labels]}`` dict
            (e.g. ``{'Ga': ['3S', '3P', '4S', '4P', '3D', ...]}``) selects
            an explicit all-electron basis from ``basispath``; this is the
            way to include semicore shells (such as ``3S`` / ``3P``) that
            the preset rules would not add.  The Hubbard manifold is always
            selected by its shell *label* (e.g. ``'3P'``) so that
            polarization shells sharing the same angular momentum are
            excluded.
        projection : {'ortho-atomic', 'atomic'}, optional
            Hubbard projector type written to the QE ``HUBBARD`` card.

            - ``'ortho-atomic'`` (default): Löwdin-orthogonalized atomic
              orbitals.  Matches the QE+U literature standard
              (``hubbard_projectors = 'ortho-atomic'``) and reproduces
              the Lee-Son eACBN0 values (PRR 2, 043410 (2020)).
            - ``'atomic'``: bare atomic orbitals.  Retained for
              reproducing pre-2018 ACBN0 results; typically yields U
              values 20-40% lower than ortho-atomic on the same system.

            The choice must be self-consistent: the same projector type
            is used to build the occupations entering the ACBN0 numerator
            *and* the +U Hamiltonian in QE.
        gaussian_threshold : float, optional
            Residual-norm tolerance for the contracted-Gaussian fit of the
            pseudo-atomic wavefunctions
            (:func:`PAOFLOW.projection.upf_gaussfit.gaussian_fit`).  The
            fit adds Gaussians (up to 5) until the per-shell residual falls
            below this value.  The default ``0.01`` is appropriate for most
            norm-conserving pseudos; harder-to-fit shells (e.g. some ONCV
            sets) may need a looser value such as ``0.05``.
        """
        from os import chdir

        from .inputs.file_io import struct_from_inputfile_QE
        from .projection.upf_gaussfit import gaussian_fit
        from .utils.header import header

        header(style='small')
        print('\nPerforming ACBN0 self-consistent determination of Hubbard U corrections.\n')

        datafilepath = join(outputdir, 'data.pkl')
        with open('compute_hartree.py', 'w') as f:
            f.write('from PAOFLOW.ACBN0 import ACBN0_Hartree\n')
            f.write(f"H = ACBN0_Hartree('{datafilepath}')\n")
            f.write(f"H.hartree_energy('{outputdir}')")

        self.prefix = prefix
        self.pthr = pthr
        self.workdir = workdir
        self.mpi_qe = mpi_qe
        self.nproc = nproc
        self.qpath = qe_path
        self.mpi_python = mpi_python
        self.mpi_hartree = mpi_hartree if mpi_hartree is not None else mpi_python
        self.ppath = python_path
        self.qoption = qe_options
        self.projection = projection
        self.outputdir = outputdir

        self.use_local_basis = bool(use_local_basis)
        self.basispath = basispath
        self.configuration = configuration
        self.gaussian_threshold = float(gaussian_threshold)
        # Per-iteration cache of the PAO basis metadata dumped by the
        # local-basis PAOFLOW driver (populated by ``_load_pao_basis_meta``).
        self._pao_meta = None
        # Whether a converged charge density from a previous ``run_dft``
        # call is available on disk.  When True, the next scf reuses it as
        # its starting potential (``startingpot='file'``) instead of the
        # superposition of atomic densities, so each self-consistent
        # iteration restarts from the previous step's density.
        self._scf_density_available = False

        if self.use_local_basis:
            cfg = self.configuration
            if isinstance(cfg, dict):
                # An explicit {element: [shells]} mapping builds the AE
                # basis and therefore always needs the radial basis files.
                needs_basis = True
                cfg_name = 'custom dict'
            else:
                cfg = (cfg or 'standard').lower()
                needs_basis = cfg in ('standard', 'extended')
                cfg_name = cfg
            if needs_basis and not self.basispath:
                msg = (
                    'use_local_basis with configuration=%r requires '
                    "'basispath' pointing to the per-element radial basis "
                    'files.' % cfg_name
                )
                raise ValueError(msg)

        self.uVals = {}
        self.vVals = {}
        self.occ_states = {}
        self.occ_values = {}
        self.hubbard_occ = {}
        self.hubbard_tag = 'HUBBARD (' + self.projection + ')'

        chdir(self.workdir)

        # Get structure information
        self.blocks, self.cards = struct_from_inputfile_QE(f'{self.prefix}.scf.in')
        if 'nspin' in self.blocks['system']:
            self.nspin = int(self.blocks['system']['nspin'])
        else:
            self.nspin = 1

        # Decide whether PAOFLOW must expand the IBZ wedge to the full BZ.
        # If the nscf was run with nosym=.true. and noinv=.true., QE already
        # wrote the full BZ and no expansion is needed.  Otherwise, ask the
        # symmetry expander to fill the full BZ from the wedge; both Hks
        # and Sks are expanded (see open_grid_wrapper).
        #
        # The nscf step is optional: when ``<prefix>.nscf.in`` is absent the
        # driver runs the scf alone (with whatever bands the scf template
        # requests) and reads the symmetry flags from the scf input instead.
        self._has_nscf = isfile(f'{self.prefix}.nscf.in')
        sym_file = f'{self.prefix}.nscf.in' if self._has_nscf else f'{self.prefix}.scf.in'
        sym_blocks, _ = struct_from_inputfile_QE(sym_file)
        nscf_sys = sym_blocks.get('system', {})

        def _qe_true(v):
            # Accept any QE logical-true shorthand: .true., .t., true, t
            # (with arbitrary case and trailing commas/whitespace).
            return str(v).strip().rstrip(',').strip().lower().strip('.') in (
                't',
                'true',
            )

        nosym = _qe_true(nscf_sys.get('nosym', '.false.'))
        noinv = _qe_true(nscf_sys.get('noinv', '.false.'))
        self.expand_wedge = not (nosym and noinv)
        print(f'ACBN0: nscf nosym={nosym}, noinv={noinv} -> expand_wedge={self.expand_wedge}\n')

        # Generate gaussian fits
        print('Generating gaussian fits for pseudopotential basis states.\n')
        # The ATOMIC_SPECIES card only stores the UPF file name; the
        # directory holding the pseudopotentials is given by ``pseudo_dir``
        # in the &control namelist.  Resolve the full path so the Gaussian
        # fitter can open the file regardless of the current working dir.
        pseudo_dir = self.blocks.get('control', {}).get('pseudo_dir', '')
        pseudo_dir = expanduser(pseudo_dir.strip().strip('"').strip("'"))
        self.basis = {}
        self.basis_labels = {}
        self.uspecies = []
        for s in self.cards['ATOMIC_SPECIES'][1:]:
            ele, _, pp = s.split()
            self.uspecies.append(ele)
            pp_path = pp
            if not isfile(pp_path) and pseudo_dir:
                pp_path = join(pseudo_dir, pp)
            atno, basis, labels = gaussian_fit(pp_path, threshold=self.gaussian_threshold)
            self.basis[ele] = basis
            self.basis_labels[ele] = labels

        # Store U values from input template
        if 'HUBBARD' in self.cards:
            self.hubbard_tag = self.cards['HUBBARD'][0]
            for h in self.cards['HUBBARD'][1:]:
                tokens = h.split()
                kind = tokens[0]
                if kind == 'U':
                    _, sym, uval = tokens
                    self.uVals[sym] = float(uval)

                    ele, occ = sym.split('-')
                    if ele not in self.occ_states:
                        self.occ_states[ele] = []
                    self.occ_states[ele].append(occ)

                elif kind == 'V':
                    # V <label1> <label2> <atom_idx1> <atom_idx2> <value>
                    _, sym1, sym2, idx1, idx2, vval = tokens
                    self.vVals[(sym1, sym2, int(idx1), int(idx2))] = float(vval)

                else:
                    # Other HUBBARD entries (J, J0, B, E, ...) are preserved
                    # verbatim via self.hubbard_card() only if added by the
                    # user; not parsed here.
                    pass

            # Store occupations from input template
            nat = len(self.uspecies)
            for s in self.blocks['system']:
                if 'hubbard_occ' in s:
                    i, j = map(int, re.findall(r'\(([^\)]+),([^\)]+)\)', s)[0])
                    if i > nat:
                        msg = f'hubbard_occ index 1 (value:{i}) out of range for {nat} species.'
                        raise ValueError(msg)

                    spec = self.uspecies[i - 1]
                    nstates = len(self.occ_states[spec])
                    if j > nstates:
                        msg = f'hubbard_occ index 2 (value:{j}) out of range, {nstates} states listed for {spec}.'
                        raise ValueError(msg)

                    state = self.occ_states[spec][j - 1]
                    self.occ_values[f'{spec}-{state}'] = float(self.blocks['system'][s])
                    self.hubbard_occ[s] = self.blocks['system'][s]

    def set_hubbard_parameters(self, hubbard):
        htype = type(hubbard)
        if htype in [list, tuple]:
            for h in hubbard:
                ele, occ = h.split('-')
                self.uVals[h] = 0.01
                if ele not in self.occ_states:
                    self.occ_states[ele] = []
                self.occ_states[ele].append(occ)

        elif htype is dict:
            for k, v in hubbard.items():
                self.uVals[k] = 0.01
                if v is None:
                    continue

                try:
                    vtype = type(v)
                    if vtype not in [list, tuple]:
                        self.uVals[k] = float(v)

                    else:
                        if len(v) >= 1 and v[0] is not None:
                            self.uVals[k] = v[0]

                        if len(v) >= 2 and v[1] is not None:
                            ele, occ = k.split('-')
                            if ele not in self.occ_states:
                                self.occ_states[ele] = [occ]

                            elif occ not in self.occ_states[ele]:
                                self.occ_states[ele].append(occ)

                            i = 1 + self.uspecies.index(ele)
                            j = 1 + self.occ_states[ele].index(occ)

                            key = f'hubbard_occ({i},{j})'
                            self.hubbard_occ[key] = float(v[1])

                except Exception as e:
                    print(
                        'Dictionary values should either be the initial U value, or a list/tuple containing (initial U, hubbard occupation).'
                    )
                    raise e

        else:
            msg = 'Input type should be either list or dict.'
            raise TypeError(msg)

    def set_intersite_V_parameters(self, vpairs):
        """Register intersite Hubbard V pairs and their initial values.

        Parameters
        ----------
        vpairs : list, tuple, or dict
            Either an iterable of ``(label1, label2, atom_idx1, atom_idx2,
            V_init)`` tuples, or a dict mapping
            ``(label1, label2, atom_idx1, atom_idx2)`` to ``V_init``.
            ``label*`` follow the QE ``species-orbital`` convention (e.g.
            ``'Ni-3d'``); ``atom_idx*`` are 1-based atom indices as used in
            the QE ``HUBBARD`` card.  ``V_init`` may be ``None``, in which
            case 0.01 eV is used as a seed.
        """
        if isinstance(vpairs, dict):
            items = vpairs.items()
        elif isinstance(vpairs, (list, tuple)):
            items = []
            for entry in vpairs:
                if len(entry) != 5:
                    msg = 'Each V entry must be (label1, label2, atom_idx1, atom_idx2, V_init).'
                    raise ValueError(msg)
                key = (entry[0], entry[1], int(entry[2]), int(entry[3]))
                items.append((key, entry[4]))
        else:
            msg = 'Input type should be either list/tuple of tuples or dict.'
            raise TypeError(msg)

        for key, v_init in items:
            self.vVals[key] = 0.01 if v_init is None else float(v_init)

    def optimize_hubbard_U(self, convergence_threshold=0.01):
        print('\nBeginning self-consistent loop.\n')
        itr = 0
        converged = False
        while not converged:
            itr += 1
            print(f'Iteration #{itr}\n')

            self.run_dft(self.prefix, self.uspecies, self.uVals)

            save_prefix = self.blocks['control']['prefix'].strip('"').strip("'")
            self.run_paoflow(self.prefix, save_prefix)

            new_U = self.run_acbn0(self.prefix)

            converged = True
            print('\nNew U values:')
            for k, v in new_U.items():
                print(f'  {k} : {v}')
                if converged and np.abs(self.uVals[k] - v) > convergence_threshold:
                    converged = False
            print(flush=True)

            self.uVals = new_U

    def exec_QE(self, executable, fname):
        exe = expanduser(join(self.qpath, executable))
        fout = fname.replace('in', 'out')

        command = f'{self.mpi_qe} {exe} {self.qoption}'
        print(
            f'Starting Process: {self.mpi_qe} {exe} {self.qoption} < {fname} > {fout}',
            flush=True,
        )
        with open(fname, 'r') as qe_in, open(fout, 'w') as qe_out:
            subprocess.run(
                [tok for tok in command.split(' ') if tok],
                stdin=qe_in,
                stdout=qe_out,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def _python_exec(self):
        # ``python_path`` may point either at the directory containing the
        # python interpreter (the historical convention) or directly at the
        # interpreter executable.  Detect the latter so both forms work.
        ppath = expanduser(self.ppath)
        if ppath and isfile(ppath):
            return ppath
        return join(ppath, 'python')

    def exec_PAOFLOW(self):
        python_exec = self._python_exec()
        command = f'{self.mpi_python} {python_exec} acbn0.py'
        print(f'Starting Process: {command} > paoflow.out', flush=True)
        with open('paoflow.out', 'w') as paoflow_out:
            subprocess.run(
                [tok for tok in command.split(' ') if tok],
                stdout=paoflow_out,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def hubbard_card(self):
        if len(self.uVals) == 0 and len(self.vVals) == 0:
            msg = (
                'No U or V found. Add them to the template inputfiles or '
                'with set_hubbard_parameters / set_intersite_V_parameters.'
            )
            raise ValueError(msg)

        card = [self.hubbard_tag]
        for k, v in self.uVals.items():
            card.append(f' U {k} {v}')

        # Emit each undirected V channel only once.  QE's HUBBARD card
        # check (PW/src/read_cards.f90, card_hubbard) considers two V
        # entries to be the same channel whenever they share the same
        # unordered atom-index pair AND the same unordered (n,l)
        # manifold pair -- because internally the manifold key strips
        # the species prefix.  Concretely:
        #   ``V Ga-4s As-4p 1 2``  and  ``V Ga-4p As-4s 1 2``
        # both reduce to atom-pair {1,2} with manifold-set {4s,4p} and
        # are rejected by QE even though they couple physically
        # different orbital pairs.  We canonicalise on
        #   (frozenset(atom_indices), frozenset(orbital_labels))
        # where ``orbital_labels`` is the manifold name stripped of the
        # species prefix.  When two ACBN0-computed V values collapse
        # to the same canonical channel we emit their *average* (this
        # is a QE limitation, not an ACBN0 one).
        def _orb(sym):
            return sym.split('-', 1)[1] if '-' in sym else sym

        grouped = {}  # canonical -> (representative entry, [values])
        for (sym1, sym2, idx1, idx2), v in self.vVals.items():
            canonical = (
                frozenset((idx1, idx2)),
                frozenset((_orb(sym1), _orb(sym2))),
            )
            if canonical in grouped:
                grouped[canonical][1].append(v)
            else:
                grouped[canonical] = ((sym1, sym2, idx1, idx2), [v])
        for (sym1, sym2, idx1, idx2), vals in grouped.values():
            v_emit = sum(vals) / len(vals)
            card.append(f' V {sym1} {sym2} {idx1} {idx2} {v_emit}')

        return card

    def run_dft(self, prefix, species, uVals):
        from .inputs.file_io import create_atomic_inputfile, struct_from_inputfile_QE

        blocks, cards = struct_from_inputfile_QE(f'{prefix}.scf.in')
        cards['HUBBARD'] = self.hubbard_card()
        for k, v in self.hubbard_occ.items():
            blocks['system'][k] = v
        # Restart the scf from the charge density converged in the
        # previous iteration.  Across the self-consistent U/U+V loop the
        # only thing that changes between successive scf runs is the
        # (small) update to the Hubbard parameters, so the previous
        # density is an excellent starting guess and dramatically cuts
        # the number of scf steps.  The first run has no density on disk
        # and therefore starts from the atomic superposition.
        if self._scf_density_available:
            blocks.setdefault('electrons', {})
            blocks['electrons']['startingpot'] = "'file'"
        create_atomic_inputfile('scf', blocks, cards)

        blocks, cards = (
            struct_from_inputfile_QE(f'{prefix}.nscf.in') if self._has_nscf else (None, None)
        )
        if self._has_nscf:
            cards['HUBBARD'] = self.hubbard_card()
            for k, v in self.hubbard_occ.items():
                blocks['system'][k] = v
            create_atomic_inputfile('nscf', blocks, cards)

        if self.use_local_basis:
            # The local-basis projection (PAOFLOW.projections) replaces
            # projwfc.x entirely, so no <prefix>.projwfc.in is read and
            # projwfc.x is never launched.
            executables = {'scf': 'pw.x', 'nscf': 'pw.x'}
            calcs = ['scf', 'nscf'] if self._has_nscf else ['scf']
        else:
            blocks, cards = struct_from_inputfile_QE(f'{prefix}.projwfc.in')
            create_atomic_inputfile('projwfc', blocks, cards)

            executables = {'scf': 'pw.x', 'nscf': 'pw.x', 'projwfc': 'projwfc.x -nd 1'}
            calcs = ['scf', 'nscf', 'projwfc'] if self._has_nscf else ['scf', 'projwfc']
        for c in calcs:
            self.exec_QE(executables[c], f'{c}.in')

        # The scf has now written a converged charge density to
        # ``outdir/<prefix>.save``; subsequent iterations can reuse it.
        self._scf_density_available = True

    def run_paoflow(self, prefix, save_prefix):
        from .inputs.file_io import create_acbn0_inputfile

        fstr = f'{prefix}_PAO_bands' + '{}.in'
        calcs = []
        if self.nspin == 1:
            calcs.append(fstr.format(''))
        else:
            calcs.append(fstr.format('_up'))
            calcs.append(fstr.format('_down'))

        # QE writes the wavefunction save folder to ``outdir/<prefix>.save``.
        # PAOFLOW resolves ``savedir`` relative to its workpath ("./"), so the
        # QE ``outdir`` (if any) must be folded into the save path, otherwise
        # PAOFLOW would look for ``<prefix>.save`` in the run directory.
        outdir = self.blocks.get('control', {}).get('outdir', '')
        outdir = outdir.strip().strip('"').strip("'")
        savedir = f'{save_prefix}.save'
        if outdir:
            savedir = join(outdir, savedir)

        create_acbn0_inputfile(
            savedir,
            self.pthr,
            self.outputdir,
            expand_wedge=self.expand_wedge,
            use_local_basis=self.use_local_basis,
            basispath=self.basispath,
            configuration=self.configuration,
        )
        self.exec_PAOFLOW()

    def read_cell_atoms(self, fname):
        lines = None
        with open(fname, 'r') as f:
            lines = f.readlines()

        il = 0
        while 'lattice parameter' not in lines[il]:
            il += 1
        alat = float(lines[il].split()[4])

        while 'number of atoms/cell' not in lines[il]:
            il += 1
        nat = int(lines[il].split()[4])

        while 'crystal axes:' not in lines[il]:
            il += 1
        il += 1
        lattice = np.array([[float(v) for v in lines[il + i].split()[3:6]] for i in range(3)])

        while 'site n.' not in lines[il]:
            il += 1
        il += 1
        # species = []
        positions = np.empty((nat, 3), dtype=float)
        for i in range(nat):
            ls = lines[il + i].split()
            # species.append(ls[1])
            positions[i, :] = np.array([float(v) for v in ls[6:9]])

        lattice *= alat
        positions *= alat
        return lattice, positions

    def hubbard_orbital(self, ele):
        orb = ele[-1]
        orbitals = {'s': 0, 'p': 1, 'd': 2}

        if orb in orbitals:
            return orbitals[orb]

        else:
            raise Exception(f'Element {ele} has no defined Hubbard orbital')

    @staticmethod
    def _acbn0_denominator(nlm_up, nlm_dn):
        """Compute the U and J denominators of Agapito PRX 5, 011006 (2015),
        Eqs. (10–11), for arbitrary spin polarization.

        For ``nspin == 1`` pass ``nlm_dn = nlm_up``: ``Nbb == Naa`` and the
        result reduces to ``den_U = 2*(Naa + Nab)`` and ``den_J = 2*Naa``.
        """
        Naa = Nbb = 0.0
        Nab = 0.0
        n = nlm_up.shape[0]
        for i1 in range(n):
            for i2 in range(n):
                Nab += nlm_up[i1] * nlm_dn[i2]
                if i1 != i2:
                    Naa += nlm_up[i1] * nlm_up[i2]
                    Nbb += nlm_dn[i1] * nlm_dn[i2]
        den_U = Naa.real + Nbb.real + 2 * Nab.real
        den_J = Naa.real + Nbb.real
        return den_U, den_J

    def _projwfc_hubbard_manifolds(self, state_lines):
        """Build the Hubbard manifolds from ``projwfc.x`` ``state #`` lines.

        Returns a dict mapping each Hubbard orbital key to
        ``(basis_dm, basis_2e)`` PAO-index arrays, where ``basis_dm`` is the
        density-matrix manifold (all atoms of the species, matching L) and
        ``basis_2e`` is the single representative shell used for the
        two-electron integrals.

        Selection is label-aware: the projwfc ``wfc N`` index is mapped to
        the pseudo's N-th ``PP_PSWFC`` shell label (via
        :attr:`self.basis_labels`) so that a semicore shell of the same L as
        the Hubbard valence shell (e.g. Mg ``2s`` alongside the ``3s``
        valence) is excluded.  Without this filter the first (semicore)
        shell would be picked, giving an unphysically large U.
        """
        manifolds = {}
        for orb in self.uVals:
            ostates = []
            ustates = []
            species_label = orb.split('-')[0]
            horb = self.hubbard_orbital(orb)
            shell_label = self._shell_label(orb)
            labels = self.basis_labels.get(species_label)
            for n, sl in enumerate(state_lines):
                mat = re.search(
                    r'atom\s+\d+\s*\(\s*(\S+)\s*\)\s*,\s*wfc\s+(\d+)\s*\(\s*l\s*=\s*(\d+)',
                    sl,
                )
                if mat is None:
                    continue
                oele = mat.group(1).strip()
                wfc_idx = int(mat.group(2))
                oL = int(mat.group(3))
                if oL != horb:
                    continue
                # Map wfc index -> pseudo shell label and keep only the
                # Hubbard valence shell, excluding same-L semicore shells.
                if labels is not None and 0 <= wfc_idx - 1 < len(labels):
                    if labels[wfc_idx - 1].upper() != shell_label:
                        continue
                if species_label in oele:
                    ostates.append(n)
                    if species_label == oele:
                        ustates.append(n)
            if not ustates:
                raise RuntimeError(
                    f'No projwfc states matched Hubbard manifold {orb!r} '
                    f'(species {species_label!r}, shell {shell_label!r}).'
                )
            sstates = [ustates[0]]
            for us in ustates[1:]:
                if us == 1 + sstates[-1]:
                    sstates.append(us)
                else:
                    break
            manifolds[orb] = (np.array(ostates), np.array(sstates))
        return manifolds

    def _load_pao_basis_meta(self):
        """Read the per-orbital PAO metadata dumped by the local-basis
        projection driver (``<outputdir>/pao_basis.dat``).

        Returns a list of dicts (sorted by PAO index) with keys
        ``index`` (0-based PAO index), ``atom`` (1-based atom index),
        ``elem`` (species label), ``l``, ``m`` and ``label`` (shell label,
        e.g. ``'3P'``).
        """
        if self._pao_meta is not None:
            return self._pao_meta
        path = join(self.outputdir, 'pao_basis.dat')
        meta = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                meta.append(
                    {
                        'index': int(parts[0]),
                        'atom': int(parts[1]),
                        'elem': parts[2],
                        'l': int(parts[3]),
                        'm': int(parts[4]),
                        'label': parts[5],
                    }
                )
        meta.sort(key=lambda d: d['index'])
        self._pao_meta = meta
        return meta

    @staticmethod
    def _shell_label(orb):
        """Return the upper-case shell label of a Hubbard orbital key.

        ``'Si-3p' -> '3P'``, ``'Fe-3d' -> '3D'``.
        """
        return orb.split('-')[1].upper()

    def _local_hubbard_manifolds(self):
        """Build the Hubbard manifolds from the local-basis PAO metadata.

        Selection is by shell *label* (not merely by angular momentum), so
        polarization shells that share the Hubbard L (e.g. ``4P`` alongside
        the ``3P`` valence shell in the ``standard`` basis) are excluded.

        Returns a dict mapping each Hubbard orbital key to
        ``(basis_dm, basis_2e)`` PAO-index arrays.
        """
        meta = self._load_pao_basis_meta()
        manifolds = {}
        for orb in self.uVals:
            species_label = orb.split('-')[0]
            horb = self.hubbard_orbital(orb)
            shell_label = self._shell_label(orb)

            ostates = []
            ustates = []
            for e in meta:
                if e['l'] != horb or e['label'].upper() != shell_label:
                    continue
                if species_label in e['elem']:
                    ostates.append(e['index'])
                    if species_label == e['elem']:
                        ustates.append(e['index'])
            if not ustates:
                raise RuntimeError(
                    f'No PAO orbitals matched Hubbard manifold {orb!r} '
                    f'(species {species_label!r}, shell {shell_label!r}) in '
                    f'{join(self.outputdir, "pao_basis.dat")}.'
                )
            # First contiguous run of the species' orbitals = the valence
            # shell on the first matching atom.
            sstates = [ustates[0]]
            for us in ustates[1:]:
                if us == 1 + sstates[-1]:
                    sstates.append(us)
                else:
                    break
            manifolds[orb] = (np.array(ostates), np.array(sstates))
        return manifolds

    def _local_gauss_basis(self, nawf, coords):
        """Build a length-``nawf`` Gaussian-basis list aligned with the PAO
        ordering of the local-basis Hamiltonian.

        Only the Hubbard valence-shell positions are populated with the
        UPF-derived contracted Gaussians (the only entries indexed by the
        two-electron / density-matrix code); all other positions hold an
        inert placeholder so that index access remains valid.
        """
        from collections import defaultdict

        from .utils.pyints import CGBF

        meta = self._load_pao_basis_meta()
        placeholder = CGBF(np.zeros(3), 'X')
        placeholder.pnorms.append(1.0)
        placeholder.pexps.append(1.0)
        placeholder.pcoefs.append(1.0)
        placeholder.powers.append((0, 0, 0))

        gauss_basis = [placeholder] * nawf
        for orb in self.uVals:
            species_label = orb.split('-')[0]
            horb = self.hubbard_orbital(orb)
            shell_label = self._shell_label(orb)

            by_atom = defaultdict(list)
            for e in meta:
                if e['l'] != horb or e['label'].upper() != shell_label:
                    continue
                if species_label == e['elem']:
                    by_atom[e['atom']].append(e)

            for atom_idx, entries in by_atom.items():
                entries.sort(key=lambda d: d['index'])
                pos = coords[atom_idx - 1]
                gs = self._atom_shell_gaussians(species_label, pos, horb, shell_label)
                if len(gs) != len(entries):
                    raise RuntimeError(
                        f'Gaussian shell mismatch for {orb!r} on atom '
                        f'{atom_idx}: {len(gs)} Gaussians vs {len(entries)} '
                        f'PAO orbitals. The UPF must provide exactly one '
                        f'l={horb} shell labelled {shell_label!r} for '
                        f'species {species_label!r}.'
                    )
                for e, bf in zip(entries, gs):
                    gauss_basis[e['index']] = bf
        return gauss_basis

    def _atom_shell_gaussians(self, ele, pos_angstrom, L, shell_label=None):
        """Build the CGBFs of the ``(ele, L)`` shell centred at
        ``pos_angstrom`` (Cartesian Ångström).  Returns a list with one
        CGBF per magnetic component (``2L+1`` Gaussians).

        When ``shell_label`` is given (e.g. ``'2S'``) only the UPF
        pseudo-wavefunction carrying that label is used.  This is required
        for pseudopotentials that include semicore states of the same
        angular momentum as the Hubbard valence shell (e.g. Li ``1s`` +
        ``2s``), where selecting purely by ``L`` would return more Gaussian
        shells than the single label-selected PAO valence orbital.
        """
        from .utils.pyints import CGBF

        gauss = []
        origin_bohr = np.asarray(pos_angstrom) * ANGS_TO_BOHR
        labels = self.basis_labels.get(ele)
        wanted = None if shell_label is None else shell_label.upper()
        for ishell, shell in enumerate(self.basis[ele]):
            if wanted is not None and labels is not None:
                if labels[ishell].upper() != wanted:
                    continue
            for subshell in shell:
                lx, ly, lz, _, _ = subshell[0]
                if lx + ly + lz != L:
                    continue
                bf = CGBF(origin_bohr, ele)
                for lx, ly, lz, coeff, zeta in subshell:
                    bf.pnorms.append(1.0)
                    bf.pexps.append(zeta)
                    bf.pcoefs.append(coeff)
                    bf.powers.append((lx, ly, lz))
                gauss.append(bf)
        return gauss

    def run_acbn0(self, prefix):
        BOHR_RADIUS_ANGS = 0.529177e0

        lattice, coords = self.read_cell_atoms('scf.out')
        lattice *= BOHR_RADIUS_ANGS
        coords *= BOHR_RADIUS_ANGS
        nspin = self.nspin

        kpnts, kwght, Sks, Hks_up, Hks_dw = self.read_ham_data(nspin)

        if self.use_local_basis:
            # Local-basis path: the Hubbard manifolds and the index-aligned
            # Gaussian basis are derived from the PAO metadata dumped by the
            # projection driver (pao_basis.dat) rather than from projwfc.out.
            manifolds = self._local_hubbard_manifolds()
            gauss_basis = self._local_gauss_basis(Hks_up.shape[0], coords)
        else:
            sind = 0
            state_lines = open('projwfc.out', 'r').readlines()
            while 'state #' not in state_lines[sind]:
                sind += 1
            send = sind
            while 'state #' in state_lines[send]:
                send += 1
            state_lines = state_lines[sind:send]

            species = []
            # for s in list(set(species)):
            for k, v in self.uVals.items():
                species_label = k.split('-')[0]
                species.append(species_label)

            gauss_basis = self.getbasis(self.basis, species, lattice, coords)
            manifolds = self._projwfc_hubbard_manifolds(state_lines)

        uVals = {}

        # Cache the per-k generalized eigendecomposition once per spin
        # channel; reused across every Hubbard orbital below so we don't
        # re-diagonalize H(k), S(k) for each species.
        eigvec_up = self._eigh_all_k(Hks_up, Sks)
        eigvec_dn = self._eigh_all_k(Hks_dw, Sks) if nspin == 2 else None

        for orb, v in self.uVals.items():
            basis_dm, basis_2e = manifolds[orb]

            dk, nlm = self.Dk(basis_dm, basis_2e, Hks_up, Sks, eigvec_cache=eigvec_up)
            nlm = self.Nmm(nlm, Hks_up, kwght)

            if nspin == 1:
                dk_dn = None
                nlmd = nlm
            else:
                dk_dn, nlmd = self.Dk(basis_dm, basis_2e, Hks_dw, Sks, eigvec_cache=eigvec_dn)
                nlmd = self.Nmm(nlmd, Hks_dw, kwght)

            den_U, den_J = self._acbn0_denominator(nlm, nlmd)

            DR_up = self.DR(dk, kwght)

            DR_dn = DR_up
            if nspin == 2:
                DR_dn = self.DR(dk_dn, kwght)

            data = {
                'DR_up': DR_up,
                'DR_dn': DR_dn,
                'basis': gauss_basis,
                'basis_2e': basis_2e,
            }
            with open(join(self.outputdir, 'data.pkl'), 'wb') as f:
                pickle.dump(data, f)

            # compute hartree energy in parallel
            python_exec = self._python_exec()
            command = f'{self.mpi_hartree} {python_exec} compute_hartree.py'
            subprocess.run([tok for tok in command.split(' ') if tok], check=True)

            with open(join(self.outputdir, 'tmp_uj.pkl'), 'rb') as f:
                uj = pickle.load(f)
            num_U = uj['U']
            num_J = uj['J']

            hartree_to_eV = 27.211396132
            U = U_eff = hartree_to_eV * num_U / den_U
            if den_J == 0:
                J = 'Inf'
            else:
                J = hartree_to_eV * num_J / den_J
                U_eff -= J

            with open(f'{orb}_UJ.txt', 'w') as f:
                f.write(f'U : {U}\nJ : {J}\nU_eff : {U_eff}\n')

            uVals[orb] = U_eff

        return uVals

    def getbasis(self, basis, species, lattice, coords):
        from .utils.pyints import CGBF

        basis_functions = []
        for a, pos in zip(species, coords):
            for shell in basis[a]:
                for subshell in shell:
                    bf = CGBF(pos * 1.88973, a)
                    for lx, ly, lz, coeff, zeta in subshell:
                        bf.pnorms.append(1.0)
                        bf.pexps.append(zeta)
                        bf.pcoefs.append(coeff)
                        bf.powers.append((lx, ly, lz))

                    basis_functions.append(bf)

        return basis_functions

    def _eigh_all_k(self, Hks, Sks):
        """Solve the generalized eigenproblem ``H(k) c = ε S(k) c`` at every
        k-point and return per-k ``(eig, vec)`` tuples.

        The result depends only on ``(Hks, Sks)`` and is therefore safe to
        cache across the orbital loop in :meth:`run_acbn0` (and across the
        intersite-V loop in :class:`eACBN0`), avoiding redundant per-k
        diagonalizations when several Hubbard species are present.

        Parameters
        ----------
        Hks, Sks : (nbasis, nbasis, nkpnts) complex ndarray

        Returns
        -------
        list of (eig, vec)
            ``eig`` is ``(nbasis,)`` real, ``vec`` is ``(nbasis, nbasis)``
            complex; length ``nkpnts``.
        """
        from scipy.linalg import eigh

        nkpnts = Hks.shape[2]
        return [eigh(Hks[:, :, ik], Sks[:, :, ik]) for ik in range(nkpnts)]

    def Dk(self, basis_dm, basis_2e, Hks, Sks, eigvec_cache=None):
        """Build the per-k density matrix ``D(k)`` and the per-k Mulliken
        projection ``n_{lm}(k)`` on ``basis_2e`` for the occupied manifold.

        Parameters
        ----------
        basis_dm, basis_2e : 1D int ndarray
            PAO indices defining the density-matrix block and the
            two-electron-integral block, respectively.
        Hks, Sks : (nbasis, nbasis, nkpnts) complex ndarray
        eigvec_cache : list of (eig, vec) or None, optional
            If provided, the per-k generalized eigenproblem is *not*
            re-solved; values from the cache are used instead.  Produced by
            :meth:`_eigh_all_k`.
        """
        from scipy.linalg import eigh

        nbasis, _, nkpnts = Hks.shape
        size_dm, size_2e = basis_dm.shape[0], basis_2e.shape[0]
        D_k = np.zeros((nbasis, nbasis, nkpnts), dtype=complex)
        nlm_k = np.zeros((size_2e, nbasis, nkpnts), dtype=complex)

        # Find the density matrix for each k
        for ik in range(nkpnts):
            if eigvec_cache is not None:
                eig, vec = eigvec_cache[ik]
            else:
                eig, vec = eigh(Hks[:, :, ik], Sks[:, :, ik])

            occ_ind = np.where(eig <= 0.0)[0]
            nocc = len(occ_ind)

            lm_dm = np.zeros((size_dm, nocc), dtype=complex)
            lm_2e = np.zeros((size_2e, nocc), dtype=complex)

            sk_dm = Sks[basis_dm, :, ik]
            sk_2e = Sks[basis_2e, :, ik]
            for im in range(nocc):
                vim = vec[:, im]
                lm_dm[:, im] = np.conj(vim[basis_dm]) * sk_dm.dot(vim)
                lm_2e[:, im] = np.conj(vim[basis_2e]) * sk_2e.dot(vim)

            evec2 = vec[:, occ_ind]
            nlm_k[:, :nocc, ik] = lm_2e
            lm_dm = np.sum(lm_dm, axis=0)

            D_k[:, :, ik] = np.tensordot(np.conj(evec2 * lm_dm), evec2, axes=([1], [1]))

        nlm_k[:, :, 0] = 0
        return D_k, nlm_k

    def Nmm(self, nlm, Hks, kwght):
        # Vectorised: nlm_aux[lm, b] = sum_k kwght[k] * nlm[lm, b, k].
        kwght = np.asarray(kwght)
        total_w = float(np.sum(kwght))
        nlm_aux = np.tensordot(nlm, kwght, axes=([2], [0]))  # (lm_size, nbasis)
        return np.sum(nlm_aux, axis=1) / total_w

    def DR(self, Dk, kwght):
        # Vectorised: D[a, b] = sum_k kwght[k] * Dk[a, b, k] / sum(kwght).
        kwght = np.asarray(kwght)
        total_w = float(np.sum(kwght))
        D = np.tensordot(Dk, kwght, axes=([2], [0]))  # (nawf, nawf)
        return (D / total_w).real

    def read_ham_data(self, nspin):
        kpnts = np.loadtxt(open(join(self.outputdir, 'k.txt'), 'r'))
        kwght = np.loadtxt(open(join(self.outputdir, 'wk.txt'), 'r'))

        if len(kpnts.shape) == 1:
            kpnts = np.array([kpnts])
            kwght = np.array([kwght])
        nkpnts = kpnts.shape[0]

        kovp = np.load(join(self.outputdir, 'kovp.npy'))
        nbasis = int(np.sqrt(kovp.shape[0] / nkpnts))
        kovp = kovp.reshape((nbasis, nbasis, nkpnts))

        kham_up = kham_dn = None
        for ispin in range(nspin):
            fname = 'kham'
            if nspin == 2:
                fname += '_up' if ispin == 0 else '_dn'
            fname += '.npy'

            kham = np.load(join(self.outputdir, fname))
            kham = kham.reshape((nbasis, nbasis, nkpnts))

            if ispin == 0:
                kham_up = kham
            elif ispin == 1:
                kham_dn = kham

        return kpnts, kwght, kovp, kham_up, kham_dn


class eACBN0(ACBN0):
    """Extended ACBN0 (DFT+U+V) driver.

    This class implements the *extended* ACBN0 self-consistent scheme that
    augments the standard on-site Hubbard U calculation with the intersite
    Hubbard V correction, following the formulation of

        S.-H. Lee and Y.-W. Son, *First-principles approach with a
        pseudo-hybrid density functional for extended Hubbard interactions*,
        Phys. Rev. Research **2**, 043410 (2020).

    Subclass of :class:`ACBN0`.  ``eACBN0`` reuses the entire on-site U
    machinery (template parsing, ``pw.x`` / ``projwfc.x`` / PAOFLOW launches,
    ACBN0 Hartree integrals, HUBBARD card emission) and adds the bookkeeping
    and numerics required to evaluate V for an arbitrary set of atomic pairs
    — including their periodic images.

    Theoretical background
    ----------------------
    Given a pair of Hubbard-active sites ``(I, J)`` separated by a Bravais
    lattice translation ``R``, the intersite Hubbard parameter is defined as
    (Eq. 8 of the reference)

    .. math::

        V^{IJ}(R) = \\frac{
            \\frac{1}{2}\\sum_{ijkl, \\sigma\\sigma'}
                P^{IJ\\sigma}_{ij}(R)\\, P^{JI\\sigma'}_{kl}(-R)\\,
                (ij|kl)
        }{
            \\sum_{ij, \\sigma\\sigma'} n^{II\\sigma}_{ii} n^{JJ\\sigma'}_{jj}
            - \\sum_{ij, \\sigma} n^{IJ\\sigma}_{ij}(R) n^{JI\\sigma}_{ji}(-R)
        }

    with the *bare* on-site / intersite occupation matrices ``n`` defined by
    Eq. (2) and their *renormalized* counterparts ``P`` defined by Eqs. (4)
    and (5).  The four-centre electron-repulsion integrals ``(ij|kl)`` are
    evaluated in the auxiliary Gaussian basis fitted to the PAO numerical
    wavefunctions (the same basis used by :class:`ACBN0_Hartree`).

    Workflow
    --------
    The typical usage mirrors :class:`ACBN0` but adds a pair-selection
    step and uses :meth:`optimize_hubbard_UV` instead of
    ``optimize_hubbard_U``::

        from PAOFLOW.ACBN0 import eACBN0

        e = eACBN0('MnO', workdir='./',
                   mpi_qe='mpirun -np 8',
                   mpi_python='mpirun -np 1',
                   mpi_hartree='mpirun -np 8',
                   qe_path='/path/to/qe/bin',
                   python_path='/path/to/python/bin',
                   outputdir='./tmp/')

        e.set_hubbard_parameters({'Mn-3d': 5.0, 'O-2p': 1.0})
        e.set_intersite_pairs(cutoff=2.5, V_init=0.5)
        e.optimize_hubbard_UV(convergence_threshold=0.05, max_iter=25,
                              mixing=0.7)

    Each outer iteration of :meth:`eACBN0.optimize_hubbard_UV`:

    1. Writes the current ``HUBBARD`` card (U on every declared orbital, V
       on every registered pair) into the SCF/NSCF templates and runs
       ``pw.x`` (scf), ``pw.x`` (nscf) and ``projwfc.x``.
    2. Runs PAOFLOW to produce the PAO Hamiltonian/overlap and the
       reduced-zone k-grid dumps.
    3. Calls :meth:`eACBN0.run_acbn0` for the on-site U values.
    4. Calls :meth:`eACBN0.run_eacbn0_V` which, for every registered pair,
       builds the pair density matrices and dispatches the four-centre
       integral evaluation to ``compute_hartree_v.py`` under
       ``mpi_hartree``.
    5. Applies linear mixing to both U and V and checks for convergence
       (``max |Δ| < convergence_threshold`` across all parameters).

    Public API
    ----------
    Class :class:`eACBN0` adds the following methods on top of
    :class:`ACBN0`:

    - :meth:`eACBN0.set_intersite_pairs` — register V pairs either from an
      explicit list (with optional ``(n_a, n_b, n_c)`` image labels) or via
      a real-space cutoff search restricted to user-selected species pairs.
    - :meth:`eACBN0.print_intersite_pairs` — print a human-readable summary
      of every registered pair (atom indices, image, distance, seed V).
    - :meth:`eACBN0.run_eacbn0_V` — evaluate Eq. (8) for every registered
      pair using the current PAOFLOW dump and ``projwfc.out``.  Collapses
      multiple images of the same ``(label1, label2, i1, i2)`` key to the
      minimum-image representative.
    - :meth:`eACBN0.optimize_hubbard_UV` — joint U+V self-consistent loop
      with linear mixing.

    Internal helpers (prefixed with an underscore) implement geometry
    parsing from the QE input cards (:meth:`eACBN0._geometry_from_cards`),
    periodic neighbour enumeration with cell-shape-aware image windows
    (:meth:`eACBN0._image_window`, :meth:`eACBN0._enumerate_pairs`),
    construction of per-site contracted-Gaussian basis functions
    (:meth:`eACBN0._atom_shell_gaussians`), pair density-matrix assembly
    (:meth:`eACBN0._pair_density_matrices`) and the MPI launcher for the
    four-centre integrals (:meth:`eACBN0._launch_compute_hartree_v`).

    Conventions
    -----------
    - Energies are in eV; lengths are in Ångström in the public API and in
      Bohr internally where the Gaussian integrals require it.
    - Atom indices follow the QE convention: 1-based, ordered as listed in
      the ``ATOMIC_POSITIONS`` card.
    - Periodic images are labelled by integer triples ``(n_a, n_b, n_c)``
      applied to atom ``i2`` of a pair: ``R = n_a a_1 + n_b a_2 + n_c a_3``.
    - k-points returned by PAOFLOW are in crystal coordinates by default;
      pass ``kpnts_are_cartesian=True`` to :meth:`eACBN0.run_eacbn0_V` (and
      hence to :meth:`eACBN0.optimize_hubbard_UV`) if your PAOFLOW dump
      uses Cartesian Bohr\\ :sup:`-1` instead.
    - ``mpi_hartree`` (kwarg of the underlying :class:`ACBN0`) controls the
      MPI launcher for the pure-Python Coulomb integrals; it can — and
      usually should — differ from ``mpi_python``, which controls the
      PAOFLOW step.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :class:`ACBN0`.

    Notes
    -----
    - Spin-restricted runs (``nspin = 1``) reuse a single density matrix
      for both spin channels by halving the doubly-occupied DM, so that the
      on-site limit (I = J, R = 0) reproduces the standard
      :class:`ACBN0` result exactly.
    - The underlying PAOFLOW call uses
      ``pao_hamiltonian(write_binary=True, expand_wedge=True)``.  The resulting ``Hks``
      is on the full BZ produced by QE The reshape to ``(nawf, nawf, nk1, nk2, nk3, nspin)`` in
      :func:`PAOFLOW.hamiltonian.do_build_pao_hamiltonian.do_build_pao_hamiltonian`
      is now guarded by an array-size check, so the same code path also
      handles the IBZ k-grid that arises from the bare ACBN0 (U-only)
      stage when run without ``nosym/noinv``.
    - :func:`PAOFLOW.hamiltonian.do_build_pao_hamiltonian.build_Hks` now applies a
      *per-k* projectability filter (``pthr_local``, defaulting to
      ``0.5 * pthr``) in addition to the global ``pthr``.  Bands whose
      local PAO projection at a given k vanishes (typically one of a
      degenerate set at high-symmetry points such as Γ, where QE's gauge
      choice can leave it with near-zero atomic-orbital content) are
      dropped from the construction at that k-point only.  Without this
      guard the previous code renormalised the vanishing projection vector
      to unit norm, corrupting ``H(k)`` at high-symmetry points and, in
      turn, the occupations entering the ACBN0 / eACBN0 numerators and
      denominators.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Per-image neighbour metadata, keyed by
        # ``(label1, label2, atom_idx1, atom_idx2, n_a, n_b, n_c)``.
        # ``self.vVals`` (inherited from ACBN0) remains the source of truth
        # for what gets written into the QE ``HUBBARD`` card.
        self.vPairs = {}

    # ------------------------------------------------------------------ #
    # Geometry parsing                                                   #
    # ------------------------------------------------------------------ #
    def _alat_bohr(self):
        """Return ``alat`` in Bohr from the QE ``&system`` block.

        Supports either ``celldm(1)`` (Bohr) or ``A`` (Ångström).  Returns
        ``None`` if neither is present (e.g. ``CELL_PARAMETERS`` is in
        explicit units).
        """
        sys = self.blocks.get('system', {})
        for k, v in sys.items():
            kl = k.lower().replace(' ', '')
            if kl == 'celldm(1)':
                return float(v.replace('d', 'e').replace('D', 'e'))
            if kl == 'a':
                return float(v.replace('d', 'e').replace('D', 'e')) / BOHR_RADIUS_ANGS
        return None

    def _lattice_from_ibrav(self):
        """Reconstruct lattice vectors (Å) from the QE ``ibrav`` + ``celldm``
        (or ``A``/``B``/``C``/``cos*``) parameters in the ``&system`` block.

        Used when the template specifies the cell through ``ibrav`` instead
        of an explicit ``CELL_PARAMETERS`` card.
        """
        from .inputs.lattice_format import celldm_from_namelist, lattice_format_QE

        sys = self.blocks.get('system', {})
        ibrav_raw = None
        for k, v in sys.items():
            if k.lower().replace(' ', '') == 'ibrav':
                ibrav_raw = v
                break
        if ibrav_raw is None:
            raise ValueError(
                'Neither a CELL_PARAMETERS card nor ibrav was found in the '
                'template; cannot determine the cell geometry for eACBN0.'
            )
        ibrav = int(float(str(ibrav_raw).replace('d', 'e').replace('D', 'e')))
        if ibrav == 0:
            raise ValueError(
                'ibrav=0 requires an explicit CELL_PARAMETERS card, which was '
                'not found in the template.'
            )
        celldm = celldm_from_namelist(sys, ibrav)
        lattice_bohr = lattice_format_QE(ibrav, celldm)
        return lattice_bohr * BOHR_RADIUS_ANGS

    def _geometry_from_cards(self):
        """Parse lattice (Å) and Cartesian atomic positions (Å) from the
        input cards captured in :attr:`self.cards`.

        Returns
        -------
        lattice : (3, 3) ndarray
            Lattice vectors as rows, in Ångström.
        positions : (nat, 3) ndarray
            Atomic positions in Cartesian Ångström.
        species : list[str]
            Per-atom species labels (length ``nat``).
        """
        if 'CELL_PARAMETERS' not in self.cards:
            # No explicit cell card: reconstruct the lattice from the QE
            # ``ibrav`` + ``celldm`` (or A/B/C/cos*) parameters instead.
            lattice = self._lattice_from_ibrav()
        else:
            header = self.cards['CELL_PARAMETERS'][0].lower()
            if 'angstrom' in header:
                cell_unit = 'angstrom'
            elif 'bohr' in header:
                cell_unit = 'bohr'
            else:
                cell_unit = 'alat'

            vecs = []
            for ln in self.cards['CELL_PARAMETERS'][1:4]:
                vecs.append([float(x) for x in ln.split()[:3]])
            lattice = np.asarray(vecs, dtype=float)

            if cell_unit == 'bohr':
                lattice *= BOHR_RADIUS_ANGS
            elif cell_unit == 'alat':
                alat_bohr = self._alat_bohr()
                if alat_bohr is None:
                    raise ValueError(
                        "CELL_PARAMETERS in 'alat' but no celldm(1)/A found in &system block."
                    )
                lattice *= alat_bohr * BOHR_RADIUS_ANGS

        if 'ATOMIC_POSITIONS' not in self.cards:
            raise ValueError('ATOMIC_POSITIONS card not found in template.')

        pos_header = self.cards['ATOMIC_POSITIONS'][0].lower()
        if 'crystal' in pos_header:
            pos_unit = 'crystal'
        elif 'angstrom' in pos_header:
            pos_unit = 'angstrom'
        elif 'bohr' in pos_header:
            pos_unit = 'bohr'
        else:
            pos_unit = 'alat'

        species = []
        raw = []
        for ln in self.cards['ATOMIC_POSITIONS'][1:]:
            tokens = ln.split()
            if len(tokens) < 4:
                continue
            species.append(tokens[0])
            raw.append([float(x) for x in tokens[1:4]])
        raw = np.asarray(raw, dtype=float)

        if pos_unit == 'crystal':
            positions = raw @ lattice
        elif pos_unit == 'bohr':
            positions = raw * BOHR_RADIUS_ANGS
        elif pos_unit == 'alat':
            alat_bohr = self._alat_bohr()
            if alat_bohr is None:
                raise ValueError(
                    "ATOMIC_POSITIONS in 'alat' but no celldm(1)/A found in &system block."
                )
            positions = raw * alat_bohr * BOHR_RADIUS_ANGS
        else:
            positions = raw

        return lattice, positions, species

    # ------------------------------------------------------------------ #
    # Neighbour search                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _image_window(lattice, cutoff):
        """Return per-axis half-windows ``(n_a, n_b, n_c)`` large enough
        that every image within ``cutoff`` is enclosed.

        Uses the inter-planar spacing along each reciprocal direction so
        the search is correct even for highly skewed cells.
        """
        # Inter-planar spacing along a_i is |a_i · n_i| where n_i is the
        # unit normal to the plane spanned by the other two vectors.
        a1, a2, a3 = lattice
        n12 = np.cross(a1, a2)
        n23 = np.cross(a2, a3)
        n31 = np.cross(a3, a1)
        d_a = abs(np.dot(a1, n23)) / np.linalg.norm(n23)
        d_b = abs(np.dot(a2, n31)) / np.linalg.norm(n31)
        d_c = abs(np.dot(a3, n12)) / np.linalg.norm(n12)
        return (
            int(np.ceil(cutoff / d_a)),
            int(np.ceil(cutoff / d_b)),
            int(np.ceil(cutoff / d_c)),
        )

    def _enumerate_pairs(
        self,
        cutoff,
        species_pairs,
        include_onsite,
    ):
        """Return a list of ``(label1, label2, i1, i2, image, R_cart, dist)``
        tuples for every pair of declared Hubbard-active atoms whose
        separation lies within ``cutoff`` (Å).

        ``i1`` and ``i2`` are 1-based atom indices (QE convention),
        ``image = (n_a, n_b, n_c)`` is the lattice translation applied to
        atom ``i2``.
        """
        lattice, positions, species = self._geometry_from_cards()
        nat = len(species)

        # Hubbard-active orbitals per species, derived from the canonical
        # source of truth ``self.uVals`` (keyed by ``'species-orbital'``).
        species_orbs = {}
        for label in self.uVals:
            ele, orb = label.split('-')
            species_orbs.setdefault(ele, []).append(orb)

        # Build the set of (label1, label2) pairs to search for.
        if species_pairs is None:
            labels = sorted({f'{ele}-{orb}' for ele, orbs in species_orbs.items() for orb in orbs})
            wanted_pairs = set()
            for i, a in enumerate(labels):
                for b in labels[i:]:
                    wanted_pairs.add((a, b))
                    wanted_pairs.add((b, a))
        else:
            wanted_pairs = {tuple(p) for p in species_pairs}

        # Precompute label per atom (only for atoms whose species has any
        # declared Hubbard orbital).
        atom_labels = []
        for ele in species:
            if ele in species_orbs:
                atom_labels.append([f'{ele}-{orb}' for orb in species_orbs[ele]])
            else:
                atom_labels.append([])

        n_a, n_b, n_c = self._image_window(lattice, cutoff)

        pairs = []
        cutoff2 = cutoff * cutoff
        for i in range(nat):
            if not atom_labels[i]:
                continue
            for j in range(nat):
                if not atom_labels[j]:
                    continue
                for na in range(-n_a, n_a + 1):
                    for nb in range(-n_b, n_b + 1):
                        for nc in range(-n_c, n_c + 1):
                            if not include_onsite and i == j and na == 0 and nb == 0 and nc == 0:
                                continue
                            R = na * lattice[0] + nb * lattice[1] + nc * lattice[2]
                            d_vec = positions[j] + R - positions[i]
                            d2 = float(d_vec @ d_vec)
                            if d2 > cutoff2:
                                continue
                            dist = float(np.sqrt(d2))
                            for l1 in atom_labels[i]:
                                for l2 in atom_labels[j]:
                                    if (l1, l2) not in wanted_pairs:
                                        continue
                                    pairs.append(
                                        (
                                            l1,
                                            l2,
                                            i + 1,
                                            j + 1,
                                            (na, nb, nc),
                                            R,
                                            dist,
                                        )
                                    )
        return pairs

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def set_intersite_pairs(
        self,
        pairs=None,
        cutoff=None,
        include_onsite=False,
        species_pairs=None,
        V_init=0.01,
    ):
        """Register intersite Hubbard V pairs.

        Exactly one of ``pairs`` or ``cutoff`` must be given (or both —
        in which case the explicit list is registered first, then the
        cutoff search augments it).

        Parameters
        ----------
        pairs : list of tuple, optional
            Explicit pair list.  Each entry is either ``(label1, label2,
            atom_idx1, atom_idx2, V_init)`` (home cell, image = (0,0,0))
            or ``(label1, label2, atom_idx1, atom_idx2, n_a, n_b, n_c,
            V_init)``.  Atom indices are 1-based.  ``V_init`` may be
            ``None`` to use the ``V_init`` keyword default.
        cutoff : float, optional
            Real-space cutoff in Ångström.  All ordered ``(i, j, R)``
            atom pairs whose separation lies within ``cutoff`` are
            enumerated for every label pair selected by
            ``species_pairs``.
        include_onsite : bool, default False
            When ``True`` the cutoff search also includes the trivial
            ``(i, i, R = 0)`` pairs.
        species_pairs : list of (str, str), optional
            Restrict the cutoff search to these ``(label1, label2)``
            pairs.  Defaults to every unordered pair of currently
            declared Hubbard orbitals (both directions stored).
        V_init : float, default 0.01
            Seed V value (eV) used when an entry's ``V_init`` is ``None``
            and for cutoff-enumerated pairs.
        """
        if pairs is None and cutoff is None:
            raise ValueError('Provide either `pairs` or `cutoff`.')

        if pairs is not None:
            self._register_explicit_pairs(pairs, V_init)

        if cutoff is not None:
            self._register_cutoff_pairs(
                cutoff,
                species_pairs,
                include_onsite,
                V_init,
            )

    def _register_explicit_pairs(self, pairs, V_init_default):
        lattice, positions, _ = self._geometry_from_cards()
        for entry in pairs:
            if len(entry) == 5:
                l1, l2, i1, i2, v = entry
                image = (0, 0, 0)
            elif len(entry) == 8:
                l1, l2, i1, i2, na, nb, nc, v = entry
                image = (int(na), int(nb), int(nc))
            else:
                raise ValueError(
                    'Each explicit V entry must be a 5-tuple '
                    '(label1, label2, atom_idx1, atom_idx2, V_init) '
                    'or an 8-tuple including the (n_a, n_b, n_c) image.'
                )
            i1, i2 = int(i1), int(i2)
            R = image[0] * lattice[0] + image[1] * lattice[1] + image[2] * lattice[2]
            d_vec = positions[i2 - 1] + R - positions[i1 - 1]
            dist = float(np.linalg.norm(d_vec))

            key = (l1, l2, i1, i2)
            full_key = key + image
            self.vPairs[full_key] = {
                'image': image,
                'R_cart': R,
                'distance': dist,
            }
            self.vVals[key] = float(V_init_default) if v is None else float(v)

    def _register_cutoff_pairs(
        self,
        cutoff,
        species_pairs,
        include_onsite,
        V_init,
    ):
        found = self._enumerate_pairs(cutoff, species_pairs, include_onsite)
        for l1, l2, i1, i2, image, R, dist in found:
            key = (l1, l2, i1, i2)
            full_key = key + image
            self.vPairs[full_key] = {
                'image': image,
                'R_cart': R,
                'distance': dist,
            }
            # Seed only if the user hasn't already set this V via an
            # explicit call or template entry.
            if key not in self.vVals:
                self.vVals[key] = float(V_init)

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #
    def print_intersite_pairs(self):
        """Print a human-readable summary of the registered V pairs."""
        if not self.vPairs:
            print('No intersite V pairs registered.')
            return

        print(f'\n{len(self.vPairs)} intersite V pair(s) registered:')
        print(
            f'  {"label1":>10s} {"label2":>10s} {"i1":>4s} {"i2":>4s} '
            f'{"image":>12s} {"d (A)":>9s} {"V_init (eV)":>12s}'
        )
        for key, meta in sorted(self.vPairs.items(), key=lambda kv: kv[1]['distance']):
            l1, l2, i1, i2, na, nb, nc = key
            v = self.vVals[(l1, l2, i1, i2)]
            print(
                f'  {l1:>10s} {l2:>10s} {i1:>4d} {i2:>4d} '
                f'({na:>2d},{nb:>2d},{nc:>2d}) {meta["distance"]:>9.4f} '
                f'{v:>12.4f}'
            )

    # ------------------------------------------------------------------ #
    # Phase 3: pair density matrices and intersite V evaluation          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _orbital_L(orb_label):
        """Map a Hubbard orbital label (e.g. ``'3d'``, ``'2p'``, ``'5f'``)
        to its angular-momentum quantum number L."""
        sym = orb_label[-1].lower()
        table = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        if sym not in table:
            raise ValueError(f'Unsupported orbital symbol in {orb_label!r}.')
        return table[sym]

    def _parse_state_lines(self, projwfc_out='projwfc.out'):
        """Slice the ``state #`` table out of a ``projwfc.x`` output."""
        with open(projwfc_out, 'r') as f:
            lines = f.readlines()
        sind = 0
        while sind < len(lines) and 'state #' not in lines[sind]:
            sind += 1
        send = sind
        while send < len(lines) and 'state #' in lines[send]:
            send += 1
        return lines[sind:send]

    @staticmethod
    def _site_basis_indices(state_lines, atom_idx, L):
        """Return PAO Hamiltonian indices belonging to atom ``atom_idx``
        (1-based, as written by ``projwfc.x``) with angular momentum L.

        The lines look like::

          state #   1: atom   1 (MnA  ), wfc  1 (l=2 m= 1)
        """
        out = []
        for n, sl in enumerate(state_lines):
            mat = re.search(
                r'atom\s+(\d+)\s*\(\s*\S+\s*\)\s*,\s*wfc\s+\d+\s*\(\s*l\s*=\s*(\d+)', sl
            )
            if mat is None:
                continue
            if int(mat.group(1)) == atom_idx and int(mat.group(2)) == L:
                out.append(n)
        return np.asarray(out, dtype=int)

    def _site_basis_indices_local(self, atom_idx, L, shell_label):
        """Local-basis analogue of :meth:`_site_basis_indices`.

        Returns the PAO Hamiltonian indices on atom ``atom_idx`` (1-based)
        belonging to the valence shell ``shell_label`` (e.g. ``'3D'``).
        Selecting by label (rather than by L alone) excludes polarization
        shells that share the angular momentum L, so the returned size is
        ``2L+1`` and matches the ``_atom_shell_gaussians`` count.
        """
        meta = self._load_pao_basis_meta()
        target = shell_label.upper()
        out = [
            e['index']
            for e in meta
            if e['atom'] == atom_idx and e['l'] == L and e['label'].upper() == target
        ]
        return np.asarray(sorted(out), dtype=int)

    def _pair_density_matrices(
        self,
        basis_I,
        basis_J,
        Hks,
        Sks,
        kpnts,
        kwght,
        R_bohr,
        eigvec_cache=None,
    ):
        """Compute spin-channel pair density matrices following Eqs. (2),
        (4) and (5) of Phys. Rev. Research 2, 043410 (2020).

        Parameters
        ----------
        basis_I, basis_J : (n_I,) / (n_J,) int ndarray
            PAO basis indices on atom I (home cell) and atom J.
        Hks, Sks : (nbasis, nbasis, nkpts) complex ndarray
            k-resolved Hamiltonian (in eV; Fermi level at 0) and overlap.
        kpnts : (nkpts, 3) ndarray
            k-points in 2π/a_lattice units **OR** Cartesian
            (Bohr$^{-1}$): only the phase ``k·R`` is used, so as long as
            ``kpnts`` and ``R_bohr`` share the same convention with the
            saved Bloch eigenvectors the phase is correct.  See the
            ``kpnts_are_cartesian`` argument of :meth:`run_eacbn0_V`.
        kwght : (nkpts,) ndarray
            Brillouin-zone integration weights.
        R_bohr : (3,) ndarray
            Lattice translation applied to atom J, in Bohr.

        Returns
        -------
        result : dict
            Keys ``'P_II'``, ``'P_JJ'``, ``'P_IJ'``, ``'P_JI'`` (the
            renormalized real-space density matrices, in PAO basis;
            ``P_II`` and ``P_JJ`` at R=0, ``P_IJ`` at +R, ``P_JI`` at
            -R) and the bare counterparts ``'n_II'``, ``'n_JJ'``,
            ``'n_IJ'``, ``'n_JI'`` with the same R conventions.  All
            shapes are ``(n_I, n_I)``, ``(n_J, n_J)``, ``(n_I, n_J)``,
            ``(n_J, n_I)`` respectively.
        """
        from scipy.linalg import eigh

        nkpts = Hks.shape[2]
        n_I = basis_I.size
        n_J = basis_J.size

        total_w = float(np.sum(kwght))

        # Block layout: each entry is (key, left coefficients, right coefficients,
        # phase factor). The same four blocks are filled for the bare ``n_*``
        # and the renormalised ``P_*`` matrices, the only difference being
        # whether the right-hand-side carries the band weight ``N_w``.
        block_shapes = {
            'II': (n_I, n_I),
            'JJ': (n_J, n_J),
            'IJ': (n_I, n_J),
            'JI': (n_J, n_I),
        }
        n_blocks = {k: np.zeros(s, dtype=complex) for k, s in block_shapes.items()}
        P_blocks = {k: np.zeros(s, dtype=complex) for k, s in block_shapes.items()}

        for ik in range(nkpts):
            w = kwght[ik]
            phase_pos = np.exp(1j * float(np.dot(kpnts[ik], R_bohr)))
            phase_neg = np.conj(phase_pos)

            if eigvec_cache is not None:
                eig, vec = eigvec_cache[ik]
            else:
                eig, vec = eigh(Hks[:, :, ik], Sks[:, :, ik])
            occ_ind = np.where(eig <= 0.0)[0]
            if occ_ind.size == 0:
                continue
            evec = vec[:, occ_ind]  # (nbasis, nocc)

            # In the non-orthonormal PAO basis the eigenvectors satisfy
            #   H c = ε S c  ⇒  c^† S c = 1
            # so the proper Mulliken-type projection amplitudes that
            # add up to the band normalisation are the pair
            #   c[α, m]     (bra coefficient)
            #   (S c)[α, m] (overlap-corrected ket coefficient).
            # We must keep BOTH:  using only (S c) (as the previous
            # implementation did) yields the S·D·S block, which is
            # orbital-character dependent and produces unphysical
            # manifold weights N_w > 1 in a non-orthonormal basis.
            Sv = Sks[:, :, ik] @ evec  # (nbasis, nocc) = S c
            cI = evec[basis_I, :]  # c[basis_I]
            sI = Sv[basis_I, :]  # (S c)[basis_I]
            cJ = evec[basis_J, :]
            sJ = Sv[basis_J, :]

            # Mulliken band weight on the (I+J) manifold (Eq. 4):
            #   N_w[m] = Σ_{α∈I+J} Re( c*_α (S c)_α ) ∈ [0, 1].
            N_w = (
                np.einsum('im,im->m', np.conj(cI), sI).real
                + np.einsum('jm,jm->m', np.conj(cJ), sJ).real
            )

            # (key, left c, left Sc, right c, right Sc, phase) — used to
            # accumulate both bare ``n_*`` (Eq. 2) and renormalised
            # ``P_*`` (Eq. 5) blocks, Hermitianised by construction.
            blocks = (
                ('II', cI, sI, cI, sI, 1.0 + 0.0j),
                ('JJ', cJ, sJ, cJ, sJ, 1.0 + 0.0j),
                ('IJ', cI, sI, cJ, sJ, phase_pos),
                ('JI', cJ, sJ, cI, sI, phase_neg),
            )
            for key, ca, sa, cb, sb, phase in blocks:
                prefac = 0.5 * w * phase
                n_blocks[key] += prefac * (np.conj(ca) @ sb.T + np.conj(sa) @ cb.T)
                # ``P_*`` matrices weight the right-hand-side coefficients
                # (broadcast over the band index) by ``N_w[m]``.
                sb_w = sb * N_w
                cb_w = cb * N_w
                P_blocks[key] += prefac * (np.conj(ca) @ sb_w.T + np.conj(sa) @ cb_w.T)

        scale = 1.0 / total_w
        result = {f'P_{k}': v * scale for k, v in P_blocks.items()}
        result.update({f'n_{k}': v * scale for k, v in n_blocks.items()})
        return result

    @staticmethod
    def _overlap_is_identity(Sks, atol=1e-6):
        """Return ``True`` if every per-k overlap block equals the identity.

        With the current PAOFLOW code the internal projection path
        (``use_local_basis=True``) already computes and stores the true
        non-orthogonal atomic-orbital overlap S(k) in ``kovp.npy``, so this
        check should normally return ``False``.

        The check is retained as a safety net for stale ``kovp.npy`` files
        that were written by an older PAOFLOW version (which incorrectly
        dumped the identity).  :meth:`run_eacbn0_V` raises
        :exc:`RuntimeError` when it returns ``True`` because intersite V
        requires the genuine off-site S_{IJ}(k) blocks.
        """
        nbasis = Sks.shape[0]
        eye = np.eye(nbasis, dtype=Sks.dtype)
        return bool(np.allclose(Sks, eye[:, :, None], atol=atol))

    def run_eacbn0_V(self, kpnts_are_cartesian=False):
        """Compute intersite Hubbard V for every pair registered in
        :attr:`self.vPairs` and return the updated mapping.

        Assumes that ``run_dft`` + ``run_paoflow`` have already produced
        ``projwfc.out`` and the PAOFLOW Hamiltonian/overlap dumps in
        :attr:`self.outputdir`.

        Parameters
        ----------
        kpnts_are_cartesian : bool, default False
            By default PAOFLOW writes ``k.txt`` in crystal coordinates
            (units of the reciprocal lattice vectors).  Set this to
            ``True`` if the k-points are already in Cartesian
            Bohr$^{-1}$ (in which case ``R_bohr`` is used directly).

        Returns
        -------
        new_V : dict
            ``{(label1, label2, atom_idx1, atom_idx2): V_eV}`` —
            one entry per *image-collapsed* pair.  When several images
            of the same pair are registered the values reported here are
            those of the closest (minimum-image) image.
        """
        # Geometry from QE output of the current SCF iteration (in Bohr).

        # ``read_cell_atoms`` returns lattice + positions in Bohr.
        lattice_B, coords_B = self.read_cell_atoms('scf.out')
        lattice_A = lattice_B * BOHR_RADIUS_ANGS
        coords_A = coords_B * BOHR_RADIUS_ANGS
        recip = 2 * np.pi * np.linalg.inv(lattice_B).T  # Bohr^-1

        # Refresh image vectors of the registered pairs from this geometry
        # (they were originally computed from the input-template lattice
        # in :meth:`set_intersite_pairs`; small rounding differences are
        # possible).
        for full_key, meta in self.vPairs.items():
            na, nb, nc = full_key[4], full_key[5], full_key[6]
            R_A = na * lattice_A[0] + nb * lattice_A[1] + nc * lattice_A[2]
            i1, i2 = full_key[2], full_key[3]
            d_vec = coords_A[i2 - 1] + R_A - coords_A[i1 - 1]
            meta['R_cart'] = R_A
            meta['distance'] = float(np.linalg.norm(d_vec))

        kpnts, kwght, Sks, Hks_up, Hks_dn = self.read_ham_data(self.nspin)
        if self.nspin == 1:
            Hks_dn = Hks_up

        # Intersite V requires the genuine non-orthogonal atomic overlap S(k):
        # the off-site S_IJ(k) blocks carry the bond charge that defines
        # n^{IJ}(R) and the denominator of Eq. (8).  With the current PAOFLOW
        # code both the internal-basis path (use_local_basis=True, any
        # configuration) and the projwfc.x path store the true S(k) in
        # kovp.npy, so the check below is a safety net for stale output
        # written by older PAOFLOW versions that erroneously dumped S(k) = I.
        if self._overlap_is_identity(Sks):
            raise RuntimeError(
                'Intersite Hubbard V requires the true non-orthogonal atomic '
                'overlap S(k), but kovp.npy contains the identity matrix. '
                'This indicates a stale kovp.npy written by an older PAOFLOW '
                'version that incorrectly set S(k) = I for the internal '
                'projection path. Delete the output directory and re-run the '
                'ACBN0 driver so that projections() recomputes the genuine '
                'S(k) and writes it to kovp.npy.'
            )

        # Convert k-points to Cartesian Bohr^-1 if needed.
        if kpnts_are_cartesian:
            k_cart = kpnts
        else:
            k_cart = kpnts @ recip  # (nk, 3)

        state_lines = None if self.use_local_basis else self._parse_state_lines('projwfc.out')

        # Cache the per-k generalized eigendecomposition once per spin
        # channel; reused across every (I, J) pair below so we don't
        # re-diagonalize H(k), S(k) for each Hubbard pair.
        eigvec_up = self._eigh_all_k(Hks_up, Sks)
        eigvec_dn = self._eigh_all_k(Hks_dn, Sks) if self.nspin == 2 else eigvec_up

        # Collapse images to their minimum-image representative per
        # (l1, l2, i1, i2) (these are the keys that QE actually carries).
        best_image = {}
        for full_key, meta in self.vPairs.items():
            l1, l2, i1, i2, na, nb, nc = full_key
            key = (l1, l2, i1, i2)
            prev = best_image.get(key)
            if prev is None or meta['distance'] < prev[1]['distance']:
                best_image[key] = (full_key, meta)

        new_V = {}
        for key, (full_key, meta) in best_image.items():
            l1, l2, i1, i2 = key
            ele1, orb1 = l1.split('-')
            ele2, orb2 = l2.split('-')
            L1 = self._orbital_L(orb1)
            L2 = self._orbital_L(orb2)

            if self.use_local_basis:
                basis_I = self._site_basis_indices_local(i1, L1, orb1.upper())
                basis_J = self._site_basis_indices_local(i2, L2, orb2.upper())
            else:
                basis_I = self._site_basis_indices(state_lines, i1, L1)
                basis_J = self._site_basis_indices(state_lines, i2, L2)
            if basis_I.size == 0 or basis_J.size == 0:
                raise RuntimeError(
                    f'No PAO states found for {key!r}: '
                    f'basis_I.size={basis_I.size}, basis_J.size={basis_J.size}'
                )

            R_bohr = meta['R_cart'] * ANGS_TO_BOHR

            # --- σ = up (and σ = dn when spin-polarized) -----------------
            up = self._pair_density_matrices(
                basis_I,
                basis_J,
                Hks_up,
                Sks,
                k_cart,
                kwght,
                R_bohr,
                eigvec_cache=eigvec_up,
            )
            if self.nspin == 2:
                dn = self._pair_density_matrices(
                    basis_I,
                    basis_J,
                    Hks_dn,
                    Sks,
                    k_cart,
                    kwght,
                    R_bohr,
                    eigvec_cache=eigvec_dn,
                )
            else:
                # Spin-restricted: split the doubly-occupied DM in half
                # so that the on-site I=J case reproduces the existing
                # ACBN0 result (which folds in the factor of 2 through
                # the (a_b + ab_ba) spin combination).
                dn = {k: v.copy() for k, v in up.items()}

            # --- Denominator (Eq. 8 denom of Lee-Son PRR 2020) ----------
            # The denominator uses the *bare* occupation matrices n
            # (Eq. 2), NOT the renormalized P (Eq. 5).  This asymmetry
            # is the whole point of the ACBN0-like construction: the
            # numerator carries the manifold-weight factor N_w (through
            # P), while the denominator does not.  The N_w-induced
            # difference between num and den is what yields the proper
            # orbital-character-dependent screening of V_IJ.
            den = 0.0
            for nII in (up['n_II'], dn['n_II']):
                for nJJ in (up['n_JJ'], dn['n_JJ']):
                    den += float((np.diag(nII).real[:, None] * np.diag(nJJ).real[None, :]).sum())
            for nIJ, nJI in [(up['n_IJ'], up['n_JI']), (dn['n_IJ'], dn['n_JI'])]:
                # Σ_{ij} n^{IJ}_{ij} n^{JI}_{ji}  (note the transpose)
                den -= float((nIJ * nJI.T).real.sum())

            # --- Numerator (Eq. 8 num): launch the MPI kernel -----------
            lbl1 = orb1.upper() if self.use_local_basis else None
            lbl2 = orb2.upper() if self.use_local_basis else None
            gauss_I = self._atom_shell_gaussians(ele1, coords_A[i1 - 1], L1, lbl1)
            gauss_J = self._atom_shell_gaussians(
                ele2,
                coords_A[i2 - 1] + meta['R_cart'],
                L2,
                lbl2,
            )
            if len(gauss_I) != basis_I.size or len(gauss_J) != basis_J.size:
                raise RuntimeError(
                    'Mismatch between PAO basis size and Gaussian shell '
                    f'size for pair {key!r}: '
                    f'PAO {basis_I.size}/{basis_J.size} vs Gauss '
                    f'{len(gauss_I)}/{len(gauss_J)}'
                )

            data = {
                'gauss_I': gauss_I,
                'gauss_J': gauss_J,
                'P_II_up': up['P_II'],
                'P_II_dn': dn['P_II'],
                'P_JJ_up': up['P_JJ'],
                'P_JJ_dn': dn['P_JJ'],
                'P_IJ_up': up['P_IJ'],
                'P_IJ_dn': dn['P_IJ'],
                'P_JI_up': up['P_JI'],
                'P_JI_dn': dn['P_JI'],
            }
            datapath = join(self.outputdir, 'data_v.pkl')
            with open(datapath, 'wb') as f:
                pickle.dump(data, f)

            self._write_compute_hartree_v(datapath)
            self._launch_compute_hartree_v()

            with open(join(self.outputdir, 'tmp_v.pkl'), 'rb') as f:
                num = pickle.load(f)['num']

            # The 1/2 prefactor in Eq. (8) accounts for double-counting
            # of the (I,J) and (J,I) entries when summing over ordered
            # pairs in the energy expression.
            V_IJ = 0.5 * HARTREE_TO_EV * float(num.real) / den if den != 0 else 0.0
            new_V[key] = V_IJ

            with open(f'{l1}_{l2}_{i1}_{i2}_V.txt', 'w') as f:
                f.write(f'pair          : {key}\n')
                f.write(f'image         : ({full_key[4]},{full_key[5]},{full_key[6]})\n')
                f.write(f'distance (A)  : {meta["distance"]:.6f}\n')
                f.write(f'V (eV)        : {V_IJ:.6f}\n')

        return new_V

    def _write_compute_hartree_v(self, datapath):
        with open('compute_hartree_v.py', 'w') as f:
            f.write('from PAOFLOW.ACBN0 import eACBN0_Hartree\n')
            f.write(f"H = eACBN0_Hartree('{datapath}')\n")
            f.write(f"H.intersite_energy('{self.outputdir}')\n")

    def _launch_compute_hartree_v(self):
        python_exec = self._python_exec()
        mpi = getattr(self, 'mpi_hartree', None) or self.mpi_python
        command = f'{mpi} {python_exec} compute_hartree_v.py'
        subprocess.run([tok for tok in command.split(' ') if tok], check=True)

    # ------------------------------------------------------------------ #
    # Phase 4: joint U+V self-consistent loop                             #
    # ------------------------------------------------------------------ #
    def optimize_hubbard_UV(
        self,
        convergence_threshold=0.01,
        max_iter=50,
        mixing=1.0,
        kpnts_are_cartesian=False,
    ):
        """Joint ACBN0 + eACBN0 self-consistent loop.

        Each outer iteration runs ``pw.x`` (scf+nscf), ``projwfc.x``,
        PAOFLOW and then *both* :meth:`run_acbn0` and
        :meth:`run_eacbn0_V`.  Convergence is declared when **all** U
        and V parameters change by less than ``convergence_threshold``
        between consecutive iterations.

        Parameters
        ----------
        convergence_threshold : float, default 0.01
            Maximum allowed absolute change (in eV) of any U or V value
            between iterations.
        max_iter : int, default 50
            Hard cap on the number of outer iterations.
        mixing : float, default 1.0
            Linear-mixing factor applied to **both** U and V: 1.0 fully
            replaces the previous parameters with the freshly-computed
            ones; smaller values dampen oscillations (e.g.
            ``mixing=0.5`` uses ``new = 0.5*old + 0.5*computed``).
        kpnts_are_cartesian : bool, default False
            Forwarded to :meth:`run_eacbn0_V`.

        Raises
        ------
        RuntimeError
            If convergence is not achieved within ``max_iter`` iterations.
        """
        if not self.vPairs:
            raise RuntimeError(
                'No intersite V pairs registered. Call '
                'set_intersite_pairs(...) before optimize_hubbard_UV.'
            )

        print('\nBeginning joint U+V self-consistent loop.\n')
        for itr in range(1, max_iter + 1):
            print(f'Iteration #{itr}\n')

            self.run_dft(self.prefix, self.uspecies, self.uVals)

            save_prefix = self.blocks['control']['prefix'].strip('"').strip("'")
            self.run_paoflow(self.prefix, save_prefix)

            new_U = self.run_acbn0(self.prefix)
            new_V = self.run_eacbn0_V(kpnts_are_cartesian=kpnts_are_cartesian)

            # ---- Convergence check + mixing -------------------------
            converged = True
            print('\nNew U values:')
            for k, v in new_U.items():
                old = self.uVals[k]
                mixed = mixing * v + (1.0 - mixing) * old
                print(f'  {k} : old={old:.4f}  new={v:.4f}  mixed={mixed:.4f}')
                if abs(mixed - old) > convergence_threshold:
                    converged = False
                new_U[k] = mixed

            print('\nNew V values:')
            for k, v in new_V.items():
                old = self.vVals.get(k, 0.0)
                mixed = mixing * v + (1.0 - mixing) * old
                l1, l2, i1, i2 = k
                print(f'  {l1} {l2} {i1} {i2} : old={old:.4f}  new={v:.4f}  mixed={mixed:.4f}')
                if abs(mixed - old) > convergence_threshold:
                    converged = False
                new_V[k] = mixed
            print(flush=True)

            self.uVals = new_U
            for k, v in new_V.items():
                self.vVals[k] = v

            if converged:
                print(f'Converged after {itr} iteration(s).')
                return

        raise RuntimeError(f'Joint U+V loop did not converge in {max_iter} iterations.')
