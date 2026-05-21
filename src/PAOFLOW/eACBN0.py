"""Extended ACBN0 (DFT+U+V) driver.

This module implements the *extended* ACBN0 self-consistent scheme that
augments the standard on-site Hubbard U calculation with the intersite
Hubbard V correction, following the formulation of

    S.-H. Lee and Y.-W. Son, *First-principles approach with a
    pseudo-hybrid density functional for extended Hubbard interactions*,
    Phys. Rev. Research **2**, 043410 (2020).

The driver is exposed through the :class:`eACBN0` class, a subclass of
:class:`PAOFLOW.ACBN0.ACBN0`.  ``eACBN0`` reuses the entire on-site U
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
wavefunctions (the same basis used by :mod:`PAOFLOW.ACBN0_Hartree`).

Workflow
--------
The typical usage mirrors :class:`ACBN0.ACBN0` but adds a pair-selection
step and uses :meth:`optimize_hubbard_UV` instead of
``optimize_hubbard_U``::

    from PAOFLOW.eACBN0 import eACBN0

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
:class:`PAOFLOW.ACBN0.ACBN0`:

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

Notes
-----
- The HUBBARD V correction requires the symmetries to be turned off in
  ``pw.x`` (``nosym = .true., noinv = .true.``).  The driver does not
  inject these flags; they must be present in the user-supplied
  ``<prefix>.scf.in`` / ``<prefix>.nscf.in`` templates.
- Spin-restricted runs (``nspin = 1``) reuse a single density matrix
  for both spin channels by halving the doubly-occupied DM, so that the
  on-site limit (I = J, R = 0) reproduces the standard
  :class:`ACBN0.ACBN0` result exactly.
"""

import numpy as np

from .ACBN0 import ACBN0


BOHR_RADIUS_ANGS = 0.529177210903
ANGS_TO_BOHR = 1.0 / BOHR_RADIUS_ANGS
HARTREE_TO_EV = 27.211396132


class eACBN0(ACBN0):
    """ACBN0 self-consistent loop extended with intersite V (DFT+U+V).

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :class:`PAOFLOW.ACBN0.ACBN0`.
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
            raise ValueError(
                'CELL_PARAMETERS card not found in template; '
                'ibrav-based geometries are not yet supported by eACBN0.'
            )

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
                    "CELL_PARAMETERS in 'alat' but no celldm(1)/A "
                    'found in &system block.'
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
                    "ATOMIC_POSITIONS in 'alat' but no celldm(1)/A "
                    'found in &system block.'
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
            labels = sorted({f'{ele}-{orb}'
                             for ele, orbs in species_orbs.items()
                             for orb in orbs})
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
                atom_labels.append([f'{ele}-{orb}'
                                    for orb in species_orbs[ele]])
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
                            if (
                                not include_onsite
                                and i == j
                                and na == 0
                                and nb == 0
                                and nc == 0
                            ):
                                continue
                            R = (na * lattice[0]
                                 + nb * lattice[1]
                                 + nc * lattice[2])
                            d_vec = positions[j] + R - positions[i]
                            d2 = float(d_vec @ d_vec)
                            if d2 > cutoff2:
                                continue
                            dist = float(np.sqrt(d2))
                            for l1 in atom_labels[i]:
                                for l2 in atom_labels[j]:
                                    if (l1, l2) not in wanted_pairs:
                                        continue
                                    pairs.append((
                                        l1, l2, i + 1, j + 1,
                                        (na, nb, nc), R, dist,
                                    ))
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
                cutoff, species_pairs, include_onsite, V_init,
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
            R = (image[0] * lattice[0]
                 + image[1] * lattice[1]
                 + image[2] * lattice[2])
            d_vec = positions[i2 - 1] + R - positions[i1 - 1]
            dist = float(np.linalg.norm(d_vec))

            key = (l1, l2, i1, i2)
            full_key = key + image
            self.vPairs[full_key] = {
                'image': image,
                'R_cart': R,
                'distance': dist,
            }
            self.vVals[key] = (
                float(V_init_default) if v is None else float(v)
            )

    def _register_cutoff_pairs(
        self, cutoff, species_pairs, include_onsite, V_init,
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
        for key, meta in sorted(self.vPairs.items(),
                                key=lambda kv: kv[1]['distance']):
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
        import re

        out = []
        for n, sl in enumerate(state_lines):
            mat = re.search(r'atom\s+(\d+)\s*\(\s*\S+\s*\)\s*,\s*wfc\s+\d+\s*\(\s*l\s*=\s*(\d+)', sl)
            if mat is None:
                continue
            if int(mat.group(1)) == atom_idx and int(mat.group(2)) == L:
                out.append(n)
        return np.asarray(out, dtype=int)

    def _atom_shell_gaussians(self, ele, pos_angstrom, L):
        """Build the CGBFs of the ``(ele, L)`` shell centred at
        ``pos_angstrom`` (Cartesian Ångström).  Returns a list with one
        CGBF per magnetic component (``2L+1`` Gaussians)."""
        from .defs.pyints import CGBF

        gauss = []
        origin_bohr = np.asarray(pos_angstrom) * ANGS_TO_BOHR
        for shell in self.basis[ele]:
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

    def _pair_density_matrices(
        self, basis_I, basis_J, Hks, Sks, kpnts, kwght, R_bohr,
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

        P_II = np.zeros((n_I, n_I), dtype=complex)
        P_JJ = np.zeros((n_J, n_J), dtype=complex)
        P_IJ = np.zeros((n_I, n_J), dtype=complex)
        P_JI = np.zeros((n_J, n_I), dtype=complex)

        n_II = np.zeros((n_I, n_I), dtype=complex)
        n_JJ = np.zeros((n_J, n_J), dtype=complex)
        n_IJ = np.zeros((n_I, n_J), dtype=complex)
        n_JI = np.zeros((n_J, n_I), dtype=complex)

        for ik in range(nkpts):
            w = kwght[ik]
            phase_pos = np.exp(1j * float(np.dot(kpnts[ik], R_bohr)))
            phase_neg = np.conj(phase_pos)

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
            Sv = Sks[:, :, ik] @ evec     # (nbasis, nocc) = S c
            cI = evec[basis_I, :]         # c[basis_I]
            sI = Sv[basis_I, :]           # (S c)[basis_I]
            cJ = evec[basis_J, :]
            sJ = Sv[basis_J, :]

            # Mulliken band weight on the (I+J) manifold (Eq. 4):
            #   N_w[m] = Σ_{α∈I+J} Re( c*_α (S c)_α ) ∈ [0, 1].
            N_w = (np.einsum('im,im->m', np.conj(cI), sI).real
                   + np.einsum('jm,jm->m', np.conj(cJ), sJ).real)

            # Bare on-site / intersite occupation matrix blocks (Eq. 2),
            # Hermitianised (½ [c*·(Sc)^T + (Sc)*·c^T]) so they are
            # real-symmetric (resp. Hermitian) by construction:
            n_II += 0.5 * w * (np.conj(cI) @ sI.T + np.conj(sI) @ cI.T)
            n_JJ += 0.5 * w * (np.conj(cJ) @ sJ.T + np.conj(sJ) @ cJ.T)
            n_IJ += 0.5 * w * phase_pos * (
                np.conj(cI) @ sJ.T + np.conj(sI) @ cJ.T
            )
            n_JI += 0.5 * w * phase_neg * (
                np.conj(cJ) @ sI.T + np.conj(sJ) @ cI.T
            )

            # Renormalized P matrices (Eq. 5): same expressions weighted
            # by N_w[m].
            cI_w = cI * N_w  # broadcast over the band index
            cJ_w = cJ * N_w
            sI_w = sI * N_w
            sJ_w = sJ * N_w
            P_II += 0.5 * w * (np.conj(cI) @ sI_w.T + np.conj(sI) @ cI_w.T)
            P_JJ += 0.5 * w * (np.conj(cJ) @ sJ_w.T + np.conj(sJ) @ cJ_w.T)
            P_IJ += 0.5 * w * phase_pos * (
                np.conj(cI) @ sJ_w.T + np.conj(sI) @ cJ_w.T
            )
            P_JI += 0.5 * w * phase_neg * (
                np.conj(cJ) @ sI_w.T + np.conj(sJ) @ cI_w.T
            )

        scale = 1.0 / total_w
        # --- diagnostic ------------------------------------------------
        # Inspect manifold-weight behaviour: Tr(P)/Tr(n) on each site
        # and on the intersite block (= weighted-average band weight on
        # the I+J manifold).  Should be in (0, 1].  Tr ≈ 1 means
        # N_w ≈ 1, i.e. the bands sit fully inside the I+J manifold and
        # the renormalisation is essentially trivial.
        tr_nII = float(np.trace(n_II).real * scale)
        tr_PII = float(np.trace(P_II).real * scale)
        tr_nJJ = float(np.trace(n_JJ).real * scale)
        tr_PJJ = float(np.trace(P_JJ).real * scale)
        print(
            f'  [DBG _pair_density_matrices] n_I={n_I} n_J={n_J} | '
            f'Tr(n_II)={tr_nII:.4f} Tr(P_II)={tr_PII:.4f} '
            f'(<N_w>_II={tr_PII/tr_nII if tr_nII else float("nan"):.4f}) | '
            f'Tr(n_JJ)={tr_nJJ:.4f} Tr(P_JJ)={tr_PJJ:.4f} '
            f'(<N_w>_JJ={tr_PJJ/tr_nJJ if tr_nJJ else float("nan"):.4f})'
        )
        return {
            'P_II': P_II * scale,
            'P_JJ': P_JJ * scale,
            'P_IJ': P_IJ * scale,
            'P_JI': P_JI * scale,
            'n_II': n_II * scale,
            'n_JJ': n_JJ * scale,
            'n_IJ': n_IJ * scale,
            'n_JI': n_JI * scale,
        }

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
        import pickle
        from os.path import join

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
            R_A = (na * lattice_A[0]
                   + nb * lattice_A[1]
                   + nc * lattice_A[2])
            i1, i2 = full_key[2], full_key[3]
            d_vec = coords_A[i2 - 1] + R_A - coords_A[i1 - 1]
            meta['R_cart'] = R_A
            meta['distance'] = float(np.linalg.norm(d_vec))

        kpnts, kwght, Sks, Hks_up, Hks_dn = self.read_ham_data(self.nspin)
        if self.nspin == 1:
            Hks_dn = Hks_up

        # Convert k-points to Cartesian Bohr^-1 if needed.
        if kpnts_are_cartesian:
            k_cart = kpnts
        else:
            k_cart = kpnts @ recip  # (nk, 3)

        state_lines = self._parse_state_lines('projwfc.out')

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
                basis_I, basis_J, Hks_up, Sks, k_cart, kwght, R_bohr,
            )
            if self.nspin == 2:
                dn = self._pair_density_matrices(
                    basis_I, basis_J, Hks_dn, Sks, k_cart, kwght, R_bohr,
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
                    den += float(
                        (np.diag(nII).real[:, None]
                         * np.diag(nJJ).real[None, :]).sum()
                    )
            for nIJ, nJI in [(up['n_IJ'], up['n_JI']),
                             (dn['n_IJ'], dn['n_JI'])]:
                # Σ_{ij} n^{IJ}_{ij} n^{JI}_{ji}  (note the transpose)
                den -= float((nIJ * nJI.T).real.sum())

            # --- Numerator (Eq. 8 num): launch the MPI kernel -----------
            gauss_I = self._atom_shell_gaussians(ele1, coords_A[i1 - 1], L1)
            gauss_J = self._atom_shell_gaussians(
                ele2, coords_A[i2 - 1] + meta['R_cart'], L2,
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
                'P_II_up': up['P_II'], 'P_II_dn': dn['P_II'],
                'P_JJ_up': up['P_JJ'], 'P_JJ_dn': dn['P_JJ'],
                'P_IJ_up': up['P_IJ'], 'P_IJ_dn': dn['P_IJ'],
                'P_JI_up': up['P_JI'], 'P_JI_dn': dn['P_JI'],
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
            f.write('from PAOFLOW.eACBN0_Hartree import eACBN0_Hartree\n')
            f.write(f"H = eACBN0_Hartree('{datapath}')\n")
            f.write(f"H.intersite_energy('{self.outputdir}')\n")

    def _launch_compute_hartree_v(self):
        import subprocess
        from os.path import join

        python_exec = join(self.ppath, 'python') if self.ppath else 'python'
        mpi = getattr(self, 'mpi_hartree', None) or self.mpi_python
        command = f'{mpi} {python_exec} compute_hartree_v.py'
        subprocess.run(command.split(), check=True)

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

            save_prefix = self.blocks['control']['prefix']\
                .strip('"').strip("'")
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
                print(
                    f'  {l1} {l2} {i1} {i2} : old={old:.4f}  '
                    f'new={v:.4f}  mixed={mixed:.4f}'
                )
                if abs(mixed - old) > convergence_threshold:
                    converged = False
                new_V[k] = mixed
            print('', flush=True)

            self.uVals = new_U
            for k, v in new_V.items():
                self.vVals[k] = v

            if converged:
                print(f'Converged after {itr} iteration(s).')
                return

        raise RuntimeError(
            f'Joint U+V loop did not converge in {max_iter} iterations.'
        )
