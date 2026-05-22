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
   Gaussian basis (:mod:`PAOFLOW.defs.upf_gaussfit`) so the Coulomb
   integrals required by the ACBN0 formula can be evaluated
   analytically.
3. On every outer iteration, inject the current ``HUBBARD`` card into
   the SCF/NSCF templates and run ``pw.x`` (scf), ``pw.x`` (nscf),
   ``projwfc.x``, PAOFLOW and finally :mod:`PAOFLOW.ACBN0_Hartree`
   (under MPI).
4. Recompute U for every Hubbard-active orbital from the ACBN0
   numerator (renormalized two-electron integrals) and denominator
   (occupation-matrix invariants).
5. Iterate until ``max |Δ U| < convergence_threshold``.

The class is also the base of :class:`PAOFLOW.eACBN0.eACBN0`, which
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
- The Hartree integrals (:mod:`PAOFLOW.ACBN0_Hartree`) are launched
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
  basis from :func:`PAOFLOW.defs.upf_gaussfit.gaussian_fit`.
- :attr:`blocks` / :attr:`cards` — parsed namelists / cards of the
  SCF input template, as returned by
  :func:`PAOFLOW.defs.file_io.struct_from_inputfile_QE`.
- :attr:`nspin` — read from the ``&system`` block (defaults to 1).
- :attr:`hubbard_tag` — header line of the ``HUBBARD`` card
  (e.g. ``'HUBBARD (atomic)'``).
- :attr:`hubbard_occ` — fixed ``hubbard_occ(i,j)`` entries to be
  written back into the ``&system`` namelist.

Notes
-----
- The driver does not modify or add symmetry flags.  When using the
  output ``HUBBARD`` card with intersite V (via :class:`eACBN0`) the
  user must supply ``nosym = .true., noinv = .true.`` in the SCF/NSCF
  templates — QE crashes otherwise.
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
  :func:`PAOFLOW.defs.do_build_pao_hamiltonian.do_build_pao_hamiltonian`
  is therefore skipped for ACBN0; both IBZ and full-BZ k-grids are
  accepted.
- :func:`PAOFLOW.defs.do_build_pao_hamiltonian.build_Hks` now applies a
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

import pickle
import subprocess
from os.path import join

import numpy as np


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
    ):
        """Initialize the ACBN0 self-consistent U driver.

        Parameters
        ----------
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
        """
        from os import chdir

        from .defs.file_io import struct_from_inputfile_QE
        from .defs.header import header
        from .defs.upf_gaussfit import gaussian_fit

        header()
        print('\nPerforming ACBN0 self-consistent determination of Hubbard U corrections.\n')

        datafilepath = join(outputdir, 'data.pkl')
        with open('compute_hartree.py', 'w') as f:
            f.write('from PAOFLOW import ACBN0_Hartree\n')
            f.write(f"H = ACBN0_Hartree.ACBN0_Hartree('{datafilepath}')\n")
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

        # Generate gaussian fits
        print('Generating gaussian fits for pseudopotential basis states.\n')
        self.basis = {}
        self.uspecies = []
        for s in self.cards['ATOMIC_SPECIES'][1:]:
            ele, _, pp = s.split()
            self.uspecies.append(ele)
            atno, basis = gaussian_fit(pp, threshold=0.01)
            self.basis[ele] = basis

        # Store U values from input template
        if 'HUBBARD' in self.cards:
            import re

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
            print('', flush=True)

            self.uVals = new_U

    def exec_QE(self, executable, fname):
        exe = join(self.qpath, executable)
        fout = fname.replace('in', 'out')

        command = f'{self.mpi_qe} {exe} {self.qoption}'
        print(
            f'Starting Process: {self.mpi_qe} {exe} {self.qoption} < {fname} > {fout}', flush=True
        )
        with open(fname, 'r') as qe_in, open(fout, 'w') as qe_out:
            subprocess.run(
                command.split(' '), stdin=qe_in, stdout=qe_out, stderr=subprocess.STDOUT, check=True
            )

    def exec_PAOFLOW(self):
        python_exec = join(self.ppath, 'python')
        command = f'{self.mpi_python} {python_exec} acbn0.py'
        print(f'Starting Process: {command} > paoflow.out', flush=True)
        with open('paoflow.out', 'w') as paoflow_out:
            subprocess.run(
                command.split(' '),
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
            card.append(' U {} {}'.format(k, v))

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
            card.append(
                ' V {} {} {} {} {}'.format(
                    sym1,
                    sym2,
                    idx1,
                    idx2,
                    v_emit,
                )
            )

        return card

    def run_dft(self, prefix, species, uVals):
        from .defs.file_io import create_atomic_inputfile, struct_from_inputfile_QE

        blocks, cards = struct_from_inputfile_QE(f'{prefix}.scf.in')
        cards['HUBBARD'] = self.hubbard_card()
        for k, v in self.hubbard_occ.items():
            blocks['system'][k] = v
        create_atomic_inputfile('scf', blocks, cards)

        blocks, cards = struct_from_inputfile_QE(f'{prefix}.nscf.in')
        cards['HUBBARD'] = self.hubbard_card()
        for k, v in self.hubbard_occ.items():
            blocks['system'][k] = v
        create_atomic_inputfile('nscf', blocks, cards)

        blocks, cards = struct_from_inputfile_QE(f'{prefix}.projwfc.in')
        create_atomic_inputfile('projwfc', blocks, cards)

        executables = {'scf': 'pw.x', 'nscf': 'pw.x', 'projwfc': 'projwfc.x -nd 1'}
        for c in ['scf', 'nscf', 'projwfc']:
            self.exec_QE(executables[c], f'{c}.in')

    def run_paoflow(self, prefix, save_prefix):
        from .defs.file_io import create_acbn0_inputfile

        fstr = f'{prefix}_PAO_bands' + '{}.in'
        calcs = []
        if self.nspin == 1:
            calcs.append(fstr.format(''))
        else:
            calcs.append(fstr.format('_up'))
            calcs.append(fstr.format('_down'))

        create_acbn0_inputfile(save_prefix, self.pthr, self.outputdir)
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

    def run_acbn0(self, prefix):
        import re

        BOHR_RADIUS_ANGS = 0.529177e0

        lattice, coords = self.read_cell_atoms('scf.out')
        lattice *= BOHR_RADIUS_ANGS
        coords *= BOHR_RADIUS_ANGS
        nspin = self.nspin

        sind = 0
        state_lines = open('projwfc.out', 'r').readlines()
        while 'state #' not in state_lines[sind]:
            sind += 1
        send = sind
        while 'state #' in state_lines[send]:
            send += 1
        state_lines = state_lines[sind:send]

        kpnts, kwght, Sks, Hks_up, Hks_dw = self.read_ham_data(nspin)
        uVals = {}
        species = []
        # for s in list(set(species)):
        for k, v in self.uVals.items():
            species_label = k.split('-')[0]
            species.append(species_label)

        gauss_basis = self.getbasis(self.basis, species, lattice, coords)

        for orb, v in self.uVals.items():
            ostates = []
            ustates = []
            species_label = orb.split('-')[0]
            horb = self.hubbard_orbital(orb)
            for n, sl in enumerate(state_lines):
                stateN = re.findall(r'\(([^\)]+)\)', sl)
                oele = stateN[0].strip()
                oL = int(re.split('=| ', stateN[1])[1])
                if species_label in oele and oL == horb:
                    ostates.append(n)
                    if species_label == oele:
                        ustates.append(n)
            sstates = [ustates[0]]
            for i, us in enumerate(ustates[1:]):
                if us == 1 + sstates[i]:
                    sstates.append(us)
                else:
                    break

            basis_dm = np.array(ostates)
            basis_2e = np.array(sstates)

            dk, nlm = self.Dk(basis_dm, basis_2e, Hks_up, Sks)
            nlm = self.Nmm(nlm, Hks_up, kwght)
            nnlm = nlm.shape[0]

            dk_dn = None
            den_U, den_J = 0, 0
            if nspin == 1:
                Naa, Nab = 0.0, 0.0
                for i1, m1 in enumerate(nlm):
                    for i2, m2 in enumerate(nlm):
                        nlm12 = m1 * m2
                        Nab += nlm12
                        if i1 != i2:
                            Naa += nlm12
                den_U = 2 * (Naa.real + Nab.real)
                den_J = 2 * Naa.real

            else:
                dk_dn, nlmd = self.Dk(basis_dm, basis_2e, Hks_dw, Sks)
                nlmd = self.Nmm(nlmd, Hks_dw, kwght)

                Naa, Nbb, Nab = 0.0, 0.0, 0.0
                for i1 in range(nnlm):
                    for i2 in range(nnlm):
                        Nab += nlm[i1] * nlmd[i2]
                        if i1 != i2:
                            Naa += nlm[i1] * nlm[i2]
                            Nbb += nlmd[i1] * nlmd[i2]
                den_U = Naa.real + Nbb.real + 2 * Nab.real
                den_J = Naa.real + Nbb.real

            DR_up = self.DR(dk, kwght)

            DR_dn = DR_up
            if nspin == 2:
                DR_dn = self.DR(dk_dn, kwght)

            data = {'DR_up': DR_up, 'DR_dn': DR_dn, 'basis': gauss_basis, 'basis_2e': basis_2e}
            with open(join(self.outputdir, 'data.pkl'), 'wb') as f:
                pickle.dump(data, f)

            # compute hartree energy in parallel
            python_exec = join(self.ppath, 'python')
            command = f'{self.mpi_hartree} {python_exec} compute_hartree.py'
            subprocess.run(command.split(' '), check=True)

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
        from .defs.pyints import CGBF

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

    def Dk(self, basis_dm, basis_2e, Hks, Sks):
        from scipy.linalg import eigh

        nbasis, _, nkpnts = Hks.shape
        size_dm, size_2e = basis_dm.shape[0], basis_2e.shape[0]
        D_k = np.zeros((nbasis, nbasis, nkpnts), dtype=complex)
        nlm_k = np.zeros((size_2e, nbasis, nkpnts), dtype=complex)

        # Find the density matrix for each k
        for ik in range(nkpnts):
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
        lm_size, nbasis, nkp = nlm.shape
        nlm_aux = np.zeros((lm_size, nbasis), dtype=complex)
        for ik, wght in enumerate(kwght):
            nlm_aux += wght * nlm[:, :, ik]

        return np.sum(nlm_aux / np.sum(kwght), axis=1)

    def DR(self, Dk, kwght):
        nawf = Dk.shape[0]

        D = np.zeros((nawf, nawf), dtype=complex)
        for ik, wght in enumerate(kwght):
            D += wght * Dk[:, :, ik]

        return D.real / np.sum(kwght)

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
