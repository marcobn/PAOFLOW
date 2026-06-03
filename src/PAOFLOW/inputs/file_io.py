def struct_from_outputfile_QE(fname: str):
    """Parse structural information from a Quantum ESPRESSO output file.

    Parameters
    ----------
    fname : str
        Path to the QE ``.out`` file.

    Returns
    -------
    dict
        Structure dictionary with keys:

        - ``'lattice'`` : np.ndarray, shape ``(3, 3)`` — lattice vectors
          in Bohr.
        - ``'abc'`` : np.ndarray, shape ``(nat, 3)`` — fractional atomic
          coordinates.
        - ``'species'`` : list of str — atomic species labels.
        - ``'lunit'`` : str — length unit (``'bohr'``).
        - ``'aunit'`` : str — atomic position unit (``'alat'``).

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not exist.
    """
    import os
    from os.path import isfile, join

    import numpy as np

    if not isfile(fname):
        msg = 'File {} does not exist.'.format(join(os.getcwd(), fname))
        raise FileNotFoundError(msg)

    struct = {'lunit': 'bohr', 'aunit': 'alat'}
    with open(fname, 'r') as f:
        lines = f.readlines()

        eL = 0
        try:
            struct['species'] = []
            celldm = np.empty(6, dtype=float)
            while 'bravais-lattice' not in lines[eL]:
                eL += 1
            ibrav = int(lines[eL].split()[3])

            while 'celldm' not in lines[eL]:
                eL += 1
            celldm[:3] = [float(v) for i, v in enumerate(lines[eL].split()) if i % 2 == 1]
            celldm[3:] = [float(v) for i, v in enumerate(lines[eL + 1].split()) if i % 2 == 1]

            if ibrav != 0:
                from .lattice_format import lattice_format_QE

                struct['lattice'] = lattice_format_QE(ibrav, celldm)
            else:
                while 'crystal axes' not in lines[eL]:
                    eL += 1
                coord = []
                for l in lines[eL + 1 : eL + 4]:
                    coord.append([celldm[0] * float(v) for v in l.split()[3:6]])
                struct['lattice'] = np.array(coord)

            while 'site n.' not in lines[eL]:
                eL += 1
            eL += 1
            apos = []
            while 'End' not in lines[eL] and lines[eL] != '\n':
                line = lines[eL].split()
                struct['species'].append(line[1])
                apos.append([float(v) for v in line[6:9]])
                eL += 1
            apos = celldm[0] * np.array(apos)
            struct['abc'] = apos @ np.linalg.inv(struct['lattice'])

        except Exception as e:
            print('ERROR: Could not read the QE output.')
            raise e

    return struct


def read_relaxed_coordinates_QE(fname: str):
    """Read relaxed atomic positions (and optionally cell parameters) from a QE output file.

    Parameters
    ----------
    fname : str
        Path to the QE ``.out`` (relaxation) file.

    Returns
    -------
    dict
        Updated structure dictionary (from :func:`struct_from_outputfile_QE`)
        with two additional keys:

        - ``'lattice'`` : np.ndarray, shape ``(nsteps+1, 3, 3)`` —
          lattice vectors at each ionic step (index 0 is the initial
          structure).
        - ``'abc'`` : np.ndarray, shape ``(nsteps+1, nat, 3)`` —
          fractional atomic positions at each ionic step.

    Raises
    ------
    Exception
        If no atomic positions or cell coordinates are found in the file.
    """
    import re

    import numpy as np

    abc = []
    cell_params = []
    struct = struct_from_outputfile_QE(fname)

    with open(fname, 'r') as f:
        lines = f.readlines()

        eL = 0
        nL = len(lines)

        try:

            def read_apos(sind):
                apos = []
                while lines[sind] != '\n' and 'End final coordinates' not in lines[sind]:
                    apos.append([float(v) for v in lines[sind].split()[1:4]])
                    sind += 1
                return sind, apos

            while eL < nL:
                while (
                    eL < nL
                    and 'CELL_PARAMETERS' not in lines[eL]
                    and 'ATOMIC_POSITIONS' not in lines[eL]
                ):
                    eL += 1
                if eL >= nL:
                    break
                if 'ATOMIC_POSITIONS' in lines[eL]:
                    unit = lines[eL].split()[1].strip('(){{}}')
                    if len(unit) > 1:
                        struct['aunit'] = unit
                    eL, apos = read_apos(eL + 1)
                    abc.append(apos)
                elif 'CELL_PARAMETERS' in lines[eL]:
                    coord = []
                    unit = lines[eL].split()[1].strip('(){{}}')

                    alat = 1
                    if 'alat' in unit or len(unit) == 0:
                        struct['lunit'] = 'alat'
                        if 'alat' in unit:
                            cpattern = re.search('\(([^\)]+)\)', lines[eL])
                            if cpattern is not None:
                                alat = float(cpattern.group(0)[1:-1].split('=')[1])
                    else:
                        struct['lunit'] = unit
                    for l in lines[eL + 1 : eL + 4]:
                        coord.append(alat * np.array([float(v) for v in l.split()]))
                    cell_params.append(coord)
                    eL += 4

                    while 'ATOMIC_POSITIONS' not in lines[eL]:
                        eL += 1
                    eL, apos = read_apos(eL + 1)
                    abc.append(apos)

        except Exception as e:
            print('WARNING: No atomic positions or cell coordinates were found.', flush=True)
            raise e

    struct['lattice'] = np.array([struct['lattice']] + cell_params)
    struct['abc'] = np.array([struct['abc']] + abc)

    return struct


def struct_from_inputfile_QE(fname: str) -> dict:
    """Parse structural information and Namelist blocks from a Quantum ESPRESSO input file.

    Parameters
    ----------
    fname : str
        Path to the QE input file (e.g. ``.scf.in``).

    Returns
    -------
    blocks : dict
        Nested dictionary mapping Namelist names (lower-case) to
        ``{keyword: value}`` dictionaries.
    cards : dict
        Dictionary mapping card names (e.g. ``'ATOMIC_POSITIONS'``,
        ``'ATOMIC_SPECIES'``, ``'K_POINTS'``, ``'CELL_PARAMETERS'``,
        ``'HUBBARD'``) to lists of raw text lines.

    Notes
    -----
    Currently only control Namelist blocks are parsed; inline comments are
    stripped.  The ``HUBBARD`` card accepts an arbitrary number of entries
    (``U``, ``V``, ``J``, ...), terminated by the next card or end of file.
    """
    import re
    from os.path import isfile

    if not isfile(fname):
        raise FileNotFoundError('File {} does not exist.'.format(fname))

    fstr = None
    with open(fname, 'r') as f:
        fstr = f.read()

    # Process blocks
    cards = {}
    blocks = {}
    natom = ntype = 0
    pattern = re.compile('&(.*?)/@')
    comment = lambda v: v != '' and v[0] != '!'
    matches = pattern.findall(fstr.replace(' ', '').replace('\n', '@ '))
    for match in matches:
        match = match.replace(',@', '@')
        match = [s.replace(' ', '').split('!')[0] for s in re.split(', |@', match) if s != '']

        # Split inline commas without destroying Hubbard_occ tags
        block_args = []
        for m in match:
            hcinds = set([s.end(0) - 1 for s in list(re.finditer('\(([^\)]+),', m))])

            if len(hcinds) == 0:
                for s in m.split(','):
                    if s != '':
                        block_args.append(s)

            else:
                cinds = set([i for i, c in enumerate(m) if c == ','])
                if cinds == hcinds:
                    block_args.append(m)

                else:
                    iprev = 0
                    for i in sorted(cinds.difference(hcinds)):
                        block_args.append(m[iprev:i])
                        iprev = i + 1
                    block_args.append(m[iprev:])

        match = None
        block = block_args.pop(0).lower()

        blocks[block] = {}
        for s in block_args:
            k, v = s.split('=')
            k = k.lower()
            blocks[block][k] = v
            if k == 'ntyp':
                ntype = int(v)
            elif k == 'nat':
                natom = int(v)

    # Process CARDS
    fstr = list(filter(comment, fstr.split('\n')))

    def scan_blank_lines(nl):
        nl += 1
        while fstr[nl] == '':
            nl += 1
        return nl

    il = 0
    nf = len(fstr)
    while il < nf and 'ATOMIC_POSITIONS' not in fstr[il]:
        il += 1
    if il < nf:
        cards['ATOMIC_POSITIONS'] = [fstr[il]]
        il = scan_blank_lines(il)
        for i in range(natom):
            cards['ATOMIC_POSITIONS'].append(fstr[il + i])

    sl = 0
    while sl < nf and 'ATOMIC_SPECIES' not in fstr[sl]:
        sl += 1
    if sl < nf:
        cards['ATOMIC_SPECIES'] = [fstr[sl]]
        sl = scan_blank_lines(sl)
        for i in range(ntype):
            cards['ATOMIC_SPECIES'].append(fstr[sl + i])

    hl = 0
    while hl < nf and 'HUBBARD' not in fstr[hl]:
        hl += 1
    if hl < nf:
        cards['HUBBARD'] = [fstr[hl]]
        hl = scan_blank_lines(hl)
        # Accept an arbitrary number of HUBBARD entries (U, V, J, J0, B, E, ...)
        # until the next card keyword or EOF.
        _card_keywords = {
            'K_POINTS',
            'CELL_PARAMETERS',
            'ATOMIC_POSITIONS',
            'ATOMIC_SPECIES',
            'OCCUPATIONS',
            'CONSTRAINTS',
            'ADDITIONAL_K_POINTS',
            'SOLVENTS',
            'HUBBARD',
        }
        while hl < nf:
            tokens = fstr[hl].split()
            if not tokens or tokens[0] in _card_keywords:
                break
            cards['HUBBARD'].append(fstr[hl])
            hl += 1

    kl = 0
    while kl < nf and 'K_POINTS' not in fstr[kl]:
        kl += 1
    if kl < nf:
        cards['K_POINTS'] = [fstr[kl]]
        if 'gamma' in fstr[kl].lower():
            pass
        else:
            cards['K_POINTS'].append(fstr[kl + 1])
            if 'automatic' not in fstr[kl]:
                nk = int(fstr[kl + 1])
                kl += 2
                for i in range(nk):
                    cards['K_POINTS'].append(fstr[kl + i])

    cl = 0
    while cl < nf and 'CELL_PARAM' not in fstr[cl]:
        cl += 1
    if cl < nf:
        cards['CELL_PARAMETERS'] = []
        for i in range(4):
            cards['CELL_PARAMETERS'].append(fstr[cl + i])

    return blocks, cards


def create_atomic_inputfile(calculation, blocks, cards):
    """Write a Quantum ESPRESSO input file from Namelist blocks and card data.

    Parameters
    ----------
    calculation : str
        Base name of the output file; the file is written to
        ``{calculation}.in``.
    blocks : dict
        Nested dictionary of Namelist blocks as returned by
        :func:`struct_from_inputfile_QE`.
    cards : dict
        Dictionary of card data as returned by :func:`struct_from_inputfile_QE`.
        The ``'ATOMIC_SPECIES'`` card, if present, is written first and then
        removed from the dict before writing the remaining cards.

    Returns
    -------
    None
        Writes a file ``{calculation}.in`` to the current directory.
    """
    with open(f'{calculation}.in', 'w') as f:
        f.write('\n')
        for kb, vb in blocks.items():
            f.write(f' &{kb}\n')
            for ks, vs in vb.items():
                f.write(f'  {ks} = {vs}\n')
            f.write(' /\n\n')

        if 'ATOMIC_SPECIES' in cards:
            for s in cards['ATOMIC_SPECIES']:
                f.write(s + '\n')
            f.write('\n')
            del cards['ATOMIC_SPECIES']

        for kc, vc in cards.items():
            for s in vc:
                f.write(s + '\n')
            f.write('\n')


def create_acbn0_inputfile(
    prefix,
    pthr,
    outputdir,
    expand_wedge=False,
    use_local_basis=False,
    basispath=None,
    configuration='standard',
):
    """Generate a PAOFLOW Python driver script for an ACBN0 calculation.

    Parameters
    ----------
    prefix : str
        QE calculation prefix; the save directory is ``{prefix}.save``.
    pthr : float
        Projectability threshold passed to ``paoflow.projectability()``.
    outputdir : str
        Output directory passed to the :class:`~PAOFLOW.PAOFLOW.PAOFLOW`
        constructor.
    expand_wedge : bool, optional
        Forwarded to ``paoflow.pao_hamiltonian``.  ``False`` (default)
        assumes QE produced the full BZ (``nosym=.true., noinv=.true.``);
        ``True`` expands the symmetry-reduced wedge to the full grid.
    use_local_basis : bool, optional
        When ``True`` the projections are computed internally by PAOFLOW
        (:meth:`PAOFLOW.PAOFLOW.PAOFLOW.projections`) instead of reading
        ``atomic_proj.xml`` from ``projwfc.x``.  The local projection
        orthonormalises the atomic orbitals, so the wavefunction overlap is
        set to the identity before the +U Hamiltonian is built, and the PAO
        basis metadata is dumped to ``<outputdir>/pao_basis.dat`` for the
        Hubbard-manifold selection in :meth:`ACBN0.run_acbn0`.
    basispath : str, optional
        Directory with the per-element radial basis files; forwarded to
        ``projections`` when ``use_local_basis`` is ``True`` and
        ``configuration`` is ``'standard'`` / ``'extended'``.
    configuration : str, optional
        Projection-basis preset (``'minimal'``, ``'standard'`` or
        ``'extended'``) forwarded to ``projections``.

    Returns
    -------
    None
        Writes the script ``acbn0.py`` to the current directory.
    """
    if use_local_basis:
        _create_acbn0_local_basis_inputfile(
            prefix, pthr, outputdir, expand_wedge, basispath, configuration
        )
        return

    with open('acbn0.py', 'w') as f:
        f.write('from PAOFLOW import PAOFLOW\n\n')
        f.write(
            f"paoflow = PAOFLOW.PAOFLOW(outputdir='{outputdir}', savedir='{prefix}.save', save_overlaps=True, acbn0=True)\n"
        )
        f.write('paoflow.read_atomic_proj_QE()\n')
        f.write(f'paoflow.projectability(pthr={pthr})\n')
        f.write(f'paoflow.pao_hamiltonian(write_binary=True,expand_wedge={bool(expand_wedge)})\n')
        f.write('paoflow.finish_execution()\n')


def _create_acbn0_local_basis_inputfile(
    prefix, pthr, outputdir, expand_wedge, basispath, configuration
):
    """Write ``acbn0.py`` for the local-basis (projwfc-free) ACBN0 path.

    The generated script projects the DFT eigenstates onto PAOFLOW's
    internal atomic basis, fixes the (orthonormal) overlap to the identity
    so that the non-orthogonal correction in
    :func:`PAOFLOW.hamiltonian.do_build_pao_hamiltonian.do_build_pao_hamiltonian`
    is a no-op, dumps the PAO-orbital metadata and finally builds and writes
    the +U Hamiltonian / overlap that :meth:`ACBN0.run_acbn0` consumes.

    The metadata dump and the identity overlap must be set *before*
    ``pao_hamiltonian`` because, with ``acbn0=True``, that call writes the
    binary dumps and terminates the process with ``sys.exit(0)``.
    """
    cfg = (configuration or 'standard').lower()
    basis_arg = 'None' if basispath is None else repr(basispath)

    lines = [
        'import os',
        'import numpy as np',
        'from PAOFLOW import PAOFLOW',
        '',
        'paoflow = PAOFLOW.PAOFLOW('
        f"outputdir='{outputdir}', savedir='{prefix}.save', "
        'save_overlaps=True, acbn0=True)',
        f'paoflow.projections(basispath={basis_arg}, configuration={cfg!r})',
        '',
        'arry, attr = paoflow.data_controller.data_dicts()',
        '',
        '# The local projection orthonormalises the atomic orbitals, so the',
        '# wavefunction overlap is the identity.  Set it explicitly so that',
        '# do_non_ortho (called for acbn0=True) leaves H(k) unchanged and the',
        '# dumped overlap kovp.npy is the identity (ortho-atomic scheme).',
        "nawf = attr['nawf']",
        "nkpnts = attr['nkpnts']",
        "arry['Sks'] = np.repeat(np.eye(nawf, dtype=complex)[:, :, None], nkpnts, axis=2)",
        '',
        '# Dump the PAO-orbital metadata (PAO order matches the dumped',
        '# Hamiltonian/overlap matrices) so that ACBN0.run_acbn0 can select',
        '# the Hubbard manifold by shell label and build the aligned',
        '# Gaussian basis.  Columns: index atom_index element l m label',
        "tau = list(arry['tau'])",
        '',
        '',
        'def _atom_index(t):',
        '    for ia, ta in enumerate(tau):',
        '        if np.allclose(ta, t):',
        '            return ia + 1',
        '    return 0',
        '',
        '',
        f"_meta_path = os.path.join({outputdir!r}, 'pao_basis.dat')",
        "with open(_meta_path, 'w') as _fmeta:",
        "    for _i, _b in enumerate(arry['basis']):",
        "        _fmeta.write('{0} {1} {2} {3} {4} {5}\\n'.format(",
        "            _i, _atom_index(_b['tau']), _b['atom'], _b['l'], _b['m'], _b['label']))",
        '',
        f'paoflow.projectability(pthr={pthr})',
        f'paoflow.pao_hamiltonian(write_binary=True, expand_wedge={bool(expand_wedge)})',
        'paoflow.finish_execution()',
        '',
    ]
    with open('acbn0.py', 'w') as f:
        f.write('\n'.join(lines))
