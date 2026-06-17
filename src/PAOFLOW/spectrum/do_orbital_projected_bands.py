import numpy as np
from os.path import join
from mpi4py import MPI


# Angular-momentum label -> quantum number l
L_LABELS = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

# Quantum number l -> spectroscopic label, for human-readable orbital tags.
L_NAMES = {0: 's', 1: 'p', 2: 'd', 3: 'f'}


def build_orbital_map(data_controller):
    """Map each PAO orbital onto its atom index and angular momentum.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Either the per-orbital ``basis`` list (built by
        :func:`projection.do_atwfc_proj.build_pswfc_basis_all`) or the
        ``shells``/``atoms`` metadata together with ``naw`` must be present.

    Returns
    -------
    orb_atom : np.ndarray, shape ``(nawf,)``, int
        0-based atom index that each orbital belongs to.
    orb_l : np.ndarray, shape ``(nawf,)``, int
        Angular momentum quantum number :math:`l` of each orbital.

    Notes
    -----
    When ``basis`` is available it is used directly, since it encodes the
    exact orbital ordering of the Hamiltonian (and therefore of ``v_k``).
    Atom boundaries are detected from the orbital positions ``tau`` so that
    two atoms of the same species are distinguished.  Otherwise the map is
    reconstructed from ``shells`` and ``atoms`` ((2l+1) orbitals per shell).

    Ad-hoc spin-orbit coupling (:meth:`PAOFLOW.adhoc_spin_orbit`) doubles the
    Hamiltonian as ``[spin-up block][spin-down block]`` but leaves ``basis``
    single, so the map (from either source) is tiled when ``nawf`` is twice its
    length, to line up with the spin-doubled ``v_k``.
    """
    arry, attr = data_controller.data_dicts()

    # Preferred: authoritative per-orbital basis built by do_atwfc_proj.
    if 'basis' in arry and len(arry['basis']) > 0:
        basis = arry['basis']
        orb_l = np.array([b['l'] for b in basis], dtype=int)
        orb_atom = np.zeros(len(basis), dtype=int)
        prev_tau = np.asarray(basis[0]['tau'], dtype=float)
        ia = 0
        for mu in range(1, len(basis)):
            tau = np.asarray(basis[mu]['tau'], dtype=float)
            if not np.allclose(tau, prev_tau):
                ia += 1
                prev_tau = tau
            orb_atom[mu] = ia
        # Ad-hoc SOC doubles the Hamiltonian ([up][down]) without doubling
        # 'basis', so nawf == 2*len(basis); tile the map to match v_k.
        if attr.get('nawf', len(basis)) == 2 * len(basis):
            orb_atom = np.concatenate([orb_atom, orb_atom])
            orb_l = np.concatenate([orb_l, orb_l])
        return orb_atom, orb_l

    # Fallback: reconstruct from shells/atoms (DFT-only path without basis).
    if 'naw' not in arry:
        data_controller.build_arrays_adhoc_soc()

    shells = arry.get('shells', None)
    atoms = arry.get('atoms', None)
    if atoms is None or not isinstance(shells, dict):
        raise Exception(
            "orbital_projected_bands requires either 'basis' or the "
            "'shells'/'atoms' metadata to map orbital character."
        )

    orb_atom, orb_l = [], []
    for ia, sp in enumerate(atoms):
        for l in shells[sp]:
            for _ in range(2 * int(l) + 1):
                orb_atom.append(ia)
                orb_l.append(int(l))
    orb_atom = np.array(orb_atom, dtype=int)
    orb_l = np.array(orb_l, dtype=int)

    # Ad-hoc SOC doubles the basis as [spin-up block][spin-down block]; detect
    # it structurally (nawf twice the spatial-orbital count) so it works
    # regardless of the flag name ('adhoc_SO' or legacy 'do_spin_orbit').
    if attr.get('nawf', orb_l.shape[0]) == 2 * orb_l.shape[0]:
        orb_atom = np.concatenate([orb_atom, orb_atom])
        orb_l = np.concatenate([orb_l, orb_l])

    return orb_atom, orb_l


def orbital_projected_bands(data_controller):
    """Write orbital-character-projected band weights ("fat bands") to file.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``E_k`` (shape ``(nkpi, nbnd, nspin)``),
        ``v_k`` (shape ``(nkpi, nawf, nbnd, nspin)``).
        Selection is read from the optional arrays ``orb_proj_atoms``
        (list of atom indices, ``None`` = all atoms) and ``orb_proj_l``
        (list of :math:`l` quantum numbers, ``None`` = all momenta).
        Output base name is taken from ``orb_proj_fname``.
        Required attributes: ``npool``, ``opath``.

    Returns
    -------
    None
        Writes one text file per spin channel to
        ``{opath}/{orb_proj_fname}_{ispin}.dat``.  Each line contains the
        k-point index (int), the band energy (float, eV) and the
        orbital-projected spectral weight (float), matching the column
        layout of :func:`do_site_projected_bands.site_projeted_bands` so the
        result can be plotted with ``GPAO.plot_weighted_bands``.

    Notes
    -----
    The weight of band :math:`n` at k-point :math:`k` is

    .. math::

        w_{nk} = \\sum_{\\mu \\in \\text{selection}}
            |\\langle \\mu | n, k \\rangle|^2 .

    The scattered ``E_k`` and ``v_k`` arrays are gathered to rank 0
    (collective call) and only rank 0 writes the output.
    """
    from ..utils.communication import gather_full

    arry, attr = data_controller.data_dicts()
    rank = MPI.COMM_WORLD.Get_rank()
    npool = attr['npool']

    # Gather the (MPI-scattered) band data onto rank 0.  Collective: every
    # rank must participate before the rank-0-only write below.
    E_k = gather_full(arry['E_k'], npool)
    v_k = gather_full(arry['v_k'], npool)

    if rank != 0:
        return

    nkpi, nawf, nbnd, nspin = v_k.shape

    orb_atom, orb_l = build_orbital_map(data_controller)
    if orb_l.shape[0] != nawf:
        raise Exception(
            'orbital_projected_bands: orbital map length (%d) does not match '
            'the number of orbitals nawf (%d).' % (orb_l.shape[0], nawf)
        )

    # Build the orbital selection mask from the requested atoms / momenta.
    atoms_sel = arry.get('orb_proj_atoms', None)
    l_sel = arry.get('orb_proj_l', None)

    mask = np.ones(nawf, dtype=bool)
    if atoms_sel is not None:
        mask &= np.isin(orb_atom, np.array(atoms_sel, dtype=int))
    if l_sel is not None:
        mask &= np.isin(orb_l, np.array(l_sel, dtype=int))

    if not mask.any():
        raise Exception(
            'orbital_projected_bands: the orbital selection is empty. Check the '
            "'atoms' indices and 'orbitals' characters."
        )

    fbase = arry.get('orb_proj_fname', 'orbital-projected-bands')

    for ispin in range(nspin):
        # weight[k, n] = sum_{mu in selection} |v_k[k, mu, n]|^2.
        # Slice the spin channel first so 'mask' is the only advanced index;
        # otherwise numpy moves the masked axis to the front (it would be
        # separated from the integer spin index by a slice).
        v_spin = v_k[:, :, :, ispin]
        weight = np.sum(np.abs(v_spin[:, mask, :]) ** 2, axis=1)

        with open(join(attr['opath'], '%s_%d.dat' % (fbase, ispin)), 'w') as f:
            for n in range(nbnd):
                for k in range(nkpi):
                    f.write(
                        '%d %s %s\n'
                        % (k, float(np.real(E_k[k, n, ispin])), float(weight[k, n]))
                    )


def species_orbital_projected_bands(data_controller):
    """Write orbital-resolved "fat band" weights for one atomic species.

    Unlike :func:`orbital_projected_bands`, which sums the weight over a whole
    angular-momentum channel, this routine projects onto **each individual
    orbital** of the requested species separately: the basis functions are
    grouped by their shell ``label`` together with the ``l`` and ``m`` quantum
    numbers, and the spectral weight of each group is summed over every
    equivalent atom of the species.

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``E_k`` ``(nkpi, nbnd, nspin)``, ``v_k``
        ``(nkpi, nawf, nbnd, nspin)`` and the per-orbital ``basis`` (built by
        the atomic-wavefunction projection) whose records carry ``atom`` (the
        species symbol), ``l``, ``m`` and ``label``.  The species symbol is read
        from ``orb_proj_species`` and the output base name from
        ``orb_proj_fname``.  Required attributes: ``npool``, ``opath``.

    Returns
    -------
    list of (str, str)
        On rank 0, one ``(path, label)`` pair per file written, ordered as the
        orbitals appear in the basis; ``label`` is a short human-readable tag
        such as ``'Mo 4D (d, m=1)'`` suitable for a plot title / colour-bar.
        Empty list on the other ranks.  Each file uses the same three-column
        layout (k-index, eigenvalue, weight) as
        :func:`orbital_projected_bands` and is named
        ``{opath}/{orb_proj_fname}_{species}_{shell}_m{m}_{ispin}.dat``.

    Notes
    -----
    The per-orbital ``basis`` is required: without it only the angular momentum
    ``l`` is known (not ``m``), so individual orbitals cannot be resolved -- use
    :func:`orbital_projected_bands` in that case.
    """
    from ..utils.communication import gather_full

    arry, attr = data_controller.data_dicts()
    rank = MPI.COMM_WORLD.Get_rank()
    npool = attr['npool']

    # Collective: every rank must reach the gather before the rank-0-only write.
    E_k = gather_full(arry['E_k'], npool)
    v_k = gather_full(arry['v_k'], npool)

    if rank != 0:
        return []

    nkpi, nawf, nbnd, nspin = v_k.shape

    if 'basis' not in arry or len(arry['basis']) == 0:
        raise Exception(
            "species_orbital_projected_bands requires the per-orbital 'basis' "
            'to resolve individual orbitals; it is not available for this run '
            "(only an l-resolved projection is possible -- use "
            "'orbital_projected_bands' instead)."
        )

    basis = arry['basis']
    nbasis = len(basis)
    # Ad-hoc SOC doubles the Hamiltonian ([up][down]) without doubling 'basis',
    # so nawf is either nbasis (no SOC) or 2*nbasis (ad-hoc SOC).
    soc_doubled = nawf == 2 * nbasis
    if nawf != nbasis and not soc_doubled:
        raise Exception(
            'species_orbital_projected_bands: basis length (%d) is not '
            'compatible with the number of orbitals nawf (%d).' % (nbasis, nawf)
        )

    species = arry.get('orb_proj_species', None)
    if species is None:
        raise Exception(
            "species_orbital_projected_bands: no species requested; set "
            "'orb_proj_species'."
        )

    # Group this species' basis indices by individual orbital (shell, l, m).
    # Equivalent atoms of the species share the same key and are summed.
    groups = {}
    for mu, b in enumerate(basis):
        if b['atom'] != species:
            continue
        key = (str(b['label']).strip(), int(b['l']), int(b['m']))
        groups.setdefault(key, []).append(mu)

    if not groups:
        present = sorted({b['atom'] for b in basis})
        raise Exception(
            "species_orbital_projected_bands: no orbitals found for species "
            "'%s'. Species present in the basis: %s." % (species, present)
        )

    fbase = arry.get('orb_proj_fname', 'orbital-projected-bands')
    written = []

    for (label, l, m), idx in groups.items():
        idx = np.array(idx, dtype=int)
        if soc_doubled:
            # Spin-down copies live at the same offset in the second block;
            # include them so the weight is summed over both spin channels.
            idx = np.concatenate([idx, idx + nbasis])
        shell = label.replace(' ', '')
        lname = L_NAMES.get(l, 'l%d' % l)
        nice = '%s %s (%s, m=%d)' % (species, shell, lname, m)
        for ispin in range(nspin):
            # Slice the spin channel first so 'idx' is the only advanced index.
            v_spin = v_k[:, :, :, ispin]
            weight = np.sum(np.abs(v_spin[:, idx, :]) ** 2, axis=1)

            path = join(
                attr['opath'],
                '%s_%s_%s_m%d_%d.dat' % (fbase, species, shell, m, ispin),
            )
            with open(path, 'w') as f:
                for n in range(nbnd):
                    for k in range(nkpi):
                        f.write(
                            '%d %s %s\n'
                            % (k, float(np.real(E_k[k, n, ispin])), float(weight[k, n]))
                        )
            written.append((path, nice))

    return written


def character_projected_bands(data_controller):
    """Write "fat band" weights summed over all atoms for each l character.

    For every requested angular-momentum character (``s``, ``p``, ``d``, ``f``)
    the spectral weight is summed over **all** orbitals of that character in the
    system, irrespective of species or atom.  One file per character and spin
    channel is written.

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``E_k`` ``(nkpi, nbnd, nspin)``, ``v_k``
        ``(nkpi, nawf, nbnd, nspin)`` and the orbital-map source used by
        :func:`build_orbital_map` (the per-orbital ``basis`` or the
        ``shells``/``atoms`` metadata -- only ``l`` is needed here, so the
        basis is optional).  The requested characters are read from
        ``orb_proj_characters`` (a list among ``'s','p','d','f'``; ``None``
        selects every character present) and the output base name from
        ``orb_proj_fname``.  Required attributes: ``npool``, ``opath``.

    Returns
    -------
    list of (str, str)
        On rank 0, one ``(path, label)`` pair per file written; empty on the
        other ranks.  Each file uses the same three-column layout (k-index,
        eigenvalue, weight) as :func:`orbital_projected_bands` and is named
        ``{opath}/{orb_proj_fname}_{char}_{ispin}.dat``.
    """
    from ..utils.communication import gather_full

    arry, attr = data_controller.data_dicts()
    rank = MPI.COMM_WORLD.Get_rank()
    npool = attr['npool']

    # Collective: every rank must reach the gather before the rank-0-only write.
    E_k = gather_full(arry['E_k'], npool)
    v_k = gather_full(arry['v_k'], npool)

    if rank != 0:
        return []

    nkpi, nawf, nbnd, nspin = v_k.shape

    _, orb_l = build_orbital_map(data_controller)
    if orb_l.shape[0] != nawf:
        raise Exception(
            'character_projected_bands: orbital map length (%d) does not match '
            'the number of orbitals nawf (%d).' % (orb_l.shape[0], nawf)
        )

    # Resolve the requested characters; default to every character present.
    requested = arry.get('orb_proj_characters', None)
    if requested is None:
        ls = sorted({int(l) for l in orb_l})
    else:
        try:
            ls = [L_LABELS[c.lower()] for c in requested]
        except (KeyError, AttributeError):
            raise Exception(
                "character_projected_bands: 'characters' must be a list of "
                "labels among 's','p','d','f'."
            )

    fbase = arry.get('orb_proj_fname', 'orbital-projected-bands')
    written = []

    for l in ls:
        mask = orb_l == l
        if not mask.any():
            # A requested character that is absent from the basis is skipped.
            continue
        cname = L_NAMES.get(l, 'l%d' % l)
        for ispin in range(nspin):
            # Slice the spin channel first so 'mask' is the only advanced index.
            v_spin = v_k[:, :, :, ispin]
            weight = np.sum(np.abs(v_spin[:, mask, :]) ** 2, axis=1)

            path = join(attr['opath'], '%s_%s_%d.dat' % (fbase, cname, ispin))
            with open(path, 'w') as f:
                for n in range(nbnd):
                    for k in range(nkpi):
                        f.write(
                            '%d %s %s\n'
                            % (k, float(np.real(E_k[k, n, ispin])), float(weight[k, n]))
                        )
            written.append((path, 'total %s' % cname))

    return written
