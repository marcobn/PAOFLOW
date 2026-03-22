#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Species-pair-keyed format normalizers (backwards compatible)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _normalize_species_pair_hoppings(hoppings):
    """Detect species-pair-keyed hoppings and convert to shell-tag format.

    New format (species-pair-keyed)::

        hoppings = {
            "Si-Si": [
                {"r_ref": 4.44, "params": {"sss": -0.1, "sps": 0.2, ...}},
                {"r_ref": 7.28, "params": {"sss": -0.05, ...}},
            ]
        }

    Old format (shell-tag-keyed)::

        hoppings = {"nn": {"sss": -0.1, ...}, "nnn": {"sss": -0.05, ...}}
        # or flat (single-shell):
        hoppings = {"sss": -0.1, ...}

    Returns
    -------
    (hoppings_norm, n_shells, converted) where:
      - hoppings_norm : dict  — hoppings in old shell-tag format (ready for
        the existing parsing logic).  If already in old format, returned as-is.
        For multi-species, returns a dict keyed by species pair, each value
        being a shell-tag dict: {"Si-Si": {"nn": {...}, "nnn": {...}}, ...}.
      - n_shells : int  — number of neighbour shells (1, 2, or 3).
        -1 signals distance-dependent mode.
        -2 signals multi-species shell mode.
      - converted : bool — True if new-format conversion was performed.
    """
    if not isinstance(hoppings, dict) or len(hoppings) == 0:
        return hoppings, 1, False

    first_key = next(iter(hoppings))
    first_val = hoppings[first_key]

    # Distance-dependent format: value is a dict with "channels" key
    if isinstance(first_val, dict) and 'channels' in first_val and '-' in first_key:
        # Return as-is — SK_EDTB handles this format directly
        return hoppings, -1, True  # n_shells=-1 signals distance-dependent

    # New format: top-level key contains "-" and value is a list of shell dicts
    if isinstance(first_val, list) and '-' in first_key:
        shell_tags = ['nn', 'nnn', 'nnnn']
        if len(hoppings) == 1:
            # Single species pair — flatten to shell-tag format
            shells = first_val
            shells_sorted = sorted(shells, key=lambda s: s['r_ref'])
            n_shells = len(shells_sorted)
            if n_shells > 3:
                raise ValueError(f'Too many neighbour shells ({n_shells}); maximum 3 supported.')
            hoppings_norm = {}
            for idx, shell_data in enumerate(shells_sorted):
                hoppings_norm[shell_tags[idx]] = shell_data['params']
            return hoppings_norm, n_shells, True
        else:
            # Multi-species pairs — keep species-pair keys, convert each to shell-tag
            hoppings_norm = {}
            n_shells = None
            for pair_key, shells in hoppings.items():
                shells_sorted = sorted(shells, key=lambda s: s['r_ref'])
                ns = len(shells_sorted)
                if ns > 3:
                    raise ValueError(
                        f"Too many neighbour shells ({ns}) for pair '{pair_key}'; "
                        f'maximum 3 supported.'
                    )
                if n_shells is None:
                    n_shells = ns
                pair_dict = {}
                for idx, shell_data in enumerate(shells_sorted):
                    pair_dict[shell_tags[idx]] = shell_data['params']
                hoppings_norm[pair_key] = pair_dict
            return hoppings_norm, -2, True  # -2 signals multi-species shell mode

    # Old format — pass through unchanged
    return hoppings, 0, False


def _normalize_species_pair_gamma(gamma_spec):
    """Unwrap species-pair-keyed gamma for single-species systems.

    New format::

        gamma = {"Pt-Pt": {"ss": 0.1, "sp": 0.2, ...}}

    Old format (returned as-is)::

        gamma = {"ss": 0.1, ...}   # per-channel
        gamma = 0.05               # global scalar

    For multi-species, returns the full dict keyed by species pair.

    Returns the bare gamma specification (float, flat dict, or species-pair dict).
    """
    if isinstance(gamma_spec, dict) and len(gamma_spec) > 0:
        first_key = next(iter(gamma_spec))
        first_val = gamma_spec[first_key]
        if isinstance(first_val, (dict, float, int)) and '-' in first_key:
            if len(gamma_spec) == 1:
                return first_val
            # Multi-species: return the full species-pair-keyed dict
            return gamma_spec
    return gamma_spec


def Slater_Koster(data_controller, params):
    """Build a generalized Slater-Koster tight-binding model (two-center, up to 3NN).

    This routine populates the PAOFLOW data containers with a real-space
    tight-binding Hamiltonian constructed in the two-center approximation,
    up to third nearest neighbors. It uses a simple orbital basis per
    atom (s,p and d) and derives hopping matrix elements from the standard
    Slater-Koster direction-cosine expressions.

    High-level workflow:
    - Read lattice vectors and atomic positions from params and set up basic
        attributes (number of atoms, orbitals, hoppings).
    - Compute reciprocal lattice vectors and the cell volume.
    - Build a 3x3x3 supercell of atomic positions to identify neighbors and
        determine a first-neighbor cutoff.
    - Construct the Dnm matrix (orbital-position differences) for gradient
        calculations.
    - Allocate the real-space Hamiltonian array HRs and fill on-site energies
        and hopping terms using direction cosines and SK parameters.

    Data structure expectations (params):
    - params['model']['a_vectors']: 3x3 lattice vectors.
    - params['model']['atoms']: dict keyed by string indices ("0", "1", ...),
        each containing:
            - 'name': atomic species label
            - 'tau': fractional/cartesian position in lattice units
            - 'orbitals': list of orbital labels used to map on-site terms
            - on-site energies keyed by orbital label
    - params['model']['hoppings']: either a flat dict of SK parameters with keys
        'sss', 'sps', 'pps', 'ppp', etc., or a shell dict with 'nn' and optional
        'nnn'/'nnnn' blocks containing those keys.

    Output side-effects (data_controller):
    - arry['a_vectors'], arry['b_vectors'], arry['tau'], arry['atoms'],
        arry['shells'], arry['norbitals'], arry['sctau'], arry['Dnm'],
        arry['HRs']
    - attr['alat'], attr['omega'], attr['natoms'], attr['nawf'], attr['bnd'],
        attr['nbnds'], attr['nk1'], attr['nk2'], attr['nk3'], attr['nkpnts'],
        attr['nspin'], attr['dftSO'], attr['shift'], attr['cutoff']

    Notes:
    - The neighbor search uses a 3x3x3 supercell for nn only, 5x5x5 when nnn
        is enabled, and 7x7x7 when nnnn is enabled. Shells are determined from
        distinct neighbor distances and mid-point cutoffs.
    - Only the unpolarized case is supported; spin-orbit is disabled here.
    """

    import numpy as np

    arry, attr = data_controller.data_dicts()
    # Lattice Vectors
    arry['a_vectors'] = np.array(params['model']['a_vectors'])
    attr['alat'] = params.get('alat', 1.0)

    # Atomic coordinates
    natoms = len(params['model']['atoms'])
    tau = np.zeros((natoms, 3), dtype=float)
    for ia in range(natoms):
        tau[ia] = np.array(params['model']['atoms'][str(ia)]['tau'])
    atoms = []
    shells = []
    for ia in range(natoms):
        atoms.append(params['model']['atoms'][str(ia)]['name'])
        shells.append(params['model']['atoms'][str(ia)])
    arry['tau'] = tau
    arry['atoms'] = atoms
    attr['natoms'] = natoms
    arry['shells'] = shells
    attr['nspin'] = 1  # only unpolarized case for now
    attr['dftSO'] = False  # no spin-orbit

    # ── Multi-shell support ──────────────────────────────────────
    # When atoms specify 'configuration' (e.g. ['3S','3P','3D','4S','4P']),
    # each pair of shell groups gets its own set of SK hopping parameters.
    _config_l_map = {'S': 0, 'P': 1, 'D': 2, 'F': 3}
    _l_to_orbitals = {
        0: ['s'],
        1: ['px', 'py', 'pz'],
        2: ['dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2'],
    }
    _lpair_required_keys = {
        (0, 0): ['sss'],
        (0, 1): ['sps'],
        (0, 2): ['sds'],
        (1, 1): ['pps', 'ppp'],
        (1, 2): ['pds', 'pdp'],
        (2, 2): ['dds', 'ddp', 'ddd'],
    }

    is_multishell = any(
        'configuration' in params['model']['atoms'][str(ia)] for ia in range(natoms)
    )

    atom_config = []  # config labels per atom
    atom_angular_orbs = []  # angular orbital names per atom (may have duplicates)
    atom_orbital_groups = []  # config group index per orbital

    if is_multishell:
        for ia in range(natoms):
            atom_dict = params['model']['atoms'][str(ia)]
            if 'configuration' not in atom_dict:
                raise ValueError(
                    f"Atom {ia}: 'configuration' key required when multi-shell mode is used."
                )
            config = atom_dict['configuration']
            atom_config.append(config)
            ang_orbs = []
            groups = []
            for ig, cfg_label in enumerate(config):
                l_val = _config_l_map[cfg_label[-1].upper()]
                if l_val not in _l_to_orbitals:
                    raise ValueError(
                        f"Unsupported angular momentum l={l_val} for config '{cfg_label}'."
                    )
                for orb_name in _l_to_orbitals[l_val]:
                    ang_orbs.append(orb_name)
                    groups.append(ig)
            atom_angular_orbs.append(ang_orbs)
            atom_orbital_groups.append(groups)
            # Populate 'orbitals' for downstream code
            atom_dict['orbitals'] = ang_orbs

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    attr['omega'] = attr['alat'] ** 3 * volume
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    hoppings = params['model']['hoppings']

    # ── Species-pair-keyed format auto-detection ──────────────
    hoppings, _sp_n_shells, _sp_converted = _normalize_species_pair_hoppings(hoppings)

    if _sp_converted:
        use_second_neighbors = _sp_n_shells >= 2
        use_third_neighbors = _sp_n_shells >= 3
    else:
        use_third_neighbors = isinstance(hoppings, dict) and 'nnnn' in hoppings
        use_second_neighbors = isinstance(hoppings, dict) and (
            'nnn' in hoppings or use_third_neighbors
        )
        if (use_second_neighbors or use_third_neighbors) and 'nn' not in hoppings:
            raise ValueError(
                'Neighbor hoppings require a "nn" block in params["model"]["hoppings"].'
            )
        if use_third_neighbors and 'nnn' not in hoppings:
            raise ValueError(
                'Third-neighbor hoppings require a "nnn" block in params["model"]["hoppings"].'
            )

    # dimensions of the supercell for two-center approximation
    if use_third_neighbors:
        cell_range = 3
    elif use_second_neighbors:
        cell_range = 2
    else:
        cell_range = 1
    nk1 = nk2 = nk3 = 2 * cell_range + 1
    nkpnts = nk1 * nk2 * nk3
    attr['nk1'] = nk1
    attr['nk2'] = nk2
    attr['nk3'] = nk3
    attr['nkpnts'] = nkpnts

    # on site and hopping parameters
    norbitals = np.zeros(natoms, dtype=int)
    for ia in range(natoms):
        norbitals[ia] = len(params['model']['atoms'][str(ia)]['orbitals'])

    nawf = 0
    for ia in range(natoms):
        nawf += norbitals[ia]
    attr['nawf'] = nawf
    attr['bnd'] = nawf
    attr['nbnds'] = nawf
    attr['shift'] = 0

    # Cumulative orbital block start per atom (correct for heterogeneous norbitals)
    atom_block_start = np.zeros(natoms, dtype=int)
    for ia in range(1, natoms):
        atom_block_start[ia] = atom_block_start[ia - 1] + norbitals[ia - 1]

    # Define the Dnm matrix for correct calculation of gradient
    basis = []
    for i in range(attr['natoms']):
        for k in range(len(arry['shells'][i]['orbitals'])):
            basis.append(arry['shells'][i]['tau'])
    arry['Dnm'] = np.empty((attr['nawf'], attr['nawf'], 3))
    for i in range(3):
        for n in range(attr['nawf']):
            for m in range(attr['nawf']):
                arry['Dnm'][n, m, i] = basis[n][i] - basis[m][i]

    # generate all the orbitals positions in the supercell
    sctau = np.zeros((natoms, nk1, nk2, nk3, 3), dtype=float)
    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                for ia in range(natoms):
                    sctau[ia, i, j, k, :] = (
                        tau[ia]
                        + i * arry['a_vectors'][0]
                        + j * arry['a_vectors'][1]
                        + k * arry['a_vectors'][2]
                    )
    sctau = np.reshape(sctau, (natoms * nk1 * nk2 * nk3, 3), order='C')
    # make the list of neighbors and find cutoff for two-center approximation
    distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    cosines = lambda x, y: (y - x) / np.sqrt(np.sum((x - y) ** 2))
    dist = []
    for ia in range(natoms):
        for n in range(natoms * nk1 * nk2 * nk3):
            dist.append(distance(tau[ia], sctau[n]))
    unique_dist = np.sort(np.unique(dist))
    # Merge nearly-degenerate distances (e.g. from floating-point
    # precision in DFT atomic positions).  Use relative tolerance 1e-3.
    _merged = [unique_dist[0]]
    for ud in unique_dist[1:]:
        if _merged[-1] > 0 and abs(ud - _merged[-1]) / _merged[-1] < 1e-3:
            continue  # skip — same shell as previous
        _merged.append(ud)
    unique_dist = np.array(_merged)

    if unique_dist.size < 2:
        raise ValueError('Unable to determine nearest-neighbor distances for Slater-Koster model.')

    dist_1 = unique_dist[1]
    if use_third_neighbors:
        if unique_dist.size < 4:
            raise ValueError(
                'Unable to determine third-neighbor distances for Slater-Koster model.'
            )
        dist_2 = unique_dist[2]
        dist_3 = unique_dist[3]
        cutoff_1 = dist_1 + (dist_2 - dist_1) / 2.0
        cutoff_2 = dist_2 + (dist_3 - dist_2) / 2.0
        if unique_dist.size > 4:
            dist_4 = unique_dist[4]
            cutoff_3 = dist_3 + (dist_4 - dist_3) / 2.0
        else:
            cutoff_3 = dist_3 + (dist_3 - dist_2) / 2.0
    elif use_second_neighbors:
        if unique_dist.size < 3:
            raise ValueError(
                'Unable to determine second-neighbor distances for Slater-Koster model.'
            )
        dist_2 = unique_dist[2]
        cutoff_1 = dist_1 + (dist_2 - dist_1) / 2.0
        if unique_dist.size > 3:
            dist_3 = unique_dist[3]
            cutoff_2 = dist_2 + (dist_3 - dist_2) / 2.0
        else:
            cutoff_2 = dist_2 + (dist_2 - dist_1) / 2.0
        cutoff_3 = None
    else:
        if unique_dist.size < 3:
            raise ValueError('Unable to determine cutoff for first-neighbor shell.')
        cutoff_1 = dist_1 + (unique_dist[2] - dist_1) / 2.0
        cutoff_2 = None
        cutoff_3 = None
    sctau = np.reshape(sctau, (natoms, nk1, nk2, nk3, 3), order='C')

    # debug
    arry['sctau'] = sctau
    attr['cutoff'] = cutoff_1
    attr['cutoff_1'] = cutoff_1
    if cutoff_2 is not None:
        attr['cutoff_2'] = cutoff_2
    if cutoff_3 is not None:
        attr['cutoff_3'] = cutoff_3
    arry['norbitals'] = norbitals

    HRs = np.zeros((nawf, nawf, nk1, nk2, nk3, 1), dtype=complex)

    # on-site matrix elements
    if is_multishell:
        # Multi-shell: on-site energy per configuration group.
        # d-shells may optionally be split into t2g/eg.
        for ia in range(natoms):
            atom_dict = params['model']['atoms'][str(ia)]
            config = atom_config[ia]
            start = atom_block_start[ia]
            idx = 0
            for ig, cfg_label in enumerate(config):
                l_val = _config_l_map[cfg_label[-1].upper()]
                norb_l = 2 * l_val + 1
                if l_val <= 1:
                    e = atom_dict[cfg_label]
                    for io in range(norb_l):
                        HRs[start + idx + io, start + idx + io, 0, 0, 0, 0] = e
                elif l_val == 2:
                    key_t2g = f'{cfg_label}_t2g'
                    key_eg = f'{cfg_label}_eg'
                    if key_t2g in atom_dict:
                        e_t2g = atom_dict[key_t2g]
                        e_eg = atom_dict[key_eg]
                    else:
                        e_t2g = e_eg = atom_dict[cfg_label]
                    for io in range(3):  # dxy, dyz, dzx (t2g)
                        HRs[start + idx + io, start + idx + io, 0, 0, 0, 0] = e_t2g
                    for io in range(3, 5):  # dx2-y2, dz2 (eg)
                        HRs[start + idx + io, start + idx + io, 0, 0, 0, 0] = e_eg
                idx += norb_l
    else:
        for ia in range(natoms):
            for no in range(norbitals[ia]):
                HRs[
                    atom_block_start[ia] + no,
                    atom_block_start[ia] + no,
                    0,
                    0,
                    0,
                    0,
                ] = params['model']['atoms'][str(ia)][
                    params['model']['atoms'][str(ia)]['orbitals'][no]
                ]

    # hopping matrix elements
    orbital_order = ('s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2')
    p_index_map = {'px': 0, 'py': 1, 'pz': 2}
    d_orbitals = set(orbital_order[4:])
    supported_orbitals = set(orbital_order)

    for shell in arry['shells']:
        for orb in shell['orbitals']:
            if orb not in supported_orbitals:
                raise ValueError(f'Unsupported orbital label: {orb}')

    if 'nn' in hoppings:
        hoppings_shells = {'nn': hoppings['nn']}
        if 'nnn' in hoppings:
            hoppings_shells['nnn'] = hoppings['nnn']
        if 'nnnn' in hoppings:
            hoppings_shells['nnnn'] = hoppings['nnnn']
    else:
        hoppings_shells = {'nn': hoppings}

    if use_second_neighbors and 'nnn' not in hoppings_shells:
        raise KeyError('Second-neighbor hoppings requested but no "nnn" block provided.')
    if use_third_neighbors and 'nnnn' not in hoppings_shells:
        raise KeyError('Third-neighbor hoppings requested but no "nnnn" block provided.')

    required_keys = ['sss', 'sps', 'pps', 'ppp']
    has_d = any(orb in d_orbitals for shell in arry['shells'] for orb in shell['orbitals'])
    if has_d:
        required_keys.extend(['sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd'])

    if is_multishell:
        # Multi-shell validation: each group-pair sub-dict must contain
        # the SK keys required by the angular-momentum pair.
        for shell_name, shell_hop_dict in hoppings_shells.items():
            for pair_key, pair_hoppings in shell_hop_dict.items():
                parts = pair_key.split('-')
                if len(parts) != 2:
                    raise KeyError(
                        f"Invalid group-pair key '{pair_key}' in {shell_name}. "
                        "Expected format: 'GroupA-GroupB' (e.g. '3S-3P')."
                    )
                la = _config_l_map[parts[0][-1].upper()]
                lb = _config_l_map[parts[1][-1].upper()]
                lpair = (min(la, lb), max(la, lb))
                needed = _lpair_required_keys[lpair]
                missing_keys = [k for k in needed if k not in pair_hoppings]
                if missing_keys:
                    raise KeyError(
                        f"Missing SK keys for {shell_name}/'{pair_key}' "
                        f'(l-pair {lpair}): {", ".join(missing_keys)}'
                    )
    else:
        for shell_name, shell_hoppings in hoppings_shells.items():
            missing_keys = [key for key in required_keys if key not in shell_hoppings]
            if missing_keys:
                raise KeyError(
                    f'Missing Slater-Koster hopping keys for '
                    f'{shell_name}: {", ".join(missing_keys)}'
                )

    elements = sorted(set(arry['atoms']))
    shell_names = ', '.join(sorted(hoppings_shells.keys()))
    if is_multishell:
        config_str = ', '.join(atom_config[0])
        print(
            f'Slater_Koster (multi-shell): elements={elements}, '
            f'config=[{config_str}], '
            f'neighbor_shells={len(hoppings_shells)} ({shell_names})'
        )
    else:
        print(
            f'Slater_Koster: elements={elements}, '
            f'neighbor_shells={len(hoppings_shells)} ({shell_names})'
        )

    # Standard Slater-Koster normalization factors for cubic harmonics
    # (Slater & Koster, Phys. Rev. 94, 1498, 1954)
    sq3 = np.sqrt(3.0)  # √3 ≈ 1.7321
    hsq3 = sq3 / 2.0  # √3/2 ≈ 0.8660

    def _sd_value(d_orb, lx, ly, lz, shell_hoppings):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        if d_orb == 'dxy':
            return sq3 * lx * ly * shell_hoppings['sds']
        if d_orb == 'dyz':
            return sq3 * ly * lz * shell_hoppings['sds']
        if d_orb == 'dzx':
            return sq3 * lz * lx * shell_hoppings['sds']
        if d_orb == 'dx2-y2':
            return hsq3 * (l2 - m2) * shell_hoppings['sds']
        if d_orb == 'dz2':
            return (n2 - 0.5 * (l2 + m2)) * shell_hoppings['sds']
        return None

    def _pd_value(p_orb, d_orb, lx, ly, lz, shell_hoppings):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        if p_orb == 'px':
            if d_orb == 'dxy':
                return (
                    sq3 * l2 * ly * shell_hoppings['pds']
                    + ly * (1.0 - 2.0 * l2) * shell_hoppings['pdp']
                )
            if d_orb == 'dyz':
                return (
                    sq3 * lx * ly * lz * shell_hoppings['pds']
                    - 2.0 * lx * ly * lz * (shell_hoppings['pdp'])
                )
            if d_orb == 'dzx':
                return (
                    sq3 * l2 * lz * shell_hoppings['pds']
                    + lz * (1.0 - 2.0 * l2) * shell_hoppings['pdp']
                )
            if d_orb == 'dx2-y2':
                return (
                    hsq3 * lx * (l2 - m2) * shell_hoppings['pds']
                    + lx * (1.0 - l2 + m2) * shell_hoppings['pdp']
                )
            if d_orb == 'dz2':
                return (
                    lx * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['pds']
                    - sq3 * lx * n2 * shell_hoppings['pdp']
                )
        if p_orb == 'py':
            if d_orb == 'dxy':
                return (
                    sq3 * m2 * lx * shell_hoppings['pds']
                    + lx * (1.0 - 2.0 * m2) * shell_hoppings['pdp']
                )
            if d_orb == 'dyz':
                return (
                    sq3 * m2 * lz * shell_hoppings['pds']
                    + lz * (1.0 - 2.0 * m2) * shell_hoppings['pdp']
                )
            if d_orb == 'dzx':
                return (
                    sq3 * lx * ly * lz * shell_hoppings['pds']
                    - 2.0 * lx * ly * lz * (shell_hoppings['pdp'])
                )
            if d_orb == 'dx2-y2':
                return (
                    hsq3 * ly * (l2 - m2) * shell_hoppings['pds']
                    - ly * (1.0 + l2 - m2) * shell_hoppings['pdp']
                )
            if d_orb == 'dz2':
                return (
                    ly * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['pds']
                    - sq3 * ly * n2 * shell_hoppings['pdp']
                )
        if p_orb == 'pz':
            if d_orb == 'dxy':
                return (
                    sq3 * lx * ly * lz * shell_hoppings['pds']
                    - 2.0 * lx * ly * lz * (shell_hoppings['pdp'])
                )
            if d_orb == 'dyz':
                return (
                    sq3 * n2 * ly * shell_hoppings['pds']
                    + ly * (1.0 - 2.0 * n2) * shell_hoppings['pdp']
                )
            if d_orb == 'dzx':
                return (
                    sq3 * n2 * lx * shell_hoppings['pds']
                    + lx * (1.0 - 2.0 * n2) * shell_hoppings['pdp']
                )
            if d_orb == 'dx2-y2':
                return (
                    hsq3 * lz * (l2 - m2) * shell_hoppings['pds']
                    - lz * (l2 - m2) * shell_hoppings['pdp']
                )
            if d_orb == 'dz2':
                return (
                    lz * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['pds']
                    + sq3 * lz * (l2 + m2) * shell_hoppings['pdp']
                )
        return None

    def _dd_value(d_orb_a, d_orb_b, lx, ly, lz, shell_hoppings):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        lm = lx * ly
        ln = lx * lz
        mn = ly * lz
        l2m2 = l2 * m2
        l2n2 = l2 * n2
        m2n2 = m2 * n2
        diff_lm = l2 - m2

        if d_orb_a == d_orb_b == 'dxy':
            return (
                3.0 * l2m2 * shell_hoppings['dds']
                + (l2 + m2 - 4.0 * l2m2) * shell_hoppings['ddp']
                + (n2 + l2m2) * shell_hoppings['ddd']
            )
        if d_orb_a == d_orb_b == 'dyz':
            return (
                3.0 * m2n2 * shell_hoppings['dds']
                + (m2 + n2 - 4.0 * m2n2) * shell_hoppings['ddp']
                + (l2 + m2n2) * shell_hoppings['ddd']
            )
        if d_orb_a == d_orb_b == 'dzx':
            return (
                3.0 * l2n2 * shell_hoppings['dds']
                + (l2 + n2 - 4.0 * l2n2) * shell_hoppings['ddp']
                + (m2 + l2n2) * shell_hoppings['ddd']
            )
        if d_orb_a == d_orb_b == 'dx2-y2':
            return (
                0.75 * diff_lm**2 * shell_hoppings['dds']
                + (l2 + m2 - diff_lm**2) * shell_hoppings['ddp']
                + (n2 + 0.25 * diff_lm**2) * shell_hoppings['ddd']
            )
        if d_orb_a == d_orb_b == 'dz2':
            term = n2 - 0.5 * (l2 + m2)
            return (
                term**2 * shell_hoppings['dds']
                + 3.0 * n2 * (l2 + m2) * shell_hoppings['ddp']
                + 0.75 * (l2 + m2) ** 2 * shell_hoppings['ddd']
            )

        if (d_orb_a, d_orb_b) in (('dxy', 'dyz'), ('dyz', 'dxy')):
            return (
                3.0 * lx * m2 * lz * shell_hoppings['dds']
                + ln * (1.0 - 4.0 * m2) * shell_hoppings['ddp']
                + ln * (m2 - 1.0) * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dxy', 'dzx'), ('dzx', 'dxy')):
            return (
                3.0 * l2 * ly * lz * shell_hoppings['dds']
                + mn * (1.0 - 4.0 * l2) * shell_hoppings['ddp']
                + mn * (l2 - 1.0) * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dyz', 'dzx'), ('dzx', 'dyz')):
            return (
                3.0 * ly * n2 * lx * shell_hoppings['dds']
                + lm * (1.0 - 4.0 * n2) * shell_hoppings['ddp']
                + lm * (n2 - 1.0) * shell_hoppings['ddd']
            )

        if (d_orb_a, d_orb_b) in (('dxy', 'dx2-y2'), ('dx2-y2', 'dxy')):
            return (
                1.5 * lm * diff_lm * shell_hoppings['dds']
                + 2.0 * lm * (m2 - l2) * shell_hoppings['ddp']
                + 0.5 * lm * diff_lm * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dyz', 'dx2-y2'), ('dx2-y2', 'dyz')):
            return (
                1.5 * mn * diff_lm * shell_hoppings['dds']
                - mn * (1.0 + 2.0 * diff_lm) * shell_hoppings['ddp']
                + mn * (1.0 + 0.5 * diff_lm) * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dzx', 'dx2-y2'), ('dx2-y2', 'dzx')):
            return (
                1.5 * ln * diff_lm * shell_hoppings['dds']
                + ln * (1.0 - 2.0 * diff_lm) * shell_hoppings['ddp']
                - ln * (1.0 - 0.5 * diff_lm) * shell_hoppings['ddd']
            )

        if (d_orb_a, d_orb_b) in (('dxy', 'dz2'), ('dz2', 'dxy')):
            return sq3 * (
                lm * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['dds']
                - 2.0 * lm * n2 * shell_hoppings['ddp']
                + 0.5 * lm * (1.0 + n2) * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dyz', 'dz2'), ('dz2', 'dyz')):
            return sq3 * (
                mn * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['dds']
                + mn * (l2 + m2 - n2) * shell_hoppings['ddp']
                - 0.5 * mn * (l2 + m2) * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dzx', 'dz2'), ('dz2', 'dzx')):
            return sq3 * (
                ln * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['dds']
                + ln * (l2 + m2 - n2) * shell_hoppings['ddp']
                - 0.5 * ln * (l2 + m2) * shell_hoppings['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dx2-y2', 'dz2'), ('dz2', 'dx2-y2')):
            return sq3 * (
                0.5 * diff_lm * (n2 - 0.5 * (l2 + m2)) * shell_hoppings['dds']
                + n2 * (m2 - l2) * shell_hoppings['ddp']
                + 0.25 * (1.0 + n2) * diff_lm * shell_hoppings['ddd']
            )

        return None

    def _sk_sp_value(orb_a, orb_b, lx, ly, lz, shell_hoppings):
        if orb_a == 's' and orb_b in d_orbitals:
            return _sd_value(orb_b, lx, ly, lz, shell_hoppings)
        if orb_b == 's' and orb_a in d_orbitals:
            return _sd_value(orb_a, lx, ly, lz, shell_hoppings)
        if orb_a in p_index_map and orb_b in d_orbitals:
            return _pd_value(orb_a, orb_b, lx, ly, lz, shell_hoppings)
        if orb_b in p_index_map and orb_a in d_orbitals:
            value = _pd_value(orb_b, orb_a, lx, ly, lz, shell_hoppings)
            return -value if value is not None else None
        if orb_a in d_orbitals and orb_b in d_orbitals:
            return _dd_value(orb_a, orb_b, lx, ly, lz, shell_hoppings)

        if orb_a == 's' and orb_b == 's':
            return shell_hoppings['sss']

        if orb_a == 's' and orb_b in p_index_map:
            return (lx, ly, lz)[p_index_map[orb_b]] * shell_hoppings['sps']

        if orb_b == 's' and orb_a in p_index_map:
            return -(lx, ly, lz)[p_index_map[orb_a]] * shell_hoppings['sps']

        if orb_a == orb_b and orb_a in p_index_map:
            ll = (lx, ly, lz)[p_index_map[orb_a]]
            return ll**2 * shell_hoppings['pps'] + (1.0 - ll**2) * shell_hoppings['ppp']

        if (orb_a, orb_b) in (('px', 'py'), ('py', 'px')):
            return lx * ly * (shell_hoppings['pps'] - shell_hoppings['ppp'])

        if (orb_a, orb_b) in (('py', 'pz'), ('pz', 'py')):
            return ly * lz * (shell_hoppings['pps'] - shell_hoppings['ppp'])

        if (orb_a, orb_b) in (('px', 'pz'), ('pz', 'px')):
            return lx * lz * (shell_hoppings['pps'] - shell_hoppings['ppp'])

        return None

    def _validate_sk_tables():
        if not has_d:
            return
        if is_multishell:
            # Multi-shell: validate d-d symmetry for each group pair
            # that involves d-orbitals, in each neighbor shell.
            dd_pairs = []
            for ia in range(natoms):
                config = atom_config[ia]
                for ga in range(len(config)):
                    for gb in range(ga, len(config)):
                        la = _config_l_map[config[ga][-1].upper()]
                        lb = _config_l_map[config[gb][-1].upper()]
                        if la == 2 and lb == 2:
                            pair_key = f'{config[ga]}-{config[gb]}'
                            if pair_key not in dd_pairs:
                                dd_pairs.append(pair_key)
                break  # same config for all atoms

            test_dirs = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
                np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
            ]
            for pair_key in dd_pairs:
                for shell_name in hoppings_shells:
                    if pair_key not in hoppings_shells[shell_name]:
                        continue
                    sh = hoppings_shells[shell_name][pair_key]
                    for vec in test_dirs:
                        lx, ly, lz = vec
                        for a in d_orbitals:
                            for b in d_orbitals:
                                v_ab = _dd_value(a, b, lx, ly, lz, sh)
                                v_ba = _dd_value(b, a, lx, ly, lz, sh)
                                if v_ab is None or v_ba is None:
                                    continue
                                if not np.isclose(v_ab, v_ba, atol=1e-12, rtol=1e-12):
                                    raise ValueError(
                                        f'Inconsistent d-d SK symmetry for '
                                        f'{a},{b} ({pair_key}/{shell_name}) '
                                        f'at direction '
                                        f'({lx:.3f},{ly:.3f},{lz:.3f})'
                                    )
            return

        test_dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
            np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
        ]

        for vec in test_dirs:
            lx, ly, lz = vec
            for a in d_orbitals:
                for b in d_orbitals:
                    v_ab = _dd_value(a, b, lx, ly, lz, hoppings_shells['nn'])
                    v_ba = _dd_value(b, a, lx, ly, lz, hoppings_shells['nn'])
                    if v_ab is None or v_ba is None:
                        continue
                    if not np.isclose(v_ab, v_ba, atol=1e-12, rtol=1e-12):
                        raise ValueError(
                            'Inconsistent d-d SK symmetry for '
                            f'{a},{b} at direction ({lx:.3f},{ly:.3f},{lz:.3f})'
                        )

    if params.get('model', {}).get('validate_sk', False):
        _validate_sk_tables()

    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                for ia in range(natoms):
                    for ib in range(natoms):
                        dist_val = distance(tau[ia], sctau[ib, i, j, k, :])
                        if dist_val <= 0:
                            continue
                        if dist_val < cutoff_1:
                            shell_key = 'nn'
                        elif cutoff_2 is not None and dist_val < cutoff_2:
                            shell_key = 'nnn'
                        elif cutoff_3 is not None and dist_val < cutoff_3:
                            shell_key = 'nnnn'
                        else:
                            continue

                        lx = cosines(tau[ia], sctau[ib, i, j, k, :])[0]
                        ly = cosines(tau[ia], sctau[ib, i, j, k, :])[1]
                        lz = cosines(tau[ia], sctau[ib, i, j, k, :])[2]

                        shell_hop_data = hoppings_shells[shell_key]
                        orbitals_a = arry['shells'][ia]['orbitals']
                        orbitals_b = arry['shells'][ib]['orbitals']

                        if is_multishell:
                            # Multi-shell: route each orbital pair to
                            # the group-pair-specific SK parameters.
                            config_a = atom_config[ia]
                            config_b = atom_config[ib]
                            groups_a = atom_orbital_groups[ia]
                            groups_b = atom_orbital_groups[ib]
                            for noa, orb_a in enumerate(orbitals_a):
                                for nob, orb_b in enumerate(orbitals_b):
                                    ga = groups_a[noa]
                                    gb = groups_b[nob]
                                    cfg_ga = config_a[ga]
                                    cfg_gb = config_b[gb]
                                    if ga <= gb:
                                        pair_key = f'{cfg_ga}-{cfg_gb}'
                                    else:
                                        pair_key = f'{cfg_gb}-{cfg_ga}'
                                    pair_hoppings = shell_hop_data[pair_key]
                                    value = _sk_sp_value(
                                        orb_a,
                                        orb_b,
                                        lx,
                                        ly,
                                        lz,
                                        pair_hoppings,
                                    )
                                    if value is not None:
                                        HRs[
                                            atom_block_start[ia] + noa,
                                            atom_block_start[ib] + nob,
                                            i,
                                            j,
                                            k,
                                            0,
                                        ] = value
                        else:
                            for noa, orb_a in enumerate(orbitals_a):
                                for nob, orb_b in enumerate(orbitals_b):
                                    value = _sk_sp_value(
                                        orb_a,
                                        orb_b,
                                        lx,
                                        ly,
                                        lz,
                                        shell_hop_data,
                                    )
                                    if value is not None:
                                        HRs[
                                            atom_block_start[ia] + noa,
                                            atom_block_start[ib] + nob,
                                            i,
                                            j,
                                            k,
                                            0,
                                        ] = value

        arry['HRs'] = HRs

    # Reorder orbitals from SK convention to SOC/Abate-Asdente convention
    # SOC expects m=0,+1,-1 ordering per l-shell:
    #   p: (pz, px, py)   — SK has (px, py, pz)
    #   d: (dz2, dxz, dyz, dx2-y2, dxy) — SK has (dxy, dyz, dzx, dx2-y2, dz2)
    _sk_orb_to_soc = {
        's': 0,
        'pz': 0,
        'px': 1,
        'py': 2,
        'dz2': 0,
        'dzx': 1,
        'dyz': 2,
        'dx2-y2': 3,
        'dxy': 4,
    }
    _l_block_size = {0: 1, 1: 3, 2: 5}
    _orb_l = {
        's': 0,
        'px': 1,
        'py': 1,
        'pz': 1,
        'dxy': 2,
        'dyz': 2,
        'dzx': 2,
        'dx2-y2': 2,
        'dz2': 2,
    }
    perm = list(range(nawf))
    for ia in range(natoms):
        orbs = shells[ia]['orbitals']
        # Find starting index of each l-block within this atom
        l_start = {}  # l -> first absolute index
        idx = atom_block_start[ia]
        for orb in orbs:
            l = _orb_l[orb]
            if l not in l_start:
                l_start[l] = idx
            idx += 1
        # Build permutation for this atom
        for orb in orbs:
            l = _orb_l[orb]
            old_abs = atom_block_start[ia] + orbs.index(orb)
            new_abs = l_start[l] + _sk_orb_to_soc[orb]
            perm[new_abs] = old_abs
    arry['HRs'] = arry['HRs'][perm, :, :, :, :, :]
    arry['HRs'] = arry['HRs'][:, perm, :, :, :, :]

    # Convert shells to dict keyed by species name for SOC compatibility
    shells_dict = {}
    for sh in shells:
        name = sh['name']
        if name not in shells_dict:
            shells_dict[name] = list(dict.fromkeys(_orb_l[o] for o in sh['orbitals']))
    arry['shells'] = shells_dict


def SK_EDTB(data_controller, params):
    """Build an environment-dependent Slater-Koster tight-binding model (up to 3NN).

    Extends the two-center ``Slater_Koster`` with environment-dependent
    screening corrections in the style of Porezag & Frauenheim.

    For each bond (i, j) the two-center SK integrals are modulated by a
    scalar factor that depends on the local atomic environment:

        V_λ^eff(i,j) = V_λ^(2c) · exp( -γ_λ · S_ij )

    where S_ij = Σ_{k ≠ i,j} f_c(d_ik) · f_c(d_jk) is a screening sum
    over nearby mediating atoms *k*, and f_c is a smooth cutoff function.

    The γ_λ screening strengths can be specified:
      - per SK channel  ('sss', 'pps', …)
      - per l-pair       ('ss', 'sp', 'pp', 'sd', 'pd', 'dd')
      - as a single global value

    Additional input (on top of ``Slater_Koster`` requirements):

    params['model']['screening'] : dict
        r_cut : float
            Cutoff radius (same units as a_vectors) for mediating atoms.
        gamma : float or dict
            If float → single global screening strength.
            If dict  → per-channel {'sss': γ1, 'pps': γ2, …}
                    or per-l-pair {'ss': γ1, 'sp': γ2, …}.
        onsite_shift : dict, optional
            Per-orbital environment-dependent on-site correction strength.
            Keys are orbital labels ('s', 'p', 'd'); values are floats.
            Each on-site energy receives Δε = η · Σ_k f_c(d_ik).
    """

    import numpy as np

    arry, attr = data_controller.data_dicts()
    # Lattice Vectors
    arry['a_vectors'] = np.array(params['model']['a_vectors'])
    attr['alat'] = params.get('alat', 1.0)

    # Atomic coordinates
    natoms = len(params['model']['atoms'])
    tau = np.zeros((natoms, 3), dtype=float)
    for ia in range(natoms):
        tau[ia] = np.array(params['model']['atoms'][str(ia)]['tau'])
    atoms = []
    shells = []
    for ia in range(natoms):
        atoms.append(params['model']['atoms'][str(ia)]['name'])
        shells.append(params['model']['atoms'][str(ia)])
    arry['tau'] = tau
    arry['atoms'] = atoms
    attr['natoms'] = natoms
    arry['shells'] = shells
    attr['nspin'] = 1
    attr['dftSO'] = False

    # ── Multi-shell support ──────────────────────────────────────
    _config_l_map = {'S': 0, 'P': 1, 'D': 2, 'F': 3}
    _l_to_orbitals = {
        0: ['s'],
        1: ['px', 'py', 'pz'],
        2: ['dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2'],
    }
    _lpair_required_keys = {
        (0, 0): ['sss'],
        (0, 1): ['sps'],
        (0, 2): ['sds'],
        (1, 1): ['pps', 'ppp'],
        (1, 2): ['pds', 'pdp'],
        (2, 2): ['dds', 'ddp', 'ddd'],
    }

    is_multishell = any(
        'configuration' in params['model']['atoms'][str(ia)] for ia in range(natoms)
    )

    atom_config = []
    atom_angular_orbs = []
    atom_orbital_groups = []

    if is_multishell:
        for ia in range(natoms):
            atom_dict = params['model']['atoms'][str(ia)]
            if 'configuration' not in atom_dict:
                raise ValueError(
                    f"Atom {ia}: 'configuration' key required when multi-shell mode is used."
                )
            config = atom_dict['configuration']
            atom_config.append(config)
            ang_orbs = []
            groups = []
            for ig, cfg_label in enumerate(config):
                l_val = _config_l_map[cfg_label[-1].upper()]
                if l_val not in _l_to_orbitals:
                    raise ValueError(
                        f"Unsupported angular momentum l={l_val} for config '{cfg_label}'."
                    )
                for orb_name in _l_to_orbitals[l_val]:
                    ang_orbs.append(orb_name)
                    groups.append(ig)
            atom_angular_orbs.append(ang_orbs)
            atom_orbital_groups.append(groups)
            atom_dict['orbitals'] = ang_orbs

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    attr['omega'] = attr['alat'] ** 3 * volume
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    hoppings = params['model']['hoppings']

    # ── Species-pair-keyed format auto-detection ──────────────
    hoppings, _sp_n_shells, _sp_converted = _normalize_species_pair_hoppings(hoppings)

    # ── Distance-dependent hopping mode ──────────────────────
    _dd_mode = _sp_n_shells == -1
    _multi_species = _sp_n_shells == -2
    alat_phys = params.get('alat', 1.0)

    if _dd_mode:
        _dd_multi = len(hoppings) > 1  # multi-species DD

        if _dd_multi:
            # Store per-pair DD specs
            _dd_pairs = {}
            _max_r_c_alat = 0.0
            for pk, spec in hoppings.items():
                r_0 = spec['r_0'] / alat_phys
                r_c = spec['r_c'] / alat_phys
                n_c = spec['n_c']
                _dd_pairs[pk] = {
                    'r_0': r_0,
                    'r_c': r_c,
                    'n_c': n_c,
                    'channels': spec['channels'],
                    'spec': spec,
                }
                _max_r_c_alat = max(_max_r_c_alat, r_c)
            _dd_r_c_alat = _max_r_c_alat

            def _dd_hop_values_pair(dist_val, pair_key):
                """Evaluate Goodwin-style V_λ(r) for a specific species pair."""
                pd = _dd_pairs[pair_key]
                hop = {}
                ratio = pd['r_0'] / dist_val
                exp_arg = (
                    -((dist_val / pd['r_c']) ** pd['n_c']) + (pd['r_0'] / pd['r_c']) ** pd['n_c']
                )
                for ch_name, ch_p in pd['channels'].items():
                    V0 = ch_p['V0']
                    n_ch = ch_p['n']
                    hop[ch_name] = V0 * ratio**n_ch * np.exp(n_ch * exp_arg)
                return hop

            # For single-pair compat
            _dd_spec = next(iter(hoppings.values()))
            _dd_channels = _dd_spec['channels']
            _dd_n_c = _dd_spec['n_c']
            _dd_r_0_alat = _dd_spec['r_0'] / alat_phys
            _dd_hop_values = None  # not used in multi-species DD
        else:
            # Single species pair
            _dd_pair_key = next(iter(hoppings))
            _dd_spec = hoppings[_dd_pair_key]
            _dd_r_0_alat = _dd_spec['r_0'] / alat_phys
            _dd_r_c_alat = _dd_spec['r_c'] / alat_phys
            _dd_n_c = _dd_spec['n_c']
            _dd_channels = _dd_spec['channels']
            _dd_multi = False

            def _dd_hop_values(dist_val):
                """Evaluate Goodwin-style V_λ(r) for all SK channels."""
                hop = {}
                ratio = _dd_r_0_alat / dist_val
                exp_arg = (
                    -((dist_val / _dd_r_c_alat) ** _dd_n_c)
                    + (_dd_r_0_alat / _dd_r_c_alat) ** _dd_n_c
                )
                for ch_name, ch_p in _dd_channels.items():
                    V0 = ch_p['V0']
                    n_ch = ch_p['n']
                    hop[ch_name] = V0 * ratio**n_ch * np.exp(n_ch * exp_arg)
                return hop

        # cell_range must cover the hopping cutoff r_c
        a_norms = np.linalg.norm(arry['a_vectors'], axis=1)
        cell_range = max(1, int(np.ceil(_dd_r_c_alat / a_norms.min())) + 1)
        use_second_neighbors = False
        use_third_neighbors = False
    elif _sp_converted:
        if _multi_species:
            # Determine n_shells from the first species pair
            _first_pair_hops = next(iter(hoppings.values()))
            _eff_n_shells = len(_first_pair_hops)
            use_second_neighbors = _eff_n_shells >= 2
            use_third_neighbors = _eff_n_shells >= 3
        else:
            use_second_neighbors = _sp_n_shells >= 2
            use_third_neighbors = _sp_n_shells >= 3
    else:
        use_third_neighbors = isinstance(hoppings, dict) and 'nnnn' in hoppings
        use_second_neighbors = isinstance(hoppings, dict) and (
            'nnn' in hoppings or use_third_neighbors
        )
        if (use_second_neighbors or use_third_neighbors) and 'nn' not in hoppings:
            raise ValueError(
                'Neighbor hoppings require a "nn" block in params["model"]["hoppings"].'
            )
        if use_third_neighbors and 'nnn' not in hoppings:
            raise ValueError(
                'Third-neighbor hoppings require a "nnn" block in params["model"]["hoppings"].'
                'Third-neighbor hoppings require a "nnn" block in params["model"]["hoppings"].'
            )

    # dimensions of the supercell
    if not _dd_mode:
        if use_third_neighbors:
            cell_range = 3
        elif use_second_neighbors:
            cell_range = 2
        else:
            cell_range = 1
    nk1 = nk2 = nk3 = 2 * cell_range + 1
    nkpnts = nk1 * nk2 * nk3
    attr['nk1'] = nk1
    attr['nk2'] = nk2
    attr['nk3'] = nk3
    attr['nkpnts'] = nkpnts

    # orbital counts
    norbitals = np.zeros(natoms, dtype=int)
    for ia in range(natoms):
        norbitals[ia] = len(params['model']['atoms'][str(ia)]['orbitals'])

    nawf = 0
    for ia in range(natoms):
        nawf += norbitals[ia]
    attr['nawf'] = nawf
    attr['bnd'] = nawf
    attr['nbnds'] = nawf
    attr['shift'] = 0

    atom_block_start = np.zeros(natoms, dtype=int)
    for ia in range(1, natoms):
        atom_block_start[ia] = atom_block_start[ia - 1] + norbitals[ia - 1]

    # Dnm matrix
    basis = []
    for i in range(attr['natoms']):
        for k in range(len(arry['shells'][i]['orbitals'])):
            basis.append(arry['shells'][i]['tau'])
    arry['Dnm'] = np.empty((attr['nawf'], attr['nawf'], 3))
    for i in range(3):
        for n in range(attr['nawf']):
            for m in range(attr['nawf']):
                arry['Dnm'][n, m, i] = basis[n][i] - basis[m][i]

    # generate supercell positions
    sctau = np.zeros((natoms, nk1, nk2, nk3, 3), dtype=float)
    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                for ia in range(natoms):
                    sctau[ia, i, j, k, :] = (
                        tau[ia]
                        + i * arry['a_vectors'][0]
                        + j * arry['a_vectors'][1]
                        + k * arry['a_vectors'][2]
                    )
    sctau = np.reshape(sctau, (natoms * nk1 * nk2 * nk3, 3), order='C')

    distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    cosines = lambda x, y: (y - x) / np.sqrt(np.sum((x - y) ** 2))

    if _dd_mode:
        # Distance-dependent mode: no shell assignment needed
        cutoff_1 = _dd_r_c_alat
        cutoff_2 = None
        cutoff_3 = None
    else:
        from scipy.spatial.distance import cdist as _cdist

        dist = _cdist(tau, sctau.reshape(-1, 3)).ravel().tolist()
        unique_dist = np.sort(np.unique(dist))
        # Merge nearly-degenerate distances (e.g. from floating-point
        # precision in DFT atomic positions).  Use relative tolerance 1e-3.
        _merged = [unique_dist[0]]
        for ud in unique_dist[1:]:
            if _merged[-1] > 0 and abs(ud - _merged[-1]) / _merged[-1] < 1e-3:
                continue  # skip — same shell as previous
            _merged.append(ud)
        unique_dist = np.array(_merged)
        if unique_dist.size < 2:
            raise ValueError('Unable to determine nearest-neighbor distances for SK_EDTB model.')

        dist_1 = unique_dist[1]
        if use_third_neighbors:
            if unique_dist.size < 4:
                raise ValueError('Unable to determine third-neighbor distances for SK_EDTB model.')
            dist_2 = unique_dist[2]
            dist_3 = unique_dist[3]
            cutoff_1 = dist_1 + (dist_2 - dist_1) / 2.0
            cutoff_2 = dist_2 + (dist_3 - dist_2) / 2.0
            if unique_dist.size > 4:
                dist_4 = unique_dist[4]
                cutoff_3 = dist_3 + (dist_4 - dist_3) / 2.0
            else:
                cutoff_3 = dist_3 + (dist_3 - dist_2) / 2.0
        elif use_second_neighbors:
            if unique_dist.size < 3:
                raise ValueError('Unable to determine second-neighbor distances for SK_EDTB model.')
            dist_2 = unique_dist[2]
            cutoff_1 = dist_1 + (dist_2 - dist_1) / 2.0
            if unique_dist.size > 3:
                dist_3 = unique_dist[3]
                cutoff_2 = dist_2 + (dist_3 - dist_2) / 2.0
            else:
                cutoff_2 = dist_2 + (dist_2 - dist_1) / 2.0
            cutoff_3 = None
        else:
            if unique_dist.size < 3:
                raise ValueError('Unable to determine cutoff for first-neighbor shell.')
            cutoff_1 = dist_1 + (unique_dist[2] - dist_1) / 2.0
            cutoff_2 = None
            cutoff_3 = None
    sctau = np.reshape(sctau, (natoms, nk1, nk2, nk3, 3), order='C')

    arry['sctau'] = sctau
    attr['cutoff'] = cutoff_1
    attr['cutoff_1'] = cutoff_1
    if cutoff_2 is not None:
        attr['cutoff_2'] = cutoff_2
    if cutoff_3 is not None:
        attr['cutoff_3'] = cutoff_3
    arry['norbitals'] = norbitals

    HRs = np.zeros((nawf, nawf, nk1, nk2, nk3, 1), dtype=complex)

    # ══════════════════════════════════════════════════════════════
    #  EDTB: Environment-dependent screening
    # ══════════════════════════════════════════════════════════════

    screening = params['model'].get('screening', None)
    if screening is None:
        raise ValueError(
            "SK_EDTB requires a 'screening' block in params['model']. "
            'Use Slater_Koster instead for a pure two-center model.'
        )

    r_cut_input = screening['r_cut']
    gamma_spec = _normalize_species_pair_gamma(screening['gamma'])
    onsite_shift_spec = screening.get('onsite_shift', None)

    # Convert r_cut from physical units (Bohr) to internal lattice units.
    # The model stores a_vectors in units of alat; distances in the code
    # are therefore in alat units.  The user specifies r_cut in Bohr.
    # (alat_phys was computed earlier, above the DD-mode block.)
    r_cut = r_cut_input / alat_phys

    # Smooth cutoff: cosine taper in [0.8·r_cut, r_cut]
    r_taper = 0.8 * r_cut

    def _f_cutoff(r):
        """Smooth cutoff function: 1 for r < r_taper, cosine taper to 0 at r_cut."""
        if r <= r_taper:
            return 1.0
        if r >= r_cut:
            return 0.0
        return 0.5 * (1.0 + np.cos(np.pi * (r - r_taper) / (r_cut - r_taper)))

    # Map from SK parameter name → screening strength γ
    _sk_param_names = [
        'sss',
        'sps',
        'pps',
        'ppp',
        'sds',
        'pds',
        'pdp',
        'dds',
        'ddp',
        'ddd',
    ]
    _sk_to_lpair = {
        'sss': 'ss',
        'sps': 'sp',
        'pps': 'pp',
        'ppp': 'pp',
        'sds': 'sd',
        'pds': 'pd',
        'pdp': 'pd',
        'dds': 'dd',
        'ddp': 'dd',
        'ddd': 'dd',
    }

    def _get_gamma(sk_key, gamma_val=None):
        """Return the screening strength for a given SK parameter name.

        gamma_val overrides gamma_spec when provided (for per-pair gamma).
        """
        gv = gamma_val if gamma_val is not None else gamma_spec
        if isinstance(gv, (int, float)):
            return float(gv)
        if sk_key in gv:
            return gv[sk_key]
        lp = _sk_to_lpair.get(sk_key)
        if lp and lp in gv:
            return gv[lp]
        return 0.0

    # Check if gamma_spec is species-pair-keyed (multi-species)
    _multi_species_gamma = (
        isinstance(gamma_spec, dict) and len(gamma_spec) > 0 and '-' in next(iter(gamma_spec))
    )

    if _multi_species_gamma:
        # Build a gamma_map per species pair
        _gamma_maps = {}
        for pair_key, gval in gamma_spec.items():
            _gamma_maps[pair_key] = {k: _get_gamma(k, gval) for k in _sk_param_names}
        # Default (fallback) gamma_map — all zeros
        gamma_map = {k: 0.0 for k in _sk_param_names}
    else:
        _gamma_maps = None
        gamma_map = {k: _get_gamma(k) for k in _sk_param_names}

    # Build the flat supercell positions for screening sum
    sctau_flat = sctau.reshape(natoms * nk1 * nk2 * nk3, 3)

    # Precompute screening sums S_ij for every bond (ia, sctau[ib,i,j,k])
    # and environment-dependent on-site coordination C_i for each atom.
    #
    # S_ij = Σ_{k ≠ i,j} f_c(d_ik) · f_c(d_jk)
    #
    # We store S for every (ia, ib_sc) pair where ib_sc is the supercell index.
    # For efficiency, compute f_c(d) for all atom-supercell pairs first.

    n_sc = natoms * nk1 * nk2 * nk3

    # f_c values: f_c_table[ia, n] = f_c(|tau[ia] - sctau_flat[n]|)
    # Vectorized with cdist for performance
    from scipy.spatial.distance import cdist as _cdist

    _dist_table = _cdist(tau, sctau_flat)  # (natoms, n_sc)
    f_c_table = np.where(
        _dist_table <= r_taper,
        1.0,
        np.where(
            _dist_table >= r_cut,
            0.0,
            0.5 * (1.0 + np.cos(np.pi * (_dist_table - r_taper) / (r_cut - r_taper))),
        ),
    )
    f_c_table[_dist_table < 1e-10] = 0.0  # exclude self
    del _dist_table

    # On-site coordination number (for environment-dependent on-site shifts)
    coord_i = np.zeros(natoms, dtype=float)
    if onsite_shift_spec is not None:
        for ia in range(natoms):
            coord_i[ia] = np.sum(f_c_table[ia])

    # S_ij screening sums — precomputed via matrix multiply
    # S_all[ia, jsc] = Σ_{k ≠ j} f_c(d_ik) · f_c(d_jk)
    # Precompute f_c for all supercell-supercell atom pairs.
    _dist_sc = _cdist(sctau_flat, sctau_flat)  # (n_sc, n_sc)
    _f_c_sc = np.where(
        _dist_sc <= r_taper,
        1.0,
        np.where(
            _dist_sc >= r_cut,
            0.0,
            0.5 * (1.0 + np.cos(np.pi * (_dist_sc - r_taper) / (r_cut - r_taper))),
        ),
    )
    _f_c_sc[_dist_sc < 1e-10] = 0.0  # exclude self
    del _dist_sc
    # S_all[ia, jsc] = f_c_table[ia, :] · f_c_sc[jsc, :]
    S_all = f_c_table @ _f_c_sc.T  # (natoms, n_sc)
    del _f_c_sc
    _nk123 = nk1 * nk2 * nk3
    _nk23 = nk2 * nk3

    def _screened_hoppings(shell_hoppings, S_ij, gmap=None):
        """Return a copy of shell_hoppings with screening applied."""
        if gmap is None:
            gmap = gamma_map
        screened = {}
        for key, val in shell_hoppings.items():
            g = gmap.get(key, 0.0)
            screened[key] = val * np.exp(-g * S_ij)
        return screened

    # ══════════════════════════════════════════════════════════════

    # on-site matrix elements
    if is_multishell:
        for ia in range(natoms):
            atom_dict = params['model']['atoms'][str(ia)]
            config = atom_config[ia]
            start = atom_block_start[ia]
            idx = 0
            for ig, cfg_label in enumerate(config):
                l_val = _config_l_map[cfg_label[-1].upper()]
                norb_l = 2 * l_val + 1
                if l_val <= 1:
                    e = atom_dict[cfg_label]
                    # EDTB on-site shift
                    if onsite_shift_spec is not None:
                        orb_type = 's' if l_val == 0 else 'p'
                        eta = onsite_shift_spec.get(orb_type, 0.0)
                        e = e + eta * coord_i[ia]
                    for io in range(norb_l):
                        HRs[start + idx + io, start + idx + io, 0, 0, 0, 0] = e
                elif l_val == 2:
                    key_t2g = f'{cfg_label}_t2g'
                    key_eg = f'{cfg_label}_eg'
                    if key_t2g in atom_dict:
                        e_t2g = atom_dict[key_t2g]
                        e_eg = atom_dict[key_eg]
                    else:
                        e_t2g = e_eg = atom_dict[cfg_label]
                    if onsite_shift_spec is not None:
                        eta = onsite_shift_spec.get('d', 0.0)
                        e_t2g = e_t2g + eta * coord_i[ia]
                        e_eg = e_eg + eta * coord_i[ia]
                    for io in range(3):
                        HRs[start + idx + io, start + idx + io, 0, 0, 0, 0] = e_t2g
                    for io in range(3, 5):
                        HRs[start + idx + io, start + idx + io, 0, 0, 0, 0] = e_eg
                idx += norb_l
    else:
        for ia in range(natoms):
            for no in range(norbitals[ia]):
                orb_label = params['model']['atoms'][str(ia)]['orbitals'][no]
                e = params['model']['atoms'][str(ia)][orb_label]
                # EDTB on-site shift
                if onsite_shift_spec is not None:
                    if orb_label == 's':
                        orb_type = 's'
                    elif orb_label in ('px', 'py', 'pz'):
                        orb_type = 'p'
                    else:
                        orb_type = 'd'
                    eta = onsite_shift_spec.get(orb_type, 0.0)
                    e = e + eta * coord_i[ia]
                HRs[
                    atom_block_start[ia] + no,
                    atom_block_start[ia] + no,
                    0,
                    0,
                    0,
                    0,
                ] = e

    # hopping matrix elements
    orbital_order = ('s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2')
    p_index_map = {'px': 0, 'py': 1, 'pz': 2}
    d_orbitals = set(orbital_order[4:])
    supported_orbitals = set(orbital_order)

    for shell in arry['shells']:
        for orb in shell['orbitals']:
            if orb not in supported_orbitals:
                raise ValueError(f'Unsupported orbital label: {orb}')

    if _dd_mode:
        # ── Distance-dependent mode: validate channels ────────
        hoppings_shells = {}  # not used, but prevent NameError
        required_keys = ['sss', 'sps', 'pps', 'ppp']
        has_d = any(orb in d_orbitals for shell in arry['shells'] for orb in shell['orbitals'])
        if has_d:
            required_keys.extend(['sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd'])

        if _dd_multi:
            for pk, pd in _dd_pairs.items():
                missing_ch = [k for k in required_keys if k not in pd['channels']]
                if missing_ch:
                    raise KeyError(
                        f'Missing SK channels in distance-dependent hoppings '
                        f"for pair '{pk}': {', '.join(missing_ch)}"
                    )
        else:
            missing_ch = [k for k in required_keys if k not in _dd_channels]
            if missing_ch:
                raise KeyError(
                    f'Missing SK channels in distance-dependent hoppings: {", ".join(missing_ch)}'
                )

        elements = sorted(set(arry['atoms']))
        if isinstance(gamma_spec, (int, float)):
            gamma_str = f'{gamma_spec:.4g}'
        elif _multi_species_gamma:
            gamma_str = ', '.join(
                f'{pk}: {gv:.4g}'
                if isinstance(gv, (int, float))
                else f'{pk}: {{{", ".join(f"{kk}={vv:.4g}" for kk, vv in gv.items())}}}'
                for pk, gv in gamma_spec.items()
            )
        else:
            gamma_str = ', '.join(f'{k}={v:.4g}' for k, v in gamma_spec.items())

        if _dd_multi:
            pair_strs = ', '.join(sorted(_dd_pairs.keys()))
            print(
                f'SK_EDTB (distance-dependent, multi-species): elements={elements}, '
                f'pairs=[{pair_strs}], '
                f'channels={len(_dd_channels)}, '
                f'r_cut_screen={r_cut_input:.3f} Bohr ({r_cut:.4f} alat), '
                f'gamma=[{gamma_str}]'
            )
        else:
            print(
                f'SK_EDTB (distance-dependent): elements={elements}, '
                f'r_0={_dd_spec["r_0"]:.3f} Bohr, '
                f'r_c={_dd_spec["r_c"]:.3f} Bohr ({_dd_r_c_alat:.4f} alat), '
                f'n_c={_dd_n_c:.2f}, channels={len(_dd_channels)}, '
                f'r_cut_screen={r_cut_input:.3f} Bohr ({r_cut:.4f} alat), '
                f'gamma=[{gamma_str}]'
            )
    else:
        if _multi_species:
            # Multi-species mode: hoppings is {pair_key: {shell_tag: {sk_params}}}
            # We store the full structure and build a flat hoppings_shells from
            # any pair for validation/cutoff detection.
            _ms_hoppings = hoppings  # species-pair → {shell_tag → params}
            _first_pair_hops = next(iter(hoppings.values()))
            hoppings_shells = dict(_first_pair_hops)
        elif 'nn' in hoppings:
            hoppings_shells = {'nn': hoppings['nn']}
            if 'nnn' in hoppings:
                hoppings_shells['nnn'] = hoppings['nnn']
            if 'nnnn' in hoppings:
                hoppings_shells['nnnn'] = hoppings['nnnn']
        else:
            hoppings_shells = {'nn': hoppings}

        if use_second_neighbors and 'nnn' not in hoppings_shells:
            raise KeyError('Second-neighbor hoppings requested but no "nnn" block provided.')
        if use_third_neighbors and 'nnnn' not in hoppings_shells:
            raise KeyError('Third-neighbor hoppings requested but no "nnnn" block provided.')

        required_keys = ['sss', 'sps', 'pps', 'ppp']
        has_d = any(orb in d_orbitals for shell in arry['shells'] for orb in shell['orbitals'])
        if has_d:
            required_keys.extend(['sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd'])

        if is_multishell:
            for shell_name, shell_hop_dict in hoppings_shells.items():
                for pair_key, pair_hoppings in shell_hop_dict.items():
                    parts = pair_key.split('-')
                    if len(parts) != 2:
                        raise KeyError(
                            f"Invalid group-pair key '{pair_key}' in {shell_name}. "
                            "Expected format: 'GroupA-GroupB' (e.g. '3S-3P')."
                        )
                    la = _config_l_map[parts[0][-1].upper()]
                    lb = _config_l_map[parts[1][-1].upper()]
                    lpair = (min(la, lb), max(la, lb))
                    needed = _lpair_required_keys[lpair]
                    missing_keys = [k for k in needed if k not in pair_hoppings]
                    if missing_keys:
                        raise KeyError(
                            f"Missing SK keys for {shell_name}/'{pair_key}' "
                            f'(l-pair {lpair}): {", ".join(missing_keys)}'
                        )
        elif _multi_species:
            for pair_key, pair_shells in _ms_hoppings.items():
                for shell_name, shell_hoppings_ms in pair_shells.items():
                    missing_keys = [key for key in required_keys if key not in shell_hoppings_ms]
                    if missing_keys:
                        raise KeyError(
                            f'Missing Slater-Koster hopping keys for '
                            f'{pair_key}/{shell_name}: {", ".join(missing_keys)}'
                        )
        else:
            for shell_name, shell_hoppings in hoppings_shells.items():
                missing_keys = [key for key in required_keys if key not in shell_hoppings]
                if missing_keys:
                    raise KeyError(
                        f'Missing Slater-Koster hopping keys for '
                        f'{shell_name}: {", ".join(missing_keys)}'
                    )

        elements = sorted(set(arry['atoms']))
        shell_names = ', '.join(sorted(hoppings_shells.keys()))
        if isinstance(gamma_spec, (int, float)):
            gamma_str = f'{gamma_spec:.4g}'
        elif _multi_species_gamma:
            gamma_str = ', '.join(
                f'{pk}: {gv:.4g}'
                if isinstance(gv, (int, float))
                else f'{pk}: {{{", ".join(f"{k}={v:.4g}" for k, v in gv.items())}}}'
                for pk, gv in gamma_spec.items()
            )
        else:
            gamma_str = ', '.join(f'{k}={v:.4g}' for k, v in gamma_spec.items())
        if is_multishell:
            config_str = ', '.join(atom_config[0])
            print(
                f'SK_EDTB (multi-shell): elements={elements}, '
                f'config=[{config_str}], '
                f'neighbor_shells={len(hoppings_shells)} ({shell_names}), '
                f'r_cut={r_cut_input:.3f} Bohr ({r_cut:.4f} alat), '
                f'gamma=[{gamma_str}]'
            )
        else:
            print(
                f'SK_EDTB: elements={elements}, '
                f'neighbor_shells={len(hoppings_shells)} ({shell_names}), '
                f'r_cut={r_cut_input:.3f} Bohr ({r_cut:.4f} alat), '
                f'gamma=[{gamma_str}]'
            )

    # SK angular-integral functions (identical to Slater_Koster)
    sq3 = np.sqrt(3.0)
    hsq3 = sq3 / 2.0

    def _sd_value(d_orb, lx, ly, lz, sh):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        if d_orb == 'dxy':
            return sq3 * lx * ly * sh['sds']
        if d_orb == 'dyz':
            return sq3 * ly * lz * sh['sds']
        if d_orb == 'dzx':
            return sq3 * lz * lx * sh['sds']
        if d_orb == 'dx2-y2':
            return hsq3 * (l2 - m2) * sh['sds']
        if d_orb == 'dz2':
            return (n2 - 0.5 * (l2 + m2)) * sh['sds']
        return None

    def _pd_value(p_orb, d_orb, lx, ly, lz, sh):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        if p_orb == 'px':
            if d_orb == 'dxy':
                return sq3 * l2 * ly * sh['pds'] + ly * (1.0 - 2.0 * l2) * sh['pdp']
            if d_orb == 'dyz':
                return sq3 * lx * ly * lz * sh['pds'] - 2.0 * lx * ly * lz * sh['pdp']
            if d_orb == 'dzx':
                return sq3 * l2 * lz * sh['pds'] + lz * (1.0 - 2.0 * l2) * sh['pdp']
            if d_orb == 'dx2-y2':
                return hsq3 * lx * (l2 - m2) * sh['pds'] + lx * (1.0 - l2 + m2) * sh['pdp']
            if d_orb == 'dz2':
                return lx * (n2 - 0.5 * (l2 + m2)) * sh['pds'] - sq3 * lx * n2 * sh['pdp']
        if p_orb == 'py':
            if d_orb == 'dxy':
                return sq3 * m2 * lx * sh['pds'] + lx * (1.0 - 2.0 * m2) * sh['pdp']
            if d_orb == 'dyz':
                return sq3 * m2 * lz * sh['pds'] + lz * (1.0 - 2.0 * m2) * sh['pdp']
            if d_orb == 'dzx':
                return sq3 * lx * ly * lz * sh['pds'] - 2.0 * lx * ly * lz * sh['pdp']
            if d_orb == 'dx2-y2':
                return hsq3 * ly * (l2 - m2) * sh['pds'] - ly * (1.0 + l2 - m2) * sh['pdp']
            if d_orb == 'dz2':
                return ly * (n2 - 0.5 * (l2 + m2)) * sh['pds'] - sq3 * ly * n2 * sh['pdp']
        if p_orb == 'pz':
            if d_orb == 'dxy':
                return sq3 * lx * ly * lz * sh['pds'] - 2.0 * lx * ly * lz * sh['pdp']
            if d_orb == 'dyz':
                return sq3 * n2 * ly * sh['pds'] + ly * (1.0 - 2.0 * n2) * sh['pdp']
            if d_orb == 'dzx':
                return sq3 * n2 * lx * sh['pds'] + lx * (1.0 - 2.0 * n2) * sh['pdp']
            if d_orb == 'dx2-y2':
                return hsq3 * lz * (l2 - m2) * sh['pds'] - lz * (l2 - m2) * sh['pdp']
            if d_orb == 'dz2':
                return lz * (n2 - 0.5 * (l2 + m2)) * sh['pds'] + sq3 * lz * (l2 + m2) * sh['pdp']
        return None

    def _dd_value(d_orb_a, d_orb_b, lx, ly, lz, sh):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        lm = lx * ly
        ln = lx * lz
        mn = ly * lz
        l2m2 = l2 * m2
        l2n2 = l2 * n2
        m2n2 = m2 * n2
        diff_lm = l2 - m2

        if d_orb_a == d_orb_b == 'dxy':
            return (
                3.0 * l2m2 * sh['dds']
                + (l2 + m2 - 4.0 * l2m2) * sh['ddp']
                + (n2 + l2m2) * sh['ddd']
            )
        if d_orb_a == d_orb_b == 'dyz':
            return (
                3.0 * m2n2 * sh['dds']
                + (m2 + n2 - 4.0 * m2n2) * sh['ddp']
                + (l2 + m2n2) * sh['ddd']
            )
        if d_orb_a == d_orb_b == 'dzx':
            return (
                3.0 * l2n2 * sh['dds']
                + (l2 + n2 - 4.0 * l2n2) * sh['ddp']
                + (m2 + l2n2) * sh['ddd']
            )
        if d_orb_a == d_orb_b == 'dx2-y2':
            return (
                0.75 * diff_lm**2 * sh['dds']
                + (l2 + m2 - diff_lm**2) * sh['ddp']
                + (n2 + 0.25 * diff_lm**2) * sh['ddd']
            )
        if d_orb_a == d_orb_b == 'dz2':
            term = n2 - 0.5 * (l2 + m2)
            return (
                term**2 * sh['dds']
                + 3.0 * n2 * (l2 + m2) * sh['ddp']
                + 0.75 * (l2 + m2) ** 2 * sh['ddd']
            )

        if (d_orb_a, d_orb_b) in (('dxy', 'dyz'), ('dyz', 'dxy')):
            return (
                3.0 * lx * m2 * lz * sh['dds']
                + ln * (1.0 - 4.0 * m2) * sh['ddp']
                + ln * (m2 - 1.0) * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dxy', 'dzx'), ('dzx', 'dxy')):
            return (
                3.0 * l2 * ly * lz * sh['dds']
                + mn * (1.0 - 4.0 * l2) * sh['ddp']
                + mn * (l2 - 1.0) * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dyz', 'dzx'), ('dzx', 'dyz')):
            return (
                3.0 * ly * n2 * lx * sh['dds']
                + lm * (1.0 - 4.0 * n2) * sh['ddp']
                + lm * (n2 - 1.0) * sh['ddd']
            )

        if (d_orb_a, d_orb_b) in (('dxy', 'dx2-y2'), ('dx2-y2', 'dxy')):
            return (
                1.5 * lm * diff_lm * sh['dds']
                + 2.0 * lm * (m2 - l2) * sh['ddp']
                + 0.5 * lm * diff_lm * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dyz', 'dx2-y2'), ('dx2-y2', 'dyz')):
            return (
                1.5 * mn * diff_lm * sh['dds']
                - mn * (1.0 + 2.0 * diff_lm) * sh['ddp']
                + mn * (1.0 + 0.5 * diff_lm) * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dzx', 'dx2-y2'), ('dx2-y2', 'dzx')):
            return (
                1.5 * ln * diff_lm * sh['dds']
                + ln * (1.0 - 2.0 * diff_lm) * sh['ddp']
                - ln * (1.0 - 0.5 * diff_lm) * sh['ddd']
            )

        if (d_orb_a, d_orb_b) in (('dxy', 'dz2'), ('dz2', 'dxy')):
            return sq3 * (
                lm * (n2 - 0.5 * (l2 + m2)) * sh['dds']
                - 2.0 * lm * n2 * sh['ddp']
                + 0.5 * lm * (1.0 + n2) * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dyz', 'dz2'), ('dz2', 'dyz')):
            return sq3 * (
                mn * (n2 - 0.5 * (l2 + m2)) * sh['dds']
                + mn * (l2 + m2 - n2) * sh['ddp']
                - 0.5 * mn * (l2 + m2) * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dzx', 'dz2'), ('dz2', 'dzx')):
            return sq3 * (
                ln * (n2 - 0.5 * (l2 + m2)) * sh['dds']
                + ln * (l2 + m2 - n2) * sh['ddp']
                - 0.5 * ln * (l2 + m2) * sh['ddd']
            )
        if (d_orb_a, d_orb_b) in (('dx2-y2', 'dz2'), ('dz2', 'dx2-y2')):
            return sq3 * (
                0.5 * diff_lm * (n2 - 0.5 * (l2 + m2)) * sh['dds']
                + n2 * (m2 - l2) * sh['ddp']
                + 0.25 * (1.0 + n2) * diff_lm * sh['ddd']
            )

        return None

    def _sk_sp_value(orb_a, orb_b, lx, ly, lz, sh):
        if orb_a == 's' and orb_b in d_orbitals:
            return _sd_value(orb_b, lx, ly, lz, sh)
        if orb_b == 's' and orb_a in d_orbitals:
            return _sd_value(orb_a, lx, ly, lz, sh)
        if orb_a in p_index_map and orb_b in d_orbitals:
            return _pd_value(orb_a, orb_b, lx, ly, lz, sh)
        if orb_b in p_index_map and orb_a in d_orbitals:
            value = _pd_value(orb_b, orb_a, lx, ly, lz, sh)
            return -value if value is not None else None
        if orb_a in d_orbitals and orb_b in d_orbitals:
            return _dd_value(orb_a, orb_b, lx, ly, lz, sh)

        if orb_a == 's' and orb_b == 's':
            return sh['sss']
        if orb_a == 's' and orb_b in p_index_map:
            return (lx, ly, lz)[p_index_map[orb_b]] * sh['sps']
        if orb_b == 's' and orb_a in p_index_map:
            return -(lx, ly, lz)[p_index_map[orb_a]] * sh['sps']

        if orb_a == orb_b and orb_a in p_index_map:
            ll = (lx, ly, lz)[p_index_map[orb_a]]
            return ll**2 * sh['pps'] + (1.0 - ll**2) * sh['ppp']

        if (orb_a, orb_b) in (('px', 'py'), ('py', 'px')):
            return lx * ly * (sh['pps'] - sh['ppp'])
        if (orb_a, orb_b) in (('py', 'pz'), ('pz', 'py')):
            return ly * lz * (sh['pps'] - sh['ppp'])
        if (orb_a, orb_b) in (('px', 'pz'), ('pz', 'px')):
            return lx * lz * (sh['pps'] - sh['ppp'])

        return None

    # ── Hopping loop with EDTB screening ─────────────────────
    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                for ia in range(natoms):
                    for ib in range(natoms):
                        pos_j = sctau[ib, i, j, k, :]
                        dist_val = distance(tau[ia], pos_j)
                        if dist_val <= 0:
                            continue

                        if _dd_mode:
                            # Distance-dependent: accept all bonds within r_c
                            if dist_val > _dd_r_c_alat:
                                continue
                        else:
                            # Discrete-shell: assign shell by cutoff
                            if dist_val < cutoff_1:
                                shell_key = 'nn'
                            elif cutoff_2 is not None and dist_val < cutoff_2:
                                shell_key = 'nnn'
                            elif cutoff_3 is not None and dist_val < cutoff_3:
                                shell_key = 'nnnn'
                            else:
                                continue

                        _dir = cosines(tau[ia], pos_j)
                        lx, ly, lz = _dir[0], _dir[1], _dir[2]

                        # EDTB: look up precomputed screening sum
                        _jsc = ib * _nk123 + (i % nk1) * _nk23 + (j % nk2) * nk3 + (k % nk3)
                        S_ij = S_all[ia, _jsc]

                        if _dd_mode:
                            if _dd_multi:
                                sp_a, sp_b = sorted([atoms[ia], atoms[ib]])
                                _pair_key = f'{sp_a}-{sp_b}'
                                pd = _dd_pairs[_pair_key]
                                if dist_val > pd['r_c']:
                                    continue
                                _pair_gmap = (
                                    _gamma_maps.get(_pair_key, gamma_map)
                                    if _gamma_maps
                                    else gamma_map
                                )
                                shell_hop_data = _screened_hoppings(
                                    _dd_hop_values_pair(dist_val, _pair_key), S_ij, gmap=_pair_gmap
                                )
                            else:
                                shell_hop_data = _screened_hoppings(_dd_hop_values(dist_val), S_ij)
                        elif _multi_species:
                            # Look up species pair key
                            sp_a, sp_b = sorted([atoms[ia], atoms[ib]])
                            _pair_key = f'{sp_a}-{sp_b}'
                            _pair_gmap = (
                                _gamma_maps.get(_pair_key, gamma_map) if _gamma_maps else gamma_map
                            )
                            shell_hop_data = _screened_hoppings(
                                _ms_hoppings[_pair_key][shell_key], S_ij, gmap=_pair_gmap
                            )
                        else:
                            shell_hop_data = _screened_hoppings(hoppings_shells[shell_key], S_ij)

                        orbitals_a = arry['shells'][ia]['orbitals']
                        orbitals_b = arry['shells'][ib]['orbitals']

                        if is_multishell and not _dd_mode:
                            config_a = atom_config[ia]
                            config_b = atom_config[ib]
                            groups_a = atom_orbital_groups[ia]
                            groups_b = atom_orbital_groups[ib]
                            for noa, orb_a in enumerate(orbitals_a):
                                for nob, orb_b in enumerate(orbitals_b):
                                    ga = groups_a[noa]
                                    gb = groups_b[nob]
                                    cfg_ga = config_a[ga]
                                    cfg_gb = config_b[gb]
                                    if ga <= gb:
                                        pair_key = f'{cfg_ga}-{cfg_gb}'
                                    else:
                                        pair_key = f'{cfg_gb}-{cfg_ga}'
                                    # Screen the group-pair hoppings
                                    pair_hoppings = _screened_hoppings(
                                        hoppings_shells[shell_key][pair_key], S_ij
                                    )
                                    value = _sk_sp_value(
                                        orb_a,
                                        orb_b,
                                        lx,
                                        ly,
                                        lz,
                                        pair_hoppings,
                                    )
                                    if value is not None:
                                        HRs[
                                            atom_block_start[ia] + noa,
                                            atom_block_start[ib] + nob,
                                            i,
                                            j,
                                            k,
                                            0,
                                        ] = value
                        else:
                            for noa, orb_a in enumerate(orbitals_a):
                                for nob, orb_b in enumerate(orbitals_b):
                                    value = _sk_sp_value(
                                        orb_a,
                                        orb_b,
                                        lx,
                                        ly,
                                        lz,
                                        shell_hop_data,
                                    )
                                    if value is not None:
                                        HRs[
                                            atom_block_start[ia] + noa,
                                            atom_block_start[ib] + nob,
                                            i,
                                            j,
                                            k,
                                            0,
                                        ] = value

    arry['HRs'] = HRs

    # Reorder orbitals from SK convention to SOC/Abate-Asdente convention
    # SOC expects m=0,+1,-1 ordering per l-shell:
    #   p: (pz, px, py)   — SK has (px, py, pz)
    #   d: (dz2, dxz, dyz, dx2-y2, dxy) — SK has (dxy, dyz, dzx, dx2-y2, dz2)
    _sk_orb_to_soc = {
        's': 0,
        'pz': 0,
        'px': 1,
        'py': 2,
        'dz2': 0,
        'dzx': 1,
        'dyz': 2,
        'dx2-y2': 3,
        'dxy': 4,
    }
    _l_block_size = {0: 1, 1: 3, 2: 5}
    _orb_l = {
        's': 0,
        'px': 1,
        'py': 1,
        'pz': 1,
        'dxy': 2,
        'dyz': 2,
        'dzx': 2,
        'dx2-y2': 2,
        'dz2': 2,
    }
    perm = list(range(nawf))
    for ia in range(natoms):
        orbs = shells[ia]['orbitals']
        # Find starting index of each l-block within this atom
        l_start = {}  # l -> first absolute index
        idx = atom_block_start[ia]
        for orb in orbs:
            l = _orb_l[orb]
            if l not in l_start:
                l_start[l] = idx
            idx += 1
        # Build permutation for this atom
        for orb in orbs:
            l = _orb_l[orb]
            old_abs = atom_block_start[ia] + orbs.index(orb)
            new_abs = l_start[l] + _sk_orb_to_soc[orb]
            perm[new_abs] = old_abs
    arry['HRs'] = arry['HRs'][perm, :, :, :, :, :]
    arry['HRs'] = arry['HRs'][:, perm, :, :, :, :]

    # Convert shells to dict keyed by species name for SOC compatibility
    shells_dict = {}
    for sh in shells:
        name = sh['name']
        if name not in shells_dict:
            shells_dict[name] = list(dict.fromkeys(_orb_l[o] for o in sh['orbitals']))
    arry['shells'] = shells_dict


def graphene(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 1

    attr['bnd'] = 4
    attr['shift'] = 100.0
    attr['Efermi'] = 0.0
    attr['nspin'] = 1
    attr['nawf'] = 2
    attr['nspin'] = 1
    attr['natoms'] = 2

    attr['alat'] = 2.46 * ANGSTROM_AU

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H00
    arry['HRs'][0, 1, 0, 0, 0, 0] = params['t']
    arry['HRs'][1, 0, 0, 0, 0, 0] = params['t']

    # H10
    arry['HRs'][1, 0, 1, 0, 0, 0] = params['t']

    # H20
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H01
    arry['HRs'][1, 0, 0, 1, 0, 0] = params['t']

    # H02
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry['a_vectors'] = arry['a_vectors']

    # Atomic coordinates
    arry['tau'] = np.zeros((2, 3), dtype=float)

    arry['tau'][0, 0] = 0.50000
    arry['tau'][0, 1] = 0.28867
    arry['tau'][1, 0] = 1.00000
    arry['tau'][1, 1] = 0.57735

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['C', 'C']


def graphene2(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 1

    attr['nawf'] = 2
    attr['nspin'] = 1
    attr['natoms'] = 2

    arry['naw'] = np.array([1, 1])

    attr['alat'] = 2.46 * ANGSTROM_AU

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H00
    arry['HRs'][0, 0, 0, 0, 0, 0] = params['delta'] / 2
    arry['HRs'][1, 1, 0, 0, 0, 0] = -params['delta'] / 2

    # H00
    arry['HRs'][0, 1, 0, 0, 0, 0] = params['t']
    arry['HRs'][1, 0, 0, 0, 0, 0] = params['t']

    # H10
    arry['HRs'][1, 0, 1, 0, 0, 0] = params['t']

    # H20
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H01
    arry['HRs'][1, 0, 0, 1, 0, 0] = params['t']

    # H02
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry['a_vectors'] = arry['a_vectors']

    # Atomic coordinates
    arry['tau'] = np.zeros((2, 3), dtype=float)

    arry['tau'][0, 0] = 0.50000
    arry['tau'][0, 1] = 0.28867
    arry['tau'][1, 0] = 1.00000
    arry['tau'][1, 1] = 0.57735

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['C', 'C']


def cubium(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 3
    attr['Efermi'] = 6 * params['t']
    attr['nawf'] = 1
    attr['nspin'] = 1
    attr['natoms'] = 1
    attr['bnd'] = 1
    attr['shift'] = 0
    attr['dftSO'] = False
    attr['nkpnts'] = attr['nk1'] * attr['nk2'] * attr['nk3']
    attr['nbnds'] = 1
    attr['nelec'] = 2

    attr['alat'] = 1.0 * ANGSTROM_AU
    attr['omega'] = attr['alat'] ** 3

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H000
    arry['HRs'][0, 0, 0, 0, 0, 0] = 0.0 - attr['Efermi']

    # H100
    arry['HRs'][0, 0, 1, 0, 0, 0] = params['t']

    # H200
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H010
    arry['HRs'][0, 0, 0, 1, 0, 0] = params['t']

    # H020
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # H001
    arry['HRs'][0, 0, 0, 0, 1, 0] = params['t']

    # H002
    arry['HRs'][:, :, 0, 0, 2, 0] = np.conj(arry['HRs'][:, :, 0, 0, 1, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Atomic coordinates
    arry['tau'] = np.zeros((1, 3), dtype=float)

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['Cu']


def cubium2(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 3

    attr['nawf'] = 2
    attr['nspin'] = 1
    attr['natoms'] = 1
    attr['bnd'] = 2
    attr['shift'] = 0
    attr['dftSO'] = False
    attr['nkpnts'] = attr['nk1'] * attr['nk2'] * attr['nk3']
    attr['nbnds'] = 2
    attr['nelec'] = 2
    attr['alat'] = 1.0 * ANGSTROM_AU
    attr['omega'] = attr['alat'] ** 3

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H000
    arry['HRs'][0, 0, 0, 0, 0, 0] = -params['Eg'] / 2 - 6.0 * params['t']
    arry['HRs'][1, 1, 0, 0, 0, 0] = params['Eg'] / 2 + 6.0 * params['t']

    # H100
    arry['HRs'][0, 0, 1, 0, 0, 0] = params['t']
    arry['HRs'][1, 1, 1, 0, 0, 0] = -params['t']

    # H200
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H010
    arry['HRs'][0, 0, 0, 1, 0, 0] = params['t']
    arry['HRs'][1, 1, 0, 1, 0, 0] = -params['t']

    # H020
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # H001
    arry['HRs'][0, 0, 0, 0, 1, 0] = params['t']
    arry['HRs'][1, 1, 0, 0, 1, 0] = -params['t']

    # H002
    arry['HRs'][:, :, 0, 0, 2, 0] = np.conj(arry['HRs'][:, :, 0, 0, 1, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Atomic coordinates
    arry['tau'] = np.zeros((1, 3), dtype=float)

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['Cu']


def Kane_Mele(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 1

    attr['nawf'] = 4
    attr['bnd'] = 4
    attr['shift'] = 100.0
    attr['Efermi'] = 0.0
    attr['nspin'] = 1
    attr['natoms'] = 2

    arry['naw'] = [2, 2]

    if 'alat' not in params:
        alat = 1.0
    else:
        alat = params['alat']

    attr['alat'] = alat * ANGSTROM_AU

    t = params['t']
    soc_par = params['soc_par']
    r_par = params['r_par']
    v_par = params['v_par']

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H00
    arry['HRs'][0, 0, 0, 0, 0, 0] = t * v_par
    arry['HRs'][1, 1, 0, 0, 0, 0] = t * v_par
    arry['HRs'][2, 2, 0, 0, 0, 0] = -t * v_par
    arry['HRs'][3, 3, 0, 0, 0, 0] = -t * v_par

    # H00
    arry['HRs'][0, 2, 0, 0, 0, 0] = t
    arry['HRs'][1, 3, 0, 0, 0, 0] = t
    arry['HRs'][2, 0, 0, 0, 0, 0] = t
    arry['HRs'][3, 1, 0, 0, 0, 0] = t

    # H10
    arry['HRs'][2, 0, 1, 0, 0, 0] = t
    arry['HRs'][3, 1, 1, 0, 0, 0] = t

    arry['HRs'][0, 0, 1, 0, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][1, 1, 1, 0, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][2, 2, 1, 0, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][3, 3, 1, 0, 0, 0] = -complex(0.0, soc_par)

    ##H20
    # arry['HRs'][:,:,2,0,0,0] = np.conj(arry['HRs'][:,:,1,0,0,0]).T

    # H01
    arry['HRs'][2, 0, 0, 1, 0, 0] = t
    arry['HRs'][3, 1, 0, 1, 0, 0] = t

    arry['HRs'][0, 0, 0, 1, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][1, 1, 0, 1, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][2, 2, 0, 1, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][3, 3, 0, 1, 0, 0] = complex(0.0, soc_par)

    ##H02
    # arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

    # H21
    arry['HRs'][0, 0, 2, 1, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][1, 1, 2, 1, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][2, 2, 2, 1, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][3, 3, 2, 1, 0, 0] = -complex(0.0, soc_par)

    ##H12
    ##arry['HRs'][:,:,1,2,0,0] = np.conj(arry['HRs'][:,:,2,1,0,0]).T

    r3h = np.sqrt(3.0) / 2.0

    arry['HRs'][0, 3, 0, 0, 0, 0] += r_par * complex(
        -r3h, 0.5
    )  # 1j * r_par * (0.5 * 1 - r3h * -1j)
    arry['HRs'][1, 2, 0, 0, 0, 0] += r_par * complex(r3h, 0.5)  # 1j * r_par * (0.5 * 1 - r3h * 1j)
    arry['HRs'][3, 0, 0, 0, 0, 0] += r_par * complex(-r3h, -0.5)
    arry['HRs'][2, 1, 0, 0, 0, 0] += r_par * complex(r3h, -0.5)

    arry['HRs'][0, 3, 1, 0, 0, 0] += -r_par * complex(
        r3h, 0.5
    )  # -1j * r_par * (0.5 * 1 + r3h * -1j)
    arry['HRs'][1, 2, 1, 0, 0, 0] += -r_par * complex(
        -r3h, 0.5
    )  # -1j * r_par * (0.5 * 1 + r3h * 1j)

    arry['HRs'][0, 3, 0, 1, 0, 0] += complex(0.0, r_par)  # -1j * r_par * -1 * 1
    arry['HRs'][1, 2, 0, 1, 0, 0] += complex(0.0, r_par)  # -1j * r_par * -1 * 1

    # H02
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T
    # H20
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T
    # H12
    arry['HRs'][:, :, 1, 2, 0, 0] = np.conj(arry['HRs'][:, :, 2, 1, 0, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry['a_vectors'] = arry['a_vectors']

    # Spin properties
    arry['Sj'] = np.zeros((3, 4, 4), dtype=complex)

    arry['Sj'][2, 0, 0] = 0.5
    arry['Sj'][2, 1, 1] = -0.5
    arry['Sj'][2, 2, 2] = 0.5
    arry['Sj'][2, 3, 3] = -0.5

    arry['Sj'][0, 0, 1] = 0.5
    arry['Sj'][0, 1, 0] = 0.5
    arry['Sj'][0, 2, 3] = 0.5
    arry['Sj'][0, 3, 2] = 0.5

    arry['Sj'][1, 0, 1] = -complex(0.0, 0.5)
    arry['Sj'][1, 1, 0] = complex(0.0, 0.5)
    arry['Sj'][1, 2, 3] = -complex(0.0, 0.5)
    arry['Sj'][1, 3, 2] = complex(0.0, 0.5)
    # Atomic coordinates
    arry['tau'] = np.zeros((2, 3), dtype=float)
    arry['tau'][0] = np.dot([1 / 3, 1 / 3, 0.0], arry['a_vectors'])
    arry['tau'][1] = np.dot([2 / 3, 2 / 3, 0.0], arry['a_vectors'])

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    attr['omega'] = alat**3 * arry['a_vectors'][0, :].dot(
        np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])
    )

    arry['species'] = ['KM', 'KM']


def build_TB_model(data_controller, parameters):
    if parameters['label'].upper() == 'GRAPHENE':
        graphene(data_controller, parameters)
    elif parameters['label'].upper() == 'GRAPHENE2':
        graphene2(data_controller, parameters)
    elif parameters['label'].upper() == 'CUBIUM':
        cubium(data_controller, parameters)
    elif parameters['label'].upper() == 'CUBIUM2':
        cubium2(data_controller, parameters)
    elif parameters['label'].upper() == 'KANE_MELE':
        Kane_Mele(data_controller, parameters)
    elif parameters['label'].upper() == 'SLATER_KOSTER':
        Slater_Koster(data_controller, parameters)
    elif parameters['label'].upper() == 'SK_EDTB':
        SK_EDTB(data_controller, parameters)
    else:
        print(f'ERROR: Label "{parameters["label"]}" not found in builtin models.')
