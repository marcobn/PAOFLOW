"""Tight-binding models of PAOFLOW: Slater-Koster and EDTB builders, PythTB models, hard-coded
models and the hopping-list models used to validate the optical responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ..DataController import DataController

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

    # n_sc = natoms * nk1 * nk2 * nk3

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
        """Return a copy of shell_hoppings with screening applied.

        Handles both the flat single-configuration format
        (``{'sss': val, ...}``) and the multi-configuration shell-pair
        format (``{'3S-3P': {'sps': val, ...}, ...}``), where each shell
        pair carries its own SK sub-dict.
        """
        if gmap is None:
            gmap = gamma_map
        screened = {}
        for key, val in shell_hoppings.items():
            if isinstance(val, dict):
                # Multi-shell: key is a shell-pair ('3S-3P'), val an SK dict.
                screened[key] = {
                    sk: sv * np.exp(-gmap.get(sk, 0.0) * S_ij) for sk, sv in val.items()
                }
            else:
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
                                    _dd_hop_values_pair(dist_val, _pair_key),
                                    S_ij,
                                    gmap=_pair_gmap,
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
                                _ms_hoppings[_pair_key][shell_key],
                                S_ij,
                                gmap=_pair_gmap,
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Predefined PythTB models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def cubium_pythtb(t):
    from pythtb import Lattice, TBModel

    cell = np.eye(3)
    orb = [[0, 0, 0]]
    lat = Lattice(cell, orb, periodic_dirs='all')
    model = TBModel(lattice=lat, spinful=False)
    # on-site
    onsite = -6 * t
    model.set_onsite(onsite, ind_i=0)
    # hopping
    model.set_hop(t, 0, 0, [1, 0, 0])
    model.set_hop(t, 0, 0, [0, 1, 0])
    model.set_hop(t, 0, 0, [0, 0, 1])

    return model


def cubium2_pythtb(t, Eg):
    from pythtb import Lattice, TBModel

    cell = np.eye(3)
    orb = [[0, 0, 0], [0, 0, 0]]
    lat = Lattice(cell, orb, periodic_dirs='all')
    model = TBModel(lattice=lat, spinful=False)
    # on-site
    onsite = -Eg / 2 - 6.0 * t
    model.set_onsite(onsite, ind_i=0)
    model.set_onsite(-onsite, ind_i=1)
    # hopping
    model.set_hop(t, 0, 0, [1, 0, 0])
    model.set_hop(t, 0, 0, [0, 1, 0])
    model.set_hop(t, 0, 0, [0, 0, 1])
    model.set_hop(-t, 1, 1, [1, 0, 0])
    model.set_hop(-t, 1, 1, [0, 1, 0])
    model.set_hop(-t, 1, 1, [0, 0, 1])

    return model


def ssh_pythtb(v, w):
    from pythtb import Lattice, TBModel

    r"""Su-Schrieffer-Heeger (SSH) model.

    This function constructs the SSH model with the specified hopping parameters.
    The SSH model is a one-dimensional tight-binding model that describes a chain of atoms
    with alternating hopping parameters. The tight-binding Hamiltonian for the SSH model can be
    written as:

    .. math::
       H = v \sum_{i} (c_{i, 1}^{\dagger} c_{i, 2} + \text{h.c.}) + w \sum_{i} (c_{i, 2}^{\dagger} c_{i+1, 1} + \text{h.c.})


    Parameters
    ----------
    v : float
        The intercell hopping within the unit cell.
    w : float
        The intracell hopping to neighboring unit cells.

    Returns
    -------
    TBModel
        The tight-binding model for the SSH lattice.
    """
    lat_vecs = [[1]]
    orb_vecs = [[0], [1 / 2]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0])

    model = TBModel(lattice=lat, spinful=False)

    model.set_hop(v, 0, 1, [0])
    model.set_hop(w, 1, 0, [1])  # B of cell i to A of cell i + 1

    return model


def checkerboard_pythtb(delta, t):
    from pythtb import Lattice, TBModel

    r"""Checkerboard tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a checkerboard tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D square
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (1, 0), \quad \mathbf{a}_2 = (0, 1)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \left(0, 0\right), \quad \mathbf{\tau}_2 = \left(\frac{1}{2}, \frac{1}{2}\right)

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.}) + \Delta \sum_i n_i

    Parameters
    ----------
    t : float
        Nearest neighbor hopping amplitude.

    delta : float
        On-site energy. Positive for one sublattice, negative for the other.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat_vecs = [[1, 0], [0, 1]]
    orb_vecs = [[0, 0], [1 / 2, 1 / 2]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    model = TBModel(lattice=lat, spinful=False)

    # set on-site energies
    model.set_onsite([-delta, delta], mode='set')

    model.set_hop(t, 1, 0, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])
    model.set_hop(t, 1, 0, [1, 1])

    return model


def graphene_pythtb(delta: float, t: float):
    from pythtb import Lattice, TBModel

    r"""Graphene tight-binding model.

    This function creates a graphene tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = a(1, 0), \quad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right),

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{\tau}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = \Delta \sum_i n_i + t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.})

    Parameters
    ----------
    delta : float
        On-site energy difference between the two orbitals.
    t : float
        Hopping parameter between nearest neighbor orbitals.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    model = TBModel(lattice=lat, spinful=False)

    model.set_onsite([delta / 2, -delta / 2])
    model.set_hop(t, 0, 1, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])

    return model


def haldane_pythtb(delta: float, t1: float, t2: float, phi: float = np.pi / 2):
    from pythtb import Lattice, TBModel

    r"""Haldane tight-binding model.

    This function creates a Haldane tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (1, 0), \quad \mathbf{a}_2 = \left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{\tau}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = \Delta \sum_i (-)^i c_i^\dagger c_i + t_1 \sum_{\langle i,j \rangle} (c_i^\dagger c_j
        + \text{h.c.}) + t_2 \sum_{\langle\langle i,j \rangle\rangle} (ic_i^\dagger c_j + \text{h.c.})

    Parameters
    ----------
    delta : float
        Onsite mass term. Opposite sign for the two sublattices.
    t1 : float
        Nearest neighbor hopping amplitude.
    t2 : float
        Next-nearest neighbor hopping amplitude. Peierls phase is included.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Haldane model describes a two-dimensional topological insulator with a
    non-trivial band structure. It is characterized by a finite Chern number
    and exhibits edge states that are protected by time-reversal symmetry [haldane]_.

    References
    ----------
    .. [haldane] Haldane, F. D. M. (1988).
        O(3) Nonlinear :math:`\sigma` Model and the Quantum Hall Effect in Two Dimensions.
        *Physical Review Letters*, 61(20), 2015–2018.
    """

    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
    model = TBModel(lattice=lat, spinful=False)

    model.set_onsite([-delta, delta], mode='set')

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t1, 0, 1, lvec, mode='set')

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2 * np.exp(1j * phi), 0, 0, lvec, mode='set')
    for lvec in ([-1, 0], [1, -1], [0, 1]):
        model.set_hop(t2 * np.exp(1j * phi), 1, 1, lvec, mode='set')

    return model


def kane_mele_pythtb(delta, t, soc, rashba):
    from pythtb import Lattice, TBModel

    r"""Kane-Mele tight-binding model.

    This function creates a Kane-Mele tight-binding model with the specified
    parameters. The model is defined on a 2D honeycomb lattice with two sublattices.
    The lattice vectors are given by:

    .. math::

        \mathbf{a}_1 = a(1, 0), \quad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right),

    and the orbital positions are given by:

    .. math::

        \mathbf{r}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{r}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The Hamiltonian in second-quantized form is given by:

    .. math::

        H = \Delta \sum_{i} c_i^\dagger c_i +
        t \sum_{\langle i,j \rangle} ( c_i^\dagger c_j + h.c.) +
        \lambda_{SO} \sum_{\langle \langle i,j \rangle \rangle} ( c_i^\dagger \sigma_z c_j + \text{h.c.}) + \\
        \lambda_{R} \sum_{\langle i,j \rangle} ( c_i^\dagger \mathbf{\sigma} \times
        \mathbf{\hat{d}}_{\langle i,j \rangle} c_j + \text{h.c.})

    Parameters
    ----------
    onsite : float
        On-site energy.
    t : float, complex
        Hopping parameter.
    soc : float, complex
        Spin-orbit coupling strength.
    rashba : float, complex
        Rashba coupling strength.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Kane-Mele model describes a two-dimensional topological insulator with spin-orbit coupling.
    It is defined on a honeycomb lattice and includes both intrinsic and Rashba spin-orbit coupling [kane-mele]_.

    References
    ----------
    .. [kane-mele] Kane, C. L., & Mele, E. J. (2005). Quantum Spin Hall Effect in Graphene. *Physical Review Letters*, 95(22), 226801.
    """

    # define lattice vectors
    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    # define coordinates of orbitals
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    # make two dimensional tight-binding Kane-Mele model
    ret_model = TBModel(lattice=lat, spinful=True)

    # set on-site energies
    ret_model.set_onsite([delta, -delta])

    # useful definitions
    sigma_x = np.array([0, 1, 0, 0])
    sigma_y = np.array([0, 0, 1, 0])
    sigma_z = np.array([0, 0, 0, 1])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])

    # spin-independent first-neighbor hoppings
    ret_model.set_hop(t, 0, 1, [0, 0])
    ret_model.set_hop(t, 0, 1, [0, -1])
    ret_model.set_hop(t, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    nnn_hop = 1j * soc * sigma_z
    ret_model.set_hop(nnn_hop, 0, 0, [0, 1])
    ret_model.set_hop(-nnn_hop, 0, 0, [1, 0])
    ret_model.set_hop(nnn_hop, 0, 0, [1, -1])
    ret_model.set_hop(-nnn_hop, 1, 1, [0, 1])
    ret_model.set_hop(nnn_hop, 1, 1, [1, 0])
    ret_model.set_hop(-nnn_hop, 1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)

    # bond unit vectors are (np.sqrt(3) / 2, 1/2) then (0,-1) then (-np.sqrt(3) / 2, 1/2)
    ret_model.set_hop(
        1j * rashba * ((1 / 2) * sigma_x - (np.sqrt(3) / 2) * sigma_y),
        0,
        1,
        [0, 0],
        mode='add',
    )
    ret_model.set_hop(1j * rashba * -sigma_x, 0, 1, [0, -1], mode='add')
    ret_model.set_hop(
        1j * rashba * ((1 / 2) * sigma_x + (np.sqrt(3) / 2) * sigma_y),
        0,
        1,
        [-1, 0],
        mode='add',
    )

    return ret_model


def fu_kane_mele_pythtb(t, soc, dt=[0, 0, 0, 0]):
    from pythtb import Lattice, TBModel

    r"""Fu-Kane-Mele tight-binding model.

    This function creates a Fu-Kane-Mele tight-binding model on a diamond
    lattice. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (0, 1, 1), \quad \mathbf{a}_2 = (1, 0, 1),
        \quad \mathbf{a}_3 = (1, 1, 0)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = (0, 0, 0),
        \quad \mathbf{\tau}_2 = \frac{1}{4} \mathbf{a}_1 + \frac{1}{4} \mathbf{a}_2
        + \frac{1}{4} \mathbf{a}_3

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = t \sum_{\langle ij \rangle} c_i^{\dagger} c_j
        + i \lambda_{SO} \sum_{\langle\langle ij \rangle\rangle} c_i^{\dagger}
        \vec{\sigma} \cdot (\mathbf{d}_{ij}^{1} \times \mathbf{d}_{ij}^{2}) c_j

    where the first term is a nearest-neighbor hopping term connecting the two fcc sublattices
    of the diamond lattice, and the second term is a spin-orbit coupling term connecting
    second-neighbor sites within the same sublattice. Here, :math:`\mathbf{d}_{ij}^{1,2}`
    are the two nearest-neighbor bond vectors connecting sites :math:`i` and :math:`j`.

    Due to inversion symmetry, each band is doubly degenerate. The degeneracy is lifted by symmetry
    lowering perturbations of the four nearest-neighbor hoppings :math:`t \rightarrow t + \delta t_p`
    with :math:`p = 1, 2, 3, 4` indexing the four bonds connected to each site.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    t : float
        Spin-independent nearest-neighbor hopping amplitude.
    soc : float
        Spin-orbit coupling strength. Modulates next-nearest neighbor
        hopping amplitudes.
    dt : list[float, float, float, float], optional
        Offsets added to the four nearest-neighbor hoppings along the
        bonds connected to each site. The entries are applied in the
        following order:

        - `dt[0]` : bond along ``R = [0, 0, 0]``
        - `dt[1]` : bond along ``R = [-1, 0, 0]``
        - `dt[2]` : bond along ``R = [0, -1, 0]``
        - `dt[3]` : bond along ``R = [0, 0, -1]``

        The default is ``[0, 0, 0, 0]``, which corresponds to uniform
        hopping amplitudes. This parameter allows for symmetry-lowering
        perturbations to the nearest-neighbor hoppings.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    - The Fu-Kane-Mele model describes a three-dimensional topological insulator with a
      non-trivial band structure. It is characterized by a strong :math:`\mathbb{Z}_2` invariant
      and exhibits surface Dirac cones that are protected by time-reversal and inversion
      symmetry [1]_.

    References
    ----------
    .. [1] \ L. Fu, C. L. Kane, and E. J. Mele, *Phys. Rev. Lett.*, **98**, 106803
        (2007).
    """

    lat_vecs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    orb_vecs = [[0, 0, 0], [0.25, 0.25, 0.25]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1, 2])

    model = TBModel(lattice=lat, spinful=True)

    # spin-independent first-neighbor hops
    for idx, lvec in enumerate([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
        model.set_hop(t + dt[idx], 0, 1, lvec)

    # spin-dependent second-neighbor hops
    lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
    dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])
    for j in range(6):
        spin = np.array([0.0] + dir_list[j])
        model.set_hop(1j * soc * spin, 0, 0, lvec_list[j])
        model.set_hop(-1j * soc * spin, 1, 1, lvec_list[j])

    return model


# Predefined hard-coded models


def graphene(data_controller, params):
    from ..utils.constants import ANGSTROM_AU

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

    # Inter-cell nearest-neighbour hoppings.  PAOFLOW's convention is
    # H(k) = sum_n exp(-2 pi i k.n) HRs[..., n], so HRs[i, j, n] couples orbital i in the
    # home cell to orbital j in cell -n: with A at (1/3, 1/3) and B at (2/3, 2/3) the two
    # other B -> A bonds reach A in the cells +a_1 and +a_2 (n = (-1, 0, 0), (0, -1, 0)),
    # which keeps the real-space cells consistent with 'tau' (needed by the position
    # operator of the non-linear optics; the band structure does not depend on it).
    # H(-1,0,0) ~ index 2
    arry['HRs'][1, 0, 2, 0, 0, 0] = params['t']

    # H(1,0,0) ~ index 1
    arry['HRs'][:, :, 1, 0, 0, 0] = np.conj(arry['HRs'][:, :, 2, 0, 0, 0]).T

    # H(0,-1,0) ~ index 2
    arry['HRs'][1, 0, 0, 2, 0, 0] = params['t']

    # H(0,1,0) ~ index 1
    arry['HRs'][:, :, 0, 1, 0, 0] = np.conj(arry['HRs'][:, :, 0, 2, 0, 0]).T

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
    from ..utils.constants import ANGSTROM_AU

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
    from ..utils.constants import ANGSTROM_AU

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


def pythtb_to_hopping_model(my_model) -> HoppingModel:
    r"""Hopping list of a PythTB model (``pythtb`` 2.x ``TBModel``) in PAOFLOW's conventions.

    Parameters
    ----------
    my_model : pythtb.TBModel
        Periodic in every direction of its lattice (finite directions are not supported).

    Returns
    -------
    HoppingModel
        Lattice vectors padded with unit vectors to 3D, ``alat = 1``; spinful models in the block
        order (every orbital up, then every orbital down), each spin-orbital at its site;
        ``nelec`` is ``None`` (PythTB models carry no filling); the grid holds the hoppings
        (:math:`2\max|R| + 1` per axis).

    Notes
    -----
    PythTB's hopping ``(h, i, j, R)`` is :math:`\langle i, 0|H|j, \mathbf R\rangle`, the
    ``(i, j, R, h)`` of :class:`HoppingModel`, which :func:`hoppings_to_HRs` stores at the
    index :math:`-\mathbf R` that PAOFLOW's Fourier convention and the orbital centres need.
    """
    lattice = my_model._lattice
    table = my_model._hoptable
    ndim = int(table.dim_r)
    norb = int(lattice._orb_vecs_cart.shape[0])
    spinful = bool(my_model._spinful)
    lattice_vectors = np.eye(3)
    lattice_vectors[:ndim, :ndim] = lattice._lat_vectors
    cartesian = np.zeros((norb, 3))
    cartesian[:, :ndim] = lattice._orb_vecs_cart
    sites = np.linalg.solve(lattice_vectors.T, cartesian.T).T
    cells = np.zeros((len(table.amplitudes), 3), dtype=int)
    cells[:, :ndim] = np.asarray(table.lattice_vecs, dtype=int).reshape(-1, ndim)
    bonds = [
        (int(i), int(j), cell, amplitude)
        for i, j, cell, amplitude in zip(table.from_idx, table.to_idx, cells, table.amplitudes)
    ]
    hoppings = _hermitian_hoppings(
        norb, list(enumerate(my_model._site_energies)), bonds, spinful=spinful
    )
    grid = 2 * np.max(np.abs(cells), axis=0) + 1 if len(cells) else np.ones(3, dtype=int)
    return HoppingModel(
        'pythtb', hoppings, 2 * norb if spinful else norb, lattice_vectors, 1.0,
        np.concatenate([sites, sites]) if spinful else sites, None,
        tuple(int(n) for n in grid), spinful=spinful,
    )  # fmt: skip


def build_from_pythTB(data_controller: DataController, my_model) -> None:
    r"""Load a PythTB model into the DataController (:func:`pythtb_to_hopping_model`).

    Parameters
    ----------
    data_controller : DataController
    my_model : pythtb.TBModel

    Returns
    -------
    None
        What :func:`build_from_hoppings` sets (no ``nelec``), plus ``attributes['norb']`` and
        ``arrays['Hks']`` (FFT of ``HRs``).

    Notes
    -----
    Until 13 Nov 2026 the hoppings were stored at the index :math:`+\mathbf R` and spinful
    models had the centres (A, A, B, B): band structures of time-reversal-odd models came out
    at :math:`-\mathbf k` and the optical responses broke the point-group selection rules.
    """
    model = pythtb_to_hopping_model(my_model)
    build_from_hoppings(data_controller, model)
    arrays, attributes = data_controller.data_dicts()
    attributes['norb'] = model.number_of_orbitals
    arrays['Hks'] = np.fft.fftn(arrays['HRs'], axes=(2, 3, 4))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hopping-list models (the validation models of the optical responses)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hopping = tuple[int, int, Sequence[int], complex]


def hoppings_to_HRs(
    hoppings: Sequence[Hopping], number_of_orbitals: int, real_space_grid: Sequence[int]
) -> NDArray[np.complex128]:
    r"""PAOFLOW ``HRs`` of a hopping list.

    Parameters
    ----------
    hoppings : sequence of ``(i, j, R, h)``
        Amplitude ``h`` (eV) from orbital ``i`` in the home cell to orbital ``j`` in the cell ``R``.
    number_of_orbitals : int
    real_space_grid : sequence of 3 int
        Grid of ``HRs``; every entry must exceed twice the largest ``|R|`` along its axis.

    Returns
    -------
    ndarray ``(number_of_orbitals, number_of_orbitals, n1, n2, n3, 1)``, complex
        ``h`` accumulated at the grid index ``-R`` (modulo the grid).

    Raises
    ------
    ValueError
        If a hopping does not fit in ``real_space_grid``.

    Notes
    -----
    With PAOFLOW's convention :math:`H(\mathbf k) = \sum_{\mathbf n} e^{-2\pi i\mathbf k\cdot\mathbf n}
    \,\mathrm{HRs}[\ldots, \mathbf n]` the index :math:`-\mathbf R` gives the lattice-gauge sum
    :math:`\sum_{\mathbf R} h_{ij}(\mathbf R)\,e^{2\pi i\mathbf k\cdot\mathbf R}`, the cell of orbital
    ``j`` being :math:`\mathbf R` as the orbital centres and the position operator require.
    """
    grid = np.asarray(real_space_grid, dtype=int)
    HRs = np.zeros(
        (number_of_orbitals, number_of_orbitals, grid[0], grid[1], grid[2], 1), dtype=complex
    )
    for i, j, cell, amplitude in hoppings:
        if np.any(2 * np.abs(np.asarray(cell)) >= grid):
            raise ValueError(f'real_space_grid {tuple(grid)} cannot hold the cell R = {cell}')
        index = (-np.asarray(cell)) % grid
        HRs[i, j, index[0], index[1], index[2], 0] += amplitude
    return HRs


@dataclass(frozen=True)
class HoppingModel:
    r"""Tight-binding model given as a list of hoppings between orbital centres.

    Attributes
    ----------
    name : str
    hoppings : tuple of ``(i, j, R, h)``
        Amplitude ``h`` (eV) from orbital ``i`` in the home cell to orbital ``j`` in the cell ``R``
        (integer lattice coordinates). The list is Hermitian as given: every ``(i, j, R, h)`` has its
        partner ``(j, i, -R, conj(h))``, on-site terms appear once.
    number_of_orbitals : int
    lattice_vectors : ndarray ``(3, 3)``
        Rows, alat units.
    alat : float
        Lattice parameter (bohr).
    orbital_positions : ndarray ``(number_of_orbitals, 3)``
        Orbital centres, crystal coordinates.
    nelec : float or None
        Electrons per cell; ``None`` leaves it to the caller (the PythTB models carry no filling).
    real_space_grid : tuple of 3 int
        Grid of the ``HRs`` array (:func:`hoppings_to_HRs`).
    spinful : bool
        The orbitals are spin-orbitals and every band holds one electron (PAOFLOW's ``dftSO``);
        otherwise the bands are spin-degenerate and hold two. Spin-orbitals are in PAOFLOW's block
        order (every orbital spin up, then every orbital spin down), the basis of the ``Sj``
        operators of :func:`build_from_hoppings`.

    Notes
    -----
    The Bloch Hamiltonian in the atomic gauge is

    .. math::

        H^A_{ij}(\mathbf k) = \sum_{\mathbf R} h_{ij}(\mathbf R)\,e^{i\mathbf k\cdot\mathbf d},
        \qquad \mathbf d = a\,(\mathbf R + \boldsymbol\tau_j - \boldsymbol\tau_i)\,A,

    with :math:`a` = ``alat``, :math:`A` = ``lattice_vectors`` and :math:`\boldsymbol\tau` =
    ``orbital_positions``.
    """

    name: str
    hoppings: tuple
    number_of_orbitals: int
    lattice_vectors: NDArray[np.float64]
    alat: float
    orbital_positions: NDArray[np.float64]
    nelec: float | None
    real_space_grid: tuple[int, int, int]
    spinful: bool = False

    @property
    def volume(self) -> float:
        """Cell volume (bohr^3)."""
        return self.alat**3 * abs(float(np.linalg.det(self.lattice_vectors)))

    def real_space_hamiltonian(self) -> NDArray[np.complex128]:
        """``HRs`` of the model in PAOFLOW's layout (:func:`hoppings_to_HRs`)."""
        return hoppings_to_HRs(self.hoppings, self.number_of_orbitals, self.real_space_grid)


def honeycomb_model(
    delta: float = 0.5,
    t: float = -1.0,
    t2: float = 0.0,
    phi: float = 0.0,
    nelec: float = 2.0,
    alat: float = 4.65,
    strain: float = 0.0,
) -> HoppingModel:
    r"""Gapped graphene (``t2 = 0``) or the Haldane model on the honeycomb lattice (``a_1 || x``).

    Parameters
    ----------
    delta : float, optional
        Sublattice potential, :math:`\pm\Delta/2` on A and B (eV): breaks inversion.
    t : float, optional
        Nearest-neighbour hopping (eV).
    t2 : float, optional
        Next-nearest-neighbour hopping :math:`t_2 e^{\pm i\phi}` (eV).
    phi : float, optional
        Haldane flux phase: time reversal is broken for ``t2 != 0`` and ``phi`` not a multiple of
        :math:`\pi`.
    nelec : float, optional
        Electrons per cell (spin-degenerate bands: 2 fills the lower band).
    alat : float, optional
        Lattice parameter (bohr); the default is that of graphene, 2.46 A.
    strain : float, optional
        Relative change of the bond ``R = 0``: breaks the threefold axis.

    Returns
    -------
    HoppingModel
        Two orbitals at :math:`(1/3, 1/3)` and :math:`(2/3, 2/3)`, a 10 alat vacuum along ``z``.

    Notes
    -----
    .. math::

        H = \frac{\Delta}{2}\sum_i \xi_i\,c_i^\dagger c_i + t\sum_{\langle ij\rangle} c_i^\dagger c_j
        + t_2\sum_{\langle\langle ij\rangle\rangle} e^{i\nu_{ij}\phi}\,c_i^\dagger c_j,

    with :math:`\xi = \pm 1` on A and B and :math:`\nu_{ij} = \pm 1` for the counter-clockwise
    and clockwise second neighbours (F. D. M. Haldane, Phys. Rev. Lett. 61, 2015 (1988)).
    Uses in the validation of the optical responses: the point group of gapped graphene, D3h,
    allows only :math:`\sigma^{yyy} = -\sigma^{yxx} = -\sigma^{xxy}`; ``strain`` lowers it so that
    every in-plane component survives (shift and injection currents, the Berry-curvature dipole of
    the doped model); the Haldane flux gives the time-reversal-odd responses (magnetic injection,
    gyration current) and changes sign with ``phi``.
    """
    lattice_vectors = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    orbital_positions = np.array([[1.0 / 3, 1.0 / 3, 0.0], [2.0 / 3, 2.0 / 3, 0.0]])
    hoppings = [(0, 0, (0, 0, 0), 0.5 * delta), (1, 1, (0, 0, 0), -0.5 * delta)]
    # nearest neighbours: B at tau_B + R for R in {0, -a1, -a2} relative to A
    for cell in ((0, 0, 0), (-1, 0, 0), (0, -1, 0)):
        bond = t * (1.0 + strain) if cell == (0, 0, 0) else t  # ``strain`` breaks C3
        hoppings.append((0, 1, cell, bond))
        hoppings.append((1, 0, tuple(-np.array(cell)), bond))
    if t2 != 0.0:
        # second neighbours: the six lattice vectors; +phi on A for the counter-clockwise set
        for cell, sign in (
            ((1, 0, 0), 1),
            ((0, 1, 0), -1),
            ((-1, 1, 0), 1),
            ((-1, 0, 0), -1),
            ((0, -1, 0), 1),
            ((1, -1, 0), -1),
        ):
            hoppings.append((0, 0, cell, t2 * np.exp(1j * sign * phi)))
            hoppings.append((1, 1, cell, t2 * np.exp(-1j * sign * phi)))
    return HoppingModel(
        'honeycomb', tuple(hoppings), 2, lattice_vectors, float(alat), orbital_positions,
        float(nelec), (6, 6, 1),
    )  # fmt: skip


def rice_mele_model(
    t: float = 1.0, dt: float = 0.3, delta: float = 0.4, nelec: float = 2.0, alat: float = 2.0
) -> HoppingModel:
    r"""Rice-Mele chain along ``x``: alternating hoppings :math:`t \pm \delta t` and a staggered potential.

    Parameters
    ----------
    t : float, optional
        Mean hopping (eV).
    dt : float, optional
        Dimerization: :math:`t + \delta t` inside the cell, :math:`t - \delta t` between cells (eV).
    delta : float, optional
        Staggered on-site potential :math:`\pm\Delta` (eV): breaks inversion.
    nelec : float, optional
        Electrons per cell (spin-degenerate bands).
    alat : float, optional
        Cell length (bohr); 10 alat of vacuum along ``y`` and ``z``.

    Returns
    -------
    HoppingModel

    Notes
    -----
    .. math::

        H = \Delta\sum_i (c_{iA}^\dagger c_{iA} - c_{iB}^\dagger c_{iB})
        + \sum_i \left[(t + \delta t)\,c_{iA}^\dagger c_{iB} + (t - \delta t)\,c_{iB}^\dagger
        c_{i+1,A} + \mathrm{h.c.}\right]

    (M. J. Rice and E. J. Mele, Phys. Rev. Lett. 49, 1455 (1982)). The only second-order
    component is :math:`xxx`; the shift current vanishes at :math:`\Delta = 0` (SSH, inversion
    symmetric), changes sign with :math:`\Delta` and with :math:`\delta t`, the 1D ferroelectric
    of B. M. Fregoso, T. Morimoto and J. E. Moore, Phys. Rev. B 96, 075421 (2017).
    """
    lattice_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    orbital_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    hoppings = (
        (0, 0, (0, 0, 0), delta),
        (1, 1, (0, 0, 0), -delta),
        (0, 1, (0, 0, 0), t + dt),
        (1, 0, (0, 0, 0), t + dt),
        (1, 0, (1, 0, 0), t - dt),
        (0, 1, (-1, 0, 0), t - dt),
    )
    return HoppingModel(
        'rice_mele', hoppings, 2, lattice_vectors, float(alat), orbital_positions, float(nelec),
        (8, 1, 1),
    )  # fmt: skip


def weyl_model(t: float = 1.0, b: float = 0.3, alat: float = 1.0) -> HoppingModel:
    r"""Two-band lattice Weyl semimetal with two nodes of opposite chirality at different energies.

    Parameters
    ----------
    t : float, optional
        Hopping (eV): the cones have the isotropic velocity :math:`t\,a/\hbar`.
    b : float, optional
        Energy offset of the nodes, :math:`\pm b` (eV).
    alat : float, optional
        Lattice parameter of the simple-cubic cell (bohr).

    Returns
    -------
    HoppingModel
        One spinful orbital (two spin-orbitals) at the origin, one electron per cell.

    Notes
    -----
    .. math::

        H(\mathbf k) = t\left[\sin k_x\,\sigma_x + \sin k_y\,\sigma_y
        + (2 - \cos k_x - \cos k_y - \cos k_z)\,\sigma_z\right] + b\sin k_z,

    with nodes at :math:`\mathbf k = (0, 0, \pm\pi/2)`, chiralities :math:`\pm 1`, energies
    :math:`\pm b` (the :math:`b\sin k_z` term shifts them without tilting the cones). Time reversal
    and inversion are both broken. With the Fermi level at one node the other is Pauli-blocked for
    :math:`\hbar\omega < 4b`, where the trace of the circular injection tensor is quantized by the
    chirality of the first (F. de Juan et al., Nat. Commun. 8, 15995 (2017)).
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    hoppings = []

    def add(cell: tuple[int, int, int], matrix: NDArray[np.complex128]) -> None:
        for i in range(2):
            for j in range(2):
                if matrix[i, j] != 0:
                    hoppings.append((i, j, np.array(cell), matrix[i, j]))

    add((0, 0, 0), 2.0 * t * pauli_z)
    for axis, sigma in ((0, pauli_x), (1, pauli_y)):
        unit = np.eye(3, dtype=int)[axis]
        add(tuple(unit), t * sigma / 2j)
        add(tuple(-unit), -t * sigma / 2j)
    for axis in range(3):
        unit = np.eye(3, dtype=int)[axis]
        add(tuple(unit), -t * pauli_z / 2.0)
        add(tuple(-unit), -t * pauli_z / 2.0)
    add((0, 0, 1), b * np.eye(2) / 2j)
    add((0, 0, -1), -b * np.eye(2) / 2j)
    return HoppingModel(
        'weyl', tuple(hoppings), 2, np.eye(3), float(alat), np.zeros((2, 3)), 1.0, (3, 3, 3),
        spinful=True,
    )  # fmt: skip


def single_band_model(
    hoppings: Mapping[tuple[int, int, int], complex],
    lattice_vectors: ArrayLike,
    alat: float,
    nelec: float,
    real_space_grid: Sequence[int],
    remote_band_energy: float = 12.0,
) -> HoppingModel:
    r"""One band of arbitrary hoppings plus a decoupled empty band: a metal with only intraband terms.

    Parameters
    ----------
    hoppings : mapping ``R -> h``
        Hopping of the band to the cell ``R`` (eV); the partner :math:`h(-\mathbf R) =
        h(\mathbf R)^*` is added, so give one of each pair. Complex amplitudes on odd shells break
        time reversal (:math:`\varepsilon(\mathbf k) \neq \varepsilon(-\mathbf k)`).
    lattice_vectors : array_like ``(3, 3)``
        Rows, alat units.
    alat : float
        Lattice parameter (bohr).
    nelec : float
        Electrons per cell (spin-degenerate: below 2 the band is partly filled).
    real_space_grid : sequence of 3 int
    remote_band_energy : float, optional
        On-site energy of the second, decoupled orbital (eV): an empty band far above, which keeps
        the band-edge bookkeeping of the response codes defined and contributes nothing.

    Returns
    -------
    HoppingModel

    Notes
    -----
    :math:`\varepsilon(\mathbf k) = \sum_{\mathbf R} h(\mathbf R)\,e^{2\pi i\mathbf k\cdot\mathbf R}`.
    One band has no interband coherence, so the optical responses reduce to the Drude terms,
    which have closed forms: the dielectric function :math:`1 - \omega_p^2/\omega(\omega + i\gamma)`,
    :math:`\omega_p^2 = (4\pi/\Omega)\,g_s\sum_{\mathbf k} f\,\partial^2\varepsilon`, and the
    second-order Drude current :math:`\propto\tau^2\sum_{\mathbf k} f\,\partial_a\partial_b
    \partial_c\varepsilon` (non-zero only without time reversal). Validation uses: the 2D band
    ``{(1,0,0): -1.0, (0,1,0): -0.8, (2,0,0): 0.25j, (1,1,0): 0.1+0.15j, (1,-1,0): -0.1j}``
    (``np.diag([1, 1, 10])``, alat 5 bohr, 0.8 electrons, grid (5, 5, 1)) for the nonlinear Drude
    current, the 3D band ``{(1,0,0): -1.0, (0,1,0): -0.8, (0,0,1): -0.5, (1,1,0): 0.2}``
    (simple cubic, alat 5 bohr, 0.8 electrons, grid (5, 5, 5)) for the Drude dielectric function.
    """
    entries = [(1, 1, (0, 0, 0), float(remote_band_energy))]
    for cell, amplitude in hoppings.items():
        entries.append((0, 0, cell, amplitude))
        entries.append((0, 0, tuple(-np.array(cell)), np.conj(amplitude)))
    return HoppingModel(
        'single_band', tuple(entries), 2, np.asarray(lattice_vectors, dtype=float), float(alat),
        np.zeros((2, 3)), float(nelec), tuple(int(n) for n in real_space_grid),
    )  # fmt: skip


def _hermitian_hoppings(
    number_of_orbitals: int,
    onsite: Sequence[tuple[int, ArrayLike]],
    bonds: Sequence[tuple[int, int, Sequence[int], ArrayLike]],
    spinful: bool = False,
) -> tuple:
    r"""Hermitian hopping list of a :class:`HoppingModel` from on-site terms and one entry per bond.

    Parameters
    ----------
    number_of_orbitals : int
        Spatial orbitals per cell (twice as many spin-orbitals when ``spinful``).
    onsite : sequence of ``(i, e)``
        On-site energy of orbital ``i`` (eV): a number, or a Hermitian 2x2 spin matrix.
    bonds : sequence of ``(i, j, R, h)``
        Amplitude from orbital ``i`` in the home cell to orbital ``j`` in the cell ``R`` (eV): a
        number, or a 2x2 spin matrix :math:`h_{ss'}` when ``spinful``. Give each bond once; the
        partner ``(j, i, -R)`` with :math:`h^\dagger` is added here.
    spinful : bool, optional
        Spin-orbitals in the block order: orbital ``i`` with spin ``s`` (0 up, 1 down) is
        ``i + s * number_of_orbitals``; a number is a spin-independent term.

    Returns
    -------
    tuple of ``(i, j, R, h)``
        Zero amplitudes are dropped.
    """
    n = int(number_of_orbitals)
    spins = (0, 1) if spinful else (0,)

    def spin_matrix(value: ArrayLike) -> NDArray[np.complex128]:
        matrix = np.asarray(value, dtype=complex)
        if matrix.ndim == 0:
            return matrix * np.eye(len(spins))
        return matrix.reshape(len(spins), len(spins))

    entries = []
    for i, energy in onsite:
        matrix = spin_matrix(energy)
        for s in spins:
            for s2 in spins:
                if matrix[s, s2] != 0:
                    entries.append((int(i) + s * n, int(i) + s2 * n, (0, 0, 0), matrix[s, s2]))
    for i, j, cell, amplitude in bonds:
        matrix = spin_matrix(amplitude)
        cell = tuple(int(c) for c in cell)
        back = tuple(-c for c in cell)
        for s in spins:
            for s2 in spins:
                if matrix[s, s2] != 0:
                    entries.append((int(i) + s * n, int(j) + s2 * n, cell, matrix[s, s2]))
                    entries.append((int(j) + s2 * n, int(i) + s * n, back, np.conj(matrix[s, s2])))
    return tuple(entries)


def kane_mele_model(
    t: float = 1.0,
    soc: float = 0.06,
    rashba: float = 0.05,
    delta: float = 0.2,
    nelec: float = 2.0,
    alat: float = 4.65,
) -> HoppingModel:
    r"""Kane-Mele model: spinful honeycomb lattice with intrinsic and Rashba spin-orbit coupling.

    Parameters
    ----------
    t : float, optional
        Nearest-neighbour hopping (eV).
    soc : float, optional
        Intrinsic spin-orbit coupling :math:`\lambda_{SO}` (eV), on the second neighbours.
    rashba : float, optional
        Rashba coupling :math:`\lambda_R` (eV), on the first neighbours: breaks the mirror
        :math:`z \to -z` and :math:`S_z` conservation.
    delta : float, optional
        Staggered potential :math:`\lambda_v` (eV), :math:`\pm\lambda_v` on A and B: breaks
        inversion. With ``rashba = 0`` the gap closes at :math:`\lambda_v = 3\sqrt{3}\,
        \lambda_{SO}` (quantum spin Hall below, trivial above).
    nelec : float, optional
        Electrons per cell (spinful bands hold one: 2 fills the two lower bands).
    alat : float, optional
        Lattice parameter (bohr).

    Returns
    -------
    HoppingModel
        Four spin-orbitals (A up, B up | A down, B down) at :math:`(1/3, 1/3)` and
        :math:`(2/3, 2/3)`, ``a_1 || x``, a 10 alat vacuum along ``z``; ``spinful``.

    Notes
    -----
    .. math::

        H = t\sum_{\langle ij\rangle} c_i^\dagger c_j
        + i\lambda_{SO}\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}\,c_i^\dagger s_z c_j
        + i\lambda_R\sum_{\langle ij\rangle} c_i^\dagger(\mathbf s\times\hat{\mathbf d}_{ij})_z c_j
        + \lambda_v\sum_i \xi_i\,c_i^\dagger c_i,

    with :math:`\hat{\mathbf d}_{ij}` the unit vector from :math:`i` to :math:`j`,
    :math:`\nu_{ij} = \pm 1` and :math:`\xi = \pm 1` on A and B (C. L. Kane and E. J. Mele,
    Phys. Rev. Lett. 95, 226801 (2005)); the hoppings are those of :func:`kane_mele_pythtb`.
    With ``rashba = 0`` the spectrum at K and K' is :math:`\pm(\lambda_v \pm 3\sqrt{3}\,
    \lambda_{SO})`. Point group D3h (``rashba = 0``) or C3v: the in-plane second-order
    responses obey :math:`\sigma^{yyy} = -\sigma^{yxx} = -\sigma^{xxy}`, :math:`\sigma^{xxx} =
    0`. Time reversal holds, so the T-odd responses (magnetic injection, gyration) vanish.
    """
    root3 = np.sqrt(3.0)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    lattice_vectors = np.array([[1.0, 0.0, 0.0], [0.5, root3 / 2, 0.0], [0.0, 0.0, 10.0]])
    sites = np.array([[1.0 / 3, 1.0 / 3, 0.0], [2.0 / 3, 2.0 / 3, 0.0]])
    bonds = []
    # first neighbours: B in the cells 0, -a2, -a1 of A; d = unit vector from A to B
    for cell, (dx, dy) in (
        ((0, 0, 0), (root3 / 2, 0.5)),
        ((0, -1, 0), (0.0, -1.0)),
        ((-1, 0, 0), (-root3 / 2, 0.5)),
    ):
        bonds.append((0, 1, cell, t * np.eye(2) + 1j * rashba * (dy * sigma_x - dx * sigma_y)))
    # second neighbours: i lambda_SO nu_ij s_z, opposite signs on A and B
    for cell, sign in (((0, 1, 0), 1), ((1, 0, 0), -1), ((1, -1, 0), 1)):
        bonds.append((0, 0, cell, 1j * sign * soc * sigma_z))
        bonds.append((1, 1, cell, -1j * sign * soc * sigma_z))
    hoppings = _hermitian_hoppings(2, [(0, delta), (1, -delta)], bonds, spinful=True)
    return HoppingModel(
        'kane_mele', hoppings, 4, lattice_vectors, float(alat), np.concatenate([sites, sites]),
        float(nelec), (6, 6, 1), spinful=True,
    )  # fmt: skip


#: Nearest-neighbour three-band parameters of G.-B. Liu et al., Phys. Rev. B 88, 085433 (2013),
#: Table II (arXiv:1305.6089v4): ``(a [A], eps1, eps2, t0, t1, t2, t11, t12, t22)`` (eV).
TMD_NEAREST_NEIGHBOUR = {
    'GGA': {
        'MoS2': (3.190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338, 0.057),
        'WS2': (3.191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061),
        'MoSe2': (3.326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290, 0.130),
        'WSe2': (3.325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329, 0.034),
        'MoTe2': (3.557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239, 0.252),
        'WTe2': (3.560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270, 0.190),
    },
    'LDA': {
        'MoS2': (3.129, 1.238, 2.366, -0.218, 0.444, 0.533, 0.250, 0.360, 0.047),
        'WS2': (3.132, 1.355, 2.569, -0.238, 0.626, 0.557, 0.324, 0.405, -0.076),
        'MoSe2': (3.254, 1.001, 2.239, -0.222, 0.350, 0.488, 0.244, 0.314, 0.129),
        'WSe2': (3.253, 1.124, 2.447, -0.242, 0.506, 0.514, 0.305, 0.353, 0.025),
        'MoTe2': (3.472, 0.618, 2.126, -0.202, 0.254, 0.423, 0.241, 0.263, 0.269),
        'WTe2': (3.476, 0.623, 2.251, -0.209, 0.388, 0.442, 0.272, 0.295, 0.200),
    },
}

#: Spin-orbit parameter lambda (eV) of the same paper, Table IV (GGA).
TMD_SPIN_ORBIT = {
    'MoS2': 0.073,
    'WS2': 0.211,
    'MoSe2': 0.091,
    'WSe2': 0.228,
    'MoTe2': 0.107,
    'WTe2': 0.237,
}


def tmd_model(
    material: str = 'MoS2',
    functional: str = 'GGA',
    soc: bool | float = True,
    thickness: float | None = None,
) -> HoppingModel:
    r"""Three-band model of a group-VIB transition-metal dichalcogenide monolayer (MX2).

    Parameters
    ----------
    material : str, optional
        ``'MoS2'``, ``'WS2'``, ``'MoSe2'``, ``'WSe2'``, ``'MoTe2'`` or ``'WTe2'``.
    functional : str, optional
        ``'GGA'`` or ``'LDA'``: the parameter set (:data:`TMD_NEAREST_NEIGHBOUR`).
    soc : bool or float, optional
        On-site spin-orbit coupling :math:`\lambda\,L_zS_z`: ``True`` takes :math:`\lambda`
        from :data:`TMD_SPIN_ORBIT` (fitted to GGA, so ``functional='GGA'`` only), a number is
        :math:`\lambda` in eV, ``False`` gives the spin-degenerate model.
    thickness : float, optional
        Height of the cell along ``z`` (angstrom), the layer thickness that turns sheet
        responses into the per-volume ones PAOFLOW writes; default 10 alat.

    Returns
    -------
    HoppingModel
        The metal :math:`d_{z^2}, d_{xy}, d_{x^2-y^2}` orbitals at the origin (block-ordered
        spin-orbitals with ``soc``), ``a_1 = a\hat x``, two electrons per cell (the valence
        band filled).

    Raises
    ------
    ValueError
        For an unknown material or functional, or ``soc=True`` with the LDA set.

    Notes
    -----
    Nearest-neighbour model of G.-B. Liu, W.-Y. Shan, Y. Yao, W. Yao and D. Xiao, Phys. Rev. B
    88, 085433 (2013). The hopping to the neighbour :math:`\mathbf R_1 = a\hat x` is

    .. math::

        E(\mathbf R_1) = \begin{pmatrix} t_0 & t_1 & t_2\\ -t_1 & t_{11} & t_{12}\\
        t_2 & -t_{12} & t_{22}\end{pmatrix},

    those to the C3 images :math:`\mathbf R = C_3^n\mathbf R_1` are
    :math:`D(-2\pi n/3)\,E(\mathbf R_1)\,D(-2\pi n/3)^T` (:math:`D(\theta)` rotates
    :math:`(d_{xy}, d_{x^2-y^2})` by :math:`2\theta`), and :math:`E(-\mathbf R) =
    E(\mathbf R)^T`; :math:`H(\mathbf k) = \sum_{\mathbf R}E(\mathbf R)\,e^{i\mathbf k\cdot
    \mathbf R}` is the paper's closed form, Eqs. (4)-(11). The spin-orbit term is
    :math:`(\lambda/2)\,L_z\otimes\sigma_z` with :math:`L_z = 2i` between :math:`d_{xy}` and
    :math:`d_{x^2-y^2}` (Eqs. (25)-(27)); in this three-band model the valence splitting at K is
    exactly :math:`2\lambda` and of opposite sign at K'. Direct gap at K (without SOC):
    :math:`\varepsilon_1 - 3t_0 - [\varepsilon_2 - \tfrac32(t_{11} + t_{22}) -
    3\sqrt3\,|t_{12}|]` (1.663 eV for GGA MoS2). Point group D3h with the mirror
    :math:`x \to -x`: :math:`\sigma^{yyy} = -\sigma^{yxx} = -\sigma^{xxy}`, :math:`\sigma^{xxx} =
    0`. All orbitals sit on the metal site, so the optical matrix elements are those of
    :math:`\partial H/\partial\mathbf k` alone (no intra-atomic dipoles).
    """
    from ..utils.constants import ANGSTROM_AU

    if functional not in TMD_NEAREST_NEIGHBOUR:
        raise ValueError(f'functional must be one of {sorted(TMD_NEAREST_NEIGHBOUR)}')
    table = TMD_NEAREST_NEIGHBOUR[functional]
    if material not in table:
        raise ValueError(f'material must be one of {sorted(table)}')
    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = table[material]
    if soc is True:
        if functional != 'GGA':
            raise ValueError('the tabulated lambda is fitted to GGA: pass soc=<lambda in eV>')
        coupling = TMD_SPIN_ORBIT[material]
    else:
        coupling = float(soc)
    spinful = coupling != 0.0
    hop_r1 = np.array([[t0, t1, t2], [-t1, t11, t12], [t2, -t12, t22]])
    bonds = []
    # R1 = a1, C3 R1 = -a1 + a2, C3^2 R1 = -a2
    for n, cell in enumerate(((1, 0, 0), (-1, 1, 0), (0, -1, 0))):
        angle = -4.0 * np.pi * n / 3.0  # D(-theta) rotates (d_xy, d_x2-y2) by -2 theta
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        hop = rotation @ hop_r1 @ rotation.T
        bonds += [(mu, nu, cell, hop[mu, nu]) for mu in range(3) for nu in range(3)]
    onsite = [(0, eps1), (1, eps2), (2, eps2)]
    if spinful:
        # (lambda / 2) L_z sigma_z, L_z = 2i between d_xy and d_x2-y2
        bonds.append((1, 2, (0, 0, 0), 1j * coupling * np.diag([1.0, -1.0])))
    hoppings = _hermitian_hoppings(3, onsite, bonds, spinful=spinful)
    height = 10.0 if thickness is None else float(thickness) / a
    lattice_vectors = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, height]])
    norb = 6 if spinful else 3
    return HoppingModel(
        f'tmd_{material}', hoppings, norb, lattice_vectors, a * ANGSTROM_AU, np.zeros((norb, 3)),
        2.0, (6, 6, 1), spinful=spinful,
    )  # fmt: skip


def parabolic_model(
    n_bands: int = 2,
    gap: float = 1.5,
    t_c: float = 0.5,
    t_sigma: float = 0.4,
    t_pi: float = 0.1,
    kane_p: float = 1.0,
    asymmetry: float = 0.3,
    phase: float = 0.0,
    alat: float = 10.0,
) -> HoppingModel:
    r"""Parabolic-band semiconductor: one s conduction band and ``n_bands - 1`` p valence bands (1D, 2D, 3D).

    Parameters
    ----------
    n_bands : {2, 3, 4}, optional
        2: chain along ``x`` (s, p_x); 3: square lattice (s, p_x, p_y); 4: simple cubic (s, p_x, p_y,
        p_z). The dimension is ``n_bands - 1``; 10 alat of vacuum along the other axes.
    gap : float, optional
        Energy of the s level above the p levels at Gamma before the inversion-breaking term (eV).
    t_c : float, optional
        Conduction hopping (eV): :math:`2t_c\sum_\alpha(1 - \cos k_\alpha a)`.
    t_sigma, t_pi : float, optional
        Valence hoppings along and across the p lobe (eV): light and heavy holes.
    kane_p : float, optional
        Interband (Kane) momentum coupling :math:`P` (eV): :math:`H_{s p_\alpha} = iP\sin k_\alpha a`.
    asymmetry : float, optional
        Inversion-breaking amplitude (eV). 2 and 3 bands: an on-site s-p mixing :math:`Q` along the
        polar axis (``x`` for the chain, ``y`` on the square lattice); 4 bands: Kane's
        :math:`B` term, :math:`H_{s p_x} \mathrel{+}= B\sin k_y a\sin k_z a` and cyclic (T_d).
    phase : float, optional
        Phase :math:`\phi` of the asymmetry term, :math:`Q e^{i\phi}` (or :math:`Be^{i\phi}`): 0 keeps
        time reversal, :math:`\pi/2` breaks it and keeps the product PT.
    alat : float, optional
        Lattice parameter (bohr).

    Returns
    -------
    HoppingModel
        All orbitals at the origin (no intra-cell dipoles), the valence bands filled
        (``nelec = 2 (n_bands - 1)``), spin-degenerate.

    Raises
    ------
    ValueError
        For ``n_bands`` other than 2, 3, 4.

    Notes
    -----
    With :math:`d = ` ``n_bands - 1`` and :math:`c_\alpha = \cos k_\alpha a`, :math:`s_\alpha = \sin k_\alpha a`,

    .. math::

        H_{ss} = E_g + 2t_c\sum_{\alpha}(1 - c_\alpha),\quad
        H_{p_\alpha p_\alpha} = -2t_\sigma(1 - c_\alpha) - 2t_\pi\sum_{\beta\neq\alpha}(1 - c_\beta),\quad
        H_{s p_\alpha} = iP s_\alpha + A_\alpha,

    with :math:`A = Qe^{i\phi}` on the polar axis (2, 3 bands) or :math:`A_x = Be^{i\phi}s_ys_z` and cyclic
    (4 bands). Near Gamma the bands are parabolic, with the Kane masses (:math:`A = 0`)
    :math:`\hbar^2/m_c = 2a^2(t_c + P^2/E_g)` for the conduction band, :math:`-2a^2(t_\sigma + P^2/E_g)` for
    the light hole along its lobe and :math:`-2a^2 t_\pi` for the heavy holes across (E. O. Kane, J. Phys.
    Chem. Solids 1, 249 (1957)). At Gamma the polar level pair is :math:`E_g/2 \pm
    \sqrt{E_g^2/4 + |Q|^2}`. Symmetry: :math:`P` is odd under inversion and even under time reversal,
    so :math:`A = 0` is centrosymmetric (every second-order response vanishes); :math:`\phi = 0` gives
    a polar (2, 3 bands; point group :math:`C_{1v}` with the mirror :math:`x \to -x` in 2D) or
    :math:`T_d` (4 bands) crystal with time reversal: shift current and circular injection, no T-odd
    response. :math:`\phi = \pi/2` breaks P and T but keeps PT (a magnetoelectric toroidal order): the
    linear shift current and the circular injection vanish and the magnetic injection and the
    gyration current appear (H. Watanabe and Y. Yanase, Phys. Rev. X 11, 011001 (2021)); the bands
    become asymmetric, :math:`\varepsilon(\mathbf k)\neq\varepsilon(-\mathbf k)`. With two bands the
    three-band (virtual) SHG terms vanish identically.
    """
    if n_bands not in (2, 3, 4):
        raise ValueError('n_bands must be 2, 3 or 4')
    dim = n_bands - 1
    unit = np.eye(3, dtype=int)
    amplitude = complex(asymmetry) * np.exp(1j * phase)
    onsite = [(0, gap + 2.0 * t_c * dim)]
    bonds = []
    for alpha in range(dim):
        onsite.append((1 + alpha, -2.0 * (t_sigma + (dim - 1) * t_pi)))
        step = tuple(unit[alpha])
        bonds.append((0, 0, step, -t_c))
        bonds.append((0, 1 + alpha, step, 0.5 * kane_p))
        bonds.append((0, 1 + alpha, tuple(-unit[alpha]), -0.5 * kane_p))
        for beta in range(dim):
            bonds.append((1 + beta, 1 + beta, step, t_sigma if beta == alpha else t_pi))
    if dim < 3:
        bonds.append((0, dim, (0, 0, 0), amplitude))  # p_x (chain) or p_y (square lattice)
    else:
        # B sin(k_b a) sin(k_c a) = -B/4 sum_{s, s'} s s' exp(i(s k_b + s' k_c) a), (a, b, c) cyclic
        for alpha, (b, c) in enumerate(((1, 2), (2, 0), (0, 1))):
            for sb in (1, -1):
                for sc in (1, -1):
                    cell = tuple(sb * unit[b] + sc * unit[c])
                    bonds.append((0, 1 + alpha, cell, -0.25 * sb * sc * amplitude))
    hoppings = _hermitian_hoppings(n_bands, onsite, bonds)
    lattice_vectors = np.diag([1.0 if axis < dim else 10.0 for axis in range(3)])
    grid = tuple(4 if axis < dim else 1 for axis in range(3))
    return HoppingModel(
        f'parabolic_{n_bands}', hoppings, n_bands, lattice_vectors, float(alat),
        np.zeros((n_bands, 3)), 2.0 * dim, grid,
    )  # fmt: skip


def _kane_mele_from_legacy(params: Mapping) -> HoppingModel:
    """:func:`kane_mele_model` from the parameters of the ``'Kane_Mele'`` label.

    ``t``, ``soc_par`` (:math:`\\lambda_{SO}`), ``r_par`` (:math:`\\lambda_R`, default 0),
    ``v_par`` (:math:`\\lambda_v / t`, default 0) in eV, ``alat`` in angstrom (default 1); an
    unknown key raises ``TypeError``.
    """
    from ..utils.constants import ANGSTROM_AU

    unknown = set(params) - {'label', 't', 'soc_par', 'r_par', 'v_par', 'alat'}
    if unknown:
        raise TypeError(f'Kane_Mele: unknown parameters {sorted(unknown)}')
    t = params['t']
    return kane_mele_model(
        t=t,
        soc=params['soc_par'],
        rashba=params.get('r_par', 0.0),
        delta=t * params.get('v_par', 0.0),
        alat=params.get('alat', 1.0) * ANGSTROM_AU,
    )


HOPPING_MODELS = {
    'HONEYCOMB': honeycomb_model,
    'RICE_MELE': rice_mele_model,
    'WEYL': weyl_model,
    'SINGLE_BAND': single_band_model,
    'TMD': tmd_model,
    'PARABOLIC': parabolic_model,
}


def build_from_hoppings(data_controller: DataController, model: HoppingModel) -> None:
    r"""Load a :class:`HoppingModel` into the DataController in place of a DFT calculation.

    Parameters
    ----------
    data_controller : DataController
    model : HoppingModel

    Returns
    -------
    None
        Sets ``arrays['HRs', 'a_vectors', 'b_vectors', 'tau', 'Dnm', 'naw', 'atoms']`` and
        ``attributes['nawf', 'bnd', 'nbnds', 'natoms', 'nspin', 'alat', 'omega', 'nelec', 'Efermi',
        'shift', 'dftSO', 'adhoc_SO', 'do_spin_orbit', 'nk1', 'nk2', 'nk3', 'nkpnts']``
        (``nelec`` only when the model has one); every orbital is its own site, so ``tau`` holds
        the orbital centres (Cartesian, alat units) and ``Dnm`` their differences
        :math:`\tau_n - \tau_m` (Cartesian, bohr, as the QE route) used by
        ``gradient_and_momenta``. Spinful models also get the spin operators
        ``arrays['Sj']`` (:math:`\sigma/2` in the block order of :class:`HoppingModel`).
    """
    arrays, attributes = data_controller.data_dicts()
    lattice_vectors = np.asarray(model.lattice_vectors, dtype=float)
    norb = int(model.number_of_orbitals)
    grid = tuple(int(n) for n in model.real_space_grid)
    arrays['HRs'] = model.real_space_hamiltonian()
    arrays['a_vectors'] = lattice_vectors
    arrays['b_vectors'] = np.linalg.inv(lattice_vectors).T
    arrays['tau'] = np.asarray(model.orbital_positions, dtype=float) @ lattice_vectors
    centres_bohr = float(model.alat) * arrays['tau']
    arrays['Dnm'] = centres_bohr[:, None, :] - centres_bohr[None, :, :]
    arrays['naw'] = np.ones(norb, dtype=int)
    arrays['atoms'] = ['X'] * norb
    attributes['nawf'] = attributes['bnd'] = attributes['nbnds'] = attributes['natoms'] = norb
    attributes['nspin'] = 1
    attributes['alat'] = float(model.alat)
    attributes['omega'] = model.volume
    if model.nelec is not None:
        attributes['nelec'] = float(model.nelec)
    attributes['Efermi'] = 0.0
    attributes['shift'] = np.inf
    spinful = bool(model.spinful)
    attributes['dftSO'] = attributes['adhoc_SO'] = attributes['do_spin_orbit'] = spinful
    if spinful:
        pauli = (
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]]),
            np.diag([1.0, -1.0]).astype(complex),
        )
        arrays['Sj'] = 0.5 * np.stack([np.kron(sigma, np.eye(norb // 2)) for sigma in pauli])
    attributes['nk1'], attributes['nk2'], attributes['nk3'] = grid
    attributes['nkpnts'] = int(np.prod(grid))


def build_TB_model(data_controller: DataController, params: dict) -> None:
    """Build the tight-binding model named by ``params['label']`` (``PAOFLOW(model=params)``).

    Parameters
    ----------
    data_controller : DataController
    params : dict
        ``'label'`` (case-insensitive) and the parameters of the model. The hopping-list models
        (``'honeycomb'``, ``'rice_mele'``, ``'weyl'``, ``'single_band'``, ``'tmd'``,
        ``'parabolic'``) take the keyword arguments of :func:`honeycomb_model`,
        :func:`rice_mele_model`, :func:`weyl_model`, :func:`single_band_model`, :func:`tmd_model` and
        :func:`parabolic_model` (``alat`` in bohr);
        ``'Kane_Mele'`` builds :func:`kane_mele_model` from ``t``, ``soc_par``, ``r_par``,
        ``v_par`` (in units of ``t``) and ``alat`` (angstrom); an unknown key raises
        ``TypeError``. The ``*_pythtb`` labels go through :func:`build_from_pythTB`.

    Returns
    -------
    None
        Fills the DataController; an unknown label exits.
    """
    model_name = params['label']
    match model_name.upper():
        case 'CUBIUM_PYTHTB':
            my_model = cubium_pythtb(params['t'])
            build_from_pythTB(data_controller, my_model)
        case 'CUBIUM2_PYTHTB':
            my_model = cubium2_pythtb(params['t'], params['Eg'])
            build_from_pythTB(data_controller, my_model)
        case 'SSH_PYTHTB':
            my_model = ssh_pythtb(params['v'], params['w'])
            build_from_pythTB(data_controller, my_model)
        case 'CHECKERBOARD_PYTHTB':
            my_model = checkerboard_pythtb(params['delta'], params['t'])
            build_from_pythTB(data_controller, my_model)
        case 'GRAPHENE_PYTHTB':
            my_model = graphene_pythtb(params['delta'], params['t'])
            build_from_pythTB(data_controller, my_model)
        case 'HALDANE_PYTHTB':
            my_model = haldane_pythtb(params['delta'], params['t1'], params['t2'], params['phi'])
            build_from_pythTB(data_controller, my_model)
        case 'KANE_MELE_PYTHTB':
            my_model = kane_mele_pythtb(
                params['delta'], params['t'], params['soc'], params['rashba']
            )
            build_from_pythTB(data_controller, my_model)
        case 'FU_KANE_MELE_PYTHTB':
            my_model = fu_kane_mele_pythtb(params['t'], params['soc'], params['dt'])
            build_from_pythTB(data_controller, my_model)
        case 'GRAPHENE':
            graphene(data_controller, params)
        case 'CUBIUM':
            cubium(data_controller, params)
        case 'CUBIUM2':
            cubium2(data_controller, params)
        case 'KANE_MELE':
            build_from_hoppings(data_controller, _kane_mele_from_legacy(params))
        case 'SLATER_KOSTER':
            Slater_Koster(data_controller, params)
        case 'SK_EDTB':
            SK_EDTB(data_controller, params)
        case label if label in HOPPING_MODELS:
            arguments = {key: value for key, value in params.items() if key != 'label'}
            build_from_hoppings(data_controller, HOPPING_MODELS[label](**arguments))
        case _:
            print(f'ERROR: Label "{model_name}" not found in builtin models.')
            import sys

            sys.exit()
