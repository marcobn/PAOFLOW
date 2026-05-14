# edtb_params.py
#
# Standalone EDTB parameter and geometry I/O — no PAOFLOW dependency.
#
# Defines the canonical schema for environment-dependent tight-binding
# parameters (species-pair resolved, distance-tagged) and provides
# read/write/validate/conversion utilities.
#
# Parameter file schema (JSON)
# ----------------------------
# {
#   "edtb_version": "1.0",
#   "description": "...",                          # optional
#
#   "basis": {
#     "<species>": {
#       "orbitals": ["s", "px", "py", "pz", ...],  # full orbital list
#       "l_channels": ["s", "p", ...]               # angular-momentum channels
#     }
#   },
#
#   "onsite": {
#     "<species>": {"s": <float>, "p": <float>, "t2g": <float>, "eg": <float>}
#   },
#
#   "hoppings": {
#     "<sp1>-<sp2>": [                              # alphabetically sorted key
#       {"r_ref": <float>, "params": {"sss": <float>, "sps": <float>, ...}},
#       ...                                         # one entry per shell
#     ]
#   },
#
#   "screening": {                                  # absent for pure SK
#     "r_cut": <float>,
#     "gamma": {
#       "<sp1>-<sp2>": {"ss": <float>, "sp": <float>, ...}
#     }
#   }
# }
#
# Geometry file schema (JSON)
# ---------------------------
# {
#   "alat": <float>,                                # lattice parameter (Bohr)
#   "a_vectors": [[...], [...], [...]],             # in units of alat
#   "atoms": [
#     {"species": "<name>", "tau": [x, y, z]},      # tau in units of alat
#     ...
#   ]
# }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

CURRENT_VERSION = '1.0'

SK_PARAM_NAMES = ['sss', 'sps', 'pps', 'ppp', 'sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd']

L_ORBITALS = {
    's': ['s'],
    'p': ['px', 'py', 'pz'],
    'd': ['dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2'],
}

# Canonical l-pair labels (sorted: l_lo ≤ l_hi)
LPAIR_LABELS = ['ss', 'sp', 'sd', 'pp', 'pd', 'dd']

# Active SK integrals for each l-pair
LPAIR_SK_NAMES = {
    'ss': ['sss'],
    'sp': ['sps'],
    'sd': ['sds'],
    'pp': ['pps', 'ppp'],
    'pd': ['pds', 'pdp'],
    'dd': ['dds', 'ddp', 'ddd'],
}

# Map individual orbital name → l-channel
_ORB_TO_L = {
    's': 's',
    'px': 'p',
    'py': 'p',
    'pz': 'p',
    'dxy': 'd',
    'dyz': 'd',
    'dzx': 'd',
    'dx2-y2': 'd',
    'dz2': 'd',
}

# Map d-orbital name → cubic-harmonic on-site group
_ORB_TO_D_GROUP = {
    'dxy': 't2g',
    'dyz': 't2g',
    'dzx': 't2g',
    'dx2-y2': 'eg',
    'dz2': 'eg',
}


# Angular-momentum sort key (s=0, p=1, d=2, f=3)
_L_ORDER = {'s': 0, 'p': 1, 'd': 2, 'f': 3}


# ═══════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════


def species_pair_key(sp1: str, sp2: str) -> str:
    """Canonical species-pair key (alphabetically sorted, hyphen-separated).

    Examples
    --------
    >>> species_pair_key("Ge", "Si")
    'Ge-Si'
    >>> species_pair_key("Pt", "Pt")
    'Pt-Pt'
    """
    a, b = sorted([sp1, sp2])
    return f'{a}-{b}'


def active_sk_names_for_basis(l_channels_a: List[str], l_channels_b: List[str]) -> List[str]:
    """Return the SK parameter names active for a species pair."""
    names = []
    seen = set()
    for la in l_channels_a:
        for lb in l_channels_b:
            lp = ''.join(sorted([la, lb], key=lambda x: _L_ORDER.get(x, 99)))
            if lp in LPAIR_SK_NAMES:
                for n in LPAIR_SK_NAMES[lp]:
                    if n not in seen:
                        names.append(n)
                        seen.add(n)
    return names


def active_gamma_labels(l_channels_a: List[str], l_channels_b: List[str]) -> List[str]:
    """Return active γ labels for a species pair."""
    labels = []
    seen = set()
    for la in l_channels_a:
        for lb in l_channels_b:
            lp = ''.join(sorted([la, lb], key=lambda x: _L_ORDER.get(x, 99)))
            if lp in LPAIR_LABELS and lp not in seen:
                labels.append(lp)
                seen.add(lp)
    return labels


def _orbital_to_onsite_group(orb: str, onsite_keys: set) -> str:
    """Map an orbital name to its on-site group key.

    Handles both flat (s, p, d) and crystal-field-split (s, p, t2g, eg).
    """
    l = _ORB_TO_L[orb]
    if l in onsite_keys:
        return l
    if l == 'd':
        group = _ORB_TO_D_GROUP[orb]
        if group in onsite_keys:
            return group
    raise KeyError(
        f"Cannot find on-site group for orbital '{orb}' in available keys {sorted(onsite_keys)}"
    )


# ═══════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════


def validate_params(params: dict) -> List[str]:
    """Validate an EDTB parameter dict against the schema.

    Returns
    -------
    list of str
        Error messages (empty list = valid).
    """
    errors = []

    # ── Version ──
    if 'edtb_version' not in params:
        errors.append("Missing 'edtb_version'")

    # ── Basis ──
    if 'basis' not in params:
        errors.append("Missing 'basis'")
        return errors  # cannot validate further

    basis = params['basis']
    species_list = sorted(basis.keys())

    for sp in species_list:
        b = basis[sp]
        if 'orbitals' not in b:
            errors.append(f"basis['{sp}']: missing 'orbitals'")
        if 'l_channels' not in b:
            errors.append(f"basis['{sp}']: missing 'l_channels'")
        # Consistency: every orbital must map to a declared l-channel
        if 'orbitals' in b and 'l_channels' in b:
            for orb in b['orbitals']:
                l = _ORB_TO_L.get(orb)
                if l is None:
                    errors.append(f"basis['{sp}']: unknown orbital '{orb}'")
                elif l not in b['l_channels']:
                    errors.append(
                        f"basis['{sp}']: orbital '{orb}' (l={l}) "
                        f'not in l_channels {b["l_channels"]}'
                    )

    # ── Onsite ──
    if 'onsite' not in params:
        errors.append("Missing 'onsite'")
    else:
        for sp in species_list:
            if sp not in params['onsite']:
                errors.append(f"onsite: missing species '{sp}'")

    # ── Hoppings ──
    if 'hoppings' not in params:
        errors.append("Missing 'hoppings'")
    else:
        for i, sp1 in enumerate(species_list):
            for sp2 in species_list[i:]:
                key = species_pair_key(sp1, sp2)
                if key not in params['hoppings']:
                    errors.append(f"hoppings: missing species pair '{key}'")
                else:
                    shells = params['hoppings'][key]
                    if isinstance(shells, dict) and shells.get('type') == 'distance_dependent':
                        # ── Distance-dependent format ──
                        for field in ('r_0', 'r_c', 'n_c', 'channels'):
                            if field not in shells:
                                errors.append(f"hoppings['{key}']: missing '{field}'")
                        if 'channels' in shells:
                            lc_a = basis.get(sp1, {}).get('l_channels', [])
                            lc_b = basis.get(sp2, {}).get('l_channels', [])
                            expected_sk = set(active_sk_names_for_basis(lc_a, lc_b))
                            got = set(shells['channels'].keys())
                            missing = expected_sk - got
                            if missing:
                                errors.append(
                                    f"hoppings['{key}']: missing DD channels "
                                    f'{sorted(missing)} for '
                                    f'l_channels {lc_a}\u00d7{lc_b}'
                                )
                            for ch_name, ch_val in shells['channels'].items():
                                if (
                                    not isinstance(ch_val, dict)
                                    or 'V0' not in ch_val
                                    or 'n' not in ch_val
                                ):
                                    errors.append(
                                        f"hoppings['{key}'].channels['{ch_name}']: "
                                        f"must have 'V0' and 'n'"
                                    )
                    elif not isinstance(shells, list) or len(shells) == 0:
                        errors.append(
                            f"hoppings['{key}']: must be a non-empty list of shells "
                            f'or a distance_dependent dict'
                        )
                    else:
                        # Expected SK param names for this species pair
                        lc_a = basis.get(sp1, {}).get('l_channels', [])
                        lc_b = basis.get(sp2, {}).get('l_channels', [])
                        expected_sk = set(active_sk_names_for_basis(lc_a, lc_b))
                        for s, shell in enumerate(shells):
                            if 'r_ref' not in shell:
                                errors.append(f"hoppings['{key}'][{s}]: missing 'r_ref'")
                            if 'params' not in shell:
                                errors.append(f"hoppings['{key}'][{s}]: missing 'params'")
                            elif expected_sk:
                                got = set(shell['params'].keys())
                                missing = expected_sk - got
                                extra = got - expected_sk
                                if missing:
                                    errors.append(
                                        f"hoppings['{key}'][{s}]: missing SK "
                                        f'params {sorted(missing)} for '
                                        f'l_channels {lc_a}\u00d7{lc_b}'
                                    )
                                if extra:
                                    errors.append(
                                        f"hoppings['{key}'][{s}]: unexpected "
                                        f'SK params {sorted(extra)} for '
                                        f'l_channels {lc_a}\u00d7{lc_b}'
                                    )

    # ── Screening (optional) ──
    if 'screening' in params:
        scr = params['screening']
        if 'r_cut' not in scr:
            errors.append("screening: missing 'r_cut'")
        if 'gamma' not in scr:
            errors.append("screening: missing 'gamma'")
        else:
            for i, sp1 in enumerate(species_list):
                for sp2 in species_list[i:]:
                    key = species_pair_key(sp1, sp2)
                    if key not in scr['gamma']:
                        errors.append(f"screening.gamma: missing species pair '{key}'")
                    elif isinstance(scr['gamma'][key], dict):
                        lc_a = basis.get(sp1, {}).get('l_channels', [])
                        lc_b = basis.get(sp2, {}).get('l_channels', [])
                        expected_g = set(active_gamma_labels(lc_a, lc_b))
                        got_g = set(scr['gamma'][key].keys())
                        missing_g = expected_g - got_g
                        extra_g = got_g - expected_g
                        if missing_g:
                            errors.append(
                                f"screening.gamma['{key}']: missing "
                                f'labels {sorted(missing_g)} for '
                                f'l_channels {lc_a}\u00d7{lc_b}'
                            )
                        if extra_g:
                            errors.append(
                                f"screening.gamma['{key}']: unexpected "
                                f'labels {sorted(extra_g)} for '
                                f'l_channels {lc_a}\u00d7{lc_b}'
                            )

    return errors


def validate_geometry(geometry: dict) -> List[str]:
    """Validate a geometry dict against the schema.

    Returns
    -------
    list of str
        Error messages (empty list = valid).
    """
    errors = []

    if 'alat' not in geometry:
        errors.append("Missing 'alat'")
    if 'a_vectors' not in geometry:
        errors.append("Missing 'a_vectors'")
    else:
        av = geometry['a_vectors']
        if not isinstance(av, list) or len(av) != 3:
            errors.append("'a_vectors' must be a list of 3 vectors")
        else:
            for i, v in enumerate(av):
                if not isinstance(v, list) or len(v) != 3:
                    errors.append(f'a_vectors[{i}]: must be a 3-element list')

    if 'atoms' not in geometry:
        errors.append("Missing 'atoms'")
    else:
        for i, atom in enumerate(geometry['atoms']):
            if 'species' not in atom:
                errors.append(f"atoms[{i}]: missing 'species'")
            if 'tau' not in atom:
                errors.append(f"atoms[{i}]: missing 'tau'")
            elif not isinstance(atom['tau'], list) or len(atom['tau']) != 3:
                errors.append(f"atoms[{i}]: 'tau' must be a 3-element list")

    return errors


# ═══════════════════════════════════════════════════════════════
#  Read / Write — Parameters
# ═══════════════════════════════════════════════════════════════


def write_params(filepath: Union[str, Path], params: dict, *, validate: bool = True) -> None:
    """Write EDTB parameters to a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Output path.
    params : dict
        EDTB parameter dict conforming to the schema.
    validate : bool
        If True, validate before writing.

    Raises
    ------
    ValueError
        If *validate* is True and the dict has schema violations.
    """
    if validate:
        errors = validate_params(params)
        if errors:
            raise ValueError('Invalid parameter dict:\n  ' + '\n  '.join(errors))
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)


def read_params(filepath: Union[str, Path], *, validate: bool = True) -> dict:
    """Read EDTB parameters from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Input path.
    validate : bool
        If True, validate after reading.

    Returns
    -------
    dict
        EDTB parameter dict.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        params = json.load(f)
    if validate:
        errors = validate_params(params)
        if errors:
            raise ValueError(f"Invalid parameter file '{filepath}':\n  " + '\n  '.join(errors))
    return params


# ═══════════════════════════════════════════════════════════════
#  Read / Write — Geometry
# ═══════════════════════════════════════════════════════════════


def write_geometry(filepath: Union[str, Path], geometry: dict, *, validate: bool = True) -> None:
    """Write a geometry file (JSON).

    Parameters
    ----------
    filepath : str or Path
        Output path.
    geometry : dict
        Geometry dict with keys ``alat``, ``a_vectors``, ``atoms``.
    validate : bool
        Validate before writing.
    """
    if validate:
        errors = validate_geometry(geometry)
        if errors:
            raise ValueError('Invalid geometry dict:\n  ' + '\n  '.join(errors))
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(geometry, f, indent=2)


def read_geometry(filepath: Union[str, Path], *, validate: bool = True) -> dict:
    """Read a geometry file (JSON).

    Returns
    -------
    dict
        Geometry dict with keys ``alat``, ``a_vectors``, ``atoms``.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        geometry = json.load(f)
    if validate:
        errors = validate_geometry(geometry)
        if errors:
            raise ValueError(f"Invalid geometry file '{filepath}':\n  " + '\n  '.join(errors))
    return geometry


# ═══════════════════════════════════════════════════════════════
#  Shell-distance computation (lattice + basis)
# ═══════════════════════════════════════════════════════════════


def compute_shell_distances(a_vectors, tau_list, n_shells=3, r_max=20.0, tol=0.01):
    """Compute neighbor-shell distances from lattice vectors and atomic basis.

    Parameters
    ----------
    a_vectors : array-like, shape (3, 3)
        Lattice vectors in Bohr (already scaled by alat).
    tau_list : list of array-like, shape (natoms, 3)
        Atomic positions in Bohr.
    n_shells : int
        Number of distinct shells to return.
    r_max : float
        Cutoff for neighbor search (Bohr).
    tol : float
        Distance tolerance for grouping into shells (Bohr).

    Returns
    -------
    list of float
        Sorted shell distances (length ≤ n_shells).
    """
    a_vecs = np.asarray(a_vectors)
    taus = [np.asarray(t) for t in tau_list]
    natoms = len(taus)

    nmax = int(np.ceil(r_max / np.min(np.linalg.norm(a_vecs, axis=1)))) + 1
    all_dists = set()

    for ia in range(natoms):
        for ib in range(natoms):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in range(-nmax, nmax + 1):
                        if ia == ib and n1 == 0 and n2 == 0 and n3 == 0:
                            continue
                        R = taus[ib] - taus[ia] + n1 * a_vecs[0] + n2 * a_vecs[1] + n3 * a_vecs[2]
                        d = np.linalg.norm(R)
                        if 0 < d < r_max:
                            all_dists.add(round(d, 6))

    # Group into shells by distance tolerance
    shells = []
    for d in sorted(all_dists):
        if not shells or abs(d - shells[-1]) > tol:
            shells.append(d)
        if len(shells) >= n_shells:
            break

    return shells


def compute_pair_shell_distances(a_vectors, atoms, sp1, sp2, n_shells=3, r_max=20.0, tol=0.01):
    """Compute neighbor-shell distances for a specific species pair.

    Unlike ``compute_shell_distances`` (which pools all atom pairs),
    this considers only bonds from *sp1* atoms to *sp2* atoms.

    Parameters
    ----------
    a_vectors : array-like, shape (3, 3)
        Lattice vectors in Bohr (already scaled by alat).
    atoms : list of dict
        Atom dicts with ``"species"`` and ``"tau"`` keys (positions in Bohr).
    sp1, sp2 : str
        Species names.
    n_shells : int
        Number of distinct shells to return.
    r_max : float
        Cutoff for neighbor search (Bohr).
    tol : float
        Distance tolerance for grouping into shells (Bohr).

    Returns
    -------
    list of float
        Sorted shell distances for the *sp1*--*sp2* pair (length <= *n_shells*).
    """
    a_vecs = np.asarray(a_vectors)
    nmax = int(np.ceil(r_max / np.min(np.linalg.norm(a_vecs, axis=1)))) + 1

    idx_a = [i for i, at in enumerate(atoms) if at['species'] == sp1]
    idx_b = [i for i, at in enumerate(atoms) if at['species'] == sp2]
    taus = [np.asarray(at['tau']) for at in atoms]

    all_dists: set = set()
    for ia in idx_a:
        for ib in idx_b:
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in range(-nmax, nmax + 1):
                        if ia == ib and n1 == 0 and n2 == 0 and n3 == 0:
                            continue
                        R = taus[ib] - taus[ia] + n1 * a_vecs[0] + n2 * a_vecs[1] + n3 * a_vecs[2]
                        d = float(np.linalg.norm(R))
                        if 0 < d < r_max:
                            all_dists.add(round(d, 6))

    shells: List[float] = []
    for d in sorted(all_dists):
        if not shells or abs(d - shells[-1]) > tol:
            shells.append(d)
        if len(shells) >= n_shells:
            break

    return shells


# ═══════════════════════════════════════════════════════════════
#  Conversion: old model dict → new EDTB param + geometry dicts
# ═══════════════════════════════════════════════════════════════


def from_model_dict(
    model_dict: dict, shell_distances: Optional[Dict[str, float]] = None
) -> Tuple[dict, dict]:
    """Convert a PAOFLOW model dict to the new schema.

    Accepts **both** the old shell-tag-keyed format::

        "hoppings": {"nn": {"sss": ...}, "nnn": {...}}

    and the new species-pair-keyed format::

        "hoppings": {"Pt-Pt": [{"r_ref": 5.247, "params": {"sss": ...}}, ...]}

    Format detection is automatic.

    Parameters
    ----------
    model_dict : dict
        Model dict with keys
        ``label``, ``alat``, ``model.{a_vectors, atoms, hoppings, screening}``.
    shell_distances : dict, optional
        *(Old format only)* Shell tag → reference distance (Bohr), e.g.
        ``{"nn": 5.247, "nnn": 7.420, "nnnn": 9.090}``.
        If None, distances are computed automatically from the lattice.

    Returns
    -------
    params : dict
        New-format EDTB parameter dict.
    geometry : dict
        New-format geometry dict.
    """
    m = model_dict['model']
    alat = float(model_dict.get('alat', 1.0))
    label = model_dict.get('label', '')

    # ── Detect hopping format ──
    first_hop_value = next(iter(m['hoppings'].values()))
    dd_format = isinstance(first_hop_value, dict) and 'channels' in first_hop_value
    new_format = isinstance(first_hop_value, list)

    # ── Parse atoms ──
    atoms_old = m['atoms']
    natoms = len(atoms_old)

    species_info: Dict[str, dict] = {}
    geom_atoms = []

    # Map configuration label (e.g. '2S', '3P', '3D') → l-channel
    _CONFIG_TO_L = {'S': 's', 'P': 'p', 'D': 'd', 'F': 'f'}

    for ia in range(natoms):
        atom = atoms_old[str(ia)]
        sp = atom['name']
        geom_atoms.append({'species': sp, 'tau': atom['tau']})

        if sp not in species_info:
            if 'configuration' in atom:
                # Configuration-based atom (e.g. ['2S', '2P', '3D'])
                config = atom['configuration']
                orbitals = []
                l_channels = []
                for cfg_label in config:
                    l_char = cfg_label[-1].upper()  # '2S' → 'S'
                    lc = _CONFIG_TO_L.get(l_char)
                    if lc:
                        orbitals.extend(L_ORBITALS[lc])
                        if lc not in l_channels:
                            l_channels.append(lc)
                species_info[sp] = {'orbitals': orbitals, 'l_channels': l_channels}
            else:
                # Orbital-based atom (e.g. orbitals=['s','px','py','pz'])
                orbitals = list(atom.get('orbitals', []))
                l_channels = []
                for orb in orbitals:
                    l = _ORB_TO_L.get(orb)
                    if l and l not in l_channels:
                        l_channels.append(l)
                species_info[sp] = {'orbitals': orbitals, 'l_channels': l_channels}

    # ── Basis ──
    basis = {sp: dict(info) for sp, info in species_info.items()}

    # ── Onsite ──
    onsite: Dict[str, dict] = {}
    for ia in range(natoms):
        atom = atoms_old[str(ia)]
        sp = atom['name']
        if sp in onsite:
            continue
        l_channels = species_info[sp]['l_channels']
        on: Dict[str, float] = {}

        if 'configuration' in atom:
            # Configuration-based: keys like '2S', '2P', '3D',
            # and for d-orbitals possibly '3D_t2g' / '3D_eg'
            config = atom['configuration']
            for cfg_label in config:
                l_char = cfg_label[-1].upper()
                lc = _CONFIG_TO_L.get(l_char)
                if not lc:
                    continue
                if lc == 'd':
                    key_t2g = f'{cfg_label}_t2g'
                    key_eg = f'{cfg_label}_eg'
                    if key_t2g in atom and key_eg in atom:
                        val_t2g = atom[key_t2g]
                        val_eg = atom[key_eg]
                        if abs(val_t2g - val_eg) > 1e-10:
                            on['t2g'] = val_t2g
                            on['eg'] = val_eg
                        else:
                            on['d'] = val_t2g
                    elif cfg_label in atom:
                        on['d'] = atom[cfg_label]
                else:
                    if cfg_label in atom:
                        on[lc] = atom[cfg_label]
        else:
            # Orbital-based: keys like 's', 'px', 'dxy'
            for lc in l_channels:
                if lc in ('s', 'p'):
                    rep = L_ORBITALS[lc][0]  # "s" or "px"
                    if rep in atom:
                        on[lc] = atom[rep]
                elif lc == 'd':
                    val_t2g = atom.get('dxy')
                    val_eg = atom.get('dx2-y2')
                    if val_t2g is not None and val_eg is not None:
                        if abs(val_t2g - val_eg) > 1e-10:
                            on['t2g'] = val_t2g
                            on['eg'] = val_eg
                        else:
                            on['d'] = val_t2g
                    elif val_t2g is not None:
                        on['d'] = val_t2g
        onsite[sp] = on

    species_list = sorted(species_info.keys())

    if dd_format:
        # ── Distance-dependent format: pass through ──
        hoppings = {}
        for k, v in m['hoppings'].items():
            hoppings[k] = {
                'type': 'distance_dependent',
                'r_0': v['r_0'],
                'r_c': v['r_c'],
                'n_c': v['n_c'],
                'channels': {ch: dict(cp) for ch, cp in v['channels'].items()},
            }

        # ── Screening ──
        screening = None
        if 'screening' in m:
            scr = m['screening']
            gamma_raw = scr['gamma']
            if isinstance(gamma_raw, dict):
                first_g = next(iter(gamma_raw.values()))
                if isinstance(first_g, (dict, float, int)):
                    if any('-' in k for k in gamma_raw):
                        gamma = {
                            k: (dict(v) if isinstance(v, dict) else v) for k, v in gamma_raw.items()
                        }
                    else:
                        gamma = {}
                        for i, sp1 in enumerate(species_list):
                            for sp2 in species_list[i:]:
                                key = species_pair_key(sp1, sp2)
                                gamma[key] = dict(gamma_raw)
            else:
                gamma = {}
                for i, sp1 in enumerate(species_list):
                    for sp2 in species_list[i:]:
                        key = species_pair_key(sp1, sp2)
                        gamma[key] = float(gamma_raw)
            screening = {'r_cut': scr['r_cut'], 'gamma': gamma}
            if 'onsite_shift' in scr:
                screening['onsite_shift'] = dict(scr['onsite_shift'])

    elif new_format:
        # ── New format: hoppings are already species-pair-keyed ──
        hoppings = {k: list(v) for k, v in m['hoppings'].items()}

        # ── Screening ──
        screening = None
        if 'screening' in m:
            scr = m['screening']
            gamma_raw = scr['gamma']
            # gamma may already be pair-keyed or bare
            if isinstance(gamma_raw, dict):
                first_g = next(iter(gamma_raw.values()))
                if isinstance(first_g, (dict, float, int)):
                    # Check if top-level keys are species-pair keys (contain '-')
                    if any('-' in k for k in gamma_raw):
                        gamma = {
                            k: (dict(v) if isinstance(v, dict) else v) for k, v in gamma_raw.items()
                        }
                    else:
                        # Bare l-pair dict → wrap for every species pair
                        gamma = {}
                        for i, sp1 in enumerate(species_list):
                            for sp2 in species_list[i:]:
                                key = species_pair_key(sp1, sp2)
                                gamma[key] = dict(gamma_raw)
            else:
                gamma = {}
                for i, sp1 in enumerate(species_list):
                    for sp2 in species_list[i:]:
                        key = species_pair_key(sp1, sp2)
                        gamma[key] = float(gamma_raw)
            screening = {'r_cut': scr['r_cut'], 'gamma': gamma}
            if 'onsite_shift' in scr:
                screening['onsite_shift'] = dict(scr['onsite_shift'])
    else:
        # ── Old format: shell-tag keyed, species-blind ──
        shell_tags = sorted(m['hoppings'].keys(), key=len)
        a_vecs_bohr = np.array(m['a_vectors']) * alat

        # Build atom list in Bohr for per-pair distance computation
        atoms_bohr = [
            {
                'species': atoms_old[str(ia)]['name'],
                'tau': (np.array(atoms_old[str(ia)]['tau']) * alat).tolist(),
            }
            for ia in range(natoms)
        ]

        # ── Hoppings ──
        # Old format is species-blind → same SK params for all pairs,
        # but r_ref is computed per pair from the actual lattice geometry.
        hoppings: Dict[str, list] = {}
        for i, sp1 in enumerate(species_list):
            for sp2 in species_list[i:]:
                key = species_pair_key(sp1, sp2)

                # Per-pair shell distances
                if shell_distances is not None:
                    pair_dists = shell_distances
                else:
                    dists = compute_pair_shell_distances(
                        a_vecs_bohr,
                        atoms_bohr,
                        sp1,
                        sp2,
                        n_shells=len(shell_tags),
                    )
                    pair_dists = {}
                    for j, tag in enumerate(shell_tags):
                        pair_dists[tag] = dists[j] if j < len(dists) else 0.0

                shells = []
                for tag in shell_tags:
                    hop_data = m['hoppings'][tag]
                    shells.append(
                        {
                            'r_ref': round(pair_dists[tag], 6),
                            'params': {k: v for k, v in hop_data.items()},
                        }
                    )
                hoppings[key] = shells

        # ── Screening ──
        screening = None
        if 'screening' in m:
            scr = m['screening']
            gamma_raw = scr['gamma']
            gamma: Dict[str, Any] = {}
            for i, sp1 in enumerate(species_list):
                for sp2 in species_list[i:]:
                    key = species_pair_key(sp1, sp2)
                    if isinstance(gamma_raw, dict):
                        gamma[key] = {k: v for k, v in gamma_raw.items()}
                    else:
                        gamma[key] = float(gamma_raw)
            screening = {'r_cut': scr['r_cut'], 'gamma': gamma}

    # ── Assemble ──
    params = {
        'edtb_version': CURRENT_VERSION,
        'description': f"Converted from '{label}' model",
        'basis': basis,
        'onsite': onsite,
        'hoppings': hoppings,
    }
    if screening is not None:
        params['screening'] = screening

    geometry = {
        'alat': alat,
        'a_vectors': m['a_vectors'],
        'atoms': geom_atoms,
    }

    return params, geometry


# ═══════════════════════════════════════════════════════════════
#  Conversion: new EDTB params + geometry → old PAOFLOW model dict
# ═══════════════════════════════════════════════════════════════


def to_model_dict(params: dict, geometry: dict) -> dict:
    """Convert new-format params + geometry to a model dict.

    Emits the species-pair-keyed hopping format, consistent with
    ``sk_fitting.build_model_dict``.

    Parameters
    ----------
    params : dict
        EDTB parameter dict (new format).
    geometry : dict
        Geometry dict (new format).

    Returns
    -------
    dict
        Model dict with species-pair-keyed hoppings.
    """
    # species_list = sorted(params['basis'].keys())

    # ── Atoms ──
    atoms_dict = {}
    for ia, atom in enumerate(geometry['atoms']):
        sp = atom['species']
        basis_sp = params['basis'][sp]
        on = params['onsite'][sp]
        atom_d = {
            'name': sp,
            'tau': atom['tau'],
            'orbitals': list(basis_sp['orbitals']),
        }
        on_keys = set(on.keys())
        for orb in basis_sp['orbitals']:
            group = _orbital_to_onsite_group(orb, on_keys)
            atom_d[orb] = on[group]
        atoms_dict[str(ia)] = atom_d

    # ── Hoppings (pass-through, species-pair-keyed) ──
    hoppings = {}
    for pair_key, shells in params['hoppings'].items():
        if isinstance(shells, dict) and shells.get('type') == 'distance_dependent':
            hoppings[pair_key] = {
                'type': 'distance_dependent',
                'r_0': shells['r_0'],
                'r_c': shells['r_c'],
                'n_c': shells['n_c'],
                'channels': {ch: dict(cp) for ch, cp in shells['channels'].items()},
            }
        else:
            hoppings[pair_key] = [
                {'r_ref': shell['r_ref'], 'params': dict(shell['params'])} for shell in shells
            ]

    # ── Model dict ──
    has_screening = 'screening' in params
    label = 'SK_EDTB' if has_screening else 'Slater_Koster'

    model = {
        'label': label,
        'alat': geometry['alat'],
        'model': {
            'a_vectors': geometry['a_vectors'],
            'atoms': atoms_dict,
            'hoppings': hoppings,
        },
    }

    if has_screening:
        scr = params['screening']
        gamma = {k: (dict(v) if isinstance(v, dict) else v) for k, v in scr['gamma'].items()}
        screening_out: Dict[str, Any] = {'r_cut': scr['r_cut'], 'gamma': gamma}
        if 'onsite_shift' in scr:
            screening_out['onsite_shift'] = dict(scr['onsite_shift'])
        model['model']['screening'] = screening_out

    return model


# ═══════════════════════════════════════════════════════════════
#  Pretty-print summary
# ═══════════════════════════════════════════════════════════════


def summarize_params(params: dict) -> str:
    """Return a human-readable summary of an EDTB parameter dict."""
    lines = []
    lines.append(f'EDTB Parameters  (v{params.get("edtb_version", "?")})')
    if 'description' in params:
        lines.append(f'  {params["description"]}')

    basis = params.get('basis', {})
    species = sorted(basis.keys())
    lines.append(f'\nSpecies: {", ".join(species)}')

    for sp in species:
        b = basis[sp]
        lines.append(
            f'  {sp}: l_channels={b.get("l_channels", "?")}, norb={len(b.get("orbitals", []))}'
        )
        on = params.get('onsite', {}).get(sp, {})
        on_str = ', '.join(f'{k}={v:.4f}' for k, v in on.items())
        lines.append(f'    onsite: {on_str}')

    hoppings = params.get('hoppings', {})
    lines.append(f'\nHopping blocks: {len(hoppings)} species pairs')
    for key in sorted(hoppings.keys()):
        shells = hoppings[key]
        if isinstance(shells, dict) and shells.get('type') == 'distance_dependent':
            lines.append(
                f'  {key}: distance-dependent  r_0={shells.get("r_0", "?")}  r_c={shells.get("r_c", "?")} Bohr'
            )
            channels = shells.get('channels', {})
            for ch, cp in sorted(channels.items()):
                lines.append(f'    {ch}: V0={cp["V0"]:.4f}, n={cp["n"]:.4f}')
        elif isinstance(shells, list):
            lines.append(f'  {key}: {len(shells)} shells')
            for s, sh in enumerate(shells):
                r = sh.get('r_ref', '?')
                npar = len(sh.get('params', {}))
                lines.append(f'    shell {s + 1}: r_ref={r} Bohr, {npar} params')
        else:
            lines.append(f'  {key}: {shells}')

    if 'screening' in params:
        scr = params['screening']
        lines.append(f'\nScreening: r_cut={scr["r_cut"]} Bohr')
        gamma = scr.get('gamma', {})
        for key in sorted(gamma.keys()):
            g = gamma[key]
            if isinstance(g, dict):
                g_str = ', '.join(f'{k}={v:.4f}' for k, v in g.items())
            else:
                g_str = f'global={g:.4f}'
            lines.append(f'  γ[{key}]: {g_str}')
    else:
        lines.append('\nScreening: none (pure SK)')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  EDTBModel — user-facing interface class
# ═══════════════════════════════════════════════════════════════


class EDTBModel:
    """User-facing interface for environment-dependent tight-binding models.

    Holds validated parameter and geometry data and provides a clean API
    for saving, loading, converting, and computing band structures.

    Construction
    ------------
    Direct::

        model = EDTBModel(params_dict, geometry_dict)

    From files::

        model = EDTBModel.from_files("params.json", "geometry.json")

    From a PAOFLOW model dict (old *or* new format)::

        model = EDTBModel.from_model_dict(model_dict)

    From a fitted ``SKFitter`` / ``SKFitterEDTB``::

        model = EDTBModel.from_fitter(fitter, p_opt)

    Serialisation
    -------------
    ::

        model.save("params.json", "geometry.json")
        md = model.to_model_dict()  # for PAOFLOW.PAOFLOW(model=md)

    Band computation
    ----------------
    ::

        result = model.compute_bands(ibrav=2, nk=500)
        print(result["bands_file"])   # path to bands_0.dat

    Transferability
    ---------------
    ::

        new_model = model.with_geometry(new_geometry)  # same params, new cell
    """

    # ── Constructor ───────────────────────────────────────────

    def __init__(
        self,
        params: dict,
        geometry: dict,
        *,
        validate: bool = True,
    ):
        """Create an EDTB model from parameter and geometry dicts.

        Parameters
        ----------
        params : dict
            EDTB parameter dict conforming to the ``edtb_params`` schema.
        geometry : dict
            Geometry dict with keys ``alat``, ``a_vectors``, ``atoms``.
        validate : bool
            If True (default), validate both dicts on construction.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if validate:
            errors = validate_params(params)
            if errors:
                raise ValueError('Invalid parameter dict:\n  ' + '\n  '.join(errors))
            errors = validate_geometry(geometry)
            if errors:
                raise ValueError('Invalid geometry dict:\n  ' + '\n  '.join(errors))
        # Store deep copies to prevent mutation
        import copy

        self._params = copy.deepcopy(params)
        self._geometry = copy.deepcopy(geometry)

    # ── Class-method constructors ─────────────────────────────

    @classmethod
    def from_files(
        cls,
        params_path: Union[str, Path],
        geometry_path: Union[str, Path],
        *,
        validate: bool = True,
    ) -> 'EDTBModel':
        """Load an EDTB model from JSON parameter and geometry files.

        Parameters
        ----------
        params_path : str or Path
            Path to the parameter file.
        geometry_path : str or Path
            Path to the geometry file.
        validate : bool
            Validate after loading (default True).

        Returns
        -------
        EDTBModel
        """
        params = read_params(params_path, validate=validate)
        geometry = read_geometry(geometry_path, validate=validate)
        return cls(params, geometry, validate=False)

    @classmethod
    def from_model_dict(
        cls,
        model_dict: dict,
        *,
        shell_distances: Optional[Dict[str, float]] = None,
    ) -> 'EDTBModel':
        """Convert a PAOFLOW model dict to an EDTBModel.

        Accepts **both** the old shell-tag-keyed format
        (``"nn"``/``"nnn"``/``"nnnn"`` keys) and the new species-pair-keyed
        format.  Detection is automatic.

        Parameters
        ----------
        model_dict : dict
            PAOFLOW model dict with ``label``, ``alat``,
            ``model.{a_vectors, atoms, hoppings, [screening]}``.
        shell_distances : dict, optional
            *(Old format only)* Shell tag → distance mapping (Bohr).
            If None, distances are computed from the lattice geometry.

        Returns
        -------
        EDTBModel
        """
        params, geometry = from_model_dict(model_dict, shell_distances=shell_distances)
        return cls(params, geometry, validate=True)

    @classmethod
    def from_fitter(cls, fitter, p_opt) -> 'EDTBModel':
        """Build an EDTBModel from a fitted SKFitter or SKFitterEDTB.

        Calls ``fitter.build_model_dict(p_opt)`` internally and converts
        the result to the new schema.

        Parameters
        ----------
        fitter : SKFitter | SKFitterEDTB | MultiGeomEDTB
            Fitted fitter object.
        p_opt : array-like
            Optimised parameter vector.

        Returns
        -------
        EDTBModel
        """
        model_dict = fitter.build_model_dict(np.asarray(p_opt))
        return cls.from_model_dict(model_dict)

    # ── Serialisation ─────────────────────────────────────────

    def save(
        self,
        params_path: Union[str, Path],
        geometry_path: Optional[Union[str, Path]] = None,
        *,
        validate: bool = True,
    ) -> None:
        """Write parameter (and optionally geometry) files.

        Parameters
        ----------
        params_path : str or Path
            Output path for the parameter file.
        geometry_path : str or Path, optional
            Output path for the geometry file.  If None, only the
            parameter file is written.
        validate : bool
            Validate before writing (default True).
        """
        write_params(params_path, self._params, validate=validate)
        if geometry_path is not None:
            write_geometry(geometry_path, self._geometry, validate=validate)

    def to_model_dict(self) -> dict:
        """Convert to a PAOFLOW model dict.

        The returned dict can be passed directly to
        ``PAOFLOW.PAOFLOW(model=...)``.

        Returns
        -------
        dict
            Model dict with species-pair-keyed hoppings.
        """
        return to_model_dict(self._params, self._geometry)

    # ── Geometry transfer ─────────────────────────────────────

    def with_geometry(self, geometry: dict) -> 'EDTBModel':
        """Return a new EDTBModel with the same parameters but a different geometry.

        This is the main transferability mechanism: train once,
        apply to arbitrary cells / strains / defects with the same
        species.

        Parameters
        ----------
        geometry : dict
            New geometry dict (``alat``, ``a_vectors``, ``atoms``).

        Returns
        -------
        EDTBModel
            New model sharing the same parameters.

        Raises
        ------
        ValueError
            If the new geometry contains species not in the model.
        """
        # Check species compatibility
        geom_species = {a['species'] for a in geometry['atoms']}
        model_species = set(self._params['basis'].keys())
        unknown = geom_species - model_species
        if unknown:
            raise ValueError(
                f'Geometry contains species not in the model: '
                f'{sorted(unknown)}.  Model species: {sorted(model_species)}.'
            )
        return EDTBModel(self._params, geometry)

    @classmethod
    def from_geometry_file(
        cls,
        params_path: Union[str, Path],
        geometry_path: Union[str, Path],
    ) -> 'EDTBModel':
        """Shortcut: load parameters from one file, geometry from another.

        Identical to :meth:`from_files` — provided for readability in
        transferability workflows where the same params file is reused.
        """
        return cls.from_files(params_path, geometry_path)

    # ── Band computation ──────────────────────────────────────

    def compute_bands(
        self,
        *,
        ibrav: int = 0,
        nk: int = 500,
        outputdir: Optional[str] = None,
        band_path: Optional[str] = None,
        high_sym_points: Optional[dict] = None,
        smearing: str = 'gauss',
        verbose: bool = False,
    ) -> dict:
        """Compute the band structure using PAOFLOW.

        Parameters
        ----------
        ibrav : int
            Bravais lattice type (default 0).
        nk : int
            Number of k-points along the path (default 500).
        outputdir : str, optional
            Directory for output files.  If None, a temporary directory
            is used (based on the model label and alat).
        band_path : str, optional
            Custom band path (e.g. ``"L-G-X"``).
        high_sym_points : dict, optional
            Custom high-symmetry point coordinates.
        smearing : str
            Smearing type (default ``"gauss"``).
        verbose : bool
            Print PAOFLOW output (default False).

        Returns
        -------
        dict
            ``bands_file`` : str — path to ``bands_0.dat``
            ``sym_file`` : str — path to ``kpath_points.txt``
            ``paoflow`` : PAOFLOW object (for further analysis)
        """
        from PAOFLOW import PAOFLOW as PF

        model_dict = self.to_model_dict()

        if outputdir is None:
            outputdir = f'edtb_{self.label}_{self.geometry["alat"]:.2f}'

        pao = PF.PAOFLOW(
            savedir=None,
            model=model_dict,
            outputdir=outputdir,
            smearing=smearing,
            verbose=verbose,
        )
        arry, attr = pao.data_controller.data_dicts()

        bands_kw = {'ibrav': ibrav, 'nk': nk}
        if band_path is not None:
            bands_kw['band_path'] = band_path
        if high_sym_points is not None:
            bands_kw['high_sym_points'] = high_sym_points

        pao.bands(**bands_kw)

        bands_file = f'{attr["outputdir"]}/bands_0.dat'
        sym_file = f'{attr["outputdir"]}/kpath_points.txt'

        return {
            'bands_file': bands_file,
            'sym_file': sym_file,
            'paoflow': pao,
        }

    # ── Properties ────────────────────────────────────────────

    @property
    def params(self) -> dict:
        """The EDTB parameter dict (read-only copy)."""
        import copy

        return copy.deepcopy(self._params)

    @property
    def geometry(self) -> dict:
        """The geometry dict (read-only copy)."""
        import copy

        return copy.deepcopy(self._geometry)

    @property
    def species(self) -> List[str]:
        """Sorted list of species in the model."""
        return sorted(self._params['basis'].keys())

    @property
    def n_species(self) -> int:
        """Number of distinct species."""
        return len(self._params['basis'])

    @property
    def n_shells(self) -> int:
        """Maximum number of neighbor shells across species pairs."""
        hoppings = self._params.get('hoppings', {})
        if not hoppings:
            return 0
        return max(len(shells) for shells in hoppings.values())

    @property
    def has_screening(self) -> bool:
        """Whether screening (EDTB) parameters are present."""
        return 'screening' in self._params

    @property
    def label(self) -> str:
        """Model label: ``'SK_EDTB'`` or ``'Slater_Koster'``."""
        return 'SK_EDTB' if self.has_screening else 'Slater_Koster'

    @property
    def alat(self) -> float:
        """Lattice parameter from the geometry (Bohr)."""
        return self._geometry['alat']

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        header = f'EDTBModel  [{self.label}]  alat={self.alat:.4f} Bohr'
        geom_info = f'  geometry: {len(self._geometry["atoms"])} atoms, species={self.species}'
        return header + '\n' + geom_info + '\n' + summarize_params(self._params)

    def __repr__(self) -> str:
        return (
            f'EDTBModel(label={self.label!r}, '
            f'species={self.species}, '
            f'n_shells={self.n_shells}, '
            f'alat={self.alat:.4f})'
        )

    def __str__(self) -> str:
        return self.summary()
