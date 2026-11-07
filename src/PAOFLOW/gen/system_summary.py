"""What a Quantum ESPRESSO run tells the generator: save, ``pw.x`` input and pseudopotential headers.

The summaries are light (an XML/namelist read, no PAOFLOW objects) and feed two consumers: the
defaults of the generator questions (spin-orbit, 2D, metal, valence electrons, the coupling source
for ultrasoft/PAW pseudopotentials) and the system card of the browser GUI.  Every function returns
``None`` or an empty value when the files are missing or unreadable; nothing raises on bad input.
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903

#: Pseudopotential types that are not norm-conserving (no native DFPT vertex).
AUGMENTED_TYPES = ('US', 'USPP', 'PAW')


def read_upf_header(path: str) -> dict[str, Any] | None:
    """Element, type, relativistic treatment and valence charge of a UPF file (v1 or v2).

    Parameters
    ----------
    path : str
        The UPF file.

    Returns
    -------
    dict or None
        ``element``, ``pseudo_type`` (``'NC'``, ``'SL'``, ``'US'``, ``'PAW'``),
        ``relativistic`` (``'scalar'``, ``'full'``, ``'no'`` or ``None``), ``z_valence`` and
        ``functional``; ``None`` when the file cannot be read.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            head = handle.read(20000)
    except OSError:
        return None
    match = re.search(r'<PP_HEADER\b(.*?)/?>', head, re.S)
    if match and '=' in match.group(1):
        attributes = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', match.group(1)))
        z_valence = attributes.get('z_valence')
        return {
            'element': attributes.get('element', '').strip(),
            'pseudo_type': attributes.get('pseudo_type', '').strip().upper(),
            'relativistic': attributes.get('relativistic', '').strip().lower() or None,
            'z_valence': float(z_valence) if z_valence else None,
            'functional': attributes.get('functional', '').strip(),
        }
    block = re.search(r'<PP_HEADER>(.*?)</PP_HEADER>', head, re.S)
    if block is None:
        return None
    lines = [line.split() for line in block.group(1).strip().splitlines()]
    try:
        element = lines[1][0]
        pseudo_type = lines[2][0].upper()
        z_line = next(line for line in lines if 'valence' in ' '.join(line).lower())
        z_valence = float(z_line[0])
    except (IndexError, StopIteration, ValueError):
        return None
    return {
        'element': element,
        'pseudo_type': pseudo_type,
        'relativistic': None,
        'z_valence': z_valence,
        'functional': '',
    }


def _float(element: ET.Element | None) -> float | None:
    if element is None or element.text is None:
        return None
    try:
        return float(element.text)
    except ValueError:
        return None


def _bool(element: ET.Element | None) -> bool:
    return element is not None and (element.text or '').strip().lower() == 'true'


def _formula(names: list[str]) -> str:
    counts = Counter(names)
    return ''.join('%s%s' % (name, count if count > 1 else '') for name, count in counts.items())


def is_two_dimensional(cell_bohr: np.ndarray, positions_bohr: np.ndarray) -> bool:
    """Whether a cell holds a slab: vacuum along c (c ⊥ the plane, atoms within half of it).

    Parameters
    ----------
    cell_bohr : numpy.ndarray, shape ``(3, 3)``
        Lattice vectors (rows).
    positions_bohr : numpy.ndarray, shape ``(nat, 3)``
        Cartesian positions.

    Returns
    -------
    bool

    Notes
    -----
    The atoms are projected on the unit normal of the a-b plane,

    .. math::

        z_i = \\mathbf{r}_i \\cdot \\hat{n}, \\qquad \\hat{n} = \\frac{\\mathbf{a}\\times\\mathbf{b}}
        {|\\mathbf{a}\\times\\mathbf{b}|},

    and the cell is a slab when the height :math:`h = \\mathbf{c}\\cdot\\hat{n}` exceeds both
    in-plane lengths and the atoms occupy less than half of it (periodic images folded).
    """
    a, b, c = np.asarray(cell_bohr, dtype=float)
    normal = np.cross(a, b)
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        return False
    normal /= norm
    height = abs(float(np.dot(c, normal)))
    if height <= max(np.linalg.norm(a), np.linalg.norm(b)):
        return False
    heights = np.sort(np.mod(np.asarray(positions_bohr, dtype=float) @ normal, height))
    if heights.size < 2:
        return True
    gaps = np.diff(np.concatenate([heights, [heights[0] + height]]))
    return float(gaps.max()) > 0.5 * height


def read_save_summary(savedir: str) -> dict[str, Any] | None:
    """Summary of a ``<prefix>.save`` folder (its ``data-file-schema.xml``).

    Parameters
    ----------
    savedir : str
        The save folder.

    Returns
    -------
    dict or None
        ``prefix``, ``formula``, ``nat``, ``species`` (name, mass, pseudo file and its UPF
        header), ``alat_bohr``, ``cell_bohr``, ``bravais_index``, ``lsda``, ``noncolin``,
        ``spinorbit``, ``nbnd``, ``nelec``, ``fermi_energy_ev``, ``homo_ev``, ``lumo_ev``,
        ``gap_ev``, ``occupations``, ``kgrid``, ``nks``, ``functional``, ``metal``,
        ``two_dimensional`` and ``augmented`` (any US/PAW pseudopotential); ``None`` when the
        file is missing or not a QE schema.
    """
    path = os.path.join(savedir, 'data-file-schema.xml')
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    output = root.find('output')
    if output is None:
        return None
    structure = output.find('atomic_structure')
    bands = output.find('band_structure')
    if structure is None or bands is None:
        return None
    names, positions = [], []
    for atom in structure.findall('atomic_positions/atom'):
        names.append(atom.attrib.get('name', '?'))
        positions.append([float(x) for x in (atom.text or '0 0 0').split()])
    cell = np.array(
        [
            [float(x) for x in (structure.find('cell/' + v).text or '').split()]
            for v in ('a1', 'a2', 'a3')
        ]
    )
    species = []
    for entry in output.findall('atomic_species/species'):
        pseudo = (entry.findtext('pseudo_file') or '').strip()
        upf = os.path.join(savedir, pseudo) if pseudo else ''
        species.append(
            {
                'name': entry.attrib.get('name', '?'),
                'mass_amu': _float(entry.find('mass')),
                'pseudo_file': pseudo,
                'pseudo': read_upf_header(upf) if upf else None,
            }
        )
    homo = _float(bands.find('highestOccupiedLevel'))
    lumo = _float(bands.find('lowestUnoccupiedLevel'))
    fermi = _float(bands.find('fermi_energy'))
    occupations = (bands.findtext('occupations_kind') or '').strip() or None
    grid = bands.find('starting_k_points/monkhorst_pack')
    kgrid = (
        [int(grid.attrib[k]) for k in ('nk1', 'nk2', 'nk3')]
        if grid is not None and all(k in grid.attrib for k in ('nk1', 'nk2', 'nk3'))
        else None
    )
    gap = (lumo - homo) * HARTREE_EV if homo is not None and lumo is not None else None
    smearing = output.find('band_structure/smearing')
    return {
        'prefix': os.path.splitext(os.path.basename(os.path.normpath(savedir)))[0],
        'formula': _formula(names),
        'nat': len(names),
        'species': species,
        'alat_bohr': float(structure.attrib.get('alat', 'nan')),
        'cell_bohr': cell.tolist(),
        'bravais_index': int(structure.attrib['bravais_index'])
        if structure.attrib.get('bravais_index')
        else None,
        'lsda': _bool(bands.find('lsda')),
        'noncolin': _bool(bands.find('noncolin')),
        'spinorbit': _bool(bands.find('spinorbit')),
        'nbnd': int(bands.findtext('nbnd') or 0),
        'nelec': _float(bands.find('nelec')),
        'fermi_energy_ev': fermi * HARTREE_EV if fermi is not None else None,
        'homo_ev': homo * HARTREE_EV if homo is not None else None,
        'lumo_ev': lumo * HARTREE_EV if lumo is not None else None,
        'gap_ev': gap,
        'occupations': occupations,
        'smearing': smearing.text.strip() if smearing is not None and smearing.text else None,
        'kgrid': kgrid,
        'nks': int(bands.findtext('nks') or 0),
        'functional': (output.findtext('dft/functional') or '').strip(),
        'metal': occupations not in (None, 'fixed') and (gap is None or gap <= 0.0),
        'two_dimensional': is_two_dimensional(cell, np.array(positions)) if positions else False,
        'augmented': any(
            (s['pseudo'] or {}).get('pseudo_type') in AUGMENTED_TYPES for s in species
        ),
    }


def _namelist_value(text: str, name: str) -> str | None:
    match = re.search(r"(?im)^[^!]*\b%s\s*=\s*('[^']*'|\"[^\"]*\"|[^,\s]+)" % re.escape(name), text)
    return match.group(1).strip('\'"') if match else None


def _card_lines(text: str, card: str) -> list[list[str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().upper().startswith(card):
            rows = []
            for following in lines[index + 1 :]:
                tokens = following.split()
                if not tokens or tokens[0].startswith('!') or tokens[0].startswith('#'):
                    if rows:
                        break
                    continue
                if tokens[0].upper() in (
                    'ATOMIC_SPECIES',
                    'ATOMIC_POSITIONS',
                    'K_POINTS',
                    'CELL_PARAMETERS',
                    'HUBBARD',
                    'OCCUPATIONS',
                    'CONSTRAINTS',
                ) or tokens[0].startswith('&'):
                    break
                rows.append(tokens)
            return rows
    return []


def read_pw_input_summary(path: str) -> dict[str, Any] | None:
    """Summary of a ``pw.x`` input: prefix, species, occupations, bands and valence electrons.

    Parameters
    ----------
    path : str
        The input file.

    Returns
    -------
    dict or None
        ``prefix``, ``pseudo_dir`` (absolute), ``nat``, ``species`` (name, mass, file, UPF
        header when found), ``counts`` (atoms per species), ``occupations``, ``metal``,
        ``nbnd``, ``noncolin``, ``spinorbit``, ``nelec`` (from the UPF valence charges, ``None``
        if a pseudopotential is missing) and ``augmented``; ``None`` if unreadable.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError:
        return None
    directory = os.path.dirname(os.path.abspath(path))
    pseudo_dir = _namelist_value(text, 'pseudo_dir')
    pseudo_dir = os.path.normpath(os.path.join(directory, os.path.expanduser(pseudo_dir or '.')))
    species = []
    for tokens in _card_lines(text, 'ATOMIC_SPECIES'):
        if len(tokens) < 3:
            continue
        upf = os.path.join(pseudo_dir, tokens[2])
        try:
            mass = float(tokens[1])
        except ValueError:
            mass = None
        species.append(
            {
                'name': tokens[0],
                'mass_amu': mass,
                'pseudo_file': tokens[2],
                'pseudo': read_upf_header(upf) if os.path.isfile(upf) else None,
            }
        )
    counts = Counter(tokens[0] for tokens in _card_lines(text, 'ATOMIC_POSITIONS'))
    nelec = None
    if species and all(s['pseudo'] and s['pseudo'].get('z_valence') for s in species):
        nelec = sum(s['pseudo']['z_valence'] * counts.get(s['name'], 0) for s in species)
    occupations = (_namelist_value(text, 'occupations') or '').lower() or None
    nbnd = _namelist_value(text, 'nbnd')
    nat = _namelist_value(text, 'nat')

    def flag(name: str) -> bool:
        value = (_namelist_value(text, name) or '').lower()
        return value in ('.true.', 'true', 't', '.t.')

    return {
        'prefix': _namelist_value(text, 'prefix'),
        'pseudo_dir': pseudo_dir,
        'nat': int(float(nat)) if nat else sum(counts.values()),
        'species': species,
        'counts': dict(counts),
        'occupations': occupations,
        'metal': occupations == 'smearing',
        'nbnd': int(float(nbnd)) if nbnd else None,
        'noncolin': flag('noncolin'),
        'spinorbit': flag('lspinorb'),
        'nelec': nelec,
        'augmented': any(
            (s['pseudo'] or {}).get('pseudo_type') in AUGMENTED_TYPES for s in species
        ),
    }


def find_saves(workdir: str) -> list[str]:
    """``*.save`` folders of a directory (sorted names).

    Parameters
    ----------
    workdir : str
        The directory.

    Returns
    -------
    list of str
    """
    return sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(workdir, '*.save')) if os.path.isdir(d)
    )
