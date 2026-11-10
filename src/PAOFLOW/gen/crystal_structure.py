"""Crystal structure of a working folder: read from a QE save or a ``pw.x`` input, drawn in 3D.

The Setup tab of ``paoflow-studio`` shows it under the working folder (``GET
/api/workspace/structure``).  :func:`structure_from_save` and :func:`structure_from_pw_input`
give the cell and the atoms; :func:`structure_figure` gives a Plotly specification (the colour
roles of :mod:`PAOFLOW.results`) with the unit cell, the atoms in the Jmol colours with sizes
from their covalent radii, and the bonds between atoms closer than the sum of the radii
(+ 15 %).  A bond to an atom outside the cell shown is drawn up to its middle.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

BOHR_ANGSTROM = 0.529177210903

#: Jmol colours and covalent radii (Å; B. Cordero et al., Dalton Trans. 2832 (2008)) by element.
ELEMENTS: dict[str, tuple[str, float]] = {
    'H': ('#ffffff', 0.31),
    'He': ('#d9ffff', 0.28),
    'Li': ('#cc80ff', 1.28),
    'Be': ('#c2ff00', 0.96),
    'B': ('#ffb5b5', 0.84),
    'C': ('#909090', 0.76),
    'N': ('#3050f8', 0.71),
    'O': ('#ff0d0d', 0.66),
    'F': ('#90e050', 0.57),
    'Ne': ('#b3e3f5', 0.58),
    'Na': ('#ab5cf2', 1.66),
    'Mg': ('#8aff00', 1.41),
    'Al': ('#bfa6a6', 1.21),
    'Si': ('#f0c8a0', 1.11),
    'P': ('#ff8000', 1.07),
    'S': ('#ffff30', 1.05),
    'Cl': ('#1ff01f', 1.02),
    'Ar': ('#80d1e3', 1.06),
    'K': ('#8f40d4', 2.03),
    'Ca': ('#3dff00', 1.76),
    'Sc': ('#e6e6e6', 1.70),
    'Ti': ('#bfc2c7', 1.60),
    'V': ('#a6a6ab', 1.53),
    'Cr': ('#8a99c7', 1.39),
    'Mn': ('#9c7ac7', 1.39),
    'Fe': ('#e06633', 1.32),
    'Co': ('#f090a0', 1.26),
    'Ni': ('#50d050', 1.24),
    'Cu': ('#c88033', 1.32),
    'Zn': ('#7d80b0', 1.22),
    'Ga': ('#c28f8f', 1.22),
    'Ge': ('#668f8f', 1.20),
    'As': ('#bd80e3', 1.19),
    'Se': ('#ffa100', 1.20),
    'Br': ('#a62929', 1.20),
    'Kr': ('#5cb8d1', 1.16),
    'Rb': ('#702eb0', 2.20),
    'Sr': ('#00ff00', 1.95),
    'Y': ('#94ffff', 1.90),
    'Zr': ('#94e0e0', 1.75),
    'Nb': ('#73c2c9', 1.64),
    'Mo': ('#54b5b5', 1.54),
    'Tc': ('#3b9e9e', 1.47),
    'Ru': ('#248f8f', 1.46),
    'Rh': ('#0a7d8c', 1.42),
    'Pd': ('#006985', 1.39),
    'Ag': ('#c0c0c0', 1.45),
    'Cd': ('#ffd98f', 1.44),
    'In': ('#a67573', 1.42),
    'Sn': ('#668080', 1.39),
    'Sb': ('#9e63b5', 1.39),
    'Te': ('#d47a00', 1.38),
    'I': ('#940094', 1.39),
    'Xe': ('#429eb0', 1.40),
    'Cs': ('#57178f', 2.44),
    'Ba': ('#00c900', 2.15),
    'La': ('#70d4ff', 2.07),
    'Ce': ('#ffffc7', 2.04),
    'Pr': ('#d9ffc7', 2.03),
    'Nd': ('#c7ffc7', 2.01),
    'Pm': ('#a3ffc7', 1.99),
    'Sm': ('#8fffc7', 1.98),
    'Eu': ('#61ffc7', 1.98),
    'Gd': ('#45ffc7', 1.96),
    'Tb': ('#30ffc7', 1.94),
    'Dy': ('#1fffc7', 1.92),
    'Ho': ('#00ff9c', 1.92),
    'Er': ('#00e675', 1.89),
    'Tm': ('#00d452', 1.90),
    'Yb': ('#00bf38', 1.87),
    'Lu': ('#00ab24', 1.87),
    'Hf': ('#4dc2ff', 1.75),
    'Ta': ('#4da6ff', 1.70),
    'W': ('#2194d6', 1.62),
    'Re': ('#267dab', 1.51),
    'Os': ('#266696', 1.44),
    'Ir': ('#175487', 1.41),
    'Pt': ('#d0d0e0', 1.36),
    'Au': ('#ffd123', 1.36),
    'Hg': ('#b8b8d0', 1.32),
    'Tl': ('#a6544d', 1.45),
    'Pb': ('#575961', 1.46),
    'Bi': ('#9e4fb5', 1.48),
    'Po': ('#ab5c00', 1.40),
    'At': ('#754f45', 1.50),
    'Rn': ('#428296', 1.50),
    'Fr': ('#420066', 2.60),
    'Ra': ('#007d00', 2.21),
    'Ac': ('#70abfa', 2.15),
    'Th': ('#00baff', 2.06),
    'Pa': ('#00a1ff', 2.00),
    'U': ('#008fff', 1.96),
    'Np': ('#0080ff', 1.90),
    'Pu': ('#006bff', 1.87),
    'Am': ('#545cf2', 1.80),
    'Cm': ('#785ce3', 1.69),
}
UNKNOWN_ELEMENT = ('#ff1493', 1.2)
BOND_TOLERANCE = 1.15
MAX_REPEAT = 3
#: Atoms of the cell times the repetitions above which fewer cells per axis are shown.
MAX_SHOWN_ATOMS = 2000


def element_of(label: str) -> str:
    """Element of a species label (``Fe1``, ``Fe_up``, ``O2-`` -> ``Fe``, ``Fe``, ``O``).

    Parameters
    ----------
    label : str
        Species name of the QE input or save.

    Returns
    -------
    str
        The element symbol, or the label when no element is recognised.
    """
    match = re.match(r'([A-Z][a-z]?)', label.strip().capitalize())
    if match and match.group(1) in ELEMENTS:
        return match.group(1)
    match = re.match(r'([A-Z])', label.strip().capitalize())
    if match and match.group(1) in ELEMENTS:
        return match.group(1)
    return label


def structure_from_save(savedir: str) -> dict[str, Any] | None:
    """Cell and atoms of a ``<prefix>.save`` (its ``data-file-schema.xml``, final structure).

    Parameters
    ----------
    savedir : str
        The save folder.

    Returns
    -------
    dict or None
        ``cell`` (rows a1, a2, a3 in Å), ``species`` (labels), ``positions`` (Cartesian, Å);
        ``None`` when the file is missing or has no structure.
    """
    try:
        root = ET.parse(os.path.join(savedir, 'data-file-schema.xml')).getroot()
    except (OSError, ET.ParseError):
        return None
    structure = root.find('output/atomic_structure')
    if structure is None:
        structure = root.find('input/atomic_structure')
    if structure is None:
        return None
    try:
        cell = np.array(
            [
                [float(x) for x in structure.findtext('cell/' + v).split()]
                for v in ('a1', 'a2', 'a3')
            ]
        )
        atoms = structure.findall('atomic_positions/atom')
        positions = np.array([[float(x) for x in (a.text or '').split()] for a in atoms])
    except (AttributeError, ValueError):
        return None
    if not len(atoms) or positions.shape != (len(atoms), 3):
        return None
    return {
        'cell': cell * BOHR_ANGSTROM,
        'species': [a.attrib.get('name', '?') for a in atoms],
        'positions': positions * BOHR_ANGSTROM,
    }


def _namelist_number(text: str, name: str, pattern: str | None = None) -> float | None:
    key = pattern if pattern is not None else re.escape(name)
    match = re.search(r'(?im)^[^!]*\b%s\s*=\s*([-+0-9.eEdD]+)' % key, text)
    if not match:
        return None
    return float(match.group(1).lower().replace('d', 'e'))


def _card(text: str, card: str) -> tuple[str, list[list[str]]]:
    """Option and rows of a card (rows until the next card, namelist or blank line after rows)."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if tokens and tokens[0].upper() == card:
            option = ' '.join(tokens[1:]).strip('{}() ').lower()
            rows = []
            for following in lines[index + 1 :]:
                tokens = following.split('!')[0].split()
                if not tokens:
                    if rows:
                        break
                    continue
                if tokens[0].startswith('&') or tokens[0].upper() in (
                    'ATOMIC_SPECIES',
                    'ATOMIC_POSITIONS',
                    'K_POINTS',
                    'CELL_PARAMETERS',
                    'HUBBARD',
                    'OCCUPATIONS',
                    'CONSTRAINTS',
                    'ATOMIC_FORCES',
                    'ADDITIONAL_K_POINTS',
                    'SOLVENTS',
                ):
                    break
                rows.append(tokens)
            return option, rows
    return '', []


def _celldm(text: str, ibrav: int) -> np.ndarray | None:
    """``celldm(1..6)``, from ``celldm`` or from ``A, B, C, cosAB, cosAC, cosBC`` (QE abc2celldm)."""
    celldm = np.zeros(6)
    for index in range(6):
        value = _namelist_number(text, 'celldm', r'celldm\(\s*%d\s*\)' % (index + 1))
        if value is not None:
            celldm[index] = value
    if celldm[0]:
        return celldm
    a = _namelist_number(text, 'A')
    if not a:
        return None
    b = _namelist_number(text, 'B') or a
    c = _namelist_number(text, 'C') or a
    cos_ab = _namelist_number(text, 'cosAB') or 0.0
    cos_ac = _namelist_number(text, 'cosAC') or 0.0
    cos_bc = _namelist_number(text, 'cosBC') or 0.0
    celldm[:3] = a / BOHR_ANGSTROM, b / a, c / a
    if ibrav == 14:
        celldm[3:] = cos_bc, cos_ac, cos_ab
    elif ibrav in (-12, -13):
        celldm[4] = cos_ac
    else:
        celldm[3] = cos_ab
    return celldm


def structure_from_pw_input(path: str) -> dict[str, Any] | None:
    """Cell and atoms of a ``pw.x`` input (``ibrav`` with ``celldm`` or ``A, B, C``, or
    ``CELL_PARAMETERS``; positions in ``alat``, ``bohr``, ``angstrom`` or ``crystal``).

    Parameters
    ----------
    path : str
        The input file.

    Returns
    -------
    dict or None
        As :func:`structure_from_save`; ``None`` when the structure cannot be read.
    """
    from ..inputs.lattice_format import lattice_format_QE

    try:
        with open(path, encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError:
        return None
    ibrav_value = _namelist_number(text, 'ibrav')
    ibrav = int(ibrav_value) if ibrav_value is not None else 0
    celldm = _celldm(text, ibrav)
    alat = celldm[0] if celldm is not None else None  # bohr
    try:
        if ibrav != 0:
            if celldm is None:
                return None
            cell = np.asarray(lattice_format_QE(ibrav, celldm), dtype=float)  # bohr
        else:
            option, rows = _card(text, 'CELL_PARAMETERS')
            if len(rows) < 3:
                return None
            vectors = np.array([[float(x) for x in row[:3]] for row in rows[:3]])
            if option.startswith('angstrom'):
                cell = vectors / BOHR_ANGSTROM
            elif option.startswith('bohr') or (not option and alat is None):
                cell = vectors
            else:  # alat, or no option with celldm(1)
                if alat is None:
                    return None
                cell = vectors * alat
            if alat is None:
                alat = float(np.linalg.norm(cell[0]))
        option, rows = _card(text, 'ATOMIC_POSITIONS')
        rows = [row for row in rows if len(row) >= 4]
        if not rows:
            return None
        species = [row[0] for row in rows]
        values = np.array([[float(x.lower().replace('d', 'e')) for x in row[1:4]] for row in rows])
    except (ValueError, IndexError):
        return None
    if option.startswith('crystal_sg'):
        return None
    if option.startswith('crystal'):
        positions = values @ cell
    elif option.startswith('bohr'):
        positions = values
    elif option.startswith('angstrom'):
        positions = values / BOHR_ANGSTROM
    else:  # alat (the default)
        positions = values * alat
    return {
        'cell': cell * BOHR_ANGSTROM,
        'species': species,
        'positions': positions * BOHR_ANGSTROM,
    }


def cell_facts(cell: np.ndarray) -> dict[str, float]:
    """Lengths (Å), angles (degrees) and volume (Å^3) of a cell.

    Parameters
    ----------
    cell : ndarray, shape (3, 3)
        Rows a1, a2, a3 in Å.

    Returns
    -------
    dict
        ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``, ``volume``.
    """
    a, b, c = (float(np.linalg.norm(v)) for v in cell)

    def angle(u, v):
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    return {
        'a': a,
        'b': b,
        'c': c,
        'alpha': angle(cell[1], cell[2]),
        'beta': angle(cell[0], cell[2]),
        'gamma': angle(cell[0], cell[1]),
        'volume': float(abs(np.linalg.det(cell))),
    }


def _formula(species: list[str]) -> str:
    counts: dict[str, int] = {}
    for label in species:
        element = element_of(label)
        counts[element] = counts.get(element, 0) + 1
    return ''.join('%s%s' % (e, n if n > 1 else '') for e, n in counts.items())


def structure_figure(structure: dict[str, Any], repeat: int = 1) -> dict[str, Any]:
    """Plotly specification of a structure: cell, atoms and bonds.

    Parameters
    ----------
    structure : dict
        ``cell``, ``species`` and ``positions`` (Å), as :func:`structure_from_save`.
    repeat : int, optional
        Cells per axis shown (1 to :data:`MAX_REPEAT`); atoms on the faces of the block are shown
        on both sides.

    Returns
    -------
    dict
        ``figure``, ``species`` (``label``, ``element``, ``color``, ``count`` of the cell),
        ``formula``, ``nat`` and ``cell`` (:func:`cell_facts`).
    """
    from .band_paths import camera_eye

    cell = np.asarray(structure['cell'], dtype=float)
    labels = list(structure['species'])
    repeat = int(min(max(1, repeat), MAX_REPEAT))
    while repeat > 1 and len(labels) * repeat**3 > MAX_SHOWN_ATOMS:
        repeat -= 1  # a large cell: fewer copies
    fractional = np.asarray(structure['positions'], dtype=float) @ np.linalg.inv(cell)
    fractional -= np.floor(fractional + 1.0e-6)  # wrapped into [0, 1)
    nat = len(labels)
    elements = [element_of(label) for label in labels]
    radius = np.array([ELEMENTS.get(e, UNKNOWN_ELEMENT)[1] for e in elements])

    # Images in the shell around the block: those inside the block are shown, the others only
    # end the half bonds that leave it.
    shifts = np.array(
        [(i, j, k) for i in range(-1, repeat + 1) for j in range(-1, repeat + 1) for k in range(-1, repeat + 1)],
        dtype=float,
    )  # fmt: skip
    image_frac = (fractional[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    image_atom = np.tile(np.arange(nat), len(shifts))
    tolerance = 1.0e-3
    shown = np.all((image_frac > -tolerance) & (image_frac < repeat + tolerance), axis=1)
    image_xyz = image_frac @ cell
    shown_index = np.flatnonzero(shown)
    xyz = image_xyz[shown_index]
    atom = image_atom[shown_index]

    bond_x, bond_y, bond_z = [], [], []
    pairs = []
    chunk = max(1, 2_000_000 // max(1, len(image_xyz)))  # rows of the distance matrix at a time
    for start in range(0, len(xyz), chunk):
        rows = slice(start, start + chunk)
        cutoff = (radius[atom[rows]][:, None] + radius[image_atom][None, :]) * BOND_TOLERANCE
        distance = np.linalg.norm(xyz[rows, None, :] - image_xyz[None, :, :], axis=2)
        found = np.argwhere((distance < cutoff) & (distance > 0.3))
        found[:, 0] += start
        pairs.append(found)
    for i, j in np.concatenate(pairs) if pairs else []:
        if shown[j]:
            if shown_index[i] >= j:
                continue  # each bond inside the block once
            end = image_xyz[j]
        else:
            end = 0.5 * (xyz[i] + image_xyz[j])
        bond_x += [float(xyz[i][0]), float(end[0]), None]
        bond_y += [float(xyz[i][1]), float(end[1]), None]
        bond_z += [float(xyz[i][2]), float(end[2]), None]

    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    edges = [
        (a, b)
        for a in range(8)
        for b in range(a + 1, 8)
        if np.abs(corners[a] - corners[b]).sum() == 1
    ]
    cx, cy, cz = [], [], []
    for a, b in edges:
        for point in (corners[a] @ cell, corners[b] @ cell):
            cx.append(float(point[0]))
            cy.append(float(point[1]))
            cz.append(float(point[2]))
        cx.append(None)
        cy.append(None)
        cz.append(None)

    data = [
        {
            'type': 'scatter3d',
            'mode': 'lines',
            'name': 'cell',
            'x': cx,
            'y': cy,
            'z': cz,
            'line': {'color': 'axis', 'width': 3},
            'hoverinfo': 'skip',
        },
        {
            'type': 'scatter3d',
            'mode': 'lines',
            'name': 'bonds',
            'x': bond_x,
            'y': bond_y,
            'z': bond_z,
            'line': {'color': 'neutral', 'width': 5},
            'hoverinfo': 'skip',
        },
    ]
    species_rows = []
    for label in dict.fromkeys(labels):
        element = element_of(label)
        color, cov = ELEMENTS.get(element, UNKNOWN_ELEMENT)
        members = np.flatnonzero(np.array(labels)[atom] == label)
        text = [
            '%s %d · (%.4f, %.4f, %.4f) crystal · (%.3f, %.3f, %.3f) Å'
            % (label, atom[m] + 1, *(image_frac[shown_index[m]]), *xyz[m])
            for m in members
        ]
        data.append(
            {
                'type': 'scatter3d',
                'mode': 'markers',
                'name': label,
                'x': [float(v) for v in xyz[members, 0]],
                'y': [float(v) for v in xyz[members, 1]],
                'z': [float(v) for v in xyz[members, 2]],
                'text': text,
                'hovertemplate': '%{text}<extra></extra>',
                'marker': {
                    'size': round((6.0 + 9.0 * cov) / repeat**0.6, 1),
                    'color': color,
                    'opacity': 1.0,
                    'line': {'color': 'axis', 'width': 1},
                },
            }
        )
        species_rows.append(
            {'label': label, 'element': element, 'color': color, 'count': labels.count(label)}
        )
    tips = cell * 1.12
    data.append(
        {
            'type': 'scatter3d',
            'mode': 'text',
            'name': 'axes',
            'x': [float(t[0]) for t in tips],
            'y': [float(t[1]) for t in tips],
            'z': [float(t[2]) for t in tips],
            'text': ['a', 'b', 'c'],
            'textfont': {'color': 'muted', 'size': 12},
            'hoverinfo': 'skip',
        }
    )
    axis = {
        'showbackground': False,
        'showgrid': False,
        'zeroline': False,
        'showticklabels': False,
        'showspikes': False,
        'title': {'text': ''},
    }
    view = corners @ cell
    figure = {
        'data': data,
        'layout': {
            'scene': {
                'xaxis': dict(axis),
                'yaxis': dict(axis),
                'zaxis': dict(axis),
                'aspectmode': 'data',
                'camera': {'eye': camera_eye(view - view.mean(axis=0), distance=2.1)},
            },
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
            'showlegend': False,
        },
    }
    return {
        'figure': figure,
        'species': species_rows,
        'formula': _formula(labels),
        'nat': nat,
        'cell': cell_facts(cell),
        'shown': int(len(xyz)),
        'repeat': repeat,
    }


def folder_structure(folder: str, source: str) -> dict[str, Any] | None:
    """The structure of a save or a ``pw.x`` input of a folder.

    Parameters
    ----------
    folder : str
        The working folder.
    source : str
        A save folder (``<prefix>.save``, or a path to one) or a ``pw.x`` input, relative to
        ``folder``.

    Returns
    -------
    dict or None
        As :func:`structure_from_save`, plus ``kind`` (``'save'`` or ``'pw.x input'``).
    """
    path = os.path.join(folder, source)
    if os.path.isdir(path):
        structure = structure_from_save(path)
        kind = 'save'
    else:
        structure = structure_from_pw_input(path)
        kind = 'pw.x input'
    if structure is None:
        return None
    structure['kind'] = kind
    return structure
