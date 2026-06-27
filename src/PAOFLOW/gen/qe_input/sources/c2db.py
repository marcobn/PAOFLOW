"""C2DB (Computational 2D Materials Database) adapter.

Resolves a C2DB reference (material UID such as ``2C-1``, a material-page URL
``https://c2db.fysik.dtu.dk/material/<uid>``, or a bare ``c2db:<uid>`` token),
downloads the per-material structure JSON, and scrapes the HTML overview table
for the band gap and magnetic flag.  Everything is normalized into a
:class:`MaterialRecord` flagged as two-dimensional.

Only the Python standard library and numpy are used (no ASE dependency): the
C2DB structure download is a minimal ASE-atoms dict that ``json`` parses
directly.
"""

import json
import re

from ..record import MaterialRecord
from ..writer import EGAP_METAL_THRESHOLD
from .base import DatabaseSource, download_text

C2DB_HOST = 'https://c2db.fysik.dtu.dk'

# Atomic-number -> symbol table (Z = 1..118), used to label the JSON numbers.
_ELEMENTS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni '
    'Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I '
    'Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt '
    'Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr '
    'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
).split()


def _symbol(z):
    """Return the element symbol for atomic number *z*."""
    if 1 <= z <= len(_ELEMENTS):
        return _ELEMENTS[z - 1]
    raise RuntimeError('Unsupported atomic number {} in C2DB entry'.format(z))


def resolve_uid(identifier):
    """Extract the C2DB material UID from any accepted reference."""
    token = identifier.strip()
    if token.lower().startswith('c2db:'):
        return token.split(':', 1)[1].strip()
    match = re.search(r'/(?:material|row)/([^/?#]+)', token)
    if match:
        return match.group(1)
    if 'c2db.fysik.dtu.dk' in token.lower():
        raise RuntimeError("Could not extract a C2DB UID from '{}'".format(identifier))
    # Treat a bare token (e.g. '2C-1', 'MoS2-165798be3any') as the UID itself.
    return token


def _ordered_species(symbols):
    """Return ``[(element, count), ...]`` in first-appearance order."""
    order = []
    counts = {}
    for sym in symbols:
        if sym not in counts:
            counts[sym] = 0
            order.append(sym)
        counts[sym] += 1
    return [(sym, counts[sym]) for sym in order]


def _geometry_from_atoms(atoms):
    """Build a writer-compatible geometry dict from an ASE-atoms JSON dict."""
    numbers = atoms['numbers']
    positions = atoms['positions']
    cell = atoms['cell']
    symbols = [_symbol(int(z)) for z in numbers]

    cell_rows = ['{:22.16f} {:22.16f} {:22.16f}'.format(*[float(c) for c in row]) for row in cell]
    pos_rows = []
    for sym, pos in zip(symbols, positions):
        pos_rows.append(
            '{:<3s} {:18.12f} {:18.12f} {:18.12f}'.format(sym, *[float(p) for p in pos])
        )

    return {
        'cell_header': 'CELL_PARAMETERS (angstrom)',
        'cell_unit': 'angstrom',
        'cell_rows': cell_rows,
        'pos_header': 'ATOMIC_POSITIONS (angstrom)',
        'pos_unit': 'angstrom',
        'pos_rows': pos_rows,
        'atom_order': symbols,
    }


def _strip_tags(html):
    """Return a plain-text rendering of *html* (tags -> spaces)."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = text.replace('&nbsp;', ' ')
    return re.sub(r'\s+', ' ', text)


def scrape_overview(html):
    """Scrape ``(gap_eV, magnetic)`` from a C2DB material page.

    Each value is ``None`` when it cannot be located, so the caller can fall
    back to a CLI override.
    """
    text = _strip_tags(html)

    gap = None
    gap_match = re.search(r'Band gap \(PBE\)\s*\[eV\]\s*(-?\d+(?:\.\d+)?)', text)
    if gap_match:
        try:
            gap = float(gap_match.group(1))
        except ValueError:
            gap = None

    magnetic = None
    mag_match = re.search(r'(?<!anisotropy)\bMagnetic\b\s*(Yes|No)\b', text)
    if mag_match:
        magnetic = mag_match.group(1).lower() == 'yes'

    return gap, magnetic


class C2dbSource(DatabaseSource):
    """Adapter for the C2DB 2D-materials database (c2db.fysik.dtu.dk)."""

    name = 'c2db'

    def matches(self, identifier):
        token = identifier.strip().lower()
        if token.startswith('c2db:') or 'c2db.fysik.dtu.dk' in token:
            return True
        # Bare C2DB UID like '2C-1' or 'MoS2-165798be3any': formula then '-id'.
        return bool(re.fullmatch(r'[A-Za-z0-9]+-[A-Za-z0-9]+', identifier.strip()))

    def fetch(self, identifier, metallic=None, magnetic=None, kpoints=None, **options):
        uid = resolve_uid(identifier)
        base = '{}/material/{}'.format(C2DB_HOST, uid)

        raw = download_text('{}/download/json'.format(base))
        data = json.loads(raw)
        if not data:
            raise RuntimeError("Empty C2DB structure JSON for '{}'".format(uid))
        atoms = next(iter(data.values()))
        geometry = _geometry_from_atoms(atoms)

        pbc = atoms.get('pbc', [True, True, False])
        dimensionality = '2D' if not all(pbc) else '3D'

        # Resolve metallic/magnetic: explicit override wins, else scrape HTML.
        gap = None
        scraped_mag = None
        if metallic is None or magnetic is None:
            try:
                gap, scraped_mag = scrape_overview(download_text(base))
            except RuntimeError:
                pass

        if metallic is None:
            metallic = True if gap is None else gap <= EGAP_METAL_THRESHOLD
        if magnetic is None:
            magnetic = bool(scraped_mag)

        return MaterialRecord(
            compound=uid,
            geometry=geometry,
            species=_ordered_species(geometry['atom_order']),
            natoms=len(geometry['atom_order']),
            metallic=metallic,
            magnetic=magnetic,
            dimensionality=dimensionality,
            kpoints=tuple(kpoints) if kpoints else None,
            energy_cutoff=None,
            bravais_hint=None,
            spacegroup=None,
            source=self.name,
        )
