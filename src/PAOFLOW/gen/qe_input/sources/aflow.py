"""AFLOW database adapter.

Resolves an AFLOW reference (AFLOWDATA directory URL, material-page URL with
``?id=aflow:<auid>``, or a bare ``aflow:<auid>`` token), downloads the
``aflowlib.json`` reference card and the ``CONTCAR.relax.qe`` geometry, and
normalizes them into a :class:`MaterialRecord`.
"""

import json
import re
import urllib.parse

from ..record import MaterialRecord
from ..writer import EGAP_METAL_THRESHOLD, SPIND_THRESHOLD
from .base import DatabaseSource, download_text

# AFLUX summons endpoint used to resolve an AUID to its data directory (aurl).
AFLUX_API = 'https://aflow.org/API/aflux/'


def entry_file_url(base_url, filename):
    """Join an AFLOW entry base URL with a file name."""
    return base_url.rstrip('/') + '/' + filename


def aurl_to_url(aurl):
    """Convert an AFLOW aurl ('host:AFLOWDATA/...') to an https file directory URL."""
    host, _, path = aurl.partition(':')
    return 'https://{}/{}'.format(host, path.lstrip('/'))


def resolve_auid(auid):
    """Resolve an AFLOW AUID to its data-directory URL via the AFLUX API."""
    query = "{}?auid('{}'),aurl".format(AFLUX_API, auid)
    text = download_text(query)
    try:
        records = json.loads(text)
    except ValueError as exc:
        raise RuntimeError('Unexpected AFLUX response for {}'.format(auid)) from exc
    if not records or 'aurl' not in records[0]:
        raise RuntimeError("AUID '{}' not found in the AFLOW database".format(auid))
    return aurl_to_url(records[0]['aurl'])


def resolve_entry_url(url):
    """Normalize any accepted AFLOW reference into a data-directory URL.

    Accepts a direct AFLOWDATA directory URL, an ``aflow.org`` material page
    URL carrying ``?id=aflow:<auid>``, or a bare ``aflow:<auid>`` token.
    """
    candidate = url.strip()

    # Bare AUID token, e.g. "aflow:0a66d228d896a855".
    if candidate.lower().startswith('aflow:'):
        return resolve_auid(candidate)

    parsed = urllib.parse.urlparse(candidate)
    params = urllib.parse.parse_qs(parsed.query)
    # Material page: ...?id=aflow:<auid> (id may also be passed as 'auid').
    for key in ('id', 'auid'):
        if key in params and params[key]:
            value = params[key][0]
            match = re.search(r'aflow:[0-9a-fA-F]+', value)
            if match:
                return resolve_auid(match.group(0))

    # Already an AFLOWDATA directory URL.
    if 'AFLOWDATA' in candidate:
        return candidate

    raise RuntimeError(
        "Could not interpret '{}' as an AFLOW entry. Provide an AFLOWDATA "
        'directory URL, a material page URL with ?id=aflow:<auid>, or a bare '
        'aflow:<auid> token.'.format(url)
    )


def parse_contcar_qe(text):
    """Extract the geometry cards from a CONTCAR.relax.qe file.

    Returns a dict with the verbatim ``ATOMIC_POSITIONS`` and
    ``CELL_PARAMETERS`` cards (header + data rows), their unit strings, and the
    ordered list of element symbols (one entry per atom).
    """
    lines = text.splitlines()
    pos_header = pos_unit = None
    cell_header = cell_unit = None
    pos_rows = []
    cell_rows = []
    atom_order = []

    section = None
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith('ATOMIC_POSITIONS'):
            section = 'positions'
            pos_header = stripped
            pos_unit = _card_unit(stripped)
            continue
        if upper.startswith('CELL_PARAMETERS'):
            section = 'cell'
            cell_header = stripped
            cell_unit = _card_unit(stripped)
            continue
        # Any other namelist/card terminator ends the current section.
        if upper.startswith(('&', '/', 'K_POINTS', 'ATOMIC_SPECIES', 'HUBBARD')):
            section = None
            continue
        if not stripped or stripped.startswith(('#', '!')):
            continue

        # Strip trailing inline comments (AFLOW annotates rows with "! //").
        data = stripped.split('!', 1)[0].rstrip()
        if not data:
            continue

        if section == 'positions':
            pos_rows.append(data)
            atom_order.append(data.split()[0])
        elif section == 'cell':
            cell_rows.append(data)

    if not cell_rows:
        raise RuntimeError('CELL_PARAMETERS block not found in CONTCAR.relax.qe')
    if not pos_rows:
        raise RuntimeError('ATOMIC_POSITIONS block not found in CONTCAR.relax.qe')

    return {
        'cell_header': cell_header,
        'cell_unit': cell_unit,
        'cell_rows': cell_rows,
        'pos_header': pos_header,
        'pos_unit': pos_unit,
        'pos_rows': pos_rows,
        'atom_order': atom_order,
    }


def _card_unit(header):
    """Return the unit token inside a card header, e.g. '(angstrom)' -> 'angstrom'."""
    start = header.find('(')
    end = header.find(')')
    if start != -1 and end != -1 and end > start:
        return header[start + 1 : end].strip()
    parts = header.split()
    return parts[1] if len(parts) > 1 else ''


def is_magnetic(spind):
    """True if any per-atom local moment exceeds the magnetic threshold."""
    if not spind:
        return False
    return any(abs(float(s)) > SPIND_THRESHOLD for s in spind)


def resolve_species(aflow):
    """Return a list of (element, num_each_type) preserving AFLOW order."""
    species = list(aflow.get('species', []))
    composition = aflow.get('composition', [])
    counts = list(composition) if composition else [None] * len(species)
    if len(counts) < len(species):
        counts += [None] * (len(species) - len(counts))
    return list(zip(species, counts))


class AflowSource(DatabaseSource):
    """Adapter for the AFLOW (aflow.org / aflowlib) database."""

    name = 'aflow'

    def matches(self, identifier):
        token = identifier.strip().lower()
        return (
            token.startswith('aflow:')
            or 'aflowlib' in token
            or 'aflowdata' in token
            or 'aflow.org' in token
        )

    def fetch(self, identifier, metallic=None, magnetic=None, kpoints=None, **options):
        entry_url = resolve_entry_url(identifier)
        aflow = json.loads(download_text(entry_file_url(entry_url, 'aflowlib.json')))
        contcar = parse_contcar_qe(download_text(entry_file_url(entry_url, 'CONTCAR.relax.qe')))

        species = resolve_species(aflow)
        egap = float(aflow.get('Egap', 0.0) or 0.0)
        kpts = aflow.get('kpoints_static')
        kpoints = tuple(kpoints) if kpoints else (tuple(kpts) if kpts else None)
        fallback_cutoff = aflow.get('energy_cutoff')

        if metallic is None:
            metallic = egap <= EGAP_METAL_THRESHOLD
        if magnetic is None:
            magnetic = is_magnetic(aflow.get('spinD', []))

        return MaterialRecord(
            compound=aflow.get('compound', 'system'),
            geometry=contcar,
            species=species,
            natoms=aflow.get('natoms') or len(contcar['atom_order']),
            metallic=metallic,
            magnetic=magnetic,
            dimensionality='3D',
            kpoints=kpoints,
            energy_cutoff=float(fallback_cutoff) if fallback_cutoff is not None else None,
            bravais_hint=(aflow.get('Bravais_lattice_relax') or aflow.get('bravais_lattice_relax')),
            spacegroup=aflow.get('sg') or aflow.get('spacegroup_relax'),
            source=self.name,
        )
