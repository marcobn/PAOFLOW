#!/usr/bin/env python3
"""Generate a Quantum ESPRESSO scf input from an AFLOW database entry.

Given the URL of an AFLOW entry (e.g.
https://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/FCC/Si1_ICSD_29287/), this tool
downloads the ``aflowlib.json`` reference card and the ``CONTCAR.relax.qe``
geometry, then assembles a ready-to-run ``<compound>.scf.in`` file.

Only the Python standard library is used.
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

# AFLUX summons endpoint used to resolve an AUID to its data directory (aurl).
AFLUX_API = 'https://aflow.org/API/aflux/'

# Threshold below which an AFLOW Egap is treated as metallic (needs smearing).
EGAP_METAL_THRESHOLD = 1.0e-6
# Threshold above which a per-atom local moment marks the system as magnetic.
SPIND_THRESHOLD = 1.0e-3
# Default Gaussian smearing width (Ry) for metals.
DEFAULT_DEGAUSS = 0.05
# Initial guess for starting_magnetization on every magnetic species.
DEFAULT_START_MAG = 0.1
# Safety margin applied to the estimated PAO basis size when setting nbnd, so
# that the requested number of bands comfortably spans the extended basis.
NBND_MARGIN = 1.15
# Map of spectroscopic shell letters to angular momentum (fallback parsing).
L_OF_LETTER = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def download_text(url, timeout=30):
    """Return the decoded text body of *url* or raise a RuntimeError."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        raise RuntimeError('HTTP {} fetching {}'.format(exc.code, url)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError('Could not fetch {}: {}'.format(url, exc.reason)) from exc


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


# --------------------------------------------------------------------------- #
# CONTCAR.relax.qe parsing
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Pseudopotential folder helpers
# --------------------------------------------------------------------------- #
def find_pseudo_file(pseudo_dir, element):
    """Return the .upf filename in *pseudo_dir* matching *element* (case-insensitive)."""
    target = element.lower() + '.upf'
    matches = []
    for name in os.listdir(pseudo_dir):
        lower = name.lower()
        if not lower.endswith('.upf'):
            continue
        if lower == target:
            return name
        if lower.split('.')[0] == element.lower():
            matches.append(name)
    if matches:
        return matches[0]
    raise RuntimeError("No pseudopotential '{}.upf' found in {}".format(element, pseudo_dir))


def lmax_from_upf(upf_path):
    """Return the maximum angular momentum among the pseudo-wavefunctions.

    Reads the ``PP_CHI`` entries of a UPF file (``l="..."`` attributes, with a
    fallback to spectroscopic ``label`` letters). Returns ``None`` if no
    pseudo-wavefunction information is found.
    """
    try:
        with open(upf_path, 'r', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError:
        return None

    l_values = [int(v) for v in re.findall(r'\bl\s*=\s*"(\d+)"', text)]
    if l_values:
        return max(l_values)

    letters = re.findall(r'label\s*=\s*"\s*\d*\s*([A-Za-z])', text)
    found = [L_OF_LETTER[c.lower()] for c in letters if c.lower() in L_OF_LETTER]
    return max(found) if found else None


def extended_orbitals_per_atom(lmax):
    """Number of PAO orbitals per atom in the extended basis.

    The extended basis includes one shell for every angular momentum from 0 up
    to ``lmax + 1`` (the leading polarization channel), giving
    ``sum_{l=0}^{lmax+1} (2l + 1) = (lmax + 2) ** 2`` orbitals.
    """
    return (lmax + 2) ** 2


def load_atomic_masses(pseudo_dir):
    """Build a {symbol: atomic_mass} map from PeriodicTableJSON.json."""
    path = os.path.join(pseudo_dir, 'PeriodicTableJSON.json')
    if not os.path.isfile(path):
        raise RuntimeError('PeriodicTableJSON.json not found in {}'.format(pseudo_dir))
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    masses = {}
    elements = data.get('elements', data) if isinstance(data, dict) else data
    if isinstance(elements, list):
        for entry in elements:
            sym = entry.get('symbol')
            mass = entry.get('atomic_mass')
            if sym is not None and mass is not None:
                masses[sym] = mass
    elif isinstance(elements, dict):
        for sym, entry in elements.items():
            mass = entry.get('atomic_mass') if isinstance(entry, dict) else entry
            if mass is not None:
                masses[sym] = mass
    return masses


def load_reference_cutoffs(pseudo_dir):
    """Build a {element: ecutwfc_Ry} map from reference.json (hn in Hartree)."""
    path = os.path.join(pseudo_dir, 'reference.json')
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    cutoffs = {}
    for element, entry in data.items():
        if isinstance(entry, dict) and 'hn' in entry:
            try:
                cutoffs[element] = float(entry['hn']) * 2.0  # Hartree -> Rydberg
            except (TypeError, ValueError):
                continue
    return cutoffs


# --------------------------------------------------------------------------- #
# Input assembly
# --------------------------------------------------------------------------- #
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


def build_input(aflow, contcar, pseudo_dir, soc, degauss, nbnd_override=None):
    """Assemble the full QE scf input text."""
    compound = aflow.get('compound', 'system')
    species = resolve_species(aflow)
    ntyp = len(species)
    nat = aflow.get('natoms') or len(contcar['atom_order'])
    egap = float(aflow.get('Egap', 0.0) or 0.0)
    metallic = egap <= EGAP_METAL_THRESHOLD
    magnetic = is_magnetic(aflow.get('spinD', []))

    masses = load_atomic_masses(pseudo_dir)
    cutoffs = load_reference_cutoffs(pseudo_dir)

    species_rows = []
    ecut_values = []
    orbitals_per_element = {}
    for element, _count in species:
        upf = find_pseudo_file(pseudo_dir, element)
        mass = masses.get(element)
        if mass is None:
            raise RuntimeError("Atomic mass for '{}' not in PeriodicTableJSON.json".format(element))
        species_rows.append('  {:<3s} {:>10.4f}  {}'.format(element, float(mass), upf))
        if element in cutoffs:
            ecut_values.append(cutoffs[element])
        lmax = lmax_from_upf(os.path.join(pseudo_dir, upf))
        if lmax is None:
            lmax = 2  # generous default (covers s/p/d extended basis)
            sys.stderr.write(
                "Warning: could not read pseudo-wavefunctions for '{}'; "
                'assuming l_max=2 for nbnd.\n'.format(element)
            )
        orbitals_per_element[element] = extended_orbitals_per_atom(lmax)

    if ecut_values:
        ecutwfc = max(ecut_values)
    else:
        fallback = aflow.get('energy_cutoff')
        if fallback is None:
            raise RuntimeError('ecutwfc unavailable: provide reference.json in the pseudo folder.')
        ecutwfc = float(fallback)
        sys.stderr.write(
            'Warning: reference.json missing/incomplete; '
            'using aflow energy_cutoff={} Ry.\n'.format(ecutwfc)
        )

    # nbnd large enough to span the extended PAO basis used in PAOFLOW
    # projections: sum of per-atom extended-basis orbitals, doubled for
    # spin-orbit (spinor) calculations, plus a safety margin.
    if nbnd_override is not None:
        nbnd = nbnd_override
    else:
        nawf = sum(
            orbitals_per_element.get(el, extended_orbitals_per_atom(2))
            for el in contcar['atom_order']
        )
        if soc:
            nawf *= 2
        nbnd = max(nawf + 2, int(math.ceil(nawf * NBND_MARGIN)))

    out = []
    # &control
    out.append(' &control')
    out.append("    calculation = 'scf'")
    out.append("    restart_mode = 'from_scratch',")
    out.append("    prefix = '{}',".format(compound))
    out.append("    pseudo_dir = '{}',".format(pseudo_dir))
    out.append("    outdir = './'")
    out.append(" /")

    # &system
    out.append(' &system')
    out.append('    ibrav = 0,')
    out.append('    nat = {},'.format(nat))
    out.append('    ntyp = {},'.format(ntyp))
    out.append('    ecutwfc = {:.1f},'.format(ecutwfc))
    out.append('    nbnd = {},'.format(nbnd))
    if metallic:
        out.append("    occupations = 'smearing',")
        out.append('    degauss = {},'.format(degauss))
    if soc:
        out.append('    noncolin = .true.,')
        out.append('    lspinorb = .true.,')
        for i in range(1, ntyp + 1):
            out.append('    starting_magnetization({}) = {},'.format(i, DEFAULT_START_MAG))
    elif magnetic:
        out.append('    nspin = 2,')
        for i in range(1, ntyp + 1):
            out.append('    starting_magnetization({}) = {},'.format(i, DEFAULT_START_MAG))
    out.append(' /')

    # &electrons
    out.append(' &electrons')
    out.append('    mixing_beta = {}'.format(0.2 if metallic else 0.7))
    out.append(' /')

    # ATOMIC_SPECIES
    out.append('ATOMIC_SPECIES')
    out.extend(species_rows)

    # CELL_PARAMETERS
    out.append(contcar['cell_header'])
    out.extend('  ' + row for row in contcar['cell_rows'])

    # ATOMIC_POSITIONS
    out.append(contcar['pos_header'])
    out.extend('  ' + row for row in contcar['pos_rows'])

    # K_POINTS
    kpts = aflow.get('kpoints_static') or [1, 1, 1]
    out.append('K_POINTS {automatic}')
    out.append('  {} {} {} 0 0 0'.format(kpts[0], kpts[1], kpts[2]))

    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate a Quantum ESPRESSO scf input from an AFLOW entry.'
    )
    parser.add_argument(
        'url',
        help='AFLOW entry: an AFLOWDATA directory URL '
        '(e.g. https://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/FCC/Si1_ICSD_29287/), '
        'a material page URL (https://aflow.org/material/?id=aflow:<auid>), '
        'or a bare aflow:<auid> token',
    )
    parser.add_argument(
        '-p',
        '--pseudo-dir',
        required=True,
        help='Folder with <element>.upf, PeriodicTableJSON.json and reference.json',
    )
    parser.add_argument(
        '--soc',
        action='store_true',
        help='Pseudopotentials are relativistic: add noncolin/lspinorb cards',
    )
    parser.add_argument(
        '-o',
        '--out',
        default=None,
        help='Output path (default: <compound>.scf.in in the current directory)',
    )
    parser.add_argument(
        '--degauss',
        type=float,
        default=DEFAULT_DEGAUSS,
        help='Smearing width in Ry for metallic systems (default: %(default)s)',
    )
    parser.add_argument(
        '--nbnd',
        type=int,
        default=None,
        help='Override nbnd (default: estimated from the extended PAO basis)',
    )
    args = parser.parse_args(argv)

    pseudo_dir = os.path.abspath(os.path.expanduser(args.pseudo_dir))
    if not os.path.isdir(pseudo_dir):
        parser.error('pseudo-dir does not exist: {}'.format(pseudo_dir))

    try:
        entry_url = resolve_entry_url(args.url)
        aflow_text = download_text(entry_file_url(entry_url, 'aflowlib.json'))
        aflow = json.loads(aflow_text)
        contcar_text = download_text(entry_file_url(entry_url, 'CONTCAR.relax.qe'))
        contcar = parse_contcar_qe(contcar_text)
        content = build_input(aflow, contcar, pseudo_dir, args.soc, args.degauss, args.nbnd)
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write('Error: {}\n'.format(exc))
        return 1

    out_path = args.out or '{}.scf.in'.format(aflow.get('compound', 'system'))
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(content)

    sys.stderr.write('Wrote {}\n'.format(os.path.abspath(out_path)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
