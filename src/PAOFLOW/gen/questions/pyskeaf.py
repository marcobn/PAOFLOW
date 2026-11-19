"""Questions of the ``pyskeaf`` workflow: quantum oscillations from the Fermi-surface BXSF files.

A Property run with the Fermi surface writes one single-band file per crossing band,
``Fermi_surf_band_<n>.bxsf``, in its output folder.  ``main.pyskeaf.py`` reads them with
:meth:`PAOFLOW.PAOFLOW.PAOFLOW.pyskeaf` (no QE save needed) and writes the de Haas-van Alphen
frequencies against the field direction, ``qo_EF_<energy>_freqvsangle_<band>.out``, which the
Results tab plots.
"""

from __future__ import annotations

import glob
import os
import re

from .model import Answers, Context, Note, Question, Workflow, float_list, int_list

BXSF = 'Fermi-surface files'
FIELD = 'Magnetic field'
ORBITS = 'Extremal orbits'
BXSF_PATTERN = re.compile(r'Fermi_surf_band_(\d+)\.bxsf$')
FIELD_DIRECTIONS = ['rotation', 'non_principal', 'b1', 'b2', 'b3']


def bxsf_bands(folder: str) -> list[int]:
    """Band numbers of the ``Fermi_surf_band_<n>.bxsf`` files of a folder.

    Parameters
    ----------
    folder : str
        Folder.

    Returns
    -------
    list of int
        Sorted band numbers (empty when there is none).
    """
    bands = []
    for path in glob.glob(os.path.join(folder, 'Fermi_surf_band_*.bxsf')):
        match = BXSF_PATTERN.search(os.path.basename(path))
        if match:
            bands.append(int(match.group(1)))
    return sorted(bands)


def bxsf_band_label(path: str) -> int | None:
    """The band of a ``Fermi_surf_band_<n>.bxsf`` file, from its ``BAND:`` line.

    The file number is the position of the file (``n`` = 1, 2, ...), which ``PAOFLOW.pyskeaf``
    selects with ``bands``; the band is the index of the band in the run (band 2 when band 1
    does not cross the window).

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    int or None
    """
    try:
        with open(path, encoding='utf-8', errors='replace') as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                if line.strip().startswith('BAND:'):
                    return int(line.split(':')[1])
    except (OSError, ValueError):
        return None
    return None


def _detected_folder(context: Context) -> str:
    """The folder of the BXSF files: ``output``, the working directory or a run's ``output``."""

    def search() -> str:
        workdir = context.workdir
        for candidate in ['output', '.'] + sorted(
            os.path.relpath(path, workdir)
            for path in glob.glob(os.path.join(workdir, '*', 'output'))
        ):
            if bxsf_bands(os.path.join(workdir, candidate)):
                return candidate
        return 'output'

    return context.cached('bxsf-folder', search)


def _bands_found(answers: Answers, context: Context) -> list[int]:
    folder = answers.get('outputdir') or _detected_folder(context)
    return bxsf_bands(os.path.join(context.workdir, folder))


def _bands_note(answers: Answers, context: Context) -> str:
    folder = os.path.join(context.workdir, answers.get('outputdir') or _detected_folder(context))
    parts = []
    for number in _bands_found(answers, context):
        band = bxsf_band_label(os.path.join(folder, 'Fermi_surf_band_%d.bxsf' % number))
        parts.append('%d (band %d)' % (number, band) if band not in (None, number) else str(number))
    return 'BXSF files (file number, and its band): %s' % ', '.join(parts)


def _parse_bands(text: str) -> str | list[int]:
    text = text.strip()
    if text.lower() in ('', 'all'):
        return 'all'
    bands = int_list(text)
    if not bands or any(band < 1 for band in bands):
        raise ValueError('give "all" or band numbers such as 5 6')
    return bands


def _parse_energies(text: str) -> float | list[float]:
    values = float_list(text)
    if not values:
        raise ValueError('give one energy (eV) or several separated by commas')
    return values[0] if len(values) == 1 else values


def _rotation(answers: Answers) -> bool:
    return answers.get('b_field') == 'rotation'


def _one_direction(answers: Answers) -> bool:
    return answers.get('b_field') == 'non_principal'


def _at_least_two(value: int, answers: Answers, context: Context) -> str | None:
    return None if value >= 2 else 'a rotation needs two field directions or more'


def _missing_files_note(answers: Answers, context: Context) -> str:
    folder = answers.get('outputdir') or _detected_folder(context)
    return (
        'No Fermi_surf_band_*.bxsf in %s yet: run a Property run with the Fermi surface first '
        '(main.pyskeaf.py reads them when it runs).' % folder
    )


PYSKEAF_ITEMS = (
    Question(
        'outputdir',
        'Folder with the Fermi_surf_band_*.bxsf files (output of a Property run with the Fermi '
        'surface)',
        default=lambda a, c: _detected_folder(c),
        label='BXSF folder',
        help='relative to the working folder; the results are written there too',
        widget='folder',
        section=BXSF,
    ),
    Note(_missing_files_note, when=lambda a, c: not _bands_found(a, c), section=BXSF),
    Note(_bands_note, when=lambda a, c: bool(_bands_found(a, c)), section=BXSF),
    Question(
        'bands',
        'Bands ("all", or the file numbers of Fermi_surf_band_<n>.bxsf such as 1 2)',
        default='all',
        parse=_parse_bands,
        label='Bands',
        section=BXSF,
    ),
    Question(
        'fermi_energy',
        'Energy of the orbits relative to E_F (eV); several values scan, e.g. -0.05, 0, 0.05',
        default='0',
        parse=_parse_energies,
        label='Energy',
        unit='eV',
        section=BXSF,
    ),
    Question(
        'num_interpolation',
        'Interpolated points per reciprocal-cell side',
        kind='int',
        default=100,
        section=BXSF,
    ),
    Question(
        'b_field',
        'Field direction (rotation: from (θ, φ) start to end; non_principal: one (θ, φ); '
        'b1, b2, b3: along a reciprocal vector)',
        kind='choice',
        choices=FIELD_DIRECTIONS,
        default='rotation',
        label='Field direction',
        section=FIELD,
    ),
    Question(
        'theta',
        'θ, azimuth of the field in the xy plane (deg)',
        kind='float',
        default=0.0,
        unit='deg',
        when=lambda a, c: _one_direction(a),
        section=FIELD,
    ),
    Question(
        'phi',
        'φ, polar angle of the field from +z (deg)',
        kind='float',
        default=0.0,
        unit='deg',
        when=lambda a, c: _one_direction(a),
        section=FIELD,
    ),
    Question(
        'theta_start',
        'θ (azimuth in the xy plane) at the start of the rotation (deg)',
        kind='float',
        default=0.0,
        unit='deg',
        when=lambda a, c: _rotation(a),
        section=FIELD,
    ),
    Question(
        'theta_end',
        'θ at the end of the rotation (deg)',
        kind='float',
        default=0.0,
        unit='deg',
        when=lambda a, c: _rotation(a),
        section=FIELD,
    ),
    Question(
        'phi_start',
        'φ (polar angle from +z) at the start of the rotation (deg)',
        kind='float',
        default=0.0,
        unit='deg',
        when=lambda a, c: _rotation(a),
        section=FIELD,
    ),
    Question(
        'phi_end',
        'φ at the end of the rotation (deg)',
        kind='float',
        default=90.0,
        unit='deg',
        when=lambda a, c: _rotation(a),
        section=FIELD,
    ),
    Question(
        'num_angles',
        'Field directions of the rotation (equally spaced, ends included)',
        kind='int',
        default=37,
        validate=_at_least_two,
        when=lambda a, c: _rotation(a),
        section=FIELD,
    ),
    Question(
        'minimum_frequency',
        'Smallest frequency reported (kT)',
        kind='float',
        default=0.0,
        unit='kT',
        level='advanced',
        section=ORBITS,
    ),
    Question(
        'frequency_tolerance',
        'Fractional frequency difference of orbits averaged together',
        kind='float',
        default=0.01,
        level='advanced',
        section=ORBITS,
    ),
    Question(
        'orbit_tolerance',
        'Fractional distance (of the reciprocal cell) of orbits averaged together',
        kind='float',
        default=0.05,
        level='advanced',
        section=ORBITS,
    ),
    Question(
        'allow_wall_orbits',
        'Keep the extremal orbits close to the supercell walls?',
        kind='bool',
        default=True,
        level='advanced',
        section=ORBITS,
    ),
    Question(
        'verbose',
        'Also write the short, long and orbit-outline files?',
        kind='bool',
        default=False,
        level='advanced',
        section=ORBITS,
    ),
    Question(
        'plot',
        'Also generate a plotting script (plot.pyskeaf.py: the figures saved as PDF)?',
        kind='bool',
        default=True,
        section='Output',
        gui=False,  # the Studio: "Write the plot scripts" of its Output box (options.plot)
    ),
)


def assemble_pyskeaf(answers: Answers, context: Context) -> dict:
    """Configuration of :func:`PAOFLOW.gen.pyskeaf_driver.build_pyskeaf_script`.

    Parameters
    ----------
    answers : dict
        The answers.
    context : Context
        The session context.

    Returns
    -------
    dict
        ``outputdir``, ``bands``, ``fermi_energy``, ``num_interpolation``, ``b_field``,
        ``azimuthal`` and ``polar`` (degrees: scalars, or (start, end) of a rotation),
        ``num_angles`` and the orbit settings.
    """
    b_field = answers['b_field']
    if b_field == 'rotation':
        azimuthal = (answers['theta_start'], answers['theta_end'])
        polar = (answers['phi_start'], answers['phi_end'])
        num_angles = answers['num_angles']
    elif b_field == 'non_principal':
        azimuthal, polar, num_angles = answers['theta'], answers['phi'], 1
    else:
        azimuthal, polar, num_angles = 0.0, 0.0, 1
    return {
        'outputdir': answers['outputdir'],
        'bands': answers['bands'],
        'fermi_energy': answers['fermi_energy'],
        'num_interpolation': answers['num_interpolation'],
        'b_field': b_field,
        'azimuthal': azimuthal,
        'polar': polar,
        'num_angles': num_angles,
        'minimum_frequency': answers['minimum_frequency'],
        'frequency_tolerance': answers['frequency_tolerance'],
        'orbit_tolerance': answers['orbit_tolerance'],
        'allow_wall_orbits': answers['allow_wall_orbits'],
        'verbose': answers['verbose'],
        'plot': answers['plot'],
    }


PYSKEAF = Workflow(
    name='pyskeaf',
    title='Quantum oscillations (pyskeaf)',
    description='de Haas–van Alphen frequencies against the field direction (θ, φ) from the '
    'Fermi-surface BXSF files of a Property run (main.pyskeaf.py).',
    items=PYSKEAF_ITEMS,
    assemble=assemble_pyskeaf,
)
