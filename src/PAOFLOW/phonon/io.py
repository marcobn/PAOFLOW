"""I/O for the PAOFLOW finite-displacement phonon workflow.

Responsibilities (Stage 1):

* write complete, ready-to-run Quantum ESPRESSO ``pw.x`` SCF inputs for the
  phonopy-generated displaced supercells (phonopy naming ``supercell-NNN.in``);
* harvest atomic forces from the corresponding ``pw.x`` outputs;
* ingest externally produced ``FORCE_SETS`` files;
* persist a ``FORCE_SETS`` for provenance.

All structural quantities follow the phonopy QE convention (Bohr lengths,
Ry/au forces), matching ``Phonopy(..., calculator='qe')`` set up in
:mod:`PAOFLOW.phonon.do_phonopy`.
"""

import os

import numpy as np


def resolve_phonon_dir(data_controller, phonon_dir='phonon'):
    """Return (and create) the directory holding the displaced supercells.

    ``phonon_dir`` is resolved relative to the PAOFLOW output directory
    (``attr['opath']``) unless an absolute path is given.
    """
    _, attr = data_controller.data_dicts()
    base = attr.get('opath', '.')
    path = phonon_dir if os.path.isabs(phonon_dir) else os.path.join(base, phonon_dir)
    if attr.get('rank', 0) == 0 or 'rank' not in attr:
        os.makedirs(path, exist_ok=True)
    return path


def _pp_filenames(data_controller):
    """Build a ``{element_symbol: pseudopotential_filename}`` mapping."""
    from .structure import _element_symbol

    arry, _ = data_controller.data_dicts()
    species = arry.get('species', None)
    pp = {}
    if species is not None:
        for entry in species:
            label, pseudo = entry[0], entry[1]
            sym = _element_symbol(label)
            pp.setdefault(sym, os.path.basename(str(pseudo)))
    return pp


def _supercell_kgrid(data_controller, supercell_matrix):
    """Scale the unit-cell Monkhorst-Pack grid down to the supercell.

    For a diagonal supercell the per-axis k-density is divided by the diagonal
    multiplicity; for a general matrix the (cube-root of the) determinant is
    used as an isotropic fallback.
    """
    _, attr = data_controller.data_dicts()
    nk = np.array(
        [attr.get('nk1', 1) or 1, attr.get('nk2', 1) or 1, attr.get('nk3', 1) or 1],
        dtype=float,
    )
    sc = np.asarray(supercell_matrix, dtype=float)
    if np.count_nonzero(sc - np.diag(np.diagonal(sc))) == 0:
        diag = np.diagonal(sc)
    else:
        diag = np.full(3, round(abs(np.linalg.det(sc)) ** (1.0 / 3.0)))
    diag = np.where(diag == 0, 1, diag)
    kg = np.maximum(1, np.ceil(nk / diag)).astype(int)
    return kg


def _namelists(data_controller, supercell, prefix, pp_dir):
    """Assemble the ``&control/&system/&electrons`` namelist block."""
    _, attr = data_controller.data_dicts()

    nat = len(supercell)
    ntyp = len(set(map(str, supercell.symbols)))
    ecutwfc = float(attr.get('ecutwfc', 60.0))  # Ry
    ecutrho = float(attr.get('ecutrho', 4.0 * ecutwfc))  # Ry
    nspin = int(attr.get('nspin', 1))
    insulator = bool(attr.get('insulator', True))

    lines = []
    lines.append('&control')
    lines.append("    calculation = 'scf'")
    lines.append("    prefix = '%s'" % prefix)
    lines.append("    outdir = './tmp_%s'" % prefix)
    lines.append("    pseudo_dir = '%s'" % pp_dir)
    lines.append('    tprnfor = .true.')
    lines.append('    tstress = .false.')
    lines.append('/')
    lines.append('&system')
    lines.append('    ibrav = 0')
    lines.append('    nat = %d' % nat)
    lines.append('    ntyp = %d' % ntyp)
    lines.append('    ecutwfc = %.2f' % ecutwfc)
    lines.append('    ecutrho = %.2f' % ecutrho)
    if not insulator:
        lines.append("    occupations = 'smearing'")
        lines.append("    smearing = 'mp'")
        lines.append('    degauss = %.4f' % float(attr.get('degauss', 0.02)))
    if nspin == 2:
        lines.append('    nspin = 2')
        lines.append('    starting_magnetization(1) = 0.1')
    lines.append('/')
    lines.append('&electrons')
    lines.append('    conv_thr = 1.0d-8')
    lines.append('    mixing_beta = 0.7')
    lines.append('/')
    return '\n'.join(lines)


def _qe_input_text(data_controller, cell, supercell_matrix, prefix, pp_dir, pp_filenames, kgrid):
    """Compose a complete ``pw.x`` SCF input for one (super)cell."""
    from phonopy.interface.qe import get_pwscf_structure

    header = _namelists(data_controller, cell, prefix, pp_dir)
    structure = get_pwscf_structure(cell, pp_filenames=pp_filenames)
    k1, k2, k3 = kgrid
    kpoints = 'K_POINTS automatic\n %d %d %d 0 0 0\n' % (int(k1), int(k2), int(k3))
    return header + '\n' + structure + '\n' + kpoints


def write_displaced_supercells(
    data_controller,
    phonon_dir='phonon',
    pp_dir=None,
    prefix=None,
    kgrid=None,
):
    """Write QE SCF inputs for every phonopy-generated displaced supercell.

    Requires that displacements have already been generated on the stored
    :class:`phonopy.Phonopy` object (``arry['phonopy']``).

    Returns
    -------
    list[str]
        Absolute paths of the written ``supercell-NNN.in`` files (perfect
        supercell ``supercell.in`` excluded).
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)

    if prefix is None:
        savedir = attr.get('savedir', None)
        prefix = os.path.basename(str(savedir)).replace('.save', '') if savedir else 'supercell'
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')

    pp_filenames = _pp_filenames(data_controller)
    if kgrid is None:
        kgrid = _supercell_kgrid(data_controller, phonon.supercell_matrix)

    rank = getattr(data_controller, 'rank', 0)
    written = []

    supercells = phonon.supercells_with_displacements
    ndisp = len(supercells)
    width = max(3, len(str(ndisp)))

    if rank == 0:
        # Perfect supercell (reference / provenance).
        with open(os.path.join(out_dir, 'supercell.in'), 'w') as f:
            f.write(
                _qe_input_text(
                    data_controller,
                    phonon.supercell,
                    phonon.supercell_matrix,
                    prefix,
                    pp_dir,
                    pp_filenames,
                    kgrid,
                )
            )
        for i, cell in enumerate(supercells, start=1):
            fname = os.path.join(out_dir, 'supercell-{0:0{w}}.in'.format(i, w=width))
            with open(fname, 'w') as f:
                f.write(
                    _qe_input_text(
                        data_controller,
                        cell,
                        phonon.supercell_matrix,
                        prefix,
                        pp_dir,
                        pp_filenames,
                        kgrid,
                    )
                )
            written.append(fname)

        # Persist the displacement dataset for provenance / restart.
        phonon.save(os.path.join(out_dir, 'phonopy_disp.yaml'))

        if attr.get('verbose', False):
            print('Wrote %d displaced supercell inputs to %s' % (ndisp, out_dir))

    return written


def harvest_qe_forces(data_controller, phonon_dir='phonon'):
    """Parse forces from ``supercell-NNN.out`` files and set them on phonopy.

    Returns
    -------
    numpy.ndarray
        Force sets of shape ``(num_displacements, natoms_supercell, 3)`` in
        Ry/au.
    """
    from phonopy.interface.qe import parse_set_of_forces

    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    ndisp = len(phonon.supercells_with_displacements)
    width = max(3, len(str(ndisp)))

    filenames = []
    for i in range(1, ndisp + 1):
        fn = os.path.join(out_dir, 'supercell-{0:0{w}}.out'.format(i, w=width))
        if not os.path.isfile(fn):
            raise FileNotFoundError('Missing QE output for displacement %d: %s' % (i, fn))
        filenames.append(fn)

    natoms = len(phonon.supercell)
    force_sets = parse_set_of_forces(natoms, filenames, verbose=attr.get('verbose', False))
    force_sets = np.array(force_sets)

    phonon.forces = force_sets
    return force_sets


def ingest_force_sets(data_controller, force_sets_path):
    """Load an external ``FORCE_SETS`` onto the stored phonopy object.

    The displacement/forces dataset fully replaces the current dataset, so the
    supercell matrix used to build the phonopy object must match the one that
    produced the ``FORCE_SETS``.
    """
    from phonopy.file_IO import parse_FORCE_SETS

    arry, _ = data_controller.data_dicts()
    phonon = arry['phonopy']

    natoms = len(phonon.supercell)
    dataset = parse_FORCE_SETS(natom=natoms, filename=force_sets_path)
    phonon.dataset = dataset
    return dataset


def write_force_sets(data_controller, phonon_dir='phonon', filename='FORCE_SETS'):
    """Write the current displacement/forces dataset to ``FORCE_SETS``."""
    from phonopy.file_IO import write_FORCE_SETS

    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    rank = getattr(data_controller, 'rank', 0)
    if rank != 0:
        return None

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    path = os.path.join(out_dir, filename)
    write_FORCE_SETS(phonon.dataset, filename=path)
    return path
