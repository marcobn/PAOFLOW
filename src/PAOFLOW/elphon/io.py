"""QE input writing for the finite-difference electron-phonon workflow.

Reuses the phonon module's QE-input helpers to write one static SCF input per
displaced supercell.  Each displaced run gets a **unique prefix** (hence a
unique ``tmp_<prefix>`` output directory) so its wavefunctions survive and can be
re-projected by the analyse phase; the band count ``nbnd`` is sized to the PAO
projection basis of the chosen configuration.  A JSON manifest records the
displacement metadata (including the per-run prefix) so the analyse phase can
pair the +/- cells and locate each ``.save``.
"""

import json
import os
import warnings

from ..phonon.io import (
    _pp_filenames,
    _qe_input_text,
    _supercell_kgrid,
    resolve_phonon_dir,
)

MANIFEST = 'displacements.json'


def _disp_prefix(base_prefix, index, width):
    return '%s_disp%0*d' % (base_prefix, width, index)


def write_eph_displaced_supercells(
    data_controller,
    cells,
    meta,
    elphon_dir='elphon',
    pp_dir=None,
    prefix=None,
    kgrid=None,
    hubbard_card=None,
    configuration='standard',
    nbnd=None,
    is_plusminus='auto',
):
    """Write ``supercell.in`` (reference) + ``disp-NNN.in`` for every cell.

    Each displaced input carries a unique QE ``prefix`` so the runs do not
    clobber each other's ``.save``.  ``nbnd`` defaults to the PAO basis size for
    ``configuration`` (:func:`PAOFLOW.elphon.basis.supercell_nbnd`); when the
    pseudopotentials cannot be located it falls back to the QE default (no
    ``nbnd`` line) with a warning.

    Returns the list of written ``disp-NNN.in`` paths.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']
    out_dir = resolve_phonon_dir(data_controller, elphon_dir)

    base_prefix = prefix
    if base_prefix is None:
        savedir = attr.get('savedir', None)
        base_prefix = (
            os.path.basename(str(savedir)).replace('.save', '') if savedir else 'supercell'
        )
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')
    pp_filenames = _pp_filenames(data_controller)
    if kgrid is None:
        kgrid = _supercell_kgrid(data_controller, phonon.supercell_matrix)

    if nbnd is None:
        from .basis import supercell_nbnd

        try:
            nbnd = supercell_nbnd(
                data_controller, phonon.supercell, configuration=configuration, pp_dir=pp_dir
            )
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(
                'Could not size nbnd from the %r basis (%s); falling back to the '
                'QE default. Pass nbnd=... or a valid pp_dir to set it explicitly.'
                % (configuration, exc)
            )
            nbnd = None

    written = []
    if getattr(data_controller, 'rank', 0) != 0:
        return written

    width = max(3, len(str(len(cells))))

    def _text(cell, pfx):
        return _qe_input_text(
            data_controller,
            cell,
            phonon.supercell_matrix,
            pfx,
            pp_dir,
            pp_filenames,
            kgrid,
            hubbard_card=hubbard_card,
            nbnd=nbnd,
        )

    reference_prefix = base_prefix + '_ref'
    with open(os.path.join(out_dir, 'supercell.in'), 'w') as f:
        f.write(_text(phonon.supercell, reference_prefix))

    manifest_meta = []
    for i, (cell, m) in enumerate(zip(cells, meta), start=1):
        pfx = _disp_prefix(base_prefix, i, width)
        path = os.path.join(out_dir, 'disp-{0:0{w}}.in'.format(i, w=width))
        with open(path, 'w') as f:
            f.write(_text(cell, pfx))
        written.append(path)
        entry = dict(m)
        entry['prefix'] = pfx
        entry['input'] = os.path.basename(path)
        manifest_meta.append(entry)

    with open(os.path.join(out_dir, MANIFEST), 'w') as f:
        json.dump(
            {
                'displacement_distance': meta[0]['distance'] if meta else None,
                'configuration': configuration,
                'nbnd': nbnd,
                'is_plusminus': is_plusminus,
                'reference_prefix': reference_prefix,
                'displacements': manifest_meta,
            },
            f,
            indent=2,
        )

    return written
