"""PAO basis-size helpers for the electron-phonon supercell inputs.

The DFT band count (``nbnd``) of a displaced supercell must cover the PAO
projection basis, whose size is fixed by the basis configuration
(``minimal`` / ``standard`` / ``extended``) -- the *same framework* used to
generate the projection basis in a regular PAOFLOW run
(:func:`PAOFLOW.basis_gen.driver._default_shells`).  A shell ``<n><L>`` carries
``2L + 1`` orbitals, so the number of PAO orbitals of a supercell is the sum of
``2L + 1`` over every shell of every atom for the chosen preset.
"""

import os

from ..phonon.io import _pp_filenames

_L_INDEX = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}


def _orbitals_per_shell(label):
    """Number of magnetic orbitals ``2L + 1`` carried by a ``<n><L>`` shell."""
    if len(label) < 2 or label[1].upper() not in _L_INDEX:
        raise ValueError('shell label %r not understood' % label)
    return 2 * _L_INDEX[label[1].upper()] + 1


def species_pao_orbitals(upf_path, configuration='standard'):
    """PAO orbital count of one species for the given basis configuration."""
    from ..basis_gen.driver import _default_shells
    from ..inputs.read_upf import UPF

    if not os.path.isfile(upf_path):
        raise FileNotFoundError('Pseudopotential %s not found to size nbnd.' % upf_path)
    upf = UPF(upf_path)
    shells = _default_shells(upf, preset=configuration)
    return sum(_orbitals_per_shell(s) for s in shells)


def supercell_nbnd(data_controller, cell, configuration='standard', pp_dir=None, margin=0):
    """Band count that covers the PAO basis of ``cell`` (the projection size).

    Sums the per-species PAO orbital count over every atom of ``cell`` using the
    ``configuration`` preset, then adds ``margin`` empty bands.  Raises when a
    species pseudopotential cannot be located (the caller may then fall back to
    the QE default band count).
    """
    arry, attr = data_controller.data_dicts()
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')

    pp_filenames = _pp_filenames(data_controller)  # {symbol: basename}

    cache = {}
    nawf = 0
    for sym in map(str, cell.symbols):
        if sym not in cache:
            fname = pp_filenames.get(sym)
            if fname is None:
                raise ValueError('No pseudopotential for species %r to size nbnd.' % sym)
            cache[sym] = species_pao_orbitals(
                os.path.join(pp_dir, fname), configuration=configuration
            )
        nawf += cache[sym]

    return int(nawf + int(margin))
