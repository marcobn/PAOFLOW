"""One entry point for the pseudopotential data of a run: UPF files (QE) or POTCAR datasets (VASP).

The projections, the basis builders, the generator and the position dipoles read the
projectors, augmentation charges, partial waves and pseudo-orbitals of every species through
the attributes of :class:`~PAOFLOW.inputs.read_upf.UPF`. For a VASP run
:class:`~PAOFLOW.inputs.read_potcar.PotcarDataset` provides the same attributes, so the consumers
call :func:`load_pseudopotentials` (or :func:`load_pseudopotential`) instead of opening
``fpath/<pseudo>`` themselves. For QE the loader builds exactly the ``UPF`` objects the callers
built before.
"""

from __future__ import annotations

import os
import re
import warnings

from .read_potcar import PotcarDataset, find_potcar_datasets, read_potcar
from .read_upf import UPF


def _element_of(species_name: str) -> str:
    return re.split(r'\d+', species_name)[0]


def read_pseudopotential_file(path: str) -> UPF | PotcarDataset:
    """Parse a pseudopotential file of either format.

    Parameters
    ----------
    path : str
        A UPF file (v1 or v2) or a ``POTCAR``.

    Returns
    -------
    UPF or PotcarDataset
        The first dataset of a concatenated ``POTCAR`` (with a warning when it holds several;
        :func:`~PAOFLOW.inputs.read_potcar.read_potcar` returns them all).
    """
    with open(path) as handle:
        head = ''
        for line in handle:
            if line.strip():
                head = line.strip()
                break
    if head.startswith('<UPF') or head.startswith('<PP_INFO') or head.startswith('<?xml'):
        return UPF(path)
    datasets = read_potcar(path)
    if not datasets:
        raise ValueError('%s: neither a UPF file nor a POTCAR' % path)
    if len(datasets) > 1:
        warnings.warn(
            '%s holds %d datasets; using the first (%s)' % (path, len(datasets), datasets[0].title),
            RuntimeWarning,
            stacklevel=2,
        )
    return datasets[0]


def load_pseudopotentials(data_controller) -> dict[str, UPF | PotcarDataset]:
    """The pseudopotential data of every species of the run.

    Parameters
    ----------
    data_controller : DataController
        Required arrays: ``species`` (``(name, file)`` for QE, ``(name, TITEL)`` for VASP);
        attributes ``dft``, ``fpath``; optional ``potcar_dir``.

    Returns
    -------
    dict
        ``{species: UPF}`` for QE input, ``{species: PotcarDataset}`` for VASP input (read from
        ``fpath/POTCAR`` or from the PAW library ``potcar_dir`` / ``$PAOFLOW_POTCAR_DIR``, see
        :func:`~PAOFLOW.inputs.read_potcar.find_potcar_datasets`).

    Raises
    ------
    FileNotFoundError
        VASP input whose datasets are found nowhere (the message lists the places searched).
    ValueError
        Two species of a VASP run with the same element (``Fe`` and ``Fe_pv``): the atoms of
        ``vasprun.xml`` carry the bare symbol and cannot be told apart.
    """
    arry, attr = data_controller.data_dicts()
    species = list(arry['species'])
    if attr.get('dft', 'QE') == 'VASP':
        names = [name for name, _ in species]
        elements = [_element_of(name) for name in names]
        if len(set(elements)) != len(elements):
            raise ValueError(
                'species %s of the VASP run share an element: the atoms cannot be assigned to '
                'their POTCAR datasets' % names
            )
        datasets = find_potcar_datasets(
            [title for _, title in species], attr.get('fpath'), attr.get('potcar_dir')
        )
        return dict(zip(names, datasets))
    return {name: UPF(os.path.join(attr['fpath'], pseudo)) for name, pseudo in species}


def load_pseudopotential(data_controller, name: str) -> tuple[UPF | PotcarDataset, str]:
    """The pseudopotential data and the file name (QE) or title (VASP) of one species.

    Parameters
    ----------
    data_controller : DataController
        See :func:`load_pseudopotentials`.
    name : str
        A species name of ``arry['species']``, or an element symbol (the first species of that
        element is used, as the basis presets do).

    Returns
    -------
    tuple
        ``(data, pseudo)``.

    Raises
    ------
    RuntimeError
        No species matches ``name``.
    """
    arry, attr = data_controller.data_dicts()
    species = list(arry.get('species') or [])
    matches = [(sp, pseudo) for sp, pseudo in species if sp == name]
    if not matches:
        matches = [(sp, pseudo) for sp, pseudo in species if _element_of(sp) == name]
    if not matches:
        raise RuntimeError('no species %r in the run (species: %s)' % (name, species))
    species_name, pseudo = matches[0]
    if attr.get('dft', 'QE') == 'VASP':
        return load_pseudopotentials(data_controller)[species_name], pseudo
    return UPF(os.path.join(attr['fpath'], pseudo)), pseudo
