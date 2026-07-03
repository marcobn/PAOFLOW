"""phonopy driver entry points for PAOFLOW.

Builds and drives the :class:`phonopy.Phonopy` object: structure conversion
and object creation (Stage 0), then the finite-displacement harmonic workflow
(Stage 1) -- displacement generation, second-order force constants and the
derived dispersion, density of states and thermal properties.
"""

import os

import numpy as np

from .structure import paoflow_to_phonopy

# Frequency conversion factor: phonopy returns THz; multiply to obtain cm^-1.
THZ_TO_CM1 = 33.35641


def _normalize_supercell_matrix(supercell_matrix):
    """Coerce a user supercell specification into a ``(3, 3)`` integer array.

    Accepts a scalar (isotropic diagonal), a length-3 sequence (anisotropic
    diagonal) or a full ``(3, 3)`` matrix.
    """
    sc = np.asarray(supercell_matrix)
    if sc.ndim == 0:
        sc = np.diag([int(sc)] * 3)
    elif sc.ndim == 1:
        if sc.shape[0] != 3:
            raise ValueError('A diagonal supercell_matrix must have length 3.')
        sc = np.diag(sc.astype(int))
    elif sc.shape != (3, 3):
        raise ValueError('supercell_matrix must be a scalar, length-3 vector or 3x3 matrix.')
    return np.asarray(sc, dtype=int)


def init_phonopy(data_controller):
    """Create a :class:`phonopy.Phonopy` object and store it on the controller.

    Reads from DataController
    -------------------------
    Structure: ``alat``, ``a_vectors``, ``tau``, ``atoms`` (via
    :func:`paoflow_to_phonopy`).
    Configuration: ``phonon_supercell_matrix`` (required),
    ``phonon_primitive_matrix`` (optional), ``phonon_displacement_distance``
    (optional).

    Writes to DataController
    ------------------------
    ``phonopy`` : the :class:`phonopy.Phonopy` instance handle.

    Returns
    -------
    phonopy.Phonopy
        The initialised object (also stored under ``arry['phonopy']``).
    """
    from phonopy import Phonopy

    arry, attr = data_controller.data_dicts()

    if 'phonon_supercell_matrix' not in attr or attr['phonon_supercell_matrix'] is None:
        raise ValueError(
            'phonon_supercell_matrix must be set before initialising phonopy. '
            'Call PAOFLOW.phonons(...) / phonon setup with supercell_matrix=...'
        )

    unitcell = paoflow_to_phonopy(data_controller)
    supercell_matrix = _normalize_supercell_matrix(attr['phonon_supercell_matrix'])

    # Default to the identity primitive matrix so the phonopy primitive cell
    # coincides with the PAOFLOW unit cell; the user may request 'auto'/'P' or
    # supply an explicit 3x3 matrix.
    primitive_matrix = attr.get('phonon_primitive_matrix', None)
    if primitive_matrix is None:
        primitive_matrix = 'P'
    elif not isinstance(primitive_matrix, str):
        primitive_matrix = np.asarray(primitive_matrix, dtype=float)

    phonon = Phonopy(
        unitcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        calculator='qe',
    )

    arry['phonopy'] = phonon

    if attr.get('verbose', False):
        print('phonopy initialised:')
        print('  unit cell atoms      :', len(unitcell))
        print('  supercell matrix     :\n', supercell_matrix)
        print('  supercell atoms      :', len(phonon.supercell))
        print('  primitive matrix     :', primitive_matrix)

    return phonon


def generate_displacements(data_controller):
    """Generate symmetry-reduced finite displacements on the phonopy object.

    Reads ``phonon_displacement_distance`` (Bohr) from the controller and
    populates ``phonon.supercells_with_displacements``.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    distance = attr.get('phonon_displacement_distance', 0.01)
    phonon.generate_displacements(distance=distance)

    if attr.get('verbose', False):
        ndisp = len(phonon.supercells_with_displacements)
        print('Generated %d displaced supercells (distance = %g Bohr).' % (ndisp, distance))

    return phonon.supercells_with_displacements


def produce_force_constants(data_controller, forces=None):
    """Produce the second-order force constants (fc2) from the force sets.

    ``forces`` may be supplied explicitly (array ``(ndisp, natoms, 3)`` in
    Ry/au); otherwise the forces/dataset already attached to the phonopy
    object (via :mod:`PAOFLOW.phonon.io`) are used.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    if forces is not None:
        phonon.forces = np.asarray(forces)

    phonon.produce_force_constants(show_drift=attr.get('verbose', False))
    return phonon.force_constants


def attach_nac(data_controller, born=None, dielectric=None, born_file=None, factor=None):
    """Attach non-analytical correction (NAC) parameters to the phonopy object.

    Enables LO-TO splitting near :math:`\\Gamma` by setting
    ``phonon.nac_params``.  Once attached, phonopy applies the correction
    automatically to the dispersion, DOS and thermal properties.

    Parameters
    ----------
    born : array_like, optional
        Born effective charges ``(natom_prim, 3, 3)`` in units of the
        elementary charge.  Required (with ``dielectric``) unless ``born_file``
        is given.
    dielectric : array_like, optional
        ``(3, 3)`` high-frequency dielectric tensor.
    born_file : str, optional
        Path to a phonopy ``BORN`` file; when supplied it takes precedence and
        provides ``born``, ``dielectric`` and the unit-conversion ``factor``.
    factor : float, optional
        NAC unit-conversion factor.  Defaults to the phonopy Quantum ESPRESSO
        value when building ``nac_params`` from explicit arrays.

    Returns
    -------
    dict
        The ``nac_params`` dictionary that was attached.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    if born_file is not None:
        from .io import read_born_file

        nac_params = read_born_file(data_controller, born_file)
    else:
        if born is None or dielectric is None:
            raise ValueError('attach_nac requires either born_file or both born and dielectric.')
        if factor is None:
            from phonopy.interface.calculator import get_calculator_physical_units

            factor = float(get_calculator_physical_units('qe').nac_factor)
        nac_params = {
            'born': np.asarray(born, dtype=float),
            'dielectric': np.asarray(dielectric, dtype=float),
            'factor': float(factor),
        }

    phonon.nac_params = nac_params
    arry['born_charges'] = nac_params['born']
    arry['dielectric_tensor'] = nac_params['dielectric']

    if attr.get('verbose', False):
        zsum = np.asarray(nac_params['born']).sum(axis=0)
        print('NAC parameters attached (LO-TO splitting enabled):')
        print('  dielectric tensor (diag):', np.diag(nac_params['dielectric']))
        print('  sum of Born charges (acoustic sum rule):\n', zsum)

    return nac_params


def _write_rows(data_controller, fname, header, rows):
    """Write a whitespace-delimited table to ``opath/<fname>`` on rank 0."""
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    path = os.path.join(attr.get('opath', '.'), fname)
    with open(path, 'w') as f:
        if header:
            f.write('# ' + header + '\n')
        for row in rows:
            f.write(' '.join('% 16.8e' % v for v in row) + '\n')
    return path


# Display names for the Greek high-symmetry labels emitted by PAOFLOW's
# ``_getHighSymPoints`` (which uses a ``g`` prefix, e.g. ``gG`` for Gamma).
_GREEK_LABELS = {
    'gG': r'$\Gamma$',
    'gS': r'$\Sigma$',
    'gS1': r'$\Sigma_1$',
    'gD': r'$\Delta$',
}


def _phonopy_label(name):
    """Convert a PAOFLOW high-symmetry point name to a display label."""
    if name in _GREEK_LABELS:
        return _GREEK_LABELS[name]
    import re

    m = re.match(r'^([A-Za-z]+)(\d+)$', name)
    if m:
        return r'$%s_{%s}$' % (m.group(1), m.group(2))
    return name


def _qe_path_to_phonopy(special_points, band_path):
    """Convert PAOFLOW ``(special_points, band_path)`` to phonopy band paths.

    ``band_path`` is a string such as ``'gG-X-W-K-gG-L|U-X'`` where ``-``
    separates consecutive points within a continuous segment and ``|`` marks a
    discontinuity. The result is ``(band_paths, labels)`` where ``band_paths``
    is the phonopy list-of-segments format and ``labels`` is the flat list of
    display labels (one per high-symmetry point).
    """
    band_paths = []
    labels = []
    for segment in band_path.split('|'):
        names = segment.split('-')
        band_paths.append([list(np.asarray(special_points[n], dtype=float)) for n in names])
        labels.extend(_phonopy_label(n) for n in names)
    return band_paths, labels


def default_q_path(data_controller):
    """Derive a high-symmetry q-path (phonopy format) from the QE ``ibrav``.

    Returns ``(band_paths, labels)`` or ``(None, None)`` when no ibrav-based
    path is available (``ibrav`` unset/``0`` or an unsupported lattice).
    """
    arry, attr = data_controller.data_dicts()
    ibrav = attr.get('ibrav', None)
    if ibrav is None:
        return None, None
    try:
        ibrav = int(ibrav)
    except (TypeError, ValueError):
        return None, None
    if ibrav == 0:
        return None, None
    try:
        from ..spectrum.kpnts_interpolation_mesh import _getHighSymPoints

        cell = np.asarray(arry['a_vectors'], dtype=float)
        special_points, band_path = _getHighSymPoints(ibrav, attr['alat'], cell)
        return _qe_path_to_phonopy(special_points, band_path)
    except Exception:
        return None, None


def _band_tick_positions(distances, connections):
    """Distances of the high-symmetry ticks, aligned with the flat label list.

    ``distances`` is one array per consecutive point pair; ``connections`` is
    the phonopy ``path_connections`` list (``False`` marks the end of a
    continuous segment). At a discontinuity two labels share the same distance,
    so the number of ticks equals the number of high-symmetry points.
    """
    if connections is None:
        return [seg[0] for seg in distances] + [distances[-1][-1]]
    positions = []
    group_start = True
    for i, conn in enumerate(connections):
        if group_start:
            positions.append(distances[i][0])
            group_start = False
        positions.append(distances[i][-1])
        if not conn:
            group_start = True
    return positions


def compute_phonon_bands(
    data_controller,
    q_path=None,
    q_labels=None,
    npoints=101,
    units='THz',
    fname='phonon',
):
    """Compute the phonon dispersion and write ``<fname>_band.dat``.

    When ``q_path`` is ``None`` a high-symmetry path is derived from the QE
    ``ibrav`` (see :func:`default_q_path`); if that is unavailable the path is
    generated automatically via seekpath. Otherwise ``q_path`` is a sequence of
    path segments in fractional reciprocal coordinates.

    Writes ``<fname>_band.dat`` (distance + branch frequencies) and, when tick
    labels are available, ``<fname>_band.labels`` (tick distance + label).
    """
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    scale = THZ_TO_CM1 if str(units).lower() in ('cm-1', 'cm^-1', 'cm') else 1.0

    # When no explicit path is given, prefer the ibrav-based high-symmetry path.
    if q_path is None:
        auto_path, auto_labels = default_q_path(data_controller)
        if auto_path is not None:
            q_path, q_labels = auto_path, auto_labels

    connections = None
    flat_labels = None
    if q_path is None:
        try:
            phonon.auto_band_structure(npoints=npoints, with_eigenvectors=False)
        except ModuleNotFoundError:
            if getattr(data_controller, 'rank', 0) == 0:
                print(
                    'Phonon bands skipped: no ibrav-based path is available and '
                    "automatic high-symmetry paths require the optional 'seekpath' "
                    'package. Install seekpath or pass an explicit q_path to '
                    'phonons().'
                )
            return None
    else:
        qpoints, connections = get_band_qpoints_and_path_connections(q_path, npoints=npoints)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=q_labels)
        if q_labels is not None:
            flat_labels = list(q_labels)

    bsd = phonon.get_band_structure_dict()
    distances = bsd['distances']
    frequencies = bsd['frequencies']

    rows = []
    for seg_d, seg_f in zip(distances, frequencies):
        for d, fq in zip(seg_d, seg_f):
            rows.append([d] + list(np.asarray(fq) * scale))

    band_path = _write_rows(
        data_controller,
        fname + '_band.dat',
        'distance  frequencies(%s)' % units,
        rows,
    )

    # Tick labels: explicit/ibrav labels take precedence over phonopy-resolved.
    if flat_labels is None:
        labels = bsd.get('labels', None)
        if labels is not None:
            flat_labels = []
            for pair in labels:
                if not flat_labels:
                    flat_labels.append(pair[0])
                flat_labels.append(pair[1])

    if flat_labels is not None and getattr(data_controller, 'rank', 0) == 0:
        tick_distances = _band_tick_positions(distances, connections)
        lpath = os.path.join(attr.get('opath', '.'), fname + '_band.labels')
        with open(lpath, 'w') as f:
            f.write('# tick_distance  label\n')
            for d, lab in zip(tick_distances, flat_labels):
                f.write('% 16.8e  %s\n' % (d, lab))

    return band_path


def compute_phonon_dos(
    data_controller,
    mesh=None,
    sigma=None,
    units='THz',
    fname='phonon',
):
    """Compute the total phonon DOS and write ``<fname>_dos.dat``."""
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    if mesh is None:
        mesh = attr.get('phonon_q_mesh', None) or [20, 20, 20]

    scale = THZ_TO_CM1 if str(units).lower() in ('cm-1', 'cm^-1', 'cm') else 1.0

    phonon.run_mesh(mesh, is_gamma_center=True)
    phonon.run_total_dos(sigma=sigma)
    dosd = phonon.get_total_dos_dict()

    freqs = np.asarray(dosd['frequency_points']) * scale
    dos = np.asarray(dosd['total_dos'])
    rows = list(zip(freqs, dos))

    return _write_rows(
        data_controller,
        fname + '_dos.dat',
        'frequency(%s)  total_dos' % units,
        rows,
    )


def compute_thermal_properties(
    data_controller,
    mesh=None,
    t_min=0.0,
    t_max=1000.0,
    t_step=10.0,
    fname='phonon',
):
    """Compute harmonic thermal properties and write ``<fname>_thermal.dat``.

    Columns: temperature (K), Helmholtz free energy (kJ/mol), entropy
    (J/K/mol), constant-volume heat capacity (J/K/mol).
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    if mesh is None:
        mesh = attr.get('phonon_q_mesh', None) or [20, 20, 20]

    phonon.run_mesh(mesh, is_gamma_center=True)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    tpd = phonon.get_thermal_properties_dict()

    rows = list(
        zip(
            tpd['temperatures'],
            tpd['free_energy'],
            tpd['entropy'],
            tpd['heat_capacity'],
        )
    )

    return _write_rows(
        data_controller,
        fname + '_thermal.dat',
        'T(K)  free_energy(kJ/mol)  entropy(J/K/mol)  Cv(J/K/mol)',
        rows,
    )
