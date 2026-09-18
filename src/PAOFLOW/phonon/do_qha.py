"""Quasi-harmonic approximation (QHA) driver for PAOFLOW (Stage 4).

For a set of isotropically strained volumes the static DFT energy ``E(V)`` is
combined with the harmonic vibrational free energy ``F_vib(T, V)`` to obtain the
temperature-dependent equilibrium volume ``V(T)``, volumetric thermal expansion
``alpha(T)``, isothermal bulk modulus ``B(T)``, Gibbs free energy ``G(T)``,
constant-pressure heat capacity ``C_p(T)`` and the thermodynamic Gruneisen
parameter ``gamma(T)``.

Two back-ends are used depending on how many volumes are sampled:

* ``nvolumes = 5`` (default): the full equation-of-state QHA implemented by
  :class:`phonopy.PhonopyQHA` (Vinet EOS by default).  A four-parameter EOS
  needs at least four volume points, so this is the accurate, recommended
  choice.
* ``nvolumes = 3``: a low-order (parabolic) fit of ``G(V) = E(V) + F_vib(T, V)``
  at each temperature.  A quadratic is exactly determined by three points and
  is valid for the small volume changes probed here; it produces the same set
  of output files as the EOS route.

The routine mirrors the two-phase ``generate`` / ``analyse`` pattern used by the
rest of the phonon workflow: with ``forces=None`` it writes the Quantum
ESPRESSO inputs (a static SCF for ``E(V)`` plus the displaced supercells for the
phonons at every volume); with ``forces='qe'`` it harvests the results and runs
the QHA.
"""

import os

import numpy as np

from .do_phonopy import _normalize_supercell_matrix, _write_rows, produce_force_constants
from .structure import paoflow_to_phonopy

# CODATA conversions for the PhonopyQHA inputs / thermodynamic relations.
RY_TO_EV = 13.605693122994  # Rydberg -> electron-volt
BOHR_TO_ANG = 0.529177210903  # Bohr -> Angstrom
EV_ANG3_TO_GPA = 160.21766208  # eV/Angstrom^3 -> GPa
KJMOL_TO_EV = 1.0 / 96.4853075  # kJ/mol -> eV (per formula unit)
AVOGADRO = 6.02214076e23  # 1/mol


def _volume_scales(nvolumes, strain):
    """Return the isotropic linear scale factors for the volume scan.

    ``strain`` is the maximum linear strain; the cell volume at scale ``s`` is
    ``omega * s**3``.  Supports the two sampling densities exposed to the user.
    """
    nvolumes = int(nvolumes)
    if nvolumes not in (3, 5):
        raise ValueError('quasi_harmonic supports nvolumes = 3 or 5.')
    return 1.0 + np.linspace(-float(strain), float(strain), nvolumes)


def _vol_dirname(index):
    """Sub-directory name for the ``index``-th sampled volume."""
    return 'vol-%02d' % index


def _init_phonopy_scaled(data_controller, scale):
    """Build (and store) a phonopy object for an isotropically strained cell.

    Mirrors :func:`PAOFLOW.phonon.do_phonopy.init_phonopy` but applies the
    linear ``scale`` factor to the lattice so the same supercell / primitive
    configuration is reused at every sampled volume.
    """
    from phonopy import Phonopy

    arry, attr = data_controller.data_dicts()

    if attr.get('phonon_supercell_matrix', None) is None:
        raise ValueError(
            'phonon_supercell_matrix must be set before the QHA volume scan. '
            'Call PAOFLOW.quasi_harmonic(..., supercell_matrix=...).'
        )

    unitcell = paoflow_to_phonopy(data_controller, scale=scale)
    supercell_matrix = _normalize_supercell_matrix(attr['phonon_supercell_matrix'])

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
    return phonon


def _write_static_scf(data_controller, phonon, vol_dir, pp_dir, prefix, hubbard_card):
    """Write the unit-cell static SCF input used to obtain ``E(V)``."""
    from .io import _pp_filenames, _qe_input_text, resolve_phonon_dir

    arry, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None

    out_dir = resolve_phonon_dir(data_controller, vol_dir)

    if prefix is None:
        savedir = attr.get('savedir', None)
        prefix = os.path.basename(str(savedir)).replace('.save', '') if savedir else 'scf'
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')

    pp_filenames = _pp_filenames(data_controller)
    kgrid = [
        int(attr.get('nk1', 1) or 1),
        int(attr.get('nk2', 1) or 1),
        int(attr.get('nk3', 1) or 1),
    ]

    text = _qe_input_text(
        data_controller,
        phonon.unitcell,
        phonon.supercell_matrix,
        prefix,
        pp_dir,
        pp_filenames,
        kgrid,
        hubbard_card=hubbard_card,
    )

    path = os.path.join(out_dir, 'scf.in')
    with open(path, 'w') as f:
        f.write(text)
    return path


def generate_qha_inputs(
    data_controller,
    nvolumes=5,
    strain=0.02,
    qha_dir='qha',
    pp_dir=None,
    prefix=None,
    kgrid=None,
    hubbard_card=None,
):
    """Write the QE inputs for every sampled volume (generation phase).

    For each isotropically strained cell this writes, under
    ``<outputdir>/<qha_dir>/vol-NN/``:

    * ``scf.in`` -- a static unit-cell SCF for the electronic energy ``E(V)``;
    * ``supercell-NNN.in`` -- the phonopy displaced supercells for the phonons.

    Run ``pw.x`` on every input, then re-call with ``forces='qe'``.

    Returns
    -------
    list[str]
        The sampled-volume directories that were populated.
    """
    from .do_phonopy import generate_displacements
    from .io import write_displaced_supercells

    scales = _volume_scales(nvolumes, strain)
    dirs = []
    for i, scale in enumerate(scales):
        vol_dir = os.path.join(qha_dir, _vol_dirname(i))
        phonon = _init_phonopy_scaled(data_controller, scale)
        generate_displacements(data_controller)
        write_displaced_supercells(
            data_controller,
            phonon_dir=vol_dir,
            pp_dir=pp_dir,
            prefix=prefix,
            kgrid=kgrid,
            hubbard_card=hubbard_card,
        )
        _write_static_scf(data_controller, phonon, vol_dir, pp_dir, prefix, hubbard_card)
        dirs.append(vol_dir)
    return dirs


def _collect_volume_data(data_controller, nvolumes, strain, qha_dir, mesh, t_min, t_max, t_step):
    """Harvest ``E(V)`` and the harmonic thermal properties at every volume."""
    from .do_phonopy import generate_displacements
    from .io import harvest_qe_forces, parse_qe_total_energy, resolve_phonon_dir

    scales = _volume_scales(nvolumes, strain)

    volumes = []
    energies = []
    free_energy = []
    entropy = []
    cv = []
    temperatures = None
    phonons = []

    for i, scale in enumerate(scales):
        vol_dir = os.path.join(qha_dir, _vol_dirname(i))
        out_dir = resolve_phonon_dir(data_controller, vol_dir)

        phonon = _init_phonopy_scaled(data_controller, scale)
        generate_displacements(data_controller)
        harvest_qe_forces(data_controller, phonon_dir=vol_dir)
        produce_force_constants(data_controller)

        phonon.run_mesh(mesh, is_gamma_center=True)
        phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
        tp = phonon.get_thermal_properties_dict()

        if temperatures is None:
            temperatures = np.asarray(tp['temperatures'], dtype=float)
        free_energy.append(np.asarray(tp['free_energy'], dtype=float))
        entropy.append(np.asarray(tp['entropy'], dtype=float))
        cv.append(np.asarray(tp['heat_capacity'], dtype=float))

        e_ry = parse_qe_total_energy(os.path.join(out_dir, 'scf.out'))
        energies.append(e_ry * RY_TO_EV)
        volumes.append(float(phonon.unitcell.volume) * BOHR_TO_ANG**3)
        phonons.append(phonon)

    return {
        'volumes': np.asarray(volumes),  # Angstrom^3
        'energies': np.asarray(energies),  # eV
        'temperatures': temperatures,  # K
        'free_energy': np.asarray(free_energy).T,  # (ntemp, nvol) kJ/mol
        'entropy': np.asarray(entropy).T,  # (ntemp, nvol) J/K/mol
        'cv': np.asarray(cv).T,  # (ntemp, nvol) J/K/mol
        'phonons': phonons,  # Phonopy objects (fc2 built) for mode Gruneisen
    }


def _qha_via_phonopy(data, eos, pressure, t_max):
    """Run :class:`phonopy.PhonopyQHA` (>= 4 volumes) and extract the results."""
    from phonopy import PhonopyQHA

    qha = PhonopyQHA(
        volumes=data['volumes'],
        electronic_energies=data['energies'],
        temperatures=data['temperatures'],
        free_energy=data['free_energy'],
        cv=data['cv'],
        entropy=data['entropy'],
        eos=eos,
        pressure=pressure,
        t_max=t_max,
    )

    ntemp = len(qha.volume_temperature)
    result = {
        'temperatures': np.asarray(data['temperatures'][:ntemp]),
        'volume': np.asarray(qha.volume_temperature),
        'thermal_expansion': np.asarray(qha.thermal_expansion),
        'bulk_modulus': np.asarray(qha.bulk_modulus_temperature),
        'gibbs': np.asarray(qha.gibbs_temperature),
        'heat_capacity': np.asarray(qha.heat_capacity_P_polyfit),
        'gruneisen': np.asarray(qha.gruneisen_temperature),
        'B0': float(qha.bulk_modulus),
    }
    return qha, result


def _qha_via_quadratic(data, pressure):
    """Parabolic ``F(V; T)`` QHA for the three-volume scan.

    At each temperature a quadratic is fit to ``G(V) = E(V) + F_vib(T, V)``
    (+ ``pV`` when a pressure is requested).  The minimum gives the equilibrium
    volume and Gibbs energy; the curvature gives the bulk modulus.  Thermal
    expansion follows from central differences of ``V(T)``, and the
    thermodynamic Gruneisen parameter from ``gamma = alpha * B * V / C_v``.
    """
    volumes = np.asarray(data['volumes'], dtype=float)  # Ang^3
    energies = np.asarray(data['energies'], dtype=float)  # eV
    temps = np.asarray(data['temperatures'], dtype=float)  # K
    fvib = np.asarray(data['free_energy'], dtype=float) * KJMOL_TO_EV  # eV, (ntemp, nvol)
    cv = np.asarray(data['cv'], dtype=float)  # J/K/mol, (ntemp, nvol)

    # p V term: convert pressure (GPa) so that p V is expressed in eV.
    pv = 0.0
    if pressure:
        pv = (float(pressure) / EV_ANG3_TO_GPA) * volumes  # eV per volume point

    ntemp = len(temps)
    v_eq = np.full(ntemp, np.nan)
    g_eq = np.full(ntemp, np.nan)
    bulk = np.full(ntemp, np.nan)  # GPa
    cv_eq = np.full(ntemp, np.nan)  # J/K/mol
    for it in range(ntemp):
        g = energies + fvib[it] + pv
        a, b, _c = np.polyfit(volumes, g, 2)  # g = a V^2 + b V + c
        if a <= 0:
            continue
        v0 = -b / (2.0 * a)
        v_eq[it] = v0
        g_eq[it] = np.polyval([a, b, _c], v0)
        # B = V d^2G/dV^2 = V * 2a  (eV/Ang^3 -> GPa).
        bulk[it] = v0 * (2.0 * a) * EV_ANG3_TO_GPA
        # Heat capacity at the equilibrium volume (quadratic interpolation).
        cv_eq[it] = np.polyval(np.polyfit(volumes, cv[it], 2), v0)

    # Volumetric thermal expansion via central differences (drops endpoints).
    interior = slice(1, ntemp - 1)
    dv = v_eq[2:] - v_eq[:-2]
    dt = temps[2:] - temps[:-2]
    alpha = np.full(ntemp, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        alpha[interior] = dv / dt / v_eq[interior]

    # Thermodynamic Gruneisen and C_p at constant pressure (per mole of cells).
    v_molar = v_eq * 1.0e-30 * AVOGADRO  # m^3/mol
    b_pa = bulk * 1.0e9  # Pa
    gamma = np.full(ntemp, np.nan)
    cp = np.full(ntemp, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        gamma[interior] = alpha[interior] * b_pa[interior] * v_molar[interior] / cv_eq[interior]
        cp[interior] = (
            cv_eq[interior]
            + alpha[interior] ** 2 * b_pa[interior] * v_molar[interior] * temps[interior]
        )

    # Report only the temperatures with a well-defined derivative.
    valid = np.isfinite(alpha)
    result = {
        'temperatures': temps[valid],
        'volume': v_eq[valid],
        'thermal_expansion': alpha[valid],
        'bulk_modulus': bulk[valid],
        'gibbs': g_eq[valid],
        'heat_capacity': cp[valid],
        'gruneisen': gamma[valid],
        'B0': float(bulk[valid][0]) if np.any(valid) else float('nan'),
    }
    return result


def _write_qha_outputs(data_controller, data, result, fname):
    """Write the QHA tables (``<fname>_*.dat``) using the shared row writer."""
    _write_rows(
        data_controller,
        fname + '_ev.dat',
        'volume(Ang^3)  static_energy(eV)',
        list(zip(data['volumes'], data['energies'])),
    )

    tables = [
        ('_volume.dat', 'T(K)  volume(Ang^3)', 'volume'),
        ('_thermal_expansion.dat', 'T(K)  thermal_expansion(1/K)', 'thermal_expansion'),
        ('_bulk_modulus.dat', 'T(K)  bulk_modulus(GPa)', 'bulk_modulus'),
        ('_gibbs.dat', 'T(K)  gibbs_free_energy(eV)', 'gibbs'),
        ('_heat_capacity.dat', 'T(K)  Cp(J/K/mol)', 'heat_capacity'),
        ('_gruneisen.dat', 'T(K)  gruneisen', 'gruneisen'),
    ]
    temps = result['temperatures']
    for suffix, header, key in tables:
        _write_rows(
            data_controller,
            fname + suffix,
            header,
            list(zip(temps, result[key])),
        )


def compute_gruneisen_band(
    data_controller,
    phonons,
    q_path=None,
    q_labels=None,
    npoints=101,
    units='THz',
    cutoff_frequency=None,
    fname='qha',
):
    """Mode Grueneisen parameters along a q-path (dispersion-style output).

    Uses three bracketing volumes (V0-, V0, V0+) centred on the middle of the
    volume scan to finite-difference ``gamma_qv = -(V/omega) domega/dV`` via
    :class:`phonopy.PhonopyGruneisen`.  Writes ``<fname>_gruneisen_band.dat``
    (distance, then the per-branch Grueneisen parameter and frequency) and, when
    tick labels are available, ``<fname>_gruneisen_band.labels``.

    The mode Grueneisen parameter carries a ``1/omega**2`` factor, so the
    acoustic branches diverge as ``omega -> 0`` at Gamma and produce spurious
    spikes there.  Modes whose frequency is below ``cutoff_frequency`` (in the
    output ``units``) are therefore masked (written as ``nan``, so a plot simply
    leaves a gap).  ``cutoff_frequency=None`` uses 1% of the maximum frequency
    along the path; pass ``0`` to disable the masking.

    Requires at least three volumes with force constants already built.
    """
    from phonopy import PhonopyGruneisen
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    from .do_phonopy import THZ_TO_CM1, _band_tick_positions, default_q_path

    if phonons is None or len(phonons) < 3:
        return None

    # Central triplet: smallest symmetric volume bracket around equilibrium.
    c = len(phonons) // 2
    center, minus, plus = phonons[c], phonons[c - 1], phonons[c + 1]

    gru = PhonopyGruneisen(center, plus, minus)

    scale = THZ_TO_CM1 if str(units).lower() in ('cm-1', 'cm^-1', 'cm') else 1.0

    # Resolve the q-path exactly like the phonon dispersion (do_phonopy
    # .compute_phonon_bands): an explicit path wins, otherwise the ibrav-based
    # high-symmetry path, otherwise the automatic seekpath path.
    if q_path is None:
        auto_path, auto_labels = default_q_path(data_controller)
        if auto_path is not None:
            q_path, q_labels = auto_path, auto_labels

    flat_labels = None
    if q_path is not None:
        qpoints, connections = get_band_qpoints_and_path_connections(q_path, npoints=npoints)
        if q_labels is not None:
            flat_labels = list(q_labels)
    else:
        try:
            from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

            qpoints, label_pairs, connections = get_band_qpoints_by_seekpath(
                center.primitive, npoints, is_const_interval=True
            )
            flat_labels = []
            for pair in label_pairs:
                if not flat_labels:
                    flat_labels.append(pair[0])
                flat_labels.append(pair[1])
        except ModuleNotFoundError:
            if getattr(data_controller, 'rank', 0) == 0:
                print(
                    'Grueneisen dispersion skipped: no ibrav-based q-path is '
                    'available and the automatic path needs the optional '
                    "'seekpath' package.  Install seekpath or pass ibrav / "
                    'q_path to quasi_harmonic().'
                )
            return None

    gru.set_band_structure(qpoints)
    _q, distances, frequencies, _ev, gruneisen = gru.get_band_structure()

    # Frequency cutoff (output units) below which the acoustic 1/omega**2
    # divergence is masked out near Gamma.
    all_freqs = np.abs(np.concatenate([np.asarray(f).ravel() for f in frequencies])) * scale
    fmax = float(np.nanmax(all_freqs)) if all_freqs.size else 0.0
    cutoff = 0.01 * fmax if cutoff_frequency is None else float(cutoff_frequency)

    rows = []
    for seg_d, seg_f, seg_g in zip(distances, frequencies, gruneisen):
        for d, fq, gv in zip(seg_d, seg_f, seg_g):
            fq = np.asarray(fq) * scale
            gv = np.asarray(gv, dtype=float).copy()
            if cutoff > 0.0:
                gv[np.abs(fq) < cutoff] = np.nan
            row = [d]
            for g_branch, f_branch in zip(gv, fq):
                row.append(g_branch)
                row.append(f_branch)
            rows.append(row)

    band_path = _write_rows(
        data_controller,
        fname + '_gruneisen_band.dat',
        'distance  [gruneisen  frequency(%s)] per branch' % units,
        rows,
    )

    if flat_labels is not None and getattr(data_controller, 'rank', 0) == 0:
        tick_distances = _band_tick_positions(distances, connections)
        _, attr = data_controller.data_dicts()
        lpath = os.path.join(attr.get('opath', '.'), fname + '_gruneisen_band.labels')
        with open(lpath, 'w') as f:
            f.write('# tick_distance  label\n')
            for d, lab in zip(tick_distances, flat_labels):
                f.write('% 16.8e  %s\n' % (d, lab))

    return band_path


def run_qha(
    data_controller,
    nvolumes=5,
    strain=0.02,
    qha_dir='qha',
    mesh=None,
    t_min=0.0,
    t_max=1000.0,
    t_step=10.0,
    eos='vinet',
    pressure=0.0,
    ibrav=None,
    q_path=None,
    q_labels=None,
    q_npoints=101,
    gruneisen_band=True,
    gruneisen_cutoff=None,
    units='THz',
    fname='qha',
):
    """Harvest the volume scan and compute the QHA quantities (analysis phase)."""
    arry, attr = data_controller.data_dicts()

    if mesh is None:
        mesh = attr.get('phonon_q_mesh', None) or [20, 20, 20]
    if ibrav is not None:
        attr['ibrav'] = ibrav

    data = _collect_volume_data(
        data_controller, nvolumes, strain, qha_dir, mesh, t_min, t_max, t_step
    )

    if int(nvolumes) >= 4:
        qha, result = _qha_via_phonopy(data, eos, pressure, t_max)
        arry['qha'] = qha
    else:
        result = _qha_via_quadratic(data, pressure)
        arry['qha'] = None

    arry['qha_data'] = data
    arry['qha_result'] = result

    _write_qha_outputs(data_controller, data, result, fname)

    if gruneisen_band:
        compute_gruneisen_band(
            data_controller,
            data.get('phonons'),
            q_path=q_path,
            q_labels=q_labels,
            npoints=q_npoints,
            units=units,
            cutoff_frequency=gruneisen_cutoff,
            fname=fname,
        )

    return result
