"""Electron-phonon runs on a model crystal, one function per property, with the files of PAOFLOW.

These are the steps of the ``main.py`` that ``paoflow-gen`` (workflow ``ep_model``) and the Setup
tab of ``paoflow-studio`` write.  Each function computes one property of an
:class:`~PAOFLOW.models.electron_phonon.model.ElectronPhononModel` with
:mod:`PAOFLOW.models.electron_phonon.properties` and writes it in the format of the ab initio
electron-phonon workflows (:mod:`PAOFLOW.elphon.workflows`), so the Results tab and
:mod:`PAOFLOW.results` read them:

=====================================  ==========================================================
function                               files in ``outdir``
=====================================  ==========================================================
:func:`band_structure`                 ``bands_0.dat``, ``kpath_points.txt``
:func:`phonon_dispersion`              ``<model>_band.dat``, ``<model>_band.labels``, ``<model>_dos.dat``
:func:`superconductivity`              ``alpha2F.dat``, ``alpha2F_tr.dat``, ``eliashberg.npz``,
                                       ``lambda_summary.dat``
:func:`migdal_eliashberg`              ``migdal_eliashberg/``: ``gap_vs_T.dat``, ``max_eigenvalue.dat``, EPW-format files
:func:`resistivity`                    ``resistivity.dat`` and the Drude emissivity files
:func:`self_energy`                    ``self_energy_<T>K.dat``
:func:`mobility`                       ``mobility.dat``, ``rates_<T>K.dat``
:func:`phonon_assisted_absorption`     ``epsilon2_indabs_<T>K.dat``, ``alpha_<T>K.dat``
=====================================  ==========================================================

The models are small: every step runs serially on one core, in seconds to a minute on the
default grids.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .model import BOHR_M, E_CHARGE, ElectronPhononModel
from . import properties

#: Points per periodic direction of the grids that locate the Fermi level and the band edges.
EDGE_GRID = {1: 400, 2: 96, 3: 32}


# ------------------------------------------------------------------------------------- the path
def path_points(
    model: ElectronPhononModel,
    band_path: str,
    high_symmetry_points: dict[str, Sequence[float]],
    nk: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[tuple[float, str]]]:
    """Points of a band path, their distance along it and the labelled ticks.

    Parameters
    ----------
    model : ElectronPhononModel
    band_path : str
        Labels joined by ``-``, segments split by ``|`` (``'gG-X-M-gG|R-X'``; ``gG`` is Gamma).
    high_symmetry_points : dict
        Crystal coordinates of every label.
    nk : int, optional
        Points of the whole path, shared by the lines in proportion to their length.

    Returns
    -------
    k_crystal : ndarray, shape ``(npts, 3)``
    distance : ndarray, shape ``(npts,)``
        Path length (2 pi / alat), continuous across a ``|`` break.
    ticks : list of (float, str)
        Position and label of every high-symmetry point (``Γ`` for ``gG``; ``X|M`` at a break).

    Raises
    ------
    ValueError
        If a label has no coordinates or a segment has a single point.
    """
    segments = [segment.split('-') for segment in band_path.split('|') if segment]
    for segment in segments:
        if len(segment) < 2:
            raise ValueError('every segment of the band path needs two points: %r' % band_path)
        for label in segment:
            if label not in high_symmetry_points:
                raise ValueError('no coordinates for the point %r of the band path' % label)
    scale = model.alat / (2 * np.pi)
    lines = [
        (np.asarray(high_symmetry_points[a], float), np.asarray(high_symmetry_points[b], float))
        for segment in segments
        for a, b in zip(segment[:-1], segment[1:])
    ]
    lengths = [np.linalg.norm(model.to_cartesian(end - start)) * scale for start, end in lines]
    total = sum(lengths) or 1.0
    kpoints, distances, ticks = [], [], []
    position = 0.0
    line = 0
    for segment in segments:
        pretty = ['Γ' if label in ('gG', 'G', 'Gamma') else label for label in segment]
        if ticks and abs(ticks[-1][0] - position) < 1e-12:
            ticks[-1] = (position, ticks[-1][1] + '|' + pretty[0])
        else:
            ticks.append((position, pretty[0]))
        for index in range(len(segment) - 1):
            start, end = lines[line]
            count = max(2, int(round(nk * lengths[line] / total)))
            last = index == len(segment) - 2
            fractions = np.linspace(0.0, 1.0, count + 1 if last else count, endpoint=last)
            kpoints.append(start + fractions[:, None] * (end - start))
            distances.append(position + fractions * lengths[line])
            position += lengths[line]
            ticks.append((position, pretty[index + 1]))
            line += 1
    return np.concatenate(kpoints), np.concatenate(distances), ticks


def _write_ticks(path: str, ticks: list[tuple[float, str]]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        for position, label in ticks:
            handle.write('%.8f %s\n' % (position, label))


def energy_reference(model: ElectronPhononModel, nk: int | None = None) -> dict:
    """Fermi level of a metal, band edges of an insulator (the zero of the outputs).

    Parameters
    ----------
    model : ElectronPhononModel
    nk : int, optional
        Points per periodic direction of the grid (default :data:`EDGE_GRID`).

    Returns
    -------
    dict
        ``metal`` (bool), ``zero`` (the Fermi level of a metal, the valence-band maximum of an
        insulator; eV), ``vbm``, ``cbm`` and ``gap`` (eV; ``None`` for a metal).
    """
    states = properties.electron_states(model, nk or EDGE_GRID[model.dimension])
    filled = model.nelec / 2.0
    if abs(filled - round(filled)) < 1e-9 and 0 < round(filled) < model.number_of_orbitals:
        vbm, cbm = properties.band_edges(states, model.nelec)
        if cbm > vbm + 1e-6:
            return {'metal': False, 'zero': vbm, 'vbm': vbm, 'cbm': cbm, 'gap': cbm - vbm}
    fermi = properties.fermi_level(states.energies, model.nelec, 0.01)
    return {'metal': True, 'zero': fermi, 'vbm': None, 'cbm': None, 'gap': None}


# ------------------------------------------------------------------------------- bands, phonons
def band_structure(
    model: ElectronPhononModel,
    outdir: str,
    *,
    band_path: str,
    high_symmetry_points: dict[str, Sequence[float]],
    nk: int = 200,
) -> dict:
    """Electron bands along a path, from the Fermi level (metal) or the valence top (insulator).

    Parameters
    ----------
    model : ElectronPhononModel
    outdir : str
        Output directory (created).
    band_path : str
        :func:`path_points`.
    high_symmetry_points : dict
        Crystal coordinates of the labels.
    nk : int, optional
        Points of the path.

    Returns
    -------
    dict
        ``distance``, ``energies`` (from the zero), ``ticks`` and :func:`energy_reference`.
        Writes ``bands_0.dat`` (distance in 2 pi/alat, then one column per band) and
        ``kpath_points.txt``.
    """
    kpoints, distance, ticks = path_points(model, band_path, high_symmetry_points, nk)
    reference = energy_reference(model)
    energies = np.linalg.eigvalsh(model.hamiltonian(kpoints)) - reference['zero']
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(
        os.path.join(outdir, 'bands_0.dat'),
        np.column_stack([distance, energies]),
        fmt='%14.8f',
        header='%s: path distance (2pi/alat), band energies (eV) from the %s'
        % (model.name, 'Fermi level' if reference['metal'] else 'valence-band maximum'),
    )
    _write_ticks(os.path.join(outdir, 'kpath_points.txt'), ticks)
    if reference['metal']:
        print(
            '%s: metal, Fermi level %.4f eV (the zero of the bands)'
            % (model.name, reference['zero'])
        )
    else:
        print(
            '%s: insulator, gap %.4f eV (zero at the valence top)' % (model.name, reference['gap'])
        )
    print('  wrote %s' % os.path.join(outdir, 'bands_0.dat'))
    return {'distance': distance, 'energies': energies, 'ticks': ticks, **reference}


def phonon_dispersion(
    model: ElectronPhononModel,
    outdir: str,
    *,
    band_path: str,
    high_symmetry_points: dict[str, Sequence[float]],
    nk: int = 200,
    dos_grid: int | None = None,
    dos_smearing_mev: float = 0.5,
) -> dict:
    """Phonon dispersion along a path and phonon density of states.

    Parameters
    ----------
    model : ElectronPhononModel
    outdir : str
        Output directory (created).
    band_path : str
        :func:`path_points`.
    high_symmetry_points : dict
        Crystal coordinates of the labels.
    nk : int, optional
        Points of the path.
    dos_grid : int, optional
        q-points per periodic direction of the DOS (default :data:`EDGE_GRID`).
    dos_smearing_mev : float, optional
        Gaussian width of the DOS (meV).

    Returns
    -------
    dict
        ``distance``, ``energies_mev`` (npts, nmode), ``dos_energies_mev``, ``dos`` (states/meV
        per cell, integral ``3 natom``), ``unstable`` (an imaginary mode on the path or the grid).
        Writes ``<model>_band.dat``, ``<model>_band.labels`` and ``<model>_dos.dat``.
    """
    qpoints, distance, ticks = path_points(model, band_path, high_symmetry_points, nk)
    energies = 1e3 * model.phonons(qpoints)[0]
    grid = properties.crystal_grid(model, dos_grid or EDGE_GRID[model.dimension])
    grid_energies = 1e3 * model.phonons(grid)[0]
    top = max(float(grid_energies.max()), float(energies.max()), 1e-3)
    axis = np.linspace(0.0, 1.1 * top, 600)
    dos = properties.gaussian_delta(
        axis[:, None] - grid_energies.ravel()[None, :], dos_smearing_mev
    ).sum(1) / len(grid)
    unstable = bool(min(energies.min(), grid_energies.min()) < -1e-3 * top)
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, model.name)
    np.savetxt(
        stem + '_band.dat',
        np.column_stack([distance, energies]),
        fmt='%14.8f',
        header='%s: q-path distance (2pi/alat), phonon energies (meV)' % model.name,
    )
    _write_ticks(stem + '_band.labels', ticks)
    np.savetxt(
        stem + '_dos.dat',
        np.column_stack([axis, dos]),
        fmt='%14.8f',
        header='%s: phonon energy (meV), DOS (states/meV per cell)' % model.name,
    )
    print(
        '%s: phonons up to %.2f meV%s'
        % (model.name, top, '; UNSTABLE: imaginary modes (negative energies)' if unstable else '')
    )
    for suffix in ('_band.dat', '_band.labels', '_dos.dat'):
        print('  wrote %s' % (stem + suffix))
    return {
        'distance': distance,
        'energies_mev': energies,
        'ticks': ticks,
        'dos_energies_mev': axis,
        'dos': dos,
        'unstable': unstable,
    }


# ------------------------------------------------------------------------------------- metals
def superconductivity(
    model: ElectronPhononModel,
    outdir: str,
    *,
    nk: int = 40,
    nq: int = 8,
    sigma_ev: float = 0.02,
    adaptive_smearing: float | None = 0.7,
    mu_star: float = 0.1,
    transport: bool = True,
    sigma_w_frac: float = 0.03,
    nomega: int = 600,
) -> dict:
    """Mode-resolved coupling, Eliashberg function, :math:`\\lambda`, :math:`\\omega_{\\log}`, :math:`T_c`.

    Parameters
    ----------
    model : ElectronPhononModel
        A metal.
    outdir : str
        Output directory (created).
    nk : int, optional
        k-points per periodic direction of the double delta.
    nq : int, optional
        q-points per periodic direction (irreducible wedge in 3D); must divide ``nk``.
    sigma_ev : float, optional
        Gaussian width, the floor of the adaptive widths (eV).
    adaptive_smearing : float, optional
        Prefactor of the adaptive widths (0.7); ``None`` or 0: the fixed width.
    mu_star : float, optional
        Coulomb pseudopotential of the McMillan and Allen-Dynes :math:`T_c`.
    transport : bool, optional
        Also :math:`\\alpha^2F_{tr}` and the plasma energy (for :func:`resistivity`).
    sigma_w_frac : float, optional
        Gaussian width of the modes in :math:`\\alpha^2F`, fraction of the highest phonon energy.
    nomega : int, optional
        Frequency points of :math:`\\alpha^2F`.

    Returns
    -------
    dict
        :func:`~PAOFLOW.models.electron_phonon.properties.eliashberg_function` plus the modes
        (``lambda_qv``, ``omega_qv_thz``, ``q_points``, ``q_weights``), ``N_F``, ``E_F`` and, with
        ``transport``, ``a2F_tr``, ``lambda_tr`` and ``omega_plasma_ev``.  Writes ``alpha2F.dat``,
        ``alpha2F_tr.dat``, ``eliashberg.npz`` (the input of :func:`migdal_eliashberg`,
        :func:`resistivity` and :func:`self_energy`) and ``lambda_summary.dat``.

    Raises
    ------
    ValueError
        If ``nq`` does not divide ``nk``.
    """
    from ...elphon.eph_kq import THZ_TO_EV
    from ...elphon.workflows.superconductivity import smearing_label, write_eliashberg

    if nk % nq:
        raise ValueError('nq = %d must divide nk = %d (k + q on the k-grid)' % (nq, nk))
    adaptive = adaptive_smearing if adaptive_smearing else None
    q_points, q_weights = properties.irreducible_q_points(model, nq)
    coupling = properties.mode_resolved_coupling(
        model, nk, q_points, sigma=sigma_ev, adaptive=adaptive, transport=transport
    )
    result = properties.eliashberg_function(
        coupling['lambda_qv'],
        coupling['omega_ev'],
        q_weights=q_weights,
        mu_star=mu_star,
        sigma_w_frac=sigma_w_frac,
        nomega=nomega,
    )
    out = {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float, np.ndarray, np.floating))
    }
    out.update(
        lambda_qv=coupling['lambda_qv'],
        omega_qv_thz=np.maximum(coupling['omega_ev'], 0.0) / THZ_TO_EV,
        q_points=q_points,
        q_weights=q_weights,
        N_F=coupling['N_F'],
        E_F=coupling['E_F'],
        nk=nk,
        nq=nq,
        sigma_ev=sigma_ev,
        adaptive_smearing=adaptive or 0.0,
        sigma_w_frac=sigma_w_frac,
        mu_star=mu_star,
        model_name=model.name,
    )
    if transport:
        transport_result = properties.eliashberg_function(
            coupling['lambda_tr_qv'],
            coupling['omega_ev'],
            q_weights=q_weights,
            mu_star=mu_star,
            sigma_w_frac=sigma_w_frac,
            nomega=nomega,
        )
        out.update(
            a2F_tr=transport_result['a2F'],
            lambda_tr=transport_result['lambda'],
            lambda_tr_qv=coupling['lambda_tr_qv'],
            omega_plasma_ev=properties.plasma_energy(
                coupling['states'], coupling['E_F'], 300.0, model.cell_volume
            ),
        )
    grids = '%s, k %d^%d, q %d^%d (%d irreducible)' % (
        model.name,
        nk,
        model.dimension,
        nq,
        model.dimension,
        len(q_points),
    )
    write_eliashberg(
        out, outdir, mu_star=mu_star, label=grids + ', ' + smearing_label(sigma_ev, adaptive)
    )
    lines = [
        '%s electron-phonon coupling (PAOFLOW model crystal)' % model.name,
        'grids                 %s' % grids,
        'smearing              %s' % smearing_label(sigma_ev, adaptive),
        'E_F                   %.6f eV' % out['E_F'],
        'N_F (per spin, cell)  %.6f /eV' % out['N_F'],
        'lambda                %.6f' % out['lambda'],
        'omega_log             %.4f meV' % (1e3 * out['omega_log']),
        'Tc McMillan           %.4f K   (mu* = %.2f)' % (out['Tc_mcmillan'], mu_star),
        'Tc Allen-Dynes        %.4f K' % out['Tc_allen_dynes'],
    ]
    if transport:
        lines += [
            'lambda_tr             %.6f' % out['lambda_tr'],
            'hbar omega_p          %.6f eV' % out['omega_plasma_ev'],
        ]
    with open(os.path.join(outdir, 'lambda_summary.dat'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return out


def _eliashberg_npz(outdir: str) -> str:
    path = os.path.join(outdir, 'eliashberg.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError('%s not found: run superconductivity() first' % path)
    return path


def migdal_eliashberg(
    outdir: str,
    *,
    temperatures: Sequence[float] | None = None,
    mu_star: float = 0.1,
    wscut_ev: float | None = None,
    pade: bool = True,
    acon: bool = False,
) -> dict:
    """Isotropic Migdal-Eliashberg equations on the Eliashberg function of :func:`superconductivity`.

    Parameters
    ----------
    outdir : str
        Folder of ``eliashberg.npz``; the results go to ``outdir/migdal_eliashberg``.
    temperatures : sequence of float, optional
        Temperatures (K); by default from 1 K to 1.6 times the Allen-Dynes :math:`T_c`.
    mu_star : float, optional
        Coulomb pseudopotential.
    wscut_ev : float, optional
        Matsubara cutoff (eV); by default ten times the highest phonon energy (at least 0.1 eV).
    pade : bool, optional
        Real-axis gap and quasiparticle DOS by Padé approximants.
    acon : bool, optional
        Also by the iterative analytic continuation (slower).

    Returns
    -------
    dict
        :func:`PAOFLOW.elphon.workflows.superconductivity.migdal_eliashberg`: ``result``
        (gap and :math:`Z` per temperature, ``Tc_gap``), ``linear`` (largest eigenvalue of the
        linearised kernel, ``Tc_linear``) and ``lambda``.

    Notes
    -----
    The Eliashberg function is rebuilt from the modes with the width of :func:`superconductivity`
    (``sigma_w_frac`` of the highest phonon energy).
    """
    from ...elphon.eph_kq import THZ_TO_EV
    from ...elphon.workflows.superconductivity import migdal_eliashberg as solve

    source = _eliashberg_npz(outdir)
    with np.load(source) as data:
        allen_dynes = float(data['Tc_allen_dynes'])
        omega_max_ev = float(data['omega_qv_thz'].max()) * THZ_TO_EV
        width_fraction = float(data['sigma_w_frac']) if 'sigma_w_frac' in data else 0.03
        name = str(data['model_name']) if 'model_name' in data else 'model'
    tc_scale = max(allen_dynes, 1.0)
    if temperatures is None:
        step = 0.25 if tc_scale < 8 else (0.5 if tc_scale < 25 else 1.0)
        temperatures = np.arange(1.0, 1.6 * tc_scale + step, step)
    wscut = wscut_ev if wscut_ev else max(0.1, 10.0 * omega_max_ev)
    width_mev = 1e3 * width_fraction * omega_max_ev
    return solve(
        source,
        os.path.join(outdir, 'migdal_eliashberg'),
        name,
        temps=list(np.asarray(temperatures, dtype=float)),
        mu_star=mu_star,
        wscut=wscut,
        degaussq_mev=width_mev,
        degaussq_linear_mev=width_mev,
        pade=pade,
        acon=acon,
        linear_temps=(max(0.25, 0.5 * tc_scale), 2.0 * tc_scale + 1.0, 31),
    )


def resistivity(
    outdir: str,
    *,
    temperatures: Sequence[float] = (10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    residual_uohm_cm: float = 0.0,
) -> dict:
    """Resistivity :math:`\\rho(T)` (and the Drude emissivity) from :math:`\\alpha^2F_{tr}`.

    Parameters
    ----------
    outdir : str
        Folder of ``eliashberg.npz`` (written with ``transport=True``) and of the outputs.
    temperatures : sequence of float, optional
        Temperatures (K).
    residual_uohm_cm : float, optional
        Residual resistivity (micro-ohm cm) of a Matthiessen channel.

    Returns
    -------
    dict
        :func:`PAOFLOW.elphon.workflows.transport.metal_resistivity` (``rows``: T, rho in
        micro-ohm cm; ``lambda_tr``, ``plasma_ev``, ``emissivity``).

    Notes
    -----
    .. math::

        \\rho(T) = \\frac{1}{\\varepsilon_0\\omega_p^2\\tau_{tr}(T)},\\qquad
        \\frac{\\hbar}{\\tau_{tr}} = 2\\pi\\int_0^\\infty d\\omega\\,
        \\frac{\\hbar\\omega/k_BT}{\\sinh^2(\\hbar\\omega/2k_BT)}\\frac{\\alpha^2F_{tr}(\\omega)}{2},

    linear in :math:`T` above the Debye scale and :math:`\\propto T^5` (Bloch-Grüneisen) far
    below it, provided the q-grid resolves the long-wavelength phonons.
    """
    from ...elphon.workflows.transport import metal_resistivity

    return metal_resistivity(
        _eliashberg_npz(outdir),
        outdir,
        temps_k=list(temperatures),
        rho0_uohm_cm=residual_uohm_cm,
    )


def self_energy(
    outdir: str,
    *,
    temperatures: Sequence[float] = (0.0, 100.0),
    energy_window_ev: float | None = None,
    n_energies: int = 961,
) -> dict:
    """Electron self-energy of the Eliashberg function (Engelsberg-Schrieffer) and the mass enhancement.

    Parameters
    ----------
    outdir : str
        Folder of ``eliashberg.npz`` and of the outputs.
    temperatures : sequence of float, optional
        Temperatures (K).
    energy_window_ev : float, optional
        Energies from ``-window`` to ``window`` around the Fermi level (eV); by default five times
        the highest phonon energy.
    n_energies : int, optional
        Energy points.

    Returns
    -------
    dict
        ``energies_ev``, ``self_energy`` (ntemp, nenergy, complex, eV), ``mass_enhancement``
        (:math:`-\\partial\\,\\mathrm{Re}\\Sigma/\\partial\\omega` at the Fermi level, per
        temperature; equal to :math:`\\lambda` at 0 K).  Writes ``self_energy_<T>K.dat``.
    """
    from ...elphon.self_energy import eliashberg_self_energy, mass_enhancement

    with np.load(_eliashberg_npz(outdir)) as data:
        omega, a2f = np.asarray(data['omega']), np.asarray(data['a2F'])
        coupling = float(data['lambda'])
    window = energy_window_ev or 5.0 * float(omega[a2f > 1e-6 * a2f.max()].max())
    energies = np.linspace(-window, window, int(n_energies))
    values, enhancement = [], []
    print('Eliashberg self-energy (lambda = %.4f):' % coupling)
    for temperature in temperatures:
        sigma = eliashberg_self_energy(omega, a2f, energies, float(temperature))
        values.append(sigma)
        enhancement.append(mass_enhancement(energies, sigma)[0])
        path = os.path.join(outdir, 'self_energy_%.1fK.dat' % float(temperature))
        np.savetxt(
            path,
            np.column_stack([energies, sigma.real, sigma.imag]),
            fmt='%16.9f',
            header='omega - E_F (eV), Re Sigma, Im Sigma (eV); T = %.1f K, mass enhancement %.5f'
            % (float(temperature), enhancement[-1]),
        )
        print(
            '  T = %6.1f K: mass enhancement -dRe Sigma/dw = %.4f   wrote %s'
            % (float(temperature), enhancement[-1], path)
        )
    return {
        'energies_ev': energies,
        'self_energy': np.array(values),
        'mass_enhancement': np.array(enhancement),
        'lambda': coupling,
    }


# ----------------------------------------------------------------------------- semiconductors
def _carrier_densities(
    states: properties.ElectronStates,
    filled: int,
    potential: float,
    temperature: float,
    volume_cm3: float,
) -> tuple[float, float]:
    """Electrons in the empty bands and holes in the full ones (cm^-3, spin-degenerate bands)."""
    occupation, vacancy = properties.fermi_occupations(states.energies, potential, temperature)
    scale = 2.0 / (len(states.kpoints) * volume_cm3)
    return float(occupation[:, filled:].sum() * scale), float(vacancy[:, :filled].sum() * scale)


def _neutral_potential(
    states: properties.ElectronStates,
    filled: int,
    temperature: float,
    volume_cm3: float,
    doping_cm3: float,
) -> float:
    """Chemical potential with ``n_e - n_h = doping`` (bisection)."""
    low, high = states.energies.min() - 1.0, states.energies.max() + 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        electrons, holes = _carrier_densities(states, filled, middle, temperature, volume_cm3)
        low, high = (middle, high) if electrons - holes < doping_cm3 else (low, middle)
    return middle


def _seebeck(
    states: properties.ElectronStates,
    rates: NDArray[np.float64],
    potential: float,
    temperature: float,
) -> float:
    """Seebeck coefficient (uV/K) along x: :math:`S = -\\langle\\varepsilon - \\mu\\rangle_\\sigma/eT`."""
    occupation, vacancy = properties.fermi_occupations(states.energies, potential, temperature)
    lifetime = np.where(rates > 0, 1.0 / np.maximum(rates, 1e-300), 0.0)
    weight = states.velocities[..., 0] ** 2 * lifetime * occupation * vacancy
    if weight.sum() <= 0:
        return float('nan')
    return float(-1e6 * np.sum(weight * (states.energies - potential)) / weight.sum() / temperature)


def mobility(
    model: ElectronPhononModel,
    outdir: str,
    *,
    nk: int | None = None,
    temperatures: Sequence[float] = (100.0, 150.0, 200.0, 300.0, 400.0, 500.0),
    window_ev: float = 0.15,
    final_window_ev: float = 0.25,
    smearing_ev: float = 0.005,
    adaptive_smearing: float | None = 0.7,
    doping_cm3: float = 0.0,
    carriers: str = 'electrons',
) -> dict:
    """Phonon-limited electron and hole mobilities (SERTA and MRTA) of a semiconductor.

    Parameters
    ----------
    model : ElectronPhononModel
        An insulator (an even filling with a gap).
    outdir : str
        Output directory (created).
    nk : int, optional
        k-points (and q-points) per periodic direction; by default 1200, 120 and 40 in 1D, 2D, 3D.
    temperatures : sequence of float, optional
        Temperatures (K).
    window_ev : float, optional
        Carriers within this energy of their band edge get a rate (the conducting states).
    final_window_ev : float, optional
        Final states of the scattering within this energy of the band edge.
    smearing_ev : float, optional
        Width of the energy delta, or the floor of the adaptive widths (eV).
    adaptive_smearing : float, optional
        Prefactor of the adaptive widths (0.7); ``None`` or 0: the fixed width.
    doping_cm3 : float, optional
        Net doping ``n_e - n_h`` (cm^-3) that sets the chemical potential; 0: intrinsic.
    carriers : {'electrons', 'holes', 'both'}, optional
        Carriers whose rates are computed.  The heavy p holes of the 3D model fill ten times
        more of the window than the electrons and take about a minute and a half at 40^3 (the
        electrons a second).

    Returns
    -------
    dict
        ``temps_k``, ``fermi_levels_ev`` (from the valence top), ``n_e_cm3``, ``n_h_cm3``,
        ``mobility_e_serta``, ``mobility_h_serta``, ``mobility_e_mrta``, ``mobility_h_mrta``
        (cm^2/Vs, NaN for the carriers left out), ``sigma_serta``, ``sigma_mrta`` (S/m, from the
        carriers computed), ``seebeck_mrta`` (uV/K), ``gap_ev``.  Writes ``mobility.dat`` (the
        columns of the ab initio ``mobility.dat``) and ``rates_<T>K.dat``.

    Raises
    ------
    ValueError
        If the model is a metal or ``carriers`` is unknown.

    Notes
    -----
    Along x, with the formula of
    :func:`~PAOFLOW.models.electron_phonon.properties.carrier_mobility`.  Carrier densities use
    the cell volume, including the long vectors of the non-periodic directions of 1D and 2D
    models; the mobility does not depend on it.  With only acoustic phonons and one valley the
    3D mobility goes as :math:`T^{-3/2}` (Bardeen-Shockley).
    """
    if carriers not in ('electrons', 'holes', 'both'):
        raise ValueError("carriers must be 'electrons', 'holes' or 'both'")
    adaptive = adaptive_smearing if adaptive_smearing else None
    grid = nk or {1: 1200, 2: 120, 3: 40}[model.dimension]
    states = properties.electron_states(model, grid)
    filled = int(round(model.nelec / 2.0))
    vbm, cbm = properties.band_edges(states, model.nelec)
    if cbm <= vbm:
        raise ValueError('%s is a metal: the mobility needs a gap' % model.name)
    temperatures = np.asarray(temperatures, dtype=float)
    volume_cm3 = model.cell_volume * (100.0 * BOHR_M) ** 3
    conduction = list(range(filled, model.number_of_orbitals))
    valence = list(range(filled))
    potentials = np.array(
        [_neutral_potential(states, filled, t, volume_cm3, doping_cm3) for t in temperatures]
    )
    densities = np.array(
        [
            _carrier_densities(states, filled, p, t, volume_cm3)
            for p, t in zip(potentials, temperatures)
        ]
    )
    lowest_conduction = states.energies[:, filled]
    highest_valence = states.energies[:, filled - 1]
    windows = {
        'electrons': (
            np.nonzero(lowest_conduction < cbm + window_ev)[0],
            np.nonzero(lowest_conduction < cbm + final_window_ev)[0],
            conduction,
        ),
        'holes': (
            np.nonzero(highest_valence > vbm - window_ev)[0],
            np.nonzero(highest_valence > vbm - final_window_ev)[0],
            valence,
        ),
    }
    chosen = ('electrons', 'holes') if carriers == 'both' else (carriers,)
    serta = np.zeros((len(temperatures), len(states.kpoints), model.number_of_orbitals))
    mrta = np.zeros_like(serta)
    for kind in chosen:
        initial, final, bands = windows[kind]
        kind_serta, kind_mrta = properties.carrier_scattering_rates(
            model, states, initial, final, temperatures, potentials,
            smearing=smearing_ev, adaptive=adaptive,
        )  # fmt: skip
        serta[:, :, bands] = kind_serta[:, :, bands]
        mrta[:, :, bands] = kind_mrta[:, :, bands]
    columns = {}
    for name, rates in (('serta', serta), ('mrta', mrta)):
        for kind, tag, column in (('electrons', 'e', 0), ('holes', 'h', 1)):
            columns['mobility_%s_%s' % (tag, name)] = (
                properties.carrier_mobility(
                    states, rates, windows[kind][2], temperatures, potentials, kind
                )
                if kind in chosen
                else np.full(len(temperatures), np.nan)
            )
        columns['sigma_' + name] = (
            100.0
            * E_CHARGE
            * (
                densities[:, 0] * np.nan_to_num(columns['mobility_e_' + name])
                + densities[:, 1] * np.nan_to_num(columns['mobility_h_' + name])
            )
        )
    seebeck = np.array(
        [_seebeck(states, mrta[it], potentials[it], t) for it, t in enumerate(temperatures)]
    )
    os.makedirs(outdir, exist_ok=True)
    table = np.column_stack(
        [
            temperatures,
            potentials - vbm,
            densities[:, 0],
            densities[:, 1],
            columns['mobility_e_serta'],
            columns['mobility_h_serta'],
            columns['mobility_e_mrta'],
            columns['mobility_h_mrta'],
            columns['sigma_serta'],
            columns['sigma_mrta'],
            seebeck,
        ]
    )
    np.savetxt(
        os.path.join(outdir, 'mobility.dat'),
        table,
        fmt='%14.6e',
        header='T (K), mu_F (eV, from the valence top), n_e, n_h (cm^-3), mobility e/h SERTA, '
        'e/h MRTA (cm^2/Vs), sigma SERTA, MRTA (S/m), Seebeck MRTA (uV/K); %s, k %d^%d, %s'
        % (
            model.name,
            grid,
            model.dimension,
            'electrons and holes' if carriers == 'both' else carriers,
        ),
    )
    written = [os.path.join(outdir, 'mobility.dat')]
    window_states = []
    for kind in chosen:
        k_indices, _final, bands = windows[kind]
        for band in bands:
            window_states += [(k, band) for k in k_indices]
    rows = np.array(window_states, dtype=int)
    for it, temperature in enumerate(temperatures):
        data = np.column_stack(
            [
                states.energies[rows[:, 0], rows[:, 1]] - vbm,
                serta[it][rows[:, 0], rows[:, 1]],
                mrta[it][rows[:, 0], rows[:, 1]],
            ]
        )
        data = data[data[:, 1] > 0]
        path = os.path.join(outdir, 'rates_%.1fK.dat' % temperature)
        np.savetxt(
            path,
            data[np.argsort(data[:, 0])],
            fmt='%14.6e',
            header='E (eV, from the valence top), 1/tau SERTA, 1/tau MRTA (1/s)',
        )
        written.append(path)
    print(
        '%s carrier transport (k %d^%d, gap %.3f eV, doping %.3e cm^-3):'
        % (model.name, grid, model.dimension, cbm - vbm, doping_cm3)
    )
    print('    T (K)   mobility e / h (cm^2/Vs):  SERTA                MRTA')
    for it, temperature in enumerate(temperatures):
        print(
            '  %7.1f  %22.1f / %-10.1f %10.1f / %-10.1f'
            % (
                temperature,
                columns['mobility_e_serta'][it],
                columns['mobility_h_serta'][it],
                columns['mobility_e_mrta'][it],
                columns['mobility_h_mrta'][it],
            )
        )
    for path in written:
        print('  wrote %s' % path)
    return {
        'temps_k': temperatures,
        'fermi_levels_ev': potentials - vbm,
        'n_e_cm3': densities[:, 0],
        'n_h_cm3': densities[:, 1],
        **columns,
        'seebeck_mrta': seebeck,
        'gap_ev': cbm - vbm,
    }


def phonon_assisted_absorption(
    model: ElectronPhononModel,
    outdir: str,
    *,
    nk: int | None = None,
    temperatures: Sequence[float] = (10.0, 300.0, 600.0),
    photon_energies: Sequence[float] | None = None,
    window_ev: float = 0.25,
    eta_ev: float = 0.02,
    degauss_ev: float = 0.015,
    refractive_index: float = 3.4,
) -> dict:
    """Phonon-assisted absorption across the indirect gap of a semiconductor.

    Parameters
    ----------
    model : ElectronPhononModel
        An insulator with an indirect gap.
    outdir : str
        Output directory (created).
    nk : int, optional
        k-points (and q-points) per periodic direction; by default 400, 60 and 28 in 1D, 2D, 3D.
    temperatures : sequence of float, optional
        Temperatures (K).
    photon_energies : sequence of float, optional
        Photon energies (eV); by default 161 points from 0.15 eV below to 0.25 eV above the gap.
    window_ev : float, optional
        Valence states within this energy of the top and conduction states within it of the
        bottom take part.
    eta_ev : float, optional
        Broadening of the intermediate states (eV).
    degauss_ev : float, optional
        Gaussian width of the energy delta (eV).
    refractive_index : float, optional
        Refractive index of the absorption coefficient.

    Returns
    -------
    dict
        ``photon_ev``, ``temps_k``, ``eps2`` (ntemp, 3, nomega), ``eps2_average`` (over the
        periodic axes), ``alpha_cm`` (ntemp, nomega), ``gap_ev`` and ``phonon_edge_ev`` (the
        phonon at the conduction minimum).  Writes ``epsilon2_indabs_<T>K.dat`` and
        ``alpha_<T>K.dat``.

    Raises
    ------
    ValueError
        If the model is a metal.
    """
    from ...elphon.phonon_assisted_absorption import absorption_coefficient

    grid = nk or {1: 400, 2: 60, 3: 28}[model.dimension]
    states = properties.electron_states(model, grid, velocity_matrices=True)
    filled = int(round(model.nelec / 2.0))
    vbm, cbm = properties.band_edges(states, model.nelec)
    if cbm <= vbm:
        raise ValueError('%s is a metal: no indirect gap' % model.name)
    top_valence = states.energies[:, :filled].max(1)
    bottom_conduction = states.energies[:, filled:].min(1)
    initial = np.nonzero(top_valence > vbm - window_ev)[0]
    final = np.nonzero(bottom_conduction < cbm + window_ev)[0]
    gap = cbm - vbm
    photon = (
        np.linspace(gap - 0.15, gap + 0.25, 161)
        if photon_energies is None
        else np.asarray(photon_energies, dtype=float)
    )
    temperatures = np.asarray(temperatures, dtype=float)
    epsilon = properties.indirect_absorption(
        model, states, initial, final, temperatures, photon,
        eta_ev=eta_ev, degauss_ev=degauss_ev, reference_ev=0.5 * (vbm + cbm),
    )  # fmt: skip
    periodic = [axis for axis in range(3) if model.periodic_axes[axis]]
    average = epsilon[:, periodic].mean(axis=1)
    alpha = absorption_coefficient(photon, average, refractive_index=refractive_index)
    edge_point = states.kpoints[np.argmin(bottom_conduction)]
    phonon_edge = float(np.max(model.phonons(edge_point[None])[0]))
    os.makedirs(outdir, exist_ok=True)
    written = []
    for it, temperature in enumerate(temperatures):
        tag = '%.1f' % temperature
        path = os.path.join(outdir, 'epsilon2_indabs_%sK.dat' % tag)
        np.savetxt(
            path,
            np.column_stack([photon, average[it]]),
            fmt='%15.6f %22.14E',
            header='Photon energy (eV), Im eps (phonon-assisted, average over the periodic axes) '
            'for eta (eV) = %.3f' % eta_ev,
        )
        written.append(path)
        path = os.path.join(outdir, 'alpha_%sK.dat' % tag)
        np.savetxt(
            path,
            np.column_stack([photon, alpha[it]]),
            fmt='%15.6f %22.14E',
            header='Photon energy (eV), absorption coefficient (cm^-1, indirect, n = %.2f) '
            'for eta (eV) = %.3f' % (refractive_index, eta_ev),
        )
        written.append(path)
    print(
        '%s phonon-assisted absorption (k %d^%d): indirect gap %.4f eV, phonons at the conduction '
        'minimum up to %.2f meV: onsets near %.4f and %.4f eV'
        % (
            model.name,
            grid,
            model.dimension,
            gap,
            1e3 * phonon_edge,
            gap - phonon_edge,
            gap + phonon_edge,
        )
    )
    for path in written:
        print('  wrote %s' % path)
    return {
        'photon_ev': photon,
        'temps_k': temperatures,
        'eps2': epsilon,
        'eps2_average': average,
        'alpha_cm': alpha,
        'gap_ev': gap,
        'phonon_edge_ev': phonon_edge,
    }
