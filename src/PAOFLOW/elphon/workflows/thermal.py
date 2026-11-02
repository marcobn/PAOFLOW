"""Thermal workflows of the special displacement method (ZG supercells).

One ZG supercell per temperature (:func:`PAOFLOW.phonon.special_displacement.zg_configuration`,
``PAOFLOW.special_displacement``) carries the Williams-Lax thermal average of the clamped-ion
observables.  After the ``pw.x`` scf of the cell, this module gives

- :func:`thermal_spectral_function`: the supercell bands unfolded onto the primitive zone
  along a path, as a spectral function ``A(k, E; T)`` (inhomogeneous widths and band shifts of the
  static disorder, no lifetimes: see ``docs/internals/special-displacement.md``);
- :func:`ensemble_spectral_function`, :func:`band_statistics`, :func:`peak_width_at_crossing`:
  the configuration average of several seeds and the shift / width of a band;
- :func:`static_linewidth`: the adiabatic Eliashberg plateau ``pi int alpha^2F (2n + 1)`` that
  the static width of a metal is compared with;
- :func:`thermal_dielectric`: the direct dielectric function of the ZG cell
  (``epsilon_2(omega; T)`` of the Williams-Lax theory, phonon-assisted transitions included);
- :func:`zg_cells`, :func:`write_zg_run_script`: the bookkeeping of the cells written by the
  ``thermal/zg.py`` driver (``zg/seed<s>/`` with the manifest ``zg.json``).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .system import is_root

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
KB_EV = 8.617333262e-5

#: High-symmetry points of the fcc lattice in the primitive reciprocal basis (``unfold_bands``).
FCC_POINTS = {
    'G': [0.0, 0.0, 0.0],
    'X': [0.5, 0.0, 0.5],
    'L': [0.5, 0.5, 0.5],
    'W': [0.5, 0.25, 0.75],
    'K': [0.375, 0.375, 0.75],
    'U': [0.625, 0.25, 0.625],
}


def _comm():
    """``MPI.COMM_WORLD`` or ``None`` without mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD


def pao_model_from_save(
    system,
    save_dir: str,
    *,
    configuration: str = 'standard',
    pthr: float = 0.95,
    outputdir: str | None = None,
    label: str = 'PAOFLOW',
) -> tuple[dict | None, object]:
    """PAO Hamiltonian of a QE save as a model dict of :func:`~PAOFLOW.models.band_unfold.unfold_bands`.

    Parameters
    ----------
    system : ElphonSystem
        For the basis tables and the root directory.
    save_dir : str
        The QE save (primitive cell or ZG supercell).
    configuration : str
        PAO basis preset (the same for the primitive and the supercell).
    pthr : float
        Projectability threshold.
    outputdir : str, optional
        Output directory of the PAOFLOW run (default ``<root>/output``).
    label : str
        Label stored in the model dict.

    Returns
    -------
    model : dict or None
        :func:`~PAOFLOW.models.band_unfold.paoflow_to_model_dict` of the run on the root rank
        (``None`` on the other ranks: the real-space Hamiltonian lives on rank 0).
    data_controller : DataController
        Of the run (energies relative to its Fermi level).
    """
    from ...models.band_unfold import paoflow_to_model_dict
    from ...PAOFLOW import PAOFLOW

    pf = PAOFLOW(
        workpath=system.root, savedir=save_dir,
        outputdir=outputdir or os.path.join(system.root, 'output'), verbose=False,
    )  # fmt: skip
    if system.basis_path:
        pf.projections(basispath=system.basis_path, configuration=configuration)
    else:
        pf.projections(configuration=configuration)
    pf.projectability(pthr=pthr)
    pf.pao_hamiltonian()
    model = paoflow_to_model_dict(pf.data_controller, label=label) if is_root() else None
    return model, pf.data_controller


def rescale_model_alat(model: dict, alat: float) -> dict:
    """Express a model dict of :func:`~PAOFLOW.models.band_unfold.unfold_bands` in another ``alat``.

    Parameters
    ----------
    model : dict
        Output of :func:`~PAOFLOW.models.band_unfold.paoflow_to_model_dict` (lengths in units of
        its own ``alat``).
    alat : float
        The new length unit (bohr).  The unfolding needs the primitive and the supercell model in
        the same ``alat``; a supercell save carries its own, larger one.

    Returns
    -------
    dict
        The same dict with ``a_vectors``, the atomic positions and the lattice grid ``R`` rescaled
        and ``alat`` replaced.
    """
    f = float(model['alat']) / float(alat)
    model['model']['a_vectors'] = np.asarray(model['model']['a_vectors'], dtype=float) * f
    for atom in model['model']['atoms'].values():
        atom['tau'] = (np.asarray(atom['tau'], dtype=float) * f).tolist()
    model['_paoflow']['R'] = np.asarray(model['_paoflow']['R'], dtype=float) * f
    model['alat'] = float(alat)
    return model


def symmetry_points(system, points: Mapping | None = None) -> dict:
    """High-symmetry points of the unfolding path in the primitive reciprocal basis.

    Parameters
    ----------
    system : ElphonSystem
        Its coarse-grid save gives the Bravais lattice (``ibrav``) when ``points`` is ``None``.
    points : mapping, optional
        ``{label: [k1, k2, k3]}`` in crystal coordinates (e.g. :data:`FCC_POINTS`); returned as is.

    Returns
    -------
    dict
        ``{label: [k1, k2, k3]}``; the lattice defaults of PAOFLOW's band paths
        (:func:`~PAOFLOW.spectrum.kpnts_interpolation_mesh._getHighSymPoints`) with ``'G'`` for Γ.

    Raises
    ------
    ValueError
        ``ibrav = 0`` saves have no default path: pass ``points``.
    """
    if points:
        return {
            str(k): [float(x) for x in np.asarray(v, dtype=float).ravel()[:3]]
            for k, v in dict(points).items()
        }
    import xml.etree.ElementTree as ET

    from ...spectrum.kpnts_interpolation_mesh import _getHighSymPoints

    root = ET.parse(os.path.join(system.save_path, 'data-file-schema.xml')).getroot()
    structure = root.find('.//output/atomic_structure')
    if structure is None:
        structure = root.find('.//atomic_structure')
    ibrav = int(structure.get('bravais_index', 0))
    alat = float(structure.get('alat'))
    cell = structure.find('cell')
    at = np.array([[float(x) for x in cell.find(a).text.split()] for a in ('a1', 'a2', 'a3')])
    if ibrav == 0:
        raise ValueError('ibrav = 0 has no default path: set sym_points (THERMAL of system.py)')
    special, _ = _getHighSymPoints(ibrav, alat, at / alat)
    return {
        ('G' if name == 'gG' else str(name)): [
            float(x) for x in np.asarray(v, dtype=float).ravel()[:3]
        ]
        for name, v in special.items()
    }


def spectral_function_from_weights(
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    energy_grid: NDArray[np.float64],
    broadening_ev: float,
) -> NDArray[np.float64]:
    """Unfolded spectral function ``A(k, E) = sum_n W_n(k) g(E - E_n(k))`` with a Gaussian ``g``.

    Parameters
    ----------
    energies : NDArray[np.float64]
        ``(nk, nstates)`` supercell eigenvalues along the path (eV).
    weights : NDArray[np.float64]
        ``(nk, nstates)`` unfolding weights.
    energy_grid : NDArray[np.float64]
        ``(ne,)`` energies (eV).
    broadening_ev : float
        Gaussian standard deviation (eV).

    Returns
    -------
    NDArray[np.float64]
        ``(nk, ne)`` in 1/eV; ``integral A dE = sum_n W_n(k)`` at every k.
    """
    e = np.asarray(energies, dtype=float)
    w = np.asarray(weights, dtype=float)
    grid = np.asarray(energy_grid, dtype=float)
    sig = float(broadening_ev)
    lo, hi = grid.min() - 6 * sig, grid.max() + 6 * sig
    out = np.zeros((e.shape[0], grid.size))
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sig)
    for ik in range(e.shape[0]):
        sel = (e[ik] > lo) & (e[ik] < hi) & (w[ik] > 1.0e-10)
        if not sel.any():
            continue
        d = (grid[None, :] - e[ik, sel][:, None]) / sig
        out[ik] = norm * (w[ik, sel][:, None] * np.exp(-0.5 * d * d)).sum(axis=0)
    return out


def save_thermal_spectral(result: dict, path: str) -> str:
    """Write a spectral-function result (:func:`thermal_spectral_function`) as ``.npz``.

    Parameters
    ----------
    result : dict
        ``kdist``, ``sym_ticks``, ``energy_ev``, ``spectral``, ``E_pc``, ``sum_rule``,
        ``temperature_k``, ``label``, ``path``, ``broadening_ev`` and, optionally, ``E_sc``,
        ``weights``, ``configurations``.
    path : str
        Output file.

    Returns
    -------
    str
        ``path``.
    """
    ticks = result.get('sym_ticks', [])
    arrays = {
        'kdist': np.asarray(result['kdist'], dtype=float),
        'energy_ev': np.asarray(result['energy_ev'], dtype=float),
        'spectral': np.asarray(result['spectral'], dtype=float),
        'E_pc': np.asarray(result['E_pc'], dtype=float),
        'sum_rule': np.asarray(result['sum_rule'], dtype=float),
        'sym_tick_distance': np.array([float(t[0]) for t in ticks]),
        'sym_tick_label': np.array([str(t[1]) for t in ticks]),
        'temperature_k': np.nan
        if result.get('temperature_k') is None
        else float(result['temperature_k']),
        'label': str(result.get('label', '')),
        'path': str(result.get('path', '')),
        'broadening_ev': float(result.get('broadening_ev', 0.0)),
        'configurations': int(result.get('configurations', 1)),
    }
    for key in ('E_sc', 'weights'):
        if result.get(key) is not None:
            arrays[key] = np.asarray(result[key], dtype=float)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_thermal_spectral(path: str) -> dict:
    """Read a ``.npz`` of :func:`save_thermal_spectral` back into the result dictionary.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    dict
        As returned by :func:`thermal_spectral_function` (``temperature_k`` ``None`` when absent).
    """
    data = np.load(path)
    temperature = float(data['temperature_k'])
    out = {
        'kdist': data['kdist'],
        'sym_ticks': [
            (float(d), str(lab))
            for d, lab in zip(data['sym_tick_distance'], data['sym_tick_label'])
        ],
        'energy_ev': data['energy_ev'],
        'spectral': data['spectral'],
        'E_pc': data['E_pc'],
        'sum_rule': data['sum_rule'],
        'temperature_k': None if np.isnan(temperature) else temperature,
        'label': str(data['label']) if 'label' in data else '',
        'path': str(data['path']) if 'path' in data else '',
        'broadening_ev': float(data['broadening_ev']) if 'broadening_ev' in data else 0.0,
        'configurations': int(data['configurations']) if 'configurations' in data else 1,
    }
    for key in ('E_sc', 'weights'):
        if key in data:
            out[key] = data[key]
    return out


def thermal_spectral_function(
    system,
    zg_save: str,
    path_str: str,
    sym_points: dict,
    *,
    primitive_save: str | None = None,
    reference_positions=None,
    temperature_k: float | None = None,
    nk_per_seg: int = 100,
    energy_window: tuple[float, float, int] = (-0.3, 0.3, 601),
    broadening_ev: float = 0.005,
    configuration: str = 'standard',
    pthr: float = 0.95,
    outdir: str | None = None,
    label: str = 'zg',
) -> dict:
    """Spectral function of a ZG supercell unfolded onto the primitive zone along a path.

    Parameters
    ----------
    system : ElphonSystem
        Basis tables, root and (default) primitive save.
    zg_save : str
        QE save of the ZG supercell scf.
    path_str : str
        Path through ``sym_points``, e.g. ``'G-X'`` (:func:`~PAOFLOW.models.band_unfold.unfold_bands`).
    sym_points : dict
        High-symmetry points in the primitive reciprocal basis (:func:`symmetry_points`).
    primitive_save : str, optional
        Primitive-cell save of the reference bands (default ``system.save_path``).
    reference_positions : array_like, optional
        ``(natom_sc, 3)`` undisplaced positions of the supercell atoms in bohr, in the order of the
        save: the unfolding maps every atom to its ideal site, which the thermal displacements
        would otherwise hide.  ``None`` reads ``reference_positions`` from the ``<prefix>.npz`` that
        ``write_zg_inputs`` placed next to the scf input (two levels above the save).
    temperature_k : float, optional
        Stored with the result (``None`` for the undisplaced reference cell).
    nk_per_seg : int
        k-points per path segment.
    energy_window : tuple
        ``(E_min, E_max, n)`` in eV relative to the Fermi level.
    broadening_ev : float
        Gaussian width of the spectral function (eV).
    configuration, pthr
        PAO basis of both runs.
    outdir : str, optional
        Writes ``thermal_spectral_<label>.npz`` there (:func:`save_thermal_spectral`).
    label : str
        File label.

    Returns
    -------
    dict
        ``kdist``, ``sym_ticks``, ``energy_ev``, ``spectral`` ``(nk, ne)``, ``E_pc`` ``(nk, nawf_pc)``
        (primitive bands, relative to the primitive Fermi level), ``E_sc``, ``weights``, ``sum_rule``,
        ``temperature_k``, ``label``, ``path``, ``broadening_ev``.  Energies of the supercell are
        relative to its own Fermi level.  Under MPI the PAOFLOW runs are collective, the unfolding
        runs on rank 0 and the result is broadcast.
    """
    from ...models.band_unfold import unfold_bands

    comm = _comm()
    pc, _ = pao_model_from_save(
        system, primitive_save or system.save_path, configuration=configuration, pthr=pthr,
        outputdir=outdir, label='primitive',
    )  # fmt: skip
    sc, _ = pao_model_from_save(
        system, zg_save, configuration=configuration, pthr=pthr, outputdir=outdir, label=label
    )
    out = None
    if is_root(comm):
        sc = rescale_model_alat(sc, pc['alat'])
        if reference_positions is None:
            prefix = os.path.basename(os.path.normpath(zg_save)).replace('.save', '')
            npz = os.path.join(
                os.path.dirname(os.path.dirname(os.path.normpath(zg_save))), prefix + '.npz'
            )
            if os.path.isfile(npz):
                reference_positions = np.load(npz)['reference_positions']
        if reference_positions is not None:
            ref = np.asarray(reference_positions, dtype=float) / float(sc['alat'])
            atoms = sc['model']['atoms']
            if len(atoms) != ref.shape[0]:
                raise ValueError(
                    'reference_positions has %d atoms, the save %d' % (ref.shape[0], len(atoms))
                )
            for i in range(ref.shape[0]):
                atoms[str(i)]['tau'] = ref[i].tolist()
        res = unfold_bands(pc, sc, sym_points, path_str, nk_per_seg=nk_per_seg, verbose=False)
        grid = np.linspace(energy_window[0], energy_window[1], int(energy_window[2]))
        spectral = spectral_function_from_weights(res.E_sc, res.W, grid, broadening_ev)
        out = {
            'kdist': np.asarray(res.kdist),
            'sym_ticks': [(float(d), str(lab)) for d, lab in res.sym_ticks],
            'energy_ev': grid,
            'spectral': spectral,
            'E_pc': np.asarray(res.E_pc),
            'E_sc': np.asarray(res.E_sc),
            'weights': np.asarray(res.W),
            'sum_rule': np.asarray(res.sum_rule_check()),
            'temperature_k': temperature_k,
            'label': label,
            'path': path_str,
            'broadening_ev': float(broadening_ev),
            'configurations': 1,
        }
        if outdir:
            save_thermal_spectral(out, os.path.join(outdir, 'thermal_spectral_%s.npz' % label))
    if comm is not None and comm.Get_size() > 1:
        out = comm.bcast(out, root=0)
    return out


def ensemble_spectral_function(results: Sequence[dict], label: str = 'mean') -> dict:
    """Configuration average of spectral functions on the same path and energy grid.

    Parameters
    ----------
    results : sequence of dict
        Outputs of :func:`thermal_spectral_function` (several seeds at one temperature).
    label : str
        Label of the average.

    Returns
    -------
    dict
        ``spectral`` and ``sum_rule`` averaged, ``E_pc``, ``kdist``, ``sym_ticks``, ``energy_ev``,
        ``temperature_k``, ``path`` and ``broadening_ev`` of the first result, ``configurations``
        = the number of results (``E_sc`` and ``weights`` are dropped).
    """
    first = results[0]
    for other in results[1:]:
        if other['spectral'].shape != first['spectral'].shape or not (
            np.allclose(other['kdist'], first['kdist'])
            and np.allclose(other['energy_ev'], first['energy_ev'])
        ):
            raise ValueError('the spectral functions are not on the same path and energy grid')
    return {
        'kdist': np.asarray(first['kdist']),
        'sym_ticks': list(first.get('sym_ticks', [])),
        'energy_ev': np.asarray(first['energy_ev']),
        'spectral': np.mean([r['spectral'] for r in results], axis=0),
        'E_pc': np.asarray(first['E_pc']),
        'sum_rule': np.mean([r['sum_rule'] for r in results], axis=0),
        'temperature_k': first.get('temperature_k'),
        'label': label,
        'path': first.get('path', ''),
        'broadening_ev': float(first.get('broadening_ev', 0.0)),
        'configurations': int(sum(int(r.get('configurations', 1)) for r in results)),
    }


def band_statistics(
    result: dict, band: int | None, half_window_ev: float = 0.25, min_weight: float = 0.3
) -> dict:
    """Shift and spread of the unfolded weight of a primitive band along the path.

    Parameters
    ----------
    result : dict
        Output of :func:`thermal_spectral_function` or :func:`ensemble_spectral_function`.
    band : int or None
        Primitive band index (``None``: every entry is ``None``).
    half_window_ev : float
        The moments are taken within ``+- half_window_ev`` of the primitive band energy, clipped at
        the energy grid; k-points whose clipped window is narrower than ``half_window_ev`` are
        skipped.
    min_weight : float
        k-points with less integrated weight in the window are skipped.

    Returns
    -------
    dict
        Per k-point ``k_index``, ``shift_ev`` (first moment minus the primitive energy),
        ``spread_ev`` (square root of the second central moment, Gaussian broadening included),
        ``weight``; the summaries ``shift_mean_ev``, ``shift_rms_ev``, ``spread_mean_ev``,
        ``spread_max_ev``, ``spread_intrinsic_ev`` (mean spread with the broadening removed in
        quadrature), ``weight_mean`` (``None`` when no k-point qualifies).
    """
    empty = {
        'k_index': np.zeros(0, dtype=int), 'shift_ev': np.zeros(0), 'spread_ev': np.zeros(0),
        'weight': np.zeros(0), 'shift_mean_ev': None, 'shift_rms_ev': None,
        'spread_mean_ev': None, 'spread_max_ev': None, 'spread_intrinsic_ev': None,
        'weight_mean': None,
    }  # fmt: skip
    if band is None:
        return empty
    E_pc, grid, A = result['E_pc'], np.asarray(result['energy_ev']), result['spectral']
    rows = []
    for ik in range(E_pc.shape[0]):
        e0 = float(E_pc[ik, band])
        lo, hi = max(grid[0], e0 - half_window_ev), min(grid[-1], e0 + half_window_ev)
        if hi - lo < half_window_ev:  # less than half of the window inside the energy grid
            continue
        a = A[ik] * ((grid >= lo) & (grid <= hi))
        w = float(_trapezoid(a, grid))
        if w < min_weight:
            continue
        m1 = float(_trapezoid(a * grid, grid) / w)
        m2 = float(_trapezoid(a * (grid - m1) ** 2, grid) / w)
        rows.append((ik, m1 - e0, np.sqrt(max(m2, 0.0)), w))
    if not rows:
        return empty
    k_index, shift, spread, weight = (np.array(c) for c in zip(*rows))
    sigma = float(result.get('broadening_ev', 0.0))
    return {
        'k_index': k_index.astype(int),
        'shift_ev': shift,
        'spread_ev': spread,
        'weight': weight,
        'shift_mean_ev': float(shift.mean()),
        'shift_rms_ev': float(np.sqrt((shift**2).mean())),
        'spread_mean_ev': float(spread.mean()),
        'spread_max_ev': float(spread.max()),
        'spread_intrinsic_ev': float(np.sqrt(max(spread.mean() ** 2 - sigma**2, 0.0))),
        'weight_mean': float(weight.mean()),
    }


def peak_width_at_crossing(
    result: dict, band: int | None = None, energy_ev: float = 0.0, half_window_ev: float = 0.1
) -> dict:
    """Position and FWHM of the unfolded peak where a primitive band crosses an energy.

    Parameters
    ----------
    result : dict
        Output of :func:`thermal_spectral_function`.
    band : int, optional
        Primitive band index crossing ``energy_ev`` (default: the first that crosses).
    energy_ev : float
        Energy of the crossing (0 = Fermi level).
    half_window_ev : float
        The peak is searched within this range of the primitive band energy at the path point
        nearest to the crossing.

    Returns
    -------
    dict
        ``k_index``, ``kdist``, ``band``, ``E_pc_ev`` (the primitive band at that k, up to half a
        path step away from ``energy_ev``), ``peak_ev``, ``shift_ev`` (``peak_ev - E_pc_ev``: the
        displacement of the unfolded state from the primitive band, free of the path
        discretisation), ``fwhm_ev``, ``weight`` (integrated A within the window), or ``None``
        entries when no crossing is found.
    """
    E_pc, grid, A = result['E_pc'], result['energy_ev'], result['spectral']
    bands = range(E_pc.shape[1]) if band is None else [int(band)]
    for b in bands:
        d = E_pc[:, b] - energy_ev
        cross = np.nonzero((d[:-1] * d[1:] <= 0.0) & ~((d[:-1] == 0.0) & (d[1:] == 0.0)))[0]
        if cross.size:
            i0 = int(cross[0])
            ik = i0 if abs(E_pc[i0, b] - energy_ev) < abs(E_pc[i0 + 1, b] - energy_ev) else i0 + 1
            e0 = float(E_pc[ik, b])
            win = np.abs(grid - e0) <= half_window_ev
            col = A[ik] * win
            ip = int(np.argmax(col))
            half = 0.5 * col[ip]
            left = ip
            while left > 0 and col[left] > half:
                left -= 1
            right = ip
            while right < col.size - 1 and col[right] > half:
                right += 1
            de = grid[1] - grid[0]
            return {
                'k_index': ik,
                'kdist': float(result['kdist'][ik]),
                'band': b,
                'E_pc_ev': e0,
                'peak_ev': float(grid[ip]),
                'shift_ev': float(grid[ip]) - e0,
                'fwhm_ev': float((right - left) * de),
                'weight': float(_trapezoid(col, grid)),
            }
    return {
        'k_index': None,
        'kdist': None,
        'band': None,
        'E_pc_ev': None,
        'peak_ev': None,
        'shift_ev': None,
        'fwhm_ev': None,
        'weight': None,
    }


def static_linewidth(eliashberg_npz: str, temps_k: Sequence[float]) -> NDArray[np.float64]:
    """Adiabatic Eliashberg plateau ``Gamma(T) = pi int alpha^2F(w) (2 n(w, T) + 1) dw``.

    Parameters
    ----------
    eliashberg_npz : str
        ``superconductivity/output/eliashberg.npz`` (``omega`` in eV, ``a2F``).
    temps_k : sequence of float
        Temperatures (K).

    Returns
    -------
    NDArray[np.float64]
        ``Gamma`` in eV, one per temperature: ``-Im Sigma`` of a state far from the Fermi level on
        the phonon scale, i.e. the half width that a static (Williams-Lax) ensemble of displaced
        cells produces at the Fermi crossing of a metal.  The Lorentzian FWHM is ``2 Gamma``.

    Notes
    -----
    .. math::

        \\Gamma(T) = \\pi \\int_0^\\infty d\\Omega\\, \\alpha^2F(\\Omega)\\,[2 n_B(\\Omega, T) + 1]
    """
    data = np.load(eliashberg_npz)
    omega = np.asarray(data['omega'], dtype=float)
    a2f = np.asarray(data['a2F'], dtype=float)
    if omega.max() > 2.0:  # meV table
        omega = omega * 1.0e-3
    sel = omega > 0.0
    omega, a2f = omega[sel], a2f[sel]
    out = []
    for temp in np.atleast_1d(np.asarray(temps_k, dtype=float)):
        if temp > 0.0:
            x = np.clip(omega / (KB_EV * temp), 1.0e-12, 700.0)
            factor = 2.0 / np.expm1(x) + 1.0
        else:
            factor = np.ones_like(omega)
        out.append(np.pi * _trapezoid(a2f * factor, omega))
    return np.asarray(out)


def plot_thermal_spectral_function(result: dict, filename: str, vmax: float | None = None) -> str:
    """Map of ``A(k, E)`` along the path with the primitive bands overlaid.

    Parameters
    ----------
    result : dict
        Output of :func:`thermal_spectral_function`.
    filename : str
        Output figure.
    vmax : float, optional
        Colour scale maximum (default: the 99th percentile).

    Returns
    -------
    str
        ``filename``.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    k, e, a = result['kdist'], result['energy_ev'], result['spectral']
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.pcolormesh(
        k, e, a.T, shading='auto', cmap='Greys', vmin=0.0, vmax=vmax or np.percentile(a, 99)
    )
    ax.plot(k, result['E_pc'], color='#0072B2', lw=0.6, alpha=0.8)
    for d, lab in result['sym_ticks']:
        ax.axvline(d, color='k', lw=0.5)
    ax.axhline(0.0, color='k', lw=0.5, ls='--')
    ax.set_xticks([d for d, _ in result['sym_ticks']])
    ax.set_xticklabels([lab for _, lab in result['sym_ticks']])
    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(e.min(), e.max())
    ax.set_ylabel('E - E$_F$ (eV)')
    title = result.get('label', '')
    if result.get('temperature_k') is not None:
        title += '  T = %g K' % result['temperature_k']
    if int(result.get('configurations', 1)) > 1:
        title += '  (%d configurations)' % int(result['configurations'])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return filename


def write_thermal_summary(rows: Sequence[dict], path: str) -> str:
    """Write the per-temperature summary of the thermal driver (JSON + a text table).

    Parameters
    ----------
    rows : sequence of dict
        One per temperature (and the reference): ``temperature_k``, ``label``,
        ``configurations``, ``band``, ``peak_shift_ev`` (the unfolded state at the crossing
        relative to the primitive band), ``peak_vs_reference_ev``, ``fwhm_ev``, ``weight``,
        ``shift_ev`` (mean along the path), ``shift_vs_reference_ev``, ``shift_rms_ev``,
        ``spread_ev``, ``sum_rule``, ``gamma_static_ev`` (``None`` allowed).
    path : str
        The ``.json`` file; a ``.dat`` table with the same stem is written alongside.

    Returns
    -------
    str
        ``path``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump([dict(r) for r in rows], handle, indent=2)
    keys = (
        'temperature_k', 'configurations', 'band', 'peak_shift_ev', 'peak_vs_reference_ev',
        'shift_ev', 'shift_vs_reference_ev', 'shift_rms_ev', 'fwhm_ev', 'spread_ev',
        'spread_intrinsic_ev', 'weight',
        'sum_rule', 'gamma_static_ev',
    )  # fmt: skip
    with open(os.path.splitext(path)[0] + '.dat', 'w') as handle:
        handle.write('# ' + '  '.join(keys) + '   (energies in eV; reference row: T = nan)\n')
        for row in rows:
            handle.write(
                '  '.join(
                    'nan'
                    if row.get(k) is None
                    else ('%d' % row[k] if k in ('configurations', 'band') else '%.6g' % row[k])
                    for k in keys
                )
                + '\n'
            )
    return path


def load_thermal_summary(path: str) -> list[dict]:
    """Rows of :func:`write_thermal_summary` (the ``.json`` file)."""
    with open(path) as handle:
        return json.load(handle)


def plot_thermal_summary(rows: Sequence[dict], filename: str, name: str = '') -> str:
    """Band shift and width versus temperature, with the Eliashberg plateau when available.

    Parameters
    ----------
    rows : sequence of dict
        :func:`load_thermal_summary`.
    filename : str
        Output figure.
    name : str
        System name of the titles.

    Returns
    -------
    str
        ``filename``.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    thermal = [r for r in rows if r.get('temperature_k') is not None]
    reference = next((r for r in rows if r.get('temperature_k') is None), None)
    temps = np.array([r['temperature_k'] for r in thermal])

    def column(key, scale=1.0e3):
        return np.array([np.nan if r.get(key) is None else scale * r[key] for r in thermal])

    spread = (
        column('spread_intrinsic_ev')
        if any(r.get('spread_intrinsic_ev') is not None for r in thermal)
        else column('spread_ev')
    )

    fig, (ax_shift, ax_width) = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)
    ax_shift.plot(
        temps,
        column('peak_shift_ev'),
        'o-',
        color='#0072B2',
        lw=1.2,
        ms=5,
        label='state at the crossing vs the primitive band',
    )
    ax_shift.plot(
        temps,
        column('shift_ev'),
        's--',
        color='#D55E00',
        lw=1.2,
        ms=5,
        label='first moment along the path',
    )
    if reference is not None and reference.get('peak_shift_ev') is not None:
        ax_shift.axhline(
            1.0e3 * reference['peak_shift_ev'],
            color='0.4',
            lw=1.0,
            ls=':',
            label='undisplaced supercell',
        )
    ax_shift.set_xlabel('T (K)')
    ax_shift.set_ylabel('band shift (meV)')
    ax_shift.set_title('%s: shift of the crossing band' % name, fontsize=10)
    ax_shift.legend(frameon=False, fontsize=8)
    ax_width.plot(
        temps, column('fwhm_ev'), 'o-', color='#0072B2', lw=1.2, ms=5, label='FWHM at the crossing'
    )
    ax_width.plot(
        temps,
        2.3548 * spread,
        's--',
        color='#D55E00',
        lw=1.2,
        ms=5,
        label='2.35 x spread along the path (broadening removed)',
    )
    gamma = column('gamma_static_ev')
    if np.isfinite(gamma).any():
        ax_width.plot(
            temps, 2.0 * gamma, '-', color='#009E73', lw=1.6, label='2 Gamma = 2 pi int a2F (2n+1)'
        )
    ax_width.set_xlabel('T (K)')
    ax_width.set_ylabel('width (meV)')
    ax_width.set_title('static width vs the Eliashberg plateau', fontsize=10)
    ax_width.legend(frameon=False, fontsize=8, loc='upper left')
    for ax in (ax_shift, ax_width):
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return filename


def zg_cells(zg_dir: str) -> list[dict]:
    """The ZG cells written under ``<zg_dir>/seed<s>/`` and their saves.

    Parameters
    ----------
    zg_dir : str
        Directory holding one ``seed<s>/`` folder per seed, each with the manifest ``zg.json`` of
        :func:`~PAOFLOW.phonon.special_displacement.write_zg_inputs`.

    Returns
    -------
    list of dict
        One per cell, ordered by temperature (the reference first) and seed: ``seed``,
        ``temperature_k`` (``None`` for the undisplaced cell), ``prefix``, ``directory``, ``input``,
        ``save`` (the QE save of the finished scf: the schema file exists and the ``.out`` next to
        the input, when present, reports ``JOB DONE``; ``None`` otherwise), ``dims``, ``natoms``,
        ``error``.
    """
    from ...phonon.special_displacement import MANIFEST

    cells = []
    for manifest in sorted(glob.glob(os.path.join(zg_dir, 'seed*', MANIFEST))):
        directory = os.path.dirname(manifest)
        try:
            seed = int(os.path.basename(directory)[4:])
        except ValueError:
            continue
        with open(manifest) as handle:
            data = json.load(handle)
        for entry in data['configurations']:
            prefix = entry['prefix']
            saves = [
                s
                for s in glob.glob(os.path.join(directory, '*', prefix + '.save'))
                + glob.glob(os.path.join(directory, prefix + '.save'))
                if os.path.isfile(os.path.join(s, 'data-file-schema.xml'))
            ]
            out = os.path.join(directory, os.path.splitext(entry['input'])[0] + '.out')
            if saves and os.path.isfile(out):
                with open(out, errors='replace') as handle:
                    if 'JOB DONE' not in handle.read():
                        saves = []  # pw.x still running (or stopped): its save is not final
            cells.append(
                {
                    'seed': seed,
                    'temperature_k': entry.get('temperature_k'),
                    'prefix': prefix,
                    'directory': directory,
                    'input': os.path.join(directory, entry['input']),
                    'save': sorted(saves)[0] if saves else None,
                    'dims': [int(n) for n in data['dims']],
                    'natoms': int(data['natoms']),
                    'error': entry.get('error'),
                }
            )
    cells.sort(
        key=lambda c: (-1.0 if c['temperature_k'] is None else c['temperature_k'], c['seed'])
    )
    return cells


def write_zg_run_script(zg_dir: str, filename: str = 'run_zg.sh') -> str:
    """Shell script running ``pw.x`` on every ZG cell under ``zg_dir`` (skips finished runs).

    Parameters
    ----------
    zg_dir : str
        As in :func:`zg_cells`.
    filename : str
        Script name (written into ``zg_dir``, executable).

    Returns
    -------
    str
        The script path.  ``NP`` ranks, ``NK`` pools and ``PW`` are environment variables.
    """
    cells = zg_cells(zg_dir)
    inputs = [os.path.relpath(c['input'], zg_dir) for c in cells]
    lines = [
        '#!/bin/bash',
        '# pw.x scf of the ZG supercells (run from this directory): NP=8 bash %s' % filename,
        'set -e',
        'NP=${NP:-8}; NK=${NK:-1}',
        'PW=${PW:-pw.x}',
        'log() { echo "$(date +%T) $*"; }',
        'for input in %s; do' % ' '.join(inputs),
        '  dir=$(dirname $input); name=$(basename ${input%.in})',
        '  if grep -q "JOB DONE" $dir/$name.out 2>/dev/null; then log "$input done"; continue; fi',
        '  log "$input"',
        '  (cd $dir && mpirun -np $NP $PW -nk $NK -in $name.in > $name.out)',
        'done',
        'log "all ZG cells done"',
    ]
    path = os.path.join(zg_dir, filename)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    os.chmod(path, 0o755)
    return path


def thermal_dielectric(
    system, zg_save: str, outdir: str, configuration: str = 'standard', **kwargs
) -> None:
    """Direct dielectric function of a ZG supercell (``epsilon(omega; T)`` of the Williams-Lax theory).

    Parameters
    ----------
    system, zg_save, outdir
        As in :func:`thermal_spectral_function` (``outdir`` receives ``epsr_<c>.dat``, ``epsi_<c>.dat``).
    configuration : str
        PAO basis preset (``'standard'`` is enough for a supercell; the primitive optics use ``'extended'``).
    **kwargs
        Passed to :func:`~PAOFLOW.elphon.workflows.optics.direct_dielectric` (``nfft``, ``emin``, ``emax``,
        ``ne``, ``delta``, ``intraband``, ``smearing``).
    """
    from .optics import direct_dielectric

    direct_dielectric(system, zg_save, outdir, configuration=configuration, **kwargs)
