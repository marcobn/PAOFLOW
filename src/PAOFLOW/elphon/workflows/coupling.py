"""Coupling workflow: dynamical quadrupoles from single-q ``ph.x`` runs and long-range checks."""

from __future__ import annotations

import glob
import os
from typing import Sequence

import numpy as np

from ...inputs.read_upf import UPF
from ...phonon.qe_force_constants import read_dynamical_matrix_file
from ..dfpt_io import read_nscf_plane_wave_setup
from ..dfpt_vertex import CoarseVertexSetup
from ..do_pao_eph_dense_q import _phonon_modes_at_q, g_Re_at_q, prepare_dense_vertex
from ..elph_bloch import (
    AMU_RY,
    RY_TO_EV,
    RY_TO_THZ,
    band_vertex,
    precompute_dense_electrons,
    vertex_phases,
    vertex_ws_coefficients,
)
from ..coupling_cache import basis_tag
from ..long_range import LongRangeModel
from ..quadrupoles import (
    DIRECTIONS,
    quadrupoles_from_runs,
    single_q_phonon_inputs,
    write_quadrupole_fmt,
)
from .system import Electrons, ElphonSystem


def quadrupole_run_dir(system: ElphonSystem) -> str:
    """``<phonon dir>/quadrupoles``: the single-q ``ph.x`` runs of the quadrupole expansion."""
    return os.path.join(system.phonon_path, 'quadrupoles')


def quadrupole_inputs(
    system: ElphonSystem,
    *,
    magnitudes: Sequence[float] = (0.02, 0.04),
    directions: Sequence[Sequence[float]] = DIRECTIONS,
    np_default: int = 8,
) -> list[str]:
    """Write the single-q ``ph.x`` inputs (``q01/ph.in`` ...) and ``run.sh`` of the quadrupole expansion.

    Parameters
    ----------
    system : ElphonSystem
        The runs need the scf save of the phonon directory (``<phonon dir>/<prefix>.save``,
        with its ``wfc*.dat``) and ``ph.x`` run with ``epsil = .true.``.
    magnitudes : sequence of float
        ``|q|`` in ``2 pi / alat`` (two values for the q -> 0 extrapolation).
    directions : sequence of sequence of float
        Six directions determine the 18 components per atom.
    np_default : int
        Default of ``NP`` in ``run.sh``.

    Returns
    -------
    list of str
        The run directories written.
    """
    prefix = system.prefix
    qdir = quadrupole_run_dir(system)
    runs = single_q_phonon_inputs(prefix, magnitudes, directions, outdir='./')
    os.makedirs(qdir, exist_ok=True)
    lines = [
        '#!/bin/bash',
        '# single-q ph.x runs for the quadrupole expansion (run from this directory)',
        'set -e',
        'NP=${NP:-%d}' % np_default,
        'for d in q*/; do',
        '  cd $d',
        '  rm -rf _ph0; mkdir -p %s.save' % prefix,
        '  cp ../../%s.save/charge-density.dat ../../%s.save/data-file-schema.xml '
        '../../%s.save/wfc*.dat %s.save/' % (prefix, prefix, prefix, prefix),
        '  cp ../../%s.save/*.upf ../../%s.save/*.UPF %s.save/ 2>/dev/null || true'
        % (prefix, prefix, prefix),
        '  echo "$(date +%T) $d"; mpirun -np $NP ph.x -nk $NP -in ph.in > ph.out 2>&1',
        '  cd ..',
        'done',
        'echo done',
    ]
    with open(os.path.join(qdir, 'run.sh'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    written = []
    for i, (_, q, text) in enumerate(runs):
        directory = os.path.join(qdir, 'q%02d' % (i + 1))
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'ph.in'), 'w') as handle:
            handle.write(text.replace("pseudo_dir = ''", ''))
        np.savetxt(os.path.join(directory, 'q.txt'), q[None], header='q (2 pi / alat), Cartesian')
        written.append(directory)
    return written


def quadrupoles(system: ElphonSystem, output: str) -> dict:
    """Fit Z* (a check) and the dynamical quadrupoles from the finished single-q runs.

    Parameters
    ----------
    system : ElphonSystem
    output : str
        Path of the EPW-format ``quadrupole.fmt`` to write.

    Returns
    -------
    dict
        The fit (``zstar``, ``quadrupoles``, ``quadrupoles_by_magnitude``, ``sign``, ...)
        of :func:`~PAOFLOW.elphon.quadrupoles.quadrupoles_from_runs`, plus ``gamma`` (the
        Gamma dynamical matrix with the dielectric tensor) and ``path``.

    Raises
    ------
    FileNotFoundError
        If the Gamma dynamical matrix or the runs are missing.
    ValueError
        If the Gamma dynamical matrix has no dielectric tensor (``epsil``).
    """
    prefix = system.prefix
    savedir = os.path.join(system.phonon_path, prefix + '.save')
    setup = read_nscf_plane_wave_setup(savedir)
    gamma_dyn = os.path.join(system.phonon_path, prefix + '.dyn1')
    if not os.path.isfile(gamma_dyn):
        raise FileNotFoundError(
            '%s (the Gamma dynamical matrix with epsil = .true.) is needed' % gamma_dyn
        )
    gamma = read_dynamical_matrix_file(gamma_dyn)
    if gamma['epsilon'] is None:
        raise ValueError(
            'the Gamma dynamical matrix has no dielectric tensor: rerun ph.x with epsil = .true.'
        )
    alat, omega = setup['alat'], setup['omega']
    tau_cart = setup['tau_alat'] * alat
    zion = np.array(
        [UPF(os.path.join(savedir, setup['pseudo_files'][sp])).val for sp in setup['atom_species']]
    )
    runs = []
    for directory in sorted(glob.glob(os.path.join(quadrupole_run_dir(system), 'q*'))):
        if os.path.isfile(os.path.join(directory, 'q.txt')):
            runs.append((np.loadtxt(os.path.join(directory, 'q.txt')), directory))
    if len(runs) < 6:
        raise FileNotFoundError(
            'found %d finished runs under %s; six directions are needed'
            % (len(runs), quadrupole_run_dir(system))
        )
    fit = quadrupoles_from_runs(
        runs,
        prefix,
        setup['fft_grid'],
        tau_cart,
        alat,
        omega,
        gamma['epsilon'],
        zion,
        zstar_ref=gamma['born_charges'],
    )
    fit['gamma'] = gamma
    fit['path'] = write_quadrupole_fmt(output, fit['quadrupoles'])
    return fit


def long_range_path(
    system: ElphonSystem,
    electrons: Electrons,
    *,
    nk: int = 24,
    nval: int | None = None,
    direction: Sequence[float] = (1.0, 0.0, 0.0),
    quadrupoles: str | None = None,
    comm=None,
) -> dict:
    """``|g(k = Gamma, q)|^2`` of the valence bands along a Cartesian direction, with and without the model.

    The coarse coupling is interpolated three ways: with the dipole (Fröhlich) model
    subtracted on the coarse grid and restored at every dense q, with the plain
    interpolation of the full coupling, and (optionally) with the dipole + quadrupole
    model.  The LO mode of a polar crystal diverges as ``1/|q|``, which only the model
    reproduces.

    Parameters
    ----------
    system, electrons
        The system (DFPT source) and its electrons.
    nk : int
        Dense grid of the path points (multiples of ``1/nk``).
    nval : int, optional
        Valence bands summed (default ``nelec / spin degeneracy``).
    direction : sequence of float
        Cartesian direction of q in ``2 pi / alat`` (default Gamma-X of a cubic cell).
    quadrupoles : str, optional
        ``quadrupole.fmt`` for the dipole + quadrupole curve.
    comm : MPI communicator, optional

    Returns
    -------
    dict
        ``rows`` (``|q|``, LO/TO/acoustic sums for the model and the plain interpolation,
        the analytic dipole term alone and, with quadrupoles, the LO/TO/acoustic sums of
        the dipole + quadrupole model), ``header``, ``model`` (the dipole model),
        ``omega_lo_first_mev`` and ``has_quadrupoles``.

    Raises
    ------
    ValueError
        For the EPW coupling source.
    """
    if system.coupling_source != 'dfpt':
        raise ValueError('long_range_path needs the DFPT coupling source')
    nscf = electrons.nscf
    nval = int(round(system.nelec / system.degeneracy(electrons))) if nval is None else nval
    setup = CoarseVertexSetup(system.save_path, system.phonon_path, system.qgrid)
    model = LongRangeModel.from_dfpt(setup)
    models = {'model': model}
    if quadrupoles:
        models['quadrupole'] = LongRangeModel.from_coupling(
            system.phonon_path, system.save_path, quadrupoles=quadrupoles
        )
    masses = system.masses(nscf)
    common = dict(
        source='dfpt',
        masses_amu=masses,
        tau_cryst=nscf['tau_cryst'],
        orbital_positions=nscf['orbital_positions'],
        save_dir=system.save_path,
        comm=comm,
    )
    if system.cache_path:
        # the shared g(R_e, R_p) cache of the system folder; the key carries the long-range
        # signature, so every variant below has its own entry
        common['cache_dir'] = system.cache_path
        common['cache_tag'] = basis_tag(electrons.data_controller, system.pthr)
    vertex = {}
    variants = [(name, {'long_range': m}) for name, m in models.items()]
    variants.append(('plain', {'allow_missing_long_range': True}))
    for name, extra in variants:
        vertex[name] = prepare_dense_vertex(
            electrons.projections,
            nscf['kpts_cryst'],
            nscf['bg'],
            nscf['at'],
            system.phonon_path,
            system.qgrid,
            None,
            system.coarse_grid,
            None,
            **common,
            **extra,
        )
    dense = precompute_dense_electrons(
        electrons.HRs,
        nscf['at'],
        nk,
        [0.01],
        system.nelec,
        system.coarse_grid,
        orbital_positions=nscf['orbital_positions'],
    )
    V = dense['V']
    nawf = V.shape[1]
    mass_flat = np.repeat(np.asarray(masses) * AMU_RY, 3)
    q_dir = np.asarray(direction, dtype=float) @ np.linalg.inv(nscf['bg'])
    rows = []
    omega_lo_first = None
    for n in range(1, nk // 2 + 1):
        q = np.rint(q_dir * n / (nk // 2) * nk) / nk  # on the dense grid
        line = [np.linalg.norm(q @ nscf['bg'])]
        quad_line = []
        names = ('model', 'plain') + (('quadrupole',) if 'quadrupole' in vertex else ())
        for name in names:
            g_ReRp, phonon_at_q = vertex[name]['g_ReRp'], vertex[name]['phonon_at_q']
            Nint_p, W_p, Midx_p = vertex[name]['phonon_ws']
            freq_thz, z = _phonon_modes_at_q(q, phonon_at_q)
            omega_ry = np.abs(freq_thz) / RY_TO_THZ
            zmass = z / np.sqrt(mass_flat)[None, :]
            gn_flat = vertex_ws_coefficients(g_Re_at_q(g_ReRp, q, Nint_p, W_p, Midx_p), dense)
            lr = (
                models[name].vertex_pao(q, nawf, nscf['orbital_positions'])
                if name in models
                else None
            )
            ik = 0
            ikq = int(np.ravel_multi_index(np.rint(q * nk).astype(int) % nk, (nk,) * 3))
            g = band_vertex(gn_flat, vertex_phases(dense, [ik]), V[[ikq]], V[[ik]], zmass, lr)[0]
            g = (
                g[:, :nval, :nval]
                / np.sqrt(2.0 * np.where(omega_ry > 1e-8, omega_ry, np.inf))[:, None, None]
            )
            g2 = (np.abs(g) ** 2).sum(axis=(1, 2)) * (RY_TO_EV * 1e3) ** 2  # meV^2 per mode
            order = np.argsort(omega_ry)
            acoustic, to, lo = g2[order[:3]].sum(), g2[order[3:5]].sum(), g2[order[5]]
            if name == 'quadrupole':
                quad_line = [lo, to, acoustic]
            else:
                line += [lo, to, acoustic]
            if name == 'model':
                omega_lo = omega_ry[order[5]]
                if n == 1:
                    omega_lo_first = omega_lo
        # the analytic dipole vertex alone, LO mode, summed over the valence pairs
        coeff, _ = model.coefficients(q)
        freq_thz, z = _phonon_modes_at_q(q, vertex['model']['phonon_at_q'])
        zmass = z / np.sqrt(mass_flat)[None, :]
        c_lo = (zmass @ coeff)[np.argmax(np.abs(freq_thz))]
        line.append(nval * abs(c_lo) ** 2 / (2.0 * omega_lo) * (RY_TO_EV * 1e3) ** 2)
        rows.append(line + quad_line)
    header = (
        '|q| (2pi/a)   LO_model  TO_model  ac_model  LO_plain  TO_plain  ac_plain  LO_dipole_only'
        + ('  LO_quad  TO_quad  ac_quad' if 'quadrupole' in vertex else '')
        + '   [sum over the %d valence bands of |g|^2 / (2 omega), meV^2, k = Gamma]' % nval
    )
    return {
        'rows': np.array(rows),
        'header': header,
        'model': model,
        'omega_lo_first_mev': omega_lo_first * RY_TO_EV * 1e3,
        'has_quadrupoles': 'quadrupole' in vertex,
        'nval': nval,
    }


__all__ = ['long_range_path', 'quadrupole_inputs', 'quadrupole_run_dir', 'quadrupoles']
