"""Superconductivity workflow: Eliashberg function, Tc and the Migdal-Eliashberg gaps.

The functions wrap :func:`~PAOFLOW.elphon.do_pao_eph_dense_q.eliashberg_dense_q`,
:func:`~PAOFLOW.elphon.do_pao_eph.eliashberg_from_qe_coupling` and the isotropic and
anisotropic Migdal-Eliashberg solvers with the inputs of an :class:`~.system.ElphonSystem`
and write the outputs the way the examples do (``alpha2F.dat``, ``eliashberg.npz``,
``fs_coupling.npz``, the EPW-format ``me/`` and ``me_aniso/`` files).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..anisotropic_eliashberg import (
    coupling_strength,
    linearized_eigenvalues_aniso,
    migdal_eliashberg_aniso as _migdal_eliashberg_aniso,
    write_me_aniso_outputs,
)
from ..do_pao_eph import eliashberg_from_qe_coupling
from ..do_pao_eph_dense_q import eliashberg_dense_q
from ..elph_bloch import RY_TO_EV
from ..fermi_surface_coupling import read_fs_coupling, write_fs_coupling
from ..migdal_eliashberg import (
    a2f_from_epw,
    a2f_from_npz,
    linearized_eigenvalues,
    matsubara_lambda,
    migdal_eliashberg_iso,
    write_me_outputs,
)
from .system import Electrons, ElphonSystem, is_root

KB_EV = 8.617333262e-5


def eliashberg(
    system: ElphonSystem,
    electrons: Electrons,
    *,
    nk: int,
    nq: int | None = None,
    sigma_ev: float = 0.05,
    mu_star: float = 0.1,
    coarse_q: bool = False,
    transport: bool = False,
    fs_coupling: bool = False,
    fsthick_ev: float | None = None,
    adaptive_smearing: float | None = None,
    comm=None,
) -> dict:
    """Isotropic Eliashberg properties (and optionally alpha^2F_tr, the Fermi-surface coupling).

    Parameters
    ----------
    system : ElphonSystem
        The system (coupling source, grids, masses, ``nelec``, long-range model, cache).
    electrons : Electrons
        Of :meth:`ElphonSystem.electrons`.
    nk : int
        Dense electron grid per axis (EPW ``nkf``).
    nq : int, optional
        Dense phonon grid per axis (EPW ``nqf``); ``nk`` must be a multiple of it.
        Required unless ``coarse_q``.
    sigma_ev : float, optional
        Fermi-surface smearing (EPW ``degaussw``); the floor of the adaptive widths.
    mu_star : float, optional
        Coulomb pseudopotential of the McMillan and Allen-Dynes Tc.
    coarse_q : bool, optional
        Keep q on the coarse grid (``eliashberg_from_qe_coupling``): the q-points and
        phonons come from the coupling files, only the electrons are interpolated.
    transport : bool, optional
        Also the transport Eliashberg function alpha^2F_tr, lambda_tr and the plasma
        frequency (the metal transport workflow).
    fs_coupling : bool, optional
        Store the state-resolved coupling of the Fermi-surface states for the anisotropic
        Migdal-Eliashberg equations (dense q only, equal dense grids).
    fsthick_ev : float, optional
        Fermi window of the anisotropic states (EPW ``fsthick``).
    adaptive_smearing : float, optional
        Yates prefactor ``a`` of the adaptive Fermi-surface widths ``a |v_nk| dk`` per
        dense state, floored at ``sigma_ev``.
    comm : MPI communicator, optional

    Returns
    -------
    dict
        The output of the backend: ``lambda``, ``omega_log``, ``Tc_mcmillan``,
        ``Tc_allen_dynes``, ``omega``, ``a2F`` and, when requested, ``a2F_tr``,
        ``lambda_tr``, ``omega_plasma_ev`` and ``fs_coupling``.

    Raises
    ------
    ValueError
        If ``nq`` is missing for the dense-q route or does not divide ``nk``.
    """
    nscf = electrons.nscf
    common = dict(
        source=system.coupling_source,
        save_dir=system.save_path,
        masses_amu=system.masses(nscf),
        nk_dense=nk,
        sigmas_ry=[sigma_ev / RY_TO_EV],
        nelec=system.nelec,
        mu_star=mu_star,
        orbital_positions=nscf['orbital_positions'],
        spin_degeneracy=system.degeneracy(electrons),
        comm=comm,
    )
    coupling_dir = system.coupling_dir
    if coarse_q:
        return eliashberg_from_qe_coupling(
            electrons.projections,
            electrons.HRs,
            *electrons.lattice,
            coupling_dir,
            None,
            system.coarse_grid,
            None,
            allow_missing_long_range=not system.long_range,
            **common,
        )
    if nq is None:
        raise ValueError('the dense-q route needs nq (or use coarse_q=True)')
    if nk % nq:
        raise ValueError('nk (%d) must be a multiple of nq (%d)' % (nk, nq))
    arguments = system.coupling_arguments(electrons)
    arguments.pop('coupling_source')  # eliashberg_dense_q calls it ``source``
    arguments.pop('save_dir')
    arguments.pop('masses_amu')
    arguments.pop('orbital_positions')
    return eliashberg_dense_q(
        electrons.projections,
        electrons.HRs,
        *electrons.lattice,
        coupling_dir,
        system.qgrid,
        None,
        None,
        system.coarse_grid,
        None,
        nq_dense=nq,
        fs_coupling=fs_coupling,
        fsthick_ev=fsthick_ev,
        transport=transport,
        alat=nscf['alat'],
        cell_volume=nscf['omega'],
        adaptive_smearing=adaptive_smearing,
        **arguments,
        **common,
    )


def smearing_label(sigma_ev: float, adaptive_smearing: float | None) -> str:
    """``'sigma 0.050 eV'`` or ``'adaptive a = 0.70 (floor 50.0 meV)'``."""
    if adaptive_smearing is None:
        return 'sigma %.3f eV' % sigma_ev
    return 'adaptive a = %.2f (floor %.1f meV)' % (adaptive_smearing, 1e3 * sigma_ev)


def write_eliashberg(
    out: dict, outdir: str, *, mu_star: float, label: str = '', comm=None
) -> list[str]:
    """Print the Eliashberg summary and write ``alpha2F.dat``, ``eliashberg.npz`` (rank 0).

    Parameters
    ----------
    out : dict
        Of :func:`eliashberg`.  ``fs_coupling`` (when present) goes to ``fs_coupling.npz``
        and is removed from ``out``.
    outdir : str
        Output directory (created).
    mu_star : float
        Coulomb pseudopotential (header of ``alpha2F.dat``).
    label : str, optional
        Grid and smearing description for the summary line.
    comm : MPI communicator, optional

    Returns
    -------
    list of str
        Written files (empty on the other ranks).
    """
    fs_coupling = out.pop('fs_coupling', None)
    if not is_root(comm):
        return []
    print('PAOFLOW Eliashberg function%s:' % (' (%s)' % label if label else ''))
    print('  lambda   = %.4f' % out['lambda'])
    print('  w_log    = %.3f meV' % (out['omega_log'] * 1.0e3))
    print(
        '  Tc (McM) = %.2f K   Tc (AD) = %.2f K   (mu* = %.2f)'
        % (out['Tc_mcmillan'], out['Tc_allen_dynes'], mu_star)
    )
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, 'alpha2F.dat')]
    np.savetxt(
        paths[-1],
        np.column_stack([out['omega'] * 1.0e3, out['a2F']]),
        header='omega(meV)  alpha^2F  (lambda=%.4f, w_log=%.2fK, Tc_McM=%.3fK, Tc_AD=%.3fK, mu*=%.2f)'
        % (
            out['lambda'],
            out['omega_log'] / KB_EV,
            out['Tc_mcmillan'],
            out['Tc_allen_dynes'],
            mu_star,
        ),
    )
    if 'lambda_tr' in out:
        print(
            '  lambda_tr = %.4f   plasma frequency = %.3f eV'
            % (out['lambda_tr'], out['omega_plasma_ev'])
        )
        paths.append(os.path.join(outdir, 'alpha2F_tr.dat'))
        np.savetxt(
            paths[-1],
            np.column_stack([out['omega'] * 1.0e3, out['a2F_tr']]),
            header='omega(meV)  alpha^2F_tr  (lambda_tr=%.4f, hbar omega_p=%.4f eV)'
            % (out['lambda_tr'], out['omega_plasma_ev']),
        )
    paths.append(os.path.join(outdir, 'eliashberg.npz'))
    np.savez(paths[-1], **out)
    if fs_coupling is not None:
        paths.append(os.path.join(outdir, 'fs_coupling.npz'))
        write_fs_coupling(paths[-1], fs_coupling)
        print('  Fermi-surface coupling: %d states -> %s' % (fs_coupling['band'].size, paths[-1]))
    for path in paths:
        print('  wrote %s' % path)
    return paths


def migdal_eliashberg(
    source: str,
    outdir: str,
    prefix: str,
    *,
    temps: Sequence[float],
    mu_star: float = 0.1,
    wscut: float = 0.1,
    degaussq_mev: float = 0.15,
    degaussq_linear_mev: float = 0.5,
    npade: float = 90.0,
    pade: bool = True,
    acon: bool = True,
    linear_temps: Sequence[float] | None = (0.25, 6.25, 25),
    epw_a2f: str | None = None,
) -> dict:
    """Isotropic Migdal-Eliashberg equations on an alpha^2F; writes ``outdir`` in EPW's formats.

    Parameters
    ----------
    source : str
        ``eliashberg.npz`` of :func:`write_eliashberg` (the modes are re-broadened with
        ``degaussq_mev``), ignored when ``epw_a2f`` is given.
    outdir : str
        Output directory (``<prefix>.imag_iso_<T>``, ``pade_iso``, ``acon_iso``, ``qdos_iso``,
        ``gap_vs_T.dat``, ``max_eigenvalue.dat``, ``migdal_eliashberg.npz``).
    prefix : str
        File-name prefix of the EPW-format outputs.
    temps : sequence of float
        Temperatures (K).
    mu_star, wscut : float, optional
        Coulomb pseudopotential and Matsubara cutoff (eV; EPW ``muc``, ``wscut``).
    degaussq_mev, degaussq_linear_mev : float, optional
        Phonon smearing of alpha^2F (meV) for the equations and for the linearised kernel
        (EPW ``degaussq``).
    npade : float, optional
        Percentage of Matsubara points of the Padé approximant (EPW ``npade``).
    pade, acon : bool, optional
        Real-axis gap from Padé and from the analytic continuation.
    linear_temps : (Tmin, Tmax, nstemp), optional
        Temperatures of the linearised kernel (``None`` skips it).
    epw_a2f : str, optional
        An EPW ``<prefix>.a2f`` to use instead of ``source`` (its own ``degaussq``).

    Returns
    -------
    dict
        ``result`` (``migdal_eliashberg_iso``), ``linear`` (``linearized_eigenvalues`` or
        ``None``) and ``lambda`` (``2 int alpha^2F/w dw``).

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    """
    if epw_a2f:
        omega, a2F = a2f_from_epw(epw_a2f)
        omega_lin, a2F_lin = omega, a2F
        label = epw_a2f
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError('%s not found; run eliashberg.py first.' % source)
        omega, a2F = a2f_from_npz(source, degaussq_mev * 1e-3)
        omega_lin, a2F_lin = a2f_from_npz(source, degaussq_linear_mev * 1e-3)
        label = source
    lam = matsubara_lambda(omega, a2F, 1.0, 0)[0]
    print(
        'alpha^2F from %s  (lambda = %.4f, mu* = %.2f, wscut = %.3f eV)'
        % (label, lam, mu_star, wscut)
    )
    print('Isotropic Migdal-Eliashberg equations:')
    result = migdal_eliashberg_iso(
        omega, a2F, temps, mu_star, wscut, npade=npade, lpade=pade, lacon=acon, verbose=True
    )
    linear = None
    if linear_temps is not None:
        tmin, tmax, nst = linear_temps
        linear = linearized_eigenvalues(
            omega_lin, a2F_lin, np.linspace(tmin, tmax, int(nst)), mu_star, wscut
        )
    write_me_outputs(result, outdir, prefix, linear)
    print('Tc from Delta^2(T) -> 0     : %.3f K' % result['Tc_gap'])
    if linear is not None:
        print('Tc from the linearised kernel: %.3f K' % linear['Tc_linear'])
    if not result['converged'].all():
        print('WARNING: not converged at T =', result['temps'][~result['converged']])
    print('Wrote %s' % outdir)
    return {'result': result, 'linear': linear, 'lambda': lam}


def migdal_eliashberg_aniso(
    source: str,
    outdir: str,
    prefix: str,
    *,
    temps: Sequence[float],
    mu_star: float = 0.1,
    wscut: float = 0.5,
    npade: float = 90.0,
    pade: bool = True,
    linear_temps: Sequence[float] | None = None,
) -> dict:
    """Anisotropic (Fermi-surface-restricted) Migdal-Eliashberg equations on ``fs_coupling.npz``.

    Parameters
    ----------
    source : str
        ``fs_coupling.npz`` of :func:`write_eliashberg` (``fs_coupling=True``).
    outdir : str
        Output directory in EPW's formats (``<prefix>.imag_aniso_<T>``, gap distributions,
        ``lambda_FS``, ``gap_vs_T_aniso.dat``, ``migdal_eliashberg_aniso.npz``).
    prefix : str
        File-name prefix.
    temps : sequence of float
        Temperatures (K).
    mu_star, wscut, npade, pade
        As :func:`migdal_eliashberg`.
    linear_temps : (Tmin, Tmax, nstemp), optional
        Temperatures of the linearised kernel (``None`` skips it).

    Returns
    -------
    dict
        ``result``, ``linear``, ``coupling`` and ``strength``
        (:func:`~PAOFLOW.elphon.anisotropic_eliashberg.coupling_strength`).

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    """
    if not os.path.isfile(source):
        raise FileNotFoundError('%s not found; run eliashberg.py --fs-coupling first.' % source)
    coupling = read_fs_coupling(source)
    strength = coupling_strength(coupling)
    print(
        'Fermi-surface coupling: %d irreducible states (k %d^3, q %d^3, fsthick %.2f eV)'
        % (
            coupling['band'].size,
            coupling['nk_dense'],
            coupling['nq_dense'],
            coupling['fsthick_ev'],
        )
    )
    print(
        '  lambda = %.4f   lambda_nk = %.3f .. %.3f   N_F = %.4f states/eV/spin'
        % (
            strength['lambda'],
            strength['lambda_nk'].min(),
            strength['lambda_nk'].max(),
            strength['dos_ef'],
        )
    )
    print(
        'Anisotropic Migdal-Eliashberg equations (mu* = %.2f, wscut = %.2f eV):' % (mu_star, wscut)
    )
    result = _migdal_eliashberg_aniso(
        coupling, temps, mu_star, wscut, npade=npade, lpade=pade, verbose=True
    )
    linear = None
    if linear_temps is not None:
        tmin, tmax, nstemp = linear_temps
        linear = linearized_eigenvalues_aniso(
            coupling, np.linspace(tmin, tmax, int(nstemp)), mu_star, wscut
        )
    write_me_aniso_outputs(result, coupling, outdir, prefix, linear)
    print('Tc from Delta_max^2(T) -> 0    : %.2f K' % result['Tc_gap'])
    if linear is not None:
        print('Tc from the linearised kernel : %.2f K' % linear['Tc_linear'])
    if not result['converged'].all():
        print('WARNING: not converged at T =', result['temps'][~result['converged']])
    print('Wrote %s' % outdir)
    return {'result': result, 'linear': linear, 'coupling': coupling, 'strength': strength}
