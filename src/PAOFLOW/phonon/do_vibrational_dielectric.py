"""Vibrational (ionic) contribution to the dielectric function (Stage 5).

A polar zone-centre phonon modulates the macroscopic polarization and therefore
contributes a resonance to the dielectric function.  Adding the lattice
(ionic) response to the electronic high-frequency dielectric tensor
:math:`\\varepsilon^{\\infty}` gives the full low-frequency dielectric function

.. math::

    \\varepsilon_{\\alpha\\beta}(\\omega) = \\varepsilon^{\\infty}_{\\alpha\\beta}
        + \\frac{e^{2}}{\\varepsilon_{0}\\,m_{u}\\,V}
          \\sum_{\\nu} \\frac{\\bar{Z}_{\\nu,\\alpha}\\,\\bar{Z}_{\\nu,\\beta}}
                            {\\omega_{\\nu}^{2} - \\omega^{2} - i\\omega\\gamma_{\\nu}},

where :math:`\\bar{Z}_{\\nu,\\alpha} = \\sum_{k,\\beta} Z^{*}_{k,\\alpha\\beta}\\,
e_{\\nu,k\\beta}/\\sqrt{M_k}` is the *mode effective charge vector* already used
for the infrared intensities (:func:`~PAOFLOW.phonon.do_ir_raman.mode_effective_charges`),
:math:`S_{\\nu,\\alpha\\beta} = \\bar{Z}_{\\nu,\\alpha}\\bar{Z}_{\\nu,\\beta}` is the
mode-oscillator-strength tensor (Gonze & Lee, PRB 55, 10355), :math:`V` is the
primitive-cell volume and :math:`m_u` the atomic mass unit.  The mode effective
charges are returned in units of :math:`e\\,\\mathrm{amu}^{-1/2}`, so the
elementary charge :math:`e` and :math:`m_u` are restored by the SI prefactor
above.

The static limit :math:`\\varepsilon(0) = \\varepsilon^{\\infty} + \\sum_\\nu
S_\\nu/\\omega_\\nu^2` (the generalized Lyddane-Sachs-Teller relation) and, for a
polar crystal, a *reststrahlen* band where :math:`\\mathrm{Re}\\,\\varepsilon < 0`
(between the transverse :math:`\\omega_{TO}` and longitudinal :math:`\\omega_{LO}`
frequencies) follow directly.  Acoustic modes are rigid translations
(:math:`\\omega_\\nu \\le 0`, :math:`\\bar{Z}_\\nu = 0`) and do not contribute.
"""

import os

import numpy as np

from ..utils.constants import (
    BOHR_RADIUS_SI,
    ELECTRONVOLT_SI,
    HBAR,
    SPEED_OF_LIGHT,
    UMA_SI,
)
from .do_ir_raman import _irrep_labels, _unit_scale, mode_effective_charges

# Vacuum permittivity (F/m); not present in utils.constants.
_EPS0_SI = 8.8541878128e-12

# Dielectric tensor components written to disk (matches do_epsilon / io.py).
_EPS_COMPONENTS = (
    ('xx', 0, 0),
    ('yy', 1, 1),
    ('zz', 2, 2),
    ('xy', 0, 1),
    ('xz', 0, 2),
    ('yz', 1, 2),
)


def _display_to_rad_per_s(units):
    """Angular-frequency conversion factor: omega[rad/s] = K * nu[`units`]."""
    if str(units).lower() in ('cm-1', 'cm^-1', 'cm'):
        # omega = 2 pi c nu, with c in cm/s.
        return 2.0 * np.pi * SPEED_OF_LIGHT * 100.0
    # THz -> rad/s.
    return 2.0 * np.pi * 1.0e12


def compute_vibrational_dielectric(
    data_controller,
    born=None,
    dielectric=None,
    born_file=None,
    gamma=4.0,
    freq_min=None,
    freq_max=None,
    npoints=2000,
    units='cm-1',
    emit_ev=True,
    emissivity=False,
    emis_angles=(0.0,),
    emis_ntheta=64,
    emis_temperature=(300.0,),
    outdir='vibdielectric',
    fname='phonon',
    write=True,
):
    """Vibrational (ionic) dielectric function from Born charges and phonons.

    Parameters
    ----------
    born : array_like, optional
        Born effective charges ``(natom_prim, 3, 3)`` in units of the
        elementary charge.  When omitted they are read from ``born_file`` or a
        previous :func:`compute_born_and_epsilon` / :func:`attach_nac` call
        (``arry['born_charges']``).
    dielectric : array_like, optional
        High-frequency dielectric tensor :math:`\\varepsilon^{\\infty}`
        ``(3, 3)``.  When omitted it is read from ``born_file`` or from
        ``arry['dielectric_tensor']``.
    born_file : str, optional
        Path to a phonopy ``BORN`` file providing both the Born charges and
        :math:`\\varepsilon^{\\infty}`.
    gamma : float or array_like
        Phonon linewidth(s) used as the Lorentzian damping (in ``units``).  A
        scalar broadens every mode equally; an array gives a per-mode width.
    freq_min, freq_max : float, optional
        Frequency-axis limits of the dielectric function (in ``units``).
        Defaults span ``0`` to just above the highest longitudinal-optical
        frequency, so the reststrahlen band is captured.
    npoints : int
        Number of points on the frequency grid.
    units : str
        Frequency unit for inputs/outputs: ``'cm-1'`` (default) or ``'THz'``.
    emit_ev : bool
        When ``True`` (default) the per-component ``eps{r,i}_<ab>.dat`` files are
        written with the frequency axis in **eV**, so they can be plotted
        directly with :meth:`PAOFLOW.GPAO.GPAO.plot_optical` (whose axis is in
        eV).  When ``False`` the axis is written in ``units``.
    emissivity : bool
        When ``True`` the reststrahlen (phonon) emissivity is derived from the
        vibrational dielectric function via the Fresnel/Kirchhoff helpers in
        :mod:`PAOFLOW.response.do_epsilon` and written alongside the
        ``eps{r,i}`` files (directional ``refl_th*``/``emis_th*``, spectral
        hemispherical ``emish_*`` and Planck-weighted total ``emist_*``).  The
        total hemispherical emissivity is integrated over the far-IR grid only,
        so it captures the lattice (phonon) contribution to the emissivity.
    emis_angles : array_like
        Incidence angles (degrees) for the directional reflectivity/emissivity.
    emis_ntheta : int
        Polar-angle samples for the hemispherical integral.
    emis_temperature : float or array_like
        Temperature(s) (K) for the total hemispherical emissivity.
    outdir : str
        Sub-directory (under ``opath``) for the per-component dielectric files.
    fname : str
        Output basename for the static-summary file.
    write : bool
        When ``True`` the output files are written (rank 0 only).

    Returns
    -------
    dict
        ``{'grid', 'grid_ev', 'eps', 'static', 'eps_inf', 'frequencies',
        'mode_strengths', 'irreps'}``.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    born, dielectric = _resolve_born_dielectric(data_controller, born, dielectric, born_file)

    masses = np.asarray(phonon.primitive.masses, dtype=float)
    natom = masses.shape[0]
    if born.shape != (natom, 3, 3):
        raise ValueError(
            'Born charges have shape %r but the primitive cell has %d atoms; '
            'expected (%d, 3, 3).' % (tuple(born.shape), natom, natom)
        )

    eps_inf = np.asarray(dielectric, dtype=float).reshape(3, 3)

    # --- Gamma-point frequencies / eigenvectors and mode strengths --------
    qpts = phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    freqs_thz = np.asarray(qpts.frequencies)[0]
    eigvecs = np.asarray(qpts.eigenvectors)[0]
    nmodes = freqs_thz.shape[0]

    mode_charges, _ = mode_effective_charges(born, eigvecs, masses)
    # Oscillator-strength tensor S_v = outer(Zbar_v, Zbar_v) (real at Gamma).
    zbar = np.real(mode_charges)
    strengths = np.einsum('va,vb->vab', zbar, zbar)

    scale = _unit_scale(units)
    freqs = freqs_thz * scale  # display units (cm-1 / THz)

    # --- SI prefactor P = e^2 / (eps0 m_u V); result is dimensionless ------
    volume = float(phonon.primitive.volume) * BOHR_RADIUS_SI**3  # Bohr^3 -> m^3
    pref_si = ELECTRONVOLT_SI**2 / (_EPS0_SI * UMA_SI * volume)  # 1/s^2

    kdisp = _display_to_rad_per_s(units)  # omega[rad/s] = kdisp * nu[units]
    pref_disp = pref_si / kdisp**2  # prefactor in (display-unit)^2

    # Only optical, positive-frequency modes contribute.  Acoustic branches sit
    # at (numerically) zero frequency; a small relative cutoff avoids dividing
    # by their residual frequency (their mode strength is ~0 in any case).
    fmax_all = float(freqs.max()) if freqs.size else 0.0
    acoustic_tol = 1.0e-3 * fmax_all
    optical = freqs > acoustic_tol

    # --- Static dielectric and (generalized LST) LO estimate for the range -
    static = eps_inf.copy()
    for v in range(nmodes):
        if optical[v]:
            static += pref_disp * strengths[v] / freqs[v] ** 2

    if freq_min is None:
        freq_min = 0.0
    gamma_arr = np.broadcast_to(np.asarray(gamma, dtype=float), (nmodes,)).copy()
    if freq_max is None:
        freq_max = _estimate_freq_max(freqs, optical, strengths, eps_inf, pref_disp, gamma_arr)

    grid = np.linspace(float(freq_min), float(freq_max), int(npoints))

    # --- Dielectric function on the grid ----------------------------------
    eps = np.broadcast_to(eps_inf, (grid.shape[0], 3, 3)).astype(complex).copy()
    w2 = grid**2
    for v in range(nmodes):
        if not optical[v]:
            continue
        denom = freqs[v] ** 2 - w2 - 1j * grid * gamma_arr[v]
        eps += pref_disp * strengths[v][None, :, :] / denom[:, None, None]

    # eV axis (E = hbar omega) for plotting / file output.
    grid_ev = HBAR * kdisp * grid  # HBAR in eV*s, kdisp*grid in rad/s

    irreps = _irrep_labels(phonon, nmodes)

    arry['vib_dielectric_grid'] = grid
    arry['vib_dielectric_grid_ev'] = grid_ev
    arry['vib_dielectric_eps'] = eps
    arry['vib_dielectric_static'] = static
    arry['vib_dielectric_inf'] = eps_inf
    arry['vib_mode_strengths'] = strengths

    if attr.get('verbose', False) and getattr(data_controller, 'rank', 0) == 0:
        ionic = static - eps_inf
        print('Vibrational dielectric (static limit):')
        for a in range(3):
            print(
                '  eps_inf[%d] = % .4f % .4f % .4f   eps(0)[%d] = % .4f % .4f % .4f'
                % (a, *eps_inf[a], a, *static[a])
            )
        print(
            '  ionic contribution eps(0) - eps_inf (diagonal): % .4f % .4f % .4f'
            % (ionic[0, 0], ionic[1, 1], ionic[2, 2])
        )

    emissivity_result = None
    if emissivity:
        emissivity_result = _compute_emissivity(
            eps, grid_ev, emis_angles, emis_ntheta, emis_temperature
        )
        arry['vib_emissivity_hemispherical'] = emissivity_result['hemispherical']
        arry['vib_emissivity_total'] = emissivity_result['total']
        if attr.get('verbose', False) and getattr(data_controller, 'rank', 0) == 0:
            for t, T in enumerate(emissivity_result['temperatures']):
                vals = emissivity_result['total'][t]
                print(
                    '  total hemispherical emissivity at %.1f K (xx, yy, zz) = '
                    '% .4f % .4f % .4f' % (T, vals[0], vals[1], vals[2])
                )

    if write:
        axis = grid_ev if emit_ev else grid
        _write_eps_components(data_controller, outdir, axis, eps)
        _write_static_summary(
            data_controller,
            fname,
            units,
            freqs,
            optical,
            strengths,
            eps_inf,
            static,
            irreps,
        )
        if emissivity_result is not None:
            _write_emissivity_files(data_controller, outdir, axis, emissivity_result)

    return {
        'grid': grid,
        'grid_ev': grid_ev,
        'eps': eps,
        'static': static,
        'eps_inf': eps_inf,
        'frequencies': freqs,
        'mode_strengths': strengths,
        'irreps': irreps,
        'emissivity': emissivity_result,
    }


def _resolve_born_dielectric(data_controller, born, dielectric, born_file):
    """Fill in Born charges / epsilon_inf from a BORN file or stored arrays."""
    arry, _ = data_controller.data_dicts()
    if born is None or dielectric is None:
        if born_file is not None:
            from .io import read_born_file

            nac = read_born_file(data_controller, born_file)
            if born is None:
                born = nac['born']
            if dielectric is None:
                dielectric = nac['dielectric']
        else:
            if born is None:
                if arry.get('born_charges', None) is None:
                    raise ValueError(
                        'vibrational dielectric requires Born effective charges: '
                        'pass born=..., born_file=..., or run born_charges() first.'
                    )
                born = arry['born_charges']
            if dielectric is None:
                if arry.get('dielectric_tensor', None) is None:
                    raise ValueError(
                        'vibrational dielectric requires the high-frequency '
                        'dielectric tensor: pass dielectric=..., born_file=..., '
                        'or run born_charges() first.'
                    )
                dielectric = arry['dielectric_tensor']
    return np.asarray(born, dtype=float), np.asarray(dielectric, dtype=float)


def _estimate_freq_max(freqs, optical, strengths, eps_inf, pref_disp, gamma_arr):
    """Generous upper frequency bound that covers the highest LO mode.

    Uses the generalized LST estimate ``omega_LO^2 ~ omega_TO_max^2 +
    pref * sum_v Tr(S_v)/3 / min(eps_inf_diag)`` so the reststrahlen band is
    inside the default window.
    """
    if not np.any(optical):
        return 1.0
    fmax = float(freqs[optical].max())
    s_tot = float(sum(np.trace(strengths[v]) / 3.0 for v in range(len(freqs)) if optical[v]))
    eps_min = float(min(np.diag(eps_inf)))
    eps_min = eps_min if eps_min > 1.0e-6 else 1.0
    lo2 = fmax**2 + pref_disp * s_tot / eps_min
    lo = np.sqrt(max(lo2, fmax**2))
    return 1.2 * max(lo, fmax) + 5.0 * float(np.max(gamma_arr))


def _write_eps_components(data_controller, outdir, axis, eps):
    """Write ``eps{r,i}_<ab>.dat``, ``eels_<ab>.dat`` and ``refl_<ab>.dat``.

    Two-column files (no header) so :func:`read_dos_PAO` / ``plot_optical`` read
    them directly.  ``axis`` is the frequency axis (eV by default).
    """
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    path = os.path.join(attr.get('opath', '.'), outdir)
    os.makedirs(path, exist_ok=True)
    for comp, i, j in _EPS_COMPONENTS:
        e = eps[:, i, j]
        # Energy-loss -Im(1/eps) and normal-incidence reflectivity, guarded
        # against identically-zero components (e.g. off-diagonals of an
        # isotropic crystal) which would otherwise divide by zero.
        nonzero = np.abs(e) > 0.0
        eels = np.zeros_like(np.real(e))
        with np.errstate(divide='ignore', invalid='ignore'):
            eels[nonzero] = -np.imag(1.0 / e[nonzero])
        n = np.sqrt(e)
        refl = np.zeros_like(np.real(e))
        refl[nonzero] = np.abs((n[nonzero] - 1.0) / (n[nonzero] + 1.0)) ** 2
        for tag, col in (
            ('epsr', np.real(e)),
            ('epsi', np.imag(e)),
            ('eels', eels),
            ('refl', refl),
        ):
            with open(os.path.join(path, '%s_%s.dat' % (tag, comp)), 'w') as f:
                for x, y in zip(axis, col):
                    f.write('% .8e % .8e\n' % (x, y))
    return path


def _write_static_summary(
    data_controller, fname, units, freqs, optical, strengths, eps_inf, static, irreps
):
    """Write ``<fname>_vibdielectric_static.dat`` (rank 0 only)."""
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    ionic = static - eps_inf
    path = os.path.join(attr.get('opath', '.'), fname + '_vibdielectric_static.dat')
    with open(path, 'w') as f:
        f.write('# Vibrational (ionic) dielectric function: static summary\n')
        f.write('# eps(0) = eps_inf + sum_v S_v / omega_v^2  (generalized LST)\n')
        f.write('#\n# High-frequency (electronic) dielectric tensor eps_inf:\n')
        for a in range(3):
            f.write('#   % 14.8f % 14.8f % 14.8f\n' % tuple(eps_inf[a]))
        f.write('#\n# Ionic contribution eps(0) - eps_inf:\n')
        for a in range(3):
            f.write('#   % 14.8f % 14.8f % 14.8f\n' % tuple(ionic[a]))
        f.write('#\n# Static dielectric tensor eps(0):\n')
        for a in range(3):
            f.write('#   % 14.8f % 14.8f % 14.8f\n' % tuple(static[a]))
        f.write('#\n# Per-mode oscillator strengths (Tr(S_v)/3) and frequencies (%s):\n' % units)
        f.write('# mode  frequency(%s)  strength  irrep\n' % units)
        for v in range(freqs.shape[0]):
            if not optical[v]:
                continue
            lab = irreps[v] if irreps is not None and irreps[v] is not None else '-'
            s = float(np.trace(strengths[v]) / 3.0)
            f.write('%6d % 16.8e % 16.8e  %s\n' % (v + 1, freqs[v], s, lab))
    return path


def _compute_emissivity(eps, grid_ev, angles_deg, ntheta, temps):
    """Reststrahlen emissivity from the vibrational dielectric (diagonal comps).

    Reuses the Fresnel/Kirchhoff helpers in :mod:`PAOFLOW.response.do_epsilon`
    on the diagonal of :math:`\\varepsilon(\\omega) = \\varepsilon^\\infty +
    \\varepsilon^{\\text{ionic}}(\\omega)`.  Returns the directional and
    hemispherical spectral emissivities together with the Planck-weighted total
    hemispherical emissivity (integrated over the supplied far-IR grid only).
    """
    from ..response.do_epsilon import (
        directional_reflectivity,
        spectral_hemispherical_emissivity,
        total_hemispherical_emissivity,
    )
    from ..utils.constants import DEGTORAD

    angles_deg = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    temps = np.atleast_1d(np.asarray(temps, dtype=float))
    diag = ((0, 0), (1, 1), (2, 2))
    ne = grid_ev.shape[0]

    directional = np.empty((angles_deg.size, ne, 3), dtype=float)
    hemispherical = np.empty((ne, 3), dtype=float)
    total = np.empty((temps.size, 3), dtype=float)

    # The Planck weight needs strictly positive energies (the grid starts at 0).
    positive = grid_ev > 0.0
    for c, (i, j) in enumerate(diag):
        epsr = np.real(eps[:, i, j])
        epsi = np.imag(eps[:, i, j])
        for a, ang in enumerate(angles_deg):
            directional[a, :, c] = 1.0 - directional_reflectivity(epsr, epsi, float(ang) * DEGTORAD)
        emis_w = spectral_hemispherical_emissivity(epsr, epsi, int(ntheta))
        hemispherical[:, c] = emis_w
        for t, temperature in enumerate(temps):
            total[t, c] = total_hemispherical_emissivity(
                grid_ev[positive], emis_w[positive], float(temperature)
            )
    return {
        'grid_ev': grid_ev,
        'angles': angles_deg,
        'temperatures': temps,
        'directional': directional,
        'hemispherical': hemispherical,
        'total': total,
    }


def _write_emissivity_files(data_controller, outdir, axis, emis):
    """Write the reststrahlen emissivity spectra (rank 0 only).

    File names mirror :func:`PAOFLOW.response.do_epsilon.write_emissivity`:
    ``refl_th{deg}_<comp>.dat`` / ``emis_th{deg}_<comp>.dat`` (directional, per
    angle), ``emish_<comp>.dat`` (spectral hemispherical) and
    ``emist_<comp>.dat`` (temperature vs total hemispherical emissivity).
    """
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    path = os.path.join(attr.get('opath', '.'), outdir)
    os.makedirs(path, exist_ok=True)

    def _save(name, x, y):
        with open(os.path.join(path, name), 'w') as f:
            for a, b in zip(x, y):
                f.write('% .8e % .8e\n' % (a, b))

    comps = ('xx', 'yy', 'zz')
    angles = emis['angles']
    temps = emis['temperatures']
    for c, comp in enumerate(comps):
        for a, ang in enumerate(angles):
            deg = int(round(float(ang)))
            emis_dir = emis['directional'][a, :, c]
            _save('emis_th%d_%s.dat' % (deg, comp), axis, emis_dir)
            _save('refl_th%d_%s.dat' % (deg, comp), axis, 1.0 - emis_dir)
        _save('emish_%s.dat' % comp, axis, emis['hemispherical'][:, c])
        _save('emist_%s.dat' % comp, temps, emis['total'][:, c])
    return path
