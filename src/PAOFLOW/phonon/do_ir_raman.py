"""Infrared (IR) and Raman spectra of zone-centre phonons (Stages 3 & 4).

Infrared (Stage 3)
------------------
The infrared activity of a phonon mode is governed by the change in the
macroscopic dipole moment it produces. For a mode :math:`\\nu` at the zone
centre with (mass-weighted) eigenvector :math:`e_{\\nu}` the *mode effective
charge vector* is

.. math::

    \\bar{Z}_{\\nu,\\alpha} = \\sum_{k,\\beta}
        Z^{*}_{k,\\alpha\\beta}\\, \\frac{e_{\\nu,k\\beta}}{\\sqrt{M_k}},

where :math:`Z^{*}_{k}` is the Born effective charge tensor of atom :math:`k`
(units of the elementary charge), :math:`M_k` its mass and :math:`\\alpha,\\beta`
Cartesian indices. The IR oscillator strength (intensity) of the mode is

.. math::

    I_{\\nu} \\propto \\sum_{\\alpha} |\\bar{Z}_{\\nu,\\alpha}|^{2}.

The transverse-optical (TO) eigenvectors at exactly :math:`\\Gamma` are used; the
non-analytical correction (LO-TO splitting) is irrelevant to the oscillator
strengths, so only the Born charges, eigenvectors and masses are required. A
Lorentzian broadening of each mode produces a continuous spectrum
:math:`I(\\omega) = \\sum_{\\nu} I_{\\nu}\\,L(\\omega;\\omega_{\\nu},\\gamma)`.

Raman (Stage 4)
---------------
The (non-resonant, Placzek) Raman activity of a mode is governed by the
derivative of the electronic dielectric/polarisability tensor with respect to
the mode normal coordinate :math:`Q_{\\nu}`. The atomic displacement pattern of
mode :math:`\\nu` is :math:`u_{k\\beta} = e_{\\nu,k\\beta}/\\sqrt{M_k}`, so a
finite, mass-weighted displacement of amplitude :math:`\\delta` along the
eigenvector (``+``/``-``) and a central difference of the static dielectric
tensor :math:`\\varepsilon(\\pm\\delta)` give the Raman tensor

.. math::

    R^{\\nu}_{\\alpha\\beta} = \\frac{\\partial \\varepsilon_{\\alpha\\beta}}
        {\\partial Q_{\\nu}} \\approx
        \\frac{\\varepsilon_{\\alpha\\beta}(+\\delta) -
               \\varepsilon_{\\alpha\\beta}(-\\delta)}{2\\,\\delta}.

The orientationally-averaged (powder) Stokes intensity follows from the two
rotational invariants of the (symmetric) Raman tensor, the isotropic mean
:math:`a = \\tfrac{1}{3}\\mathrm{Tr}\\,R` and the anisotropy
:math:`\\gamma^2`,

.. math::

    I_{\\nu} \\propto \\frac{(\\omega_L - \\omega_{\\nu})^4}{\\omega_{\\nu}}
        \\,(n_{\\nu}+1)\\,(45\\,a^2 + 7\\,\\gamma^2),

with :math:`n_{\\nu}` the Bose-Einstein occupation at temperature :math:`T` and
:math:`\\omega_L` the laser frequency. Acoustic modes are rigid translations
and leave :math:`\\varepsilon` invariant, so they are Raman-silent by
construction.
"""

import os

import numpy as np

from .do_phonopy import THZ_TO_CM1


def _unit_scale(units):
    """Return the THz -> ``units`` conversion factor (THz or cm^-1)."""
    return THZ_TO_CM1 if str(units).lower() in ('cm-1', 'cm^-1', 'cm') else 1.0


def _lorentzian(x, x0, gamma):
    """Normalised Lorentzian of full width at half maximum ``gamma``."""
    hw = 0.5 * gamma
    return (hw / np.pi) / ((x - x0) ** 2 + hw**2)


def mode_effective_charges(born, eigvecs, masses):
    """Mode effective charge vectors and IR oscillator strengths.

    Parameters
    ----------
    born : array_like
        Born effective charges ``(natom, 3, 3)`` with ``born[k, alpha, beta]``.
    eigvecs : array_like
        Mass-weighted eigenvectors ``(3*natom, nmodes)`` (phonopy convention),
        where column ``v`` is the eigenvector of mode ``v`` and the component
        index is ``3*k + beta``.
    masses : array_like
        Atomic masses ``(natom,)``.

    Returns
    -------
    tuple
        ``(mode_charges, intensities)`` with ``mode_charges`` of shape
        ``(nmodes, 3)`` (complex) and ``intensities`` ``(nmodes,)`` real, where
        ``intensities[v] = sum_alpha |mode_charges[v, alpha]|^2``.
    """
    born = np.asarray(born, dtype=float)
    eigvecs = np.asarray(eigvecs)
    masses = np.asarray(masses, dtype=float)
    natom = masses.shape[0]
    nmodes = eigvecs.shape[1]

    # Eigenvector component index i = 3*k + beta; the physical displacement is
    # e_i / sqrt(M_k), so weight every triplet by the inverse root mass.
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    # B[alpha, 3k+beta] = Z*_{k,alpha,beta}
    bmat = born.transpose(1, 0, 2).reshape(3, natom * 3)

    mode_charges = np.zeros((nmodes, 3), dtype=complex)
    intensities = np.zeros(nmodes, dtype=float)
    for v in range(nmodes):
        disp = eigvecs[:, v] / sqrt_m
        zbar = bmat @ disp
        mode_charges[v] = zbar
        intensities[v] = float(np.real(np.vdot(zbar, zbar)))
    return mode_charges, intensities


def _irrep_labels(phonon, nmodes, degeneracy_tolerance=None):
    """Best-effort per-mode irreducible-representation labels at Gamma.

    Returns a list of length ``nmodes`` with the irrep symbol of each branch
    (e.g. ``'T1u'``), or ``None`` when phonopy cannot assign labels (missing
    spglib symmetry, unsupported point group, ...).
    """
    try:
        irreps = phonon.run_irreps([0.0, 0.0, 0.0], degeneracy_tolerance=degeneracy_tolerance)
        band_indices = irreps.band_indices
        labels = getattr(irreps, '_ir_labels', None)
        if band_indices is None or labels is None:
            return None
        out = [None] * nmodes
        for sets, lab in zip(band_indices, labels):
            for b in sets:
                if 0 <= b < nmodes:
                    out[b] = lab
        return out
    except Exception:
        return None


def compute_ir_spectrum(
    data_controller,
    born=None,
    dielectric=None,
    born_file=None,
    freq_min=None,
    freq_max=None,
    npoints=2000,
    gamma=4.0,
    units='cm-1',
    fname='phonon',
    write=True,
    intensity_tol=1.0e-4,
):
    """Compute the infrared spectrum from Born charges and Gamma eigenvectors.

    Parameters
    ----------
    born : array_like, optional
        Born effective charges ``(natom_prim, 3, 3)`` in units of the
        elementary charge. When omitted they are taken from ``born_file`` or
        from a previous :func:`compute_born_and_epsilon` / :func:`attach_nac`
        call (``arry['born_charges']``).
    dielectric : array_like, optional
        High-frequency dielectric tensor ``(3, 3)``. Unused by the oscillator
        strengths; accepted for API symmetry and read alongside ``born_file``.
    born_file : str, optional
        Path to a phonopy ``BORN`` file providing the Born charges.
    freq_min, freq_max : float, optional
        Frequency-axis limits of the broadened spectrum (in ``units``).
        Defaults span ``0`` to slightly above the highest mode.
    npoints : int
        Number of points on the broadened-spectrum frequency grid.
    gamma : float
        Lorentzian full width at half maximum (in ``units``).
    units : str
        Frequency unit for all outputs: ``'cm-1'`` (default) or ``'THz'``.
    fname : str
        Output basename; writes ``<fname>_ir_modes.dat`` and
        ``<fname>_ir_spectrum.dat``.
    write : bool
        When ``True`` the two output files are written (rank 0 only).
    intensity_tol : float
        Relative threshold (fraction of the maximum mode intensity) above which
        a mode is flagged IR-active.

    Returns
    -------
    dict
        ``{'frequencies', 'intensities', 'mode_charges', 'spectrum',
        'irreps', 'active'}``.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    # --- Born effective charges -------------------------------------------
    if born is None:
        if born_file is not None:
            from .io import read_born_file

            nac = read_born_file(data_controller, born_file)
            born = nac['born']
            if dielectric is None:
                dielectric = nac['dielectric']
        elif arry.get('born_charges', None) is not None:
            born = arry['born_charges']
        else:
            raise ValueError(
                'compute_ir_spectrum requires Born effective charges: pass '
                'born=..., born_file=..., or run born_charges() first.'
            )
    born = np.asarray(born, dtype=float)

    masses = np.asarray(phonon.primitive.masses, dtype=float)
    natom = masses.shape[0]
    if born.shape != (natom, 3, 3):
        raise ValueError(
            'Born charges have shape %r but the primitive cell has %d atoms; '
            'expected (%d, 3, 3).' % (tuple(born.shape), natom, natom)
        )

    # --- Gamma-point (TO) frequencies and eigenvectors --------------------
    qpts = phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    freqs_thz = np.asarray(qpts.frequencies)[0]
    eigvecs = np.asarray(qpts.eigenvectors)[0]
    nmodes = freqs_thz.shape[0]

    # --- Mode effective charges and oscillator strengths ------------------
    mode_charges, intensities = mode_effective_charges(born, eigvecs, masses)

    scale = _unit_scale(units)
    freqs = freqs_thz * scale

    imax = intensities.max() if intensities.size else 0.0
    active = intensities > (intensity_tol * imax) if imax > 0 else np.zeros(nmodes, bool)

    irreps = _irrep_labels(phonon, nmodes)

    # --- Broadened spectrum ------------------------------------------------
    if freq_min is None:
        freq_min = 0.0
    if freq_max is None:
        fmax = float(freqs.max()) if freqs.size else 0.0
        freq_max = 1.1 * max(fmax, 0.0) + 5.0 * gamma
    grid = np.linspace(freq_min, freq_max, int(npoints))
    spectrum = np.zeros_like(grid)
    for v in range(nmodes):
        if freqs[v] <= 0.0:
            # Acoustic / numerically-imaginary branches do not contribute.
            continue
        spectrum += intensities[v] * _lorentzian(grid, freqs[v], gamma)

    arry['ir_frequencies'] = freqs
    arry['ir_intensities'] = intensities
    arry['ir_mode_charges'] = mode_charges

    if attr.get('verbose', False) and getattr(data_controller, 'rank', 0) == 0:
        print('Infrared-active modes (intensity in arbitrary units):')
        print('  %-6s %14s %14s %-8s' % ('mode', 'freq(%s)' % units, 'intensity', 'irrep'))
        for v in range(nmodes):
            if not active[v]:
                continue
            lab = irreps[v] if irreps is not None and irreps[v] is not None else '-'
            print('  %-6d %14.4f %14.6e %-8s' % (v + 1, freqs[v], intensities[v], lab))

    if write:
        _write_ir_modes(
            data_controller, fname, units, freqs, intensities, mode_charges, active, irreps
        )
        _write_ir_spectrum(data_controller, fname, units, grid, spectrum)

    return {
        'frequencies': freqs,
        'intensities': intensities,
        'mode_charges': mode_charges,
        'spectrum': (grid, spectrum),
        'irreps': irreps,
        'active': active,
    }


def _write_ir_modes(
    data_controller, fname, units, freqs, intensities, mode_charges, active, irreps
):
    """Write the per-mode IR table ``<fname>_ir_modes.dat`` (rank 0 only)."""
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    imax = intensities.max() if intensities.size else 0.0
    norm = intensities / imax if imax > 0 else intensities
    path = os.path.join(attr.get('opath', '.'), fname + '_ir_modes.dat')
    with open(path, 'w') as f:
        f.write('# mode  frequency(%s)  intensity  intensity_norm  |Zbar|  active  irrep\n' % units)
        for v in range(freqs.shape[0]):
            zmag = float(np.sqrt(np.real(np.vdot(mode_charges[v], mode_charges[v]))))
            lab = irreps[v] if irreps is not None and irreps[v] is not None else '-'
            f.write(
                '%6d % 16.8e % 16.8e % 16.8e % 16.8e %3d  %s\n'
                % (v + 1, freqs[v], intensities[v], norm[v], zmag, int(active[v]), lab)
            )
    return path


def _write_ir_spectrum(data_controller, fname, units, grid, spectrum):
    """Write the broadened spectrum ``<fname>_ir_spectrum.dat`` (rank 0 only)."""
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    path = os.path.join(attr.get('opath', '.'), fname + '_ir_spectrum.dat')
    with open(path, 'w') as f:
        f.write('# frequency(%s)  intensity\n' % units)
        for x, y in zip(grid, spectrum):
            f.write('% 16.8e % 16.8e\n' % (x, y))
    return path


# ====================================================================== #
#  Raman spectrum (Stage 4)                                              #
# ====================================================================== #

# Boltzmann constant in cm^-1 / K (k_B / (h c)); used for the Bose factor.
_KB_CM1 = 0.6950348004


def mode_displacement_vectors(eigvecs, masses, delta):
    """Cartesian displacement patterns of every mode for a finite step ``delta``.

    The physical displacement of atom ``k`` for mode ``v`` is
    ``delta * e_{v,k} / sqrt(M_k)`` (mass-weighted eigenvector), so a unit step
    along the normal coordinate ``Q_v`` corresponds to ``delta = 1``.

    Parameters
    ----------
    eigvecs : array_like
        Mass-weighted eigenvectors ``(3*natom, nmodes)`` (phonopy convention;
        column ``v`` is mode ``v``, component index ``3*k + beta``).
    masses : array_like
        Atomic masses ``(natom,)``.
    delta : float
        Normal-coordinate step amplitude.

    Returns
    -------
    numpy.ndarray
        Displacements ``(nmodes, natom, 3)`` (same length units as the cell).
    """
    eigvecs = np.asarray(eigvecs)
    masses = np.asarray(masses, dtype=float)
    natom = masses.shape[0]
    nmodes = eigvecs.shape[1]
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    disp = np.zeros((nmodes, natom, 3), dtype=float)
    for v in range(nmodes):
        u = np.real(eigvecs[:, v]) * (delta / sqrt_m)
        disp[v] = u.reshape(natom, 3)
    return disp


def raman_invariants(tensor):
    """Rotational invariants ``(a, gamma2)`` of a Raman tensor.

    The Raman tensor ``R`` is characterised by its isotropic mean
    polarisability ``a = Tr(R)/3`` and its anisotropy ``gamma2``,

    .. math::

        \\gamma^2 = \\tfrac{1}{2}\\big[|R_{xx}-R_{yy}|^2 +
            |R_{yy}-R_{zz}|^2 + |R_{zz}-R_{xx}|^2\\big]
            + 3\\,(|R_{xy}|^2 + |R_{xz}|^2 + |R_{yz}|^2).

    The squared magnitudes make the invariants valid for the **complex** Raman
    tensor of the resonance case as well; for a real (non-resonant) tensor they
    reduce to the usual real invariants.  ``a`` is returned complex when the
    input is complex (its magnitude enters the activity), and real otherwise.

    Parameters
    ----------
    tensor : array_like
        A ``(3, 3)`` Raman tensor (symmetrised internally).

    Returns
    -------
    tuple
        ``(a, gamma2)`` with the (possibly complex) isotropic mean and the
        (real) anisotropy invariant.
    """
    r = np.asarray(tensor)
    r = 0.5 * (r + r.T)
    a = np.trace(r) / 3.0
    gamma2 = 0.5 * (
        np.abs(r[0, 0] - r[1, 1]) ** 2
        + np.abs(r[1, 1] - r[2, 2]) ** 2
        + np.abs(r[2, 2] - r[0, 0]) ** 2
    ) + 3.0 * (np.abs(r[0, 1]) ** 2 + np.abs(r[0, 2]) ** 2 + np.abs(r[1, 2]) ** 2)
    if np.iscomplexobj(r):
        return complex(a), float(gamma2)
    return float(a), float(gamma2)


def raman_powder_activity(tensor):
    """Orientationally-averaged Raman activity ``45 |a|^2 + 7 gamma2``."""
    a, gamma2 = raman_invariants(tensor)
    return 45.0 * abs(a) ** 2 + 7.0 * gamma2


def _bose_factor(freq_cm, temperature):
    """Bose-Einstein occupation ``n(omega, T)`` for ``freq_cm`` in cm^-1."""
    if temperature is None or temperature <= 0.0:
        return 0.0
    x = float(freq_cm) / (_KB_CM1 * float(temperature))
    if x <= 0.0:
        return 0.0
    # Guard against overflow for high-frequency / low-T modes.
    if x > 700.0:
        return 0.0
    return 1.0 / np.expm1(x)


def raman_tensors_from_epsilon(eps_plus, eps_minus, delta):
    """Finite-difference Raman tensors ``dε/dQ`` for every mode.

    Parameters
    ----------
    eps_plus, eps_minus : array_like
        Dielectric tensors ``(nmodes, 3, 3)`` for the ``+delta`` and
        ``-delta`` displacements along each mode. Modes that were not displaced
        should carry identical ``+/-`` tensors (or zeros) and yield a null
        Raman tensor.  May be real (non-resonant, static) or complex
        (resonance, evaluated at the laser frequency).
    delta : float
        Normal-coordinate step amplitude used to build the displacements.

    Returns
    -------
    numpy.ndarray
        Raman tensors ``(nmodes, 3, 3)`` (symmetrised), complex when the input
        dielectric tensors are complex.
    """
    eps_plus = np.asarray(eps_plus)
    eps_minus = np.asarray(eps_minus)
    r = (eps_plus - eps_minus) / (2.0 * float(delta))
    return 0.5 * (r + np.transpose(r, (0, 2, 1)))


def compute_raman_spectrum(
    data_controller,
    eps_plus,
    eps_minus,
    delta,
    computed=None,
    laser_nm=None,
    temperature=300.0,
    freq_min=None,
    freq_max=None,
    npoints=2000,
    gamma=4.0,
    units='cm-1',
    fname='phonon',
    write=True,
    intensity_tol=1.0e-4,
):
    """Compute the powder Raman spectrum from finite-difference dielectric data.

    Parameters
    ----------
    eps_plus, eps_minus : array_like
        Dielectric tensors ``(nmodes, 3, 3)`` harvested from the ``+delta`` /
        ``-delta`` displaced-cell ``do_epsilon`` runs.  Real for the
        non-resonant (static) spectrum, or complex when evaluated at a laser
        frequency for the resonance spectrum.
    delta : float
        Normal-coordinate step amplitude used to displace the cells.
    computed : array_like of bool, optional
        Mask ``(nmodes,)`` flagging the modes that were actually displaced and
        for which a Raman tensor is meaningful. Defaults to all modes.
    laser_nm : float, optional
        Excitation wavelength in nm. When given, the
        ``(omega_L - omega_nu)^4`` resonance prefactor is applied; otherwise it
        is omitted (set to 1).
    temperature : float
        Temperature in K for the Bose ``(n+1)`` Stokes factor.
    freq_min, freq_max, npoints, gamma, units, fname, write : see
        :func:`compute_ir_spectrum`.
    intensity_tol : float
        Relative threshold (fraction of the maximum activity) above which a
        mode is flagged Raman-active.

    Returns
    -------
    dict
        ``{'frequencies', 'activities', 'intensities', 'tensors',
        'invariants', 'spectrum', 'irreps', 'active'}``.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    eps_plus = np.asarray(eps_plus)
    eps_minus = np.asarray(eps_minus)

    qpts = phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    freqs_thz = np.asarray(qpts.frequencies)[0]
    nmodes = freqs_thz.shape[0]
    if eps_plus.shape != (nmodes, 3, 3) or eps_minus.shape != (nmodes, 3, 3):
        raise ValueError(
            'Dielectric arrays have shapes %r / %r but %d modes are expected; '
            'each must be (%d, 3, 3).'
            % (tuple(eps_plus.shape), tuple(eps_minus.shape), nmodes, nmodes)
        )

    if computed is None:
        computed = np.ones(nmodes, dtype=bool)
    else:
        computed = np.asarray(computed, dtype=bool)

    scale = _unit_scale(units)
    freqs = freqs_thz * scale
    freqs_cm = freqs_thz * THZ_TO_CM1  # invariants/thermal use cm^-1 frequencies

    tensors = raman_tensors_from_epsilon(eps_plus, eps_minus, delta)

    laser_cm = (1.0e7 / float(laser_nm)) if laser_nm else None

    activities = np.zeros(nmodes, dtype=float)
    intensities = np.zeros(nmodes, dtype=float)
    invariants = np.zeros((nmodes, 2), dtype=float)
    for v in range(nmodes):
        if not computed[v] or freqs_cm[v] <= 0.0:
            continue
        a, gamma2 = raman_invariants(tensors[v])
        invariants[v] = (float(np.real(a)), gamma2)
        act = 45.0 * abs(a) ** 2 + 7.0 * gamma2
        activities[v] = act
        # Stokes scattering cross-section prefactor.
        pref = _bose_factor(freqs_cm[v], temperature) + 1.0
        if laser_cm is not None:
            pref *= (laser_cm - freqs_cm[v]) ** 4
        intensities[v] = act * pref / freqs_cm[v]

    amax = activities.max() if activities.size else 0.0
    active = activities > (intensity_tol * amax) if amax > 0 else np.zeros(nmodes, bool)

    irreps = _irrep_labels(phonon, nmodes)

    if freq_min is None:
        freq_min = 0.0
    if freq_max is None:
        fmax = float(freqs.max()) if freqs.size else 0.0
        freq_max = 1.1 * max(fmax, 0.0) + 5.0 * gamma
    grid = np.linspace(freq_min, freq_max, int(npoints))
    spectrum = np.zeros_like(grid)
    for v in range(nmodes):
        if freqs[v] <= 0.0 or intensities[v] <= 0.0:
            continue
        spectrum += intensities[v] * _lorentzian(grid, freqs[v], gamma)

    arry['raman_frequencies'] = freqs
    arry['raman_activities'] = activities
    arry['raman_intensities'] = intensities
    arry['raman_tensors'] = tensors

    if attr.get('verbose', False) and getattr(data_controller, 'rank', 0) == 0:
        print('Raman-active modes (activity in arbitrary units):')
        print(
            '  %-6s %14s %14s %14s %-8s'
            % ('mode', 'freq(%s)' % units, 'activity', 'intensity', 'irrep')
        )
        for v in range(nmodes):
            if not active[v]:
                continue
            lab = irreps[v] if irreps is not None and irreps[v] is not None else '-'
            print(
                '  %-6d %14.4f %14.6e %14.6e %-8s'
                % (v + 1, freqs[v], activities[v], intensities[v], lab)
            )

    if write:
        _write_raman_modes(
            data_controller,
            fname,
            units,
            freqs,
            activities,
            intensities,
            invariants,
            active,
            irreps,
        )
        _write_raman_spectrum(data_controller, fname, units, grid, spectrum)

    return {
        'frequencies': freqs,
        'activities': activities,
        'intensities': intensities,
        'tensors': tensors,
        'invariants': invariants,
        'spectrum': (grid, spectrum),
        'irreps': irreps,
        'active': active,
    }


def _write_raman_modes(
    data_controller, fname, units, freqs, activities, intensities, invariants, active, irreps
):
    """Write the per-mode Raman table ``<fname>_raman_modes.dat`` (rank 0)."""
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    amax = activities.max() if activities.size else 0.0
    norm = activities / amax if amax > 0 else activities
    path = os.path.join(attr.get('opath', '.'), fname + '_raman_modes.dat')
    with open(path, 'w') as f:
        f.write(
            '# mode  frequency(%s)  activity  activity_norm  intensity  '
            'a_mean  gamma2  active  irrep\n' % units
        )
        for v in range(freqs.shape[0]):
            lab = irreps[v] if irreps is not None and irreps[v] is not None else '-'
            f.write(
                '%6d % 16.8e % 16.8e % 16.8e % 16.8e % 16.8e % 16.8e %3d  %s\n'
                % (
                    v + 1,
                    freqs[v],
                    activities[v],
                    norm[v],
                    intensities[v],
                    invariants[v, 0],
                    invariants[v, 1],
                    int(active[v]),
                    lab,
                )
            )
    return path


def _write_raman_spectrum(data_controller, fname, units, grid, spectrum):
    """Write the broadened Raman spectrum ``<fname>_raman_spectrum.dat`` (rank 0)."""
    _, attr = data_controller.data_dicts()
    if getattr(data_controller, 'rank', 0) != 0:
        return None
    path = os.path.join(attr.get('opath', '.'), fname + '_raman_spectrum.dat')
    with open(path, 'w') as f:
        f.write('# frequency(%s)  intensity\n' % units)
        for x, y in zip(grid, spectrum):
            f.write('% 16.8e % 16.8e\n' % (x, y))
    return path
