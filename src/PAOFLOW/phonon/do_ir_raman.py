"""Infrared (IR) spectrum from Born charges and zone-centre eigenvectors (Stage 3).

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
