"""Checks of a pw.x input before it is launched: the cheapest moment to catch a costly mistake.

Each check comes from a failure of a production campaign: k-points inflated by a grid that the lattice group does
not preserve, FFT boxes inflated by a skewed supercell basis, charge sloshing of large metallic cells with plain
mixing, Fourier interpolation from two points per axis, a coupling k-grid that the q-grid does not divide, pool
layouts and core requests that waste billed cores.
"""

from __future__ import annotations

from .diagnosis import Finding
from .pw_output import parse_pw_input
from .scheduler import balanced_pools, billed_cores


def preflight_pw_input(
    text: str,
    run: str,
    displaced: bool = False,
    metallic: bool = True,
    interpolated: bool = False,
    qgrid: tuple[int, int, int] | None = None,
    cores: int | None = None,
    cores_per_pool: int = 6,
    billing_granularity: int = 1,
    floor_bohr: float = 48.0,
) -> list:
    """Findings for a pw.x input before launch.

    Parameters
    ----------
    text : str
        The pw.x input.
    run : str
        Name of the run, for the findings.
    displaced : bool
        The cell has no crystal symmetry (displaced supercell): its grid must not rely on the lattice group.
    metallic : bool
        Metal (charge sloshing is a risk in large cells).
    interpolated : bool
        PAOFLOW will Fourier-interpolate a Hamiltonian built on this grid (needs at least three points per axis).
    qgrid : tuple of int, optional
        Phonon q-grid of the system description that uses this grid as its coupling grid (must divide it).
    cores : int, optional
        Cores to request (enables the pool and billing checks).
    cores_per_pool : int
        Cores of one k-point pool.
    billing_granularity : int
        Billing unit in cores.
    floor_bohr : float
        Smallest acceptable shortest superlattice vector of the k-grid.

    Returns
    -------
    list of Finding
    """
    from .cells import (
        fft_box,
        fft_inflation,
        grid_is_invariant,
        lattice_rotations,
        read_cell,
        reduced_basis,
        shortest_superlattice_vector,
        time_reversal_kpoints,
    )

    findings: list[Finding] = []
    settings = parse_pw_input(text)
    try:
        cell = read_cell(text)
    except (ValueError, AttributeError, IndexError):
        cell = None

    if cell is not None and settings.ecutrho:
        ratio = fft_inflation(cell, settings.ecutrho)
        if ratio > 2.0:
            findings.append(Finding('fft_inflation', run, 'FFT box %s, %.1f times the %s of a reduced basis'
                                    % (tuple(fft_box(cell, settings.ecutrho)), ratio,
                                       tuple(fft_box(reduced_basis(cell)[0], settings.ecutrho)))))  # fmt: skip
    if settings.kgrid:
        grid = 'x'.join(map(str, settings.kgrid))
        if displaced and not settings.nosym and cell is not None:
            try:
                rotations = lattice_rotations(cell)
            except ImportError:
                rotations = None
            if rotations is not None and not grid_is_invariant(settings.kgrid, rotations):
                findings.append(Finding('kpoint_inflation', run, 'grid %s is not invariant under the lattice group: '
                                        'without nosym pw.x adds off-grid points (with nosym: %d k-points)'
                                        % (grid, time_reversal_kpoints(settings.kgrid))))  # fmt: skip
        if interpolated and min(settings.kgrid) < 3:
            findings.append(
                Finding(
                    'two_point_axis',
                    run,
                    'grid %s has an axis with %d point(s)' % (grid, min(settings.kgrid)),
                )
            )
        if cell is not None:
            length = shortest_superlattice_vector(cell, settings.kgrid)
            if length < floor_bohr - 1e-6:
                findings.append(Finding('coarse_kgrid', run, 'grid %s: shortest superlattice vector %.1f bohr < %.0f'
                                        % (grid, length, floor_bohr)))  # fmt: skip
        if qgrid is not None and any(k % q for k, q in zip(settings.kgrid, qgrid)):
            findings.append(Finding('qgrid_does_not_divide_kgrid', run, 'q-grid %s does not divide k-grid %s'
                                    % ('x'.join(map(str, qgrid)), grid)))  # fmt: skip
        if cores:
            kpoints = (
                time_reversal_kpoints(settings.kgrid) if (displaced or settings.nosym) else None
            )
            if kpoints:
                layout = balanced_pools(kpoints, cores, cores_per_pool, billing_granularity)
                best = layout['one_pool_per_kpoint']
                if layout['cost'] > 1.15 * best['cost']:
                    findings.append(Finding('pool_layout', run, '%d k-points on %d cores: %d pools, %d k-points per pool, '
                                            'billed %d; one pool per k-point: %d cores (billed %d), %.0f %% of the cost'
                                            % (kpoints, cores, layout['pools'], layout['kpoints_per_pool'],
                                               layout['billed'], best['cores'], best['billed'],
                                               100 * best['cost'] / layout['cost'])))  # fmt: skip
    if cores and billing_granularity > 1 and cores % billing_granularity:
        findings.append(Finding('billing_rounding', run, '%d cores requested, %d billed'
                                % (cores, billed_cores(cores, billing_granularity))))  # fmt: skip
    if metallic and (settings.atoms or 0) >= 40:
        plain = settings.mixing_mode in (None, 'plain')
        if plain and (settings.mixing_beta is None or settings.mixing_beta > 0.4):
            findings.append(Finding('plain_mixing_large_cell', run, '%d atoms, plain mixing, beta %s'
                                    % (settings.atoms, settings.mixing_beta if settings.mixing_beta else '0.7 (default)')))  # fmt: skip
    return findings
