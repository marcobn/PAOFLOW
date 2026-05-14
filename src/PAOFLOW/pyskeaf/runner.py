"""High-level driver that orchestrates a SKEAF run for one (θ, φ) angle.

Phase 4: ``run_at_angle`` glues slice / orbit / averaging modules together.
Phase 5: ``run_skeaf`` consumes a :class:`PAOFLOW.pyskeaf.config.SkeafConfig` directly,
including the ``hvd`` H-vector setup and the ``hvd == 'r'`` rotation sweep,
and writes all five Fortran-compatible output files.
Phase 6: optional :mod:`joblib` parallelism over the angle sweep
(``cfg.n_jobs``) plus structured ``logging`` progress reports.

The Fortran main program around skeaf_v1p3p0_r149.F90 lines 1100–2950 is the
reference for the call sequence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from PAOFLOW.pyskeaf.config import SkeafConfig, read_config_in
from PAOFLOW.pyskeaf.constants import CONV_AU_TO_ANG
from PAOFLOW.pyskeaf.geometry import set_field_angle
from PAOFLOW.pyskeaf.io_bxsf import BXSFData, read_bxsf
from PAOFLOW.pyskeaf.slice_ops import (
    SliceGeometry,
    build_slice,
    make_slice_geometry,
)
from PAOFLOW.pyskeaf.orbits import (
    AveragedOrbit,
    average_orbits,
    find_closed_orbits_in_slice,
    find_extremal,
    match_chunks,
)
from PAOFLOW.pyskeaf.results import (
    Orbit,
    SKEAFResult,
    write_orbit_outlines,
    write_results_freqvsangle,
    write_results_long,
    write_results_short,
)

logger = logging.getLogger(__name__)


def run_at_angle(
    bxsf: BXSFData,
    theta: float,
    phi: float,
    *,
    numint: int,
    fermi_energy: Optional[float] = None,
    min_freq_kT: float = 0.0,
    freq_same_frac: float = 0.01,
    avg_same_frac: float = 0.05,
    allow_near_walls: bool = False,
) -> SKEAFResult:
    """Run a single-angle SKEAF analysis.

    See module docstring for details.  Returns a :class:`SKEAFResult` whose
    ``orbits`` list contains one :class:`Orbit` per averaged extremum.
    """
    if fermi_energy is None:
        fermi_energy = bxsf.fermi_energy

    geom = make_slice_geometry(bxsf, numint, theta, phi)
    n_slices = geom.numx
    logger.info(
        'run_at_angle: theta=%.4f rad, phi=%.4f rad, numint=%d (%d slices)',
        theta,
        phi,
        numint,
        n_slices,
    )

    per_slice_orbits = []
    log_every = max(1, n_slices // 10)
    for s in range(1, n_slices + 1):
        sl = build_slice(bxsf, geom, s)
        orbs = find_closed_orbits_in_slice(sl, fermi_energy=fermi_energy)
        per_slice_orbits.append(orbs)
        if logger.isEnabledFor(logging.DEBUG) and (s % log_every == 0 or s == n_slices):
            logger.debug('  slice %d/%d (%d orbits)', s, n_slices, len(orbs))

    chunks = match_chunks(per_slice_orbits)
    extrema = find_extremal(
        chunks,
        geom,
        min_freq_kT=min_freq_kT,
        allow_near_walls=allow_near_walls,
    )
    averaged = average_orbits(
        extrema,
        freq_same_frac=freq_same_frac,
        avg_same_frac=avg_same_frac,
    )
    logger.info(
        '  → %d chunk(s), %d extremum/extrema, %d averaged orbit(s)',
        len(chunks),
        len(extrema),
        len(averaged),
    )

    orbits = [_averaged_to_orbit(a, theta, phi, geom, bxsf) for a in averaged]
    return SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=fermi_energy,
        orbits=orbits,
        angles=np.array([[theta, phi]]),
        bxsf=bxsf,
    )


def _averaged_to_orbit(
    av: AveragedOrbit, theta: float, phi: float, geom: SliceGeometry, bxsf: BXSFData
) -> Orbit:
    """Convert an :class:`AveragedOrbit` to a public :class:`Orbit`.

    Computes the BZ-frame Cartesian outline (in Å⁻¹ and a.u.⁻¹) of the
    representative copy, shifted by its centroid's nearest-RUC-vector offset
    so that consumers see the orbit clustered near the RUC origin.
    """
    rep = av.representative
    rep_orbit = rep.slice_orbit
    out_ang, out_au = _compute_outline(rep_orbit, geom, bxsf)
    return Orbit(
        theta=theta,
        phi=phi,
        frequency_kT=av.frequency_kT,
        freq_uncertainty_kT=av.frequency_std_kT,
        effective_mass=av.effective_mass,
        effective_mass_std=av.effective_mass_std,
        curvature=av.curvature_kT_A2,
        curvature_std=av.curvature_std_kT_A2,
        orbit_type=int(round(av.orbit_type)) if abs(av.orbit_type) >= 0.5 else 0,
        orbit_type_std=av.orbit_type_std,
        num_copies=av.num_copies,
        avg_coords_ruc=av.avg_xyz_ruc,
        avg_coords_ruc_std=av.avg_xyz_ruc_std,
        rep_contour_xy=rep_orbit.contour_xy,
        rep_slice_index=rep_orbit.slice_index,
        rep_frequency_kT=rep_orbit.frequency_kT,
        outline_ang=out_ang,
        outline_au=out_au,
    )


def _compute_outline(
    slice_orbit, geom: SliceGeometry, bxsf: BXSFData
) -> tuple[np.ndarray, np.ndarray]:
    """Build the BZ-frame Cartesian outline shifted by RUC nearest-vector offset.

    Returns ``(outline_ang, outline_au)`` where each is shape ``(npts, 3)``.
    Mirrors the centroid-based shifting at Fortran F90:990–1015: convert the
    representative orbit's centroid to RUC fractional, take the nearest
    integer to define the subtractor, then for each contour point convert,
    subtract the offset, and rotate back to BZ-frame Cartesian.

    Both outputs are in Å⁻¹ (since ``bxsf.recip_ang`` carries the 2π factor),
    with ``outline_au`` simply scaled by ``CONV_AU_TO_ANG``.
    """
    R = geom.rotation
    M = geom.maxlreciplat
    n_slices = geom.numx
    z_frac = (slice_orbit.slice_index - 1) / (n_slices - 1)

    # Centroid in slicing-frame Å⁻¹.
    cx_slc = (4.0 * slice_orbit.avg_xy_frac[0] - 1.0) * M
    cy_slc = (4.0 * slice_orbit.avg_xy_frac[1] - 1.0) * M
    cz_slc = (4.0 * z_frac - 1.0) * M
    # → BZ-frame Cartesian Å⁻¹
    centroid_bz = R @ np.array([cx_slc, cy_slc, cz_slc])
    centroid_ruc_frac = geom.plr_inverse @ centroid_bz
    subtractor = np.round(centroid_ruc_frac)
    shift_bz = bxsf.recip_ang.T @ subtractor  # in Å⁻¹

    # Contour in slicing-frame; promote to (n, 3) by inserting z'.
    xy = slice_orbit.contour_xy  # (n, 2)
    n = xy.shape[0]
    pts3 = np.column_stack([xy, np.full(n, cz_slc)])  # (n, 3)
    bz_pts = pts3 @ R.T  # (n, 3) Å⁻¹
    bz_pts -= shift_bz[None, :]
    outline_ang = bz_pts.copy()
    outline_au = bz_pts * CONV_AU_TO_ANG
    return outline_ang, outline_au


def run_angle_sweep(
    bxsf: BXSFData,
    angles_rad: np.ndarray,
    *,
    numint: int,
    **kwargs,
) -> SKEAFResult:
    """Convenience wrapper: call :func:`run_at_angle` for each row of ``angles_rad``.

    ``angles_rad`` has shape ``(n_angles, 2)`` with columns ``(theta, phi)``.
    Returns a single :class:`SKEAFResult` whose ``orbits`` list is the
    concatenation of all per-angle orbits, in the order the angles were given.
    """
    all_orbits: List[Orbit] = []
    fe = kwargs.get('fermi_energy', bxsf.fermi_energy)
    for theta, phi in angles_rad:
        r = run_at_angle(bxsf, float(theta), float(phi), numint=numint, **kwargs)
        all_orbits.extend(r.orbits)
    return SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=fe,
        orbits=all_orbits,
        angles=np.asarray(angles_rad, dtype=float),
        bxsf=bxsf,
    )


def run_skeaf(
    config: Union[SkeafConfig, str, Path],
    bxsf: Optional[BXSFData] = None,
    *,
    write_files: bool = True,
    output_dir: Union[str, Path, None] = None,
) -> SKEAFResult:
    """Top-level driver: run SKEAF as defined by ``config``.

    Parameters
    ----------
    config
        Either a :class:`SkeafConfig` instance or a path to a ``config.in``
        file.
    bxsf
        Pre-loaded BXSF dataset.  If ``None``, loaded from ``config.filename``.
    write_files
        When True (default) the five Fortran-compatible output files are
        written to ``output_dir`` (cwd if not given).
    output_dir
        Destination directory for output files.  Created if missing.
    """
    if not isinstance(config, SkeafConfig):
        config = read_config_in(config)
    if bxsf is None:
        bxsf = read_bxsf(config.filename)

    # Resolve angle list from the hvd selector.
    if config.hvd == 'r':
        n = max(2, int(config.num_rots))
        thetas = np.linspace(config.theta_start, config.theta_end, n)
        phis = np.linspace(config.phi_start, config.phi_end, n)
    else:
        # set_field_angle returns the (theta, phi) for hvd in {a,b,c,n,r};
        # for non-'r' it produces a single angle.
        theta_one, phi_one = set_field_angle(
            bxsf.recip_ang,
            config.hvd,
            theta=config.theta,
            phi=config.phi,
        )
        thetas = np.array([theta_one])
        phis = np.array([phi_one])

    all_orbits: List[Orbit] = []
    n_angles = len(thetas)
    n_jobs = int(getattr(config, 'n_jobs', 1) or 1)
    logger.info(
        'run_skeaf: %d angle(s), numint=%d, hvd=%r, n_jobs=%d',
        n_angles,
        config.numint,
        config.hvd,
        n_jobs,
    )

    def _one(idx_theta_phi):
        idx, theta, phi = idx_theta_phi
        logger.info(
            'Angle %d/%d: theta=%.4f rad (%.3f deg), phi=%.4f rad (%.3f deg)',
            idx + 1,
            n_angles,
            theta,
            np.degrees(theta),
            phi,
            np.degrees(phi),
        )
        return run_at_angle(
            bxsf,
            float(theta),
            float(phi),
            numint=config.numint,
            fermi_energy=config.fermi_energy,
            min_freq_kT=config.min_extfreq,
            freq_same_frac=config.freq_same_frac,
            avg_same_frac=config.avg_same_frac,
            allow_near_walls=config.allow_ext_near_walls,
        )

    jobs = list(enumerate(zip(thetas, phis)))
    jobs = [(i, float(t), float(p)) for i, (t, p) in jobs]

    if n_jobs == 1 or n_angles == 1:
        results = [_one(j) for j in jobs]
    else:
        # Lazy import — joblib is already a hard dep but we keep the import
        # local so this module is cheap to import in single-thread mode.
        from joblib import Parallel, delayed

        logger.info('Dispatching %d angle(s) to %d joblib workers (loky backend)', n_angles, n_jobs)
        results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_one)(j) for j in jobs)

    for r in results:
        all_orbits.extend(r.orbits)

    result = SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=config.fermi_energy,
        orbits=all_orbits,
        angles=np.column_stack([thetas, phis]),
        config=config,
        bxsf=bxsf,
    )

    if write_files:
        out = Path(output_dir) if output_dir is not None else Path.cwd()
        out.mkdir(parents=True, exist_ok=True)
        write_results_freqvsangle(result, out / 'results_freqvsangle.out')
        write_results_short(result, out / 'results_short.out')
        write_results_long(result, out / 'results_long.out')
        write_orbit_outlines(
            result,
            out / 'results_orbitoutlines_invAng.out',
            out / 'results_orbitoutlines_invau.out',
        )
    return result
