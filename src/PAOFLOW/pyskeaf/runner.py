"""High-level driver that orchestrates a SKEAF run for one (θ, φ) angle.

Phase 4: ``run_at_angle`` glues slice / orbit / averaging modules together.
Phase 5: ``run_skeaf`` consumes a :class:`PAOFLOW.pyskeaf.config.SkeafConfig` directly,
including the ``hvd`` H-vector setup and the ``hvd == 'r'`` rotation sweep,
and writes all five Fortran-compatible output files.
Phase 6: MPI parallelism over the angle sweep when launched by Slurm or
``mpirun``, with optional :mod:`joblib` parallelism (``cfg.n_jobs``) outside
MPI, plus structured ``logging`` progress reports.

The Fortran main program around skeaf_v1p3p0_r149.F90 lines 1100–2950 is the
reference for the call sequence.
"""

from __future__ import annotations

import inspect
import logging
import re
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from PAOFLOW.pyskeaf._parallel import active_mpi_comm
from PAOFLOW.pyskeaf.config import SkeafConfig, read_config_in
from PAOFLOW.pyskeaf.constants import CONV_AU_TO_ANG
from PAOFLOW.pyskeaf.geometry import set_field_angle
from PAOFLOW.pyskeaf.io_bxsf import BXSFData, BXSFError, read_bxsf
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
    fermi_energy_filename_token,
    write_orbit_outlines,
    write_results_freqvsangle,
    write_results_long,
    write_results_short,
)
from PAOFLOW.pyskeaf.slice_ops import (
    SliceGeometry,
    build_slice,
    make_slice_geometry,
)

logger = logging.getLogger(__name__)

_RYDBERG_IN_EV = 13.605693122994
_PAOFLOW_BAND_STEM_RE = re.compile(r'^Fermi_surf_band_(\d+)$')

# One angle easily runs longer than loky's 300 s default idle timeout, so workers
# waiting on the stragglers of a sweep get reaped and the respawn race can leave
# a submitted angle without a result, hanging the run indefinitely.
_IDLE_WORKER_TIMEOUT = 86400.0


def _loky_backend_kwargs() -> dict:
    """Backend kwargs keeping idle loky workers alive, when joblib exposes them."""
    from joblib._parallel_backends import LokyBackend

    if 'idle_worker_timeout' in inspect.signature(LokyBackend.configure).parameters:
        return {'idle_worker_timeout': _IDLE_WORKER_TIMEOUT}
    return {}


@dataclass
class BXSFRun:
    """The calculation or skip outcome for one PAOFLOW BXSF file."""

    path: Path
    minimum_ev: float
    maximum_ev: float
    fermi_energy_ev: float
    result: SKEAFResult | None = None
    skipped_reason: str | None = None

    @property
    def calculated(self) -> bool:
        return self.result is not None


def _without_energy_grid(bxsf: BXSFData) -> BXSFData:
    """Return a copy of ``bxsf`` whose ``(nx, ny, nz)`` energy grid is dropped.

    Header metadata (filename, dimensions, reciprocal vectors) is preserved so
    result writers keep working, while a multi-file sweep no longer holds one
    full grid alive per completed band.
    """
    return replace(bxsf, energies=np.empty((0, 0, 0)))


def _run_mpi_angle_jobs(jobs, worker: Callable, comm):
    """Run indexed angle jobs across MPI ranks and return them in input order.

    Only the per-angle orbit lists cross MPI.  In particular, the full BXSF
    grid attached to :class:`SKEAFResult` is never gathered or broadcast.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_jobs = jobs[rank::size]
    local_results = []
    local_error = None
    try:
        local_results = [worker(job) for job in local_jobs]
    except Exception:
        local_error = traceback.format_exc()

    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = '\n\n'.join(
            f'MPI rank {worker_rank}:\n{error}'
            for worker_rank, error in enumerate(errors)
            if error is not None
        )
        raise RuntimeError(f'pyskeaf MPI angle calculation failed:\n{details}')

    gathered = comm.gather(local_results, root=0)
    ordered = None
    assembly_error = None
    if rank == 0:
        try:
            ordered = [item for rank_results in gathered for item in rank_results]
            ordered.sort(key=lambda item: item[0])
            indices = [item[0] for item in ordered]
            expected = [job[0] for job in jobs]
            if indices != expected:
                raise RuntimeError(
                    f'MPI angle result indices {indices!r} do not match expected {expected!r}.'
                )
        except Exception:
            assembly_error = traceback.format_exc()

    assembly_error = comm.bcast(assembly_error, root=0)
    if assembly_error is not None:
        raise RuntimeError(f'pyskeaf MPI result assembly failed:\n{assembly_error}')
    return comm.bcast(ordered, root=0)


def _raise_mpi_output_errors(comm, local_exception, local_traceback) -> None:
    """Synchronize output failures so secondary ranks do not hang silently."""
    errors = comm.allgather(local_traceback)
    if not any(error is not None for error in errors):
        return
    if local_exception is not None:
        raise local_exception
    details = '\n\n'.join(
        f'MPI rank {rank}:\n{error}' for rank, error in enumerate(errors) if error is not None
    )
    raise RuntimeError(f'pyskeaf MPI output writing failed:\n{details}')


def run_at_angle(
    bxsf: BXSFData,
    theta: float,
    phi: float,
    *,
    numint: int,
    fermi_energy: float | None = None,
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
    In a multi-rank MPI launch this is a collective call: every rank must call
    it, and the angles are divided among ``MPI.COMM_WORLD``.
    """
    all_orbits: list[Orbit] = []
    fe = kwargs.get('fermi_energy', bxsf.fermi_energy)
    jobs = [
        (index, float(theta), float(phi))
        for index, (theta, phi) in enumerate(np.asarray(angles_rad, dtype=float))
    ]

    def _one(job):
        index, theta, phi = job
        result = run_at_angle(bxsf, theta, phi, numint=numint, **kwargs)
        return index, result.orbits

    comm = active_mpi_comm()
    indexed_results = (
        [_one(job) for job in jobs] if comm is None else _run_mpi_angle_jobs(jobs, _one, comm)
    )
    for _, angle_orbits in indexed_results:
        all_orbits.extend(angle_orbits)
    return SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=fe,
        orbits=all_orbits,
        angles=np.asarray(angles_rad, dtype=float),
        bxsf=bxsf,
    )


def run_skeaf(
    config: SkeafConfig | str | Path,
    bxsf: BXSFData | None = None,
    *,
    write_files: bool = True,
    output_dir: str | Path | None = None,
    output_suffix: str = '',
    write_auxiliary_files: bool = True,
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
    write_auxiliary_files
        Also write short, long, and orbit-outline diagnostic files. The
        frequency-vs-angle result is always written when ``write_files`` is true.

    Notes
    -----
    In a multi-rank MPI launch this is a collective call and must be entered by
    every rank. Angles are distributed across ``MPI.COMM_WORLD`` and the
    completed orbit list is returned on every rank; only rank 0 writes files.
    Outside MPI, ``config.n_jobs`` controls optional joblib multiprocessing.
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

    all_orbits: list[Orbit] = []
    n_angles = len(thetas)
    n_jobs = int(getattr(config, 'n_jobs', 1) or 1)
    comm = active_mpi_comm()
    mpi_rank = comm.Get_rank() if comm is not None else 0
    mpi_size = comm.Get_size() if comm is not None else 1
    if mpi_rank == 0:
        logger.info(
            'run_skeaf: %d angle(s), numint=%d, hvd=%r, MPI ranks=%d, n_jobs=%d',
            n_angles,
            config.numint,
            config.hvd,
            mpi_size,
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
        result = run_at_angle(
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
        return idx, result.orbits

    jobs = list(enumerate(zip(thetas, phis)))
    jobs = [(i, float(t), float(p)) for i, (t, p) in jobs]

    if comm is not None:
        if mpi_rank == 0:
            logger.info('Dispatching %d angle(s) across %d MPI ranks', n_angles, mpi_size)
            if n_jobs != 1:
                logger.warning(
                    'Ignoring config.n_jobs=%d under MPI to avoid nested process oversubscription.',
                    n_jobs,
                )
        results = _run_mpi_angle_jobs(jobs, _one, comm)
    elif n_jobs == 1 or n_angles == 1:
        results = [_one(j) for j in jobs]
    else:
        # Lazy import — joblib is already a hard dep but we keep the import
        # local so this module is cheap to import in single-thread mode.
        from joblib import Parallel, delayed

        logger.info('Dispatching %d angle(s) to %d joblib workers (loky backend)', n_angles, n_jobs)
        results = Parallel(
            n_jobs=n_jobs,
            backend='loky',
            batch_size=1,
            timeout=getattr(config, 'angle_timeout', None),
            **_loky_backend_kwargs(),
        )(delayed(_one)(j) for j in jobs)

    for _, angle_orbits in results:
        all_orbits.extend(angle_orbits)

    result = SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=config.fermi_energy,
        orbits=all_orbits,
        angles=np.column_stack([thetas, phis]),
        config=config,
        bxsf=bxsf,
    )

    output_exception = None
    output_traceback = None
    if write_files and mpi_rank == 0:
        try:
            out = Path(output_dir) if output_dir is not None else Path.cwd()
            out.mkdir(parents=True, exist_ok=True)
            suffix = f'_{output_suffix}' if output_suffix else ''
            energy_ev = result.fermi_energy * _RYDBERG_IN_EV
            prefix = f'qo_EF_{fermi_energy_filename_token(energy_ev)}_'
            write_results_freqvsangle(result, out / f'{prefix}freqvsangle{suffix}.out')
            if write_auxiliary_files:
                write_results_short(result, out / f'{prefix}short{suffix}.out')
                write_results_long(result, out / f'{prefix}long{suffix}.out')
                write_orbit_outlines(
                    result,
                    out / f'{prefix}orbitoutlines_invAng{suffix}.out',
                    out / f'{prefix}orbitoutlines_invau{suffix}.out',
                )
        except Exception as error:
            output_exception = error
            output_traceback = traceback.format_exc()
    if write_files and comm is not None:
        _raise_mpi_output_errors(comm, output_exception, output_traceback)
    elif output_exception is not None:
        raise output_exception
    return result


def run_paoflow_bxsf_files(
    config: SkeafConfig | str | Path,
    *,
    input_dir: str | Path,
    filenames: Sequence[str | Path] | None = None,
    all_files: bool = False,
    output_dir: str | Path | None = None,
    write_files: bool = True,
    write_auxiliary_files: bool = True,
    progress_callback: Callable[[BXSFRun], None] | None = None,
) -> list[BXSFRun]:
    """Run selected PAOFLOW BXSF bands whose energy ranges contain ``E_F``.

    PAOFLOW's SKEAF-compatible per-band BXSF grids use Rydberg, while the
    Fermi-energy field in ``config.in`` uses eV. :func:`read_config_in`
    converts the latter to internal Rydberg, so eligibility is tested in Ry
    and the reported band limits are converted to eV for users.
    Select arbitrary BXSF names with ``filenames`` or every ``*.bxsf`` in
    ``input_dir`` with ``all_files=True``. If neither is supplied, the legacy
    filename in ``config.in`` is used.

    Output files include the Fermi energy in eV and use the trailing numeric
    band index when present, e.g. ``qo_EF_0_short_1.out`` for
    ``Fermi_surf_band_1.bxsf`` at 0 eV. Arbitrary filenames use their complete
    stem, e.g. ``qo_EF_0_short_manual_name.out``.
    ``progress_callback`` is invoked after each band is calculated or skipped.
    The returned :class:`BXSFRun` results carry BXSF header metadata but not
    the band energy grid, so scanning many files does not accumulate them.
    """
    if not isinstance(config, SkeafConfig):
        config = read_config_in(config)
    if filenames is not None and all_files:
        raise ValueError('Choose explicit filenames or all_files, not both.')

    directory = Path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f'BXSF input directory does not exist: {directory}')
    if filenames is not None:
        paths = [directory / Path(name) for name in filenames]
    elif all_files:
        paths = sorted(directory.glob('*.bxsf'))
    else:
        paths = [directory / config.filename]
    if not paths:
        raise FileNotFoundError(f'No BXSF files found in {directory}.')

    fermi_energy_ev = config.fermi_energy * _RYDBERG_IN_EV
    results: list[BXSFRun] = []

    def record(item: BXSFRun) -> None:
        results.append(item)
        if progress_callback is not None:
            progress_callback(item)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'BXSF file not found: {path}')
        bxsf = None  # release the previous band's grid before reading the next
        try:
            bxsf = read_bxsf(path)
        except BXSFError as error:
            record(
                BXSFRun(
                    path,
                    float('nan'),
                    float('nan'),
                    fermi_energy_ev,
                    skipped_reason=str(error),
                )
            )
            continue
        minimum_ry = float(np.min(bxsf.energies))
        maximum_ry = float(np.max(bxsf.energies))
        minimum_ev = minimum_ry * _RYDBERG_IN_EV
        maximum_ev = maximum_ry * _RYDBERG_IN_EV
        item = BXSFRun(path, minimum_ev, maximum_ev, fermi_energy_ev)
        if not minimum_ry <= config.fermi_energy <= maximum_ry:
            item.skipped_reason = (
                f'Fermi energy {fermi_energy_ev:.6f} eV is outside '
                f'[{minimum_ev:.6f}, {maximum_ev:.6f}] eV.'
            )
            record(item)
            continue

        standard_match = _PAOFLOW_BAND_STEM_RE.fullmatch(path.stem)
        item.result = run_skeaf(
            config,
            bxsf,
            write_files=write_files,
            output_dir=output_dir,
            output_suffix=standard_match.group(1) if standard_match else path.stem,
            write_auxiliary_files=write_auxiliary_files,
        )
        item.result.bxsf = _without_energy_grid(bxsf)
        record(item)
    bxsf = None
    return results
