"""Launch new pyskeaf rotations into fresh folders (never overwriting earlier runs)."""

from __future__ import annotations

import math
from pathlib import Path

from .skeaf_output import RYDBERG_IN_EV, find_bxsf_files, plain_bxsf_path


def skeaf_rotation(
    bxsf_dir: str | Path,
    destination: str | Path,
    bands: list[int],
    azimuthal: tuple[float, float],
    polar: tuple[float, float],
    num_angles: int,
    num_interpolation: int,
    fermi_energy: float = 0.0,
    frequency_tolerance: float = 0.05,
    orbit_tolerance: float = 0.1,
    allow_wall_orbits: bool = True,
    n_jobs: int | None = -1,
) -> list[Path]:
    """Run pyskeaf over a rotation for selected bands, writing into a new folder.

    Parameters
    ----------
    bxsf_dir : str or Path
        Folder with ``Fermi_surf_band_<n>.bxsf[.gz]``.
    destination : str or Path
        Output folder. It must not already contain frequency files: pyskeaf writes
        fixed file names, and silently replacing an earlier run destroys the
        reference needed for convergence checks.
    bands : list[int]
        Bands to compute.
    azimuthal, polar : tuple of float
        Start and end angles (deg); equal values keep that angle fixed.
    num_angles : int
        Number of field directions in the rotation.
    num_interpolation : int
        pyskeaf ``num_interpolation``.
    fermi_energy : float, optional
        Fermi energy in eV (relative to the BXSF reference).
    frequency_tolerance, orbit_tolerance : float, optional
        pyskeaf averaging tolerances.
    allow_wall_orbits : bool, optional
        Keep orbits near supercell walls (filter later with the copy count).
    n_jobs : int or None, optional
        joblib workers for the angle sweep (clipped to the core budget by pyskeaf).

    Returns
    -------
    list[Path]
        The ``qo_EF_<E>_freqvsangle_<band>.out`` files written. Compressed BXSF inputs
        are decompressed into ``destination/bxsf_cache``.

    Raises
    ------
    FileExistsError
        If ``destination`` already holds frequency files.
    """
    from PAOFLOW.pyskeaf.config import SkeafConfig
    from PAOFLOW.pyskeaf.runner import run_paoflow_bxsf_files

    destination = Path(destination)
    if destination.exists() and any(destination.glob('qo_EF_*_freqvsangle_*.out*')):
        raise FileExistsError(f'{destination} already contains pyskeaf output; choose a new folder')
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / 'bxsf_cache'
    sources = find_bxsf_files(bxsf_dir)
    plain = {band: plain_bxsf_path(sources[band], cache) for band in bands}
    input_dirs = {path.parent for path in plain.values()}
    if len(input_dirs) != 1:
        for band, path in plain.items():
            target = cache / path.name
            if path != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                plain[band] = target
    config = SkeafConfig(
        fermi_energy=fermi_energy / RYDBERG_IN_EV,
        numint=num_interpolation,
        hvd='r',
        theta_start=math.radians(azimuthal[0]),
        theta_end=math.radians(azimuthal[1]),
        phi_start=math.radians(polar[0]),
        phi_end=math.radians(polar[1]),
        num_rots=num_angles,
        freq_same_frac=frequency_tolerance,
        avg_same_frac=orbit_tolerance,
        allow_ext_near_walls=allow_wall_orbits,
        n_jobs=n_jobs,
    )
    run_paoflow_bxsf_files(config, input_dir=next(iter(plain.values())).parent,
                           filenames=[path.name for path in plain.values()], output_dir=destination,
                           write_auxiliary_files=False)  # fmt: skip
    return sorted(destination.glob('qo_EF_*_freqvsangle_*.out'))
