"""``config.in`` round-trip support for pyskeaf.

The Fortran code reads ``config.in`` with fixed-format ``read`` statements
(see ``skeaf_v1p3p0_r149.F90`` lines 564–589) and writes it with fixed-width
``write`` statements (lines 874–891).  The tagline at the right of every line
is decorative — only the leading numeric/character field is parsed.  This
module preserves that layout, but uses eV for the Fermi-energy field and
converts it to Rydberg for pyskeaf's internal numerical routines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

RYDBERG_IN_EV = 13.605693122994


@dataclass
class SkeafConfig:
    """All user-tunable parameters that drive a SKEAF run.

    Angles are stored internally **in radians** (matching the Fortran globals
    after conversion); the ``config.in`` file always uses **degrees**.
    """

    filename: str = ''
    fermi_energy: float = 0.0  # internal unit: Ryd; config.in uses eV
    numint: int = 1  # interpolated points per single cell side
    theta: float = 0.0  # rad — used when hvd != 'r'
    phi: float = 0.0  # rad — used when hvd != 'r'
    hvd: str = 'a'  # one of {'a','b','c','n','r'}
    min_extfreq: float = 0.0  # kT
    freq_same_frac: float = 0.01  # 0–1
    avg_same_frac: float = 0.05  # 0–1 (fraction of RUC side length)
    allow_ext_near_walls: bool = False  # 'y' / 'n' in config.in
    theta_start: float = 0.0  # rad — auto-rotation
    theta_end: float = 0.0  # rad
    phi_start: float = 0.0  # rad
    phi_end: float = 0.0  # rad
    num_rots: int = 1  # number of angles in auto-rotation
    # Phase 6 — runtime knobs (not part of the Fortran config.in format).
    n_jobs: int = 1  # joblib workers for parallel angle sweeps
    angle_timeout: float | None = None  # seconds allowed per parallel angle


def _strip_field(line: str, width: int) -> str:
    """Return the first ``width`` characters of *line*, stripped of whitespace."""
    end = len(line)
    for marker in ('!', '|'):
        idx = line.find(marker)
        if idx >= 0:
            end = min(end, idx)
    field = line[:end].strip()
    return field if field else line[:width].strip()


def read_config_in(path: str | Path = 'config.in') -> SkeafConfig:
    """Parse a ``config.in`` file in the format produced by the Fortran SKEAF.

    The Fermi energy is converted from eV in the file to Rydberg internally.
    Angles are converted from degrees in the file to radians internally.
    """
    path = Path(path)
    with path.open('r') as fh:
        lines = fh.readlines()
    if len(lines) < 15:
        raise ValueError(
            f'{path}: expected at least 15 lines, got {len(lines)}.  Is this a SKEAF config.in?'
        )

    # Field widths match the Fortran read formats:
    #   A50, F12.6, I4, F10.6, F10.6, A1, F10.6, F7.3, F7.3, A1,
    #   F10.6, F10.6, F10.6, F10.6, I5
    cfg = SkeafConfig()
    cfg.filename = _strip_field(lines[0], 50)
    fermi_energy_ev = float(_strip_field(lines[1], 12))
    cfg.fermi_energy = fermi_energy_ev / RYDBERG_IN_EV
    cfg.numint = int(_strip_field(lines[2], 4))

    # Angles in degrees in the file.
    theta_deg = float(_strip_field(lines[3], 10))
    phi_deg = float(_strip_field(lines[4], 10))
    cfg.theta = math.radians(theta_deg)
    cfg.phi = math.radians(phi_deg)

    cfg.hvd = _strip_field(lines[5], 1) or 'a'
    cfg.min_extfreq = float(_strip_field(lines[6], 10))  # Fortran reads F10.6 here
    cfg.freq_same_frac = float(_strip_field(lines[7], 7))
    cfg.avg_same_frac = float(_strip_field(lines[8], 7))

    aewn = _strip_field(lines[9], 1).lower()
    cfg.allow_ext_near_walls = aewn == 'y'

    cfg.theta_start = math.radians(float(_strip_field(lines[10], 10)))
    cfg.theta_end = math.radians(float(_strip_field(lines[11], 10)))
    cfg.phi_start = math.radians(float(_strip_field(lines[12], 10)))
    cfg.phi_end = math.radians(float(_strip_field(lines[13], 10)))
    cfg.num_rots = int(_strip_field(lines[14], 5))

    return cfg


def write_config_in(cfg: SkeafConfig, path: str | Path = 'config.in') -> None:
    """Write *cfg* using the SKEAF fixed-width layout.

    ``cfg.fermi_energy`` is stored internally in Rydberg and converted to eV
    for the second line of ``config.in``.
    """
    path = Path(path)

    def _line(payload: str, comment: str, total: int = 53) -> str:
        # All Fortran writes pad the payload field to the same total width
        # (e.g. ``A50,2x`` → 52 chars; ``F12.6,40x`` → 52; ``I4,48x`` → 52).
        # We standardise on 52 chars of payload + "! comment" suffix.
        return f'{payload:<{total - 1}}! {comment}\n'

    rad2deg = 180.0 / math.pi
    aewn = 'y' if cfg.allow_ext_near_walls else 'n'

    with path.open('w') as fh:
        fh.write(_line(f'{cfg.filename:<50}', 'Filename (50 chars. max)'))
        fermi_energy_ev = cfg.fermi_energy * RYDBERG_IN_EV
        fh.write(_line(f'{fermi_energy_ev:12.6f}', 'Fermi energy (eV)'))
        fh.write(_line(f'{cfg.numint:4d}', 'Interpolated number of points per single side'))
        fh.write(_line(f'{cfg.theta * rad2deg:10.6f}', 'Theta (degrees)'))
        fh.write(_line(f'{cfg.phi * rad2deg:10.6f}', 'Phi (degrees)'))
        fh.write(_line(f'{cfg.hvd:1s}', 'H-vector direction'))
        fh.write(_line(f'{cfg.min_extfreq:8.4f}', 'Minimum extremal FS freq. (kT)'))
        fh.write(
            _line(
                f'{cfg.freq_same_frac:7.3f}',
                'Maximum fractional diff. between orbit freqs. for averaging',
            )
        )
        fh.write(
            _line(
                f'{cfg.avg_same_frac:7.3f}',
                'Maximum distance between orbit avg. coords. for averaging',
            )
        )
        fh.write(_line(f'{aewn:1s}', 'Allow extremal orbits near super-cell walls?'))
        fh.write(_line(f'{cfg.theta_start * rad2deg:10.6f}', 'Starting theta (degrees)'))
        fh.write(_line(f'{cfg.theta_end * rad2deg:10.6f}', 'Ending theta (degrees)'))
        fh.write(_line(f'{cfg.phi_start * rad2deg:10.6f}', 'Starting phi (degrees)'))
        fh.write(_line(f'{cfg.phi_end * rad2deg:10.6f}', 'Ending phi (degrees)'))
        fh.write(_line(f'{cfg.num_rots:5d}', 'Number of rotation angles'))
