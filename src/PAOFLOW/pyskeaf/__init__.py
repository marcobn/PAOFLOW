"""pyskeaf — Python port of the Supercell K-space Extremal Area Finder (SKEAF).

Reference: P.M.C. Rourke and S.R. Julian, Comput. Phys. Commun. 183, 324 (2012).
Original Fortran 90 source: skeaf_v1p3p0_r149.F90 by Patrick Rourke.

Public API (Phase 1 — I/O only):

    from pyskeaf import read_bxsf, BXSFData
    from pyskeaf import SkeafConfig, read_config_in, write_config_in
    from pyskeaf import SKEAFResult, Orbit
"""

from PAOFLOW.pyskeaf.constants import (
    BUILD_NUMBER,
    CONV_AU_TO_ANG,
    CONV_FSAREA_TO_KT,
    CONV_FSDADE_TO_MSTAR,
)
from PAOFLOW.pyskeaf.io_bxsf import BXSFData, read_bxsf
from PAOFLOW.pyskeaf.config import SkeafConfig, read_config_in, write_config_in
from PAOFLOW.pyskeaf.results import Orbit, SKEAFResult
from PAOFLOW.pyskeaf.runner import run_at_angle, run_angle_sweep, run_skeaf

__all__ = [
    "BUILD_NUMBER",
    "CONV_AU_TO_ANG",
    "CONV_FSAREA_TO_KT",
    "CONV_FSDADE_TO_MSTAR",
    "BXSFData",
    "read_bxsf",
    "SkeafConfig",
    "read_config_in",
    "write_config_in",
    "Orbit",
    "SKEAFResult",
    "run_at_angle",
    "run_angle_sweep",
    "run_skeaf",
]

__version__ = "0.1.0.dev0"
