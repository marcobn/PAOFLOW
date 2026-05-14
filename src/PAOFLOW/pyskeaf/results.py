"""Result dataclasses and output writers for PAOFLOW.pyskeaf.

The output filenames are kept identical to those produced by the Fortran
SKEAF (``results_freqvsangle.out``, ``results_short.out``,
``results_long.out``, ``results_orbitoutlines_invAng.out``,
``results_orbitoutlines_invau.out``) so that downstream plotting scripts
(e.g. ``Cylinder/plot.py``) keep working unmodified.

Phase 1 only defines the dataclasses and stubs the writers; full output
formatting will be filled in as later phases produce real orbit data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from PAOFLOW.pyskeaf.constants import BUILD_NUMBER, CONV_AU_TO_ANG


@dataclass
class Orbit:
    """A single extremal orbit on the Fermi surface at a fixed (θ, φ).

    Mirrors the per-orbit fields written by the Fortran ``sliceext`` and the
    final averaging step.
    """

    theta: float                # rad
    phi: float                  # rad
    frequency_kT: float         # dHvA frequency (kT)
    freq_uncertainty_kT: float  # +/- estimate (kT)
    effective_mass: float       # m* / m_e (dimensionless)
    effective_mass_std: float = 0.0
    curvature: float = 0.0      # d^2 A/dk_H^2 in kT · Å²
    curvature_std: float = 0.0
    orbit_type: int = 0         # +1 = electron, -1 = hole, 0 = ambiguous
    orbit_type_std: float = 0.0
    num_copies: int = 1         # number of equivalent copies found in the supercell
    avg_coords_ruc: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    avg_coords_ruc_std: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    # The largest single-copy slice orbit — used by the outline writer.
    # ``rep_contour_xy`` is in slicing-frame Å⁻¹ (closed, NOT periodic-duplicated),
    # ``rep_slice_index`` is the 1-based slice index it was found on, and
    # ``rep_frequency_kT`` is the frequency of that single copy (which may differ
    # slightly from the averaged ``frequency_kT``).
    rep_contour_xy: np.ndarray | None = None
    rep_slice_index: int = 0
    rep_frequency_kT: float = 0.0
    outline_au: np.ndarray | None = None   # (npts, 3) in a.u.^-1
    outline_ang: np.ndarray | None = None  # (npts, 3) in Å^-1


@dataclass
class SKEAFResult:
    """Top-level container for a complete SKEAF run."""

    config_filename: str
    bxsf_filename: str
    fermi_energy: float                # Ryd
    orbits: list[Orbit] = field(default_factory=list)
    band_volume: float | None = None   # fraction of BZ filled
    dos_at_ef: float | None = None     # states per Ryd per unit cell
    angles: np.ndarray | None = None   # (n_rot, 2) array of (theta, phi) in rad
    config: object | None = None       # the :class:`PAOFLOW.pyskeaf.config.SkeafConfig`
    bxsf: object | None = None         # the :class:`PAOFLOW.pyskeaf.io_bxsf.BXSFData`


# --- writers (stubs to be fleshed out in later phases) ----------------------


def write_results_freqvsangle(
    result: SKEAFResult, path: Union[str, Path] = "results_freqvsangle.out"
) -> None:
    """Write the CSV-style ``results_freqvsangle.out`` (Fortran-compatible).

    Columns: ``Theta(deg),Phi(deg),Freq(kT),mstar(me),Curv(kTA2),Type(+e-h),NumOrbCopy``.
    Numeric format matches Fortran ``write(19, '(F10.6, ",", F10.6, ",", ES14.6, "," ...))``
    at skeaf_v1p3p0_r149.F90:2701.
    """
    header = ("Theta(deg),Phi(deg),Freq(kT),mstar(me),"
              "Curv(kTA2),Type(+e-h),NumOrbCopy\n")
    rad2deg = 180.0 / np.pi
    lines = [header]
    for orb in result.orbits:
        theta_deg = orb.theta * rad2deg
        phi_deg = orb.phi * rad2deg
        # Fortran ES14.6 is 14-wide with 6 fractional digits in scientific form.
        # F10.6 is 10-wide fixed.  F6.3 is 6-wide fixed.  I5 is right-justified width 5.
        lines.append(
            f"{theta_deg:10.6f},{phi_deg:10.6f},"
            f"{orb.frequency_kT:14.6E},{orb.effective_mass:14.6E},"
            f"{orb.curvature:14.6E},{float(orb.orbit_type):6.3f},"
            f"{orb.num_copies:5d}\n"
        )
    Path(path).write_text("".join(lines))


def write_results_short(
    result: SKEAFResult, path: Union[str, Path] = "results_short.out"
) -> None:
    """Write the human-readable ``results_short.out`` (Fortran-compatible).

    Mirrors the per-angle block at skeaf_v1p3p0_r149.F90:2780–2810.  Header
    comes from :func:`_results_header`.  Orbits are written in the order they
    appear in ``result.orbits``; the writer detects angle changes by inspecting
    consecutive ``(theta, phi)`` pairs and emits an "ANGLE n of N" delimiter.
    """
    text = _results_header(result, brief=True) + _per_angle_blocks(
        result, include_outline_pointer=False
    )
    Path(path).write_text(text)


def write_results_long(
    result: SKEAFResult, path: Union[str, Path] = "results_long.out"
) -> None:
    """Write the verbose ``results_long.out`` (every orbit copy listed).

    Same structure as ``results_short.out`` but additionally records the
    representative-copy slice index and orbit number used for the outline file
    (Fortran F90:986–988).
    """
    text = _results_header(result, brief=False) + _per_angle_blocks(
        result, include_outline_pointer=True
    )
    Path(path).write_text(text)


def write_orbit_outlines(
    result: SKEAFResult,
    path_invang: Union[str, Path] = "results_orbitoutlines_invAng.out",
    path_invau: Union[str, Path] = "results_orbitoutlines_invau.out",
) -> None:
    """Write the orbit-outline coordinate files in both Å⁻¹ and a.u.⁻¹.

    Each orbit's representative contour is converted from slicing-frame Å⁻¹
    to BZ-frame Cartesian Å⁻¹ via the rotation matrix at that angle, then
    shifted by the centroid's nearest-RUC-lattice-vector offset (Fortran
    F90:990–1015).  The a.u.⁻¹ file is the Å⁻¹ file scaled by
    ``CONV_AU_TO_ANG`` (= 0.529177209).

    Notes
    -----
    The Fortran outline writer requires ``rep_contour_xy`` to have been
    populated by :func:`PAOFLOW.pyskeaf.runner.run_at_angle`; orbits without a
    contour are silently skipped.
    """
    lines_ang: list[str] = []
    lines_au: list[str] = []

    # Group orbits by (theta, phi) so each angle gets its own header.
    by_angle: dict[tuple[float, float], list[Orbit]] = {}
    for o in result.orbits:
        by_angle.setdefault((o.theta, o.phi), []).append(o)

    rad2deg = 180.0 / np.pi
    for (theta, phi), orbs in by_angle.items():
        n_with_contour = sum(1 for o in orbs if o.outline_ang is not None)
        header_ang = (f"Theta(deg) = {theta * rad2deg:10.6f} , "
                      f"Phi(deg) = {phi * rad2deg:10.6f} , "
                      f"Number of orbits = {n_with_contour:5d} , "
                      f"(Angstrom^-1) units\n")
        header_au = (f"Theta(deg) = {theta * rad2deg:10.6f} , "
                     f"Phi(deg) = {phi * rad2deg:10.6f} , "
                     f"Number of orbits = {n_with_contour:5d} , "
                     f"(a.u.^-1) units\n")
        lines_ang.append(header_ang)
        lines_au.append(header_au)

        for o in orbs:
            if o.outline_ang is None:
                continue
            n_pts = o.outline_ang.shape[0]
            block = (
                f"Slice   = {o.rep_slice_index:6d} , "
                f"Freq(kT, average of all copies)              = "
                f"{o.frequency_kT:8.4f}\n"
                f"Orbit # = {0:6d} , "
                f"Freq(kT, this largest copy used for outline) = "
                f"{o.rep_frequency_kT:8.4f}\n"
                f"Points  = {n_pts:6d}\n"
                f"  kx            ky            kz\n"
            )
            lines_ang.append(block)
            lines_au.append(block)
            for px, py, pz in o.outline_ang:
                lines_ang.append(
                    f" {px:13.6E} {py:13.6E} {pz:13.6E}\n"
                )
            assert o.outline_au is not None
            for px, py, pz in o.outline_au:
                lines_au.append(
                    f" {px:13.6E} {py:13.6E} {pz:13.6E}\n"
                )

    Path(path_invang).write_text("".join(lines_ang))
    Path(path_invau).write_text("".join(lines_au))


# --- internal header / per-angle helpers ------------------------------------


def _results_header(result: SKEAFResult, *, brief: bool) -> str:
    """Compose the shared header for ``results_short.out`` / ``results_long.out``.

    Tag ``brief`` is currently unused (both files share an identical header in
    Fortran F90:894–945) but is reserved for future divergence.
    """
    cfg = result.config
    bxsf = result.bxsf
    label = "Short" if brief else "Long"
    out = [f" {label} results file generated by S.K.E.A.F. {BUILD_NUMBER}\n",
           " \n"]
    if bxsf is not None:
        out.append(f" XCrysDen FS filename: {bxsf.filename:<50}\n")
        out.append(f" Fermi energy: {result.fermi_energy:12.6f} Ryd \n")
        out.append(f" Original     nx = {bxsf.nx:4d}  ny = {bxsf.ny:4d}"
                   f"  nz = {bxsf.nz:4d}\n")
    if cfg is not None:
        out.append(f" New      numint = {cfg.numint:4d}\n")
        rad2deg = 180.0 / np.pi
        if cfg.hvd == "r":
            out.append(f" H-vector direction: {cfg.hvd}   "
                       f"Number of auto rotated angles = {cfg.num_rots:5d}\n")
            out.append(f" Theta = {cfg.theta_start * rad2deg:10.6f} to "
                       f"{cfg.theta_end * rad2deg:10.6f} degrees;  "
                       f"Phi = {cfg.phi_start * rad2deg:10.6f} to "
                       f"{cfg.phi_end * rad2deg:10.6f} degrees \n")
        else:
            out.append(f" H-vector direction: {cfg.hvd}   "
                       f"Theta = {cfg.theta * rad2deg:10.6f} degrees   "
                       f"Phi = {cfg.phi * rad2deg:10.6f} degrees \n")
        out.append(f" Minimum extremal FS freq.: {cfg.min_extfreq:8.4f} kT \n")
        out.append(f" Maximum fractional diff. between orbit freqs. "
                   f"for averaging: {cfg.freq_same_frac:7.3f}\n")
        out.append(f" Maximum distance (fraction of RUC side length) "
                   f"between orbit avg. coords. for averaging: "
                   f"{cfg.avg_same_frac:7.3f}\n")
        if cfg.allow_ext_near_walls:
            out.append(" Extremal orbits near super-cell walls are ALLOWED "
                       "to be included in the output.\n")
        else:
            out.append(" Extremal orbits near super-cell walls are REJECTED "
                       "from the output.\n")
    if result.dos_at_ef is not None:
        out.append(f" DOS at E_F: {result.dos_at_ef:14.6E} states/Ryd/cell\n")
    if result.band_volume is not None:
        out.append(f" Band volume (fraction of BZ): {result.band_volume:8.6f}\n")
    out.append(" \n")
    return "".join(out)


def _per_angle_blocks(result: SKEAFResult, *,
                      include_outline_pointer: bool) -> str:
    """Group orbits by ``(theta, phi)`` and emit one block per angle.

    Each block opens with ``ANGLE n of N`` (matching Fortran F90:2787) and
    lists every orbit's frequency, mass, curvature, type and copies count
    with sample standard deviations.  The long file additionally records the
    "Orbit copy chosen for outline" pointer.
    """
    rad2deg = 180.0 / np.pi
    by_angle: list[tuple[tuple[float, float], list[Orbit]]] = []
    seen: dict[tuple[float, float], int] = {}
    for o in result.orbits:
        key = (o.theta, o.phi)
        if key not in seen:
            seen[key] = len(by_angle)
            by_angle.append((key, []))
        by_angle[seen[key]][1].append(o)

    n_total = len(by_angle)
    out: list[str] = [" \n", " Predicted dHvA frequencies: \n", " \n"]
    for i, ((theta, phi), orbs) in enumerate(by_angle, start=1):
        out.append(f" ANGLE {i:5d} of {n_total:5d}    "
                   f"theta = {theta * rad2deg:10.6f} degrees, "
                   f"phi = {phi * rad2deg:10.6f} degrees\n")
        for o in orbs:
            out.append(
                f"  Freq. = {o.frequency_kT:8.4f}+/-{o.freq_uncertainty_kT:8.4f}"
                f" kT, m* = {o.effective_mass:8.4f}+/-{o.effective_mass_std:8.4f}"
                f" m_e, Curv. = {o.curvature:12.4E}+/-{o.curvature_std:12.4E}"
                f" kT A^2, orbit (1=e,-1=h): "
                f"{float(o.orbit_type):4.1f}+/-{o.orbit_type_std:4.1f}\n"
            )
            ax, ay, az = o.avg_coords_ruc
            sx, sy, sz = o.avg_coords_ruc_std
            out.append(
                f"  Orbit copies found: {o.num_copies:5d}, "
                f"RUC avg coords: ({ax:8.4f}+/-{sx:8.4f},"
                f"{ay:8.4f}+/-{sy:8.4f},{az:8.4f}+/-{sz:8.4f})\n"
            )
            if include_outline_pointer and o.rep_contour_xy is not None:
                out.append(f"  Orbit copy chosen for outline: "
                           f"from slice {o.rep_slice_index:4d}\n")
            out.append(" \n")
    return "".join(out)
