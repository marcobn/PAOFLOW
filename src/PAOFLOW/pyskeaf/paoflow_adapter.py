"""Adapter glue for embedding pyskeaf inside `PAOFLOW <https://github.com/marcobn/PAOFLOW>`_.

PAOFLOW produces band energies on a regular k-grid in memory; this module
lets a caller feed those arrays straight into :func:`PAOFLOW.pyskeaf.run_skeaf`
without writing a BXSF file to disk first.

Two entry points are exposed:

* :func:`bxsf_from_arrays` — build a :class:`PAOFLOW.pyskeaf.io_bxsf.BXSFData`
  in memory from a NumPy energy array + reciprocal lattice, with optional
  unit conversion.
* :func:`run_from_paoflow` — convenience: build the BXSFData and immediately
  call :func:`PAOFLOW.pyskeaf.run_skeaf` for one band.

Conventions
-----------
SKEAF (and therefore pyskeaf) expects energies in **Rydberg** and reciprocal
lattice vectors in **a.u.⁻¹ without the 2π factor** — the BXSF "General Grid"
convention.  PAOFLOW commonly works in **eV** and **Å⁻¹ with the 2π factor**
(``b = 2π · a*``).  The ``energy_unit`` and ``recip_unit`` parameters of
:func:`bxsf_from_arrays` handle the conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from PAOFLOW.pyskeaf.constants import CONV_AU_TO_ANG, PI
from PAOFLOW.pyskeaf.io_bxsf import BXSFData
from PAOFLOW.pyskeaf.results import SKEAFResult
from PAOFLOW.pyskeaf.runner import run_skeaf

# 1 Rydberg in electron-volts.  CODATA 2018 value, matching what PAOFLOW uses.
RYDBERG_IN_EV = 13.605693122994

EnergyUnit = str   # 'Ry' | 'Ha' | 'eV'
RecipUnit = str    # 'au' | 'au_2pi' | 'ang' | 'ang_2pi'


def _to_rydberg(energies: np.ndarray, unit: EnergyUnit) -> np.ndarray:
    """Convert a band-energy array to Rydberg from one of the supported units."""
    u = unit.lower()
    if u in ("ry", "rydberg"):
        return energies
    if u in ("ha", "hartree"):
        return energies * 2.0
    if u in ("ev",):
        return energies / RYDBERG_IN_EV
    raise ValueError(f"Unknown energy_unit {unit!r}; expected 'Ry', 'Ha', or 'eV'.")


def _to_recip_au_no2pi(recip: np.ndarray, unit: RecipUnit) -> np.ndarray:
    """Convert reciprocal lattice vectors to (a.u.)⁻¹ **without** the 2π factor.

    Supported input units:

    * ``'au'``       — already (a.u.)⁻¹, no 2π (the BXSF / SKEAF convention).
    * ``'au_2pi'``   — (a.u.)⁻¹ with the 2π factor (Quantum ESPRESSO ``bg`` in
      units of ``2π/alat`` after multiplying by ``2π/alat``).
    * ``'ang'``      — Å⁻¹ without the 2π factor.
    * ``'ang_2pi'``  — Å⁻¹ with the 2π factor (PAOFLOW default; matches the
      ``recip_ang`` field of :class:`PAOFLOW.pyskeaf.io_bxsf.BXSFData`).
    """
    u = unit.lower()
    if u in ("au", "bohr", "au_no2pi"):
        return recip
    if u in ("au_2pi", "au2pi"):
        return recip / (2.0 * PI)
    if u in ("ang", "angstrom", "a", "ang_no2pi"):
        return recip * CONV_AU_TO_ANG
    if u in ("ang_2pi", "ang2pi", "paoflow"):
        return recip * CONV_AU_TO_ANG / (2.0 * PI)
    raise ValueError(
        f"Unknown recip_unit {unit!r}; expected 'au', 'au_2pi', 'ang', 'ang_2pi'."
    )


def bxsf_from_arrays(
    energies: np.ndarray,
    recip: np.ndarray,
    fermi_energy: float,
    *,
    energy_unit: EnergyUnit = "Ry",
    recip_unit: RecipUnit = "au",
    fermi_unit: Optional[EnergyUnit] = None,
    filename: str = "paoflow_in_memory",
    origin: Optional[Sequence[float]] = None,
    enforce_periodic_endpoint: bool = True,
) -> BXSFData:
    """Build a :class:`BXSFData` from in-memory PAOFLOW arrays.

    Parameters
    ----------
    energies : ndarray, shape (nx, ny, nz)
        Band energies on a regular k-grid spanning one reciprocal unit cell.
        SKEAF's BXSF format requires the endpoint to be repeated so that
        ``E[-1, :, :] == E[0, :, :]`` along every axis.  If
        ``enforce_periodic_endpoint`` is True (default) and an ``(nx,ny,nz)``
        grid is given without that repetition (i.e. ``E.shape`` matches
        the k-grid one-to-one with no wrap-around), a wrap-around copy is
        appended along each axis.  Pass an array that *already* repeats the
        endpoint and set this flag to False to skip the copy.
    recip : ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows; units selected by ``recip_unit``.
    fermi_energy : float
        Fermi energy in the units selected by ``fermi_unit`` (defaults to
        ``energy_unit``).
    energy_unit : {'Ry', 'Ha', 'eV'}, default 'Ry'
        Unit of ``energies``.
    recip_unit : {'au', 'au_2pi', 'ang', 'ang_2pi'}, default 'au'
        Unit of ``recip``.  See :func:`_to_recip_au_no2pi` for details.
        ``'ang_2pi'`` (alias ``'paoflow'``) is the typical PAOFLOW format.
    fermi_unit : same options as ``energy_unit``, optional
        Unit of ``fermi_energy``.  Defaults to ``energy_unit``.
    filename : str
        Informational tag stored on the BXSFData (used in output headers).
    origin : sequence of 3 float, optional
        Origin of the k-grid in the same units as ``recip`` after conversion
        (defaults to (0, 0, 0); SKEAF requires the BXSF origin to be the
        Γ point).
    enforce_periodic_endpoint : bool, default True
        See ``energies`` above.

    Returns
    -------
    BXSFData
        Ready to pass to :func:`PAOFLOW.pyskeaf.run_skeaf` or any other pyskeaf API.
    """
    e = np.asarray(energies, dtype=float)
    if e.ndim != 3:
        raise ValueError(f"energies must be a 3D array, got shape {e.shape}")
    if enforce_periodic_endpoint:
        e = _wrap_periodic(e)

    e_ry = _to_rydberg(e, energy_unit)
    fe_ry = _to_rydberg(np.asarray([fermi_energy], dtype=float),
                        fermi_unit or energy_unit)[0]

    r = np.asarray(recip, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"recip must have shape (3, 3), got {r.shape}")
    recip_au = _to_recip_au_no2pi(r, recip_unit)
    recip_ang = 2.0 * PI * recip_au / CONV_AU_TO_ANG

    if origin is None:
        origin_arr = np.zeros(3)
    else:
        origin_arr = np.asarray(origin, dtype=float).reshape(3)
        if not np.allclose(origin_arr, 0.0):
            raise ValueError(
                "SKEAF requires the BXSF origin to be the Gamma point "
                "(0, 0, 0); got non-zero origin."
            )

    nx, ny, nz = e_ry.shape
    return BXSFData(
        filename=filename,
        fermi_energy=float(fe_ry),
        nx=int(nx), ny=int(ny), nz=int(nz),
        recip_au=recip_au,
        recip_ang=recip_ang,
        energies=np.ascontiguousarray(e_ry),
        origin_au=origin_arr,
    )


def _wrap_periodic(e: np.ndarray) -> np.ndarray:
    """Append a wrap-around copy along any axis whose endpoint isn't repeated.

    SKEAF expects ``E[nx-1, :, :] == E[0, :, :]`` along every axis (BXSF
    "General Grid" convention).  This helper appends a duplicate plane on
    every axis where the endpoint differs from the start, leaving the array
    untouched otherwise.
    """
    out = e
    for axis in range(3):
        first = np.take(out, indices=[0], axis=axis)
        last = np.take(out, indices=[-1], axis=axis)
        if not np.allclose(first, last):
            out = np.concatenate([out, first], axis=axis)
    return out


def run_from_paoflow(
    energies: np.ndarray,
    recip: np.ndarray,
    fermi_energy: float,
    *,
    numint: int,
    theta: float = 0.0,
    phi: float = 0.0,
    hvd: str = "n",
    energy_unit: EnergyUnit = "Ry",
    recip_unit: RecipUnit = "au",
    fermi_unit: Optional[EnergyUnit] = None,
    filename: str = "paoflow_in_memory",
    output_dir: Union[str, Path, None] = None,
    write_files: bool = False,
    config_overrides: Optional[dict] = None,
) -> SKEAFResult:
    """One-shot driver: PAOFLOW arrays → :class:`SKEAFResult`.

    Builds a :class:`BXSFData` in memory via :func:`bxsf_from_arrays`, then
    constructs a minimal :class:`PAOFLOW.pyskeaf.config.SkeafConfig` (using
    ``hvd='n'`` by default with the explicit ``theta``/``phi``) and calls
    :func:`PAOFLOW.pyskeaf.runner.run_skeaf`.

    For an angle sweep, populate ``config_overrides`` with
    ``{'hvd': 'r', 'theta_start': ..., 'theta_end': ..., 'phi_start': ...,
    'phi_end': ..., 'num_rots': ...}`` (angles in radians, matching
    :class:`SkeafConfig`).

    Parameters
    ----------
    energies, recip, fermi_energy, energy_unit, recip_unit, fermi_unit, filename
        Forwarded to :func:`bxsf_from_arrays`.
    numint : int
        Interpolation density (passed through to SKEAF).
    theta, phi : float
        Field angles in **radians**.  Used when ``hvd in {'n', 'r'}``.
    hvd : {'a', 'b', 'c', 'n', 'r'}
        H-vector selector; see :func:`PAOFLOW.pyskeaf.geometry.set_field_angle`.
    write_files, output_dir
        Forwarded to :func:`PAOFLOW.pyskeaf.runner.run_skeaf`; default is
        ``write_files=False`` (in-memory only — typical for embedding).
    config_overrides : dict, optional
        Field-name → value overrides applied to the
        :class:`PAOFLOW.pyskeaf.config.SkeafConfig` instance before running
        (e.g. ``{'min_extfreq': 0.05, 'freq_same_frac': 0.005}``).
    """
    from PAOFLOW.pyskeaf.config import SkeafConfig  # local to avoid cycles at import

    bxsf = bxsf_from_arrays(
        energies, recip, fermi_energy,
        energy_unit=energy_unit, recip_unit=recip_unit,
        fermi_unit=fermi_unit, filename=filename,
    )

    cfg = SkeafConfig(
        filename=filename,
        fermi_energy=bxsf.fermi_energy,
        numint=int(numint),
        theta=float(theta),
        phi=float(phi),
        hvd=hvd,
    )
    if config_overrides:
        for key, val in config_overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(
                    f"SkeafConfig has no field {key!r}; valid fields: "
                    f"{list(cfg.__dataclass_fields__)}"
                )
            setattr(cfg, key, val)

    return run_skeaf(cfg, bxsf,
                     write_files=write_files, output_dir=output_dir)


def run_from_paoflow_object(
    paoflow: Any,
    band_index: int,
    *,
    numint: int,
    theta: float = 0.0,
    phi: float = 0.0,
    hvd: str = "n",
    fermi_energy: Optional[float] = None,
    energy_unit: EnergyUnit = "eV",
    recip_unit: RecipUnit = "ang_2pi",
    **kwargs,
) -> SKEAFResult:
    """Run pyskeaf on one band of a PAOFLOW data container.

    The ``paoflow`` argument is duck-typed: this function looks for an
    attribute or dict key named ``'E_k'`` (or ``'E_kn'`` / ``'bands'``) of
    shape ``(..., n_bands)`` plus a reciprocal-lattice attribute named
    ``'b_vectors'`` (or ``'recip_lattice'`` / ``'b_lattice'``).  If your
    PAOFLOW build uses different names, call :func:`bxsf_from_arrays`
    directly.

    Parameters
    ----------
    paoflow
        An object or dict produced by PAOFLOW with the band-energy grid
        and reciprocal lattice attached.  See above for the names searched.
    band_index : int
        Which band (last axis of the energy array) to extract.
    fermi_energy : float, optional
        Fermi energy in ``energy_unit``.  If None, looks for an
        ``'E_F'`` / ``'fermi'`` / ``'fermi_energy'`` key on ``paoflow``.
    energy_unit, recip_unit
        Defaults match the PAOFLOW convention (eV + Å⁻¹ with 2π).
    **kwargs
        Passed through to :func:`run_from_paoflow`.
    """
    energies = _get_attr(paoflow, ("E_k", "E_kn", "bands"))
    energies = np.asarray(energies)
    if energies.ndim < 4:
        raise ValueError(
            f"Expected band-energy array with shape (nx, ny, nz, n_bands [, n_spin]); "
            f"got shape {energies.shape}."
        )
    band_arr = energies[..., band_index]
    if band_arr.ndim == 4:                        # spin-resolved → take spin 0
        band_arr = band_arr[..., 0]
    if band_arr.ndim != 3:
        raise ValueError(
            f"After indexing band {band_index}, expected a 3D array; "
            f"got shape {band_arr.shape}."
        )

    recip = np.asarray(_get_attr(paoflow, ("b_vectors", "recip_lattice", "b_lattice")))

    if fermi_energy is None:
        fermi_energy = float(_get_attr(paoflow, ("E_F", "fermi", "fermi_energy")))

    return run_from_paoflow(
        band_arr, recip, fermi_energy,
        numint=numint, theta=theta, phi=phi, hvd=hvd,
        energy_unit=energy_unit, recip_unit=recip_unit,
        filename=f"paoflow_band_{band_index}",
        **kwargs,
    )


def _get_attr(obj: Any, names: Sequence[str]) -> Any:
    """Return ``obj.<name>`` or ``obj[<name>]`` for the first matching name."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    raise AttributeError(
        f"PAOFLOW container is missing any of: {names!r}.  "
        f"Pass the arrays directly via run_from_paoflow() or bxsf_from_arrays()."
    )


__all__ = [
    "RYDBERG_IN_EV",
    "bxsf_from_arrays",
    "run_from_paoflow",
    "run_from_paoflow_object",
]

