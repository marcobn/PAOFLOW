"""Dielectric-function metric extraction for PAOFLOW vs QE benchmarking.

Helper utilities used by the Phase 1 / Phase 4 harness around the non-local
pseudopotential velocity correction.  See ``TODOs/nonlocal_velocity_correction.md``.

Supports two file formats:

* **PAOFLOW** — two whitespace-separated columns ``energy[eV]  value``,
  no header.  One file per Cartesian channel (``epsi_xx.dat``,
  ``epsi_yy.dat``, ``epsi_zz.dat``, plus matching ``epsr_*`` and
  ``eels_*``).
* **QE epsilon.x** — four columns ``energy  val_x  val_y  val_z`` with
  ``#`` comment lines.  One file per quantity
  (``epsi_<prefix>.dat`` etc.).

The metrics extracted here are the minimum needed to detect movement of
optical features when the non-local velocity correction is added:

* ``static`` — value at ``omega → 0`` (linear extrapolation of the two
  lowest-energy points).
* ``peak`` — (energy, height) of the largest local maximum within an
  energy window.
* ``mean_in_window`` — average over an energy window (good proxy for
  oscillator-strength weight in a band of interest).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrum:
    energy: np.ndarray  # (n,)  eV
    values: np.ndarray  # (n, 3)  per-Cartesian-channel value


def _load_columns(path: Path) -> np.ndarray:
    """Load whitespace-separated numeric columns, ignoring ``#`` comments."""
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def load_paoflow_spectrum(outdir: Path, basename: str) -> Spectrum:
    """Load a PAOFLOW dielectric quantity from per-Cartesian-channel files.

    Parameters
    ----------
    outdir
        Directory containing ``<basename>_xx.dat``, ``<basename>_yy.dat``,
        ``<basename>_zz.dat``.
    basename
        File-name prefix (``"epsi"``, ``"epsr"``, ``"eels"`` or ``"ieps"``).
    """
    outdir = Path(outdir)
    cols = []
    energy = None
    for axis in ('xx', 'yy', 'zz'):
        path = outdir / f'{basename}_{axis}.dat'
        if not path.exists():
            raise FileNotFoundError(path)
        data = _load_columns(path)
        if energy is None:
            energy = data[:, 0]
        elif not np.allclose(energy, data[:, 0]):
            raise ValueError(
                f'{path}: energy grid disagrees with {basename}_xx.dat '
                f'(first mismatch at index {int(np.argmax(energy != data[:, 0]))})'
            )
        cols.append(data[:, 1])
    return Spectrum(energy=np.asarray(energy), values=np.stack(cols, axis=1))


def load_qe_spectrum(path: Path) -> Spectrum:
    """Load a QE ``epsilon.x`` output file (4 columns: energy + x/y/z)."""
    data = _load_columns(Path(path))
    if data.shape[1] < 4:
        raise ValueError(
            f'{path}: expected at least 4 columns (energy, x, y, z), got {data.shape[1]}'
        )
    return Spectrum(energy=data[:, 0], values=data[:, 1:4])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _channel_average(s: Spectrum) -> np.ndarray:
    return s.values.mean(axis=1)


def static_value(s: Spectrum) -> float:
    """Linear-extrapolated value at ``omega = 0`` from the two lowest energies."""
    avg = _channel_average(s)
    e = s.energy
    if e.size < 2:
        raise ValueError('need at least 2 grid points to extrapolate')
    slope = (avg[1] - avg[0]) / (e[1] - e[0])
    return float(avg[0] - slope * e[0])


def peak_in_window(s: Spectrum, emin: float, emax: float) -> tuple[float, float]:
    """Return (energy, height) of the largest local maximum in [emin, emax].

    Falls back to the global argmax inside the window if no interior
    local maximum exists.
    """
    avg = _channel_average(s)
    mask = (s.energy >= emin) & (s.energy <= emax)
    if not mask.any():
        raise ValueError(f'no grid points inside [{emin}, {emax}] eV')

    idx_in_window = np.where(mask)[0]
    e_w = s.energy[idx_in_window]
    v_w = avg[idx_in_window]

    # Strict-interior local maxima (require neighbours, also restricted to window).
    interior = []
    for i in range(1, v_w.size - 1):
        if v_w[i] > v_w[i - 1] and v_w[i] > v_w[i + 1]:
            interior.append(i)
    if interior:
        best = max(interior, key=lambda i: v_w[i])
    else:
        best = int(np.argmax(v_w))
    return float(e_w[best]), float(v_w[best])


def mean_in_window(s: Spectrum, emin: float, emax: float) -> float:
    """Trapezoid mean of the channel-averaged spectrum over ``[emin, emax]``."""
    avg = _channel_average(s)
    mask = (s.energy >= emin) & (s.energy <= emax)
    if mask.sum() < 2:
        raise ValueError(f'need at least 2 grid points inside [{emin}, {emax}] eV')
    e = s.energy[mask]
    v = avg[mask]
    return float(np.trapezoid(v, e) / (e[-1] - e[0]))


# ---------------------------------------------------------------------------
# Convenience: full metric bundle for a PAOFLOW / QE run pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DielectricMetrics:
    """A compact set of metrics for regression-testing the dielectric tensor."""

    eps1_static: float
    eps2_peak_energy: float
    eps2_peak_height: float
    eels_peak_energy: float  # plasmon position
    eels_peak_height: float

    def asdict(self) -> dict:
        return {
            'eps1_static': self.eps1_static,
            'eps2_peak_energy': self.eps2_peak_energy,
            'eps2_peak_height': self.eps2_peak_height,
            'eels_peak_energy': self.eels_peak_energy,
            'eels_peak_height': self.eels_peak_height,
        }


def metrics_from_paoflow_output(
    outdir: Path,
    *,
    eps2_peak_window: tuple[float, float],
    eels_peak_window: tuple[float, float],
) -> DielectricMetrics:
    eps1 = load_paoflow_spectrum(outdir, 'epsr')
    eps2 = load_paoflow_spectrum(outdir, 'epsi')
    eels = load_paoflow_spectrum(outdir, 'eels')

    eps2_e, eps2_h = peak_in_window(eps2, *eps2_peak_window)
    eels_e, eels_h = peak_in_window(eels, *eels_peak_window)

    return DielectricMetrics(
        eps1_static=static_value(eps1),
        eps2_peak_energy=eps2_e,
        eps2_peak_height=eps2_h,
        eels_peak_energy=eels_e,
        eels_peak_height=eels_h,
    )


def metrics_from_qe_outputs(
    *,
    epsr_path: Path,
    epsi_path: Path,
    eels_path: Path,
    eps2_peak_window: tuple[float, float],
    eels_peak_window: tuple[float, float],
) -> DielectricMetrics:
    eps1 = load_qe_spectrum(Path(epsr_path))
    eps2 = load_qe_spectrum(Path(epsi_path))
    eels = load_qe_spectrum(Path(eels_path))

    eps2_e, eps2_h = peak_in_window(eps2, *eps2_peak_window)
    eels_e, eels_h = peak_in_window(eels, *eels_peak_window)

    return DielectricMetrics(
        eps1_static=static_value(eps1),
        eps2_peak_energy=eps2_e,
        eps2_peak_height=eps2_h,
        eels_peak_energy=eels_e,
        eels_peak_height=eels_h,
    )
