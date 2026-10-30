"""Adapters: the Migdal-Eliashberg gaps of a system folder (isotropic and anisotropic).

``superconductivity/me.py`` writes ``output/me/migdal_eliashberg.npz`` (``temps``,
``gap0_imag`` / ``gap_pade`` / ``gap_acon`` in eV, ``Z0``, ``Tc_gap``, ``Tc_linear``,
``lin_temps`` / ``max_eigenvalue``, ``w_real`` and the per-temperature ``imag_<T>``,
``pade_<T>``, ``acon_<T>``, ``qdos_<T>`` tables); ``superconductivity/me_aniso.py`` writes
``output/me_aniso/migdal_eliashberg_aniso.npz`` (``gap0`` / ``gap_edge`` per Fermi-surface
state in eV, the gap and :math:`\\lambda_{n\\mathbf{k}}` distributions, ``Tc_gap``,
``qdos``). Gaps are returned in meV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import baselines as bl

ISOTROPIC_FILE = 'migdal_eliashberg.npz'
ANISOTROPIC_FILE = 'migdal_eliashberg_aniso.npz'
KINDS = {'imag': 'gap0_imag', 'pade': 'gap_pade', 'acon': 'gap_acon'}


def _system_info(system_dir: str | Path) -> dict:
    from ...elphon.workflows import load_system

    config = load_system(str(system_dir))
    system = config.SYSTEM
    return {
        'system_dir': str(Path(system_dir).resolve()),
        'name': getattr(config, 'NAME', system.prefix),
        'prefix': system.prefix,
        'coupling_source': system.coupling_source,
        'coarse_grid': 'x'.join(str(n) for n in system.coarse_grid),
        'qgrid': 'x'.join(str(n) for n in system.qgrid),
        'superconductivity_defaults': dict(getattr(config, 'SUPERCONDUCTIVITY', {}) or {}),
    }


def _match(temperature_k: float, temps: NDArray) -> int:
    i = int(np.argmin(np.abs(temps - temperature_k)))
    if abs(temps[i] - temperature_k) > 1e-6:
        raise KeyError('no model temperature %g K (have %s)' % (temperature_k, temps))
    return i


def _temperature_tag(temperature_k: float) -> str:
    return '%06.2f' % temperature_k


class GapModel:
    """Isotropic Migdal-Eliashberg gap versus temperature (``migdal_eliashberg.npz``).

    Parameters
    ----------
    path : str or Path
        The file, or the ``me`` folder holding it.
    system : dict, optional
        Configuration of the system folder (:meth:`from_system`).

    Attributes
    ----------
    temperatures : ndarray
        Temperatures of the solutions (K).
    tc : dict
        ``'gap'`` (from :math:`\\Delta^2(T) \\to 0`) and ``'linear'`` (linearised kernel), in K.
    """

    def __init__(self, path: str | Path, system: dict | None = None) -> None:
        path = Path(path)
        self.path = path / ISOTROPIC_FILE if path.is_dir() else path
        self.system = dict(system or {})
        with np.load(self.path) as data:
            self.data = {key: np.asarray(data[key]) for key in data.files}
        self.temperatures = np.asarray(self.data['temps'], dtype=float)
        self.tc = {'gap': float(self.data['Tc_gap'])}
        if 'Tc_linear' in self.data:
            self.tc['linear'] = float(self.data['Tc_linear'])
        self.mu_star = float(self.data['mu_star']) if 'mu_star' in self.data else np.nan

    @classmethod
    def from_system(
        cls, system_dir: str | Path, output: str = 'output/me', workflow: str = 'superconductivity'
    ) -> GapModel:
        """Open ``<workflow>/<output>/migdal_eliashberg.npz`` of a system folder."""
        return cls(Path(system_dir) / workflow / output, system=_system_info(system_dir))

    def gaps(self, kind: str = 'imag') -> NDArray[np.float64]:
        """:math:`\\Delta(T)` (meV) over :attr:`temperatures`: ``'imag'`` (:math:`\\Delta(i\\omega_0)`), ``'pade'`` or ``'acon'`` gap edge."""
        if kind not in KINDS:
            raise ValueError("kind must be 'imag', 'pade' or 'acon'")
        return 1.0e3 * np.asarray(self.data[KINDS[kind]], dtype=float)

    def gap(self, temperature_k: float, kind: str = 'imag', interpolate: bool = False) -> float:
        """:math:`\\Delta` (meV) at a temperature (linear interpolation between model temperatures if asked)."""
        gaps = self.gaps(kind)
        if not interpolate:
            return float(gaps[_match(temperature_k, self.temperatures)])
        order = np.argsort(self.temperatures)
        t, g = self.temperatures[order], gaps[order]
        if temperature_k < t[0] or temperature_k > t[-1]:
            raise ValueError('%g K outside the model range %g-%g K' % (temperature_k, t[0], t[-1]))
        return float(np.interp(temperature_k, t, g))

    def gap_zero(self, kind: str = 'imag') -> float:
        """The gap (meV) at the lowest computed temperature."""
        return float(self.gaps(kind)[int(np.argmin(self.temperatures))])

    def gap_ratio(self, kind: str = 'imag', tc: str = 'gap') -> float:
        """:math:`2\\Delta(0)/k_B T_c` with :math:`\\Delta(0)` the lowest-temperature gap."""
        return 2.0e-3 * self.gap_zero(kind) / (bl.KB_EV * self.tc[tc])

    def bcs_gap(
        self, temps_k: NDArray | None = None, kind: str = 'imag', tc: str = 'gap'
    ) -> NDArray[np.float64]:
        """The BCS curve (meV) scaled to the model's :math:`\\Delta(0)` and Tc (baseline)."""
        temps = self.temperatures if temps_k is None else np.asarray(temps_k, dtype=float)
        return bl.bcs_gap_vs_t(temps, self.tc[tc], self.gap_zero(kind))

    def mass_renormalisation(self, temperature_k: float) -> float:
        """:math:`Z(i\\omega_0)` at a temperature."""
        return float(
            np.asarray(self.data['Z0'], dtype=float)[_match(temperature_k, self.temperatures)]
        )

    def quasiparticle_dos(
        self, temperature_k: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(w_ev, N_S(w)/N_F)`` of the real-axis solution at a temperature."""
        key = 'qdos_%s' % _temperature_tag(
            self.temperatures[_match(temperature_k, self.temperatures)]
        )
        if key not in self.data:
            raise KeyError('no quasiparticle DOS at %g K in %s' % (temperature_k, self.path))
        return np.asarray(self.data['w_real'], dtype=float), np.asarray(self.data[key], dtype=float)

    def real_axis(self, temperature_k: float, method: str = 'pade') -> dict[str, NDArray]:
        """``w_ev``, complex ``z`` and ``gap_ev`` of the real-axis solution (``'pade'`` or ``'acon'``)."""
        key = '%s_%s' % (
            method,
            _temperature_tag(self.temperatures[_match(temperature_k, self.temperatures)]),
        )
        if key not in self.data:
            raise KeyError('no %s solution at %g K in %s' % (method, temperature_k, self.path))
        table = np.asarray(self.data[key], dtype=float)
        return {
            'w_ev': np.asarray(self.data['w_real'], dtype=float),
            'z': table[:, 0] + 1j * table[:, 1],
            'gap_ev': table[:, 2] + 1j * table[:, 3],
        }

    def max_eigenvalue(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(temps_k, max eigenvalue)`` of the linearised kernel (1 at Tc)."""
        if 'max_eigenvalue' not in self.data:
            raise KeyError('no linearised-kernel data in %s' % self.path)
        return np.asarray(self.data['lin_temps'], dtype=float), np.asarray(
            self.data['max_eigenvalue'], dtype=float
        )

    def describe(self) -> str:
        """Markdown summary (for the report)."""
        lines = ['| parameter | value |', '|---|---|']
        for key in ('system_dir', 'name', 'prefix', 'coupling_source', 'coarse_grid', 'qgrid'):
            if key in self.system:
                lines.append('| %s | %s |' % (key, self.system[key]))
        lines.append('| migdal_eliashberg.npz | %s |' % self.path)
        lines.append('| mu_star | %g |' % self.mu_star)
        lines.append('| temperatures (K) | %s |' % ', '.join('%g' % t for t in self.temperatures))
        lines.append(
            '| Delta(iw_0) (meV) | %s |' % ', '.join('%.3f' % g for g in self.gaps('imag'))
        )
        for key, value in self.tc.items():
            lines.append('| Tc %s (K) | %.3f |' % (key, value))
        lines.append(
            '| 2 Delta(0) / k_B Tc | %.3f (BCS %.3f) |' % (self.gap_ratio(), bl.BCS_GAP_RATIO)
        )
        if 'converged' in self.data:
            lines.append(
                '| converged | %d of %d |'
                % (int(np.count_nonzero(self.data['converged'])), self.temperatures.size)
            )
        return '\n'.join(lines)


class AnisotropicGapModel:
    """Anisotropic Migdal-Eliashberg gaps per Fermi-surface state (``migdal_eliashberg_aniso.npz``).

    Parameters
    ----------
    path : str or Path
        The file, or the ``me_aniso`` folder holding it.
    system : dict, optional
        Configuration of the system folder (:meth:`from_system`).

    Attributes
    ----------
    temperatures : ndarray
    bands, weights, lambda_nk, energies_ev, k_cart : ndarray
        Per-state data (``nstate``).
    tc : float
        From :math:`\\Delta_{\\max}^2(T) \\to 0` (K).
    """

    def __init__(self, path: str | Path, system: dict | None = None) -> None:
        path = Path(path)
        self.path = path / ANISOTROPIC_FILE if path.is_dir() else path
        self.system = dict(system or {})
        with np.load(self.path) as data:
            self.data = {key: np.asarray(data[key]) for key in data.files}
        self.temperatures = np.asarray(self.data['temps'], dtype=float)
        self.bands = np.asarray(self.data['band'], dtype=int)
        self.weights = np.asarray(self.data['weight'], dtype=float)
        self.lambda_nk = np.asarray(self.data['lambda_nk'], dtype=float)
        self.energies_ev = np.asarray(self.data['energy_ev'], dtype=float)
        self.k_cart = np.asarray(self.data['k_cart'], dtype=float)
        self.tc = float(self.data['Tc_gap'])
        self.mu_star = float(self.data['mu_star']) if 'mu_star' in self.data else np.nan

    @classmethod
    def from_system(
        cls,
        system_dir: str | Path,
        output: str = 'output/me_aniso',
        workflow: str = 'superconductivity',
    ) -> AnisotropicGapModel:
        """Open ``<workflow>/<output>/migdal_eliashberg_aniso.npz`` of a system folder."""
        return cls(Path(system_dir) / workflow / output, system=_system_info(system_dir))

    def gaps(self, temperature_k: float, kind: str = 'gap0') -> NDArray[np.float64]:
        """:math:`\\Delta_{n\\mathbf{k}}` (meV) of every state: ``'gap0'`` (:math:`i\\omega_0`) or ``'edge'``."""
        key = {'gap0': 'gap0', 'edge': 'gap_edge'}[kind]
        return (
            1.0e3
            * np.asarray(self.data[key], dtype=float)[_match(temperature_k, self.temperatures)]
        )

    def gap_range(self, temperature_k: float) -> dict[str, float]:
        """``min``, ``mean`` (Fermi-surface weighted) and ``max`` gap (meV) at a temperature."""
        i = _match(temperature_k, self.temperatures)
        return {
            'min': 1.0e3 * float(self.data['gap_min'][i]),
            'mean': 1.0e3 * float(self.data['gap_mean'][i]),
            'max': 1.0e3 * float(self.data['gap_max'][i]),
        }

    def gap_distribution(
        self, temperature_k: float, kind: str = 'gap0'
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(gap_mev, distribution)`` at a temperature (``'gap0'`` or ``'edge'``)."""
        key = {'gap0': 'gap0_distribution', 'edge': 'gap_edge_distribution'}[kind]
        i = _match(temperature_k, self.temperatures)
        return np.asarray(self.data['gap_grid_mev'], dtype=float), np.asarray(
            self.data[key], dtype=float
        )[i]

    def gap_peaks(
        self,
        temperature_k: float,
        kind: str = 'gap0',
        min_fraction: float = 0.1,
        min_gap_mev: float = 0.05,
    ) -> list[float]:
        """Local maxima (meV) of the gap distribution above ``min_fraction`` of its maximum (the gaps of a multi-gap metal)."""
        grid, rho = self.gap_distribution(temperature_k, kind)
        keep = (grid > min_gap_mev) & (rho > min_fraction * rho.max())
        peaks = []
        for j in range(1, grid.size - 1):
            if keep[j] and rho[j] >= rho[j - 1] and rho[j] > rho[j + 1]:
                peaks.append(float(grid[j]))
        return peaks

    def sheet_gaps(
        self, temperature_k: float, groups: Mapping[str, Sequence[int]], kind: str = 'gap0'
    ) -> dict[str, dict[str, float]]:
        """Weighted mean gap (meV), its range and the mean :math:`\\lambda_{n\\mathbf{k}}` of band groups.

        Parameters
        ----------
        temperature_k : float
        groups : mapping
            ``name -> band indices`` (e.g. ``{'sigma': [6, 7], 'pi': [8]}`` for MgB2).
        kind : {'gap0', 'edge'}
        """
        gaps = self.gaps(temperature_k, kind)
        out = {}
        for name, bands in groups.items():
            mask = np.isin(self.bands, list(bands))
            if not np.any(mask):
                raise KeyError('no state of bands %s in the Fermi-surface set' % (list(bands),))
            w = self.weights[mask]
            out[name] = {
                'gap_mean': float(np.average(gaps[mask], weights=w)),
                'gap_min': float(gaps[mask].min()),
                'gap_max': float(gaps[mask].max()),
                'lambda_mean': float(np.average(self.lambda_nk[mask], weights=w)),
                'weight': float(w.sum() / self.weights.sum()),
                'states': int(mask.sum()),
            }
        return out

    def lambda_statistics(self) -> dict[str, float]:
        """``min``, ``max`` and weighted ``mean`` of :math:`\\lambda_{n\\mathbf{k}}`, and the stored isotropic ``lambda``."""
        out = {
            'min': float(self.lambda_nk.min()),
            'max': float(self.lambda_nk.max()),
            'mean': float(np.average(self.lambda_nk, weights=self.weights)),
        }
        if 'lambda' in self.data:
            out['lambda'] = float(self.data['lambda'])
        return out

    def quasiparticle_dos(
        self, temperature_k: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(w_ev, N_S(w)/N_F)`` at a temperature."""
        i = _match(temperature_k, self.temperatures)
        return np.asarray(self.data['w_real'], dtype=float), np.asarray(
            self.data['qdos'], dtype=float
        )[i]

    def describe(self) -> str:
        """Markdown summary (for the report)."""
        lines = ['| parameter | value |', '|---|---|']
        for key in ('system_dir', 'name', 'prefix', 'coupling_source', 'coarse_grid', 'qgrid'):
            if key in self.system:
                lines.append('| %s | %s |' % (key, self.system[key]))
        lines.append('| migdal_eliashberg_aniso.npz | %s |' % self.path)
        lines.append('| mu_star | %g |' % self.mu_star)
        lines.append(
            '| Fermi-surface states | %d (bands %s) |'
            % (self.bands.size, ', '.join(str(b) for b in np.unique(self.bands)))
        )
        stats = self.lambda_statistics()
        lines.append(
            '| lambda_nk min / mean / max | %.3f / %.3f / %.3f |'
            % (stats['min'], stats['mean'], stats['max'])
        )
        lines.append('| temperatures (K) | %s |' % ', '.join('%g' % t for t in self.temperatures))
        lines.append(
            '| Delta_nk(iw_0) min / mean / max (meV) | %s |'
            % '; '.join(
                '%g K: %.2f / %.2f / %.2f' % (t, *self.gap_range(t).values())
                for t in self.temperatures
            )
        )
        lines.append('| Tc (Delta_max^2 -> 0) (K) | %.3f |' % self.tc)
        if 'converged' in self.data:
            lines.append(
                '| converged | %d of %d |'
                % (int(np.count_nonzero(self.data['converged'])), self.temperatures.size)
            )
        return '\n'.join(lines)
