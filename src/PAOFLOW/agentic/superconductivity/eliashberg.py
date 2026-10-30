"""Adapter: the Eliashberg function of a system folder (``superconductivity/output/eliashberg.npz``).

Written by :func:`PAOFLOW.elphon.workflows.write_eliashberg` (``superconductivity/eliashberg.py``):
``omega`` (eV), ``a2F``, ``lambda``, ``omega_log`` (eV), ``omega_2`` (eV), ``mu_star``,
``Tc_mcmillan``, ``Tc_allen_dynes``, ``f1``, ``f2``, the mode-resolved ``lambda_qv`` /
``omega_q`` / ``q_weights``, and with ``--transport`` the transport function ``a2F_tr``,
``lambda_tr`` and the plasma frequency ``omega_plasma_ev``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import baselines as bl

#: Scalars reported by :meth:`EliashbergModel.describe`.
SCALARS = (
    'lambda',
    'omega_log',
    'omega_2',
    'mu_star',
    'Tc_mcmillan',
    'Tc_allen_dynes',
    'f1',
    'f2',
    'lambda_tr',
    'omega_plasma_ev',
)


class EliashbergModel:
    """Model Eliashberg observables from an ``eliashberg.npz``.

    Parameters
    ----------
    path : str or Path
        The file, or a folder holding ``eliashberg.npz``.
    system : dict, optional
        Configuration of the system folder (:meth:`from_system`), reported by :meth:`describe`.

    Attributes
    ----------
    omega_ev, a2f : ndarray
        The Eliashberg function.
    scalars : dict
        The stored scalars (:data:`SCALARS`; ``omega_log`` and ``omega_2`` in eV, Tc in K).
    transport : bool
        Whether ``a2F_tr`` is present.
    """

    def __init__(self, path: str | Path, system: dict | None = None) -> None:
        path = Path(path)
        self.path = path / 'eliashberg.npz' if path.is_dir() else path
        self.system = dict(system or {})
        with np.load(self.path) as data:
            self.data = {key: np.asarray(data[key]) for key in data.files}
        self.omega_ev = np.asarray(self.data['omega'], dtype=float)
        self.a2f = np.asarray(self.data['a2F'], dtype=float)
        self.scalars = {
            key: float(self.data[key])
            for key in SCALARS
            if key in self.data and np.ndim(self.data[key]) == 0
        }
        self.transport = 'a2F_tr' in self.data

    @classmethod
    def from_system(
        cls, system_dir: str | Path, output: str = 'output', workflow: str = 'superconductivity'
    ) -> EliashbergModel:
        """Open ``<workflow>/<output>/eliashberg.npz`` of an electron-phonon system folder."""
        from ...elphon.workflows import load_system

        config = load_system(str(system_dir))
        system = config.SYSTEM
        info = {
            'system_dir': str(Path(system_dir).resolve()),
            'name': getattr(config, 'NAME', system.prefix),
            'prefix': system.prefix,
            'coupling_source': system.coupling_source,
            'coarse_grid': 'x'.join(str(n) for n in system.coarse_grid),
            'qgrid': 'x'.join(str(n) for n in system.qgrid),
            'basis': '%s (%s, pthr %g)'
            % (system.basis_dir or 'BASIS', system.configuration, system.pthr),
            'superconductivity_defaults': dict(getattr(config, 'SUPERCONDUCTIVITY', {}) or {}),
        }
        return cls(Path(system_dir) / workflow / output / 'eliashberg.npz', system=info)

    def _function(self, transport: bool) -> NDArray[np.float64]:
        if transport:
            if not self.transport:
                raise KeyError('%s has no a2F_tr (run eliashberg.py --transport)' % self.path)
            return np.asarray(self.data['a2F_tr'], dtype=float)
        return self.a2f

    def function(
        self, omega_ev: NDArray | None = None, transport: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(omega_ev, alpha^2F)`` (or ``alpha^2F_tr``), interpolated on ``omega_ev`` if given."""
        f = self._function(transport)
        if omega_ev is None:
            return self.omega_ev, f
        w = np.asarray(omega_ev, dtype=float)
        return w, np.interp(w, self.omega_ev, f, left=0.0, right=0.0)

    def cumulative_lambda(
        self, omega_ev: NDArray | None = None, transport: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(omega_ev, lambda(omega))``: the coupling accumulated up to each frequency."""
        w, f = self.function(omega_ev, transport)
        return w, bl.cumulative_lambda(w, f)

    def moments(self, transport: bool = False) -> dict[str, float]:
        """``lambda``, ``omega_log_ev``, ``omega_2_ev`` recomputed from the stored curve."""
        return bl.a2f_moments(*self.function(transport=transport))

    def lambda_in_window(
        self, omega_min_ev: float, omega_max_ev: float, transport: bool = False
    ) -> float:
        """Contribution to :math:`\\lambda` of the modes between two frequencies (eV)."""
        w, f = self.function(transport=transport)
        keep = (w >= omega_min_ev) & (w <= omega_max_ev) & (w > 0.0)
        if np.count_nonzero(keep) < 2:
            return 0.0
        return float(np.trapezoid(2.0 * f[keep] / w[keep], w[keep]))

    def tc(self, mu_star: float | None = None, formula: str = 'allen_dynes') -> float:
        """Tc (K) from the stored moments for another ``mu_star`` (default: the run's).

        ``formula`` is ``'allen_dynes'`` (with :math:`f_1 f_2`) or ``'mcmillan'``.
        """
        mu = self.scalars['mu_star'] if mu_star is None else float(mu_star)
        lam, wlog = self.scalars['lambda'], self.scalars['omega_log']
        if formula == 'mcmillan':
            return bl.mcmillan_tc(lam, wlog, mu)
        if formula == 'allen_dynes':
            return bl.allen_dynes_tc(lam, wlog, self.scalars['omega_2'], mu)
        raise ValueError("formula must be 'allen_dynes' or 'mcmillan'")

    def mode_lambda(self) -> dict[str, NDArray[np.float64]]:
        """Mode-resolved coupling: ``omega_q_ev`` (nq, nmode), ``lambda_qv``, ``q_weights``."""
        if 'lambda_qv' not in self.data:
            raise KeyError('%s has no mode-resolved lambda' % self.path)
        return {
            'omega_q_ev': np.asarray(self.data['omega_q'], dtype=float),
            'lambda_qv': np.asarray(self.data['lambda_qv'], dtype=float),
            'q_weights': np.asarray(self.data['q_weights'], dtype=float),
        }

    def describe(self) -> str:
        """Markdown summary of the run (for the report)."""
        lines = ['| parameter | value |', '|---|---|']
        for key in (
            'system_dir',
            'name',
            'prefix',
            'coupling_source',
            'coarse_grid',
            'qgrid',
            'basis',
        ):
            if key in self.system:
                lines.append('| %s | %s |' % (key, self.system[key]))
        defaults = self.system.get('superconductivity_defaults', {})
        for key in ('nk', 'nq', 'sigma_ev', 'adaptive_smearing', 'mu_star', 'fsthick_ev'):
            if key in defaults:
                lines.append('| %s (system.py) | %s |' % (key, defaults[key]))
        lines.append('| eliashberg.npz | %s |' % self.path)
        lines.append(
            '| omega grid (meV) | %d points, %.2f-%.2f |'
            % (self.omega_ev.size, 1e3 * self.omega_ev[0], 1e3 * self.omega_ev[-1])
        )
        for key in SCALARS:
            if key in self.scalars:
                value = self.scalars[key]
                if key in ('omega_log', 'omega_2'):
                    lines.append('| %s (meV) | %.3f |' % (key, 1e3 * value))
                elif key.startswith('Tc'):
                    lines.append('| %s (K) | %.3f |' % (key, value))
                else:
                    lines.append('| %s | %.4g |' % (key, value))
        moments = self.moments()
        lines.append('| lambda recomputed from the curve | %.4f |' % moments['lambda'])
        return '\n'.join(lines)
