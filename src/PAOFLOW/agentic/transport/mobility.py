"""Adapter: PAOFLOW phonon-limited carrier transport outputs as model curves (semiconductors).

Reads ``mobility.dat`` and ``carrier_scattering.npz`` written by the transport workflow of
an electron-phonon system folder (``transport/mobility.py``; backend
:mod:`PAOFLOW.elphon.carrier_scattering`) and evaluates the observables at experimental
temperatures: electron and hole mobilities (SERTA or MRTA), carrier densities,
conductivities and the Seebeck coefficient.  Mobilities interpolate as a power law
(:math:`\\ln\\mu` linear in :math:`\\ln T`), carrier densities as activated quantities
(:math:`\\ln n` linear in :math:`1/T`); off-grid values are *interpolated models* to be
labelled as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Columns of ``mobility.dat`` (``PAOFLOW.elphon.carrier_scattering.write_mobility_outputs``).
COLUMNS = (
    'temperature_k',
    'fermi_level_ev',
    'n_e_cm3',
    'n_h_cm3',
    'mobility_e_serta',
    'mobility_h_serta',
    'mobility_e_mrta',
    'mobility_h_mrta',
    'sigma_serta',
    'sigma_mrta',
    'seebeck_mrta',
)


class MobilityModel:
    """Model carrier-transport observables from a PAOFLOW ``transport/output`` folder.

    Parameters
    ----------
    output_dir : str or Path
        Folder with ``mobility.dat`` and (optionally) ``carrier_scattering.npz``.
    system : dict, optional
        Configuration of the system folder (:meth:`from_system`), reported by :meth:`describe`.

    Attributes
    ----------
    temperatures : ndarray
        Model temperatures (K).
    table : dict
        The columns of ``mobility.dat`` (:data:`COLUMNS`), each an array over the temperatures.
    metadata : dict
        Run parameters (grids, smearing, doping, gap, electrons provenance, screening).
    """

    def __init__(self, output_dir: str | Path, system: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.system = dict(system or {})
        data = np.loadtxt(self.output_dir / 'mobility.dat', ndmin=2)
        if data.shape[1] != len(COLUMNS):
            raise ValueError(
                '%s has %d columns, expected %d'
                % (self.output_dir / 'mobility.dat', data.shape[1], len(COLUMNS))
            )
        self.table = {name: data[:, i] for i, name in enumerate(COLUMNS)}
        self.temperatures = self.table['temperature_k']
        self.scattering = None
        npz = self.output_dir / 'carrier_scattering.npz'
        if npz.exists():
            with np.load(npz, allow_pickle=True) as arrays:
                self.scattering = {key: np.asarray(arrays[key]) for key in arrays.files}
        self.metadata = self._metadata()

    @classmethod
    def from_system(
        cls, system_dir: str | Path, output: str = 'output', workflow: str = 'transport'
    ) -> 'MobilityModel':
        """Open the transport output of an electron-phonon system folder.

        Parameters
        ----------
        system_dir : str or Path
            A folder holding ``system.py``.
        output : str, optional
            Output directory of ``transport/mobility.py`` (``--tag`` runs use other names).
        workflow : str, optional
            Workflow folder name.

        Returns
        -------
        MobilityModel
        """
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
            'long_range': 'on' if system.long_range else 'off',
            'quadrupoles': system.quadrupoles or 'none',
            'transport_defaults': dict(getattr(config, 'TRANSPORT', {}) or {}),
        }
        return cls(Path(system_dir) / workflow / output, system=info)

    def _metadata(self) -> dict:
        out: dict = {'temperatures_k': self.temperatures.tolist()}
        if self.scattering is None:
            return out
        a = self.scattering
        keys = ('nk_dense', 'nq_dense', 'smearing_ev', 'adaptive_smearing', 'doping_cm3', 'vbm_ev', 'cbm_ev',
                'metal', 'scissor_mode', 'spin_degeneracy', 'coupling_save', 'electrons_save', 'electrons_dft',
                'electrons_hubbard', 'coupling_screening')  # fmt: skip
        for key in keys:
            if key in a:
                out[key] = a[key].item() if np.ndim(a[key]) == 0 else a[key]
        if 'vbm_ev' in out and 'cbm_ev' in out and not out.get('metal', False):
            out['gap_ev'] = float(out['cbm_ev']) - float(out['vbm_ev'])
        if 'scissor_ev' in a:
            out['scissor_ev'] = dict(
                zip(self.temperatures.tolist(), np.atleast_1d(a['scissor_ev']).tolist())
            )
        if 'in_window' in a:
            out['window_states'] = int(np.count_nonzero(a['in_window']))
        return out

    def bracket(self, temperature_k: float) -> tuple[float, float]:
        """The model temperatures enclosing ``temperature_k`` (equal if on the grid)."""
        temps = np.sort(self.temperatures)
        if temperature_k < temps[0] or temperature_k > temps[-1]:
            raise ValueError(
                '%g K outside the model range %g-%g K' % (temperature_k, temps[0], temps[-1])
            )
        k = int(np.searchsorted(temps, temperature_k))
        if np.isclose(temps[min(k, temps.size - 1)], temperature_k):
            return float(temperature_k), float(temperature_k)
        return float(temps[k - 1]), float(temps[k])

    def _column(self, name: str, temperature_k: float, interpolate: bool, law: str) -> float:
        values = self.table[name]
        low, high = self.bracket(temperature_k) if interpolate else (temperature_k, temperature_k)
        a = int(np.argmin(np.abs(self.temperatures - low)))
        if low == high:
            if abs(self.temperatures[a] - temperature_k) > 1e-6:
                raise KeyError(
                    'no model temperature %g K (have %s)' % (temperature_k, self.temperatures)
                )
            return float(values[a])
        b = int(np.argmin(np.abs(self.temperatures - high)))
        if law == 'power':  # ln y linear in ln T
            w = np.log(temperature_k / low) / np.log(high / low)
            return float(np.exp((1 - w) * np.log(values[a]) + w * np.log(values[b])))
        if law == 'activated':  # ln y linear in 1/T
            w = (1.0 / temperature_k - 1.0 / low) / (1.0 / high - 1.0 / low)
            return float(np.exp((1 - w) * np.log(values[a]) + w * np.log(values[b])))
        w = (temperature_k - low) / (high - low)
        return float((1 - w) * values[a] + w * values[b])

    def mobility(
        self,
        temperature_k: float,
        carrier: str = 'e',
        kind: str = 'mrta',
        interpolate: bool = False,
    ) -> float:
        """Drift mobility (cm^2/Vs) of ``carrier`` (``'e'``, ``'h'``), ``kind`` ``'serta'`` or ``'mrta'``.

        Off-grid temperatures (``interpolate=True``) follow a local power law, :math:`\\ln\\mu`
        linear in :math:`\\ln T` between the bracketing model temperatures.
        """
        if carrier not in ('e', 'h') or kind not in ('serta', 'mrta'):
            raise ValueError("carrier must be 'e' or 'h', kind 'serta' or 'mrta'")
        return self._column('mobility_%s_%s' % (carrier, kind), temperature_k, interpolate, 'power')

    def carrier_density(
        self, temperature_k: float, carrier: str = 'e', interpolate: bool = False
    ) -> float:
        """Carrier density (cm^-3); off-grid: :math:`\\ln n` linear in :math:`1/T`."""
        return self._column('n_%s_cm3' % carrier, temperature_k, interpolate, 'activated')

    def conductivity(
        self, temperature_k: float, kind: str = 'mrta', interpolate: bool = False
    ) -> float:
        """Conductivity (S/m); off-grid: :math:`\\ln\\sigma` linear in :math:`1/T`."""
        return self._column('sigma_%s' % kind, temperature_k, interpolate, 'activated')

    def seebeck(self, temperature_k: float, interpolate: bool = False) -> float:
        """Seebeck coefficient (uV/K, MRTA); off-grid: linear in T."""
        return self._column('seebeck_mrta', temperature_k, interpolate, 'linear')

    def power_law_exponent(
        self, carrier: str = 'e', kind: str = 'mrta', window: tuple[float, float] | None = None
    ) -> float:
        """Exponent :math:`p` of :math:`\\mu \\propto T^{-p}` (log-log fit over the model temperatures).

        Parameters
        ----------
        carrier, kind
            As :meth:`mobility`.
        window : (Tmin, Tmax), optional
            Temperatures used (default all).

        Returns
        -------
        float
            :math:`p` (positive for a mobility that decreases with temperature).
        """
        mu = self.table['mobility_%s_%s' % (carrier, kind)]
        t = self.temperatures
        keep = (
            np.ones_like(t, dtype=bool) if window is None else (t >= window[0]) & (t <= window[1])
        )
        if np.count_nonzero(keep) < 2:
            raise ValueError('at least two model temperatures are needed for the exponent')
        slope = np.polyfit(np.log(t[keep]), np.log(mu[keep]), 1)[0]
        return float(-slope)

    def describe(self) -> str:
        """Markdown summary of the run parameters (for the report)."""
        m = self.metadata
        lines = ['| parameter | value |', '|---|---|']
        for key in ('system_dir', 'name', 'prefix', 'coupling_source', 'coarse_grid', 'qgrid', 'basis',
                    'long_range', 'quadrupoles'):  # fmt: skip
            if key in self.system:
                lines.append('| %s | %s |' % (key, self.system[key]))
        lines.append('| temperatures (K) | %s |' % ', '.join('%g' % t for t in self.temperatures))
        for key in ('nk_dense', 'nq_dense', 'smearing_ev', 'adaptive_smearing', 'doping_cm3', 'gap_ev',
                    'scissor_mode', 'window_states', 'electrons_dft', 'electrons_hubbard', 'electrons_save',
                    'coupling_screening'):  # fmt: skip
            if key in m:
                value = m[key]
                lines.append(
                    '| %s | %s |' % (key, ('%.4g' % value) if isinstance(value, float) else value)
                )
        for kind in ('serta', 'mrta'):
            lines.append(
                '| mobility e / h %s (cm^2/Vs) | %s |'
                % (kind.upper(), ', '.join('%g K: %.0f / %.0f' % (t, e, h) for t, e, h in zip(
                    self.temperatures, self.table['mobility_e_' + kind], self.table['mobility_h_' + kind])))
            )  # fmt: skip
        return '\n'.join(lines)
