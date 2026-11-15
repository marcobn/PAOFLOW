"""Adapter: PAOFLOW metal transport outputs (resistivity, infrared optics, emissivity) as curves.

Reads ``resistivity.dat`` and ``emissivity.npz`` written by ``transport/resistivity.py`` of an
electron-phonon system folder (:func:`PAOFLOW.elphon.workflows.metal_resistivity`), and the
``eliashberg.npz`` they were built from (``lambda``, ``lambda_tr``, ``omega_log``, the
Fermi-surface plasma frequency, Tc).  The DC resistivity is a sum over scattering channels
(Matthiessen's rule with one plasma frequency), so a residual resistivity can be added after
the fact (:meth:`ResistivityModel.resistivity` with ``rho0_uohm_cm``) without a new run.
Off-grid temperatures are *interpolated models* to be labelled as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Scalars of ``eliashberg.npz`` reported by :meth:`ResistivityModel.describe`.
ELIASHBERG_KEYS = (
    'lambda',
    'lambda_tr',
    'omega_plasma_ev',
    'omega_log',
    'mu_star',
    'Tc_mcmillan',
    'Tc_allen_dynes',
)


def read_resistivity_table(path: str | Path) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """Temperatures, total resistivity and per-channel resistivities of a ``resistivity.dat``.

    Parameters
    ----------
    path : str or Path
        File written by :func:`PAOFLOW.elphon.workflows.metal_resistivity`: the header
        ``# T (K), rho total and per channel (uOhm cm): <name>, <name>, ...`` names the
        channel columns.

    Returns
    -------
    tuple
        ``(temperatures_k, total_uohm_cm, {channel: values})``.

    Raises
    ------
    ValueError
        If the channel names do not match the columns.
    """
    path = Path(path)
    with open(path) as handle:
        first = handle.readline()
    names: list[str] = []
    if first.startswith('#') and ':' in first:
        names = [s.strip() for s in first.split(':', 1)[1].split(',') if s.strip()]
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError('%s needs at least T and rho columns' % path)
    nchannel = data.shape[1] - 2
    if not names:
        names = ['channel_%d' % i for i in range(nchannel)]
    if len(names) != nchannel:
        raise ValueError(
            '%s names %d channels but has %d channel columns' % (path, len(names), nchannel)
        )
    return data[:, 0], data[:, 1], {name: data[:, 2 + i] for i, name in enumerate(names)}


class ResistivityModel:
    """Model metal-transport observables from a PAOFLOW ``transport/output`` folder.

    Parameters
    ----------
    output_dir : str or Path
        Folder with ``resistivity.dat`` and (optionally) ``emissivity.npz``.
    eliashberg : str or Path, optional
        The ``eliashberg.npz`` the run used (default:
        ``../../superconductivity/output/eliashberg.npz`` relative to ``output_dir``, if it exists).
    system : dict, optional
        Configuration of the system folder (:meth:`from_system`), reported by :meth:`describe`.

    Attributes
    ----------
    temperatures : ndarray
        Model temperatures (K).
    channels : list of str
        Scattering channels of the run (``'e-ph (a2F_tr)'``, ``'residual'``, ...).
    table : dict
        ``'total'`` and one array per channel (micro-ohm cm) over the temperatures.
    emissivity : dict or None
        The arrays of ``emissivity.npz`` (``omega_ev``, ``temps_k``, ``eps1``, ``eps2``,
        ``hemispherical_opaque``, ``hemispherical_slab``, ``directional_slab``,
        ``total_opaque``, ``total_slab``, ``planck_coverage``, ``angles_deg``, ...).
    eliashberg : dict
        Scalars of ``eliashberg.npz`` (:data:`ELIASHBERG_KEYS`) when the file is found.
    """

    def __init__(
        self,
        output_dir: str | Path,
        eliashberg: str | Path | None = None,
        system: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.system = dict(system or {})
        temps, total, per_channel = read_resistivity_table(self.output_dir / 'resistivity.dat')
        self.temperatures = temps
        self.channels = list(per_channel)
        self.table = {'total': total, **per_channel}
        self.emissivity = None
        npz = self.output_dir / 'emissivity.npz'
        if npz.exists():
            with np.load(npz, allow_pickle=True) as arrays:
                self.emissivity = {key: np.asarray(arrays[key]) for key in arrays.files}
        if eliashberg is None:
            candidate = (
                self.output_dir.parent.parent / 'superconductivity' / 'output' / 'eliashberg.npz'
            )
            eliashberg = candidate if candidate.exists() else None
        self.eliashberg_path = Path(eliashberg) if eliashberg is not None else None
        self.eliashberg: dict[str, float] = {}
        if self.eliashberg_path is not None:
            with np.load(self.eliashberg_path) as data:
                for key in ELIASHBERG_KEYS:
                    if key in data.files and np.ndim(data[key]) == 0:
                        self.eliashberg[key] = float(data[key])

    @classmethod
    def from_system(
        cls,
        system_dir: str | Path,
        output: str = 'output',
        workflow: str = 'transport',
        eliashberg: str | Path | None = None,
    ) -> ResistivityModel:
        """Open the metal-transport output of an electron-phonon system folder.

        Parameters
        ----------
        system_dir : str or Path
            A folder holding ``system.py``.
        output : str, optional
            Output directory of ``transport/resistivity.py``.
        workflow : str, optional
            Workflow folder name.
        eliashberg : str or Path, optional
            ``eliashberg.npz`` of the run (default: ``superconductivity/output/eliashberg.npz``).

        Returns
        -------
        ResistivityModel
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
            'transport_defaults': dict(getattr(config, 'TRANSPORT', {}) or {}),
            'superconductivity_defaults': dict(getattr(config, 'SUPERCONDUCTIVITY', {}) or {}),
        }
        if eliashberg is None:
            candidate = Path(system_dir) / 'superconductivity' / 'output' / 'eliashberg.npz'
            eliashberg = candidate if candidate.exists() else None
        return cls(Path(system_dir) / workflow / output, eliashberg=eliashberg, system=info)

    # ----------------------------------------------------------------- temperatures
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

    def _interpolate(
        self, temps: NDArray, values: NDArray, temperature_k: float, interpolate: bool, law: str
    ) -> NDArray:
        temps = np.asarray(temps, dtype=float)
        a = int(np.argmin(np.abs(temps - temperature_k)))
        if not interpolate:
            if abs(temps[a] - temperature_k) > 1e-6:
                raise KeyError('no model temperature %g K (have %s)' % (temperature_k, temps))
            return np.asarray(values[a])
        order = np.argsort(temps)
        ts, vs = temps[order], np.asarray(values)[order]
        if temperature_k < ts[0] or temperature_k > ts[-1]:
            raise ValueError(
                '%g K outside the model range %g-%g K' % (temperature_k, ts[0], ts[-1])
            )
        k = int(np.searchsorted(ts, temperature_k))
        if np.isclose(ts[min(k, ts.size - 1)], temperature_k):
            return np.asarray(vs[min(k, ts.size - 1)])
        low, high = ts[k - 1], ts[k]
        if law == 'power':  # ln y linear in ln T (the T^n regime below the Debye temperature)
            w = np.log(temperature_k / low) / np.log(high / low)
            return np.exp((1 - w) * np.log(vs[k - 1]) + w * np.log(vs[k]))
        w = (temperature_k - low) / (high - low)
        return (1 - w) * vs[k - 1] + w * vs[k]

    # ----------------------------------------------------------------- resistivity
    def resistivity(
        self,
        temperature_k: float,
        channel: str = 'total',
        rho0_uohm_cm: float = 0.0,
        interpolate: bool = False,
        law: str = 'linear',
    ) -> float:
        """Resistivity (micro-ohm cm) of ``channel`` (``'total'`` or a channel name) at ``temperature_k``.

        Parameters
        ----------
        temperature_k : float
        channel : str, optional
            ``'total'`` or one of :attr:`channels`.
        rho0_uohm_cm : float, optional
            Residual resistivity added to the total (Matthiessen's rule): the impurity channel
            of the measured sample, which the phonon-limited run does not contain.
        interpolate : bool, optional
            Interpolate between the bracketing model temperatures (label the result).
        law : {'linear', 'power'}
            Interpolation: linear in T (above about a fifth of the Debye temperature) or
            :math:`\\ln\\rho` linear in :math:`\\ln T` (the :math:`T^n` regime).

        Returns
        -------
        float
        """
        if channel not in self.table:
            raise KeyError('channel %r not in %s' % (channel, list(self.table)))
        value = float(
            self._interpolate(
                self.temperatures, self.table[channel], temperature_k, interpolate, law
            )
        )
        return value + (rho0_uohm_cm if channel == 'total' else 0.0)

    def slope(
        self, window: tuple[float, float] | None = None, channel: str = 'total'
    ) -> dict[str, float]:
        """Linear fit :math:`\\rho = a + b\\,T` over the model temperatures in ``window``.

        Returns
        -------
        dict
            ``slope`` (micro-ohm cm / K), ``intercept`` (micro-ohm cm), ``rms`` and ``n``.
        """
        t, rho = self.temperatures, self.table[channel]
        keep = (
            np.ones_like(t, dtype=bool) if window is None else (t >= window[0]) & (t <= window[1])
        )
        if np.count_nonzero(keep) < 2:
            raise ValueError('at least two model temperatures are needed for the slope')
        b, a = np.polyfit(t[keep], rho[keep], 1)
        rms = float(np.sqrt(np.mean((rho[keep] - a - b * t[keep]) ** 2)))
        return {'slope': float(b), 'intercept': float(a), 'rms': rms, 'n': int(keep.sum())}

    def residual_resistivity_ratio(
        self, rho0_uohm_cm: float, t_low: float, t_high: float = 300.0, interpolate: bool = True
    ) -> float:
        """RRR = (rho_ph(t_high) + rho0) / (rho_ph(t_low) + rho0) of a sample with residual ``rho0``."""
        high = self.resistivity(t_high, rho0_uohm_cm=rho0_uohm_cm, interpolate=interpolate)
        low = self.resistivity(
            t_low, rho0_uohm_cm=rho0_uohm_cm, interpolate=interpolate, law='power'
        )
        return float(high / low)

    # ----------------------------------------------------------------- optics and emissivity
    def _need_emissivity(self) -> dict:
        if self.emissivity is None:
            raise FileNotFoundError('%s has no emissivity.npz' % self.output_dir)
        return self.emissivity

    def emissivity_total(
        self, temperature_k: float, opaque: bool = True, interpolate: bool = False
    ) -> float:
        """Total hemispherical emissivity at a temperature (opaque surface or the run's slab)."""
        data = self._need_emissivity()
        key = 'total_opaque' if opaque else 'total_slab'
        return float(
            self._interpolate(data['temps_k'], data[key], temperature_k, interpolate, 'linear')
        )

    def emissivity_spectrum(
        self,
        temperature_k: float,
        kind: str = 'hemispherical_opaque',
        angle_deg: float | None = None,
        interpolate: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Spectral emissivity ``(omega_ev, emissivity)`` at a temperature.

        Parameters
        ----------
        temperature_k : float
        kind : {'hemispherical_opaque', 'hemispherical_slab', 'directional_slab'}
        angle_deg : float, optional
            Emission angle of a ``'directional_slab'`` spectrum (one of the run's angles).
        interpolate : bool, optional
            Linear interpolation between the bracketing model temperatures.
        """
        data = self._need_emissivity()
        if kind == 'directional_slab':
            if angle_deg is None:
                raise ValueError('angle_deg is needed for a directional spectrum')
            angles = np.asarray(data['angles_deg'], dtype=float)
            ia = int(np.argmin(np.abs(angles - angle_deg)))
            if abs(angles[ia] - angle_deg) > 1e-6:
                raise KeyError('angle %g deg not in the run (have %s)' % (angle_deg, angles))
            values = data[kind][:, ia, :]
        else:
            values = data[kind]
        spectrum = self._interpolate(data['temps_k'], values, temperature_k, interpolate, 'linear')
        return np.asarray(data['omega_ev'], dtype=float), np.asarray(spectrum, dtype=float)

    def dielectric(
        self, temperature_k: float, interpolate: bool = False
    ) -> tuple[NDArray, NDArray, NDArray]:
        """``(omega_ev, eps1, eps2)`` of the extended-Drude (+ interband) dielectric function."""
        data = self._need_emissivity()
        eps1 = self._interpolate(
            data['temps_k'], data['eps1'], temperature_k, interpolate, 'linear'
        )
        eps2 = self._interpolate(
            data['temps_k'], data['eps2'], temperature_k, interpolate, 'linear'
        )
        return np.asarray(data['omega_ev'], dtype=float), np.asarray(eps1), np.asarray(eps2)

    # ----------------------------------------------------------------- report
    def describe(self) -> str:
        """Markdown summary of the run parameters (for the report)."""
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
        if self.eliashberg_path is not None:
            lines.append('| eliashberg.npz | %s |' % self.eliashberg_path)
        for key in ELIASHBERG_KEYS:
            if key in self.eliashberg:
                lines.append('| %s | %.4g |' % (key, self.eliashberg[key]))
        lines.append('| channels | %s |' % ', '.join(self.channels))
        lines.append('| temperatures (K) | %s |' % ', '.join('%g' % t for t in self.temperatures))
        lines.append(
            '| rho total (uOhm cm) | %s |'
            % ', '.join(
                '%g K: %.3f' % (t, r) for t, r in zip(self.temperatures, self.table['total'])
            )
        )
        if self.emissivity is not None:
            e = self.emissivity
            lines.append('| Drude model | %s |' % e.get('drude', ''))
            if 'thickness_m' in e:
                lines.append('| slab thickness (m) | %g |' % float(e['thickness_m']))
            if 'angles_deg' in e:
                lines.append('| angles (deg) | %s |' % ', '.join('%g' % a for a in e['angles_deg']))
            lines.append(
                '| total hemispherical emissivity (opaque) | %s |'
                % ', '.join('%g K: %.4f' % (t, v) for t, v in zip(e['temps_k'], e['total_opaque']))
            )
        return '\n'.join(lines)
